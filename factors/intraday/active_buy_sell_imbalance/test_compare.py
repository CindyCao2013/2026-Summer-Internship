"""Sprint 3.5 consistency gates for active_buy_sell_imbalance."""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.active_buy_sell_imbalance.compute import (
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    assert_standard_bartimes,
    ddb_version,
    python_version,
)

NUMERIC_ATOL = 1e-10
SPEARMAN_MIN = 0.999


def assert_consistency(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
    *,
    numeric_atol: float = NUMERIC_ATOL,
    spearman_min: float = SPEARMAN_MIN,
) -> Dict[str, object]:
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)
    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping active_buy_sell_imbalance signals")

    max_diff = float(aligned["abs_diff"].max())

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["python"].corr(group["ddb"], method="spearman")

    spear = aligned.groupby("bartime", sort=False).apply(_spearman).dropna()
    min_spearman = float(spear.min()) if len(spear) else None
    assert max_diff <= numeric_atol, {"max_abs_diff": max_diff}
    if min_spearman is not None:
        assert min_spearman >= spearman_min, spear.describe()
    for label, frame in (("python", python_narrow), ("ddb", ddb_narrow)):
        if not frame["value"].between(-1, 1).all():
            raise AssertionError(f"{label} values outside [-1, 1]")

    return {
        "aligned_rows": len(aligned),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(aligned["abs_diff"].mean()),
        "spearman_min": min_spearman,
        "python_rows": len(python_narrow),
        "ddb_rows": len(ddb_narrow),
        "bartimes": sorted(
            pd.to_datetime(aligned["bartime"]).dt.strftime("%H:%M").unique()
        ),
    }


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _flow_frame(future_buy: float = 1_000.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 3,
            "date": [day] * 3,
            "bartime": [
                day + pd.Timedelta(hours=9, minutes=30),
                day + pd.Timedelta(hours=9, minutes=59),
                day + pd.Timedelta(hours=10),
            ],
            "active_buy_amt": [60.0, 60.0, future_buy],
            "active_sell_amt": [40.0, 40.0, 1.0],
            "adjfactor": [1.0, 1.0, 1.0],
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_value_and_signal_time(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_flow_frame()),
        )
        self.assertEqual(len(out), 1)
        assert_standard_bartimes(out)
        self.assertAlmostEqual(float(out.iloc[0]["value"]), 0.2)

    def test_future_bar_cannot_change_0959_signal(self):
        a = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_flow_frame(future_buy=10.0)),
        )
        b = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_flow_frame(future_buy=1_000_000.0)),
        )
        self.assertAlmostEqual(float(a.iloc[0]["value"]), float(b.iloc[0]["value"]))

    def test_sql_uses_ordered_session_cumsum(self):
        assert_no_future_leakage_contract()


class TestProductionPath(unittest.TestCase):
    def test_build_intraday_narrow_table_routes_to_ddb(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["active_buy_sell_imbalance"],
                "value": [0.2],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.active_buy_sell_imbalance.compute.ddb_version",
            return_value=fake,
        ) as mock_ddb:
            with unittest.mock.patch.object(
                cfg, "INTRADAY_ACTIVE_BUY_SELL_IMBALANCE_USE_DDB", True
            ):
                out = build_intraday_narrow_table(
                    "active_buy_sell_imbalance",
                    "2024-05-01",
                    "2024-05-31",
                )
        mock_ddb.assert_called_once()
        self.assertEqual(
            list(out.columns), ["tradetime", "symbol", "factorname", "value"]
        )


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DDB integration",
)
class TestDdbIntegration(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        start, end = "2024-05-01", "2024-05-31"
        py = python_version(start, end)
        db = ddb_version(start, end)
        self.assertFalse(py.empty, "python narrow empty")
        self.assertFalse(db.empty, "ddb narrow empty")
        print(assert_consistency(py, db))


if __name__ == "__main__":
    unittest.main(verbosity=2)
