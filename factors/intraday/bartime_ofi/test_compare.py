"""Formula, no-look-ahead, production-route and parity gates for bartime_ofi."""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from core.ddb_intraday_queries import discovery_v1_factor_script
from factors.intraday.bartime_ofi.compute import (
    FACTOR_NAME,
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
    """Enforce exact keys, numeric parity, rank parity and formula bounds."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)
    keys = ["bartime", "symbol"]
    for label, frame in (("python", python_narrow), ("ddb", ddb_narrow)):
        if frame.duplicated(keys).any():
            raise AssertionError(f"{label} contains duplicate signal keys")
        if not frame["value"].between(-1.0, 1.0).all():
            raise AssertionError(f"{label} values outside [-1, 1]")

    py_keys = set(map(tuple, python_narrow[keys].assign(
        bartime=pd.to_datetime(python_narrow["bartime"])
    ).to_numpy()))
    db_keys = set(map(tuple, ddb_narrow[keys].assign(
        bartime=pd.to_datetime(ddb_narrow["bartime"])
    ).to_numpy()))
    if py_keys != db_keys:
        raise AssertionError(
            f"Signal-key mismatch: python_only={len(py_keys - db_keys)}, "
            f"ddb_only={len(db_keys - py_keys)}"
        )

    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping bartime_ofi signals")
    max_diff = float(aligned["abs_diff"].max())
    if max_diff > numeric_atol:
        raise AssertionError({"max_abs_diff": max_diff})

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["python"].corr(group["ddb"], method="spearman")

    ranks = aligned.groupby("bartime", sort=False).apply(_spearman).dropna()
    min_spearman = float(ranks.min()) if len(ranks) else None
    if min_spearman is not None and min_spearman < spearman_min:
        raise AssertionError(ranks.describe())
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


def _flow_frame(*, future_buy: float = 1_000.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    times = ["09:59", "10:00", "10:29", "11:29", "13:29", "14:29"]
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(times),
            "date": [day] * len(times),
            "bartime": [pd.Timestamp(f"{day.date()} {time}") for time in times],
            "active_buy_amt": [60.0, future_buy, 30.0, 0.0, 10.0, 5.0],
            "active_sell_amt": [40.0, 1.0, 10.0, 10.0, 0.0, 5.0],
            "adjfactor": [2.0] * len(times),
        }
    )


class TestFormulaGate(unittest.TestCase):
    def test_current_minute_formula_at_five_standard_bartimes(self):
        out = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(_flow_frame())
        )
        assert_standard_bartimes(out)
        self.assertEqual(
            pd.to_datetime(out["bartime"]).dt.strftime("%H:%M").tolist(),
            ["09:59", "10:29", "11:29", "13:29", "14:29"],
        )
        np.testing.assert_allclose(out["value"], [0.2, 0.5, -1.0, 1.0, 0.0])
        self.assertTrue((out["factorname"] == FACTOR_NAME).all())


class TestLocalParityGate(unittest.TestCase):
    def test_identical_narrow_frames_pass(self):
        bartime = pd.Timestamp("2024-05-06 09:59:00")
        frame = pd.DataFrame(
            {
                "bartime": [bartime] * 3,
                "symbol": ["000001.SZ", "600000.SH", "600001.SH"],
                "factorname": [FACTOR_NAME] * 3,
                "value": [-0.4, 0.1, 0.8],
            }
        )
        metrics = assert_consistency(frame, frame.copy())
        self.assertEqual(metrics["aligned_rows"], 3)
        self.assertEqual(metrics["max_abs_diff"], 0.0)
        self.assertAlmostEqual(metrics["spearman_min"], 1.0)


class TestNoLookAheadGate(unittest.TestCase):
    def test_future_minute_cannot_change_0959_signal(self):
        low = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_flow_frame(future_buy=1.0)),
        )
        high = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_flow_frame(future_buy=1_000_000.0)),
        )
        self.assertEqual(float(low.iloc[0]["value"]), float(high.iloc[0]["value"]))

    def test_generated_sql_uses_current_bar_only(self):
        assert_no_future_leakage_contract()
        script = discovery_v1_factor_script(
            FACTOR_NAME, "2024-05-01", "2024-05-31"
        )
        self.assertIn("(buy_amt - sell_amt) \\ (buy_amt + sell_amt)", script)
        self.assertIn("bar_ofi as value", script)
        self.assertNotIn("cumsum(buy_amt)", script)
        self.assertNotIn("cumsum(sell_amt)", script)


class TestProductionRouteGate(unittest.TestCase):
    def test_build_intraday_narrow_table_routes_to_shared_ddb(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": [FACTOR_NAME],
                "value": [0.2],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.discovery_v1.ddb_version", return_value=fake
        ) as mock_ddb:
            with unittest.mock.patch.object(
                cfg, "INTRADAY_BARTIME_OFI_USE_DDB", True
            ):
                out = build_intraday_narrow_table(
                    FACTOR_NAME, "2024-05-01", "2024-05-31"
                )
        mock_ddb.assert_called_once()
        self.assertEqual(
            list(out.columns), ["tradetime", "symbol", "factorname", "value"]
        )

    def test_disabled_flag_routes_to_explicit_python_fallback(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": [FACTOR_NAME],
                "value": [0.2],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.discovery_v1.python_version", return_value=fake
        ) as mock_python:
            with unittest.mock.patch.object(
                cfg, "INTRADAY_BARTIME_OFI_USE_DDB", False
            ):
                build_intraday_narrow_table(
                    FACTOR_NAME, "2024-05-01", "2024-05-31"
                )
        mock_python.assert_called_once()


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DDB parity",
)
class TestLiveParityGate(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        start, end = "2024-05-01", "2024-05-31"
        py = python_version(start, end)
        db = ddb_version(start, end)
        self.assertFalse(py.empty, "python narrow empty")
        self.assertFalse(db.empty, "ddb narrow empty")
        print(assert_consistency(py, db))


if __name__ == "__main__":
    unittest.main(verbosity=2)
