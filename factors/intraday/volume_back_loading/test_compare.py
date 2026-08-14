"""Sprint 3.3 consistency gates for volume_back_loading."""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.volume_back_loading.compute import (
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    assert_signal_time,
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
        raise AssertionError("No overlapping volume_back_loading signals")

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
    if len(python_narrow) != len(ddb_narrow):
        raise AssertionError(
            f"Signal row mismatch: {len(python_narrow)} != {len(ddb_narrow)}"
        )

    return {
        "aligned_rows": len(aligned),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(aligned["abs_diff"].mean()),
        "spearman_min": min_spearman,
        "signal_time": "09:59",
        "python_rows": len(python_narrow),
        "ddb_rows": len(ddb_narrow),
    }


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


class TestFormulaContracts(unittest.TestCase):
    def test_signal_time(self):
        narrow = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-07 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["volume_back_loading"],
                "value": [1.0],
            }
        )
        assert_signal_time(narrow)

    def test_sql_uses_prior_sessions(self):
        assert_no_future_leakage_contract(lookback_days=10)

    def test_python_denominator_excludes_current_day(self):
        dates = pd.bdate_range("2024-05-01", periods=11)
        frame = pd.DataFrame(
            {
                "symbol": ["600000.SH"] * len(dates),
                "date": dates,
                "bartime": dates + pd.Timedelta(hours=14, minutes=30),
                "volume": [100.0] * 10 + [1000.0],
            }
        )
        signal_date = dates[-1] + pd.offsets.BDay(1)
        out = python_version(
            dates[-1],
            signal_date,
            store=_FrameStore(frame),
            lookback_days=10,
        )
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["value"]), 10.0)

    def test_friday_signal_stamps_monday(self):
        source = pd.Timestamp("2024-05-10")
        raw = pd.DataFrame(
            {"Symbol": ["600000.SH"], "Date": [source], "value": [1.0]}
        )
        from factors.intraday.volume_back_loading.compute import (
            _normalize_ddb_narrow,
        )

        out = _normalize_ddb_narrow(raw, "2024-05-13", "2024-05-13")
        self.assertEqual(out.iloc[0]["bartime"], pd.Timestamp("2024-05-13 09:59"))


class TestProductionPath(unittest.TestCase):
    def test_build_intraday_narrow_table_routes_to_ddb(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-07 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["volume_back_loading"],
                "value": [1.2],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.volume_back_loading.compute.ddb_version",
            return_value=fake,
        ) as mock_ddb:
            with unittest.mock.patch.object(
                cfg, "INTRADAY_VOLUME_BACK_USE_DDB", True
            ):
                out = build_intraday_narrow_table(
                    "volume_back_loading",
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
