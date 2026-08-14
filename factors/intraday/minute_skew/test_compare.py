"""Consistency gates for minute_skew."""

from __future__ import annotations

import os
import unittest
import unittest.mock

import numpy as np
import pandas as pd

from factors.intraday.minute_skew.compute import (
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    ddb_version,
    python_version,
)


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _frame(future_close: float = 10.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    times = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=31, freq="min")
    returns = np.tile([0.001, -0.002, 0.003, -0.001], 8)[:31]
    close = 10.0 * np.cumprod(1.0 + returns)
    close[-1] = future_close
    return pd.DataFrame(
        {
            "symbol": "600000.SH",
            "date": day,
            "bartime": times,
            "close": close,
            "amount": 1_000_000.0,
            "active_buy_amt": 100.0,
            "active_sell_amt": 90.0,
            "active_buy_count": 10.0,
            "active_sell_count": 9.0,
            "adjfactor": 1.0,
        }
    )


def assert_consistency(py: pd.DataFrame, db: pd.DataFrame) -> dict:
    assert_no_future_leakage_contract()
    assert_bartime_alignment(py, db)
    aligned = align_narrow(py, db)
    if aligned.empty:
        raise AssertionError("No overlapping minute_skew signals")
    max_diff = float(aligned["abs_diff"].max())
    if max_diff > 1e-10:
        raise AssertionError({"max_abs_diff": max_diff})
    spear = aligned.groupby("bartime").apply(
        lambda g: g["python"].corr(g["ddb"], method="spearman")
    ).dropna()
    if len(spear) and float(spear.min()) < 0.999:
        raise AssertionError(spear.describe())
    return {
        "aligned_rows": len(aligned),
        "max_abs_diff": max_diff,
        "spearman_min": float(spear.min()) if len(spear) else None,
    }


class TestFormulaContracts(unittest.TestCase):
    def test_matches_pandas_expanding_skew(self):
        frame = _frame()
        out = python_version("2024-05-06", "2024-05-06", store=_FrameStore(frame))
        expected = frame["close"].pct_change().expanding(min_periods=3).skew().iloc[29]
        self.assertAlmostEqual(float(out.iloc[0]["value"]), float(expected), places=14)

    def test_future_bar_cannot_change_0959(self):
        a = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(_frame(10.0))
        )
        b = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(_frame(1_000_000.0))
        )
        self.assertAlmostEqual(float(a.iloc[0]["value"]), float(b.iloc[0]["value"]))

    def test_sql_uses_ordered_cumulative_moments(self):
        assert_no_future_leakage_contract()


class TestProductionPath(unittest.TestCase):
    def test_registry_routes_to_ddb(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59")],
                "symbol": ["600000.SH"],
                "factorname": ["minute_skew"],
                "value": [0.1],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.discovery_v1.ddb_version", return_value=fake
        ) as mocked:
            with unittest.mock.patch.object(cfg, "INTRADAY_MINUTE_SKEW_USE_DDB", True):
                result = build_intraday_narrow_table(
                    "minute_skew", "2024-05-06", "2024-05-06"
                )
        mocked.assert_called_once()
        self.assertEqual(
            list(result.columns), ["tradetime", "symbol", "factorname", "value"]
        )


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DDB integration",
)
class TestDdbIntegration(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        py = python_version("2024-05-06", "2024-05-10")
        db = ddb_version("2024-05-06", "2024-05-10")
        self.assertFalse(py.empty)
        self.assertFalse(db.empty)
        print(assert_consistency(py, db))


if __name__ == "__main__":
    unittest.main(verbosity=2)
