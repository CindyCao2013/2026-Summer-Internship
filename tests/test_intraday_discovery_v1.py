"""Shared formula and integration gates for Intraday Alpha Expansion v1."""

from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

from core.ddb_intraday_queries import DISCOVERY_V1_FACTORS
from factors.intraday.discovery_v1 import (
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


def _minute_frame(*, future_multiplier: float = 1.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    bartime = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=31, freq="min")
    x = np.arange(1, len(bartime) + 1, dtype=float)
    buy = 100.0 + 3.0 * x
    sell = 90.0 + 2.0 * x
    buy[-1] *= future_multiplier  # 10:00 is after the 09:59 signal.
    return pd.DataFrame(
        {
            "symbol": "600000.SH",
            "date": day,
            "bartime": bartime,
            "close": 10.0 + 0.01 * np.sin(x),
            "amount": 1_000_000.0 + 1_000.0 * x,
            "active_buy_amt": buy,
            "active_sell_amt": sell,
            "active_buy_count": 10.0 + (x % 4),
            "active_sell_count": 9.0 + (x % 3),
            "adjfactor": 1.0,
        }
    )


class TestDiscoveryV1Contracts(unittest.TestCase):
    def test_all_factors_emit_standard_signal(self):
        store = _FrameStore(_minute_frame())
        for factor_name in sorted(DISCOVERY_V1_FACTORS):
            with self.subTest(factor=factor_name):
                result = python_version(
                    factor_name,
                    "2024-05-06",
                    "2024-05-06",
                    store=store,
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(
                    pd.Timestamp(result.iloc[0]["bartime"]).strftime("%H:%M"),
                    "09:59",
                )
                self.assertTrue(np.isfinite(float(result.iloc[0]["value"])))

    def test_future_bar_cannot_change_0959(self):
        left = _FrameStore(_minute_frame(future_multiplier=1.0))
        right = _FrameStore(_minute_frame(future_multiplier=1_000_000.0))
        for factor_name in sorted(DISCOVERY_V1_FACTORS):
            with self.subTest(factor=factor_name):
                a = python_version(
                    factor_name, "2024-05-06", "2024-05-06", store=left
                )
                b = python_version(
                    factor_name, "2024-05-06", "2024-05-06", store=right
                )
                self.assertAlmostEqual(
                    float(a.iloc[0]["value"]),
                    float(b.iloc[0]["value"]),
                    places=14,
                )

    def test_sql_no_lookahead_contracts(self):
        for factor_name in sorted(DISCOVERY_V1_FACTORS):
            with self.subTest(factor=factor_name):
                assert_no_future_leakage_contract(factor_name)

    def test_bounded_ratio_factors(self):
        store = _FrameStore(_minute_frame())
        for factor_name in ("bartime_ofi", "ofi_persistence", "large_active_buy_ratio"):
            result = python_version(
                factor_name, "2024-05-06", "2024-05-06", store=store
            )
            lower = -1.0 if factor_name == "bartime_ofi" else 0.0
            self.assertTrue(result["value"].between(lower, 1.0).all())

    def test_backend_registry_is_ddb_and_not_panel(self):
        from core.intraday_alphas import (
            INTRADAY_ALPHA_COMPUTERS,
            INTRADAY_FACTOR_BACKEND,
            PANEL_BASED_INTRADAY_FACTORS,
        )

        for factor_name in DISCOVERY_V1_FACTORS:
            self.assertIn(factor_name, INTRADAY_ALPHA_COMPUTERS)
            self.assertEqual(INTRADAY_FACTOR_BACKEND[factor_name], "ddb")
            self.assertNotIn(factor_name, PANEL_BASED_INTRADAY_FACTORS)


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DDB integration",
)
class TestDiscoveryV1DdbIntegration(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        for factor_name in sorted(DISCOVERY_V1_FACTORS):
            with self.subTest(factor=factor_name):
                py = python_version(
                    factor_name,
                    "2024-05-06",
                    "2024-05-10",
                    symbols=["000001.SZ", "600000.SH"],
                )
                db = ddb_version(
                    factor_name,
                    "2024-05-06",
                    "2024-05-10",
                    symbols=["000001.SZ", "600000.SH"],
                )
                assert_bartime_alignment(py, db)
                aligned = align_narrow(py, db)
                self.assertEqual(len(py), len(db))
                self.assertEqual(len(aligned), len(py))
                self.assertLessEqual(float(aligned["abs_diff"].max()), 1e-10)
                spearman = aligned.groupby("bartime").apply(
                    lambda g: g["python"].corr(g["ddb"], method="spearman")
                )
                self.assertGreaterEqual(float(spearman.dropna().min()), 0.999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
