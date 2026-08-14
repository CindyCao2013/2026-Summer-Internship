"""Tests for Phase-2 L2 registry, panel helpers, and SQL generation."""

from __future__ import annotations

import unittest

import pandas as pd

from research.l2_alpha.clickhouse_ssl2 import (
    minute_agg_feature_sql,
    minute_last_feature_sql,
    snapshot_feature_select_sql,
)
from research.l2_alpha.l2_factor_panel import (
    filter_bartimes,
    minute_wide_to_long,
    to_evaluation_signal,
)
from research.l2_alpha.l2_factor_registry import (
    DEFAULT_BARTIMES,
    L2_PHASE2_FACTORS,
)
from research.l2_alpha.formulas import compute_all_snapshot_features
from research.l2_alpha.schema import FACTOR_NAMES


class TestRegistry(unittest.TestCase):
    def test_first_four_only(self):
        self.assertEqual(len(L2_PHASE2_FACTORS), 4)
        self.assertIn("l2_weighted_oi_mean", L2_PHASE2_FACTORS)
        self.assertIn("l2_cancel_pressure_sum", L2_PHASE2_FACTORS)
        self.assertTrue(L2_PHASE2_FACTORS["l2_cancel_pressure_sum"]["sse_only"])

    def test_bartimes_match_preheat_grid(self):
        self.assertIn("09:59", DEFAULT_BARTIMES)
        self.assertIn("14:29", DEFAULT_BARTIMES)
        self.assertNotIn("10:00", DEFAULT_BARTIMES)
        self.assertNotIn("14:30", DEFAULT_BARTIMES)


class TestSQLGeneration(unittest.TestCase):
    def test_snapshot_sql_uses_arrays_not_expanded_cols(self):
        sql = snapshot_feature_select_sql(
            table="SSE_AL_SSL2_EXG",
            exchange_suffix=".SH",
            start="2024-06-03",
            end="2024-06-04",
            has_withdraw=True,
        )
        self.assertIn("BidVolumes", sql)
        self.assertIn("arrayResize", sql)
        self.assertNotIn("BidPrice0", sql)
        self.assertIn("BidWithdrawVolume", sql)

    def test_szse_sql_omits_withdraw_columns(self):
        sql = snapshot_feature_select_sql(
            table="SZSE_AL_SSL2_EXG",
            exchange_suffix=".SZ",
            start="2024-06-03",
            end="2024-06-04",
            has_withdraw=False,
        )
        self.assertNotIn("BidWithdrawVolume", sql)

    def test_minute_agg_sql_has_mean_max_std_sum(self):
        sql = minute_agg_feature_sql(
            table="SSE_AL_SSL2_EXG",
            exchange_suffix=".SH",
            start="2024-06-03",
            end="2024-06-04",
            has_withdraw=True,
            bartimes=["09:59", "14:29"],
        )
        self.assertIn("avg(weighted_oi) AS l2_weighted_oi_mean", sql)
        self.assertIn("max(weighted_oi) AS l2_weighted_oi_max", sql)
        self.assertIn("stddevPop(weighted_oi) AS l2_weighted_oi_std", sql)
        self.assertIn("avg(micro_bias) AS l2_microprice_bias_mean", sql)
        self.assertIn("l2_cancel_pressure_sum", sql)
        self.assertIn("sum(cancel_signed)", sql)
        self.assertIn("(toHour(ExchTime), toMinute(ExchTime)) IN", sql)
        self.assertIn("(9, 59)", sql)
        self.assertIn("(14, 29)", sql)
        # No pandas-side aggregation markers
        self.assertNotIn("groupby", sql.lower())

    def test_legacy_last_sql_still_builds(self):
        sql = minute_last_feature_sql(
            table="SSE_AL_SSL2_EXG",
            exchange_suffix=".SH",
            start="2024-06-03",
            end="2024-06-04",
        )
        self.assertIn("argMax", sql)


class TestPanelHelpers(unittest.TestCase):
    def test_wide_to_long_and_signal(self):
        wide = pd.DataFrame(
            {
                "minute_time": pd.to_datetime(
                    ["2024-06-03 09:59:00", "2024-06-03 14:29:00"]
                ),
                "symbol": ["600000.SH", "600000.SH"],
                "l2_weighted_oi_mean": [0.1, 0.2],
                "l2_microprice_bias_mean": [0.001, -0.001],
            }
        )
        long = minute_wide_to_long(
            wide,
            factor_columns=[
                "l2_weighted_oi_mean",
                "l2_microprice_bias_mean",
            ],
            aggregation_map={
                "l2_weighted_oi_mean": "mean",
                "l2_microprice_bias_mean": "mean",
            },
        )
        self.assertEqual(len(long), 4)
        self.assertEqual(set(long["bartime"]), {"09:59", "14:29"})
        filtered = filter_bartimes(long, ["09:59"])
        self.assertEqual(set(filtered["bartime"]), {"09:59"})
        signal = to_evaluation_signal(filtered, "l2_weighted_oi_mean")
        self.assertEqual(list(signal.columns), [
            "tradetime", "symbol", "factorname", "value"
        ])
        self.assertEqual(len(signal), 1)
        self.assertEqual(signal.iloc[0]["factorname"], "l2_weighted_oi_mean")


class TestFormulasStillIntact(unittest.TestCase):
    def test_reference_bundle(self):
        feats = compute_all_snapshot_features(
            bid_prices=[10.0] * 10,
            ask_prices=[10.1] * 10,
            bid_volumes=[100] * 10,
            ask_volumes=[80] * 10,
            bid_withdraw_volume=10,
            ask_withdraw_volume=5,
            bid_vwap=9.95,
            ask_vwap=10.15,
        )
        self.assertEqual(set(feats), set(FACTOR_NAMES))


if __name__ == "__main__":
    unittest.main()
