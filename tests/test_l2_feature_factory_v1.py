"""Tests for L2 Feature Factory v1 registry and SQL generation."""

from __future__ import annotations

import unittest

from research.l2_alpha.feature_factory.derived_sql import derived_feature_sql
from research.l2_alpha.feature_factory.primitives_sql import minute_primitives_sql
from research.l2_alpha.feature_factory.registry import (
    L2_FF_ALL_COLUMNS,
    expand_all_factor_names,
    expand_derived_names,
    max_lookback,
)


class TestRegistry(unittest.TestCase):
    def test_expands_to_twenty(self):
        names = expand_all_factor_names()
        self.assertEqual(len(names), 20)
        self.assertEqual(tuple(names), L2_FF_ALL_COLUMNS)
        self.assertIn("woi_delta5", names)
        self.assertIn("depth_imb_persistence20", names)
        self.assertIn("spread_mean_rank", names)

    def test_derived_exclude_ranks(self):
        derived = expand_derived_names()
        self.assertEqual(len(derived), 17)
        self.assertTrue(all(not n.endswith("_rank") for n in derived))

    def test_max_lookback(self):
        self.assertGreaterEqual(max_lookback(), 30)


class TestSQLGeneration(unittest.TestCase):
    def test_primitives_use_arrays(self):
        sql = minute_primitives_sql(
            table="SSE_AL_SSL2_EXG",
            exchange_suffix=".SH",
            start="2024-06-03",
            end="2024-06-04",
            has_withdraw=True,
        )
        self.assertIn("BidVolumes", sql)
        self.assertIn("arrayResize", sql)
        self.assertNotIn("BidPrice0", sql)
        self.assertIn("toStartOfMinute", sql)
        self.assertIn("avg(depth_oi)", sql)

    def test_derived_has_windows_and_outer_bartime_filter(self):
        sql = derived_feature_sql(
            table="SSE_AL_SSL2_EXG",
            exchange_suffix=".SH",
            start="2024-06-03",
            end="2024-06-04",
            has_withdraw=True,
            bartimes=["09:59", "14:29"],
        )
        self.assertIn("OVER (", sql)
        self.assertIn("lagInFrame", sql)
        self.assertIn("woi_delta5", sql)
        self.assertIn("depth_imb_persistence20", sql)
        # Outer filter only
        self.assertIn(
            "(toHour(minute_time), toMinute(minute_time)) IN", sql
        )
        # Must not filter ExchTime to bartimes inside primitives
        self.assertNotIn(
            "(toHour(ExchTime), toMinute(ExchTime)) IN", sql
        )

    def test_szse_cancel_null(self):
        sql = derived_feature_sql(
            table="SZSE_AL_SSL2_EXG",
            exchange_suffix=".SZ",
            start="2024-06-03",
            end="2024-06-04",
            has_withdraw=False,
            bartimes=["09:59"],
        )
        self.assertIn("cancel_mean", sql)
        self.assertIn("CAST(NULL AS Nullable(Float64)) AS cancel_mean", sql)


if __name__ == "__main__":
    unittest.main()
