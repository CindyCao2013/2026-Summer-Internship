"""Unit tests for SSL2 Array-vector formulas (no database)."""

from __future__ import annotations

import unittest

import numpy as np

from research.l2_alpha.formulas import (
    cancel_pressure,
    compute_all_snapshot_features,
    depth_imbalance,
    liquidity_skew,
    liquidity_wall,
    microprice_bias,
    relative_spread,
    top_book_imbalance,
    weighted_order_imbalance,
)
from research.l2_alpha.schema import FACTOR_NAMES


class TestTopAndDepth(unittest.TestCase):
    def test_top_book_imbalance_sign(self):
        self.assertGreater(
            top_book_imbalance([100, 50], [40, 50]), 0.0
        )
        self.assertLess(
            top_book_imbalance([40, 50], [100, 50]), 0.0
        )
        self.assertAlmostEqual(
            top_book_imbalance([50], [50]), 0.0, places=12
        )

    def test_depth_imbalance_uses_all_levels(self):
        bid = [10] * 10
        ask = [5] * 10
        self.assertAlmostEqual(depth_imbalance(bid, ask), (100 - 50) / 150)


class TestWeightedOI(unittest.TestCase):
    def test_near_levels_dominate(self):
        # Huge size only at level 10 should matter less than top size.
        bid_near = [1000] + [0] * 9
        ask_near = [10] + [0] * 9
        bid_far = [10] + [0] * 8 + [1000]
        ask_far = [10] + [0] * 9
        near = weighted_order_imbalance(bid_near, ask_near, lam=0.5)
        far = weighted_order_imbalance(bid_far, ask_far, lam=0.5)
        self.assertGreater(near, far)


class TestMicropriceAndSpread(unittest.TestCase):
    def test_microprice_bias_pulls_toward_larger_size(self):
        # Larger ask size → microprice closer to bid → negative bias.
        bias = microprice_bias(
            bid_prices=[10.0, 9.9],
            ask_prices=[10.1, 10.2],
            bid_volumes=[100, 100],
            ask_volumes=[400, 100],
        )
        self.assertLess(bias, 0.0)

    def test_relative_spread_positive(self):
        spr = relative_spread([10.0], [10.2])
        self.assertAlmostEqual(spr, 0.2 / 10.1, places=10)


class TestCancelAndWall(unittest.TestCase):
    def test_cancel_pressure(self):
        self.assertGreater(cancel_pressure(80, 20), 0.0)
        self.assertTrue(np.isnan(cancel_pressure(0, 0)))

    def test_liquidity_wall_ratio(self):
        wall = liquidity_wall([10, 10, 100], [10, 10, 10])
        self.assertAlmostEqual(wall, 100 / 150)

    def test_liquidity_skew(self):
        skew = liquidity_skew([10.0], [10.2], bid_vwap=9.9, ask_vwap=10.3)
        # ask side farther → positive skew
        self.assertGreater(skew, 0.0)


class TestBundle(unittest.TestCase):
    def test_compute_all_keys(self):
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
        self.assertTrue(all(np.isfinite(v) for v in feats.values()))


if __name__ == "__main__":
    unittest.main()
