from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from research.freeze_intraday_alpha_v1 import (
    DEFAULT_OUTPUT,
    EXPECTED_FACTORS,
    verify_spec,
)
from research.run_intraday_alpha_oos_v1 import (
    _fixed_metric,
    _residual_period_verdict,
    _residualize_frozen,
)


class TestFreezeSpecification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freeze = verify_spec(DEFAULT_OUTPUT)

    def test_factor_set_and_proxy_exclusion_are_locked(self):
        self.assertEqual(list(self.freeze["factors"]), EXPECTED_FACTORS)
        self.assertNotIn(
            "large_active_buy_ratio",
            self.freeze["factors"],
        )
        self.assertEqual(
            self.freeze["excluded_factors"]["large_active_buy_ratio"]["status"],
            "proxy_only",
        )

    def test_direction_is_train_raw_ic_sign(self):
        for factor_name, spec in self.freeze["factors"].items():
            expected = 1 if spec["train_raw_ic"] > 0 else -1
            self.assertEqual(
                spec["direction"],
                expected,
                factor_name,
            )

    def test_discovery_residual_controls_are_explicit(self):
        for factor_name, spec in self.freeze["factors"].items():
            if spec["backend"] == "candidate_ddb":
                self.assertTrue(spec["residual_controls"], factor_name)
                self.assertIn(spec["residual_direction"], (-1, 1))

    def test_oos_source_contains_no_best_tuple_search(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "research/run_intraday_alpha_oos_v1.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("_candidate_summary", source)
        self.assertNotIn("abs_icir", source)
        self.assertNotIn("nlargest(", source)


class TestFixedDirectionEvaluation(unittest.TestCase):
    def test_oos_hl_direction_is_not_inferred_from_oos_mean(self):
        spec = {
            "family": "volatility",
            "backend": "candidate_ddb",
            "research_role": "Base",
            "bartime": "14:29",
            "horizon": "Ret_30",
            "direction": -1,
            "portfolio_rule": "long_low_short_high",
            "train_raw_ic": -0.1,
            "train_raw_icir": -10.0,
            "train_fixed_direction_sharpe": 8.0,
        }
        group_ret = pd.DataFrame(
            {
                "Bartime": ["14:29:00", "14:29:00"],
                "group": ["group_HML", "group_HML"],
                "Date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
                "Ret_30": [0.02, 0.01],
            }
        )
        ic_mean = pd.DataFrame(
            {
                "Bartime": ["14:29:00"],
                "RetType": ["Ret_30"],
                "IC_Mean": [-0.03],
                "IC_IR": [-0.4],
            }
        )
        ic_ts = pd.DataFrame(
            {
                "Bartime": ["14:29:00", "14:29:00"],
                "RetType": ["Ret_30", "Ret_30"],
                "Rank_IC": [-0.02, -0.04],
            }
        )
        with mock.patch(
            "research.run_intraday_alpha_oos_v1."
            "intraday_lib.subtract_market_return",
            side_effect=lambda frame: frame,
        ):
            with mock.patch(
                "research.run_intraday_alpha_oos_v1."
                "fdl.calSharpe",
                side_effect=lambda values: float(np.mean(values)),
            ):
                with mock.patch(
                    "research.run_intraday_alpha_oos_v1."
                    "intraday_lib.intraday_turnover_b_hl",
                    return_value=4.0,
                ):
                    result = _fixed_metric(
                        "realized_volatility",
                        "validation_2024H2",
                        spec,
                        group_ret,
                        ic_mean,
                        ic_ts,
                    )
        self.assertGreater(result["oos_raw_hl_sharpe"], 0)
        self.assertLess(result["oos_fixed_direction_hl_sharpe"], 0)
        self.assertGreater(result["oos_signed_ic"], 0)
        self.assertGreater(result["oos_signed_icir"], 0)
        self.assertEqual(result["oos_fixed_direction_ic_win_rate"], 1.0)

    def test_frozen_residual_control_cannot_be_dropped(self):
        target = pd.DataFrame(
            {
                "tradetime": pd.to_datetime(["2024-07-01 10:29"]),
                "symbol": ["000001.SZ"],
                "factorname": ["bartime_ofi"],
                "value": [0.1],
            }
        )
        with self.assertRaisesRegex(ValueError, "was not built"):
            _residualize_frozen(
                "bartime_ofi",
                target,
                {},
                ["close_vwap_deviation"],
                "10:29",
            )

    def test_residual_sign_flip_is_drop(self):
        row = pd.Series(
            {
                "oos_residual_signed_ic": -0.001,
                "oos_residual_signed_icir": -0.2,
                "oos_residual_fixed_direction_hl_sharpe": 1.5,
            }
        )
        self.assertEqual(_residual_period_verdict(row), "drop")


if __name__ == "__main__":
    unittest.main()
