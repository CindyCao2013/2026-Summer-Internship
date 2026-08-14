import inspect
import json
import unittest

import numpy as np
import pandas as pd

from core.evaluation.intraday_metrics import (
    ANNUALIZATION_DAYS,
    GROUP_PANEL_COLUMNS,
    annualized_return,
    annualized_sharpe,
    benchmark_beta,
    benchmark_correlation,
    build_group_excess_panel,
    build_hl_panel,
    summarize_cross_sectional_metrics,
    summarize_ic_series,
)
from core.intraday_alphas import INTRADAY_FACTOR_BACKEND
from research import intraday_portfolio_simulator_v1 as simulator
from research import plot_intraday_evaluation_v2 as plotting
from research import run_intraday_evaluation_v2 as evaluation


class TestExactMarketExcess(unittest.TestCase):
    def setUp(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        self.market = pd.DataFrame(
            {
                "Date": dates,
                "Bartime": ["09:59", "09:59"],
                "return_window": ["Ret_30", "Ret_30"],
                "n_market_assets": [4, 4],
                "market_return": [0.025, 0.015],
            }
        )
        self.groups = pd.DataFrame(
            {
                "Date": [dates[0], dates[0], dates[1], dates[1]],
                "Bartime": ["09:59"] * 4,
                "return_window": ["Ret_30"] * 4,
                "group": ["group_0", "group_9", "group_0", "group_9"],
                "n_assets": [1, 3, 1, 3],
                "group_return_raw": [0.10, 0.00, 0.08, -0.01],
            }
        )

    def test_constituent_ew_market_is_not_mean_of_group_means(self):
        panel = build_group_excess_panel(self.groups, self.market)
        first = panel[panel["Date"] == pd.Timestamp("2024-01-02")]
        self.assertEqual(list(panel.columns), GROUP_PANEL_COLUMNS)
        self.assertAlmostEqual(first["market_return"].iloc[0], 0.025)
        self.assertNotAlmostEqual(
            first["market_return"].iloc[0],
            (0.10 + 0.00) / 2.0,
        )
        g1 = first[first["group"] == "G1"].iloc[0]
        self.assertAlmostEqual(g1["group_return_excess"], 0.075)

    def test_hl_excess_equivalence_and_direction(self):
        panel = build_group_excess_panel(self.groups, self.market)
        negative = build_hl_panel(panel, direction=-1)
        np.testing.assert_allclose(
            negative["raw_hl_return"],
            negative["excess_hl_return"],
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            negative["hl_return"],
            -negative["raw_hl_return"],
            atol=0,
            rtol=0,
        )
        positive = build_hl_panel(panel, direction=1)
        np.testing.assert_allclose(
            positive["hl_return"],
            positive["raw_hl_return"],
            atol=0,
            rtol=0,
        )

    def test_missing_exact_market_is_rejected(self):
        with self.assertRaises(ValueError):
            build_group_excess_panel(
                self.groups,
                self.market.iloc[:1],
            )


class TestBetaDiagnostics(unittest.TestCase):
    def test_known_beta_and_correlation(self):
        market = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
        hl = 2.0 * market
        self.assertAlmostEqual(benchmark_beta(hl, market), 2.0)
        self.assertAlmostEqual(benchmark_correlation(hl, market), 1.0)

    def test_beta_is_diagnostic_not_neutrality_gate(self):
        source = inspect.getsource(evaluation)
        self.assertNotIn("abs(beta)", source)
        self.assertNotIn("hl_market_beta] < 0.05", source)


class TestMetricContract(unittest.TestCase):
    def test_250_day_annualization(self):
        self.assertEqual(ANNUALIZATION_DAYS, 250)
        values = pd.Series([0.01, -0.005, 0.015, 0.0])
        self.assertAlmostEqual(
            annualized_return(values),
            values.mean() * 250,
        )
        self.assertAlmostEqual(
            annualized_sharpe(values),
            values.mean() / values.std(ddof=1) * np.sqrt(250),
        )
        ic = summarize_ic_series(
            [0.01, -0.005, 0.015, 0.0],
            direction=1,
        )
        self.assertAlmostEqual(
            ic["annualized_icir"],
            values.mean() / values.std(ddof=1) * np.sqrt(250),
        )
        self.assertAlmostEqual(ic["ic_win_rate"], 0.5)

    def test_v2_cross_sectional_schema(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
        )
        panel = pd.DataFrame(
            {
                "Date": dates,
                "Bartime": ["09:59"] * 4,
                "return_window": ["Ret_30"] * 4,
                "group": ["G1", "G10", "G1", "G10"],
                "n_assets": [10] * 4,
                "n_market_assets": [100] * 4,
                "group_return_raw": [0.00, 0.02, -0.01, 0.01],
                "market_return": [0.005, 0.005, 0.002, 0.002],
                "group_return_excess": [
                    -0.005,
                    0.015,
                    -0.012,
                    0.008,
                ],
            }
        )
        hl = build_hl_panel(panel, direction=1)
        result = summarize_cross_sectional_metrics(
            panel,
            hl,
            factor_name="synthetic",
        )
        required = {
            "metric_scope",
            "group_return_raw",
            "group_return_excess",
            "group_excess_sharpe",
            "raw_hl_return",
            "hl_return",
            "hl_sharpe",
            "hl_market_beta",
            "hl_market_corr",
            "direction_consistent",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertEqual(
            set(result["metric_scope"]),
            {"cross_sectional_group", "cross_sectional_hl"},
        )


class TestExecutionAndRegressionContracts(unittest.TestCase):
    def test_simulator_emits_explicit_ls_names(self):
        constituents = pd.DataFrame(
            {
                "Symbol": ["A", "B", "C", "D"],
                "Date": [pd.Timestamp("2024-01-02")] * 4,
                "Bartime": ["09:59"] * 4,
                "factor_group": [
                    "group_0",
                    "group_0",
                    "group_9",
                    "group_9",
                ],
                "asset_return": [0.01, 0.02, -0.01, -0.02],
            }
        )
        positions = simulator._build_positions(constituents, direction=1)
        row = simulator._simulate_day(
            positions,
            factor_name="synthetic",
            period_name="test",
            horizon="Ret_30",
            direction=1,
            one_way_cost_bps=7.5,
        )
        self.assertIn("gross_ls_return", row)
        self.assertIn("net_ls_return", row)
        self.assertNotIn("gross_return", row)
        self.assertNotIn("net_return", row)

    def test_adapter_does_not_mutate_factor_registry(self):
        self.assertEqual(
            {
                name: INTRADAY_FACTOR_BACKEND[name]
                for name in evaluation.ALL_FACTORS
            },
            {name: "ddb" for name in evaluation.ALL_FACTORS},
        )

    def test_legacy_consumers_normalize_before_analyzer(self):
        for filename in (
            "run_p2_intraday_heatmap.py",
            "intraday_Factortest.py",
        ):
            source = (evaluation.ROOT / filename).read_text(encoding="utf-8")
            self.assertLess(
                source.index("subtract_market_return(group_data_ret)"),
                source.index("analyze_group_performance_by_bartime"),
            )


class TestGeneratedV2Artifacts(unittest.TestCase):
    def setUp(self):
        self.output = (
            evaluation.ROOT / "research/results/intraday_evaluation_v2"
        )
        if not (self.output / "performance_all_v2.csv").exists():
            self.skipTest("Versioned DDB artifacts have not been generated")

    def test_complete_versioned_schema(self):
        summary = json.loads(
            (self.output / "summary.json").read_text(encoding="utf-8")
        )
        candidates = pd.read_csv(
            self.output / "intraday_alpha_library_v3_candidates.csv"
        )
        oos = pd.read_csv(
            self.output / "frozen_oos_diagnostics_v2.csv"
        )
        self.assertEqual(summary["annualization_days"], 250)
        self.assertEqual(
            summary["market_return"],
            "exact_filtered_constituent_equal_weight",
        )
        self.assertEqual(summary["beta_policy"], "diagnostic_only")
        self.assertEqual(set(candidates["factor"]), set(evaluation.ALL_FACTORS))
        self.assertEqual(len(oos), 2 * len(evaluation.ALL_FACTORS))
        required = {
            "rank_ic",
            "annualized_icir",
            "ic_win_rate",
            "g1_excess_sharpe",
            "g10_excess_sharpe",
            "hl_sharpe",
            "hl_market_beta",
            "hl_market_corr",
        }
        self.assertTrue(required.issubset(candidates.columns))
        self.assertFalse(candidates[list(required)].isna().any().any())

    def test_oos_directions_match_freeze(self):
        oos = pd.read_csv(
            self.output / "frozen_oos_diagnostics_v2.csv"
        )
        freeze = json.loads(
            (
                evaluation.ROOT
                / "research/config/intraday_alpha_freeze_v1.json"
            ).read_text(encoding="utf-8")
        )
        for row in oos.itertuples():
            self.assertEqual(
                int(row.direction),
                int(freeze["factors"][row.factor]["direction"]),
            )

    def test_execution_v2_has_only_explicit_ls_metric_names(self):
        execution = pd.read_csv(
            evaluation.ROOT
            / "research/results/intraday_portfolio_simulator_v1"
            / "intraday_portfolio_cost_v2.csv"
        )
        self.assertTrue(
            any("gross_ls_sharpe" in column for column in execution.columns)
        )
        self.assertTrue(
            any("net_ls_sharpe" in column for column in execution.columns)
        )
        self.assertFalse(
            any(column.endswith("_gross_sharpe") for column in execution.columns)
        )
        self.assertFalse(
            any(column.endswith("_net_sharpe") for column in execution.columns)
        )

    def test_plot_gate_selects_retained_monotonic_factors(self):
        performance = pd.read_csv(self.output / "performance_all_v2.csv")
        candidates = pd.read_csv(
            self.output / "intraday_alpha_library_v3_candidates.csv"
        )
        oos = pd.read_csv(
            evaluation.ROOT
            / "research/results/intraday_alpha_oos_v1"
            / "intraday_alpha_oos_v1.csv"
        )
        selected = plotting.select_qualified_factors(
            performance,
            candidates,
            oos,
        )
        qualified = set(selected.loc[selected["qualified"], "factor"])
        self.assertEqual(
            qualified,
            {
                "close_vwap_deviation",
                "intraday_amihud",
                "realized_volatility",
            },
        )
        self.assertNotIn("minute_skew", qualified)


if __name__ == "__main__":
    unittest.main()
