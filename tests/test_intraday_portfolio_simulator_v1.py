from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_portfolio_simulator_v1 import (
    _assert_gross_parity,
    _build_positions,
    _read_checkpoint,
    _simulate_day,
    _simulation_hash,
)


def _constituents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["A", "B", "C", "D"],
            "Date": pd.to_datetime(["2024-07-01"] * 4),
            "Bartime": ["14:29:00"] * 4,
            "tradetime": pd.to_datetime(["2024-07-01 14:29"] * 4),
            "factor_group": [
                "group_0",
                "group_0",
                "group_9",
                "group_9",
            ],
            "asset_return": [-0.02, 0.00, 0.03, 0.01],
        }
    )


class TestPortfolioWeights(unittest.TestCase):
    def test_positive_direction_longs_high_and_shorts_low(self):
        positions = _build_positions(_constituents(), direction=1)
        long_symbols = set(positions.loc[positions["side"] == "long", "Symbol"])
        short_symbols = set(
            positions.loc[positions["side"] == "short", "Symbol"]
        )
        self.assertEqual(long_symbols, {"C", "D"})
        self.assertEqual(short_symbols, {"A", "B"})
        self.assertAlmostEqual(
            positions.loc[
                positions["side"] == "long", "entry_weight"
            ].sum(),
            0.5,
        )
        self.assertAlmostEqual(
            positions.loc[
                positions["side"] == "short", "entry_weight"
            ].sum(),
            -0.5,
        )

    def test_negative_direction_longs_low_and_shorts_high(self):
        positions = _build_positions(_constituents(), direction=-1)
        long_symbols = set(positions.loc[positions["side"] == "long", "Symbol"])
        short_symbols = set(
            positions.loc[positions["side"] == "short", "Symbol"]
        )
        self.assertEqual(long_symbols, {"A", "B"})
        self.assertEqual(short_symbols, {"C", "D"})


class TestTransactionLedger(unittest.TestCase):
    def test_cost_uses_actual_entry_and_exit_notional(self):
        positions = _build_positions(_constituents(), direction=1)
        row = _simulate_day(
            positions,
            factor_name="factor",
            period_name="validation",
            horizon="Ret_30",
            direction=1,
            one_way_cost_bps=7.5,
        )
        expected_gross = 0.5 * 0.02 - 0.5 * -0.01
        expected_exit = (
            np.abs(positions["entry_weight"])
            * (1.0 + positions["asset_return"])
        ).sum()
        expected_traded = 1.0 + expected_exit
        self.assertAlmostEqual(row["gross_ls_return"], expected_gross)
        self.assertAlmostEqual(row["entry_traded_notional"], 1.0)
        self.assertAlmostEqual(row["exit_traded_notional"], expected_exit)
        self.assertAlmostEqual(
            row["traded_notional_turnover"], expected_traded
        )
        self.assertAlmostEqual(row["half_l1_turnover"], expected_traded / 2)
        self.assertAlmostEqual(
            row["transaction_cost"],
            expected_traded * 7.5 / 1e4,
        )
        self.assertAlmostEqual(
            row["net_ls_return"],
            expected_gross - expected_traded * 7.5 / 1e4,
        )

    def test_zero_cost_gross_matches_half_directional_hl(self):
        positions = _build_positions(_constituents(), direction=1)
        row = _simulate_day(
            positions,
            factor_name="factor",
            period_name="validation",
            horizon="Ret_30",
            direction=1,
            one_way_cost_bps=0,
        )
        ledger = pd.DataFrame([row])
        group_ret = pd.DataFrame(
            {
                "group": ["group_HML"],
                "Date": pd.to_datetime(["2024-07-01"]),
                "Bartime": ["14:29:00"],
                "Ret_30": [0.03],
            }
        )
        max_diff = _assert_gross_parity(
            ledger,
            group_ret,
            bartime="14:29",
            horizon="Ret_30",
            direction=1,
        )
        self.assertLessEqual(max_diff, 1e-12)


class TestSimulatorContracts(unittest.TestCase):
    def test_checkpoint_is_bound_to_simulation_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            checkpoint = output / "checkpoints/factor/period"
            checkpoint.mkdir(parents=True)
            pd.DataFrame(
                {
                    "Date": ["2024-07-01"],
                    "gross_return": [0.01],
                }
            ).to_csv(checkpoint / "daily_ledger.csv", index=False)
            (checkpoint / "metadata.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "summary": {"simulation_sha256": "wrong"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                _read_checkpoint(
                    output,
                    "factor",
                    "period",
                    "expected",
                )

    def test_cost_change_changes_simulation_hash(self):
        self.assertNotEqual(
            _simulation_hash("freeze", 7.5),
            _simulation_hash("freeze", 5.0),
        )

    def test_source_does_not_use_fixed_turnover_or_reselection(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "research/intraday_portfolio_simulator_v1.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("intraday_turnover_b_hl", source)
        self.assertNotIn("_candidate_summary", source)
        self.assertNotIn("abs_icir", source)


if __name__ == "__main__":
    unittest.main()
