"""Contract and gated live-parity tests for large_active_buy_ratio."""

from __future__ import annotations

import os
import unittest
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.large_active_buy_ratio.compute import (
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    assert_standard_bartimes,
    assert_unit_interval,
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
    """Validate value, rank, timestamp, range, and no-look-ahead contracts."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)
    assert_unit_interval(python_narrow)
    assert_unit_interval(ddb_narrow)
    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping large_active_buy_ratio signals")

    max_diff = float(aligned["abs_diff"].max())

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["python"].corr(group["ddb"], method="spearman")

    spear = aligned.groupby("bartime", sort=False).apply(_spearman).dropna()
    min_spearman = float(spear.min()) if len(spear) else None
    if max_diff > numeric_atol:
        raise AssertionError({"max_abs_diff": max_diff})
    if min_spearman is not None and min_spearman < spearman_min:
        raise AssertionError(spear.describe())

    return {
        "aligned_rows": len(aligned),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(aligned["abs_diff"].mean()),
        "spearman_min": min_spearman,
        "python_rows": len(python_narrow),
        "ddb_rows": len(ddb_narrow),
    }


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _proxy_frame(future_buy: float = 10.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    bartimes = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=31, freq="min")
    buy_amount = np.full(31, 10.0)
    buy_amount[[20, 25]] = 20.0
    buy_amount[30] = future_buy
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(bartimes),
            "date": [day] * len(bartimes),
            "bartime": bartimes,
            "active_buy_amt": buy_amount,
            "active_buy_count": np.ones(len(bartimes)),
            "adjfactor": np.ones(len(bartimes)),
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_bar_level_proxy_formula_and_range(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_proxy_frame()),
        )
        self.assertEqual(len(out), 1)
        assert_standard_bartimes(out)
        assert_unit_interval(out)
        # At 09:59, bars 09:50 and 09:55 are classified. The trailing
        # numerator is 20 + 20 and the trailing buy amount is 18*10 + 2*20.
        self.assertAlmostEqual(float(out.iloc[0]["value"]), 40.0 / 220.0)

    def test_future_bar_cannot_change_0959_signal(self):
        earlier = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_proxy_frame(future_buy=1.0)),
        )
        shocked = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_proxy_frame(future_buy=1_000_000.0)),
        )
        self.assertEqual(len(earlier), 1)
        self.assertEqual(len(shocked), 1)
        self.assertAlmostEqual(
            float(earlier.iloc[0]["value"]),
            float(shocked.iloc[0]["value"]),
        )

    def test_sql_shifts_mean_and_sample_std_baseline(self):
        assert_no_future_leakage_contract()

    def test_unit_interval_rejects_out_of_range_values(self):
        bad = pd.DataFrame({"value": [-0.01, 1.01]})
        with self.assertRaises(AssertionError):
            assert_unit_interval(bad)


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
