"""Formula, causality, range, and optional live parity tests."""

from __future__ import annotations

import os
import unittest
from typing import Dict

import numpy as np
import pandas as pd

from core.ddb_intraday_queries import discovery_v1_factor_script
from factors.intraday.ofi_persistence.compute import (
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    assert_standard_bartimes,
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
    """Validate numeric, ranking, timestamp, range, and causality contracts."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)
    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping ofi_persistence signals")

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
    if len(python_narrow) != len(ddb_narrow):
        raise AssertionError(
            f"Signal row mismatch: {len(python_narrow)} != {len(ddb_narrow)}"
        )
    for label, frame in (("python", python_narrow), ("ddb", ddb_narrow)):
        if not frame["value"].between(0, 1).all():
            raise AssertionError(f"{label} values outside [0, 1]")

    return {
        "aligned_rows": len(aligned),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(aligned["abs_diff"].mean()),
        "spearman_min": min_spearman,
        "python_rows": len(python_narrow),
        "ddb_rows": len(ddb_narrow),
        "bartimes": sorted(
            pd.to_datetime(aligned["bartime"]).dt.strftime("%H:%M").unique()
        ),
    }


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _causality_frame(future_buy: float, future_sell: float) -> pd.DataFrame:
    bartimes = pd.date_range("2024-05-06 09:55", periods=6, freq="min")
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 6,
            "date": [pd.Timestamp("2024-05-06")] * 6,
            "bartime": bartimes,
            "active_buy_amt": [2.0, 1.0, 2.0, 1.0, 2.0, future_buy],
            "active_sell_amt": [1.0, 2.0, 1.0, 1.0, 1.0, future_sell],
            "adjfactor": [1.0] * 6,
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_current_bar_is_included_after_five_valid_observations(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_causality_frame(1000.0, 1.0)),
        )
        self.assertEqual(len(out), 1)
        assert_standard_bartimes(out)
        self.assertAlmostEqual(float(out.iloc[0]["value"]), 3.0 / 5.0)

    def test_future_bar_cannot_change_earlier_signal(self):
        positive_future = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_causality_frame(1_000_000.0, 1.0)),
        )
        negative_future = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_causality_frame(1.0, 1_000_000.0)),
        )
        pd.testing.assert_frame_equal(positive_future, negative_future)

    def test_trailing_twenty_and_range(self):
        bartimes = pd.date_range("2024-05-06 09:30", periods=30, freq="min")
        frame = pd.DataFrame(
            {
                "symbol": ["600000.SH"] * 30,
                "date": [pd.Timestamp("2024-05-06")] * 30,
                "bartime": bartimes,
                "active_buy_amt": [2.0] * 10 + [1.0] * 20,
                "active_sell_amt": [1.0] * 10 + [2.0] * 20,
                "adjfactor": [1.0] * 30,
            }
        )
        full_day = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(frame),
            return_full_day=True,
        )
        self.assertTrue(full_day["value"].between(0, 1).all())
        self.assertAlmostEqual(float(full_day.iloc[-1]["value"]), 0.0)

    def test_ddb_builder_has_ordered_trailing_window(self):
        script = discovery_v1_factor_script(
            "ofi_persistence", "2024-05-01", "2024-05-31"
        )
        for fragment in (
            "msum(iif(isValid(bar_ofi)",
            "mcount(bar_ofi, 20, 5)",
            "context by Symbol, Date csort Bartime",
            "where Bartime in btFilter",
        ):
            self.assertIn(fragment, script)
        assert_no_future_leakage_contract()


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
