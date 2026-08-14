"""Formula, leakage, timestamp, and live parity gates for intraday_amihud."""

from __future__ import annotations

import datetime as dt
import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.intraday_amihud.compute import (
    FACTOR_NAME,
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    assert_standard_bartimes,
    compute_factor,
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
    """Apply all signal-level Python/DDB acceptance gates."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)

    keys = ["bartime", "symbol"]
    py_keys = python_narrow[keys].assign(
        bartime=lambda x: pd.to_datetime(x["bartime"])
    )
    db_keys = ddb_narrow[keys].assign(
        bartime=lambda x: pd.to_datetime(x["bartime"])
    )
    if len(py_keys) != len(db_keys) or set(map(tuple, py_keys.to_numpy())) != set(
        map(tuple, db_keys.to_numpy())
    ):
        raise AssertionError("Signal key mismatch between Python and DolphinDB")

    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping intraday_amihud signals")
    if not np.isfinite(aligned[["python", "ddb"]].to_numpy()).all():
        raise AssertionError("Non-finite values reached parity comparison")
    if (aligned[["python", "ddb"]] < 0).any().any():
        raise AssertionError("Amihud illiquidity must be non-negative")

    max_diff = float(aligned["abs_diff"].max())
    if max_diff > numeric_atol:
        raise AssertionError({"max_abs_diff": max_diff})

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["python"].corr(group["ddb"], method="spearman")

    spearman = aligned.groupby("bartime", sort=False).apply(_spearman).dropna()
    min_spearman = float(spearman.min()) if len(spearman) else None
    if min_spearman is not None and min_spearman < spearman_min:
        raise AssertionError(spearman.describe())

    return {
        "aligned_rows": len(aligned),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(aligned["abs_diff"].mean()),
        "spearman_min": min_spearman,
        "bartimes": sorted(
            pd.to_datetime(aligned["bartime"]).dt.strftime("%H:%M").unique()
        ),
    }


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _frame(
    times: pd.DatetimeIndex,
    *,
    adjfactor: float = 1.0,
    amount: float = 100.0,
) -> pd.DataFrame:
    closes = 100.0 * np.power(1.01, np.arange(len(times)))
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(times),
            "date": times.normalize(),
            "bartime": times,
            "close": closes,
            "amount": [amount] * len(times),
            "adjfactor": [adjfactor] * len(times),
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_formula_and_minimum_three_returns(self):
        times = pd.date_range("2024-05-06 09:56:00", periods=4, freq="min")
        unit_adj = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(_frame(times))
        )
        double_adj = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_frame(times, adjfactor=2.0)),
        )

        self.assertEqual(len(unit_adj), 1)
        self.assertEqual(unit_adj.iloc[0]["factorname"], FACTOR_NAME)
        # Three valid 1% returns / four adjusted amount rows.
        self.assertAlmostEqual(float(unit_adj.iloc[0]["value"]), 0.000075)
        self.assertAlmostEqual(float(double_adj.iloc[0]["value"]), 0.000075)
        self.assertAlmostEqual(
            float(unit_adj.iloc[0]["value"]),
            float(double_adj.iloc[0]["value"]),
        )

    def test_fewer_than_three_returns_emits_no_signal(self):
        times = pd.date_range("2024-05-06 09:57:00", periods=3, freq="min")
        out = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(_frame(times))
        )
        self.assertTrue(out.empty)

    def test_future_bar_cannot_change_0959_signal(self):
        times = pd.date_range("2024-05-06 09:54:00", periods=7, freq="min")
        base = _frame(times)
        changed = base.copy()
        future = changed["bartime"].dt.time == dt.time(10, 0)
        changed.loc[future, "close"] = 1_000_000.0
        changed.loc[future, "amount"] = 1.0

        left = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(base)
        )
        right = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(changed)
        )
        self.assertEqual(len(left), 1)
        self.assertAlmostEqual(float(left.iloc[0]["value"]), float(right.iloc[0]["value"]))

    def test_session_reset_blocks_prior_close(self):
        day1_times = pd.date_range("2024-05-03 09:56:00", periods=4, freq="min")
        day2_times = pd.date_range("2024-05-06 09:56:00", periods=4, freq="min")
        base = pd.concat([_frame(day1_times), _frame(day2_times)], ignore_index=True)
        changed = base.copy()
        changed.loc[changed["date"] == pd.Timestamp("2024-05-03"), "close"] *= 1000

        left = python_version(
            "2024-05-03", "2024-05-06", store=_FrameStore(base)
        )
        right = python_version(
            "2024-05-03", "2024-05-06", store=_FrameStore(changed)
        )
        day2 = pd.Timestamp("2024-05-06")
        left_value = left.loc[left["bartime"].dt.normalize() == day2, "value"].iloc[0]
        right_value = right.loc[right["bartime"].dt.normalize() == day2, "value"].iloc[0]
        self.assertAlmostEqual(float(left_value), float(right_value))

    def test_emits_exactly_five_standard_bartimes(self):
        morning = pd.date_range(
            "2024-05-06 09:30:00", "2024-05-06 11:30:00", freq="min"
        )
        afternoon = pd.date_range(
            "2024-05-06 13:00:00", "2024-05-06 14:29:00", freq="min"
        )
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_frame(morning.append(afternoon))),
        )
        assert_standard_bartimes(out)
        self.assertEqual(
            set(out["bartime"].dt.time),
            {
                dt.time(9, 59),
                dt.time(10, 29),
                dt.time(11, 29),
                dt.time(13, 29),
                dt.time(14, 29),
            },
        )

    def test_ddb_contract_is_session_ordered_and_trailing(self):
        assert_no_future_leakage_contract()

    def test_wrapper_binds_discovery_factor_name(self):
        sentinel = pd.DataFrame()
        with unittest.mock.patch(
            "factors.intraday.intraday_amihud.compute.discovery_v1.compute_factor",
            return_value=sentinel,
        ) as shared:
            out = compute_factor("2024-05-01", "2024-05-02")
        self.assertIs(out, sentinel)
        self.assertEqual(shared.call_args.args[0], FACTOR_NAME)


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DDB parity",
)
class TestDdbIntegration(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        start, end = "2024-05-01", "2024-05-31"
        py = python_version(start, end)
        db = ddb_version(start, end)
        self.assertFalse(py.empty, "Python narrow output is empty")
        self.assertFalse(db.empty, "DolphinDB narrow output is empty")
        print(assert_consistency(py, db))

    def test_configured_live_path_matches_direct_ddb(self):
        start, end = "2024-05-06", "2024-05-10"
        with unittest.mock.patch.dict(
            os.environ, {"INTRADAY_AMIHUD_USE_DDB": "true"}
        ):
            routed = compute_factor(start, end)
        direct = ddb_version(start, end)
        self.assertFalse(routed.empty, "Configured DDB output is empty")
        print(assert_consistency(routed, direct))


if __name__ == "__main__":
    unittest.main(verbosity=2)
