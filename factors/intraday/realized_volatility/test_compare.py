"""Non-live contracts and gated live parity for realized_volatility."""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.realized_volatility.compute import (
    FACTOR_NAME,
    STANDARD_BARTIMES,
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
    """Apply numeric, alignment, causality, and domain parity gates."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)
    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping realized_volatility signals")
    if len(aligned) != len(python_narrow) or len(aligned) != len(ddb_narrow):
        raise AssertionError("Signal row mismatch between Python and DolphinDB")

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
    for label, frame in (("python", python_narrow), ("ddb", ddb_narrow)):
        values = pd.to_numeric(frame["value"], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise AssertionError(f"{label} contains non-finite values")
        if (values < 0).any():
            raise AssertionError(f"{label} contains negative realized volatility")

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


def _session_frame(future_close: float = 1_000.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    minutes = ["09:30", "09:31", "09:32", "09:33", "09:34", "09:59", "10:00"]
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(minutes),
            "date": [day] * len(minutes),
            "bartime": [pd.Timestamp(f"2024-05-06 {minute}") for minute in minutes],
            "close": [100.0, 101.0, 99.0, 100.0, 102.0, 101.0, future_close],
            "adjfactor": [1.0] * len(minutes),
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_value_is_session_to_current_sqrt_sum_squared_simple_returns(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_session_frame()),
        )
        closes = np.array([100.0, 101.0, 99.0, 100.0, 102.0, 101.0])
        expected = float(np.sqrt(np.square(closes[1:] / closes[:-1] - 1).sum()))
        self.assertEqual(len(out), 1)
        assert_standard_bartimes(out)
        self.assertAlmostEqual(float(out.iloc[0]["value"]), expected)

    def test_min5_and_five_standard_bartimes(self):
        self.assertEqual(
            tuple(value.strftime("%H:%M") for value in STANDARD_BARTIMES),
            ("09:59", "10:29", "11:29", "13:29", "14:29"),
        )
        too_short = _session_frame().iloc[:5].copy()
        too_short.loc[too_short.index[-1], "bartime"] = pd.Timestamp(
            "2024-05-06 09:59"
        )
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(too_short),
        )
        self.assertTrue(out.empty)

    def test_future_bar_cannot_alter_prior_signal(self):
        low = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_session_frame(future_close=1.0)),
        )
        high = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_session_frame(future_close=1_000_000.0)),
        )
        pd.testing.assert_frame_equal(low, high)

    def test_values_are_nonnegative_and_session_resets(self):
        first = _session_frame().iloc[:-1].copy()
        second = first.copy()
        second["date"] = pd.Timestamp("2024-05-07")
        second["bartime"] = second["bartime"] + pd.Timedelta(days=1)
        second["close"] = 50.0
        out = python_version(
            "2024-05-06",
            "2024-05-07",
            store=_FrameStore(pd.concat([first, second], ignore_index=True)),
        )
        self.assertTrue((out["value"] >= 0).all())
        is_next_day = (
            pd.to_datetime(out["bartime"]).dt.date
            == pd.Timestamp("2024-05-07").date()
        )
        next_day = out[is_next_day]
        self.assertEqual(len(next_day), 1)
        self.assertEqual(float(next_day.iloc[0]["value"]), 0.0)

    def test_sql_uses_ordered_session_cumsum(self):
        assert_no_future_leakage_contract()


class TestProductionDispatch(unittest.TestCase):
    def test_build_intraday_narrow_table_dispatches_to_gated_ddb(self):
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": [FACTOR_NAME],
                "value": [0.02],
            }
        )
        flag = "INTRADAY_REALIZED_VOLATILITY_USE_DDB"
        with unittest.mock.patch.dict(os.environ, {flag: "true"}):
            with unittest.mock.patch(
                "factors.intraday.discovery_v1.ddb_version",
                return_value=fake,
            ) as mock_ddb:
                out = build_intraday_narrow_table(
                    FACTOR_NAME,
                    "2024-05-06",
                    "2024-05-06",
                )
        mock_ddb.assert_called_once()
        self.assertEqual(
            list(out.columns), ["tradetime", "symbol", "factorname", "value"]
        )
        self.assertEqual(out.iloc[0]["factorname"], FACTOR_NAME)


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DolphinDB parity",
)
class TestDdbIntegration(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        start, end = "2024-05-01", "2024-05-31"
        py = python_version(start, end)
        db = ddb_version(start, end)
        self.assertFalse(py.empty, "python narrow empty")
        self.assertFalse(db.empty, "DolphinDB narrow empty")
        print(assert_consistency(py, db))


if __name__ == "__main__":
    unittest.main(verbosity=2)
