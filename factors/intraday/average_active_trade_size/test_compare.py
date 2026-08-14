"""Contract, dispatch, and gated live-parity tests for the factor."""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.average_active_trade_size.compute import (
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
FLAG = "INTRADAY_AVERAGE_ACTIVE_TRADE_SIZE_USE_DDB"


def assert_consistency(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
    *,
    numeric_atol: float = NUMERIC_ATOL,
    spearman_min: float = SPEARMAN_MIN,
) -> Dict[str, object]:
    """Validate formula, rank, timestamp, universe, and leakage contracts."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)
    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping average_active_trade_size signals")
    if len(python_narrow) != len(ddb_narrow) or len(aligned) != len(python_narrow):
        raise AssertionError(
            "Signal key mismatch: "
            f"python={len(python_narrow)}, ddb={len(ddb_narrow)}, "
            f"aligned={len(aligned)}"
        )

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
        "bartimes": sorted(
            pd.to_datetime(aligned["bartime"]).dt.strftime("%H:%M").unique()
        ),
    }


class _FrameStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def _ticket_frame(future_amount: float = 100.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    bartimes = pd.date_range(
        day + pd.Timedelta(hours=9, minutes=30), periods=31, freq="min"
    )
    amounts = np.full(len(bartimes), 100.0)
    amounts[29] = 200.0
    amounts[30] = future_amount
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(bartimes),
            "date": [day] * len(bartimes),
            "bartime": bartimes,
            "active_buy_amt": amounts,
            "active_buy_count": np.full(len(bartimes), 10.0),
            "adjfactor": np.ones(len(bartimes)),
        }
    )


def _full_session_frame() -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    bartimes = pd.date_range(
        day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=11, minutes=30),
        freq="min",
    ).append(
        pd.date_range(
            day + pd.Timedelta(hours=13),
            day + pd.Timedelta(hours=15),
            freq="min",
        )
    )
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(bartimes),
            "date": [day] * len(bartimes),
            "bartime": bartimes,
            "active_buy_amt": np.full(len(bartimes), 100.0),
            "active_buy_count": np.full(len(bartimes), 10.0),
            "adjfactor": np.ones(len(bartimes)),
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_current_ticket_over_shifted_prior_mean(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_ticket_frame()),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(pd.Timestamp(out.iloc[0]["bartime"]).strftime("%H:%M"), "09:59")
        self.assertAlmostEqual(float(out.iloc[0]["value"]), 1.0)

    def test_prior_window_requires_ten_valid_tickets(self):
        frame = _ticket_frame()
        frame["active_buy_count"] = 0.0
        frame.loc[20:28, "active_buy_count"] = 10.0
        frame.loc[29, "active_buy_count"] = 10.0
        self.assertTrue(
            python_version(
                "2024-05-06", "2024-05-06", store=_FrameStore(frame)
            ).empty
        )
        frame.loc[19, "active_buy_count"] = 10.0
        out = python_version(
            "2024-05-06", "2024-05-06", store=_FrameStore(frame)
        )
        self.assertEqual(len(out), 1)

    def test_emits_only_the_five_standard_bartimes(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_full_session_frame()),
        )
        assert_standard_bartimes(out)
        self.assertEqual(
            set(pd.to_datetime(out["bartime"]).dt.strftime("%H:%M")),
            {"09:59", "10:29", "11:29", "13:29", "14:29"},
        )

    def test_future_bar_cannot_change_0959_signal(self):
        normal = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_ticket_frame(future_amount=1.0)),
        )
        shocked = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_ticket_frame(future_amount=1_000_000.0)),
        )
        self.assertAlmostEqual(
            float(normal.iloc[0]["value"]),
            float(shocked.iloc[0]["value"]),
        )

    def test_sql_uses_positive_shifted_baseline(self):
        assert_no_future_leakage_contract()


class TestDispatch(unittest.TestCase):
    def test_wrapper_dispatches_to_ddb_when_flag_enabled(self):
        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59")],
                "symbol": ["600000.SH"],
                "factorname": [FACTOR_NAME],
                "value": [1.0],
            }
        )
        with unittest.mock.patch.dict(os.environ, {FLAG: "true"}):
            with unittest.mock.patch(
                "factors.intraday.discovery_v1.ddb_version",
                return_value=fake,
            ) as mock_ddb:
                out = compute_factor("2024-05-06", "2024-05-06")
        mock_ddb.assert_called_once()
        self.assertIs(out, fake)
        self.assertEqual(mock_ddb.call_args.args[0], FACTOR_NAME)

    def test_production_registry_dispatches_factor(self):
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59")],
                "symbol": ["600000.SH"],
                "factorname": [FACTOR_NAME],
                "value": [1.0],
            }
        )
        with unittest.mock.patch.dict(os.environ, {FLAG: "true"}):
            with unittest.mock.patch(
                "factors.intraday.discovery_v1.ddb_version",
                return_value=fake,
            ) as mock_ddb:
                out = build_intraday_narrow_table(
                    FACTOR_NAME, "2024-05-06", "2024-05-06"
                )
        mock_ddb.assert_called_once()
        self.assertEqual(
            list(out.columns), ["tradetime", "symbol", "factorname", "value"]
        )


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
