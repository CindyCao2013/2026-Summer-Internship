"""Formula, leakage and Python/DDB parity gates for active_buy_shock."""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict

import numpy as np
import pandas as pd

from factors.intraday.active_buy_shock.compute import (
    FACTOR_NAME,
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
    """Validate keys, values, ranking, bartimes and no-look-ahead SQL."""
    assert_no_future_leakage_contract()
    assert_bartime_alignment(python_narrow, ddb_narrow)

    keys = ["bartime", "symbol"]
    for label, frame in (("python", python_narrow), ("ddb", ddb_narrow)):
        if frame.duplicated(keys).any():
            raise AssertionError(f"{label} contains duplicate signal keys")
    py_keys = set(map(tuple, python_narrow[keys].astype(str).to_numpy()))
    db_keys = set(map(tuple, ddb_narrow[keys].astype(str).to_numpy()))
    if py_keys != db_keys:
        raise AssertionError(
            f"Signal-key mismatch: python_only={len(py_keys - db_keys)}, "
            f"ddb_only={len(db_keys - py_keys)}"
        )

    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError(f"No overlapping {FACTOR_NAME} signals")
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


def _shock_frame(future_buy: float = 40.0) -> pd.DataFrame:
    day = pd.Timestamp("2024-05-06")
    bartimes = pd.date_range(
        day + pd.Timedelta(hours=9, minutes=39),
        day + pd.Timedelta(hours=10),
        freq="min",
    )
    active_buy = list(np.arange(1.0, 21.0)) + [30.0, future_buy]
    return pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(bartimes),
            "date": [day] * len(bartimes),
            "bartime": bartimes,
            "active_buy_amt": active_buy,
            "adjfactor": [2.0] * len(bartimes),
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
            "active_buy_amt": np.arange(1.0, len(bartimes) + 1.0),
            "adjfactor": np.linspace(0.9, 1.1, len(bartimes)),
        }
    )


class TestFormulaContracts(unittest.TestCase):
    def test_prior_20_sample_zscore_uses_adjusted_amount(self):
        out = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_shock_frame()),
        )
        self.assertEqual(len(out), 1)
        history = np.arange(1.0, 21.0) * 2.0
        expected = (60.0 - history.mean()) / history.std(ddof=1)
        self.assertAlmostEqual(float(out.iloc[0]["value"]), expected)

    def test_future_bar_cannot_change_0959_signal(self):
        early = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_shock_frame(future_buy=1.0)),
        )
        late = python_version(
            "2024-05-06",
            "2024-05-06",
            store=_FrameStore(_shock_frame(future_buy=1_000_000.0)),
        )
        self.assertAlmostEqual(
            float(early.iloc[0]["value"]),
            float(late.iloc[0]["value"]),
        )

    def test_rolling_baseline_resets_each_session(self):
        first = _shock_frame().iloc[:21].copy()
        day = pd.Timestamp("2024-05-07")
        second = pd.DataFrame(
            {
                "symbol": ["600000.SH"] * 5,
                "date": [day] * 5,
                "bartime": pd.date_range(
                    day + pd.Timedelta(hours=9, minutes=55),
                    periods=5,
                    freq="min",
                ),
                "active_buy_amt": np.arange(101.0, 106.0),
                "adjfactor": [1.0] * 5,
            }
        )
        out = python_version(
            "2024-05-06",
            "2024-05-07",
            store=_FrameStore(pd.concat([first, second], ignore_index=True)),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(
            pd.Timestamp(out.iloc[0]["bartime"]).date(),
            first.iloc[0]["date"].date(),
        )

    def test_only_five_standard_bartimes_are_emitted(self):
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

    def test_sql_uses_positive_shifted_session_baseline(self):
        assert_no_future_leakage_contract()


class TestProductionPath(unittest.TestCase):
    def test_registry_routes_enabled_factor_to_shared_ddb_builder(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": [FACTOR_NAME],
                "value": [2.0],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.discovery_v1.ddb_version",
            return_value=fake,
        ) as mock_ddb:
            with unittest.mock.patch.object(
                cfg, "INTRADAY_ACTIVE_BUY_SHOCK_USE_DDB", True
            ):
                out = build_intraday_narrow_table(
                    FACTOR_NAME,
                    "2024-05-01",
                    "2024-05-31",
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
