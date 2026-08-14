"""Three-level consistency: numeric, cross-sectional rank, backtest proxy.

Additional production gates (Sprint 3.1):
  - Check 1: bartime alignment / standard signal times only
  - Check 2: look-ahead contract (cumsum on csort Bartime, no forward move)
"""

from __future__ import annotations

import os
import unittest
import unittest.mock
from typing import Dict, Optional

import numpy as np
import pandas as pd

from factors.intraday.close_vwap_deviation.compute import (
    align_narrow,
    assert_bartime_alignment,
    assert_no_future_leakage_contract,
    assert_standard_bartimes,
    ddb_version,
    python_version,
)

NUMERIC_ATOL = 1e-10
SPEARMAN_MIN = 0.999
BACKTEST_REL_TOL = 0.01


def numeric_metrics(aligned: pd.DataFrame) -> Dict[str, float]:
    if aligned.empty:
        return {"n": 0, "max_abs_diff": np.nan, "mean_abs_diff": np.nan}
    diff = aligned["abs_diff"]
    return {
        "n": float(len(aligned)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def spearman_by_bartime(aligned: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman per signal time (Level 2)."""
    if aligned.empty:
        return pd.Series(dtype=float)

    def _spearman(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["python"].corr(group["ddb"], method="spearman")

    return aligned.groupby("bartime", sort=False).apply(_spearman)


def hl_sharpe_proxy(factor: pd.DataFrame) -> Optional[float]:
    """Simple long-short spread Sharpe proxy on cross-sectional mean by bartime."""
    if factor.empty:
        return None
    work = factor.copy()
    work["bartime"] = pd.to_datetime(work["bartime"])
    spreads = []
    for _, grp in work.groupby("bartime", sort=False):
        if len(grp) < 10:
            continue
        q_hi = grp["value"].quantile(0.9)
        q_lo = grp["value"].quantile(0.1)
        spreads.append(
            grp.loc[grp["value"] >= q_hi, "value"].mean()
            - grp.loc[grp["value"] <= q_lo, "value"].mean()
        )
    if len(spreads) < 2:
        return None
    s = pd.Series(spreads)
    if s.std() == 0:
        return 0.0
    return float(s.mean() / s.std() * np.sqrt(len(s)))


def backtest_proxy_metrics(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
) -> Dict[str, float]:
    py = hl_sharpe_proxy(python_narrow)
    db = hl_sharpe_proxy(ddb_narrow)
    if py is None or db is None or py == 0:
        rel_err = 0.0 if py == db else np.nan
    else:
        rel_err = abs(py - db) / abs(py)
    return {"python_hl_sharpe": py, "ddb_hl_sharpe": db, "hl_sharpe_rel_err": rel_err}


def assert_consistency(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
    *,
    numeric_atol: float = NUMERIC_ATOL,
    spearman_min: float = SPEARMAN_MIN,
    backtest_rel_tol: float = BACKTEST_REL_TOL,
    check_bartime: bool = True,
) -> Dict[str, object]:
    assert_no_future_leakage_contract()
    if check_bartime:
        assert_standard_bartimes(python_narrow)
        assert_standard_bartimes(ddb_narrow)
        assert_bartime_alignment(python_narrow, ddb_narrow)

    aligned = align_narrow(python_narrow, ddb_narrow)
    num = numeric_metrics(aligned)
    spear = spearman_by_bartime(aligned)
    bt = backtest_proxy_metrics(python_narrow, ddb_narrow)

    if num["n"] > 0:
        assert num["max_abs_diff"] <= numeric_atol, num
    valid_spear = spear.dropna()
    if len(valid_spear) > 0:
        assert float(valid_spear.min()) >= spearman_min, valid_spear.describe()
    if not np.isnan(bt.get("hl_sharpe_rel_err", np.nan)):
        assert bt["hl_sharpe_rel_err"] <= backtest_rel_tol, bt

    return {
        "numeric": num,
        "spearman_min": float(valid_spear.min()) if len(valid_spear) else None,
        "backtest": bt,
        "bartimes_python": sorted(
            {t.strftime("%H:%M") for t in pd.to_datetime(python_narrow["bartime"]).dt.time.unique()}
        ),
        "bartimes_ddb": sorted(
            {t.strftime("%H:%M") for t in pd.to_datetime(ddb_narrow["bartime"]).dt.time.unique()}
        ),
    }


class TestCompareLogic(unittest.TestCase):
    def test_numeric_and_spearman_on_identical(self):
        rows = []
        for i, sym in enumerate(["600000.SH", "000001.SZ", "000002.SZ"]):
            rows.append(
                {
                    "bartime": pd.Timestamp("2024-05-01 09:59:00"),
                    "symbol": sym,
                    "factorname": "close_vwap_deviation",
                    "value": 0.01 * (i + 1),
                }
            )
        narrow = pd.DataFrame(rows)
        metrics = assert_consistency(narrow, narrow.copy())
        self.assertEqual(metrics["numeric"]["max_abs_diff"], 0.0)
        self.assertEqual(metrics["spearman_min"], 1.0)

    def test_detects_bartime_shift(self):
        py = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-01 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["close_vwap_deviation"],
                "value": [0.1],
            }
        )
        shifted = py.copy()
        shifted["bartime"] = pd.Timestamp("2024-05-01 10:00:00")
        with self.assertRaises(AssertionError):
            assert_standard_bartimes(shifted)

    def test_detects_numeric_drift(self):
        base = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-01 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["close_vwap_deviation"],
                "value": [0.1],
            }
        )
        drift = base.copy()
        drift["value"] = 0.2
        aligned = align_narrow(base, drift)
        self.assertAlmostEqual(numeric_metrics(aligned)["max_abs_diff"], 0.1)


class TestProductionPath(unittest.TestCase):
    def test_build_intraday_narrow_table_routes_to_ddb(self):
        import factor_config as cfg
        from intraday_formulas import build_intraday_narrow_table

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-01 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["close_vwap_deviation"],
                "value": [0.01],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.close_vwap_deviation.compute.ddb_version",
            return_value=fake,
        ) as mock_ddb:
            with unittest.mock.patch.object(cfg, "INTRADAY_CLOSE_VWAP_USE_DDB", True):
                out = build_intraday_narrow_table(
                    "close_vwap_deviation",
                    "2024-05-01",
                    "2024-05-10",
                    store=None,
                )
        mock_ddb.assert_called_once()
        self.assertIn("tradetime", out.columns)


    def test_flag_routes_to_ddb(self):
        from core.intraday_alphas import compute_close_vwap_deviation

        fake = pd.DataFrame(
            {
                "bartime": [pd.Timestamp("2024-05-01 09:59:00")],
                "symbol": ["600000.SH"],
                "factorname": ["close_vwap_deviation"],
                "value": [0.01],
            }
        )
        with unittest.mock.patch(
            "factors.intraday.close_vwap_deviation.compute.ddb_version",
            return_value=fake,
        ) as mock_ddb:
            with unittest.mock.patch(
                "core.intraday_alphas._close_vwap_use_ddb",
                return_value=True,
            ):
                out = compute_close_vwap_deviation("2024-05-01", "2024-05-10")
        mock_ddb.assert_called_once()
        self.assertEqual(len(out), 1)


@unittest.skipUnless(
    os.environ.get("RUN_DDB_TESTS") == "1",
    "Set RUN_DDB_TESTS=1 to run live DDB integration",
)
class TestDdbIntegration(unittest.TestCase):
    def test_python_vs_ddb_live(self):
        start, end = "2024-05-01", "2024-05-10"
        py = python_version(start, end)
        db = ddb_version(start, end)
        self.assertFalse(py.empty, "python narrow empty")
        self.assertFalse(db.empty, "ddb narrow empty")
        metrics = assert_consistency(py, db)
        print(metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
