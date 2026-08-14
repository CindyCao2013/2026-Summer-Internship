"""Unit tests for Sprint 4.3 RV exposure audit (synthetic panels only)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.evaluation.rv_exposure_audit import (
    VERDICT_CASE_A,
    VERDICT_CASE_B,
    VERDICT_MIXED,
    build_audit_summary,
    classify_verdict,
    dominant_ic_drop_step,
    exposure_correlations,
    fama_macbeth,
    industry_demean_panel,
    progressive_fama_macbeth,
    progressive_residual_ic_chain,
    rank_zscore,
    residualize_signal,
)


def _make_independent_alpha_panel(n_days: int = 40, n_names: int = 80) -> pd.DataFrame:
    """RV predicts excess return with only weak size correlation."""
    rng = np.random.default_rng(7)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        size = rng.normal(size=n_names)
        # Mostly independent RV plus a little size bleed.
        rv = 0.15 * size + rng.normal(size=n_names)
        noise = rng.normal(scale=0.4, size=n_names)
        ret_excess = -0.8 * rv + 0.05 * size + noise
        for i in range(n_names):
            rows.append(
                {
                    "Date": date,
                    "symbol": f"S{i:03d}",
                    "rv": float(rv[i]),
                    "ret_excess": float(ret_excess[i]),
                    "ret_raw": float(ret_excess[i]),
                    "size": float(size[i]),
                    "liquidity": float(rng.normal()),
                    "hist_vol": float(0.2 * rv[i] + rng.normal()),
                    "momentum_20d": float(rng.normal()),
                    "session_mom": float(rng.normal()),
                    "industry": f"I{i % 8}",
                }
            )
    return pd.DataFrame(rows)


def _make_style_proxy_panel(n_days: int = 40, n_names: int = 80) -> pd.DataFrame:
    """RV is almost size; return comes from size, not residual RV."""
    rng = np.random.default_rng(11)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        size = rng.normal(size=n_names)
        rv = size + 0.05 * rng.normal(size=n_names)
        ret_excess = -0.7 * size + rng.normal(scale=0.3, size=n_names)
        for i in range(n_names):
            rows.append(
                {
                    "Date": date,
                    "symbol": f"S{i:03d}",
                    "rv": float(rv[i]),
                    "ret_excess": float(ret_excess[i]),
                    "ret_raw": float(ret_excess[i]),
                    "size": float(size[i]),
                    "liquidity": float(size[i] + rng.normal(scale=0.1)),
                    "hist_vol": float(rv[i]),
                    "momentum_20d": float(rng.normal()),
                    "session_mom": float(rng.normal()),
                    "industry": f"I{i % 8}",
                }
            )
    return pd.DataFrame(rows)


class TestRankZAndFM(unittest.TestCase):
    def test_rank_zscore_zero_mean_unit_std(self):
        z = rank_zscore(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertAlmostEqual(z.mean(), 0.0, places=12)
        self.assertAlmostEqual(z.std(ddof=0), 1.0, places=12)

    def test_fama_macbeth_recovers_negative_rv_loading(self):
        panel = _make_independent_alpha_panel()
        summary = fama_macbeth(panel, "ret_excess", ["rv"])
        rv_row = summary[summary["variable"] == "rv"].iloc[0]
        self.assertLess(rv_row["mean_coef"], 0.0)
        self.assertLess(rv_row["tstat_nw"], -2.0)


class TestResidualChain(unittest.TestCase):
    def test_independent_alpha_keeps_retention(self):
        panel = _make_independent_alpha_panel()
        chain = progressive_residual_ic_chain(
            panel, "rv", "ret_excess", controls=("size", "liquidity")
        )
        self.assertEqual(chain.iloc[0]["step"], "raw")
        self.assertGreater(chain.iloc[-1]["retention"], 0.50)
        self.assertLess(chain.iloc[0]["raw_ic"], 0.0)

    def test_style_proxy_kills_retention_after_size(self):
        panel = _make_style_proxy_panel()
        chain = progressive_residual_ic_chain(
            panel, "rv", "ret_excess", controls=("size",)
        )
        self.assertLess(chain.iloc[-1]["retention"], 0.30)
        self.assertEqual(dominant_ic_drop_step(chain), "size")

    def test_residualize_removes_linear_control(self):
        panel = _make_style_proxy_panel(n_days=5, n_names=60)
        resid = residualize_signal(panel, "rv", ["size"])
        work = panel.assign(resid=resid)
        # After size residualization, corr(resid, size) should be near 0.
        corrs = []
        for _, g in work.groupby("Date"):
            sub = g[["resid", "size"]].dropna()
            corrs.append(sub["resid"].corr(sub["size"]))
        self.assertLess(abs(np.nanmean(corrs)), 0.05)


class TestVerdict(unittest.TestCase):
    def test_case_a(self):
        self.assertEqual(
            classify_verdict(
                rv_tstat_full=-8.0,
                rv_mean_coef_full=-0.2,
                residual_retention=0.8,
            ),
            VERDICT_CASE_A,
        )

    def test_case_b_weak_t(self):
        self.assertEqual(
            classify_verdict(
                rv_tstat_full=-1.2,
                rv_mean_coef_full=-0.05,
                residual_retention=0.7,
            ),
            VERDICT_CASE_B,
        )

    def test_case_b_low_retention(self):
        self.assertEqual(
            classify_verdict(
                rv_tstat_full=-5.0,
                rv_mean_coef_full=-0.2,
                residual_retention=0.1,
            ),
            VERDICT_CASE_B,
        )

    def test_mixed(self):
        self.assertEqual(
            classify_verdict(
                rv_tstat_full=-3.0,
                rv_mean_coef_full=-0.1,
                residual_retention=0.4,
            ),
            VERDICT_MIXED,
        )

    def test_build_audit_summary_end_to_end(self):
        panel = _make_independent_alpha_panel()
        fm = progressive_fama_macbeth(
            panel, "ret_excess", "rv", controls=("size", "liquidity")
        )
        chain = progressive_residual_ic_chain(
            panel, "rv", "ret_excess", controls=("size", "liquidity")
        )
        corr = exposure_correlations(
            panel, "rv", ["size", "liquidity"]
        )
        summary = build_audit_summary(fm, chain, corr)
        self.assertEqual(summary["factor"], "realized_volatility")
        self.assertEqual(summary["bartime"], "14:29")
        self.assertIn(
            summary["verdict"],
            {VERDICT_CASE_A, VERDICT_CASE_B, VERDICT_MIXED},
        )
        self.assertEqual(summary["verdict"], VERDICT_CASE_A)


class TestIndustryDemean(unittest.TestCase):
    def test_within_industry_mean_near_zero(self):
        panel = _make_independent_alpha_panel(n_days=3, n_names=40)
        demeaned = industry_demean_panel(panel, ["rv", "ret_excess"])
        day = demeaned[demeaned["Date"] == demeaned["Date"].iloc[0]]
        for _, bucket in day.groupby("industry"):
            self.assertAlmostEqual(bucket["rv"].mean(), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
