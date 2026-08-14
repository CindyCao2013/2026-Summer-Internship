"""Tests for return_distribution (2.5) and tgd (3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.l2_features.return_distribution import (
    compute_return_distribution,
    compute_return_distribution_daily,
)
from core.l2_features.tgd import build_tgd20, daily_tgd_innovation, smooth_tgd, tgd20_to_wide


def test_avg_up_down_are_conditional_means_not_all_mean():
    # three up +0.02, two down -0.04, two zeros
    r = np.array([0.02, 0.02, 0.02, -0.04, -0.04, 0.0, 0.0, np.nan])
    d = compute_return_distribution(r)
    assert d.avg_up_return == pytest.approx(0.02)
    assert d.avg_down_return == pytest.approx(-0.04)
    assert d.zero_return_count == 2
    assert d.n_up == 3 and d.n_down == 2
    # NOT equal to nanmean of all finite
    assert d.avg_up_return != pytest.approx(np.nanmean(r))


def test_all_flat_distribution():
    d = compute_return_distribution(np.zeros(10))
    assert np.isnan(d.avg_up_return) and np.isnan(d.avg_down_return)
    assert d.zero_return_count == 10


def test_distribution_daily():
    rows = []
    for clock, c in [("09:31:00", 100.0), ("09:32:00", 101.0), ("09:33:00", 100.0)]:
        rows.append(
            {
                "date": "2024-06-03",
                "symbol": "S",
                "bartime": f"2024-06-03 {clock}",
                "close": c,
            }
        )
    out = compute_return_distribution_daily(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.loc[0, "n_up"] >= 1
    assert out.loc[0, "n_down"] >= 1


def _residual_panel(n_names: int = 50, n_days: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    for date in dates:
        eu = rng.normal(size=n_names)
        # ed linearly related to eu + idiosyncratic alpha
        ed = 0.3 + 0.7 * eu + rng.normal(scale=0.2, size=n_names)
        for i in range(n_names):
            rows.append(
                {
                    "date": date,
                    "symbol": f"{i:04d}.SZ",
                    "epsilon_u": eu[i],
                    "epsilon_d": ed[i],
                }
            )
    return pd.DataFrame(rows)


def test_daily_tgd_innovation_orthogonalizes():
    panel = _residual_panel(n_days=3)
    out = daily_tgd_innovation(panel, min_obs=30)
    assert "tgd_eps" in out.columns
    for _, g in out.groupby("date"):
        # residual of ed ~ eu should be ~uncorrelated with eu
        m = g.dropna(subset=["tgd_eps", "epsilon_u"])
        corr = np.corrcoef(m["tgd_eps"], m["epsilon_u"])[0, 1]
        assert abs(corr) < 0.15


def test_smooth_tgd_ma20():
    panel = _residual_panel(n_names=40, n_days=30)
    innov = daily_tgd_innovation(panel, min_obs=30)
    out = smooth_tgd(innov, window=20, min_periods=20)
    assert "TGD20" in out.columns
    # first 19 days per symbol should be NaN (min_periods=20)
    one = out[out["symbol"] == "0000.SZ"].sort_values("date")
    assert one["TGD20"].iloc[:19].isna().all()
    assert one["TGD20"].iloc[19:].notna().all()


def test_build_tgd20_and_wide():
    panel = _residual_panel(n_names=40, n_days=25)
    out = build_tgd20(panel, window=20, min_obs=30)
    wide = tgd20_to_wide(out)
    assert wide.shape[1] == 40
    assert wide.notna().any().any()
    assert out.attrs.get("tgd_stage") == "tgd20"
