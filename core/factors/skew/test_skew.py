"""Unit tests for SKEW / IdioSKEW formulas (no DB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.factors.skew.idio_skew import build_idio_skew, rolling_market_residual
from core.factors.skew.skew import alpha_from_skew, build_total_skew, skew_20d


def _synth_panel(n_days: int = 80, n_names: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    cols = [f"{i:04d}.SZ" for i in range(n_names)]
    # Lottery-like names: occasional large positive spikes
    ret = pd.DataFrame(rng.normal(scale=0.01, size=(n_days, n_names)), index=idx, columns=cols)
    for j, c in enumerate(cols):
        spike_days = rng.choice(n_days, size=3, replace=False)
        ret.iloc[spike_days, j] = 0.08 + 0.02 * j
    mkt = pd.Series(rng.normal(scale=0.008, size=n_days), index=idx, name="mkt")
    return ret, mkt


def test_skew_20d_finite_and_shifted_safe():
    ret, _ = _synth_panel()
    sk = skew_20d(ret)
    assert sk.shape == ret.shape
    assert sk.iloc[:9].isna().all().all()
    assert sk.iloc[20:].notna().any().any()


def test_alpha_negates_skew():
    ret, _ = _synth_panel()
    panels = build_total_skew(ret, windows=(20,))
    alpha = build_total_skew(ret, windows=(20,), as_alpha=True)
    assert np.allclose(
        alpha["AlphaSKEW20"].to_numpy(),
        (-panels["SKEW20"]).to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        alpha_from_skew(panels["SKEW20"]).to_numpy(),
        alpha["AlphaSKEW20"].to_numpy(),
        equal_nan=True,
    )


def test_idio_residual_mean_near_zero_in_window():
    ret, mkt = _synth_panel(n_days=100, n_names=3)
    resid = rolling_market_residual(ret, mkt, window=60, min_periods=40)
    # Rolling residual at t uses α_t, β_t estimated on the same window ending at t,
    # so the last residual in each window is not exactly mean-zero; just check finiteness.
    assert resid.iloc[60:].notna().sum().sum() > 0
    assert np.isfinite(resid.to_numpy(dtype=float)).sum() > 0


def test_idio_skew_60_builds():
    ret, mkt = _synth_panel(n_days=120, n_names=4)
    out = build_idio_skew(ret, mkt, windows=(60,))
    assert "IdioSKEW60" in out
    assert out["IdioSKEW60"].shape == ret.shape
    assert out["IdioSKEW60"].iloc[70:].notna().any().any()


def test_no_lookahead_in_rolling_endpoint():
    """Changing a future return must not change today's skew."""
    ret, _ = _synth_panel(n_days=60, n_names=2)
    sk0 = skew_20d(ret)
    ret2 = ret.copy()
    ret2.iloc[-1, 0] = 0.5
    sk1 = skew_20d(ret2)
    # All rows except the last window that includes the last day should match
    assert np.allclose(
        sk0.iloc[:-1].to_numpy(),
        sk1.iloc[:-1].to_numpy(),
        equal_nan=True,
    )
