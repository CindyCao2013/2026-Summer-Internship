"""Causal shock detectors. Information at minute t uses only observations <= t.

Existing directional_refill / liquidity_impact high-impact minutes use a
same-day 90th percentile of |r| and are NOT reused as event labels (lookahead).
LR v1 uses trailing same-session statistics and fixed depletion/widening rules.
"""

from __future__ import annotations

import numpy as np

from l2_factor_reproduction.liquidity_resilience.contracts import (
    DENOM_FLOOR_DEPTH,
    DENOM_FLOOR_SPREAD,
    DEPTH_DEPLETION_FRAC,
    FLOW_SHOCK_MULT,
    SPREAD_WIDEN_FRAC,
    TRAILING_MIN_OBS,
    TRAILING_WINDOW,
)


def trailing_median_2d(
    arr: np.ndarray,
    *,
    window: int = TRAILING_WINDOW,
    min_obs: int = TRAILING_MIN_OBS,
) -> np.ndarray:
    """Per-row trailing median of the previous `window` columns, excluding current.

    `arr` is (n_symbols, n_session_minutes). Output at t uses columns [t-window, t).
    """
    x = np.asarray(arr, dtype=float)
    if x.ndim != 2:
        raise ValueError("trailing_median_2d expects a 2-D array")
    n_sym, n = x.shape
    out = np.full((n_sym, n), np.nan, dtype=float)
    for t in range(n):
        lo = max(0, t - int(window))
        if t - lo < int(min_obs):
            continue
        window_x = x[:, lo:t]
        counts = np.isfinite(window_x).sum(axis=1)
        med = np.full(n_sym, np.nan, dtype=float)
        ok = counts >= int(min_obs)
        if np.any(ok):
            with np.errstate(all="ignore"):
                med[ok] = np.nanmedian(window_x[ok], axis=1)
        out[:, t] = med
    return out


def active_buy_shock(
    buy: np.ndarray,
    sell: np.ndarray,
    *,
    valid: np.ndarray,
    trail_buy: np.ndarray,
    mult: float = FLOW_SHOCK_MULT,
) -> np.ndarray:
    """Aggressive buy flow vs trailing same-session median of buy amount."""
    b = np.asarray(buy, dtype=float)
    s = np.asarray(sell, dtype=float)
    tr = np.asarray(trail_buy, dtype=float)
    v = np.asarray(valid, dtype=bool)
    return (
        v
        & np.isfinite(b)
        & np.isfinite(s)
        & np.isfinite(tr)
        & (tr > 0)
        & (b > s)
        & (b > 0)
        & (b >= float(mult) * tr)
    )


def active_sell_shock(
    buy: np.ndarray,
    sell: np.ndarray,
    *,
    valid: np.ndarray,
    trail_sell: np.ndarray,
    mult: float = FLOW_SHOCK_MULT,
) -> np.ndarray:
    b = np.asarray(buy, dtype=float)
    s = np.asarray(sell, dtype=float)
    tr = np.asarray(trail_sell, dtype=float)
    v = np.asarray(valid, dtype=bool)
    return (
        v
        & np.isfinite(b)
        & np.isfinite(s)
        & np.isfinite(tr)
        & (tr > 0)
        & (s > b)
        & (s > 0)
        & (s >= float(mult) * tr)
    )


def depth_depletion_shock(
    depth_pre: np.ndarray,
    depth_t0: np.ndarray,
    *,
    valid: np.ndarray,
    frac: float = DEPTH_DEPLETION_FRAC,
    floor: float = DENOM_FLOOR_DEPTH,
) -> np.ndarray:
    pre = np.asarray(depth_pre, dtype=float)
    t0 = np.asarray(depth_t0, dtype=float)
    v = np.asarray(valid, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = (pre - t0) / np.where(pre > 0, pre, np.nan)
    return (
        v
        & np.isfinite(pre)
        & np.isfinite(t0)
        & (pre >= float(floor))
        & (t0 < pre)
        & (drop >= float(frac))
    )


def spread_widening_shock(
    spread_pre: np.ndarray,
    spread_t0: np.ndarray,
    *,
    valid: np.ndarray,
    frac: float = SPREAD_WIDEN_FRAC,
    floor: float = DENOM_FLOOR_SPREAD,
) -> np.ndarray:
    pre = np.asarray(spread_pre, dtype=float)
    t0 = np.asarray(spread_t0, dtype=float)
    v = np.asarray(valid, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        widen = (t0 - pre) / np.where(pre > 0, pre, np.nan)
    return (
        v
        & np.isfinite(pre)
        & np.isfinite(t0)
        & (pre >= float(floor))
        & (t0 > pre)
        & (widen >= float(frac))
    )
