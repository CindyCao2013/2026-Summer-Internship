"""Shared recovery math. Tiny denominators are excluded, not clipped to IC-tuned percentiles."""

from __future__ import annotations

import numpy as np

from l2_factor_reproduction.liquidity_resilience.contracts import (
    DENOM_FLOOR_DEPTH,
    DENOM_FLOOR_FLOW,
    DENOM_FLOOR_OBI,
    DENOM_FLOOR_SPREAD,
)


def _as_float(*arrays: object) -> tuple:
    return tuple(np.asarray(a, dtype=float) for a in arrays)


def recovery_fraction(
    x_pre: object,
    x_t0: object,
    x_h: object,
    *,
    denom_floor: float = DENOM_FLOOR_DEPTH,
) -> np.ndarray:
    """(X_h - X_t0) / (X_pre - X_t0) for a variable depleted by the shock.

    ≈1 returned near pre, ≈0 little recovery, <0 continued deterioration, >1 overshoot.
    Events with (X_pre - X_t0) < denom_floor are NA (economically negligible depletion).
    """
    pre, t0, h = np.broadcast_arrays(*_as_float(x_pre, x_t0, x_h))
    denom = pre - t0
    out = np.full(pre.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(pre)
        & np.isfinite(t0)
        & np.isfinite(h)
        & np.isfinite(denom)
        & (denom >= float(denom_floor))
    )
    out[ok] = (h[ok] - t0[ok]) / denom[ok]
    return out


def spread_recovery_fraction(
    spread_pre: object,
    spread_t0: object,
    spread_h: object,
    *,
    denom_floor: float = DENOM_FLOOR_SPREAD,
) -> np.ndarray:
    """(spread_t0 - spread_h) / (spread_t0 - spread_pre) after widening.

    ≈1 returned to pre, ≈0 remained wide, <0 widened further.
    """
    pre, t0, h = np.broadcast_arrays(*_as_float(spread_pre, spread_t0, spread_h))
    denom = t0 - pre
    out = np.full(pre.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(pre)
        & np.isfinite(t0)
        & np.isfinite(h)
        & (denom >= float(denom_floor))
    )
    out[ok] = (t0[ok] - h[ok]) / denom[ok]
    return out


def spread_residual_width(
    spread_pre: object,
    spread_h: object,
    *,
    denom_floor: float = DENOM_FLOOR_SPREAD,
) -> np.ndarray:
    """(spread_h - spread_pre) / spread_pre. Remaining extra width vs pre."""
    pre, h = np.broadcast_arrays(*_as_float(spread_pre, spread_h))
    out = np.full(pre.shape, np.nan, dtype=float)
    ok = np.isfinite(pre) & np.isfinite(h) & (pre >= float(denom_floor))
    out[ok] = (h[ok] - pre[ok]) / pre[ok]
    return out


def replenishment_efficiency(
    x_t0: object,
    x_h: object,
    shock_size: object,
    *,
    denom_floor: float = DENOM_FLOOR_FLOW,
) -> np.ndarray:
    """Recovered depth / shock size. Units must match (depth CNY / flow CNY)."""
    t0, h, size = np.broadcast_arrays(*_as_float(x_t0, x_h, shock_size))
    out = np.full(t0.shape, np.nan, dtype=float)
    ok = np.isfinite(t0) & np.isfinite(h) & np.isfinite(size) & (size >= float(denom_floor))
    out[ok] = (h[ok] - t0[ok]) / size[ok]
    return out


def obi_restoration(
    obi_pre: object,
    obi_t0: object,
    obi_h: object,
    *,
    denom_floor: float = DENOM_FLOOR_OBI,
) -> np.ndarray:
    """1 - abs(obi_h - obi_pre) / abs(obi_t0 - obi_pre). Higher: closer to pre."""
    pre, t0, h = np.broadcast_arrays(*_as_float(obi_pre, obi_t0, obi_h))
    disp = np.abs(t0 - pre)
    out = np.full(pre.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(pre)
        & np.isfinite(t0)
        & np.isfinite(h)
        & (disp >= float(denom_floor))
    )
    out[ok] = 1.0 - np.abs(h[ok] - pre[ok]) / disp[ok]
    return out


def obi_persistence(
    obi_pre: object,
    obi_t0: object,
    obi_h: object,
    *,
    denom_floor: float = DENOM_FLOOR_OBI,
) -> np.ndarray:
    """abs(obi_h - obi_pre) / abs(obi_t0 - obi_pre). Higher: imbalance remains."""
    pre, t0, h = np.broadcast_arrays(*_as_float(obi_pre, obi_t0, obi_h))
    disp = np.abs(t0 - pre)
    out = np.full(pre.shape, np.nan, dtype=float)
    ok = (
        np.isfinite(pre)
        & np.isfinite(t0)
        & np.isfinite(h)
        & (disp >= float(denom_floor))
    )
    out[ok] = np.abs(h[ok] - pre[ok]) / disp[ok]
    return out


def event_median(values: object) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def shock_size_weighted_mean(values: object, weights: object) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))
