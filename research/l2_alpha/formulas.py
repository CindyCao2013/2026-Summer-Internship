"""Pure-Python SSL2 Array-vector formulas (reference / unit tests).

Index convention here is Python 0-based:
  bid_prices[0] == 买一 == ClickHouse BidPrices[1]
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from research.l2_alpha.schema import DEFAULT_WOI_LAMBDA, N_DEPTH_LEVELS


def _as_float_array(values: Optional[Sequence], n: int = N_DEPTH_LEVELS) -> np.ndarray:
    if values is None:
        return np.full(n, np.nan, dtype=float)
    arr = np.asarray(list(values)[:n], dtype=float)
    if len(arr) < n:
        pad = np.full(n - len(arr), np.nan, dtype=float)
        arr = np.concatenate([arr, pad])
    return arr


def _safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def top_book_imbalance(
    bid_volumes: Sequence,
    ask_volumes: Sequence,
) -> float:
    bid = _as_float_array(bid_volumes)
    ask = _as_float_array(ask_volumes)
    b0, a0 = bid[0], ask[0]
    if not np.isfinite(b0) or not np.isfinite(a0):
        return float("nan")
    return _safe_div(b0 - a0, b0 + a0)


def depth_imbalance(
    bid_volumes: Sequence,
    ask_volumes: Sequence,
    n_levels: int = N_DEPTH_LEVELS,
) -> float:
    bid = _as_float_array(bid_volumes, n_levels)
    ask = _as_float_array(ask_volumes, n_levels)
    sb = np.nansum(bid)
    sa = np.nansum(ask)
    return _safe_div(sb - sa, sb + sa)


def weighted_order_imbalance(
    bid_volumes: Sequence,
    ask_volumes: Sequence,
    *,
    lam: float = DEFAULT_WOI_LAMBDA,
    n_levels: int = N_DEPTH_LEVELS,
) -> float:
    bid = _as_float_array(bid_volumes, n_levels)
    ask = _as_float_array(ask_volumes, n_levels)
    weights = np.exp(-lam * np.arange(n_levels, dtype=float))
    # Ignore NaN levels in weighted sum.
    b = np.where(np.isfinite(bid), bid, 0.0)
    a = np.where(np.isfinite(ask), ask, 0.0)
    mask_b = np.isfinite(bid)
    mask_a = np.isfinite(ask)
    wb = float(np.sum(weights[mask_b] * b[mask_b]))
    wa = float(np.sum(weights[mask_a] * a[mask_a]))
    return _safe_div(wb - wa, wb + wa)


def microprice_bias(
    bid_prices: Sequence,
    ask_prices: Sequence,
    bid_volumes: Sequence,
    ask_volumes: Sequence,
) -> float:
    bp = _as_float_array(bid_prices)
    ap = _as_float_array(ask_prices)
    bv = _as_float_array(bid_volumes)
    av = _as_float_array(ask_volumes)
    b0, a0, bv0, av0 = bp[0], ap[0], bv[0], av[0]
    if not all(np.isfinite(x) for x in (b0, a0, bv0, av0)):
        return float("nan")
    if bv0 + av0 <= 0 or a0 + b0 <= 0:
        return float("nan")
    mid = 0.5 * (b0 + a0)
    if mid <= 0:
        return float("nan")
    mp = (b0 * av0 + a0 * bv0) / (bv0 + av0)
    return float((mp - mid) / mid)


def relative_spread(bid_prices: Sequence, ask_prices: Sequence) -> float:
    bp = _as_float_array(bid_prices)
    ap = _as_float_array(ask_prices)
    b0, a0 = bp[0], ap[0]
    if not np.isfinite(b0) or not np.isfinite(a0):
        return float("nan")
    mid = 0.5 * (b0 + a0)
    return _safe_div(a0 - b0, mid)


def cancel_pressure(
    bid_withdraw_volume: float,
    ask_withdraw_volume: float,
) -> float:
    b = float(bid_withdraw_volume) if bid_withdraw_volume is not None else float("nan")
    a = float(ask_withdraw_volume) if ask_withdraw_volume is not None else float("nan")
    if not np.isfinite(b):
        b = 0.0
    if not np.isfinite(a):
        a = 0.0
    return _safe_div(b - a, b + a)


def liquidity_skew(
    bid_prices: Sequence,
    ask_prices: Sequence,
    bid_vwap: float,
    ask_vwap: float,
) -> float:
    bp = _as_float_array(bid_prices)
    ap = _as_float_array(ask_prices)
    b0, a0 = bp[0], ap[0]
    if not all(np.isfinite(x) for x in (b0, a0, bid_vwap, ask_vwap)):
        return float("nan")
    mid = 0.5 * (b0 + a0)
    if mid <= 0:
        return float("nan")
    return float((ask_vwap - mid) / mid - (mid - bid_vwap) / mid)


def liquidity_wall(
    bid_volumes: Sequence,
    ask_volumes: Sequence,
    n_levels: int = N_DEPTH_LEVELS,
) -> float:
    bid = _as_float_array(bid_volumes, n_levels)
    ask = _as_float_array(ask_volumes, n_levels)
    finite = np.concatenate(
        [bid[np.isfinite(bid)], ask[np.isfinite(ask)]]
    )
    if finite.size == 0:
        return float("nan")
    total = float(np.nansum(bid) + np.nansum(ask))
    return _safe_div(float(np.max(finite)), total)


def compute_all_snapshot_features(
    *,
    bid_prices: Sequence,
    ask_prices: Sequence,
    bid_volumes: Sequence,
    ask_volumes: Sequence,
    bid_withdraw_volume: float = np.nan,
    ask_withdraw_volume: float = np.nan,
    bid_vwap: float = np.nan,
    ask_vwap: float = np.nan,
    lam: float = DEFAULT_WOI_LAMBDA,
) -> dict:
    return {
        "l2_top_book_imbalance": top_book_imbalance(bid_volumes, ask_volumes),
        "l2_depth_imbalance": depth_imbalance(bid_volumes, ask_volumes),
        "l2_weighted_oi": weighted_order_imbalance(
            bid_volumes, ask_volumes, lam=lam
        ),
        "l2_microprice_bias": microprice_bias(
            bid_prices, ask_prices, bid_volumes, ask_volumes
        ),
        "l2_relative_spread": relative_spread(bid_prices, ask_prices),
        "l2_cancel_pressure": cancel_pressure(
            bid_withdraw_volume, ask_withdraw_volume
        ),
        "l2_liquidity_skew": liquidity_skew(
            bid_prices, ask_prices, bid_vwap, ask_vwap
        ),
        "l2_liquidity_wall": liquidity_wall(bid_volumes, ask_volumes),
    }
