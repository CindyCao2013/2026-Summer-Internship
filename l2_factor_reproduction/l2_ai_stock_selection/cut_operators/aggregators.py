"""Interpretable aggregators over a masked primitive sequence."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    AGGREGATORS,
    MIN_COVERAGE_OBS,
    MIN_PERSISTENCE_PAIRS,
    MIN_SLOPE_OBS,
    PRIMITIVE_CLASS_AGGREGATORS,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.time_cuts import (
    consecutive_mkeys,
)


def assert_aggregator_allowed(op: str, primitive_class: str) -> None:
    allowed = PRIMITIVE_CLASS_AGGREGATORS.get(primitive_class)
    if allowed is None:
        raise KeyError("unknown primitive_class {!r}".format(primitive_class))
    if op not in allowed:
        raise ValueError(
            "aggregator {!r} is not economic for class {!r}; allowed={}".format(
                op, primitive_class, allowed
            )
        )


def _finite_masked(x: Sequence, mask: Sequence) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if arr.shape != m.shape:
        raise ValueError("x and mask length mismatch")
    return arr[m]


def agg_mean(x: Sequence, mask: Sequence, *, min_obs: int = MIN_COVERAGE_OBS) -> float:
    sl = _finite_masked(x, mask)
    sl = sl[np.isfinite(sl)]
    if sl.size < int(min_obs):
        return float("nan")
    return float(np.mean(sl))


def agg_sum(x: Sequence, mask: Sequence, *, min_obs: int = 1) -> float:
    sl = _finite_masked(x, mask)
    sl = sl[np.isfinite(sl)]
    if sl.size < int(min_obs):
        return float("nan")
    return float(np.sum(sl))


def agg_std(x: Sequence, mask: Sequence, *, min_obs: int = MIN_COVERAGE_OBS) -> float:
    sl = _finite_masked(x, mask)
    sl = sl[np.isfinite(sl)]
    if sl.size < int(min_obs):
        return float("nan")
    return float(np.std(sl, ddof=0))


def agg_median(x: Sequence, mask: Sequence, *, min_obs: int = MIN_COVERAGE_OBS) -> float:
    sl = _finite_masked(x, mask)
    sl = sl[np.isfinite(sl)]
    if sl.size < int(min_obs):
        return float("nan")
    return float(np.median(sl))


def agg_last(x: Sequence, mask: Sequence, *, min_obs: int = 1) -> float:
    sl = _finite_masked(x, mask)
    finite = np.isfinite(sl)
    if finite.sum() < int(min_obs):
        return float("nan")
    return float(sl[finite][-1])


def agg_max(x: Sequence, mask: Sequence, *, min_obs: int = 1) -> float:
    sl = _finite_masked(x, mask)
    sl = sl[np.isfinite(sl)]
    if sl.size < int(min_obs):
        return float("nan")
    return float(np.max(sl))


def agg_min(x: Sequence, mask: Sequence, *, min_obs: int = 1) -> float:
    sl = _finite_masked(x, mask)
    sl = sl[np.isfinite(sl)]
    if sl.size < int(min_obs):
        return float("nan")
    return float(np.min(sl))


def agg_persistence(
    x: Sequence,
    mask: Sequence,
    mkeys: Sequence,
    *,
    min_pairs: int = MIN_PERSISTENCE_PAIRS,
) -> float:
    """Share of consecutive same-sign observations inside the mask.

    Lunch and auction gaps are not consecutive. Zero is treated as a break.
    """
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    keys = np.asarray(mkeys, dtype=np.int32)
    if arr.shape != m.shape or keys.shape != m.shape:
        raise ValueError("x/mask/mkeys length mismatch")
    same = 0
    pairs = 0
    prev_i = None
    for i in range(arr.size):
        if not m[i] or not np.isfinite(arr[i]):
            continue
        if prev_i is not None and consecutive_mkeys(int(keys[prev_i]), int(keys[i])):
            a = arr[prev_i]
            b = arr[i]
            if a != 0 and b != 0:
                pairs += 1
                if np.sign(a) == np.sign(b):
                    same += 1
        prev_i = i
    if pairs < int(min_pairs):
        return float("nan")
    return float(same) / float(pairs)


def agg_slope(
    x: Sequence,
    mask: Sequence,
    mkeys: Sequence,
    *,
    min_obs: int = MIN_SLOPE_OBS,
) -> float:
    """OLS slope of X on mkey among masked finite points."""
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    keys = np.asarray(mkeys, dtype=float)
    ok = m & np.isfinite(arr) & np.isfinite(keys)
    if int(ok.sum()) < int(min_obs):
        return float("nan")
    t = keys[ok]
    y = arr[ok]
    t = t - t.mean()
    den = float(np.dot(t, t))
    if den <= 0:
        return float("nan")
    return float(np.dot(t, y - y.mean()) / den)


def agg_event_share(mask: Sequence, valid: Sequence) -> float:
    m = np.asarray(mask, dtype=bool)
    v = np.asarray(valid, dtype=bool)
    n_valid = int(v.sum())
    if n_valid <= 0:
        return float("nan")
    return float((m & v).sum()) / float(n_valid)


def _signed_weights(x: np.ndarray, *, signed: bool, weight: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w_plus, w_minus, w_abs) for temporal location.

    Nonnegative / unsigned variables must not be split into +/-; pass signed=False
    and an economic weight ('abs' or 'self').
    """
    arr = np.asarray(x, dtype=float)
    if not signed or str(weight).lower() in ("abs", "self", "unsigned"):
        w = np.abs(arr) if str(weight).lower() != "self" else np.where(arr > 0, arr, np.nan)
        w = np.where(np.isfinite(arr), w, np.nan)
        z = np.zeros_like(arr)
        return z, z, w
    w_plus = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    w_minus = np.where(np.isfinite(arr) & (arr < 0), -arr, 0.0)
    w_abs = w_plus + w_minus
    w_plus = np.where(np.isfinite(arr), w_plus, np.nan)
    w_minus = np.where(np.isfinite(arr), w_minus, np.nan)
    w_abs = np.where(np.isfinite(arr), w_abs, np.nan)
    return w_plus, w_minus, w_abs


def _weighted_center(t: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(t) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < 2:
        return float("nan")
    den = float(w[ok].sum())
    if den <= 0:
        return float("nan")
    return float(np.dot(t[ok], w[ok]) / den)


def _weighted_dispersion(t: np.ndarray, w: np.ndarray, center: float) -> float:
    if not np.isfinite(center):
        return float("nan")
    ok = np.isfinite(t) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < 3:
        return float("nan")
    den = float(w[ok].sum())
    if den <= 0:
        return float("nan")
    return float(np.sqrt(np.dot(w[ok], (t[ok] - center) ** 2) / den))


def agg_temporal_center(
    x: Sequence,
    mask: Sequence,
    times: Sequence,
    *,
    signed: bool = True,
    weight: str = "signed",
) -> float:
    """Mass-weighted location on the continuous minute_index 0-239."""
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    t = np.asarray(times, dtype=float)
    sl_x, sl_t = arr[m], t[m]
    _, _, w_abs = _signed_weights(sl_x, signed=signed, weight=weight)
    return _weighted_center(sl_t, w_abs)


def agg_tc_plus(x: Sequence, mask: Sequence, times: Sequence) -> float:
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    t = np.asarray(times, dtype=float)
    w_plus, _, _ = _signed_weights(arr[m], signed=True, weight="signed")
    return _weighted_center(t[m], w_plus)


def agg_tc_minus(x: Sequence, mask: Sequence, times: Sequence) -> float:
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    t = np.asarray(times, dtype=float)
    _, w_minus, _ = _signed_weights(arr[m], signed=True, weight="signed")
    return _weighted_center(t[m], w_minus)


def agg_temporal_gap(x: Sequence, mask: Sequence, times: Sequence) -> float:
    """TC_minus - TC_plus. Positive when negative mass arrives later than positive mass."""
    plus = agg_tc_plus(x, mask, times)
    minus = agg_tc_minus(x, mask, times)
    if not (np.isfinite(plus) and np.isfinite(minus)):
        return float("nan")
    return float(minus - plus)


def agg_temporal_dispersion(
    x: Sequence,
    mask: Sequence,
    times: Sequence,
    *,
    signed: bool = True,
    weight: str = "signed",
) -> float:
    arr = np.asarray(x, dtype=float)
    m = np.asarray(mask, dtype=bool)
    t = np.asarray(times, dtype=float)
    sl_x, sl_t = arr[m], t[m]
    _, _, w_abs = _signed_weights(sl_x, signed=signed, weight=weight)
    center = _weighted_center(sl_t, w_abs)
    return _weighted_dispersion(sl_t, w_abs, center)


_DISPATCH = {
    "mean": agg_mean,
    "sum": agg_sum,
    "std": agg_std,
    "median": agg_median,
    "last": agg_last,
    "max": agg_max,
    "min": agg_min,
}


def apply_aggregator(
    op: str,
    x: Sequence,
    mask: Sequence,
    *,
    mkeys: Sequence = (),
    valid: Sequence = (),
    primitive_class: str = "",
    min_obs: int = MIN_COVERAGE_OBS,
) -> float:
    if op not in AGGREGATORS:
        raise KeyError("unknown aggregator {!r}".format(op))
    if primitive_class:
        assert_aggregator_allowed(op, primitive_class)
    if op in _DISPATCH:
        kwargs = {}
        if op in ("mean", "std", "median"):
            kwargs["min_obs"] = min_obs
        return float(_DISPATCH[op](x, mask, **kwargs))
    if op == "persistence":
        return agg_persistence(x, mask, mkeys)
    if op == "slope":
        return agg_slope(x, mask, mkeys)
    if op == "event_share":
        v = valid if len(valid) else np.ones(len(mask), dtype=bool)
        return agg_event_share(mask, v)
    times = mkeys
    if op == "temporal_center":
        return agg_temporal_center(x, mask, times)
    if op == "tc_plus":
        return agg_tc_plus(x, mask, times)
    if op == "tc_minus":
        return agg_tc_minus(x, mask, times)
    if op == "temporal_gap":
        return agg_temporal_gap(x, mask, times)
    if op == "temporal_dispersion":
        return agg_temporal_dispersion(x, mask, times)
    raise KeyError("unhandled aggregator {!r}".format(op))
