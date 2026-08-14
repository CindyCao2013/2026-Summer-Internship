"""NumPy reference implementations of AlphaNet custom operators.

Input convention: ``x`` has shape ``(batch, features, time)`` with time running
from oldest to newest along the last axis.

Each sliding window of length ``d`` and stride ``stride`` emits **one scalar
per channel** (pair or feature). ``ts_zscore`` is the z-score of the *last*
observation in the window. ``ts_return`` is ``last / first - 1`` (window of
length ``d`` is equivalent to ``delay = d - 1``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EPS = 1e-8


def n_windows(time_len: int, d: int, stride: int) -> int:
    if time_len < d or d <= 0 or stride <= 0:
        return 0
    return (time_len - d) // stride + 1


def frame_windows(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    """``(B, F, T) -> (B, F, n_windows, d)``."""
    if x.ndim != 3:
        raise ValueError("expected (batch, features, time), got {}".format(x.shape))
    time_len = int(x.shape[-1])
    w = n_windows(time_len, d, stride)
    if w <= 0:
        raise ValueError("cannot frame T={} with d={} stride={}".format(time_len, d, stride))
    out = np.empty(x.shape[:2] + (w, d), dtype=x.dtype)
    for i in range(w):
        start = i * stride
        out[..., i, :] = x[..., start : start + d]
    return out


def pair_indices(n_features: int, pair_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if pair_mode == "full":
        ii, jj = np.meshgrid(np.arange(n_features), np.arange(n_features), indexing="ij")
        return ii.reshape(-1), jj.reshape(-1)
    if pair_mode == "unique":
        ii, jj = np.triu_indices(n_features, k=1)
        return ii.astype(np.int64), jj.astype(np.int64)
    raise ValueError("unknown pair_mode {!r}".format(pair_mode))


def _safe_std(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    std = np.std(arr, axis=axis)
    return np.where(std < EPS, np.nan, std)


def ts_corr(x: np.ndarray, d: int, stride: int, pair_mode: str = "full") -> np.ndarray:
    """Pearson corr of every feature pair in each window. Shape ``(B, P, W)``."""
    windows = frame_windows(x, d, stride)
    b, f, w, _ = windows.shape
    ii, jj = pair_indices(f, pair_mode)
    left = windows[:, ii]
    right = windows[:, jj]
    left_c = left - left.mean(axis=-1, keepdims=True)
    right_c = right - right.mean(axis=-1, keepdims=True)
    cov = (left_c * right_c).mean(axis=-1)
    std_l = _safe_std(left, axis=-1)
    std_r = _safe_std(right, axis=-1)
    corr = cov / (std_l * std_r)
    return np.nan_to_num(corr, nan=0.0).astype(np.float32, copy=False)


def ts_cov(x: np.ndarray, d: int, stride: int, pair_mode: str = "full") -> np.ndarray:
    windows = frame_windows(x, d, stride)
    f = windows.shape[1]
    ii, jj = pair_indices(f, pair_mode)
    left = windows[:, ii]
    right = windows[:, jj]
    left_c = left - left.mean(axis=-1, keepdims=True)
    right_c = right - right.mean(axis=-1, keepdims=True)
    cov = (left_c * right_c).mean(axis=-1)
    return cov.astype(np.float32, copy=False)


def ts_mean(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    return frame_windows(x, d, stride).mean(axis=-1).astype(np.float32, copy=False)


def ts_std(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    return frame_windows(x, d, stride).std(axis=-1).astype(np.float32, copy=False)


def ts_sum(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    return frame_windows(x, d, stride).sum(axis=-1).astype(np.float32, copy=False)


def ts_max(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    return frame_windows(x, d, stride).max(axis=-1).astype(np.float32, copy=False)


def ts_min(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    return frame_windows(x, d, stride).min(axis=-1).astype(np.float32, copy=False)


def ts_zscore(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    """Z-score of the last observation in each window. Shape ``(B, F, W)``."""
    windows = frame_windows(x, d, stride)
    mean = windows.mean(axis=-1)
    std = _safe_std(windows, axis=-1)
    last = windows[..., -1]
    z = (last - mean) / std
    return np.nan_to_num(z, nan=0.0).astype(np.float32, copy=False)


def ts_return(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    """``X_last / X_first - 1`` per window. Shape ``(B, F, W)``."""
    windows = frame_windows(x, d, stride)
    first = windows[..., 0]
    last = windows[..., -1]
    out = np.where(np.abs(first) < EPS, np.nan, last / first - 1.0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32, copy=False)


def decay_weights(d: int, dtype=np.float32) -> np.ndarray:
    """Newest observation gets weight ``d``, oldest gets ``1``."""
    return np.arange(1, d + 1, dtype=dtype)


def ts_decaylinear(x: np.ndarray, d: int, stride: int) -> np.ndarray:
    windows = frame_windows(x, d, stride)
    weights = decay_weights(d, dtype=windows.dtype)
    denom = float(weights.sum())
    return (windows * weights).sum(axis=-1).astype(np.float32, copy=False) / denom


UNARY_DISPATCH = {
    "ts_stddev": ts_std,
    "ts_zscore": ts_zscore,
    "ts_return": ts_return,
    "ts_decaylinear": ts_decaylinear,
    "ts_mean": ts_mean,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "ts_sum": ts_sum,
}
BINARY_DISPATCH = {
    "ts_corr": ts_corr,
    "ts_cov": ts_cov,
}
POOL_DISPATCH = {
    "ts_mean": ts_mean,
    "ts_max": ts_max,
    "ts_min": ts_min,
}


def apply_extract_op(
    x: np.ndarray,
    op: str,
    d: int,
    stride: int,
    pair_mode: str = "full",
) -> np.ndarray:
    if op in BINARY_DISPATCH:
        return BINARY_DISPATCH[op](x, d, stride, pair_mode=pair_mode)
    if op in UNARY_DISPATCH:
        return UNARY_DISPATCH[op](x, d, stride)
    raise KeyError("unknown extract op {!r}".format(op))


def apply_pool_op(feat: np.ndarray, op: str, d: int, stride: int) -> np.ndarray:
    """Pool over the time-window axis of a feature map ``(B, C, W)``."""
    if op not in POOL_DISPATCH:
        raise KeyError("unknown pool op {!r}".format(op))
    return POOL_DISPATCH[op](feat, d, stride)


def extract_all(
    x: np.ndarray,
    d: int,
    stride: int,
    pair_mode: str = "full",
    ops: Optional[Tuple[str, ...]] = None,
) -> dict:
    from alphanet.config import EXTRACT_OPS

    names = ops or EXTRACT_OPS
    return {op: apply_extract_op(x, op, d, stride, pair_mode=pair_mode) for op in names}
