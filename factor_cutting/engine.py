"""Object / Knife / Output abstraction + rolling cutting operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # research envs may lack numba — fall back to pure Python
    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap

Aggregate = Literal["sum", "mean", "compound"]
SplitMethod = Literal["rank_split", "quantile_split", "bucket"]


@dataclass(frozen=True)
class ObjectSpec:
    variable: str
    additive: bool = True
    formula: str = ""


@dataclass(frozen=True)
class KnifeSpec:
    variable: str
    method: SplitMethod = "rank_split"
    window: int = 20
    high_count: Optional[int] = None
    low_count: Optional[int] = None
    lambda_frac: float = 0.25
    formula: str = ""


@dataclass(frozen=True)
class OutputSpec:
    aggregate: Aggregate = "sum"
    op: Literal["difference", "ratio", "select_high", "select_low"] = "difference"
    formula: str = ""


@dataclass(frozen=True)
class CuttingSpec:
    name: str
    object: ObjectSpec
    knife: KnifeSpec
    output: OutputSpec
    paper: str = ""
    direction_paper: str = ""
    status: str = "implemented_daily"
    meta: dict = field(default_factory=dict)


def cut_difference(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    return high - low


def cut_ratio(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    return high / low.replace(0, np.nan)


@njit(cache=True)
def _rank_split_panel(
    obj: np.ndarray,
    knife: np.ndarray,
    window: int,
    high_k: int,
    low_k: int,
    agg_code: int,
    require_full: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """agg_code: 0=sum, 1=mean, 2=compound. Arrays are (T, N)."""
    t_len, n_cols = obj.shape
    out = np.full((t_len, n_cols), np.nan)
    high_leg = np.full((t_len, n_cols), np.nan)
    low_leg = np.full((t_len, n_cols), np.nan)

    for j in range(n_cols):
        for i in range(window - 1, t_len):
            n_valid = 0
            for t in range(i - window + 1, i + 1):
                o = obj[t, j]
                k = knife[t, j]
                if np.isfinite(o) and np.isfinite(k):
                    n_valid += 1
            if require_full and n_valid < window:
                continue
            if n_valid < high_k + low_k:
                continue

            o_buf = np.empty(n_valid)
            k_buf = np.empty(n_valid)
            idx = 0
            for t in range(i - window + 1, i + 1):
                o = obj[t, j]
                k = knife[t, j]
                if np.isfinite(o) and np.isfinite(k):
                    o_buf[idx] = o
                    k_buf[idx] = k
                    idx += 1

            order = np.argsort(k_buf)
            # low
            if agg_code == 0:
                lo = 0.0
                for p in range(low_k):
                    lo += o_buf[order[p]]
                hi = 0.0
                for p in range(high_k):
                    hi += o_buf[order[n_valid - high_k + p]]
            elif agg_code == 1:
                lo = 0.0
                for p in range(low_k):
                    lo += o_buf[order[p]]
                lo /= low_k
                hi = 0.0
                for p in range(high_k):
                    hi += o_buf[order[n_valid - high_k + p]]
                hi /= high_k
            else:
                lo = 1.0
                for p in range(low_k):
                    lo *= 1.0 + o_buf[order[p]]
                lo -= 1.0
                hi = 1.0
                for p in range(high_k):
                    hi *= 1.0 + o_buf[order[n_valid - high_k + p]]
                hi -= 1.0

            high_leg[i, j] = hi
            low_leg[i, j] = lo
            out[i, j] = hi - lo
    return out, high_leg, low_leg


@njit(cache=True)
def _quantile_mean_diff_panel(
    obj: np.ndarray,
    knife: np.ndarray,
    window: int,
    lambda_frac: float,
    min_effective: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_len, n_cols = obj.shape
    out = np.full((t_len, n_cols), np.nan)
    high_leg = np.full((t_len, n_cols), np.nan)
    low_leg = np.full((t_len, n_cols), np.nan)

    for j in range(n_cols):
        for i in range(window - 1, t_len):
            n_valid = 0
            for t in range(i - window + 1, i + 1):
                o = obj[t, j]
                k = knife[t, j]
                if np.isfinite(o) and np.isfinite(k):
                    n_valid += 1
            if n_valid < min_effective:
                continue
            k_take = max(1, int(n_valid * lambda_frac))
            if 2 * k_take > n_valid:
                k_take = max(1, n_valid // 2)

            o_buf = np.empty(n_valid)
            k_buf = np.empty(n_valid)
            idx = 0
            for t in range(i - window + 1, i + 1):
                o = obj[t, j]
                k = knife[t, j]
                if np.isfinite(o) and np.isfinite(k):
                    o_buf[idx] = o
                    k_buf[idx] = k
                    idx += 1

            order = np.argsort(k_buf)
            lo = 0.0
            for p in range(k_take):
                lo += o_buf[order[p]]
            lo /= k_take
            hi = 0.0
            for p in range(k_take):
                hi += o_buf[order[n_valid - k_take + p]]
            hi /= k_take
            high_leg[i, j] = hi
            low_leg[i, j] = lo
            out[i, j] = hi - lo
    return out, high_leg, low_leg


def _agg_code(aggregate: Aggregate) -> int:
    return {"sum": 0, "mean": 1, "compound": 2}[aggregate]


def rolling_rank_split_sum(
    object_panel: pd.DataFrame,
    knife_panel: pd.DataFrame,
    *,
    window: int = 20,
    high_count: int = 10,
    low_count: int = 10,
    aggregate: Aggregate = "sum",
    require_full_window: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """W-cut: rank lookback days by knife, aggregate object on high/low groups, difference."""
    obj = object_panel.astype(float)
    knife = knife_panel.reindex_like(obj).astype(float)
    f, h, lo = _rank_split_panel(
        obj.to_numpy(),
        knife.to_numpy(),
        window,
        high_count,
        low_count,
        _agg_code(aggregate),
        require_full_window,
    )
    idx, cols = obj.index, obj.columns
    return (
        pd.DataFrame(f, index=idx, columns=cols),
        pd.DataFrame(h, index=idx, columns=cols),
        pd.DataFrame(lo, index=idx, columns=cols),
    )


def rolling_quantile_mean_diff(
    object_panel: pd.DataFrame,
    knife_panel: pd.DataFrame,
    *,
    window: int = 20,
    lambda_frac: float = 0.25,
    min_effective_days: int = 10,
    aggregate: Aggregate = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantile cut: mean(object | knife top λ) − mean(object | knife bottom λ)."""
    del aggregate  # mean-only for amplitude paper
    obj = object_panel.astype(float)
    knife = knife_panel.reindex_like(obj).astype(float)
    f, h, lo = _quantile_mean_diff_panel(
        obj.to_numpy(),
        knife.to_numpy(),
        window,
        float(lambda_frac),
        min_effective_days,
    )
    idx, cols = obj.index, obj.columns
    return (
        pd.DataFrame(f, index=idx, columns=cols),
        pd.DataFrame(h, index=idx, columns=cols),
        pd.DataFrame(lo, index=idx, columns=cols),
    )


def knife_quantile_mechanism(
    future_ret: pd.DataFrame,
    knife_panel: pd.DataFrame,
    *,
    n_quantiles: int = 10,
    signal_shift: int = 1,
) -> pd.DataFrame:
    """Stage-3 diagnostic: CS mean future return by knife quantile."""
    knife = knife_panel.shift(signal_shift)
    ret = future_ret.reindex_like(knife)
    rows = []
    for dt in knife.index:
        k = knife.loc[dt]
        r = ret.loc[dt]
        mask = k.notna() & r.notna()
        if mask.sum() < n_quantiles * 20:
            continue
        q = pd.qcut(k[mask], n_quantiles, labels=False, duplicates="drop")
        for qi, grp in r[mask].groupby(q):
            rows.append(
                {"date": dt, "q": int(qi) + 1, "mean_ret": float(grp.mean()), "n": int(len(grp))}
            )
    if not rows:
        return pd.DataFrame(columns=["q", "mean_ret", "n_days"])
    long = pd.DataFrame(rows)
    return (
        long.groupby("q")
        .agg(mean_ret=("mean_ret", "mean"), n_days=("date", "nunique"), mean_n=("n", "mean"))
        .reset_index()
    )
