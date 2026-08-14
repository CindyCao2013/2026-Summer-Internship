"""Execution layer — signal → portfolio → trade (factor definition frozen).

Optimizes net alpha via rebalance frequency, buffer ranking, min-hold, weights.
Does NOT modify factor formulas.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    net_pnl_series,
    series_performance,
)
from factor_attribution import align_signal, cs_zscore


def downsample_signal(signal: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Hold cross-section between rebalance days (freq=1 → daily)."""
    if freq <= 1:
        return signal
    out = signal.copy()
    values = out.to_numpy(copy=True)
    n = len(values)
    last = values[0].copy()
    for t in range(1, n):
        if t % freq == 0:
            last = values[t].copy()
        else:
            values[t] = last
    return pd.DataFrame(values, index=out.index, columns=out.columns)


def friday_rebalance_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Rebalance only on Fridays (weekday==4); hold otherwise."""
    out = signal.copy()
    values = out.to_numpy(copy=True)
    weekdays = out.index.weekday.to_numpy()
    last = values[0].copy()
    for t in range(len(values)):
        if weekdays[t] == 4 or t == 0:
            last = values[t].copy()
        else:
            values[t] = last
    return pd.DataFrame(values, index=out.index, columns=out.columns)


def _pct_ranks(row: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile ranks in [0,1]; NaN preserved."""
    s = pd.Series(row)
    return s.rank(pct=True, method="average").to_numpy(dtype=float)


def buffer_ls_masks(
    signal: pd.DataFrame,
    *,
    entry_frac: float = 0.10,
    exit_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hysteresis long/short membership.

    Long:  enter when pct_rank >= 1-entry_frac; exit when pct_rank < 1-exit_frac
    Short: enter when pct_rank <= entry_frac;   exit when pct_rank > exit_frac
    """
    if not (0 < entry_frac < exit_frac <= 0.5):
        raise ValueError("need 0 < entry_frac < exit_frac <= 0.5")

    values = signal.to_numpy(dtype=float)
    n_t, n_n = values.shape
    long_m = np.zeros((n_t, n_n), dtype=bool)
    short_m = np.zeros((n_t, n_n), dtype=bool)
    in_long = np.zeros(n_n, dtype=bool)
    in_short = np.zeros(n_n, dtype=bool)

    long_entry = 1.0 - entry_frac
    long_exit = 1.0 - exit_frac
    short_entry = entry_frac
    short_exit = exit_frac

    for t in range(n_t):
        ranks = _pct_ranks(values[t])
        finite = np.isfinite(ranks)

        # cannot be both; clear invalid first
        in_long &= finite
        in_short &= finite

        enter_l = finite & (ranks >= long_entry)
        exit_l = (~finite) | (ranks < long_exit)
        enter_s = finite & (ranks <= short_entry)
        exit_s = (~finite) | (ranks > short_exit)

        in_long = (in_long & ~exit_l) | enter_l
        in_short = (in_short & ~exit_s) | enter_s
        # mutual exclusion: prefer long if both (rare)
        both = in_long & in_short
        in_short[both] = False

        long_m[t] = in_long
        short_m[t] = in_short

    idx, cols = signal.index, signal.columns
    return (
        pd.DataFrame(long_m, index=idx, columns=cols),
        pd.DataFrame(short_m, index=idx, columns=cols),
    )


def min_hold_masks(
    long_m: pd.DataFrame,
    short_m: pd.DataFrame,
    *,
    min_hold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Force membership to persist at least `min_hold` days after entry."""
    if min_hold <= 1:
        return long_m, short_m

    L = long_m.to_numpy(dtype=bool)
    S = short_m.to_numpy(dtype=bool)
    n_t, n_n = L.shape
    hold_l = np.zeros(n_n, dtype=int)
    hold_s = np.zeros(n_n, dtype=int)
    out_l = np.zeros_like(L)
    out_s = np.zeros_like(S)
    cur_l = np.zeros(n_n, dtype=bool)
    cur_s = np.zeros(n_n, dtype=bool)

    for t in range(n_t):
        target_l = L[t]
        target_s = S[t]

        new_l = cur_l | target_l
        new_s = cur_s | target_s

        exit_cand_l = cur_l & ~target_l
        keep_l = exit_cand_l & (hold_l < min_hold)
        allow_exit_l = exit_cand_l & (hold_l >= min_hold)
        new_l[allow_exit_l] = False
        new_l[keep_l] = True

        exit_cand_s = cur_s & ~target_s
        keep_s = exit_cand_s & (hold_s < min_hold)
        allow_exit_s = exit_cand_s & (hold_s >= min_hold)
        new_s[allow_exit_s] = False
        new_s[keep_s] = True

        both = new_l & new_s
        new_s[both] = False

        entered_l = new_l & ~cur_l
        hold_l[entered_l] = 1
        hold_l[new_l & cur_l] += 1
        hold_l[~new_l] = 0

        entered_s = new_s & ~cur_s
        hold_s[entered_s] = 1
        hold_s[new_s & cur_s] += 1
        hold_s[~new_s] = 0

        out_l[t] = new_l
        out_s[t] = new_s
        cur_l = new_l
        cur_s = new_s

    return (
        pd.DataFrame(out_l, index=long_m.index, columns=long_m.columns),
        pd.DataFrame(out_s, index=short_m.index, columns=short_m.columns),
    )


def plain_ls_masks(
    signal: pd.DataFrame,
    *,
    top_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranks = signal.rank(axis=1, pct=True, method="average")
    long_m = ranks >= (1.0 - top_frac)
    short_m = ranks <= top_frac
    return long_m, short_m


def weights_from_masks(
    signal: pd.DataFrame,
    long_m: pd.DataFrame,
    short_m: pd.DataFrame,
    *,
    method: str = "ew",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build long/short weights from membership masks.

    method:
      ew     — equal weight within book
      rank   — weight ∝ pct_rank within book (long higher better; short lower better)
      zscore — weight ∝ |z| within book (sign by side)
    """
    method = method.lower()
    z = cs_zscore(signal)
    ranks = signal.rank(axis=1, pct=True, method="average")

    if method == "ew":
        score_l = long_m.astype(float)
        score_s = short_m.astype(float)
    elif method == "rank":
        score_l = ranks.where(long_m)
        score_s = (1.0 - ranks).where(short_m)
    elif method == "zscore":
        score_l = z.clip(lower=0).where(long_m)
        score_s = (-z).clip(lower=0).where(short_m)
    else:
        raise ValueError(f"unknown weight method: {method}")

    long_n = score_l.notna().sum(axis=1).replace(0, np.nan)
    short_n = score_s.notna().sum(axis=1).replace(0, np.nan)
    # normalize scores to sum 1
    w_long = score_l.div(score_l.sum(axis=1).replace(0, np.nan), axis=0)
    w_short = score_s.div(score_s.sum(axis=1).replace(0, np.nan), axis=0)
    # fallback equal if all-zero scores
    w_long = w_long.fillna(0)
    w_short = w_short.fillna(0)
    # if method ew and mask True but score nan — already handled
    _ = long_n, short_n
    return w_long, w_short


def pnl_and_turnover_from_weights(
    w_long: pd.DataFrame,
    w_short: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    signal_shift: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """Apply weight known at t-shift to return at t."""
    wl = w_long.shift(signal_shift)
    ws = w_short.shift(signal_shift)
    r = ret.reindex_like(wl)
    gross = wl.fillna(0).mul(r).sum(axis=1) - ws.fillna(0).mul(r).sum(axis=1)
    w_ls = wl.fillna(0) - ws.fillna(0)
    to = w_ls.diff().abs().sum(axis=1)
    to.iloc[0] = w_ls.iloc[0].abs().sum()
    return gross, to


def evaluate_execution(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    label: str,
    stage: str,
    top_frac: float = 0.10,
    entry_frac: Optional[float] = None,
    exit_frac: Optional[float] = None,
    min_hold: int = 1,
    weight_method: str = "ew",
    rebalance_freq: int = 1,
    friday_only: bool = False,
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
    signal_shift: int = 1,
) -> dict:
    """One execution scheme → gross/net/TO/IC metrics. Signal definition unchanged."""
    sig = signal.reindex_like(ret)
    if friday_only:
        sig = friday_rebalance_signal(sig)
    elif rebalance_freq > 1:
        sig = downsample_signal(sig, rebalance_freq)

    if entry_frac is not None and exit_frac is not None:
        long_m, short_m = buffer_ls_masks(sig, entry_frac=entry_frac, exit_frac=exit_frac)
    else:
        long_m, short_m = plain_ls_masks(sig, top_frac=top_frac)

    if min_hold > 1:
        long_m, short_m = min_hold_masks(long_m, short_m, min_hold=min_hold)

    w_long, w_short = weights_from_masks(sig, long_m, short_m, method=weight_method)
    gross, to = pnl_and_turnover_from_weights(
        w_long, w_short, ret, signal_shift=signal_shift
    )
    net = net_pnl_series(gross, to, round_trip_cost)
    direction = 1 if gross.mean() >= 0 else -1
    gross_adj = gross * direction
    perf_g = series_performance(gross_adj.dropna())
    perf_n = series_performance(net.dropna())

    ic = daily_rank_ic_series(sig, ret, signal_shift=signal_shift)
    ic_clean = ic.dropna()
    rank_ic = float(ic_clean.mean()) if len(ic_clean) else np.nan
    icir = icir_from_daily(ic) if len(ic_clean) else np.nan

    daily_to = float(to.mean()) if to.notna().any() else np.nan
    return {
        "label": label,
        "stage": stage,
        "rank_ic": rank_ic,
        "icir": icir,
        "gross_sharpe": perf_g["sharpe"],
        "gross_annu_ret": perf_g["annu_ret"],
        "net_sharpe": perf_n["sharpe"],
        "net_annu_ret": perf_n["annu_ret"],
        "mdd_net": perf_n["max_drawdown"],
        "daily_turnover": daily_to,
        "annu_one_way_turnover": float(daily_to * 250.0 / 2.0) if pd.notna(daily_to) else np.nan,
        "implied_annu_fee": float(daily_to * 7.5 / 1e4 * 250.0) if pd.notna(daily_to) else np.nan,
        "direction": direction,
        "n_days": perf_n["n_days"],
        "rebalance_freq": rebalance_freq if not friday_only else "friday",
        "top_frac": top_frac,
        "entry_frac": entry_frac,
        "exit_frac": exit_frac,
        "min_hold": min_hold,
        "weight_method": weight_method,
        "round_trip_cost": round_trip_cost,
    }
