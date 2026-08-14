"""IC helpers for cutting legs and daily IC series."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from factor_attribution import align_signal, cs_zscore


def ic_stats(panel: pd.DataFrame, ret: pd.DataFrame, signal_shift: int = 1) -> dict:
    ic = daily_rank_ic_series(panel, ret, signal_shift=signal_shift)
    s = ic.dropna()
    return {
        "rank_ic": float(s.mean()) if len(s) else np.nan,
        "icir": icir_from_daily(ic),
        "ic_pos_ratio": float((s > 0).mean()) if len(s) else np.nan,
        "n_days": int(len(s)),
        "ic_daily": ic,
    }


def monthly_ic(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    signal_shift: int = 1,
) -> pd.Series:
    """
    Month-end signal vs next-month cumulative return (monthly rebalance IC).

    Signal at month-end t predicts product of daily returns in (t, next_month_end].
    """
    sig = align_signal(panel, signal_shift)
    r = ret.reindex_like(sig)
    # month-end dates present in index
    month_ends = sig.index.to_series().groupby(sig.index.to_period("M")).last()
    rows = []
    dates = list(month_ends.values)
    for i, d in enumerate(dates[:-1]):
        d = pd.Timestamp(d)
        d_next = pd.Timestamp(dates[i + 1])
        if d not in sig.index:
            continue
        # next-month cum ret from day after d through d_next
        window = r.loc[(r.index > d) & (r.index <= d_next)]
        if len(window) < 5:
            continue
        fut = (1.0 + window).prod(axis=0) - 1.0
        s = sig.loc[d]
        mask = s.notna() & fut.notna()
        if mask.sum() < 50:
            continue
        rows.append((d, s[mask].corr(fut[mask], method="spearman")))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({d: v for d, v in rows}).sort_index()


def monthly_ic_stats(panel: pd.DataFrame, ret: pd.DataFrame) -> dict:
    mic = monthly_ic(panel, ret)
    s = mic.dropna()
    if len(s) < 12:
        return {"monthly_rank_ic": np.nan, "monthly_icir": np.nan, "n_months": int(len(s))}
    icir = float(s.mean() / s.std() * np.sqrt(12)) if s.std() > 0 else np.nan
    return {
        "monthly_rank_ic": float(s.mean()),
        "monthly_icir": icir,
        "n_months": int(len(s)),
        "monthly_ic": mic,
    }


def apply_universe_mask(panel: pd.DataFrame, mask: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mask is None:
        return panel
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m.notna() & (m > 0))


def prepare_signal(
    panel: pd.DataFrame,
    *,
    neutralize_fn=None,
    zscore: bool = True,
) -> pd.DataFrame:
    out = panel
    if neutralize_fn is not None:
        out = neutralize_fn(out)
    if zscore:
        out = cs_zscore(out)
    return out
