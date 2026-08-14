"""Equal-weight extreme portfolios and overlapping holdings."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def equal_weight_from_mask(mask: pd.DataFrame) -> pd.DataFrame:
    """Row-normalize boolean/0-1 mask to equal weights (sum=1 when any True)."""
    m = mask.astype(float).where(mask, np.nan)
    n = m.notna().sum(axis=1).replace(0, np.nan)
    return m.div(n, axis=0).fillna(0.0)


def overlapping_weights(
    formation_weights: pd.DataFrame,
    *,
    hold_days: int,
    entry_lag: int = 1,
) -> pd.DataFrame:
    """
    Overlapping portfolio weights for multi-day holding.

    formation_weights[t]: target weights decided at close t
    entry_lag=1: first return day is t+1 (next open / next-day return)
    hold_days=H: positions from cohorts t stay active on days
                 t+entry_lag .. t+entry_lag+H-1

    Aggregate weight on day d = mean of active cohort weights
    (equal cohort blend — standard Jegadeesh-Titman overlapping).
    """
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    if entry_lag < 0:
        raise ValueError("entry_lag must be >= 0")

    w0 = formation_weights.fillna(0.0)
    # Shift formation into first holding day
    entered = w0.shift(entry_lag).fillna(0.0)

    # Sum of H overlapping books, then average
    stacked = entered.copy()
    for k in range(1, hold_days):
        stacked = stacked + entered.shift(k).fillna(0.0)
    agg = stacked / float(hold_days)

    # Renormalize long-only book to sum≈1 when any exposure
    row_sum = agg.sum(axis=1).replace(0, np.nan)
    return agg.div(row_sum, axis=0).fillna(0.0)


def long_short_weights(
    w_long: pd.DataFrame,
    w_short: pd.DataFrame,
) -> pd.DataFrame:
    """Dollar-neutral LS: +long - short (gross ≈ 2)."""
    return w_long.fillna(0.0) - w_short.fillna(0.0)


def portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    One-way turnover ≈ 0.5 * L1 weight change.

    For long-only (weights sum to 1): L1/2 equals fraction traded.
    For long-short (gross≈2): L1/2 ≈ one-side turnover of the book.
    """
    w = weights.fillna(0.0)
    l1 = w.diff().abs().sum(axis=1)
    l1.iloc[0] = w.iloc[0].abs().sum()
    return 0.5 * l1


def build_strategy_weights(
    loser_mask: pd.DataFrame,
    winner_mask: pd.DataFrame,
    *,
    hold_days: int,
    entry_lag: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Build overlapping equal-weight portfolios.

    Strategies:
      bottom10 / top10 / long_short (bottom - top)
    """
    w_loser_f = equal_weight_from_mask(loser_mask)
    w_winner_f = equal_weight_from_mask(winner_mask)

    w_bottom = overlapping_weights(w_loser_f, hold_days=hold_days, entry_lag=entry_lag)
    w_top = overlapping_weights(w_winner_f, hold_days=hold_days, entry_lag=entry_lag)
    w_ls = long_short_weights(w_bottom, w_top)

    return {
        "bottom10": w_bottom,
        "top10": w_top,
        "long_short": w_ls,
    }
