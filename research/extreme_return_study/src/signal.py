"""Extreme winner / loser signal construction."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def daily_return(close: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close daily return: close_t / close_{t-1} - 1."""
    return close / close.shift(1) - 1.0


def select_extreme_masks(
    ret_1d: pd.DataFrame,
    *,
    n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-sectional extreme selection (vectorized).

    Returns:
      loser_mask: True for bottom-n by ret_1d each day
      winner_mask: True for top-n by ret_1d each day

    Ties broken by method='first' on rank so exactly n names when possible.
    Rows with fewer than n valid names yield fewer selections.
    """
    valid_count = ret_1d.notna().sum(axis=1)
    # Rank ascending: 1 = lowest return
    rank_asc = ret_1d.rank(axis=1, method="first", ascending=True)
    # Rank descending: 1 = highest return
    rank_desc = ret_1d.rank(axis=1, method="first", ascending=False)

    n_series = pd.Series(
        np.minimum(n, valid_count.to_numpy()),
        index=ret_1d.index,
    )
    loser_mask = rank_asc.le(n_series, axis=0) & ret_1d.notna()
    winner_mask = rank_desc.le(n_series, axis=0) & ret_1d.notna()

    # Guard: if fewer than n names, still OK; if zero, all False
    has_any = (valid_count >= 1).to_numpy()[:, None]
    loser_mask = loser_mask & has_any
    winner_mask = winner_mask & has_any
    return loser_mask, winner_mask


def extreme_signal_panel(
    ret_1d: pd.DataFrame,
    *,
    n: int = 10,
) -> pd.DataFrame:
    """
    Continuous signal for IC: signed extremity of daily return.

    Higher signal = more extreme loser (for reversal hypothesis).
    Implemented as negative cross-sectional rank percentile of ret_1d
    so RankIC > 0 implies loser reversal into forward returns.
    """
    # pct rank of return: high return → high pct; negate for reversal signal
    pct = ret_1d.rank(axis=1, pct=True, method="average")
    return -pct


def formation_returns_in_universe(
    ret_1d: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Daily returns only for CSI300 members (NaN elsewhere)."""
    m = membership.reindex_like(ret_1d)
    return ret_1d.where(m == 1)
