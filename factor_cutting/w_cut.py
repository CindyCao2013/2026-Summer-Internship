"""Generalized W-cut / leg helpers used by analysis + knife search."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from factor_cutting.engine import rolling_rank_split_sum


def w_cut(
    object_panel: pd.DataFrame,
    knife_panel: pd.DataFrame,
    *,
    window: int = 20,
    return_legs: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classic W-cut: sum(object | high knife) - sum(object | low knife).

    high/low each take window // 2 days in the lookback.
    """
    half = window // 2
    factor, high, low = rolling_rank_split_sum(
        object_panel,
        knife_panel.reindex_like(object_panel),
        window=window,
        high_count=half,
        low_count=half,
        aggregate="sum",
        require_full_window=True,
    )
    if return_legs:
        return factor, high, low
    return factor
