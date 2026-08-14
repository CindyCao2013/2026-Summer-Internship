"""Ideal reversal (W-cut) — Kaiyuan 《A股反转之力的微观来源》."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from factor_cutting.engine import (
    CuttingSpec,
    KnifeSpec,
    ObjectSpec,
    OutputSpec,
)

IDEAL_REVERSAL_SPEC = CuttingSpec(
    name="ideal_reversal",
    paper="A股反转之力的微观来源",
    direction_paper="negative_ic",
    object=ObjectSpec(variable="daily_return", additive=True, formula="close/close.shift(1)-1"),
    knife=KnifeSpec(
        variable="avg_trade_amount",
        method="rank_split",
        window=20,
        high_count=10,
        low_count=10,
        formula="amount / trade_count",
    ),
    output=OutputSpec(aggregate="sum", op="difference", formula="M_high - M_low"),
)


def avg_trade_amount(
    amount: pd.DataFrame,
    trade_count: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, str]:
    """Build ATS knife. Prefer amount/trade_count; else amount/volume proxy."""
    if trade_count is not None:
        ats = amount / trade_count.replace(0, np.nan)
        return ats, "trade_count"
    if volume is None:
        raise ValueError("Need trade_count or volume for avg_trade_amount knife")
    ats = amount / volume.replace(0, np.nan)
    return ats, "amount_per_volume_proxy"


def compute_ideal_reversal(
    ret_1d: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    trade_count: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    window: int = 20,
    return_legs: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    M = sum(ret | high ATS) - sum(ret | low ATS) over past ``window`` days.

    Returns factor panel (Date × stock). If ``return_legs``, also M_high, M_low, knife_source.
    """
    from factor_cutting.w_cut import w_cut

    ats, source = avg_trade_amount(amount, trade_count=trade_count, volume=volume)
    ats = ats.reindex_like(ret_1d)
    if return_legs:
        factor, high, low = w_cut(ret_1d, ats, window=window, return_legs=True)
        return factor, high, low, source
    return w_cut(ret_1d, ats, window=window, return_legs=False)
