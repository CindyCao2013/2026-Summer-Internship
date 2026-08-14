"""Ideal amplitude — Kaiyuan 《振幅因子的隐藏结构》."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from factor_cutting.engine import (
    CuttingSpec,
    KnifeSpec,
    ObjectSpec,
    OutputSpec,
    rolling_quantile_mean_diff,
)

IDEAL_AMPLITUDE_SPEC = CuttingSpec(
    name="ideal_amplitude",
    paper="振幅因子的隐藏结构",
    direction_paper="negative_ic",
    object=ObjectSpec(variable="daily_amplitude", additive=True, formula="high/low-1"),
    knife=KnifeSpec(
        variable="close_price_state",
        method="quantile_split",
        window=20,
        lambda_frac=0.25,
        formula="close level within lookback",
    ),
    output=OutputSpec(aggregate="mean", op="difference", formula="V_high - V_low"),
)


def daily_amplitude(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    return high / low.replace(0, np.nan) - 1.0


def one_word_limit_mask(
    high: pd.DataFrame,
    low: pd.DataFrame,
    *,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """True on one-word / flat-bar days (high ≈ low)."""
    return (high - low).abs() <= tol


def compute_ideal_amplitude(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    *,
    open_: Optional[pd.DataFrame] = None,
    window: int = 20,
    lambda_frac: float = 0.25,
    min_effective_days: int = 10,
    drop_one_word_limit: bool = True,
    return_legs: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    V = mean(amp | high-close λ) - mean(amp | low-close λ).

    Effective days: finite amplitude; optionally drop one-word limit days.
    """
    amp = daily_amplitude(high, low)
    knife = close.astype(float)

    if drop_one_word_limit and open_ is not None:
        bad = one_word_limit_mask(high, low)
        amp = amp.mask(bad)
        knife = knife.mask(bad)
    # also drop zero / nan amplitude
    amp = amp.where(amp > 0)

    factor, v_high, v_low = rolling_quantile_mean_diff(
        amp,
        knife,
        window=window,
        lambda_frac=lambda_frac,
        min_effective_days=min_effective_days,
        aggregate="mean",
    )
    if return_legs:
        return factor, v_high, v_low
    return factor
