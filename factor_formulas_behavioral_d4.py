"""D4 Behavioral Alpha Density — candidate registry + 2 new hypotheses.

Reuses existing eod_engine / cn_broker bricks where possible; adds density-specific
formulas not yet in the generic factor zoo.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from factor_formulas import FactorDataCache

BehavioralD4Func = Callable[[FactorDataCache], pd.DataFrame]
BEHAVIORAL_D4_REGISTRY: Dict[str, BehavioralD4Func] = {}

D4_REPRESENTATIVE = "winner_sentiment_reversal_5d"

# (factor_name, source, hypothesis)
D4_BEHAVIORAL_DENSITY_CANDIDATES: List[Tuple[str, str, str]] = [
    (
        "max_daily_return_20d",
        "eod_engine",
        "Lottery effect: -max(daily return, 20d)",
    ),
    (
        "cn_new_high_breakout_252d",
        "cn_broker",
        "52-week high proximity: close/rolling_high_252 - 1",
    ),
    (
        "overreaction_shock_5d",
        "eod_engine",
        "Volume shock × 5d return reversal",
    ),
    (
        "winner_crowding_exhaustion_20d",
        "eod_engine",
        "20d return × volume shock exhaustion",
    ),
    (
        "cn_attention_shock_5d",
        "cn_broker",
        "Volume z-score × 5d return (attention spike)",
    ),
    (
        "cn_chase_behavior_20d",
        "cn_broker",
        "Rolling corr(intraday return, volume) — chase/proxy",
    ),
    (
        "d4_turnover_shock_reversal_5d",
        "behavioral_d4",
        "5d return × turnover shock — crowded move fade",
    ),
    (
        "d4_consecutive_gain_exhaustion_20d",
        "behavioral_d4",
        "20d return × up-day ratio — streak exhaustion",
    ),
]


def register_behavioral_d4(name: str):
    def decorator(func: BehavioralD4Func) -> BehavioralD4Func:
        if name in BEHAVIORAL_D4_REGISTRY:
            raise ValueError(f"Duplicated behavioral_d4 factor: {name}")
        BEHAVIORAL_D4_REGISTRY[name] = func
        return func

    return decorator


def _turnover_intensity(cache: FactorDataCache) -> pd.DataFrame:
    turnover = getattr(cache.data, "turnover", None)
    if turnover is not None:
        return turnover
    amount = cache.require("amount")
    return amount / cache.get("amount_mean_20d").replace(0, np.nan)


def _volume_shock(cache: FactorDataCache) -> pd.DataFrame:
    volume = cache.require("volume")
    return volume / cache.get("volume_mean_20d").replace(0, np.nan)


@register_behavioral_d4("d4_turnover_shock_reversal_5d")
def f_d4_turnover_shock_reversal_5d(cache: FactorDataCache) -> pd.DataFrame:
    turn = _turnover_intensity(cache)
    shock = turn / turn.rolling(20, min_periods=10).mean().replace(0, np.nan)
    return -(cache.get("ret_5d") * shock)


@register_behavioral_d4("d4_consecutive_gain_exhaustion_20d")
def f_d4_consecutive_gain_exhaustion_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_1d = cache.get("ret_1d")
    up_ratio = (ret_1d > 0).astype(float).rolling(20, min_periods=10).mean()
    return -(cache.get("ret_20d") * up_ratio)


def build_behavioral_d4_factor(name: str, cache: FactorDataCache) -> pd.DataFrame:
    if name not in BEHAVIORAL_D4_REGISTRY:
        raise ValueError(f"Unknown behavioral_d4 factor: {name}")
    return BEHAVIORAL_D4_REGISTRY[name](cache)
