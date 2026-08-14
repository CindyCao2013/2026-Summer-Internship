"""Size-adjusted liquidity factors + orthogonal decomposition (Level 2 EOD).

Requires EOD OHLCV + float_mktcap (and optional turnover from exchange).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from factor_formulas import FactorData, FactorDataCache
from liquidity_normalization import (
    effective_turnover,
    panel_cross_sectional_residual,
    rolling_autocorr_1,
    rolling_cv,
)

_EPS = 1e-6

# Baseline (unnormalized) + normalized + orthogonalized
LIQUIDITY_NORM_CORE_LIST = [
    "amount_stability_20d",
    "amount_per_mktcap_stability_20d",
    "turnover_stability_20d",
    "volume_stability_20d",
    "liquidity_amount_residual_20d",
    "turnover_amount_residual_20d",
]

LIQUIDITY_NORM_ALL_LIST = LIQUIDITY_NORM_CORE_LIST + [
    "liquidity_persistence_norm_20d",
    "liquidity_shock_norm_20d",
]


@dataclass
class LiquidityNormData:
    """EOD fields + float mktcap for normalized liquidity factors."""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    float_mktcap: pd.DataFrame
    turnover: Optional[pd.DataFrame] = None
    total_mktcap: Optional[pd.DataFrame] = None


class LiquidityNormCache(FactorDataCache):
    """Extends FactorDataCache with mktcap-normalized liquidity state variables."""

    def __init__(self, data: LiquidityNormData):
        super().__init__(
            FactorData(
                close=data.close,
                open=data.open,
                high=data.high,
                low=data.low,
                volume=data.volume,
                amount=data.amount,
                turnover=data.turnover,
            )
        )
        self.norm_data = data

    def get(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]

        if key == "turnover_effective":
            value = effective_turnover(
                self.norm_data.turnover,
                self.require("amount"),
                self.norm_data.float_mktcap,
            )
        elif key == "amount_per_float_mktcap":
            value = (
                self.require("amount")
                / self.norm_data.float_mktcap.replace(0, np.nan)
            )
        elif key == "log_float_mktcap":
            value = np.log(self.norm_data.float_mktcap.replace(0, np.nan))
        elif key == "turnover_cv_20d":
            value = rolling_cv(self.get("turnover_effective"), 20)
        elif key == "amount_per_mktcap_cv_20d":
            value = rolling_cv(self.get("amount_per_float_mktcap"), 20)
        elif key == "volume_cv_20d":
            value = rolling_cv(self.require("volume"), 20)
        elif key == "amount_stability_raw":
            value = -self.get("amount_cv_20d")
        elif key == "volume_stability_raw":
            value = -self.get("volume_cv_20d")
        elif key == "turnover_stability_raw":
            value = -self.get("turnover_cv_20d")
        elif key == "amount_per_mktcap_stability_raw":
            value = -self.get("amount_per_mktcap_cv_20d")
        elif key == "liquidity_amount_residual":
            value = panel_cross_sectional_residual(
                self.get("amount_stability_raw"),
                [
                    self.get("volume_stability_raw"),
                    self.get("log_float_mktcap"),
                ],
            )
        elif key == "turnover_amount_residual":
            value = panel_cross_sectional_residual(
                self.get("turnover_stability_raw"),
                [
                    self.get("amount_stability_raw"),
                    self.get("log_float_mktcap"),
                ],
            )
        else:
            return super().get(key)

        self._cache[key] = value
        return value


LiquidityNormFunc = Callable[[LiquidityNormCache], pd.DataFrame]
LIQUIDITY_NORM_REGISTRY: Dict[str, LiquidityNormFunc] = {}


def register_liquidity_norm(name: str):
    def decorator(func: LiquidityNormFunc) -> LiquidityNormFunc:
        if name in LIQUIDITY_NORM_REGISTRY:
            raise ValueError(f"Duplicated liquidity_norm factor: {name}")
        LIQUIDITY_NORM_REGISTRY[name] = func
        return func

    return decorator


@register_liquidity_norm("amount_stability_20d")
def f_amount_stability_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """Baseline (unnormalized) for decomposition comparison."""
    return cache.get("amount_stability_raw")


@register_liquidity_norm("amount_per_mktcap_stability_20d")
def f_amount_per_mktcap_stability_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """CV(amount / float_mktcap): liquidity stability net of size level."""
    return cache.get("amount_per_mktcap_stability_raw")


@register_liquidity_norm("turnover_stability_20d")
def f_turnover_stability_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """CV(turnover or amount/mktcap proxy): normalized trading intensity stability."""
    return cache.get("turnover_stability_raw")


@register_liquidity_norm("volume_stability_20d")
def f_volume_stability_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """CV(volume): share-count liquidity stability (distinct from amount when price moves)."""
    return cache.get("volume_stability_raw")


@register_liquidity_norm("liquidity_amount_residual_20d")
def f_liquidity_amount_residual_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """
    amount_stability orthogonal to volume_stability + log(float_mktcap).
    Captures liquidity quality not explained by volume stability or size.
    """
    return cache.get("liquidity_amount_residual")


@register_liquidity_norm("turnover_amount_residual_20d")
def f_turnover_amount_residual_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """turnover_stability orthogonal to amount_stability + size."""
    return cache.get("turnover_amount_residual")


@register_liquidity_norm("liquidity_persistence_norm_20d")
def f_liquidity_persistence_norm_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """Autocorr of amount/float_mktcap (size-adjusted persistence)."""
    series = cache.get("amount_per_float_mktcap")
    return rolling_autocorr_1(series, 20)


@register_liquidity_norm("liquidity_shock_norm_20d")
def f_liquidity_shock_norm_20d(cache: LiquidityNormCache) -> pd.DataFrame:
    """Normalized liquidity shock: (amount/mktcap spike) × recent return."""
    intensity = cache.get("amount_per_float_mktcap")
    mean_int = intensity.rolling(20, min_periods=10).mean()
    shock = intensity / mean_int.replace(0, np.nan)
    return -(shock * cache.get("ret_5d"))


def build_liquidity_norm_cache(
    df_close: pd.DataFrame,
    df_open: pd.DataFrame,
    df_high: pd.DataFrame,
    df_low: pd.DataFrame,
    df_volume: pd.DataFrame,
    df_amount: pd.DataFrame,
    df_float_mktcap: pd.DataFrame,
    df_total_mktcap: Optional[pd.DataFrame] = None,
    df_turnover: Optional[pd.DataFrame] = None,
) -> LiquidityNormCache:
    data = LiquidityNormData(
        close=df_close,
        open=df_open,
        high=df_high,
        low=df_low,
        volume=df_volume,
        amount=df_amount,
        turnover=df_turnover,
        float_mktcap=df_float_mktcap,
        total_mktcap=df_total_mktcap,
    )
    return LiquidityNormCache(data)


def build_liquidity_norm_factor(factor_name: str, cache: LiquidityNormCache) -> pd.DataFrame:
    if factor_name not in LIQUIDITY_NORM_REGISTRY:
        valid = sorted(LIQUIDITY_NORM_REGISTRY.keys())
        raise ValueError(f"Unknown liquidity_norm factor: {factor_name}. Valid: {valid}")
    return LIQUIDITY_NORM_REGISTRY[factor_name](cache)


def filter_liquidity_norm_factors(factor_names: List[str]) -> List[str]:
    out = []
    for name in factor_names:
        if name not in LIQUIDITY_NORM_REGISTRY:
            print(f"[SKIP] Unknown liquidity_norm factor: {name}")
            continue
        out.append(name)
    return out


def build_all_liquidity_norm_factors(cache: LiquidityNormCache) -> Dict[str, pd.DataFrame]:
    return {name: build_liquidity_norm_factor(name, cache) for name in LIQUIDITY_NORM_REGISTRY}
