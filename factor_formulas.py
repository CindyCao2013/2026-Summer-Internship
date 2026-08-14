"""日频单因子公式库：EOD 价量 / 技术类。

设计目标：
1. 不使用巨大的 if/elif 链。
2. 通过 FactorDataCache 缓存公共中间变量，避免重复 rolling。
3. 通过 FACTOR_REGISTRY 注册因子，方便扩展。
4. Factor_Test_Process.py 只负责加载数据、调用 build_factor、回测和保存结果。
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

_EPS = 1e-6

# =========================
# factor lists
# =========================

CLASSIC_FACTOR_LIST = [
    "reversal_20d",
    "reversal_60d",
    "reversal_120d",
    "reversal_240d",
    "momentum_20d",
    "momentum_60d",
    "rsi_14",
    "rsi_14_reversal",
    "volume_20d_mean",
    "volume_60d_mean",
    "amount_20d_mean",
    "amount_60d_mean",
    "turnover_20d_mean",
    "turnover_60d_mean",
    "volatility_20d",
    "volatility_60d",
    "high_low_20d",
    "high_low_60d",
    "amount_to_volatility_20d",
    "amount_to_volatility_60d",
]

NEW_EOD_FACTOR_LIST = [
    "volume_surge_no_return_20d",
    "moderate_volume_up_20d",
    "overheated_turnover_proxy_20d",
    "low_attention_reversal_20d",
    "volume_contraction_stability_20d",
    "upper_shadow_pressure_20d",
    "lower_shadow_support_20d",
    "range_contraction_20d",
    "amount_stability_20d",
    "price_volume_divergence_20d",
    "amount_acceleration_20d",
    "volume_price_efficiency_20d",
]

PRIORITY_NEW_FACTORS = [
    "low_attention_reversal_20d",
    "volume_contraction_stability_20d",
    "amount_stability_20d",
    "upper_shadow_pressure_20d",
    "lower_shadow_support_20d",
    "volume_price_efficiency_20d",
]

FACTOR_LIST = CLASSIC_FACTOR_LIST
ALL_FACTOR_LIST = CLASSIC_FACTOR_LIST + NEW_EOD_FACTOR_LIST

TURNOVER_FACTORS = {"turnover_20d_mean", "turnover_60d_mean"}


# =========================
# cache layer
# =========================

@dataclass
class FactorData:
    close: pd.DataFrame
    open: Optional[pd.DataFrame] = None
    high: Optional[pd.DataFrame] = None
    low: Optional[pd.DataFrame] = None
    volume: Optional[pd.DataFrame] = None
    amount: Optional[pd.DataFrame] = None
    turnover: Optional[pd.DataFrame] = None


class FactorDataCache:
    """Lazy cache for commonly reused factor inputs and rolling variables."""

    def __init__(self, data: FactorData):
        self.data = data
        self._cache: Dict[str, pd.DataFrame] = {}

    def require(self, field: str) -> pd.DataFrame:
        value = getattr(self.data, field)
        if value is None:
            raise ValueError(f"Required field `{field}` is not available")
        return value

    def get(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]

        close = self.data.close

        if key == "ret_1d":
            value = close / close.shift(1) - 1

        elif key.startswith("ret_"):
            window = int(key.replace("ret_", "").replace("d", ""))
            value = close / close.shift(window) - 1

        elif key == "daily_range":
            high = self.require("high")
            low = self.require("low")
            value = high / low - 1

        elif key == "rsi_14":
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14, min_periods=7).mean()
            avg_loss = loss.rolling(14, min_periods=7).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            value = 100 - 100 / (1 + rs)

        elif key.startswith("volatility_"):
            window = int(key.replace("volatility_", "").replace("d", ""))
            min_periods = 10 if window <= 20 else 20
            value = self.get("ret_1d").rolling(window, min_periods=min_periods).std()

        elif key.startswith("volume_mean_"):
            window = int(key.replace("volume_mean_", "").replace("d", ""))
            min_periods = 3 if window <= 5 else (10 if window <= 20 else 20)
            volume = self.require("volume")
            value = volume.rolling(window, min_periods=min_periods).mean()

        elif key.startswith("amount_mean_"):
            window = int(key.replace("amount_mean_", "").replace("d", ""))
            min_periods = 3 if window <= 5 else (10 if window <= 20 else 20)
            amount = self.require("amount")
            value = amount.rolling(window, min_periods=min_periods).mean()

        elif key.startswith("turnover_mean_"):
            window = int(key.replace("turnover_mean_", "").replace("d", ""))
            min_periods = 10 if window <= 20 else 20
            turnover = self.require("turnover")
            value = turnover.rolling(window, min_periods=min_periods).mean()

        elif key.startswith("high_low_mean_"):
            window = int(key.replace("high_low_mean_", "").replace("d", ""))
            min_periods = 10 if window <= 20 else 20
            value = self.get("daily_range").rolling(window, min_periods=min_periods).mean()

        elif key.startswith("daily_range_std_"):
            window = int(key.replace("daily_range_std_", "").replace("d", ""))
            min_periods = 10 if window <= 20 else 20
            value = self.get("daily_range").rolling(window, min_periods=min_periods).std()

        elif key == "upper_shadow":
            high = self.require("high")
            value = (high - close) / close

        elif key == "lower_shadow":
            low = self.require("low")
            value = (close - low) / close

        elif key == "amount_cv_20d":
            amount_mean = self.get("amount_mean_20d")
            amount = self.require("amount")
            amount_std = amount.rolling(20, min_periods=10).std()
            value = amount_std / amount_mean.replace(0, np.nan)

        else:
            raise KeyError(f"Unknown cache key: {key}")

        self._cache[key] = value
        return value


# =========================
# factor registry
# =========================

FactorFunc = Callable[[FactorDataCache], pd.DataFrame]
FACTOR_REGISTRY: Dict[str, FactorFunc] = {}


def register_factor(name: str):
    """Decorator to register a factor function."""

    def decorator(func: FactorFunc) -> FactorFunc:
        if name in FACTOR_REGISTRY:
            raise ValueError(f"Duplicated factor name: {name}")
        FACTOR_REGISTRY[name] = func
        return func

    return decorator


# =========================
# classic factors
# =========================

@register_factor("reversal_20d")
def factor_reversal_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("ret_20d")


@register_factor("reversal_60d")
def factor_reversal_60d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("ret_60d")


@register_factor("reversal_120d")
def factor_reversal_120d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("ret_120d")


@register_factor("reversal_240d")
def factor_reversal_240d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("ret_240d")


@register_factor("momentum_20d")
def factor_momentum_20d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("ret_20d")


@register_factor("momentum_60d")
def factor_momentum_60d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("ret_60d")


@register_factor("rsi_14")
def factor_rsi_14(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("rsi_14")


@register_factor("rsi_14_reversal")
def factor_rsi_14_reversal(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("rsi_14")


@register_factor("volume_20d_mean")
def factor_volume_20d_mean(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("volume_mean_20d")


@register_factor("volume_60d_mean")
def factor_volume_60d_mean(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("volume_mean_60d")


@register_factor("amount_20d_mean")
def factor_amount_20d_mean(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("amount_mean_20d")


@register_factor("amount_60d_mean")
def factor_amount_60d_mean(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("amount_mean_60d")


@register_factor("turnover_20d_mean")
def factor_turnover_20d_mean(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("turnover_mean_20d")


@register_factor("turnover_60d_mean")
def factor_turnover_60d_mean(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("turnover_mean_60d")


@register_factor("volatility_20d")
def factor_volatility_20d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("volatility_20d")


@register_factor("volatility_60d")
def factor_volatility_60d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("volatility_60d")


@register_factor("high_low_20d")
def factor_high_low_20d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("high_low_mean_20d")


@register_factor("high_low_60d")
def factor_high_low_60d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("high_low_mean_60d")


@register_factor("amount_to_volatility_20d")
def factor_amount_to_volatility_20d(cache: FactorDataCache) -> pd.DataFrame:
    amt = cache.get("amount_mean_20d")
    vol = cache.get("volatility_20d")
    return amt / vol.replace(0, np.nan)


@register_factor("amount_to_volatility_60d")
def factor_amount_to_volatility_60d(cache: FactorDataCache) -> pd.DataFrame:
    amt = cache.get("amount_mean_60d")
    vol = cache.get("volatility_60d")
    return amt / vol.replace(0, np.nan)


# =========================
# new EOD microstructure-style factors
# =========================

@register_factor("volume_surge_no_return_20d")
def factor_volume_surge_no_return_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_5d = cache.get("ret_5d")
    volume = cache.require("volume")
    vol_surge = volume / cache.get("volume_mean_20d")
    return -(vol_surge / (ret_5d.abs() + _EPS))


@register_factor("moderate_volume_up_20d")
def factor_moderate_volume_up_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20d = cache.get("ret_20d")
    vol_ratio = cache.get("volume_mean_5d") / cache.get("volume_mean_20d")
    return ret_20d * np.exp(-np.abs(vol_ratio - 1.5))


@register_factor("overheated_turnover_proxy_20d")
def factor_overheated_turnover_proxy_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_5d = cache.get("ret_5d")
    amount_ratio = cache.get("amount_mean_5d") / cache.get("amount_mean_20d")
    return -(ret_5d * amount_ratio)


@register_factor("low_attention_reversal_20d")
def factor_low_attention_reversal_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20d = cache.get("ret_20d")
    amount_ratio = cache.get("amount_mean_20d") / cache.get("amount_mean_60d")
    return -ret_20d / (amount_ratio + _EPS)


@register_factor("volume_contraction_stability_20d")
def factor_volume_contraction_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20d = cache.get("ret_20d")
    volatility = cache.get("volatility_20d")
    volume_contraction = cache.get("volume_mean_5d") / cache.get("volume_mean_20d")
    return -ret_20d / (volatility * volume_contraction + _EPS)


@register_factor("upper_shadow_pressure_20d")
def factor_upper_shadow_pressure_20d(cache: FactorDataCache) -> pd.DataFrame:
    upper_shadow = cache.get("upper_shadow")
    return -upper_shadow.rolling(20, min_periods=10).mean()


@register_factor("lower_shadow_support_20d")
def factor_lower_shadow_support_20d(cache: FactorDataCache) -> pd.DataFrame:
    lower_shadow = cache.get("lower_shadow")
    return lower_shadow.rolling(20, min_periods=10).mean()


@register_factor("range_contraction_20d")
def factor_range_contraction_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("daily_range_std_20d")


@register_factor("amount_stability_20d")
def factor_amount_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("amount_cv_20d")


@register_factor("price_volume_divergence_20d")
def factor_price_volume_divergence_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20d = cache.get("ret_20d")
    volume_change = cache.get("volume_mean_5d") / cache.get("volume_mean_20d") - 1
    return -ret_20d * volume_change


@register_factor("amount_acceleration_20d")
def factor_amount_acceleration_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount_5 = cache.get("amount_mean_5d")
    amount_20 = cache.get("amount_mean_20d")
    amount_60 = cache.get("amount_mean_60d")
    return amount_5 / amount_20 - amount_20 / amount_60


@register_factor("volume_price_efficiency_20d")
def factor_volume_price_efficiency_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20d = cache.get("ret_20d")
    amount_20 = cache.get("amount_mean_20d")
    return ret_20d / np.log1p(amount_20)


# =========================
# public API
# =========================

def build_factor(factor_name: str, cache: FactorDataCache) -> pd.DataFrame:
    """Build one factor by name."""
    if factor_name not in FACTOR_REGISTRY:
        valid = sorted(FACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown factor_name: {factor_name}. Valid factors: {valid}")
    return FACTOR_REGISTRY[factor_name](cache)


def build_factor_cache(
    df_close: pd.DataFrame,
    df_open: Optional[pd.DataFrame] = None,
    df_high: Optional[pd.DataFrame] = None,
    df_low: Optional[pd.DataFrame] = None,
    df_volume: Optional[pd.DataFrame] = None,
    df_amount: Optional[pd.DataFrame] = None,
    df_turnover: Optional[pd.DataFrame] = None,
) -> FactorDataCache:
    data = FactorData(
        close=df_close,
        open=df_open,
        high=df_high,
        low=df_low,
        volume=df_volume,
        amount=df_amount,
        turnover=df_turnover,
    )
    return FactorDataCache(data)


def available_factors() -> List[str]:
    return sorted(FACTOR_REGISTRY.keys())


def filter_available_factors(
    factor_names: Iterable[str],
    has_turnover: bool = True,
) -> List[str]:
    """Remove unavailable factors based on available data fields."""
    output = []
    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            print(f"[SKIP] Unknown factor: {name}")
            continue
        if (not has_turnover) and name in TURNOVER_FACTORS:
            print(f"[SKIP] {name}: turnover field not available")
            continue
        output.append(name)
    return output
