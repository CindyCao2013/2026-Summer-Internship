"""China A-share broker-classic factors (EOD reproduction layer).

Validated mechanisms from domestic 金工 research, screened via robust_alpha_engine.
Separate registry from generic OHLCV hypothesis mining (eod_engine_hf_v*).
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from factor_formulas import FactorDataCache
from factor_taxonomy_cn import EOD_CN_BROKER_ALL_LIST, EOD_CN_BROKER_V1_LIST

_EPS = 1e-6

CNBrokerFunc = Callable[[FactorDataCache], pd.DataFrame]
CN_BROKER_REGISTRY: Dict[str, CNBrokerFunc] = {}


def register_cn_broker(name: str):
    def decorator(func: CNBrokerFunc) -> CNBrokerFunc:
        if name in CN_BROKER_REGISTRY:
            raise ValueError(f"Duplicated cn_broker factor: {name}")
        CN_BROKER_REGISTRY[name] = func
        return func

    return decorator


def _turnover_intensity(cache: FactorDataCache) -> pd.DataFrame:
    """Turnover field or amount / rolling amount mean proxy."""
    turnover = getattr(cache.data, "turnover", None)
    if turnover is not None:
        return turnover
    amount = cache.require("amount")
    return amount / cache.get("amount_mean_20d").replace(0, np.nan)


def _intraday_return(cache: FactorDataCache) -> pd.DataFrame:
    open_ = cache.require("open")
    close = cache.data.close
    return (close - open_) / open_.replace(0, np.nan)


def _volume_zscore(cache: FactorDataCache) -> pd.DataFrame:
    volume = cache.require("volume")
    mu = volume.mean(axis=1)
    sd = volume.std(axis=1).replace(0, np.nan)
    return volume.sub(mu, axis=0).div(sd, axis=0)


def _rolling_self_percentile(df: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
    """Rolling percentile of current value vs own history (vectorized min-max proxy)."""
    rmin = df.rolling(window, min_periods=min_periods).min()
    rmax = df.rolling(window, min_periods=min_periods).max()
    return (df - rmin) / (rmax - rmin).replace(0, np.nan)


# --- 量价 ---
@register_cn_broker("cn_turnover_percentile_20d")
def f_cn_turnover_percentile_20d(cache: FactorDataCache) -> pd.DataFrame:
    turn = _turnover_intensity(cache)
    return _rolling_self_percentile(turn, window=60, min_periods=20)


@register_cn_broker("cn_turnover_change_rate_20d")
def f_cn_turnover_change_rate_20d(cache: FactorDataCache) -> pd.DataFrame:
    turn = _turnover_intensity(cache)
    t5 = turn.rolling(5, min_periods=3).mean()
    t20 = turn.rolling(20, min_periods=10).mean()
    return t5 / t20.replace(0, np.nan) - 1


@register_cn_broker("cn_volume_surge_moment_20d")
def f_cn_volume_surge_moment_20d(cache: FactorDataCache) -> pd.DataFrame:
    """方正金工: volume surge moments — EOD proxy."""
    volume = cache.require("volume")
    vmean = cache.get("volume_mean_20d")
    surge = (volume / vmean.replace(0, np.nan) - 1).clip(lower=0)
    return surge.rolling(20, min_periods=10).mean()


@register_cn_broker("cn_amount_distribution_skew_20d")
def f_cn_amount_distribution_skew_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount = cache.require("amount")
    return amount.rolling(20, min_periods=10).skew()


@register_cn_broker("cn_price_volume_divergence_20d")
def f_cn_price_volume_divergence_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20 = cache.get("ret_20d")
    vol_chg = cache.get("volume_mean_5d") / cache.get("volume_mean_20d").replace(0, np.nan) - 1
    return -ret_20 * vol_chg


# --- 行为 ---
@register_cn_broker("cn_chase_behavior_20d")
def f_cn_chase_behavior_20d(cache: FactorDataCache) -> pd.DataFrame:
    """开源金工 追涨杀跌 EOD proxy: corr(intraday_ret, volume)."""
    intra = _intraday_return(cache)
    volume = cache.require("volume")
    return intra.rolling(20, min_periods=10).corr(volume)


@register_cn_broker("cn_herding_proxy_20d")
def f_cn_herding_proxy_20d(cache: FactorDataCache) -> pd.DataFrame:
    """国盛金工 羊群效应 EOD proxy: market-sync return × volume shock."""
    ret = cache.get("ret_1d")
    mkt = ret.mean(axis=1)
    sync = ret.sub(mkt, axis=0)
    vol_shock = cache.require("volume") / cache.get("volume_mean_20d").replace(0, np.nan)
    return (sync * vol_shock).rolling(20, min_periods=10).mean()


@register_cn_broker("cn_attention_shock_5d")
def f_cn_attention_shock_5d(cache: FactorDataCache) -> pd.DataFrame:
    vol_z = _volume_zscore(cache)
    return -(vol_z * cache.get("ret_5d"))


# --- 技术 ---
@register_cn_broker("cn_new_high_breakout_252d")
def f_cn_new_high_breakout_252d(cache: FactorDataCache) -> pd.DataFrame:
    high = cache.require("high")
    close = cache.data.close
    rolling_high = high.rolling(252, min_periods=120).max()
    return close / rolling_high.replace(0, np.nan) - 1


@register_cn_broker("cn_rsi_momentum_gap_20d")
def f_cn_rsi_momentum_gap_20d(cache: FactorDataCache) -> pd.DataFrame:
    rsi_rank = cache.get("rsi_14").rank(axis=1, pct=True)
    ret_rank = cache.get("ret_20d").rank(axis=1, pct=True)
    return rsi_rank - ret_rank


@register_cn_broker("cn_shadow_combo_20d")
def f_cn_shadow_combo_20d(cache: FactorDataCache) -> pd.DataFrame:
    """东吴金工: lower shadow support minus upper shadow pressure."""
    lower = cache.get("lower_shadow")
    upper = cache.get("upper_shadow")
    return (lower - upper).rolling(20, min_periods=10).mean()


# --- A-share market structure (EOD proxy) ---
@register_cn_broker("cn_limit_up_strength_20d")
def f_cn_limit_up_strength_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret = cache.get("ret_1d")
    near_limit = (ret > 0.095).astype(float)
    return near_limit.rolling(20, min_periods=10).mean()


@register_cn_broker("cn_turnover_concentration_20d")
def f_cn_turnover_concentration_20d(cache: FactorDataCache) -> pd.DataFrame:
    turn = _turnover_intensity(cache)
    p90 = turn.rolling(60, min_periods=20).quantile(0.9)
    high_days = (turn >= p90).astype(float)
    return high_days.rolling(20, min_periods=10).mean()


def build_cn_broker_factor(factor_name: str, cache: FactorDataCache) -> pd.DataFrame:
    if factor_name not in CN_BROKER_REGISTRY:
        valid = sorted(CN_BROKER_REGISTRY.keys())
        raise ValueError(f"Unknown cn_broker factor: {factor_name}. Valid: {valid}")
    return CN_BROKER_REGISTRY[factor_name](cache)


def filter_cn_broker_factors(factor_names: List[str]) -> List[str]:
    out = []
    for name in factor_names:
        if name not in CN_BROKER_REGISTRY:
            print(f"[SKIP] Unknown cn_broker factor: {name}")
            continue
        out.append(name)
    return out
