"""EOD Alpha Engine v1: hypothesis-driven factors in 4 (+1 proxy) families.

Uses FactorDataCache from factor_formulas; does NOT use combinatorial generation.
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from factor_formulas import FactorDataCache
from factor_taxonomy import (
    EOD_ENGINE_ALL_LIST,
    EOD_ENGINE_CORE_LIST,
    EOD_ENGINE_HF_V2_LIST,
    EOD_ENGINE_HF_V3_LIST,
    EOD_ENGINE_HF_V4_LIST,
    EOD_ENGINE_HF_V5_LIST,
    EOD_ENGINE_ROBUST_LIST,
    EOD_ENGINE_PRIORITY_A_LIST,
)
from liquidity_normalization import (
    panel_cross_sectional_residual,
    rolling_autocorr_1,
    rolling_cv,
)

_EPS = 1e-6

EODEngineFunc = Callable[[FactorDataCache], pd.DataFrame]
EOD_ENGINE_REGISTRY: Dict[str, EODEngineFunc] = {}


def register_eod_engine(name: str):
    def decorator(func: EODEngineFunc) -> EODEngineFunc:
        if name in EOD_ENGINE_REGISTRY:
            raise ValueError(f"Duplicated eod_engine factor: {name}")
        EOD_ENGINE_REGISTRY[name] = func
        return func

    return decorator


def _drawup_drawdown_ratio(ret_1d: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    up = ret_1d.clip(lower=0).rolling(window, min_periods=10).sum()
    dd = ret_1d.clip(upper=0).abs().rolling(window, min_periods=10).max()
    return up / (dd + _EPS)


def _return_autocorr(ret_1d: pd.DataFrame, lag: int = 1, window: int = 5) -> pd.DataFrame:
    lagged = ret_1d.shift(lag)
    return ret_1d.rolling(window, min_periods=3).corr(lagged)


def _series_autocorr(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return df.rolling(window, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x.dropna()) > 2 else np.nan,
        raw=False,
    )


def _amihud_daily(cache: FactorDataCache) -> pd.DataFrame:
    ret_1d = cache.get("ret_1d")
    amount = cache.require("amount")
    return ret_1d.abs() / amount.replace(0, np.nan)


def _amihud_mean_20d(cache: FactorDataCache) -> pd.DataFrame:
    return _amihud_daily(cache).rolling(20, min_periods=10).mean()


def _volume_shock(cache: FactorDataCache) -> pd.DataFrame:
    volume = cache.require("volume")
    return volume / cache.get("volume_mean_20d").replace(0, np.nan)


def _cross_section_rank_mean(*dfs: pd.DataFrame) -> pd.DataFrame:
    ranked = [df.rank(axis=1, pct=True) for df in dfs]
    return sum(ranked) / len(ranked)


def _vol_regime_shock(cache: FactorDataCache) -> pd.DataFrame:
    vol_5 = cache.get("ret_1d").rolling(5, min_periods=3).std()
    vol_20 = cache.get("volatility_20d")
    return vol_5 / vol_20.replace(0, np.nan)


def _liquidity_shock(cache: FactorDataCache) -> pd.DataFrame:
    amount = cache.require("amount")
    return amount / cache.get("amount_mean_20d").replace(0, np.nan)


# --- return structure ---
@register_eod_engine("trend_consistency_20d")
def f_trend_consistency_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_1d = cache.get("ret_1d")
    signs = np.sign(ret_1d)
    return signs.rolling(20, min_periods=10).mean()


@register_eod_engine("drawup_drawdown_ratio_20d")
def f_drawup_drawdown_ratio_20d(cache: FactorDataCache) -> pd.DataFrame:
    return _drawup_drawdown_ratio(cache.get("ret_1d"), 20)


@register_eod_engine("return_autocorr_5d")
def f_return_autocorr_5d(cache: FactorDataCache) -> pd.DataFrame:
    return _return_autocorr(cache.get("ret_1d"), lag=1, window=5)


# --- liquidity structure ---
@register_eod_engine("liquidity_stability_20d")
def f_liquidity_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("amount_cv_20d")


@register_eod_engine("liquidity_shock_20d")
def f_liquidity_shock_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount = cache.require("amount")
    shock = amount / cache.get("amount_mean_20d")
    ret_5d = cache.get("ret_5d")
    return -(shock * ret_5d)


@register_eod_engine("liquidity_persistence_20d")
def f_liquidity_persistence_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount = cache.require("amount")
    return _series_autocorr(amount, 20)


@register_eod_engine("liquidity_acceleration_20d")
def f_liquidity_acceleration_20d(cache: FactorDataCache) -> pd.DataFrame:
    v5 = cache.get("volume_mean_5d")
    v20 = cache.get("volume_mean_20d")
    v60 = cache.get("volume_mean_60d")
    return v5 / v20 - v20 / v60


# --- risk structure ---
@register_eod_engine("volatility_level_20d")
def f_volatility_level_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -cache.get("volatility_20d")


@register_eod_engine("volatility_regime_change_20d")
def f_volatility_regime_change_20d(cache: FactorDataCache) -> pd.DataFrame:
    vol_5 = cache.get("ret_1d").rolling(5, min_periods=3).std()
    vol_20 = cache.get("volatility_20d")
    return vol_5 / vol_20.replace(0, np.nan) - 1


@register_eod_engine("vol_of_vol_20d")
def f_vol_of_vol_20d(cache: FactorDataCache) -> pd.DataFrame:
    vol = cache.get("volatility_20d")
    return -vol.rolling(20, min_periods=10).std()


@register_eod_engine("range_expansion_20d")
def f_range_expansion_20d(cache: FactorDataCache) -> pd.DataFrame:
    daily_range = cache.get("daily_range")
    mean_r = daily_range.rolling(20, min_periods=10).mean()
    return -(daily_range / mean_r.replace(0, np.nan))


# --- behavioral structure ---
@register_eod_engine("overreaction_shock_5d")
def f_overreaction_shock_5d(cache: FactorDataCache) -> pd.DataFrame:
    volume = cache.require("volume")
    shock = volume / cache.get("volume_mean_20d")
    return -(cache.get("ret_5d") * shock)


@register_eod_engine("underreaction_gap_20d")
def f_underreaction_gap_20d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("ret_20d") - cache.get("ret_5d")


@register_eod_engine("volume_price_divergence_20d")
def f_volume_price_divergence_20d(cache: FactorDataCache) -> pd.DataFrame:
    vol_chg = cache.get("volume_mean_5d") / cache.get("volume_mean_20d") - 1
    return -cache.get("ret_20d") * vol_chg


# --- microstructure proxy (EOD compression of intraday path) ---
@register_eod_engine("close_location_value_20d")
def f_close_location_value_20d(cache: FactorDataCache) -> pd.DataFrame:
    high = cache.require("high")
    low = cache.require("low")
    close = cache.data.close
    clv = (close - low) / (high - low + _EPS)
    return clv.rolling(20, min_periods=10).mean()


@register_eod_engine("price_inefficiency_20d")
def f_price_inefficiency_20d(cache: FactorDataCache) -> pd.DataFrame:
    open_ = cache.require("open")
    close = cache.data.close
    high = cache.require("high")
    low = cache.require("low")
    ineff = (close - open_).abs() / (high - low + _EPS)
    return -ineff.rolling(20, min_periods=10).mean()


@register_eod_engine("net_volume_pressure_20d")
def f_net_volume_pressure_20d(cache: FactorDataCache) -> pd.DataFrame:
    open_ = cache.require("open")
    close = cache.data.close
    volume = cache.require("volume")
    pressure = np.sign(close - open_) * volume
    return pressure.rolling(20, min_periods=10).mean()


# --- Priority A: literature-backed new alpha ---
@register_eod_engine("amihud_illiquidity_20d")
def f_amihud_illiquidity_20d(cache: FactorDataCache) -> pd.DataFrame:
    """Amihud (2002) illiquidity: |ret|/amount, 20d mean (negated = prefer liquid)."""
    return -_amihud_mean_20d(cache)


@register_eod_engine("amihud_shock_reversal_5d")
def f_amihud_shock_reversal_5d(cache: FactorDataCache) -> pd.DataFrame:
    amihud = _amihud_daily(cache)
    shock = amihud / _amihud_mean_20d(cache).replace(0, np.nan)
    return -(shock * cache.get("ret_5d"))


@register_eod_engine("max_daily_return_20d")
def f_max_daily_return_20d(cache: FactorDataCache) -> pd.DataFrame:
    """Lottery effect: avoid stocks with extreme single-day spikes."""
    ret_1d = cache.get("ret_1d")
    return -ret_1d.rolling(20, min_periods=10).max()


@register_eod_engine("cn_trend_pv_20d")
def f_cn_trend_pv_20d(cache: FactorDataCache) -> pd.DataFrame:
    """Liu-Zhou-Zhu style price trend scaled by volume trend (retail participation)."""
    ret_20d = cache.get("ret_20d")
    vol_trend = cache.get("volume_mean_5d") / cache.get("volume_mean_60d").replace(0, np.nan)
    return ret_20d * vol_trend


@register_eod_engine("loser_liquidity_reversal_5d")
def f_loser_liquidity_reversal_5d(cache: FactorDataCache) -> pd.DataFrame:
    """Decomposed reversal: liquidity shock on recent losers only."""
    ret_5d = cache.get("ret_5d")
    shock = _volume_shock(cache)
    factor = -(ret_5d * shock)
    return factor.where(ret_5d < 0, 0.0)


@register_eod_engine("winner_sentiment_reversal_5d")
def f_winner_sentiment_reversal_5d(cache: FactorDataCache) -> pd.DataFrame:
    """Decomposed reversal: sentiment / volume spike on recent winners only."""
    ret_5d = cache.get("ret_5d")
    shock = _volume_shock(cache)
    factor = -(ret_5d * shock)
    return factor.where(ret_5d > 0, 0.0)


@register_eod_engine("amihud_amount_orth_20d")
def f_amihud_amount_orth_20d(cache: FactorDataCache) -> pd.DataFrame:
    """Illiquidity stability orthogonal to amount stability + log(amount level)."""
    amihud_stab = -rolling_cv(_amihud_daily(cache), 20)
    amount_stab = -cache.get("amount_cv_20d")
    log_amount = np.log(cache.get("amount_mean_20d").replace(0, np.nan))
    return panel_cross_sectional_residual(amihud_stab, [amount_stab, log_amount])


# --- HF v2: flow direction, 2nd-order dynamics, CS relative structure ---
@register_eod_engine("net_inflow_asymmetry_20d")
def f_net_inflow_asymmetry_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount = cache.require("amount")
    open_ = cache.require("open")
    close = cache.data.close
    sign = np.sign(close - open_)
    up = amount.where(sign > 0, 0.0)
    down = amount.where(sign < 0, 0.0)
    asym = (up - down) / (up + down + _EPS)
    return asym.rolling(20, min_periods=10).mean()


@register_eod_engine("amount_acceleration_20d")
def f_amount_acceleration_20d(cache: FactorDataCache) -> pd.DataFrame:
    a5 = cache.get("amount_mean_5d")
    a20 = cache.get("amount_mean_20d")
    a60 = cache.get("amount_mean_60d")
    return a5 / a20.replace(0, np.nan) - a20 / a60.replace(0, np.nan)


@register_eod_engine("flow_persistence_decay_20d")
def f_flow_persistence_decay_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount = cache.require("amount")
    persistence = rolling_autocorr_1(amount, 20)
    return persistence - persistence.shift(1)


@register_eod_engine("intraday_reversal_intensity_20d")
def f_intraday_reversal_intensity_20d(cache: FactorDataCache) -> pd.DataFrame:
    high = cache.require("high")
    low = cache.require("low")
    close = cache.data.close
    midpoint = (high + low) / 2.0
    intensity = (midpoint - close) / close.replace(0, np.nan)
    return intensity.rolling(20, min_periods=10).mean()


@register_eod_engine("range_entropy_20d")
def f_range_entropy_20d(cache: FactorDataCache) -> pd.DataFrame:
    daily_range = cache.get("daily_range")
    return -rolling_cv(daily_range, 20)


@register_eod_engine("attention_shock_cs_5d")
def f_attention_shock_cs_5d(cache: FactorDataCache) -> pd.DataFrame:
    volume = cache.require("volume")
    mu = volume.mean(axis=1)
    sd = volume.std(axis=1).replace(0, np.nan)
    vol_z = volume.sub(mu, axis=0).div(sd, axis=0)
    return -(vol_z * cache.get("ret_5d"))


@register_eod_engine("winner_crowding_exhaustion_20d")
def f_winner_crowding_exhaustion_20d(cache: FactorDataCache) -> pd.DataFrame:
    shock = _volume_shock(cache)
    return -(cache.get("ret_20d") * shock)


@register_eod_engine("loser_panic_stabilization_20d")
def f_loser_panic_stabilization_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_20d = cache.get("ret_20d")
    volume = cache.require("volume")
    low_vol_state = 1.0 - (volume / cache.get("volume_mean_20d").replace(0, np.nan)).clip(0, 2) / 2.0
    decline = (-ret_20d).clip(lower=0)
    return decline * low_vol_state


@register_eod_engine("relative_liquidity_strength_20d")
def f_relative_liquidity_strength_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount_mean = cache.get("amount_mean_20d")
    cs_mean = amount_mean.mean(axis=1)
    return amount_mean.div(cs_mean.replace(0, np.nan), axis=0)


@register_eod_engine("momentum_rank_dispersion_20d")
def f_momentum_rank_dispersion_20d(cache: FactorDataCache) -> pd.DataFrame:
    r20 = cache.get("ret_20d").rank(axis=1, pct=True)
    r5 = cache.get("ret_5d").rank(axis=1, pct=True)
    return r20 - r5


@register_eod_engine("low_vol_liquidity_quality_20d")
def f_low_vol_liquidity_quality_20d(cache: FactorDataCache) -> pd.DataFrame:
    """Low vol + stable amount: penalize vol*CV (both low → higher score)."""
    vol = cache.get("volatility_20d")
    cv = cache.get("amount_cv_20d")
    return -(vol * cv)


@register_eod_engine("tail_risk_min_return_20d")
def f_tail_risk_min_return_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret_1d = cache.get("ret_1d")
    return ret_1d.rolling(20, min_periods=10).min()


@register_eod_engine("volatility_adjusted_momentum_20d")
def f_volatility_adjusted_momentum_20d(cache: FactorDataCache) -> pd.DataFrame:
    return cache.get("ret_20d") / cache.get("volatility_20d").replace(0, np.nan)


# --- HF v3: nonlinear coupling / multiscale / tail completion ---
@register_eod_engine("vol_liquidity_stress_20d")
def f_vol_liquidity_stress_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -(_amihud_mean_20d(cache) * cache.get("volatility_20d"))


@register_eod_engine("liquidity_fragility_20d")
def f_liquidity_fragility_20d(cache: FactorDataCache) -> pd.DataFrame:
    amihud = _amihud_daily(cache)
    std5 = amihud.rolling(5, min_periods=3).std()
    std20 = amihud.rolling(20, min_periods=10).std()
    return std5 / std20.replace(0, np.nan)


@register_eod_engine("vol_liquidity_rank_gap_20d")
def f_vol_liquidity_rank_gap_20d(cache: FactorDataCache) -> pd.DataFrame:
    vol_rank = cache.get("volatility_20d").rank(axis=1, pct=True)
    liq_rank = cache.get("amount_mean_20d").rank(axis=1, pct=True)
    return vol_rank - liq_rank


@register_eod_engine("momentum_timescale_conflict_20d")
def f_momentum_timescale_conflict_20d(cache: FactorDataCache) -> pd.DataFrame:
    r5 = cache.get("ret_5d").rank(axis=1, pct=True)
    r60 = cache.get("ret_60d").rank(axis=1, pct=True)
    return r5 - r60


@register_eod_engine("flow_price_rank_gap_20d")
def f_flow_price_rank_gap_20d(cache: FactorDataCache) -> pd.DataFrame:
    flow_rank = cache.get("amount_mean_5d").rank(axis=1, pct=True)
    price_rank = cache.get("ret_20d").rank(axis=1, pct=True)
    return flow_rank - price_rank


@register_eod_engine("momentum_regime_flip_20d")
def f_momentum_regime_flip_20d(cache: FactorDataCache) -> pd.DataFrame:
    r5 = cache.get("ret_5d")
    r20 = cache.get("ret_20d")
    corr = r5.rolling(20, min_periods=10).corr(r20)
    return corr.diff(5)


@register_eod_engine("downside_tail_cluster_20d")
def f_downside_tail_cluster_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret = cache.get("ret_1d")
    vol = ret.rolling(20, min_periods=10).std()
    hits = (ret < -2 * vol).astype(float)
    return hits.rolling(20, min_periods=10).sum()


@register_eod_engine("upside_fragility_20d")
def f_upside_fragility_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret = cache.get("ret_1d")
    peak = ret.rolling(5, min_periods=3).max()
    baseline = ret.rolling(20, min_periods=10).mean()
    return peak - baseline


@register_eod_engine("asymmetric_tail_ratio_20d")
def f_asymmetric_tail_ratio_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret = cache.get("ret_1d")
    down = ret.clip(upper=0).pow(2).rolling(20, min_periods=10).mean()
    up = ret.clip(lower=0).pow(2).rolling(20, min_periods=10).mean()
    return down / up.replace(0, np.nan)


@register_eod_engine("momentum_rank_churn_20d")
def f_momentum_rank_churn_20d(cache: FactorDataCache) -> pd.DataFrame:
    rank = cache.get("ret_20d").rank(axis=1, pct=True)
    return rank.diff().abs().rolling(20, min_periods=10).mean()


@register_eod_engine("return_skew_shift_20d")
def f_return_skew_shift_20d(cache: FactorDataCache) -> pd.DataFrame:
    ret = cache.get("ret_1d")
    skew20 = ret.rolling(20, min_periods=10).skew()
    skew60 = ret.rolling(60, min_periods=30).skew()
    return skew20 - skew60


@register_eod_engine("skew_20d")
def f_skew_20d(cache: FactorDataCache) -> pd.DataFrame:
    """Lottery skewness anomaly (Alpha = -SKEW20). Canonical: core/factors/skew."""
    from core.factors.skew.skew import alpha_from_skew, skew_20d

    return alpha_from_skew(skew_20d(cache.get("ret_1d")))


@register_eod_engine("skew_60d")
def f_skew_60d(cache: FactorDataCache) -> pd.DataFrame:
    """Lottery skewness anomaly (Alpha = -SKEW60). Canonical: core/factors/skew."""
    from core.factors.skew.skew import alpha_from_skew, skew_60d

    return alpha_from_skew(skew_60d(cache.get("ret_1d")))


@register_eod_engine("skew_120d")
def f_skew_120d(cache: FactorDataCache) -> pd.DataFrame:
    """Lottery skewness anomaly (Alpha = -SKEW120). Canonical: core/factors/skew."""
    from core.factors.skew.skew import alpha_from_skew, skew_120d

    return alpha_from_skew(skew_120d(cache.get("ret_1d")))


# --- HF v4: stability extensions + rank composites ---
@register_eod_engine("composite_liquidity_stability_20d")
def f_composite_liquidity_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount_stab = -cache.get("amount_cv_20d")
    vol_stab = -rolling_cv(cache.require("volume"), 20)
    range_stab = -rolling_cv(cache.get("daily_range"), 20)
    return _cross_section_rank_mean(amount_stab, vol_stab, range_stab)


@register_eod_engine("amihud_stability_20d")
def f_amihud_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -rolling_cv(_amihud_daily(cache), 20)


@register_eod_engine("return_stability_20d")
def f_return_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    return -rolling_cv(cache.get("ret_1d"), 20)


@register_eod_engine("amount_stability_60d")
def f_amount_stability_60d(cache: FactorDataCache) -> pd.DataFrame:
    return -rolling_cv(cache.require("amount"), 60, min_periods=30)


@register_eod_engine("shadow_stability_20d")
def f_shadow_stability_20d(cache: FactorDataCache) -> pd.DataFrame:
    shadow = cache.get("upper_shadow") + cache.get("lower_shadow")
    return -rolling_cv(shadow, 20)


@register_eod_engine("stability_quality_composite_20d")
def f_stability_quality_composite_20d(cache: FactorDataCache) -> pd.DataFrame:
    amount_stab = -cache.get("amount_cv_20d")
    low_vol = -cache.get("volatility_20d")
    range_stab = -rolling_cv(cache.get("daily_range"), 20)
    return _cross_section_rank_mean(amount_stab, low_vol, range_stab)


@register_eod_engine("low_vol_stability_rank_20d")
def f_low_vol_stability_rank_20d(cache: FactorDataCache) -> pd.DataFrame:
    quality = -(cache.get("volatility_20d") * cache.get("amount_cv_20d"))
    range_stab = -rolling_cv(cache.get("daily_range"), 20)
    return _cross_section_rank_mean(quality, range_stab)


@register_eod_engine("stable_reversal_blend_20d")
def f_stable_reversal_blend_20d(cache: FactorDataCache) -> pd.DataFrame:
    reversal = -cache.get("ret_20d")
    stab = -cache.get("amount_cv_20d")
    return _cross_section_rank_mean(reversal, stab)


# --- HF v5: second-order alpha (Signal × State) ---
@register_eod_engine("liquidity_conditioned_momentum_20d")
def f_liquidity_conditioned_momentum_20d(cache: FactorDataCache) -> pd.DataFrame:
    """LMQ: momentum × liquidity stability — rising on stable flow."""
    return cache.get("ret_20d") * (-cache.get("amount_cv_20d"))


@register_eod_engine("liquidity_shock_recovery_5d")
def f_liquidity_shock_recovery_5d(cache: FactorDataCache) -> pd.DataFrame:
    """LSR: Amihud recovery × short return — post-shock absorption."""
    amihud = _amihud_daily(cache)
    delta_amihud = amihud - amihud.shift(5)
    return -delta_amihud * cache.get("ret_5d")


@register_eod_engine("triple_crowding_exhaustion_20d")
def f_triple_crowding_exhaustion_20d(cache: FactorDataCache) -> pd.DataFrame:
    """WCE+: up × volume spike × vol expansion — triple crowding stress."""
    return -(cache.get("ret_20d") * _volume_shock(cache) * _vol_regime_shock(cache))


@register_eod_engine("trend_quality_composite_20d")
def f_trend_quality_composite_20d(cache: FactorDataCache) -> pd.DataFrame:
    """TQ: momentum × path efficiency × day consistency."""
    open_ = cache.require("open")
    close = cache.data.close
    high = cache.require("high")
    low = cache.require("low")
    hl_range = (high - low).replace(0, np.nan)
    efficiency = (close - open_).abs() / hl_range
    eff_mean = efficiency.rolling(20, min_periods=10).mean()
    consistency = (cache.get("ret_1d") > 0).astype(float).rolling(20, min_periods=10).mean()
    return cache.get("ret_20d") * eff_mean * consistency


@register_eod_engine("liquidity_vol_regime_20d")
def f_liquidity_vol_regime_20d(cache: FactorDataCache) -> pd.DataFrame:
    """LVR: liquidity shock relative to volatility regime."""
    return _liquidity_shock(cache) / _vol_regime_shock(cache).replace(0, np.nan)


@register_eod_engine("tail_adjusted_momentum_60d")
def f_tail_adjusted_momentum_60d(cache: FactorDataCache) -> pd.DataFrame:
    """TAM: 60d return per unit downside volatility."""
    ret_60 = cache.get("ret_60d")
    downside = cache.get("ret_1d").clip(upper=0)
    downside_vol = downside.rolling(60, min_periods=30).std()
    return ret_60 / downside_vol.replace(0, np.nan)


@register_eod_engine("flow_price_divergence_20d")
def f_flow_price_divergence_20d(cache: FactorDataCache) -> pd.DataFrame:
    """FPD: flow acceleration rank minus return rank."""
    amount_accel = (
        cache.get("amount_mean_5d") / cache.get("amount_mean_20d").replace(0, np.nan) - 1
    )
    flow_rank = amount_accel.rank(axis=1, pct=True)
    ret_rank = cache.get("ret_20d").rank(axis=1, pct=True)
    return flow_rank - ret_rank


@register_eod_engine("liquidity_accel_risk_filtered_20d")
def f_liquidity_accel_risk_filtered_20d(cache: FactorDataCache) -> pd.DataFrame:
    """LA2: second derivative of log(amount), filtered by low vol."""
    log_amount = np.log(cache.require("amount").replace(0, np.nan))
    d1 = log_amount.diff(1)
    d2 = d1.diff(1)
    la2 = d2.rolling(20, min_periods=10).mean()
    return la2 * (-cache.get("volatility_20d"))


# --- Robust alpha: residual / risk-adjusted / CS-normalized ---
def _log_amount_mean(cache: FactorDataCache) -> pd.DataFrame:
    return np.log(cache.get("amount_mean_20d").replace(0, np.nan))


@register_eod_engine("residual_momentum_60d")
def f_residual_momentum_60d(cache: FactorDataCache) -> pd.DataFrame:
    """CS residual 60d momentum after removing short mom, vol, size proxy."""
    ret_60 = cache.get("ret_60d")
    xs = [cache.get("ret_20d"), cache.get("volatility_20d"), _log_amount_mean(cache)]
    return panel_cross_sectional_residual(ret_60, xs)


@register_eod_engine("information_ratio_momentum_60d")
def f_information_ratio_momentum_60d(cache: FactorDataCache) -> pd.DataFrame:
    vol_60 = cache.get("ret_1d").rolling(60, min_periods=30).std()
    return cache.get("ret_60d") / vol_60.replace(0, np.nan)


@register_eod_engine("information_ratio_momentum_120d")
def f_information_ratio_momentum_120d(cache: FactorDataCache) -> pd.DataFrame:
    vol_120 = cache.get("ret_1d").rolling(120, min_periods=60).std()
    return cache.get("ret_120d") / vol_120.replace(0, np.nan)


@register_eod_engine("residual_liquidity_20d")
def f_residual_liquidity_20d(cache: FactorDataCache) -> pd.DataFrame:
    liq = -cache.get("amount_cv_20d")
    xs = [_log_amount_mean(cache), cache.get("volatility_20d")]
    return panel_cross_sectional_residual(liq, xs)


@register_eod_engine("relative_vol_adjusted_liquidity_20d")
def f_relative_vol_adjusted_liquidity_20d(cache: FactorDataCache) -> pd.DataFrame:
    rvl = cache.get("amount_mean_20d") / cache.get("volatility_20d").replace(0, np.nan)
    return rvl.rank(axis=1, pct=True)


@register_eod_engine("low_vol_liquidity_quality_60d")
def f_low_vol_liquidity_quality_60d(cache: FactorDataCache) -> pd.DataFrame:
    vol_60 = cache.get("ret_1d").rolling(60, min_periods=30).std()
    stab = -cache.get("amount_cv_20d")
    return _cross_section_rank_mean(-vol_60, stab)


@register_eod_engine("stability_signal_persistence_20d")
def f_stability_signal_persistence_20d(cache: FactorDataCache) -> pd.DataFrame:
    stab = -cache.get("amount_cv_20d")
    return rolling_autocorr_1(stab, 20)


@register_eod_engine("residual_reversal_20d")
def f_residual_reversal_20d(cache: FactorDataCache) -> pd.DataFrame:
    rev = -cache.get("ret_20d")
    xs = [cache.get("ret_60d"), cache.get("volatility_20d")]
    return panel_cross_sectional_residual(rev, xs)


def build_eod_engine_factor(factor_name: str, cache: FactorDataCache) -> pd.DataFrame:
    if factor_name not in EOD_ENGINE_REGISTRY:
        valid = sorted(EOD_ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown eod_engine factor: {factor_name}. Valid: {valid}")
    return EOD_ENGINE_REGISTRY[factor_name](cache)


def filter_eod_engine_factors(factor_names: List[str]) -> List[str]:
    out = []
    for name in factor_names:
        if name not in EOD_ENGINE_REGISTRY:
            print(f"[SKIP] Unknown eod_engine factor: {name}")
            continue
        out.append(name)
    return out
