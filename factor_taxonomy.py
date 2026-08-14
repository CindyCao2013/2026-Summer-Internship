"""EOD Alpha Engine taxonomy: hypothesis-driven factor families.

Each factor maps to an economic mechanism — no random combinatorial generation.
"""

from typing import Dict, List, TypedDict


class FactorMeta(TypedDict):
    family: str
    hypothesis: str
    mechanism: str
    direction_hint: str


FAMILY_RETURN = "return_structure"
FAMILY_LIQUIDITY = "liquidity_structure"
FAMILY_RISK = "risk_structure"
FAMILY_BEHAVIORAL = "behavioral_structure"
FAMILY_MICROSTRUCTURE = "microstructure_proxy"

# Core 8 signals (HF EOD alpha engine v1)
EOD_ENGINE_CORE_LIST = [
    "trend_consistency_20d",
    "liquidity_stability_20d",
    "liquidity_shock_20d",
    "volatility_level_20d",
    "volatility_regime_change_20d",
    "return_autocorr_5d",
    "drawup_drawdown_ratio_20d",
    "volume_price_divergence_20d",
]

# Extended structured pool (same families, more coverage)
EOD_ENGINE_EXTENDED_LIST = EOD_ENGINE_CORE_LIST + [
    "liquidity_persistence_20d",
    "vol_of_vol_20d",
    "range_expansion_20d",
    "overreaction_shock_5d",
    "underreaction_gap_20d",
    "close_location_value_20d",
    "price_inefficiency_20d",
    "net_volume_pressure_20d",
    "liquidity_acceleration_20d",
]

EOD_ENGINE_ALL_LIST = EOD_ENGINE_EXTENDED_LIST

# Priority A: literature-backed new alpha (Amihud, CN trend, decomposed reversal)
EOD_ENGINE_PRIORITY_A_LIST = [
    "amihud_illiquidity_20d",
    "amihud_shock_reversal_5d",
    "max_daily_return_20d",
    "cn_trend_pv_20d",
    "loser_liquidity_reversal_5d",
    "winner_sentiment_reversal_5d",
    "amihud_amount_orth_20d",
]

# Higher-moment / lottery family (research-grade SKEW pack; Alpha = -raw skew)
EOD_ENGINE_SKEW_LIST = [
    "skew_20d",
    "skew_60d",
    "skew_120d",
]

FAMILY_CROSS_SECTION = "cross_sectional_structure"
FAMILY_NONLINEAR_COUPLING = "nonlinear_liq_vol"
FAMILY_MULTISCALE = "multiscale_disagreement"
FAMILY_TAIL = "tail_structure"

# HF v2: new economic dimensions (flow direction, 2nd-order dynamics, CS relative)
EOD_ENGINE_HF_V2_LIST = [
    "net_inflow_asymmetry_20d",
    "amount_acceleration_20d",
    "flow_persistence_decay_20d",
    "intraday_reversal_intensity_20d",
    "range_entropy_20d",
    "attention_shock_cs_5d",
    "winner_crowding_exhaustion_20d",
    "loser_panic_stabilization_20d",
    "relative_liquidity_strength_20d",
    "momentum_rank_dispersion_20d",
    "low_vol_liquidity_quality_20d",
    "tail_risk_min_return_20d",
    "volatility_adjusted_momentum_20d",
]

# HF v3: mechanism coverage completion (distortion / nonlinear coupling / multiscale / tail)
EOD_ENGINE_HF_V3_LIST = [
    "vol_liquidity_stress_20d",
    "liquidity_fragility_20d",
    "vol_liquidity_rank_gap_20d",
    "momentum_timescale_conflict_20d",
    "flow_price_rank_gap_20d",
    "momentum_regime_flip_20d",
    "downside_tail_cluster_20d",
    "upside_fragility_20d",
    "asymmetric_tail_ratio_20d",
    "momentum_rank_churn_20d",
    "return_skew_shift_20d",
]

# HF v4: stability-family extensions + rank composites (target Sharpe>3 / mono H-L)
EOD_ENGINE_HF_V4_LIST = [
    "composite_liquidity_stability_20d",
    "amihud_stability_20d",
    "return_stability_20d",
    "amount_stability_60d",
    "shadow_stability_20d",
    "stability_quality_composite_20d",
    "low_vol_stability_rank_20d",
    "stable_reversal_blend_20d",
]

# HF v5: second-order alpha = Signal × State (conditional interaction layer)
EOD_ENGINE_HF_V5_LIST = [
    "liquidity_conditioned_momentum_20d",
    "liquidity_shock_recovery_5d",
    "triple_crowding_exhaustion_20d",
    "trend_quality_composite_20d",
    "liquidity_vol_regime_20d",
    "tail_adjusted_momentum_60d",
    "flow_price_divergence_20d",
    "liquidity_accel_risk_filtered_20d",
]

# Robust alpha: residual / risk-adjusted / CS-normalized (stability-first)
EOD_ENGINE_ROBUST_LIST = [
    "residual_momentum_60d",
    "information_ratio_momentum_60d",
    "information_ratio_momentum_120d",
    "residual_liquidity_20d",
    "relative_vol_adjusted_liquidity_20d",
    "low_vol_liquidity_quality_60d",
    "stability_signal_persistence_20d",
    "residual_reversal_20d",
]

# Mechanism layers for alpha universe completeness (HF-grade coverage map)
LAYER_LIQUIDITY_FLOW = "liquidity_flow"
LAYER_REVERSAL_SHOCK = "reversal_shock"
LAYER_MOMENTUM = "momentum_structure"
LAYER_MICROSTRUCTURE = "microstructure_proxy"
LAYER_BEHAVIORAL = "behavioral_crowding"
LAYER_CS_DISTORTION = "cross_sectional_distortion"
LAYER_NONLINEAR_LIQ_VOL = "nonlinear_liq_vol"
LAYER_MULTISCALE = "multiscale_disagreement"
LAYER_TAIL = "tail_structure"
LAYER_ROTATION = "rotation_leadership"

MECHANISM_LAYERS = [
    LAYER_LIQUIDITY_FLOW,
    LAYER_REVERSAL_SHOCK,
    LAYER_MOMENTUM,
    LAYER_MICROSTRUCTURE,
    LAYER_BEHAVIORAL,
    LAYER_CS_DISTORTION,
    LAYER_NONLINEAR_LIQ_VOL,
    LAYER_MULTISCALE,
    LAYER_TAIL,
    LAYER_ROTATION,
]

# Factors not yet implementable on EOD-only data (fundamental / industry required)
PLANNED_MECHANISM_GAPS = [
    {
        "id": "valuation_dispersion_pe",
        "layer": LAYER_CS_DISTORTION,
        "mechanism": "PE zscore vs industry mean — pricing distortion relative to peers",
        "blocker": "requires PE + industry classification",
    },
    {
        "id": "return_expectation_gap",
        "layer": LAYER_CS_DISTORTION,
        "mechanism": "expected_return_model minus realized_return_20d",
        "blocker": "requires return expectation model",
    },
    {
        "id": "sector_rotation_intensity",
        "layer": LAYER_ROTATION,
        "mechanism": "sector_return_dispersion_20d — capital rotation speed",
        "blocker": "requires industry classification",
    },
    {
        "id": "leadership_turnover",
        "layer": LAYER_ROTATION,
        "mechanism": "rank_change(top10 stocks over time)",
        "blocker": "requires universe leadership panel",
    },
]

# Production bundle after correlation pruning
ALPHA_BUNDLE_V1_LIST = [
    "amount_stability_20d",
    "max_daily_return_20d",
    "winner_sentiment_reversal_5d",
    "amihud_shock_reversal_5d",
    "liquidity_amount_residual_20d",
]

FAMILY_ORDER = [
    FAMILY_RETURN,
    FAMILY_LIQUIDITY,
    FAMILY_RISK,
    FAMILY_BEHAVIORAL,
    FAMILY_MICROSTRUCTURE,
    FAMILY_CROSS_SECTION,
    FAMILY_NONLINEAR_COUPLING,
    FAMILY_MULTISCALE,
    FAMILY_TAIL,
]

MECHANISM_LAYER_MAP: Dict[str, str] = {
    # liquidity / flow
    "amount_stability_20d": LAYER_LIQUIDITY_FLOW,
    "liquidity_stability_20d": LAYER_LIQUIDITY_FLOW,
    "liquidity_shock_20d": LAYER_LIQUIDITY_FLOW,
    "liquidity_persistence_20d": LAYER_LIQUIDITY_FLOW,
    "liquidity_acceleration_20d": LAYER_LIQUIDITY_FLOW,
    "amihud_illiquidity_20d": LAYER_LIQUIDITY_FLOW,
    "amihud_amount_orth_20d": LAYER_LIQUIDITY_FLOW,
    "liquidity_amount_residual_20d": LAYER_LIQUIDITY_FLOW,
    "net_inflow_asymmetry_20d": LAYER_LIQUIDITY_FLOW,
    "amount_acceleration_20d": LAYER_LIQUIDITY_FLOW,
    "flow_persistence_decay_20d": LAYER_LIQUIDITY_FLOW,
    "relative_liquidity_strength_20d": LAYER_LIQUIDITY_FLOW,
    "low_vol_liquidity_quality_20d": LAYER_LIQUIDITY_FLOW,
    # reversal / shock
    "amihud_shock_reversal_5d": LAYER_REVERSAL_SHOCK,
    "overreaction_shock_5d": LAYER_REVERSAL_SHOCK,
    "loser_liquidity_reversal_5d": LAYER_REVERSAL_SHOCK,
    "winner_sentiment_reversal_5d": LAYER_REVERSAL_SHOCK,
    "loser_panic_stabilization_20d": LAYER_REVERSAL_SHOCK,
    "intraday_reversal_intensity_20d": LAYER_REVERSAL_SHOCK,
    # momentum
    "trend_consistency_20d": LAYER_MOMENTUM,
    "cn_trend_pv_20d": LAYER_MOMENTUM,
    "underreaction_gap_20d": LAYER_MOMENTUM,
    "volatility_adjusted_momentum_20d": LAYER_MOMENTUM,
    "momentum_rank_dispersion_20d": LAYER_MULTISCALE,
    "momentum_timescale_conflict_20d": LAYER_MULTISCALE,
    "momentum_regime_flip_20d": LAYER_MULTISCALE,
    "momentum_rank_churn_20d": LAYER_ROTATION,
    # microstructure
    "close_location_value_20d": LAYER_MICROSTRUCTURE,
    "price_inefficiency_20d": LAYER_MICROSTRUCTURE,
    "net_volume_pressure_20d": LAYER_MICROSTRUCTURE,
    "range_entropy_20d": LAYER_MICROSTRUCTURE,
    # behavioral
    "volume_price_divergence_20d": LAYER_BEHAVIORAL,
    "winner_crowding_exhaustion_20d": LAYER_BEHAVIORAL,
    # nonlinear liq × vol (HF v3)
    "vol_liquidity_stress_20d": LAYER_NONLINEAR_LIQ_VOL,
    "liquidity_fragility_20d": LAYER_NONLINEAR_LIQ_VOL,
    "vol_liquidity_rank_gap_20d": LAYER_NONLINEAR_LIQ_VOL,
    # multiscale
    "flow_price_rank_gap_20d": LAYER_MULTISCALE,
    "return_skew_shift_20d": LAYER_MULTISCALE,
    # tail
    "max_daily_return_20d": LAYER_TAIL,
    "tail_risk_min_return_20d": LAYER_TAIL,
    "downside_tail_cluster_20d": LAYER_TAIL,
    "upside_fragility_20d": LAYER_TAIL,
    "asymmetric_tail_ratio_20d": LAYER_TAIL,
    # risk (map to nearest layer)
    "volatility_level_20d": LAYER_MOMENTUM,
    "volatility_regime_change_20d": LAYER_MULTISCALE,
    "vol_of_vol_20d": LAYER_TAIL,
    "range_expansion_20d": LAYER_MICROSTRUCTURE,
    "drawup_drawdown_ratio_20d": LAYER_MOMENTUM,
    "return_autocorr_5d": LAYER_MULTISCALE,
}

FACTOR_TAXONOMY: Dict[str, FactorMeta] = {
    "trend_consistency_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Persistent directional moves reflect informed flow",
        "mechanism": "Sign consistency of daily returns over 20d",
        "direction_hint": "positive",
    },
    "drawup_drawdown_ratio_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Trend quality matters more than raw momentum",
        "mechanism": "Cumulative up moves vs max drawdown",
        "direction_hint": "positive",
    },
    "return_autocorr_5d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Short-horizon autocorr reveals momentum vs mean-reversion regime",
        "mechanism": "Rolling corr(ret_t, ret_{t-1})",
        "direction_hint": "regime-dependent",
    },
    "liquidity_stability_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Stable turnover reflects institutional participation",
        "mechanism": "Negative CV of daily amount",
        "direction_hint": "positive",
    },
    "liquidity_shock_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Abnormal liquidity shocks mean-revert or overshoot",
        "mechanism": "-amount_shock * short_return",
        "direction_hint": "negative shock interaction",
    },
    "liquidity_persistence_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Persistent liquidity states carry information",
        "mechanism": "Autocorr of amount series",
        "direction_hint": "positive",
    },
    "liquidity_acceleration_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Marginal liquidity acceleration signals regime shift",
        "mechanism": "Short vs long amount ratio acceleration",
        "direction_hint": "regime-dependent",
    },
    "volatility_level_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Low vol stocks earn risk premium in A-shares",
        "mechanism": "Negative realized vol 20d",
        "direction_hint": "negative vol",
    },
    "volatility_regime_change_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Vol regime shifts precede price adjustment",
        "mechanism": "vol_5d / vol_20d - 1",
        "direction_hint": "regime-dependent",
    },
    "vol_of_vol_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Uncertainty of uncertainty is priced",
        "mechanism": "Std of rolling 20d vol",
        "direction_hint": "negative",
    },
    "range_expansion_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Range expansion signals emotional trading",
        "mechanism": "High-low range vs its mean",
        "direction_hint": "negative",
    },
    "overreaction_shock_5d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Short-term overreaction under volume spikes",
        "mechanism": "-ret_5d * volume_shock",
        "direction_hint": "negative",
    },
    "underreaction_gap_20d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Medium-term drift after slow adjustment",
        "mechanism": "ret_20d - ret_5d",
        "direction_hint": "positive",
    },
    "volume_price_divergence_20d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Price-volume mismatch signals weak trends",
        "mechanism": "-ret_20d * volume_change",
        "direction_hint": "divergence",
    },
    "close_location_value_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "hypothesis": "Close position in range reflects intraday pressure",
        "mechanism": "Mean (close-low)/(high-low)",
        "direction_hint": "positive",
    },
    "price_inefficiency_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "hypothesis": "Inefficient price paths indicate friction",
        "mechanism": "abs(close-open)/(high-low)",
        "direction_hint": "negative",
    },
    "net_volume_pressure_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "hypothesis": "Signed volume pressure proxies order flow",
        "mechanism": "Rolling mean (sign(close-open)*volume)",
        "direction_hint": "positive",
    },
    "amihud_illiquidity_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Price impact / illiquidity is distinct from amount stability",
        "mechanism": "Negative mean(|ret|/amount) over 20d",
        "direction_hint": "negative illiquidity",
    },
    "amihud_shock_reversal_5d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Illiquidity shock on recent return mean-reverts",
        "mechanism": "-(amihud/amihud_ma20) * ret_5d",
        "direction_hint": "negative",
    },
    "max_daily_return_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Lottery / max daily return stocks underperform",
        "mechanism": "Negative rolling max of daily return",
        "direction_hint": "negative tail",
    },
    "skew_20d": {
        "family": FAMILY_TAIL,
        "hypothesis": "Positive return skewness (lottery shape) is overpriced",
        "mechanism": "Alpha=-rolling_skew(ret_1d,20); canonical core/factors/skew",
        "direction_hint": "negative raw skew / positive alpha",
    },
    "skew_60d": {
        "family": FAMILY_TAIL,
        "hypothesis": "Medium-horizon skewness anomaly",
        "mechanism": "Alpha=-rolling_skew(ret_1d,60)",
        "direction_hint": "negative raw skew / positive alpha",
    },
    "skew_120d": {
        "family": FAMILY_TAIL,
        "hypothesis": "Long-horizon skewness anomaly",
        "mechanism": "Alpha=-rolling_skew(ret_1d,120)",
        "direction_hint": "negative raw skew / positive alpha",
    },
    "cn_trend_pv_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "China retail trend needs price and volume jointly",
        "mechanism": "ret_20d * (vol_5d / vol_60d)",
        "direction_hint": "positive",
    },
    "loser_liquidity_reversal_5d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Losers revert on liquidity shock (liquidity provision)",
        "mechanism": "-ret_5d * vol_shock, active when ret_5d < 0",
        "direction_hint": "negative",
    },
    "winner_sentiment_reversal_5d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Winners revert on sentiment / volume spike",
        "mechanism": "-ret_5d * vol_shock, active when ret_5d > 0",
        "direction_hint": "negative",
    },
    "amihud_amount_orth_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Illiquidity impact orthogonal to amount stability",
        "mechanism": "CS residual: amihud_stability ~ amount_stability + log_amount",
        "direction_hint": "residual",
    },
    "net_inflow_asymmetry_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Active fund flow direction, not just size",
        "mechanism": "Rolling mean (up_amount - down_amount) / total_amount",
        "direction_hint": "positive inflow",
    },
    "amount_acceleration_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Second-order capital flow acceleration",
        "mechanism": "amt_5d/amt_20d - amt_20d/amt_60d",
        "direction_hint": "regime-dependent",
    },
    "flow_persistence_decay_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Flow persistence strengthening vs fading",
        "mechanism": "Delta of amount autocorrelation",
        "direction_hint": "positive persistence",
    },
    "intraday_reversal_intensity_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "hypothesis": "Close pushed back from intraday midpoint",
        "mechanism": "Mean (high+low)/2 - close",
        "direction_hint": "positive reversal",
    },
    "range_entropy_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Stability of range structure, not range level",
        "mechanism": "CV of daily high-low range",
        "direction_hint": "negative entropy",
    },
    "attention_shock_cs_5d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Cross-sectional attention spike overreaction",
        "mechanism": "-CS volume zscore * ret_5d",
        "direction_hint": "negative",
    },
    "winner_crowding_exhaustion_20d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Crowded winners exhaust after volume surge",
        "mechanism": "-ret_20d * volume_shock",
        "direction_hint": "negative",
    },
    "loser_panic_stabilization_20d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Decline without volume = non-panic stabilization",
        "mechanism": "(-ret_20d) * low_volume_state",
        "direction_hint": "positive",
    },
    "relative_liquidity_strength_20d": {
        "family": FAMILY_CROSS_SECTION,
        "hypothesis": "Relative market capital preference",
        "mechanism": "amount_mean_20d / cross-sectional mean",
        "direction_hint": "positive",
    },
    "momentum_rank_dispersion_20d": {
        "family": FAMILY_CROSS_SECTION,
        "hypothesis": "Momentum diffusion vs convergence",
        "mechanism": "rank(ret_20d) - rank(ret_5d)",
        "direction_hint": "dispersion",
    },
    "low_vol_liquidity_quality_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Low vol plus stable liquidity compound quality",
        "mechanism": "-vol_20d * amount_stability",
        "direction_hint": "positive quality",
    },
    "tail_risk_min_return_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Extreme downside tail risk premium",
        "mechanism": "Rolling min of daily return",
        "direction_hint": "negative tail",
    },
    "volatility_adjusted_momentum_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Residual-style momentum scaled by risk",
        "mechanism": "ret_20d / vol_20d",
        "direction_hint": "positive",
    },
    "amount_stability_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Stable turnover reflects institutional participation",
        "mechanism": "Negative CV of daily amount",
        "direction_hint": "positive",
    },
    "liquidity_amount_residual_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Amount stability orthogonal to volume and size",
        "mechanism": "CS residual amount_stab ~ volume_stab + log_mktcap",
        "direction_hint": "residual",
    },
    # --- HF v3: mechanism coverage completion ---
    "vol_liquidity_stress_20d": {
        "family": FAMILY_NONLINEAR_COUPLING,
        "hypothesis": "Liquidity impact amplifies under vol stress regimes",
        "mechanism": "-amihud_mean_20d * vol_20d",
        "direction_hint": "negative stress",
    },
    "liquidity_fragility_20d": {
        "family": FAMILY_NONLINEAR_COUPLING,
        "hypothesis": "Short-horizon liquidity instability signals fragility",
        "mechanism": "std(amihud_5d) / std(amihud_20d)",
        "direction_hint": "negative fragility",
    },
    "vol_liquidity_rank_gap_20d": {
        "family": FAMILY_NONLINEAR_COUPLING,
        "hypothesis": "Mismatch between risk rank and liquidity rank is mispriced",
        "mechanism": "rank(vol_20d) - rank(amount_mean_20d)",
        "direction_hint": "divergence",
    },
    "momentum_timescale_conflict_20d": {
        "family": FAMILY_MULTISCALE,
        "hypothesis": "Short vs long momentum rank conflict signals trend instability",
        "mechanism": "rank(ret_5d) - rank(ret_60d)",
        "direction_hint": "conflict",
    },
    "flow_price_rank_gap_20d": {
        "family": FAMILY_MULTISCALE,
        "hypothesis": "Capital flow not yet reflected in price",
        "mechanism": "rank(amount_mean_5d) - rank(ret_20d)",
        "direction_hint": "divergence",
    },
    "momentum_regime_flip_20d": {
        "family": FAMILY_MULTISCALE,
        "hypothesis": "Momentum regime instability precedes adjustment",
        "mechanism": "delta rolling_corr(ret_5d, ret_20d)",
        "direction_hint": "regime-dependent",
    },
    "downside_tail_cluster_20d": {
        "family": FAMILY_TAIL,
        "hypothesis": "Crash clustering indicates fragile stocks",
        "mechanism": "count(ret < -2*sigma) over 20d",
        "direction_hint": "negative tail",
    },
    "upside_fragility_20d": {
        "family": FAMILY_TAIL,
        "hypothesis": "Unstable upside spikes vs baseline mean",
        "mechanism": "max(ret_5d) - mean(ret_20d)",
        "direction_hint": "negative fragility",
    },
    "asymmetric_tail_ratio_20d": {
        "family": FAMILY_TAIL,
        "hypothesis": "Skewed downside vs upside semivariance is priced",
        "mechanism": "downside_var / upside_var over 20d",
        "direction_hint": "negative skew",
    },
    "momentum_rank_churn_20d": {
        "family": FAMILY_MULTISCALE,
        "hypothesis": "Cross-sectional momentum rank turnover signals crowding decay",
        "mechanism": "rolling mean |delta rank(ret_20d)|",
        "direction_hint": "churn",
    },
    "return_skew_shift_20d": {
        "family": FAMILY_MULTISCALE,
        "hypothesis": "Distribution shape change vs longer baseline",
        "mechanism": "skew(ret_20d window) - skew(ret_60d window)",
        "direction_hint": "shape shift",
    },
    "composite_liquidity_stability_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Multi-dimensional flow stability is priced beyond single CV",
        "mechanism": "rank-mean(-CV_amount, -CV_volume, -CV_range)",
        "direction_hint": "positive stability",
    },
    "amihud_stability_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Stable price impact (Amihud CV) signals quality",
        "mechanism": "-CV(|ret|/amount) over 20d",
        "direction_hint": "positive stability",
    },
    "return_stability_20d": {
        "family": FAMILY_RISK,
        "hypothesis": "Smooth return path (low ret CV) is rewarded",
        "mechanism": "-CV(ret_1d) over 20d",
        "direction_hint": "positive stability",
    },
    "amount_stability_60d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Long-horizon amount stability extends 20d signal",
        "mechanism": "-CV(amount) over 60d",
        "direction_hint": "positive stability",
    },
    "shadow_stability_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "hypothesis": "Stable intraday shadow range indicates orderly trading",
        "mechanism": "-CV(upper_shadow + lower_shadow) over 20d",
        "direction_hint": "positive stability",
    },
    "stability_quality_composite_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Amount stability + low vol + range stability jointly priced",
        "mechanism": "rank-mean(-CV_amount, -vol_20d, -CV_range)",
        "direction_hint": "positive quality",
    },
    "low_vol_stability_rank_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Rank composite of vol×CV quality and range stability",
        "mechanism": "rank-mean(-vol*CV_amount, -CV_range)",
        "direction_hint": "positive quality",
    },
    "stable_reversal_blend_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Mean reversion on liquidity-stable names",
        "mechanism": "rank-mean(-ret_20d, -CV_amount)",
        "direction_hint": "reversal on stable",
    },
    "liquidity_conditioned_momentum_20d": {
        "family": FAMILY_NONLINEAR_COUPLING,
        "hypothesis": "Momentum backed by stable liquidity flow is higher quality",
        "mechanism": "ret_20d × (-amount_cv_20d)",
        "direction_hint": "quality momentum",
    },
    "liquidity_shock_recovery_5d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Liquidity recovery after shock signals smart-money absorption",
        "mechanism": "-ΔAmihud_5d × ret_5d",
        "direction_hint": "recovery",
    },
    "triple_crowding_exhaustion_20d": {
        "family": FAMILY_BEHAVIORAL,
        "hypothesis": "Up + volume spike + vol expansion = crowding exhaustion",
        "mechanism": "-ret_20d × volume_shock × vol_regime_shock",
        "direction_hint": "negative crowding",
    },
    "trend_quality_composite_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Trend quality = momentum × path efficiency × day consistency",
        "mechanism": "ret_20d × mean(|C-O|/(H-L)) × frac(positive days)",
        "direction_hint": "quality trend",
    },
    "liquidity_vol_regime_20d": {
        "family": FAMILY_NONLINEAR_COUPLING,
        "hypothesis": "Abnormal flow relative to vol regime reveals information",
        "mechanism": "liquidity_shock / volatility_shock",
        "direction_hint": "flow-vol regime",
    },
    "tail_adjusted_momentum_60d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Return per unit downside risk (stock-level Sharpe)",
        "mechanism": "ret_60d / downside_vol_60d",
        "direction_hint": "risk-adjusted momentum",
    },
    "flow_price_divergence_20d": {
        "family": FAMILY_CROSS_SECTION,
        "hypothesis": "Flow leads price — information diffusion lag",
        "mechanism": "rank(amount_accel) - rank(ret_20d)",
        "direction_hint": "flow leads price",
    },
    "liquidity_accel_risk_filtered_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Second-order liquidity change filtered by low vol",
        "mechanism": "Δ²log(amount)_20d × (-volatility_20d)",
        "direction_hint": "liquidity curvature",
    },
    "residual_momentum_60d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Momentum orthogonal to short momentum, vol, and size proxy",
        "mechanism": "CS residual: ret_60 ~ ret_20 + vol_20 + log(amount)",
        "direction_hint": "residual momentum",
    },
    "information_ratio_momentum_60d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Long-horizon risk-adjusted return (stock IR)",
        "mechanism": "ret_60d / vol_60d",
        "direction_hint": "IR momentum",
    },
    "information_ratio_momentum_120d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Very long IR momentum — lower regime sensitivity",
        "mechanism": "ret_120d / vol_120d",
        "direction_hint": "IR momentum",
    },
    "residual_liquidity_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Liquidity stability net of size and vol exposure",
        "mechanism": "CS residual: -CV_amount ~ log(amount) + vol_20",
        "direction_hint": "residual liquidity",
    },
    "relative_vol_adjusted_liquidity_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Capital participation per unit risk (CS rank)",
        "mechanism": "rank(amount_mean_20 / vol_20)",
        "direction_hint": "RVL",
    },
    "low_vol_liquidity_quality_60d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Low vol × liquidity stability composite",
        "mechanism": "rank-mean(-vol_60, -CV_amount)",
        "direction_hint": "quality",
    },
    "stability_signal_persistence_20d": {
        "family": FAMILY_LIQUIDITY,
        "hypothesis": "Persistent liquidity stability signal",
        "mechanism": "autocorr(-CV_amount, 20d)",
        "direction_hint": "persistence",
    },
    "residual_reversal_20d": {
        "family": FAMILY_RETURN,
        "hypothesis": "Short reversal net of long momentum and vol",
        "mechanism": "CS residual: -ret_20 ~ ret_60 + vol_20",
        "direction_hint": "residual reversal",
    },
}


def mechanism_layer_for(factor_name: str) -> str:
    if factor_name in MECHANISM_LAYER_MAP:
        return MECHANISM_LAYER_MAP[factor_name]
    meta = FACTOR_TAXONOMY.get(factor_name, {})
    family = meta.get("family", "unknown")
    _family_to_layer = {
        FAMILY_LIQUIDITY: LAYER_LIQUIDITY_FLOW,
        FAMILY_RETURN: LAYER_MOMENTUM,
        FAMILY_RISK: LAYER_TAIL,
        FAMILY_BEHAVIORAL: LAYER_BEHAVIORAL,
        FAMILY_MICROSTRUCTURE: LAYER_MICROSTRUCTURE,
        FAMILY_CROSS_SECTION: LAYER_CS_DISTORTION,
        FAMILY_NONLINEAR_COUPLING: LAYER_NONLINEAR_LIQ_VOL,
        FAMILY_MULTISCALE: LAYER_MULTISCALE,
        FAMILY_TAIL: LAYER_TAIL,
    }
    return _family_to_layer.get(family, "unknown")


def factors_by_family(family: str) -> List[str]:
    return [n for n, m in FACTOR_TAXONOMY.items() if m["family"] == family]
