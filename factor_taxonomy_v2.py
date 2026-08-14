"""China A-share Alpha Library v2 — research triage map (NOT a factor zoo).

HF workflow per candidate:
  Factor family → Representative → Residual vs frozen stack → Incremental IC → Keep/Drop

Three candidate types (triage labels):
  alpha_dimension   — new return-predictive information
  risk_exposure     — beta / size / vol; neutralize, don't trade raw
  data_gated        — hypothesis valid only with extra data (L2, Wind, margin)
"""

from typing import Dict, List, TypedDict

OHLCV_FROZEN_DIMENSIONS = [
    "low_vol_liquidity_quality_60d",
    "volatility_60d",
    "lower_shadow_support_20d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]


class FactorTriage(TypedDict):
    tier: str
    role: str
    batch: str
    notes: str


CN_BROKER_V1_TRIAGE: Dict[str, FactorTriage] = {
    "cn_limit_up_strength_20d": {
        "tier": "candidate_pending",
        "role": "alpha_dimension",
        "batch": "cn_structure",
        "notes": "strict_pass; cn_only cluster — pending mono/universe gates for stack v1.",
    },
    "cn_volume_surge_moment_20d": {
        "tier": "candidate_pending",
        "role": "alpha_dimension",
        "batch": "cn_liquidity",
        "notes": "strict_pass; incremental vs OHLCV — pending mono/universe gates.",
    },
    "cn_turnover_concentration_20d": {
        "tier": "drop",
        "role": "measurement_refinement",
        "batch": "cn_structure",
        "notes": "amplification_artifact after strict triage.",
    },
    "cn_attention_shock_5d": {
        "tier": "drop",
        "role": "measurement_refinement",
        "batch": "cn_behavior",
        "notes": "sign_flip_after_neutral; failed strict triage.",
    },
    "cn_herding_proxy_20d": {
        "tier": "refine",
        "role": "alpha_dimension",
        "batch": "cn_behavior",
        "notes": "Mixed with vol cluster; L2 needed for clean herding dimension.",
    },
    "cn_shadow_combo_20d": {
        "tier": "refine",
        "role": "measurement_refinement",
        "batch": "cn_technical",
        "notes": "Overlaps lower_shadow — one shadow rep only.",
    },
    "cn_new_high_breakout_252d": {
        "tier": "refine",
        "role": "measurement_refinement",
        "batch": "cn_structure",
        "notes": "Mixed with attention cluster; refine vs 52w high family.",
    },
    "cn_turnover_percentile_20d": {
        "tier": "refine",
        "role": "risk_exposure",
        "batch": "cn_structure",
        "notes": "Mixed with high_low; likely size/liquidity proxy.",
    },
    "cn_turnover_change_rate_20d": {
        "tier": "refine",
        "role": "measurement_refinement",
        "batch": "cn_structure",
        "notes": "Acceleration > level; still OHLCV-mixed.",
    },
    "cn_price_volume_divergence_20d": {
        "tier": "refine",
        "role": "measurement_refinement",
        "batch": "cn_technical",
        "notes": "Mixed with vol cluster.",
    },
    "cn_chase_behavior_20d": {
        "tier": "refine",
        "role": "alpha_dimension",
        "batch": "cn_behavior",
        "notes": "No bundle uplift; EOD proxy weak vs L2 chase.",
    },
    "cn_rsi_momentum_gap_20d": {
        "tier": "drop",
        "role": "measurement_refinement",
        "batch": "cn_technical",
        "notes": "Weak residual; TA crowded on A-share.",
    },
    "cn_amount_distribution_skew_20d": {
        "tier": "drop",
        "role": "risk_exposure",
        "batch": "cn_liquidity",
        "notes": "Weak IC; mixed with amount level.",
    },
}

ALPHA_LIBRARY_V2_BATCHES: Dict[str, List[dict]] = {
    "batch1_fundamental_core": [
        {"id": "ep_ttm", "pillar": "value", "status": "superseded_by_value_ep"},
        {"id": "bp", "pillar": "value", "status": "superseded_by_value_bp"},
        {"id": "ep_ttm_ind_neutral", "pillar": "value", "status": "superseded_by_value_ep"},
        {"id": "bp_ind_neutral", "pillar": "value", "status": "superseded_by_value_bp"},
        {"id": "value_ep", "pillar": "value", "status": "d6_brick"},
        {"id": "value_bp", "pillar": "value", "status": "d6_brick"},
        {"id": "value_cfp", "pillar": "value", "status": "d6_brick"},
        {"id": "value_composite", "pillar": "value", "status": "d6_representative"},
        {"id": "roe_stability", "pillar": "quality", "status": "composite_input"},
        {"id": "gross_profitability", "pillar": "quality", "status": "composite_input"},
        {"id": "cfo_quality", "pillar": "quality", "status": "composite_input"},
        {"id": "quality_composite", "pillar": "quality", "status": "d7_representative"},
        {"id": "cfp", "pillar": "value", "status": "planned"},
    ],
    "batch2_cn_structure": [
        {"id": "cn_limit_up_strength_20d", "status": "implemented"},
        {"id": "cn_turnover_acceleration", "status": "refine"},
        {"id": "cn_turnover_surprise", "status": "planned"},
        {"id": "cn_new_high_252d", "status": "refine"},
        {"id": "margin_change_20d", "status": "data_gated"},
    ],
    "batch3_behavior_refinement": [
        {"id": "rev_20d", "status": "planned"},
        {"id": "residual_momentum_60d", "status": "implemented"},
        {"id": "idvol_20d", "status": "planned"},
        {"id": "maxret_20d", "status": "refine"},
    ],
    "batch4_l2_microstructure_v1": [
        {"id": "cn_voi_20d", "status": "closed_no_independent_dim"},
        {"id": "cn_oir_20d", "status": "closed_no_independent_dim"},
        {"id": "cn_mpb_20d", "status": "closed_possible_enhancer"},
    ],
    "batch4_l2_microstructure_v2": [
        {"id": "cn_voi_shock", "status": "implemented", "group": "flow_shock"},
        {"id": "cn_mpb_shock", "status": "implemented", "group": "flow_shock"},
        {"id": "cn_flow_persistence", "status": "implemented", "group": "persistence"},
        {"id": "cn_imbalance_duration", "status": "implemented", "group": "persistence"},
        {"id": "cn_liquidity_consumption", "status": "implemented", "group": "consumption"},
        {"id": "cn_cancel_shock", "status": "implemented", "group": "consumption"},
        {"id": "cn_depth_oir", "status": "data_gated_ssl2"},
        {"id": "cn_limit_up_queue_pressure", "status": "data_gated"},
    ],
}

# Published stack v1 — see alpha_frozen_stack_v1.py / research/results/alpha_frozen_stack_v1.*
TARGET_ALPHA_STACK_V2 = [
    {"dim": "D1", "rep": "low_vol_liquidity_quality_60d", "source": "ohlcv_frozen", "status": "frozen"},
    {"dim": "D2", "rep": "volatility_60d", "source": "ohlcv_frozen", "status": "frozen"},
    {"dim": "D3", "rep": "lower_shadow_support_20d", "source": "ohlcv_frozen", "status": "frozen"},
    {"dim": "D4", "rep": "winner_sentiment_reversal_5d", "source": "ohlcv_frozen", "status": "frozen"},
    {"dim": "D5", "rep": "upside_fragility_20d", "source": "ohlcv_frozen", "status": "frozen"},
    {"dim": "D6", "rep": "value_composite", "source": "fundamental_value", "status": "frozen_enhancer"},
    {"dim": "D7", "rep": "quality_composite", "source": "fundamental_quality", "status": "frozen_enhancer"},
    {"dim": "CN1", "rep": "cn_limit_up_strength_20d", "source": "cn_structure", "status": "candidate_pending"},
    {"dim": "CN2", "rep": "cn_volume_surge_moment_20d", "source": "cn_liquidity", "status": "candidate_pending"},
    {"dim": "L2", "rep": "cn_voi_20d", "source": "l2", "status": "candidate_v1"},
    {"dim": "L2b", "rep": "cn_oir_20d", "source": "l2", "status": "candidate_v1"},
    {"dim": "L2c", "rep": "cn_mpb_20d", "source": "l2", "status": "candidate_v1"},
]
