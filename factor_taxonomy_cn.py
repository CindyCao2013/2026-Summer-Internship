"""China A-share alpha research map — broker-validated mechanisms over generic OHLCV mining.

Methodology:
    A-share market mechanism → broker 金工 factor system → economic hypothesis
    → formula reproduction → robust validation (robust_alpha_engine)

Not: WorldQuant Alpha101 + random OHLCV interaction search.
"""

from typing import Dict, List, TypedDict

from factor_taxonomy import (
    FAMILY_BEHAVIORAL,
    FAMILY_LIQUIDITY,
    FAMILY_MICROSTRUCTURE,
    FAMILY_RETURN,
    FAMILY_RISK,
)


class CNFactorMeta(TypedDict):
    family: str
    cn_family: str
    hypothesis: str
    mechanism: str
    direction_hint: str
    source: str  # broker / paper reference
    data_layer: str  # eod | intraday | l2


# --- China A-share Alpha Space (6 pillars) ---
CN_FAMILY_PRICE_ACTION = "price_action"
CN_FAMILY_LIQUIDITY = "liquidity"
CN_FAMILY_TRADING_BEHAVIOR = "trading_behavior"
CN_FAMILY_MICROSTRUCTURE = "microstructure"
CN_FAMILY_MARKET_STRUCTURE = "market_structure"
CN_FAMILY_FUNDAMENTAL = "fundamental"

CN_ALPHA_PILLARS: Dict[str, List[str]] = {
    CN_FAMILY_PRICE_ACTION: [
        "momentum",
        "reversal",
        "candle_shadow",
        "breakout",
    ],
    CN_FAMILY_LIQUIDITY: [
        "turnover_structure",
        "amount_distribution",
        "amihud",
        "liquidity_shock",
    ],
    CN_FAMILY_TRADING_BEHAVIOR: [
        "chase",
        "herding",
        "attention",
        "sentiment",
    ],
    CN_FAMILY_MICROSTRUCTURE: [
        "order_imbalance",  # L2 phase
        "vwap_deviation",
        "price_impact",
    ],
    CN_FAMILY_MARKET_STRUCTURE: [
        "limit_up",
        "turnover_concentration",
        "volatility_regime",
    ],
    CN_FAMILY_FUNDAMENTAL: [
        "value",
        "quality",
        "growth",
    ],
}

# Priority 1: EOD-reproducible broker classics (Phase 1 — no L2 required)
EOD_CN_BROKER_V1_LIST = [
    # 量价
    "cn_turnover_percentile_20d",
    "cn_turnover_change_rate_20d",
    "cn_volume_surge_moment_20d",
    "cn_amount_distribution_skew_20d",
    "cn_price_volume_divergence_20d",
    # 行为
    "cn_chase_behavior_20d",
    "cn_herding_proxy_20d",
    "cn_attention_shock_5d",
    # 技术
    "cn_new_high_breakout_252d",
    "cn_rsi_momentum_gap_20d",
    "cn_shadow_combo_20d",
]

# Phase 2: A-share market structure (EOD proxy until limit-up database)
EOD_CN_MARKET_STRUCTURE_LIST = [
    "cn_limit_up_strength_20d",
    "cn_turnover_concentration_20d",
]

# Phase 3: L2 v1 — level bricks (CLOSED: no independent dimension)
L2_MICROSTRUCTURE_V1_LIST = [
    "cn_voi_20d",
    "cn_oir_20d",
    "cn_mpb_20d",
]

# Phase 3b: L2 v2 — event-driven (6 factors, see factor_formulas_l2_v2.py)
L2_MICROSTRUCTURE_V2_LIST = [
    "cn_voi_shock",
    "cn_mpb_shock",
    "cn_flow_persistence",
    "cn_imbalance_duration",
    "cn_liquidity_consumption",
    "cn_cancel_shock",
]

CN_L2_PRIORITY_LIST = L2_MICROSTRUCTURE_V2_LIST + [
    "cn_order_imbalance_20d",  # v2.1: SSL2 depth OIR
]

EOD_CN_BROKER_ALL_LIST = EOD_CN_BROKER_V1_LIST + EOD_CN_MARKET_STRUCTURE_LIST

CN_FACTOR_TAXONOMY: Dict[str, CNFactorMeta] = {
    "cn_turnover_percentile_20d": {
        "family": FAMILY_LIQUIDITY,
        "cn_family": CN_FAMILY_LIQUIDITY,
        "hypothesis": "Abnormal turnover vs own history signals attention/crowding",
        "mechanism": "rolling percentile of turnover intensity (60d window)",
        "direction_hint": "context-dependent",
        "source": "中信/华泰 换手率结构",
        "data_layer": "eod",
    },
    "cn_turnover_change_rate_20d": {
        "family": FAMILY_LIQUIDITY,
        "cn_family": CN_FAMILY_LIQUIDITY,
        "hypothesis": "Turnover acceleration captures liquidity regime shift",
        "mechanism": "(turnover_5d - turnover_20d) / turnover_20d",
        "direction_hint": "acceleration",
        "source": "A股金工 换手率变化率",
        "data_layer": "eod",
    },
    "cn_volume_surge_moment_20d": {
        "family": FAMILY_LIQUIDITY,
        "cn_family": CN_FAMILY_LIQUIDITY,
        "hypothesis": "Volume surge moments contain incremental alpha (方正金工)",
        "mechanism": "mean(max(volume/vol_mean - 1, 0)) over 20d",
        "direction_hint": "surge",
        "source": "方正金工 成交量激增时刻",
        "data_layer": "eod",
    },
    "cn_amount_distribution_skew_20d": {
        "family": FAMILY_LIQUIDITY,
        "cn_family": CN_FAMILY_LIQUIDITY,
        "hypothesis": "Skewed amount distribution reflects asymmetric participation",
        "mechanism": "skew(daily amount) over 20d",
        "direction_hint": "distribution shape",
        "source": "方正金工 成交量分布",
        "data_layer": "eod",
    },
    "cn_price_volume_divergence_20d": {
        "family": FAMILY_RETURN,
        "cn_family": CN_FAMILY_PRICE_ACTION,
        "hypothesis": "Price up without volume confirmation → weak trend (A股量价背离)",
        "mechanism": "-ret_20d × volume_change_5d/20d",
        "direction_hint": "divergence",
        "source": "国内量价因子体系",
        "data_layer": "eod",
    },
    "cn_chase_behavior_20d": {
        "family": FAMILY_BEHAVIORAL,
        "cn_family": CN_FAMILY_TRADING_BEHAVIOR,
        "hypothesis": "Retail chase: intraday return correlates with volume (开源金工)",
        "mechanism": "rolling_corr(intraday_return, volume, 20d)",
        "direction_hint": "chase",
        "source": "开源金工 高频追涨杀跌 EOD proxy",
        "data_layer": "eod",
    },
    "cn_herding_proxy_20d": {
        "family": FAMILY_BEHAVIORAL,
        "cn_family": CN_FAMILY_TRADING_BEHAVIOR,
        "hypothesis": "Sync trading with market on volume spikes (国盛羊群 EOD proxy)",
        "mechanism": "mean((ret - mkt_ret) × volume_shock, 20d)",
        "direction_hint": "herding",
        "source": "国盛金工 羊群效应 EOD proxy",
        "data_layer": "eod",
    },
    "cn_attention_shock_5d": {
        "family": FAMILY_BEHAVIORAL,
        "cn_family": CN_FAMILY_TRADING_BEHAVIOR,
        "hypothesis": "Attention spike × short return — retail overreaction",
        "mechanism": "-volume_zscore × ret_5d",
        "direction_hint": "attention reversal",
        "source": "A股注意力/情绪交易",
        "data_layer": "eod",
    },
    "cn_new_high_breakout_252d": {
        "family": FAMILY_RETURN,
        "cn_family": CN_FAMILY_PRICE_ACTION,
        "hypothesis": "Price near 52-week high — breakout momentum (A股创新高因子)",
        "mechanism": "close / rolling_max(high, 252) - 1",
        "direction_hint": "breakout",
        "source": "国内技术因子体系",
        "data_layer": "eod",
    },
    "cn_rsi_momentum_gap_20d": {
        "family": FAMILY_RETURN,
        "cn_family": CN_FAMILY_PRICE_ACTION,
        "hypothesis": "RSI vs price momentum disagreement — overbought/oversold gap",
        "mechanism": "rank(RSI_14) - rank(ret_20d) cross-sectionally",
        "direction_hint": "rsi divergence",
        "source": "RSI改进 / 国内技术因子",
        "data_layer": "eod",
    },
    "cn_shadow_combo_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_PRICE_ACTION,
        "hypothesis": "Lower shadow support minus upper shadow pressure (东吴金工)",
        "mechanism": "mean(lower_shadow - upper_shadow, 20d)",
        "direction_hint": "shadow asymmetry",
        "source": "东吴金工 蜡烛图上下影线",
        "data_layer": "eod",
    },
    "cn_limit_up_strength_20d": {
        "family": FAMILY_BEHAVIORAL,
        "cn_family": CN_FAMILY_MARKET_STRUCTURE,
        "hypothesis": "Limit-up frequency captures A-share sentiment / 游资",
        "mechanism": "count(ret > 9.5%) / 20d",
        "direction_hint": "limit-up",
        "source": "A股涨停行为",
        "data_layer": "eod",
    },
    "cn_turnover_concentration_20d": {
        "family": FAMILY_LIQUIDITY,
        "cn_family": CN_FAMILY_MARKET_STRUCTURE,
        "hypothesis": "Concentrated high-turnover days — liquidity event clustering",
        "mechanism": "fraction of days with turnover > 90th pct of own 60d history",
        "direction_hint": "concentration",
        "source": "A股换手率结构",
        "data_layer": "eod",
    },
    "cn_voi_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Volume order imbalance — active buy vs sell pressure (VOI)",
        "mechanism": "20d mean of daily (active_buy_vol - active_sell_vol) / total active",
        "direction_hint": "buy_pressure",
        "source": "中信建投 高频量价 VOI",
        "data_layer": "l2",
    },
    "cn_oir_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Order book imbalance ratio — bid vs ask queue pressure (OIR)",
        "mechanism": "20d mean of daily (bid_cancel - ask_cancel) / total cancel",
        "direction_hint": "bid_pressure",
        "source": "中信建投 高频量价 OIR",
        "data_layer": "l2",
    },
    "cn_mpb_20d": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Mid-price basis / trade pressure — signed amount imbalance (MPB)",
        "mechanism": "20d mean of daily (active_buy_amt - active_sell_amt) / total active amt",
        "direction_hint": "trade_pressure",
        "source": "中信建投 高频量价 MPB",
        "data_layer": "l2",
    },
    # --- L2 v1 CLOSED: level bricks — see l2_v1_triage.csv ---
    # --- L2 v2: event-driven ---
    "cn_voi_shock": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Abnormal active-flow imbalance — VOI event shock",
        "mechanism": "zscore(VOI_t, 60d)",
        "direction_hint": "flow_shock",
        "source": "L2 v2 Group A",
        "data_layer": "l2",
    },
    "cn_mpb_shock": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Abnormal signed amount pressure — MPB event shock",
        "mechanism": "zscore(MPB_t, 60d)",
        "direction_hint": "trade_shock",
        "source": "L2 v2 Group A",
        "data_layer": "l2",
    },
    "cn_flow_persistence": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Sustained vs one-off buy pressure",
        "mechanism": "corr(VOI_t, mean(VOI_{t-5:t-1}), 20d)",
        "direction_hint": "persistence",
        "source": "L2 v2 Group B",
        "data_layer": "l2",
    },
    "cn_imbalance_duration": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Buy-pressure duration — fraction of minutes with VOI > threshold",
        "mechanism": "sum(minute VOI > 0.1) / n_minutes per day",
        "direction_hint": "duration",
        "source": "L2 v2 Group B",
        "data_layer": "l2",
    },
    "cn_liquidity_consumption": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Liquidity consumption ratio — volume vs active-flow depth proxy",
        "mechanism": "volume / (active_buy_vol + active_sell_vol); SSL2 depth → v2.5",
        "direction_hint": "consumption",
        "source": "L2 v2 Group C",
        "data_layer": "l2",
    },
    "cn_cancel_shock": {
        "family": FAMILY_MICROSTRUCTURE,
        "cn_family": CN_FAMILY_MICROSTRUCTURE,
        "hypothesis": "Cancellation asymmetry shock event",
        "mechanism": "zscore((bid_cancel - ask_cancel) / total_cancel, 60d)",
        "direction_hint": "cancel_shock",
        "source": "L2 v2 Group C",
        "data_layer": "l2",
    },
}
