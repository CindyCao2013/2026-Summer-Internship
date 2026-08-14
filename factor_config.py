"""因子研究统一配置：三条 track 共用同一 runner 框架。

Track:
  - eod_pv         日频价量 / 技术（当前主力）
  - eod_engine     HF 四大家族 structured alpha (Core 8)
  - eod_engine_priority_a  Priority A 新 alpha (7)
  - eod_latent     EOD-proxy bricks → PCA latent → rank alpha
  - fundamental    财务 / 估值 / 市值（DERIVATIVE + 财报，分阶段）
  - intraday       分钟级高频低频化（intraday_lib 回测）

改 TRACK 或 factor_list 即可切换运行目标。
"""

import datetime as dt
from pathlib import Path
from typing import List, Optional

from factor_formulas import (
    ALL_FACTOR_LIST,
    CLASSIC_FACTOR_LIST,
    NEW_EOD_FACTOR_LIST,
    PRIORITY_NEW_FACTORS,
)
from factor_formulas_fundamental import (
    FUNDAMENTAL_BATCH1_LIST,
    FUNDAMENTAL_FACTOR_LIST,
    FUNDAMENTAL_PHASE1_LIST,
    FUNDAMENTAL_PHASE2_BATCH_LIST,
    FUNDAMENTAL_QUALITY_D7_BATCH_LIST,
)
from factor_formulas_value import FUNDAMENTAL_VALUE_D6_BATCH_LIST
from factor_taxonomy import (
    ALPHA_BUNDLE_V1_LIST,
    EOD_ENGINE_ALL_LIST,
    EOD_ENGINE_CORE_LIST,
    EOD_ENGINE_HF_V2_LIST,
    EOD_ENGINE_HF_V3_LIST,
    EOD_ENGINE_HF_V4_LIST,
    EOD_ENGINE_HF_V5_LIST,
    EOD_ENGINE_ROBUST_LIST,
    EOD_ENGINE_PRIORITY_A_LIST,
)
from factor_taxonomy_cn import EOD_CN_BROKER_ALL_LIST, EOD_CN_BROKER_V1_LIST, L2_MICROSTRUCTURE_V1_LIST
from factor_formulas_l2_v2 import L2_MICROSTRUCTURE_V2_LIST
from factor_formulas_smart_money_active_v2 import SMART_MONEY_ACTIVE_V2_LIST
from factor_formulas_apm_active_v2 import APM_ACTIVE_V2_LIST
from factor_formulas_ideal_reversal_active_v2 import IDEAL_REVERSAL_ACTIVE_V2_LIST
from factor_formulas_ideal_amplitude_active_v2 import IDEAL_AMPLITUDE_V2_LIST
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_ALL_LIST,
    LIQUIDITY_NORM_CORE_LIST,
)
from intraday_formulas import (
    INTRADAY_FACTOR_LIST,
    INTRADAY_PHASE1_LIST,
    INTRADAY_PHASE2_LIST,
)

# =========================
# 选择运行 track
# =========================
# eod_pv        原有 32 个价量因子
# eod_engine    HF 四大家族 structured alpha (Core 8 / Extended)
# eod_latent    L2-proxy bricks → PCA latent → cross-sectional rank
# eod_engine_priority_a  Priority A 文献新 alpha (7)
# eod_engine_hf_v2     HF v2 新维度 (flow / CS relative / behavioral)
# eod_engine_hf_v3     HF v3 机制补全 (nonlinear liq×vol / multiscale / tail)
# eod_engine_robust    Robust alpha (residual / IR / CS-normalized)
# eod_cn_broker_v1     China A-share broker-classic reproduction (Priority 1, 11 factors)
# eod_cn_broker_all    CN Broker v1 full (13 factors incl. limit-up / turnover concentration)
# eod_liquidity_norm  size-adjusted liquidity + orthogonal decomposition
# alpha_bundle_v1     correlation-pruned 5-factor bundle (mixed registries)
# fundamental   估值 / 市值 Phase 1
# l2_microstructure_v1  L2 tick/order-book → daily VOI/OIR/MPB (3 factors)
# smart_money_active_v2  Active_* 大单集中度（聪明钱2.0）
# apm_active_v2          Active Pressure Metric（成交额加权主动买卖压力 + EWM5）
# ideal_reversal_active_v2  主动大单集中度 × 短窗反转
# ideal_amplitude_active_v2 实现振幅 / 主动净额波动（振幅质量）
TRACK = "ideal_amplitude_active_v2"

# =========================
# 回测区间
# =========================
START_DAY = dt.datetime(2021, 1, 1)
END_DAY = dt.datetime(2024, 6, 30)
PREHEAT_CALENDAR_DAYS = 400

# =========================
# 分钟数据层（MinuteBarStore — 纯 DDB 按需查询）
# =========================
# 防止误拉全历史；也可用环境变量 MINUTE_BAR_HISTORY_START 覆盖
MINUTE_BAR_HISTORY_START = dt.datetime(2020, 1, 1)
DDB_QUERY_TIMEOUT = 120  # 秒
DDB_MAX_RETRIES = 3

# =========================
# 日内择时热力图（P1）
# =========================
INTRADAY_HEATMAP_FACTORS = [
    "TGD20",
    "SmartMoneyActiveV2",
    "APM_ActiveV2",
    "IdealAmplitude_ActiveV2",
    "IdealReversal_ActiveV2",
]
INTRADAY_HEATMAP_BARTIMES = ["09:59", "10:29", "11:29", "13:29", "14:29"]
INTRADAY_HEATMAP_HORIZONS = [
    "Ret_15",
    "Ret_30",
    "Ret_60",
    "Ret_90",
    "Ret_120",
    "Ret_150",
    "Ret_180",
    "Ret_EOD",
]

# =========================
# 真分钟 Alpha Phase 2
# =========================
INTRADAY_ALPHA_STORE_START = dt.datetime(2020, 1, 1)

# =========================
# 批量 / 断点续跑
# =========================
# True：无头 Agg + 保存 summary/cum_pnl/rank_ic；False 才弹交互图
BATCH_MODE = True
SKIP_COMPLETED = True
RESUME_FROM_EXISTING = True
SAVE_RESULTS = True
METHOD = "c2c"
SHOW_GROUP_TEST_PLOTS = not BATCH_MODE

# batch_mode=False 时只跑这一个因子
SINGLE_FACTOR_NAME = "IdealAmplitude_ActiveV2"

CUSTOM_FACTOR_LIST: Optional[List[str]] = [
    "IdealAmplitude_ActiveV2",
]

# 仅 Intraday_Factor_Test_Process 使用（不影响日频 TRACK）
INTRADAY_CUSTOM_FACTOR_LIST: Optional[List[str]] = [
    "close_vwap_deviation",
    "active_buy_sell_imbalance",
    "late_session_strength",
    "volume_front_loading",
    "morning_reversal_pressure",
    "volume_back_loading",
]

# 真分钟 Alpha 评估窗口（None = 用 START_DAY/END_DAY）。
# 分钟数据由 MinuteBarStore 直接从 DolphinDB 按需查询。
INTRADAY_MINUTE_EVAL_START: Optional[dt.datetime] = None
INTRADAY_MINUTE_EVAL_END: Optional[dt.datetime] = None

# close_vwap_deviation 已通过 Sprint 3.1 production parity；保留 False 作为 Python fallback。
INTRADAY_CLOSE_VWAP_USE_DDB: bool = True

# volume_front_loading 已通过 Sprint 3.2 production parity；保留 False 作为 Python fallback。
INTRADAY_VOLUME_FRONT_USE_DDB: bool = True

# volume_back_loading 已通过 Sprint 3.3 production parity；保留 False 作为 Python fallback。
INTRADAY_VOLUME_BACK_USE_DDB: bool = True

# late_session_strength 已通过 Sprint 3.4 production parity；保留 False 作为 Python fallback。
INTRADAY_LATE_SESSION_STRENGTH_USE_DDB: bool = True

# active_buy_sell_imbalance 已通过 Sprint 3.5 production parity；保留 False 作为 Python fallback。
INTRADAY_ACTIVE_BUY_SELL_IMBALANCE_USE_DDB: bool = True

# Phase 4.1 discovery candidates. Python remains the explicit golden fallback.
INTRADAY_BARTIME_OFI_USE_DDB: bool = True
INTRADAY_OFI_PERSISTENCE_USE_DDB: bool = True
INTRADAY_ACTIVE_BUY_SHOCK_USE_DDB: bool = True
INTRADAY_AVERAGE_ACTIVE_TRADE_SIZE_USE_DDB: bool = True
INTRADAY_LARGE_ACTIVE_BUY_RATIO_USE_DDB: bool = True
INTRADAY_AMIHUD_USE_DDB: bool = True
INTRADAY_REALIZED_VOLATILITY_USE_DDB: bool = True
INTRADAY_MINUTE_SKEW_USE_DDB: bool = True

# 信号时刻涨跌停过滤（DDB get_limit_status @ Symbol/Date/Bartime）
INTRADAY_APPLY_LIMIT_FILTER: bool = True

# 口径B H-L 换手：当日开仓+平仓；与 H-L 收益定义一致（G10−G1 = 多空各 side_gross 倍权益）
# Turnover_B ≈ 2 × (long_gross + short_gross) / equity = 4 × INTRADAY_HL_SIDE_GROSS
# side_gross=1.0 → TO≈4.0；side_gross=0.5 → TO≈2.0
INTRADAY_HL_SIDE_GROSS: float = 1.0


# =========================
# 各 track 默认因子清单
# =========================
TRACK_DEFAULT_LISTS = {
    "eod_pv": ALL_FACTOR_LIST,
    "eod_engine": EOD_ENGINE_CORE_LIST,
    "eod_engine_ext": EOD_ENGINE_ALL_LIST,
    "eod_engine_priority_a": EOD_ENGINE_PRIORITY_A_LIST,
    "eod_engine_hf_v2": EOD_ENGINE_HF_V2_LIST,
    "eod_engine_hf_v3": EOD_ENGINE_HF_V3_LIST,
    "eod_engine_hf_v4": EOD_ENGINE_HF_V4_LIST,
    "eod_engine_hf_v5": EOD_ENGINE_HF_V5_LIST,
    "eod_engine_robust": EOD_ENGINE_ROBUST_LIST,
    "eod_cn_broker_v1": EOD_CN_BROKER_V1_LIST,
    "eod_cn_broker_all": EOD_CN_BROKER_ALL_LIST,
    "alpha_bundle_v1": ALPHA_BUNDLE_V1_LIST,
    "eod_latent": [],  # names resolved at runtime via build_latent_eod_factors
    "eod_liquidity_norm": LIQUIDITY_NORM_CORE_LIST,
    "eod_liquidity_norm_ext": LIQUIDITY_NORM_ALL_LIST,
    "fundamental": FUNDAMENTAL_PHASE1_LIST,
    "fundamental_batch1": FUNDAMENTAL_BATCH1_LIST,
    "fundamental_phase2": FUNDAMENTAL_PHASE2_BATCH_LIST,
    "fundamental_quality_d7": FUNDAMENTAL_QUALITY_D7_BATCH_LIST,
    "fundamental_value_d6": FUNDAMENTAL_VALUE_D6_BATCH_LIST,
    "l2_microstructure_v1": L2_MICROSTRUCTURE_V1_LIST,
    "l2_microstructure_v2": L2_MICROSTRUCTURE_V2_LIST,
    "smart_money_active_v2": SMART_MONEY_ACTIVE_V2_LIST,
    "apm_active_v2": APM_ACTIVE_V2_LIST,
    "ideal_reversal_active_v2": IDEAL_REVERSAL_ACTIVE_V2_LIST,
    "ideal_amplitude_active_v2": IDEAL_AMPLITUDE_V2_LIST,
    "intraday": INTRADAY_PHASE1_LIST + INTRADAY_PHASE2_LIST,
}

# 价量 track 常用子集（挖掘不够时优先扩这里）
EOD_PV_CLASSIC = CLASSIC_FACTOR_LIST
EOD_PV_NEW = NEW_EOD_FACTOR_LIST
EOD_PV_PRIORITY = PRIORITY_NEW_FACTORS
EOD_ENGINE_CORE = EOD_ENGINE_CORE_LIST
EOD_ENGINE_EXTENDED = EOD_ENGINE_ALL_LIST
EOD_ENGINE_PRIORITY_A = EOD_ENGINE_PRIORITY_A_LIST
EOD_ENGINE_HF_V2 = EOD_ENGINE_HF_V2_LIST
EOD_ENGINE_HF_V3 = EOD_ENGINE_HF_V3_LIST
EOD_ENGINE_HF_V4 = EOD_ENGINE_HF_V4_LIST
EOD_ENGINE_HF_V5 = EOD_ENGINE_HF_V5_LIST
EOD_ENGINE_ROBUST = EOD_ENGINE_ROBUST_LIST
EOD_CN_BROKER_V1 = EOD_CN_BROKER_V1_LIST
EOD_CN_BROKER_ALL = EOD_CN_BROKER_ALL_LIST
ALPHA_BUNDLE_V1 = ALPHA_BUNDLE_V1_LIST
LIQUIDITY_NORM_CORE = LIQUIDITY_NORM_CORE_LIST
LIQUIDITY_NORM_EXTENDED = LIQUIDITY_NORM_ALL_LIST

# ActiveV2 / 切割因子回测股票池：仅全A + 中证1000/500/沪深300
UNIVERSE_LIST = {
    "ALL": None,
    "CSI1000": "000852.SH",
    "CSI500": "000905.SH",
    "CSI300": "000300.SH",
}

RESEARCH_DIR = Path("research/results")


def resolve_factor_list(track: str = TRACK, custom: Optional[List[str]] = None) -> List[str]:
    if custom is not None:
        return list(custom)
    if CUSTOM_FACTOR_LIST is not None:
        return list(CUSTOM_FACTOR_LIST)
    return list(TRACK_DEFAULT_LISTS[track])


def resolve_batch_tag(track: str, factor_list: List[str]) -> str:
    if track == "eod_pv":
        if factor_list == PRIORITY_NEW_FACTORS:
            return "priority_new"
        if factor_list == NEW_EOD_FACTOR_LIST:
            return "new_eod"
        if factor_list == CLASSIC_FACTOR_LIST:
            return "classic"
        if factor_list == ALL_FACTOR_LIST:
            return "all"
    if track == "eod_engine":
        if factor_list == EOD_ENGINE_CORE_LIST:
            return "core"
        if factor_list == EOD_ENGINE_ALL_LIST:
            return "extended"
    if track == "eod_engine_ext":
        return "extended"
    if track == "eod_engine_priority_a":
        return "priority_a"
    if track == "eod_engine_hf_v2":
        return "hf_v2"
    if track == "eod_engine_hf_v3":
        return "hf_v3"
    if track == "eod_engine_hf_v4":
        return "hf_v4"
    if track == "eod_engine_hf_v5":
        return "hf_v5"
    if track == "eod_engine_robust":
        return "robust"
    if track == "eod_cn_broker_v1":
        return "cn_broker_v1"
    if track == "eod_cn_broker_all":
        return "cn_broker_all"
    if track == "alpha_bundle_v1":
        return "bundle_v1"
    if track == "eod_latent":
        return "latent"
    if track == "eod_liquidity_norm":
        if factor_list == LIQUIDITY_NORM_CORE_LIST:
            return "core"
        if factor_list == LIQUIDITY_NORM_ALL_LIST:
            return "extended"
    if track == "eod_liquidity_norm_ext":
        return "extended"
    if track == "fundamental":
        if factor_list == FUNDAMENTAL_PHASE1_LIST:
            return "phase1"
        if factor_list == FUNDAMENTAL_FACTOR_LIST:
            return "all"
    if track == "fundamental_batch1":
        return "batch1"
    if track == "fundamental_phase2":
        return "phase2"
    if track == "fundamental_quality_d7":
        return "quality_d7"
    if track == "fundamental_value_d6":
        return "value_d6"
    if track == "l2_microstructure_v1":
        return "l2_v1"
    if track == "l2_microstructure_v2":
        return "l2_v2"
    if track == "smart_money_active_v2":
        return "sm_active_v2"
    if track == "apm_active_v2":
        return "apm_active_v2"
    if track == "ideal_reversal_active_v2":
        return "ideal_rev_active_v2"
    if track == "ideal_amplitude_active_v2":
        return "ideal_amp_active_v2"
    if track == "intraday":
        if factor_list == INTRADAY_PHASE1_LIST:
            return "phase1"
        if factor_list == INTRADAY_PHASE2_LIST:
            return "phase2"
        if factor_list == INTRADAY_PHASE1_LIST + INTRADAY_PHASE2_LIST:
            return "phase1_plus_2"
        if factor_list == INTRADAY_FACTOR_LIST:
            return "all"
    return "custom"


def result_root_for(track: str) -> str:
    return f"result/{track}"


def manifest_path_for(track: str) -> Path:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    return RESEARCH_DIR / f"factor_run_manifest_{track}.csv"
