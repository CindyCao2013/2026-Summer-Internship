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
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_ALL_LIST,
    LIQUIDITY_NORM_CORE_LIST,
)
from intraday_formulas import INTRADAY_FACTOR_LIST, INTRADAY_PHASE1_LIST

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
TRACK = "fundamental_batch1"

# =========================
# 回测区间
# =========================
START_DAY = dt.datetime(2020, 1, 1)
END_DAY = dt.datetime(2025, 12, 31)
PREHEAT_CALENDAR_DAYS = 400

# =========================
# 批量 / 断点续跑
# =========================
BATCH_MODE = True
SKIP_COMPLETED = True
RESUME_FROM_EXISTING = True
SAVE_RESULTS = True
METHOD = "c2c"
SHOW_GROUP_TEST_PLOTS = not BATCH_MODE

# batch_mode=False 时只跑这一个因子
SINGLE_FACTOR_NAME = "amount_stability_20d"

# None = 用 track 默认清单；或手动指定列表
CUSTOM_FACTOR_LIST: Optional[List[str]] = None

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
    "intraday": INTRADAY_PHASE1_LIST,
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

UNIVERSE_LIST = {
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
    "ALL": None,
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
    if track == "intraday":
        if factor_list == INTRADAY_PHASE1_LIST:
            return "phase1"
        if factor_list == INTRADAY_FACTOR_LIST:
            return "all"
    return "custom"


def result_root_for(track: str) -> str:
    return f"result/{track}"


def manifest_path_for(track: str) -> Path:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    return RESEARCH_DIR / f"factor_run_manifest_{track}.csv"
