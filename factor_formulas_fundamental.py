"""日频财务 / 估值 / 市值因子。

Phase 1: DERIVATIVEINDICATOR 日频宽表 (EP/BP/市值)
Phase 2: ann_date 对齐财报 (ROE / roe_stability) — factor_finance 适配层
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from factor_data_loaders import DerivativeWideTables
from factor_finance import (
    calc_cfo_quality_wide,
    calc_gross_profitability_wide,
    calc_quality_composite_wide,
    calc_roe_stability_wide,
    calc_roe_wide,
)
from industry_neutral import (
    apply_valuation_sanity_mask,
    load_citics_industry_panel,
    panel_industry_demean,
)

_EPS = 1e-6

# Phase 1: 仅用 EOD 衍生指标表即可计算
FUNDAMENTAL_PHASE1_LIST = [
    "log_float_mktcap",
    "log_total_mktcap",
    "float_to_total_mktcap",
    "bp",
    "ep_ttm",
    "sp_ttm",
]

# Batch 1: industry-neutral value anchors (HF triage priority)
FUNDAMENTAL_BATCH1_LIST = [
    "ep_ttm_ind_neutral",
    "bp_ind_neutral",
    "ep_ttm",
    "bp",
]

# Phase 2: ann_date 对齐（已实现）
FUNDAMENTAL_PHASE2_IMPLEMENTED = [
    "roe",
    "roe_stability",
]

# D7 Quality block — bricks feed composite; composite is D7 representative
FUNDAMENTAL_QUALITY_D7_BRICKS = [
    "gross_profitability",
    "cfo_quality",
]

FUNDAMENTAL_QUALITY_D7_REP = "quality_composite"

FUNDAMENTAL_QUALITY_D7_BATCH_LIST = [
    FUNDAMENTAL_QUALITY_D7_REP,
    *FUNDAMENTAL_QUALITY_D7_BRICKS,
]

# Phase 2 roadmap（待接入）
FUNDAMENTAL_PHASE2_ROADMAP = [
    "relative_bp",
    "relative_ep",
    "relative_sp",
    "roa",
    "net_profit_margin",
    "asset_turnover",
    "sales_to_mktcap",
]

FUNDAMENTAL_PHASE2_LIST = FUNDAMENTAL_PHASE2_IMPLEMENTED + FUNDAMENTAL_PHASE2_ROADMAP

# Phase 2 validation batch (Quality pillar — D7)
FUNDAMENTAL_PHASE2_BATCH_LIST = FUNDAMENTAL_PHASE2_IMPLEMENTED

FUNDAMENTAL_FACTOR_LIST = FUNDAMENTAL_PHASE1_LIST + FUNDAMENTAL_PHASE2_LIST

REQUIRES_PB = {"bp", "relative_bp", "bp_ind_neutral"}
REQUIRES_PE = {"ep_ttm", "relative_ep", "ep_ttm_ind_neutral"}
REQUIRES_PS = {"sp_ttm", "relative_sp", "sales_to_mktcap"}
PHASE2_FACTORS = set(FUNDAMENTAL_PHASE2_LIST)
PHASE2_IMPLEMENTED = set(FUNDAMENTAL_PHASE2_IMPLEMENTED)
PHASE2_PENDING = set(FUNDAMENTAL_PHASE2_ROADMAP)
QUALITY_D7_FACTORS = set(FUNDAMENTAL_QUALITY_D7_BATCH_LIST)
REQUIRES_FINANCE_ANN = set(FUNDAMENTAL_PHASE2_IMPLEMENTED) | QUALITY_D7_FACTORS


@dataclass
class FundamentalDataCache:
    """Lazy cache for fundamental wide tables."""

    data: DerivativeWideTables
    close: Optional[pd.DataFrame] = None
    finance_long: Optional[pd.DataFrame] = None
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _industry: Optional[pd.DataFrame] = field(default=None, repr=False)

    def require(self, field: str) -> pd.DataFrame:
        value = getattr(self.data, field)
        if value is None:
            raise ValueError(f"Required fundamental field `{field}` is not available")
        return value

    def industry_panel(self) -> pd.DataFrame:
        if self._industry is None:
            import factor_config as cfg

            ref = self.require("float_mktcap")
            # PREHEAT_IND_DATA_CITICS 通常从 2020-01-02 起；勿用价量 preheat 起始日
            start = max(pd.to_datetime(cfg.START_DAY), pd.to_datetime(ref.index[0]))
            end = pd.to_datetime(ref.index[-1])
            ind = load_citics_industry_panel(start, end)
            self._industry = ind.reindex(index=ref.index, columns=ref.columns)
        return self._industry

    def ep_ttm_clean(self) -> pd.DataFrame:
        ep = self.get("ep_ttm")
        pe = self.require("pe_ttm")
        pb = self.data.pb
        return apply_valuation_sanity_mask(ep, pe, pb)

    def bp_clean(self) -> pd.DataFrame:
        bp = self.get("bp")
        pb = self.require("pb")
        pe = self.require("pe_ttm")
        return apply_valuation_sanity_mask(bp, pe, pb)

    def get(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]

        if key == "log_float_mktcap":
            value = np.log(self.require("float_mktcap").replace(0, np.nan))
        elif key == "log_total_mktcap":
            value = np.log(self.require("total_mktcap").replace(0, np.nan))
        elif key == "float_to_total_mktcap":
            value = self.require("float_mktcap") / self.require("total_mktcap").replace(
                0, np.nan
            )
        elif key == "bp":
            pb = self.require("pb")
            value = 1.0 / pb.replace(0, np.nan)
        elif key == "ep_ttm":
            pe = self.require("pe_ttm")
            value = 1.0 / pe.replace(0, np.nan)
        elif key == "sp_ttm":
            ps = self.require("ps_ttm")
            value = 1.0 / ps.replace(0, np.nan)
        else:
            raise KeyError(f"Unknown fundamental cache key: {key}")

        self._cache[key] = value
        return value


FundamentalFunc = Callable[[FundamentalDataCache], pd.DataFrame]
FUNDAMENTAL_REGISTRY: Dict[str, FundamentalFunc] = {}


def register_fundamental(name: str):
    def decorator(func: FundamentalFunc) -> FundamentalFunc:
        if name in FUNDAMENTAL_REGISTRY:
            raise ValueError(f"Duplicated fundamental factor: {name}")
        FUNDAMENTAL_REGISTRY[name] = func
        return func

    return decorator


@register_fundamental("log_float_mktcap")
def factor_log_float_mktcap(cache: FundamentalDataCache) -> pd.DataFrame:
    return -cache.get("log_float_mktcap")


@register_fundamental("log_total_mktcap")
def factor_log_total_mktcap(cache: FundamentalDataCache) -> pd.DataFrame:
    return -cache.get("log_total_mktcap")


@register_fundamental("float_to_total_mktcap")
def factor_float_to_total_mktcap(cache: FundamentalDataCache) -> pd.DataFrame:
    return cache.get("float_to_total_mktcap")


@register_fundamental("ep_ttm")
def factor_ep_ttm(cache: FundamentalDataCache) -> pd.DataFrame:
    return cache.ep_ttm_clean()


@register_fundamental("ep_ttm_ind_neutral")
def factor_ep_ttm_ind_neutral(cache: FundamentalDataCache) -> pd.DataFrame:
    ep = cache.ep_ttm_clean()
    return panel_industry_demean(ep, cache.industry_panel())


@register_fundamental("sp_ttm")
def factor_sp_ttm(cache: FundamentalDataCache) -> pd.DataFrame:
    return cache.get("sp_ttm")


@register_fundamental("bp")
def factor_bp(cache: FundamentalDataCache) -> pd.DataFrame:
    return cache.bp_clean()


@register_fundamental("bp_ind_neutral")
def factor_bp_ind_neutral(cache: FundamentalDataCache) -> pd.DataFrame:
    bp = cache.bp_clean()
    return panel_industry_demean(bp, cache.industry_panel())


@register_fundamental("roe")
def factor_roe(cache: FundamentalDataCache) -> pd.DataFrame:
    if cache.close is None or cache.finance_long is None or cache.finance_long.empty:
        raise ValueError("roe requires close panel and finance_long (ASHARETTMHIS)")
    return calc_roe_wide(cache.finance_long, cache.close)


@register_fundamental("roe_stability")
def factor_roe_stability(cache: FundamentalDataCache) -> pd.DataFrame:
    if cache.close is None or cache.finance_long is None or cache.finance_long.empty:
        raise ValueError("roe_stability requires close panel and finance_long")
    return calc_roe_stability_wide(cache.finance_long, cache.close)


@register_fundamental("gross_profitability")
def factor_gross_profitability(cache: FundamentalDataCache) -> pd.DataFrame:
    if cache.close is None or cache.finance_long is None or cache.finance_long.empty:
        raise ValueError("gross_profitability requires close panel and finance_long")
    return calc_gross_profitability_wide(cache.finance_long, cache.close)


@register_fundamental("cfo_quality")
def factor_cfo_quality(cache: FundamentalDataCache) -> pd.DataFrame:
    if cache.close is None or cache.finance_long is None or cache.finance_long.empty:
        raise ValueError("cfo_quality requires close panel and finance_long")
    return calc_cfo_quality_wide(cache.finance_long, cache.close)


@register_fundamental("quality_composite")
def factor_quality_composite(cache: FundamentalDataCache) -> pd.DataFrame:
    if cache.close is None or cache.finance_long is None or cache.finance_long.empty:
        raise ValueError("quality_composite requires close panel and finance_long")
    return calc_quality_composite_wide(cache.finance_long, cache.close)


def build_fundamental_cache(
    data: DerivativeWideTables,
    *,
    close: Optional[pd.DataFrame] = None,
    finance_long: Optional[pd.DataFrame] = None,
) -> FundamentalDataCache:
    return FundamentalDataCache(data=data, close=close, finance_long=finance_long)


def build_fundamental_factor(factor_name: str, cache: FundamentalDataCache) -> pd.DataFrame:
    if factor_name in PHASE2_PENDING:
        raise ValueError(
            f"{factor_name} is Phase 2 roadmap item. Not implemented yet."
        )
    if factor_name not in FUNDAMENTAL_REGISTRY:
        valid = sorted(FUNDAMENTAL_REGISTRY.keys())
        raise ValueError(f"Unknown fundamental factor: {factor_name}. Valid: {valid}")
    return FUNDAMENTAL_REGISTRY[factor_name](cache)


def filter_available_fundamental_factors(
    factor_names: Iterable[str],
    has_pb: bool,
    has_pe: bool,
    has_ps: bool,
    *,
    has_finance_ann: bool = False,
) -> List[str]:
    output = []
    for name in factor_names:
        if name in PHASE2_PENDING:
            print(f"[SKIP] {name}: Phase 2 roadmap — not implemented")
            continue
        if name in REQUIRES_FINANCE_ANN and not has_finance_ann:
            print(f"[SKIP] {name}: ann_date finance panel not available")
            continue
        if name not in FUNDAMENTAL_REGISTRY:
            print(f"[SKIP] Unknown fundamental factor: {name}")
            continue
        if name in REQUIRES_PB and not has_pb:
            print(f"[SKIP] {name}: PB field not available")
            continue
        if name in REQUIRES_PE and not has_pe:
            print(f"[SKIP] {name}: PE_TTM field not available")
            continue
        if name in REQUIRES_PS and not has_ps:
            print(f"[SKIP] {name}: PS_TTM field not available")
            continue
        output.append(name)
    return output
