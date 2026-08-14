"""D6 Value block — industry + size neutral EP / BP / CFP → value_composite.

Research question: does A-share Value provide return information independent of D1–D7?

Bricks (all industry-neutral + size-neutral):
  value_ep  — earnings yield (1/PE TTM, sanity-masked)
  value_bp  — book-to-price (1/PB, sanity-masked)
  value_cfp — operating cash flow TTM / float mktcap (ann_date CFO)

Representative:
  value_composite — equal cs_z(value_ep, value_bp, value_cfp)
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from factor_finance import calc_cfp_wide
from factor_formulas_fundamental import FundamentalDataCache, build_fundamental_cache
from factor_attribution import cs_zscore
from industry_neutral import panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual

VALUE_D6_BRICKS = [
    "value_ep",
    "value_bp",
    "value_cfp",
]

VALUE_D6_REP = "value_composite"

FUNDAMENTAL_VALUE_D6_BATCH_LIST = [
    VALUE_D6_REP,
    *VALUE_D6_BRICKS,
]

REQUIRES_FINANCE_ANN_VALUE = {"value_cfp", "value_composite"}
REQUIRES_PE_VALUE = {"value_ep", "value_composite"}
REQUIRES_PB_VALUE = {"value_bp", "value_composite"}


def neutralize_value_signal(
    raw: pd.DataFrame,
    cache: FundamentalDataCache,
) -> pd.DataFrame:
    """Industry demean then cross-sectional residual vs log float market cap."""
    ind_neutral = panel_industry_demean(raw, cache.industry_panel())
    log_size = np.log(cache.require("float_mktcap").replace(0, np.nan))
    log_size = log_size.reindex(index=ind_neutral.index, columns=ind_neutral.columns)
    return panel_cross_sectional_residual(ind_neutral, [log_size])


def _raw_value_ep(cache: FundamentalDataCache) -> pd.DataFrame:
    return cache.ep_ttm_clean()


def _raw_value_bp(cache: FundamentalDataCache) -> pd.DataFrame:
    return cache.bp_clean()


def _raw_value_cfp(cache: FundamentalDataCache) -> pd.DataFrame:
    if cache.close is None or cache.finance_long is None or cache.finance_long.empty:
        raise ValueError("value_cfp requires close panel and finance_long (ASHARETTMHIS)")
    return calc_cfp_wide(cache.finance_long, cache.close, cache.require("float_mktcap"))


ValueFunc = Callable[[FundamentalDataCache], pd.DataFrame]
VALUE_REGISTRY: Dict[str, ValueFunc] = {}


def register_value(name: str):
    def decorator(func: ValueFunc) -> ValueFunc:
        if name in VALUE_REGISTRY:
            raise ValueError(f"Duplicated value factor: {name}")
        VALUE_REGISTRY[name] = func
        return func

    return decorator


@register_value("value_ep")
def factor_value_ep(cache: FundamentalDataCache) -> pd.DataFrame:
    return neutralize_value_signal(_raw_value_ep(cache), cache)


@register_value("value_bp")
def factor_value_bp(cache: FundamentalDataCache) -> pd.DataFrame:
    return neutralize_value_signal(_raw_value_bp(cache), cache)


@register_value("value_cfp")
def factor_value_cfp(cache: FundamentalDataCache) -> pd.DataFrame:
    raw = _raw_value_cfp(cache)
    return neutralize_value_signal(raw, cache)


@register_value("value_composite")
def factor_value_composite(cache: FundamentalDataCache) -> pd.DataFrame:
    bricks = [
        factor_value_ep(cache),
        factor_value_bp(cache),
        factor_value_cfp(cache),
    ]
    z_parts = [cs_zscore(b) for b in bricks if b is not None and not b.empty]
    if not z_parts:
        ref = cache.require("float_mktcap")
        return pd.DataFrame(index=ref.index, columns=ref.columns, dtype=float)
    composite = sum(z_parts) / len(z_parts)
    return composite.reindex(
        index=cache.require("float_mktcap").index,
        columns=cache.require("float_mktcap").columns,
    )


def build_value_cache(
    data,
    *,
    close: Optional[pd.DataFrame] = None,
    finance_long: Optional[pd.DataFrame] = None,
) -> FundamentalDataCache:
    return build_fundamental_cache(data, close=close, finance_long=finance_long)


def build_value_factor(factor_name: str, cache: FundamentalDataCache) -> pd.DataFrame:
    if factor_name not in VALUE_REGISTRY:
        valid = sorted(VALUE_REGISTRY.keys())
        raise ValueError(f"Unknown value factor: {factor_name}. Valid: {valid}")
    return VALUE_REGISTRY[factor_name](cache)


def filter_available_value_factors(
    factor_names: Iterable[str],
    has_pb: bool,
    has_pe: bool,
    *,
    has_finance_ann: bool = False,
) -> List[str]:
    output = []
    for name in factor_names:
        if name not in VALUE_REGISTRY:
            print(f"[SKIP] Unknown value factor: {name}")
            continue
        if name in REQUIRES_FINANCE_ANN_VALUE and not has_finance_ann:
            print(f"[SKIP] {name}: ann_date finance panel not available")
            continue
        if name in REQUIRES_PE_VALUE and not has_pe:
            print(f"[SKIP] {name}: PE_TTM field not available")
            continue
        if name in REQUIRES_PB_VALUE and not has_pb:
            print(f"[SKIP] {name}: PB field not available")
            continue
        output.append(name)
    return output
