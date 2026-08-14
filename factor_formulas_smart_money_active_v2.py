"""Runner glue for SmartMoneyActiveV2 (主动大单集中度).

Cross-section post-process: MAD → industry fill → zscore → size+ind neutral → zscore.
Industry panel uses Citics (repo standard; design doc notes 申万一级 as conceptual).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factor_cutting.smart_money_active_v2 import (
    FORMULA_VERSION,
    cs_zscore_min_n,
    ffill_limited,
    fill_industry_then_market,
    mad_winsorize_cs,
)

SMART_MONEY_ACTIVE_V2_LIST = [
    "SmartMoneyActiveV2",
    "SmartMoneyActiveV2_raw",
]


def filter_smart_money_active_v2_factors(names: List[str]) -> List[str]:
    valid = set(SMART_MONEY_ACTIVE_V2_LIST)
    out = []
    for n in names:
        if n not in valid:
            print(f"[SKIP] Unknown SmartMoneyActiveV2 factor: {n}")
            continue
        out.append(n)
    return out


def process_factor_cross_section(
    smart_raw: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    neutralize: bool = True,
    do_ffill: bool = True,
) -> pd.DataFrame:
    """Full CS pipeline on smart_raw wide panel.

    If neutralize=True and float_mktcap given, uses Factor_Dev_Lib.panel_neutral_size_ind.
    """
    fac = smart_raw.copy().astype(float)
    if do_ffill:
        fac = ffill_limited(fac)

    fac = mad_winsorize_cs(fac)

    if industry is not None:
        fac = fill_industry_then_market(fac, industry, halt_mask=halt_mask)
    else:
        # market median only
        fac = fac.apply(lambda row: row.fillna(row.median()), axis=1)

    fac = cs_zscore_min_n(fac)

    if neutralize and float_mktcap is not None:
        import Factor_Dev_Lib as FDL

        neu = FDL.panel_neutral_size_ind(fac, nt_type="ind_cap")
        fac = cs_zscore_min_n(neu)
    elif neutralize and industry is not None and float_mktcap is None:
        # industry demean fallback
        from industry_neutral import panel_industry_demean

        fac = cs_zscore_min_n(panel_industry_demean(fac, industry))

    return fac


def build_smart_money_active_v2_factor(
    factor_name: str,
    smart_raw: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if factor_name == "SmartMoneyActiveV2_raw":
        return smart_raw
    if factor_name == "SmartMoneyActiveV2":
        return process_factor_cross_section(
            smart_raw,
            industry=industry,
            halt_mask=halt_mask,
            float_mktcap=float_mktcap,
            neutralize=True,
        )
    raise ValueError(
        f"Unknown factor {factor_name}. Valid: {SMART_MONEY_ACTIVE_V2_LIST}; "
        f"formula={FORMULA_VERSION}"
    )
