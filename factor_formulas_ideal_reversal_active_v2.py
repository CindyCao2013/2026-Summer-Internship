"""Runner glue for IdealReversal_ActiveV2 family."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from factor_cutting.ideal_reversal_active_v2 import FORMULA_VERSION
from factor_cutting.smart_money_active_v2 import (
    cs_zscore_min_n,
    ffill_limited,
    fill_industry_then_market,
    mad_winsorize_cs,
)

IDEAL_REVERSAL_ACTIVE_V2_LIST = [
    "IdealReversal_ActiveV2",
    "IdealReversal_ActiveV2_Weekly",
    "IdealReversal_ActiveV2_PureRev",
    "IdealReversal_ActiveV2_Weekly_PureRev",
    "IdealReversal_ActiveV2_Weekly_Thu",
    "IdealReversal_ActiveV2_RollingGate",
    "IdealReversal_ActiveV2_Weekly_Thu_RollingGate",
    "IdealReversal_ActiveV2_raw",
]


def filter_ideal_reversal_active_v2_factors(names: List[str]) -> List[str]:
    valid = set(IDEAL_REVERSAL_ACTIVE_V2_LIST)
    out = []
    for n in names:
        if n not in valid:
            print(f"[SKIP] Unknown IdealReversal_ActiveV2 factor: {n}")
            continue
        out.append(n)
    return out


filter_ideal_reversal_factors = filter_ideal_reversal_active_v2_factors


def process_factor_cross_section(
    factor_raw: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    neutralize: bool = True,
    do_ffill: bool = True,
) -> pd.DataFrame:
    fac = factor_raw.copy().astype(float)
    if do_ffill:
        fac = ffill_limited(fac)
    fac = mad_winsorize_cs(fac)
    if industry is not None:
        fac = fill_industry_then_market(fac, industry, halt_mask=halt_mask)
    else:
        fac = fac.apply(lambda row: row.fillna(row.median()), axis=1)
    fac = cs_zscore_min_n(fac)
    if neutralize and float_mktcap is not None:
        import Factor_Dev_Lib as FDL

        neu = FDL.panel_neutral_size_ind(fac, nt_type="ind_cap")
        fac = cs_zscore_min_n(neu)
    return fac


def build_ideal_reversal_active_v2_factor(
    factor_name: str,
    raw_by_name: Dict[str, pd.DataFrame],
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """``raw_by_name`` maps factor id → daily/weekly raw wide panel."""
    if factor_name == "IdealReversal_ActiveV2_raw":
        return raw_by_name["IdealReversal_ActiveV2"]

    if isinstance(raw_by_name, pd.DataFrame):
        raw = raw_by_name
        if factor_name not in (
            "IdealReversal_ActiveV2",
            "IdealReversal_ActiveV2_raw",
        ):
            raise ValueError(
                f"{factor_name} needs variants dict; got single DataFrame"
            )
    else:
        if factor_name not in raw_by_name:
            raise ValueError(
                f"No raw panel for {factor_name}. Have: {list(raw_by_name)}"
            )
        raw = raw_by_name[factor_name]

    if factor_name in IDEAL_REVERSAL_ACTIVE_V2_LIST:
        if factor_name.endswith("_raw"):
            return raw
        return process_factor_cross_section(
            raw,
            industry=industry,
            halt_mask=halt_mask,
            float_mktcap=float_mktcap,
            neutralize=True,
        )
    raise ValueError(
        f"Unknown factor {factor_name}. Valid: {IDEAL_REVERSAL_ACTIVE_V2_LIST}; "
        f"formula={FORMULA_VERSION}"
    )


build_ideal_reversal_factor = build_ideal_reversal_active_v2_factor
