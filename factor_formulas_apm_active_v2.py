"""Runner glue for APM_ActiveV2 (Active Pressure Metric).

Cross-section post-process: MAD → industry fill → zscore → size+ind neutral → zscore.
No PureRev — APM is directional (higher = stronger active buy pressure).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factor_cutting.apm_active_v2 import FORMULA_VERSION
from factor_cutting.smart_money_active_v2 import (
    cs_zscore_min_n,
    ffill_limited,
    fill_industry_then_market,
    mad_winsorize_cs,
)

APM_ACTIVE_V2_LIST = [
    "APM_ActiveV2",
    "APM_ActiveV2_Weekly",
    "APM_ActiveV2_Weekly_Thu",
    "APM_ActiveV2_Raw",
    "APM_ActiveV2_Session",
    "APM_ActiveV2_Smart",
    "APM_ActiveV2_Delta",
    "APM_ActiveV2_SmartV2",
    "APM_ActiveV2_SmartV2_1F",
    "APM_ActiveV2_SmartV2_1",
]

_RAW_ALIASES = {"APM_ActiveV2_raw": "APM_ActiveV2_Raw"}

_CS_FACTORS = {
    "APM_ActiveV2",
    "APM_ActiveV2_Weekly",
    "APM_ActiveV2_Weekly_Thu",
    "APM_ActiveV2_Session",
    "APM_ActiveV2_Smart",
    "APM_ActiveV2_Delta",
    "APM_ActiveV2_SmartV2",
    "APM_ActiveV2_SmartV2_1F",
    "APM_ActiveV2_SmartV2_1",
}


def filter_apm_active_v2_factors(names: List[str]) -> List[str]:
    valid = set(APM_ACTIVE_V2_LIST)
    out = []
    for n in names:
        n2 = _RAW_ALIASES.get(n, n)
        if n2 not in valid:
            print(f"[SKIP] Unknown APM_ActiveV2 factor: {n}")
            continue
        out.append(n2)
    return out


def process_factor_cross_section(
    apm_panel: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    neutralize: bool = True,
    do_ffill: bool = True,
) -> pd.DataFrame:
    """Full CS pipeline on APM wide panel."""
    fac = apm_panel.copy().astype(float)
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
    elif neutralize and industry is not None and float_mktcap is None:
        from industry_neutral import panel_industry_demean

        fac = cs_zscore_min_n(panel_industry_demean(fac, industry))

    return fac


def build_apm_active_v2_factor(
    factor_name: str,
    raw_panel: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    name = _RAW_ALIASES.get(factor_name, factor_name)
    if name == "APM_ActiveV2_Raw":
        return ffill_limited(raw_panel.copy().astype(float))
    if name in _CS_FACTORS:
        return process_factor_cross_section(
            raw_panel,
            industry=industry,
            halt_mask=halt_mask,
            float_mktcap=float_mktcap,
            neutralize=True,
        )
    raise ValueError(
        f"Unknown factor {factor_name}. Valid: {APM_ACTIVE_V2_LIST}; "
        f"formula={FORMULA_VERSION}"
    )
