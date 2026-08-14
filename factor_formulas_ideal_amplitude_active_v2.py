# ============================================================
# factor_formulas_ideal_amplitude_active_v2.py
# Runner 胶水：横截面后处理
# ============================================================
"""Runner glue for IdealAmplitude_ActiveV2.

Cross-section: MAD → industry fill → zscore → size+ind neutral → zscore.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factor_cutting.ideal_amplitude_active_v2 import FORMULA_VERSION
from factor_cutting.smart_money_active_v2 import (
    cs_zscore_min_n,
    ffill_limited,
    fill_industry_then_market,
    mad_winsorize_cs,
)

IDEAL_AMPLITUDE_V2_LIST = [
    "IdealAmplitude_ActiveV2",
    "IdealAmplitude_ActiveV2_raw",
]


def filter_ideal_amplitude_factors(names: List[str]) -> List[str]:
    valid = set(IDEAL_AMPLITUDE_V2_LIST)
    out = []
    for n in names:
        if n not in valid:
            print(f"[SKIP] Unknown IdealAmplitude factor: {n}")
            continue
        out.append(n)
    return out


def process_factor_cross_section(
    amp_smooth: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
    neutralize: bool = True,
    do_ffill: bool = True,
) -> pd.DataFrame:
    fac = amp_smooth.copy().astype(float)
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


def build_ideal_amplitude_factor(
    factor_name: str,
    amp_smooth: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    halt_mask: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if factor_name == "IdealAmplitude_ActiveV2_raw":
        return amp_smooth
    if factor_name == "IdealAmplitude_ActiveV2":
        return process_factor_cross_section(
            amp_smooth,
            industry=industry,
            halt_mask=halt_mask,
            float_mktcap=float_mktcap,
            neutralize=True,
        )
    raise ValueError(
        f"Unknown IdealAmplitude factor: {factor_name}. "
        f"Valid: {IDEAL_AMPLITUDE_V2_LIST}; formula={FORMULA_VERSION}"
    )
