"""Industry-neutral transforms for fundamental factors (CITICS L1 demean)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import Factor_Dev_Lib


def load_citics_industry_panel(start, end) -> pd.DataFrame:
    """Wide industry codes: index=TradingDay, columns=symbols."""
    ind = Factor_Dev_Lib.get_preheat_ind_data_citics(start, end)
    ind = ind.set_index("TradingDay")
    ind.index = pd.to_datetime(ind.index)
    return ind


def panel_industry_demean(signal: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional demean within industry each day (residual vs industry mean)."""
    industry = industry.reindex(index=signal.index, columns=signal.columns)
    out = signal.copy() * np.nan
    for dt in signal.index:
        s = signal.loc[dt]
        ind = industry.loc[dt]
        mask = s.notna() & ind.notna()
        if mask.sum() < 30:
            continue
        sub = pd.DataFrame({"v": s[mask], "ind": ind[mask].astype(str)})
        resid = sub["v"] - sub.groupby("ind")["v"].transform("mean")
        out.loc[dt, resid.index] = resid.values
    return out


def apply_valuation_sanity_mask(
    ep: pd.DataFrame,
    pe_ttm: pd.DataFrame,
    pb: Optional[pd.DataFrame] = None,
    pe_max: float = 150.0,
    pb_max: float = 30.0,
) -> pd.DataFrame:
    """Mask invalid / extreme valuation observations before industry neutral."""
    pe = pe_ttm.reindex_like(ep)
    valid = pe.notna() & (pe > 0) & (pe <= pe_max)
    if pb is not None:
        pb_a = pb.reindex_like(ep)
        valid = valid & pb_a.notna() & (pb_a > 0) & (pb_a <= pb_max)
    return ep.where(valid)
