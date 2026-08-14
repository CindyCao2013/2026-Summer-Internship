"""Tradability masks: ST / PT, limit, suspend. Next-session filters match the guide."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def a_share_columns(columns) -> pd.Index:
    return pd.Index([c for c in columns if str(c)[:1] in ("6", "0", "3")])


def combine_masks(*masks: Optional[pd.DataFrame]) -> pd.DataFrame:
    valid = [m for m in masks if m is not None]
    if not valid:
        raise ValueError("at least one mask is required")
    out = valid[0].astype(float)
    for m in valid[1:]:
        aligned = m.reindex(index=out.index, columns=out.columns)
        out = out.where(aligned == 1)
    return out


def next_session_tradable(mask: pd.DataFrame) -> pd.DataFrame:
    """True on date T iff T+1 is tradable (guide: drop next-day limit/suspend)."""
    return mask.shift(-1)


def apply_mask(panel: pd.DataFrame, mask: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mask is None:
        return panel
    aligned = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(aligned == 1)


def name_is_st_or_pt(name: object) -> bool:
    text = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name)
    upper = text.upper()
    return ("ST" in upper) or ("PT" in upper) or ("退" in text)
