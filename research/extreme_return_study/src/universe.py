"""CSI300 dynamic universe + tradability filters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alpha_investability import apply_tradability_mask


def universe_mask(
    membership: pd.DataFrame,
    ret: pd.DataFrame,
) -> pd.DataFrame:
    """
    Boolean-like mask (1/NaN): in CSI300 and has valid daily return.

    membership is typically 1 inside index, NaN outside.
    """
    m = membership.reindex_like(ret)
    valid = ret.notna() & (m == 1)
    return valid.astype(float).where(valid, np.nan)


def apply_universe_and_tradability(
    signal: pd.DataFrame,
    *,
    membership: pd.DataFrame,
    df_not_limit: pd.DataFrame | None = None,
    df_not_st: pd.DataFrame | None = None,
    df_trade_status: pd.DataFrame | None = None,
    close: pd.DataFrame | None = None,
    min_listing_days: int = 60,
) -> pd.DataFrame:
    """
    Restrict signal to CSI300 + tradable names.

    Tradability (A-share):
      - not limit-up / limit-down
      - not ST
      - not suspended
      - IPO seasoning (>= min_listing_days of observed closes)
    """
    out = signal.mul(universe_mask(membership, signal))
    out, _ = apply_tradability_mask(
        out,
        df_not_limit=df_not_limit,
        df_not_st=df_not_st,
        df_trade_status=df_trade_status,
        min_listing_days=min_listing_days,
        close=close,
    )
    return out


def daily_universe_size(membership: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Number of CSI300 names with valid return each day."""
    m = universe_mask(membership, ret)
    return m.notna().sum(axis=1)
