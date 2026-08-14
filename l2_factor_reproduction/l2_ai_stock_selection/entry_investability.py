"""T vs T+1 investability. The trade occurs on T+1."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import MIN_LISTING_DAYS


def listing_age_ok(price: pd.DataFrame, *, min_days: int = MIN_LISTING_DAYS) -> pd.DataFrame:
    listed = price.notna() & (price.astype(float) > 0)
    age = listed.astype(float).cumsum()
    return (age >= int(min_days)).astype(float).where(listed, np.nan)


def build_entry_tradable(
    *,
    dates: pd.DatetimeIndex,
    universe_mask_t: pd.DataFrame,
    adj_vwap: pd.DataFrame,
    trade_status_t1: Optional[pd.DataFrame] = None,
    not_limit_t1: Optional[pd.DataFrame] = None,
    min_listing_days: int = MIN_LISTING_DAYS,
) -> Dict[str, pd.DataFrame]:
    """signal_tradable_T on feature date; entry_tradable_T1 on the next session.

    Rules (no future beyond T+1 execution facts):
    - T+1 not suspended
    - T+1 adj VWAP exists and > 0
    - T+1 close not at limit (available proxy for inability to enter)
    - listing-age: cumsum of finite VWAP >= 60 by T+1
    CSI1000 membership is NOT required.
    """
    dates = pd.DatetimeIndex(dates).normalize()
    mask_t = universe_mask_t.reindex(index=dates)
    vwap = adj_vwap.reindex(index=dates)
    vwap_ok = (vwap.astype(float) > 0) & np.isfinite(vwap.astype(float))
    age = listing_age_ok(vwap, min_days=min_listing_days)
    # T+1 fields aligned onto feature date T via shift(-1): row T gets T+1 values.
    vwap_t1 = vwap_ok.shift(-1)
    age_t1 = age.shift(-1)
    ts_t1 = (
        trade_status_t1.reindex(index=dates).shift(-1)
        if trade_status_t1 is not None
        else pd.DataFrame(1.0, index=dates, columns=vwap.columns)
    )
    nl_t1 = (
        not_limit_t1.reindex(index=dates).shift(-1)
        if not_limit_t1 is not None
        else pd.DataFrame(1.0, index=dates, columns=vwap.columns)
    )
    entry = (
        (ts_t1 == 1)
        & (nl_t1 == 1)
        & vwap_t1.fillna(False)
        & (age_t1 == 1)
    )
    entry_f = entry.astype(float).where(entry, np.nan)
    signal_t = (mask_t == 1).astype(float).where(mask_t == 1, np.nan)
    return {
        "signal_tradable_T": signal_t,
        "entry_tradable_T1": entry_f.reindex(index=dates, columns=mask_t.columns),
    }


def tradability_audit(
    signal_t: pd.DataFrame,
    entry_t1: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dt in signal_t.index:
        s = signal_t.loc[dt]
        e = entry_t1.loc[dt] if dt in entry_t1.index else pd.Series(dtype=float)
        n_sig = int((s == 1).sum()) if s is not None else 0
        n_ent = int((e == 1).sum()) if len(e) else 0
        both = int(((s == 1) & (e.reindex(s.index) == 1)).sum()) if n_sig else 0
        rows.append(
            {
                "feature_date": pd.Timestamp(dt).normalize(),
                "n_signal_tradable_T": n_sig,
                "n_entry_tradable_T1": n_ent,
                "n_both": both,
                "entry_over_signal": float(n_ent / n_sig) if n_sig else float("nan"),
            }
        )
    return pd.DataFrame(rows)
