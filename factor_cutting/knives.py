"""Knife builders for factor-cutting experiments."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ranked by paper fidelity for ideal reversal W-cut
KNIFE_CANDIDATES = [
    "ats_trade_count",  # amount / trade_count  (paper)
    "ats_volume",  # amount / volume — avg price proxy for ATS
    "avg_price",  # alias of ats_volume
    "amount",
    "volume",
    "turnover",
    "turnover_proxy",  # amount / float_mktcap when TURN unavailable
    "trade_count",
    "amihud",
    "volatility_state",
]


def build_knife(
    name: str,
    *,
    amount: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    trade_count: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    ret_1d: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return Date × stock knife panel."""
    if name == "avg_price":
        name = "ats_volume"

    if name == "ats_trade_count":
        if amount is None or trade_count is None:
            raise ValueError("ats_trade_count requires amount + trade_count")
        return amount / trade_count.replace(0, np.nan)

    if name == "ats_volume":
        if amount is None or volume is None:
            raise ValueError("ats_volume requires amount + volume")
        return amount / volume.replace(0, np.nan)

    if name == "amount":
        if amount is None:
            raise ValueError("amount knife requires amount")
        return amount.astype(float)

    if name == "volume":
        if volume is None:
            raise ValueError("volume knife requires volume")
        return volume.astype(float)

    if name == "trade_count":
        if trade_count is None:
            raise ValueError("trade_count knife requires trade_count")
        return trade_count.astype(float)

    if name == "turnover":
        if turnover is not None:
            return turnover.astype(float)
        raise ValueError(
            "turnover knife requires explicit turnover panel "
            "(use turnover_proxy for amount/float_mktcap)"
        )

    if name == "turnover_proxy":
        if turnover is not None:
            return turnover.astype(float)
        if amount is None or float_mktcap is None:
            raise ValueError("turnover_proxy needs amount + float_mktcap (or turnover)")
        return amount / float_mktcap.replace(0, np.nan)

    if name == "amihud":
        if ret_1d is None or amount is None:
            raise ValueError("amihud knife requires ret_1d + amount")
        return ret_1d.abs() / amount.replace(0, np.nan)

    if name == "volatility_state":
        if ret_1d is None:
            raise ValueError("volatility_state requires ret_1d")
        return ret_1d.rolling(20, min_periods=10).std()

    raise KeyError(f"Unknown knife: {name}")


def available_knives(
    *,
    amount: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    trade_count: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    ret_1d: Optional[pd.DataFrame] = None,
    float_mktcap: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Build all knives that can be constructed from available panels."""
    out: Dict[str, pd.DataFrame] = {}
    for name in KNIFE_CANDIDATES:
        if name == "avg_price":
            continue  # alias only
        try:
            out[name] = build_knife(
                name,
                amount=amount,
                volume=volume,
                trade_count=trade_count,
                turnover=turnover,
                ret_1d=ret_1d,
                float_mktcap=float_mktcap,
            )
        except (ValueError, KeyError):
            continue
    return out
