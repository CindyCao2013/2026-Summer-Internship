"""Registry + unified compute(panel) interface for cutting factors."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from factor_cutting.active_trade import APM_SPEC, compute_apm_overnight_day_proxy
from factor_cutting.ideal_amplitude import IDEAL_AMPLITUDE_SPEC, compute_ideal_amplitude
from factor_cutting.ideal_reversal import IDEAL_REVERSAL_SPEC, compute_ideal_reversal
from factor_cutting.smart_money import SMART_MONEY_SPEC
from factor_cutting.smart_money_active_v2 import SMART_MONEY_ACTIVE_V2_SPEC
from factor_cutting.apm_active_v2 import APM_ACTIVE_V2_SPEC
from factor_cutting.ideal_reversal_active_v2 import IDEAL_REVERSAL_ACTIVE_V2_SPEC

CUTTING_FACTOR_LIST = [
    "ideal_reversal",
    "ideal_amplitude",
    "apm_overnight_day_proxy",
]

CUTTING_SPECS = {
    "ideal_reversal": IDEAL_REVERSAL_SPEC,
    "ideal_amplitude": IDEAL_AMPLITUDE_SPEC,
    "ideal_reversal_active_v2": IDEAL_REVERSAL_ACTIVE_V2_SPEC,
    "apm": APM_SPEC,
    "apm_active_v2": APM_ACTIVE_V2_SPEC,
    "smart_money": SMART_MONEY_SPEC,
    "smart_money_active_v2": SMART_MONEY_ACTIVE_V2_SPEC,
}


def compute_cutting_factor(
    name: str,
    *,
    close: pd.DataFrame,
    open_: Optional[pd.DataFrame] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    amount: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    trade_count: Optional[pd.DataFrame] = None,
    window: int = 20,
) -> pd.DataFrame:
    """Unified entry: factor.compute(panel) → Date × stock."""
    if name == "ideal_reversal":
        if amount is None:
            raise ValueError("ideal_reversal needs amount")
        ret_1d = close / close.shift(1) - 1.0
        return compute_ideal_reversal(
            ret_1d,
            amount,
            trade_count=trade_count,
            volume=volume,
            window=window,
        )

    if name == "ideal_amplitude":
        if high is None or low is None:
            raise ValueError("ideal_amplitude needs high/low")
        return compute_ideal_amplitude(
            high,
            low,
            close,
            open_=open_,
            window=window,
        )

    if name == "apm_overnight_day_proxy":
        if open_ is None:
            raise ValueError("apm proxy needs open")
        return compute_apm_overnight_day_proxy(open_, close, window=window)

    if name in ("apm", "smart_money"):
        raise NotImplementedError(f"{name} is stubbed until minute/session layer")

    raise KeyError(f"Unknown cutting factor: {name}")


def compute_all_baseline(
    close: pd.DataFrame,
    *,
    open_: Optional[pd.DataFrame] = None,
    high: Optional[pd.DataFrame] = None,
    low: Optional[pd.DataFrame] = None,
    amount: Optional[pd.DataFrame] = None,
    volume: Optional[pd.DataFrame] = None,
    trade_count: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    out = {}
    for name in CUTTING_FACTOR_LIST:
        out[name] = compute_cutting_factor(
            name,
            close=close,
            open_=open_,
            high=high,
            low=low,
            amount=amount,
            volume=volume,
            trade_count=trade_count,
        )
    return out
