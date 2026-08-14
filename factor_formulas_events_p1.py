"""P1 event factors — major-holder / insider trades (hold + decay).

Equity incentive blocked: Wind ASHARESTOCKINCENTIVEIMPLEMENT incomplete (~444 rows).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from factor_attribution import cs_zscore
from factor_formulas_sue import (
    apply_daily_decay,
    apply_event_hold,
    neutralize_size_industry,
    _sparse_event_wide,
)
from event_data_p1 import (
    aggregate_daily_holder_signal,
    aggregate_daily_insider_signal,
    load_insider_trade_long,
    load_major_holder_trade_long,
)

P1_FACTOR_LIST = [
    "major_holder_net_increase",
    "major_holder_increase_only",
    "insider_net_buy",
]

HOLD_DAYS = 60  # event holding longer than SUE


def build_p1_event_tables(bundle: dict) -> Dict[str, pd.DataFrame]:
    mjr = bundle["mjr"]
    ins = bundle["insider"]
    net = aggregate_daily_holder_signal(mjr)
    inc = net[net["surprise"] > 0].copy() if not net.empty else net
    insider = aggregate_daily_insider_signal(ins)
    return {
        "major_holder_net_increase": net,
        "major_holder_increase_only": inc,
        "insider_net_buy": insider,
    }


def load_p1_bundle(start, end, *, cache_dir=None, keep_cache=False) -> dict:
    hist = start - pd.Timedelta(days=30)
    # datetime
    import datetime as dt

    if not isinstance(hist, dt.datetime):
        hist = pd.Timestamp(hist).to_pydatetime()
    if not isinstance(start, dt.datetime):
        start = pd.Timestamp(start).to_pydatetime()
    if not isinstance(end, dt.datetime):
        end = pd.Timestamp(end).to_pydatetime()
    mjr = load_major_holder_trade_long(hist, end, cache_dir=cache_dir, keep_cache=keep_cache)
    insider = load_insider_trade_long(hist, end, cache_dir=cache_dir, keep_cache=keep_cache)
    return {"mjr": mjr, "insider": insider}


def build_p1_panels(
    event_tables: Dict[str, pd.DataFrame],
    trade_index: pd.DatetimeIndex,
    columns: pd.Index,
    *,
    mode: str = "hold",
    hold_days: int = HOLD_DAYS,
    half_life: int = 10,
) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, ev in event_tables.items():
        raw = _sparse_event_wide(ev, trade_index, columns, "surprise")
        if mode == "decay":
            panel = apply_daily_decay(raw, half_life=half_life, horizon=hold_days)
        else:
            panel = apply_event_hold(raw, hold_days=hold_days)
        out[name] = cs_zscore(panel)
    return out
