"""Shared daily→weekly hold helpers for L2 ActiveV2 factors."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Literal

import numpy as np
import pandas as pd

WeeklyMethod = Literal["friday", "every_5d", "week_mean"]


def to_weekly_hold(
    factor: pd.DataFrame,
    method: WeeklyMethod = "friday",
) -> pd.DataFrame:
    """Downsample daily factor to weekly rebalance (hold between rebalance days)."""
    if factor.empty:
        return factor
    fac = factor.sort_index().astype(float)

    if method == "friday":
        from execution_layer import friday_rebalance_signal

        return friday_rebalance_signal(fac)

    if method == "every_5d":
        from execution_layer import downsample_signal

        return downsample_signal(fac, 5)

    if method == "week_mean":
        weekly = fac.resample("W-FRI").mean()
        return weekly.reindex(fac.index, method="ffill")

    raise ValueError(f"Unknown weekly method: {method}")


def to_weekly_thu_hold(
    factor_daily: pd.DataFrame,
    *,
    agg: Literal["mean", "last"] = "mean",
) -> pd.DataFrame:
    """Thursday-signal weekly hold on a daily calendar.

    For each ISO week, aggregate Mon–Thu daily factor (``mean`` or ``last``),
    place the signal on Thursday (or last Mon–Thu session if no Thursday),
    then forward-fill until the next signal.
    """
    if factor_daily.empty:
        return factor_daily
    fac = factor_daily.sort_index().astype(float)
    if not isinstance(fac.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    out = pd.DataFrame(np.nan, index=fac.index, columns=fac.columns, dtype=float)
    iso = fac.index.isocalendar()
    years = iso.year.to_numpy(dtype=int)
    weeks = iso.week.to_numpy(dtype=int)
    keys = list(zip(years.tolist(), weeks.tolist()))

    mid_pos: Dict[tuple, list] = defaultdict(list)
    for i, (dt, key) in enumerate(zip(fac.index, keys)):
        if int(dt.weekday()) <= 3:
            mid_pos[key].append(i)

    for _key, positions in mid_pos.items():
        block = fac.iloc[positions]
        if agg == "mean":
            sig = block.mean(axis=0)
        elif agg == "last":
            sig = block.iloc[-1]
        else:
            raise ValueError(f"Unknown agg: {agg}")

        thu_pos = [p for p in positions if int(fac.index[p].weekday()) == 3]
        assign_i = thu_pos[-1] if thu_pos else positions[-1]
        out.iloc[assign_i] = sig.reindex(out.columns).to_numpy(dtype=float)

    return out.ffill()
