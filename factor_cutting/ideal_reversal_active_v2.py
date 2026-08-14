"""IdealReversal_ActiveV2 — short reversal gated by active-size concentration.

Distinct from Kaiyuan IdealReversal (daily ATS W-cut).

Uses shared brick ``active_size_concentration`` (observable ASC — NOT
\"institutional participation\").
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from factor_cutting.engine import CuttingSpec, KnifeSpec, ObjectSpec, OutputSpec
from core.l2_features.bricks.active_size.concentration import (
    compute_daily_active_size_concentration,
    concentration_one_day,
    smooth_active_size_concentration,
)

REVERSAL_WINDOW = 5
MIN_CS_N = 30
DEFAULT_WINDOWS: List[int] = [3, 5, 10]

ReversalMode = Literal["asc_gate", "pure_rev"]
WeeklyMethod = Literal["friday", "every_5d", "week_mean"]
GateType = Literal["median", "rolling_rank", "none"]

FORMULA_VERSION = "ideal_reversal_active_v2_thu_rollgate_v3"

IDEAL_REVERSAL_ACTIVE_V2_SPEC = CuttingSpec(
    name="ideal_reversal_active_v2",
    paper="理想反转因子2.0（主动大单集中度门控·研究版）",
    direction_paper="negative_ic",
    status="implemented_minute_active_size_gate",
    object=ObjectSpec(variable="daily_return", additive=True),
    knife=KnifeSpec(
        variable="active_size_concentration",
        method="quantile_split",
        window=REVERSAL_WINDOW,
        formula="sum(ActiveBuy | top20% avg_buy_size) / sum(ActiveBuy)",
    ),
    output=OutputSpec(
        op="product",
        formula="-ret_5d * asc_mean | asc_mean > CS_median(ewm ASC)",
    ),
)


def _cs_rank_pct(row: pd.Series) -> pd.Series:
    x = row.astype(float)
    n = int(np.isfinite(x.to_numpy()).sum())
    if n < 2:
        return pd.Series(np.nan, index=x.index)
    return x.rank(method="average", pct=True)


def build_reversal_factor(
    close: pd.DataFrame,
    asc_panel: pd.DataFrame,
    window: int = REVERSAL_WINDOW,
    min_cs_n: int = MIN_CS_N,
    mode: ReversalMode = "asc_gate",
) -> pd.DataFrame:
    """Build daily IdealReversal raw panel (v1 API)."""
    close = close.sort_index().astype(float)
    ret = close.pct_change(window)

    if mode == "pure_rev":
        rev = (-ret).loc[ret.index]
        n_ok = rev.notna().sum(axis=1)
        rev = rev.where(n_ok >= min_cs_n)
        return rev.apply(_cs_rank_pct, axis=1)

    if mode != "asc_gate":
        raise ValueError(f"Unknown reversal mode: {mode}")

    asc = asc_panel.reindex_like(close).astype(float)
    asc_mean = asc.rolling(window, min_periods=max(3, window // 2)).mean()

    common_idx = ret.index.intersection(asc_mean.index)
    ret = ret.loc[common_idx]
    asc_mean = asc_mean.loc[common_idx]

    median_asc = asc_mean.median(axis=1)
    high_asc = asc_mean.gt(median_asc, axis=0)

    factor = (-ret * asc_mean).where(high_asc)
    n_ok = factor.notna().sum(axis=1)
    factor = factor.where(n_ok >= min_cs_n)
    return factor


def build_reversal_factor_v2(
    close: pd.DataFrame,
    asc_panel: pd.DataFrame,
    *,
    windows: Optional[Sequence[int]] = None,
    gate_type: GateType = "rolling_rank",
    gate_threshold: float = 0.6,
    gate_roll: int = 20,
    icir_weights: Optional[Union[pd.Series, Dict[int, float]]] = None,
    min_cs_n: int = MIN_CS_N,
) -> pd.DataFrame:
    """Dynamic ASC gate + multi-window weighted reversal.

    gate_type
    ---------
    rolling_rank: CS ASC rank rolling-mean > threshold AND current rank > 0.5;
                  enhance by rolling rank mean.
    median:       legacy CS median cut; enhance by ASC level.
    none:         no gate / no ASC enhance (pure multi-window reversal).
    """
    wins = list(windows) if windows is not None else list(DEFAULT_WINDOWS)
    if not wins:
        raise ValueError("windows must be non-empty")

    close = close.sort_index().astype(float)
    rets: Dict[int, pd.DataFrame] = {}
    for w in wins:
        rets[w] = -close.pct_change(int(w))

    common_dates = rets[wins[0]].index
    for w in wins[1:]:
        common_dates = common_dates.intersection(rets[w].index)
    common_dates = common_dates[common_dates >= close.index[max(wins)]]
    for w in wins:
        rets[w] = rets[w].loc[common_dates]

    if icir_weights is None:
        weights = pd.Series(1.0, index=wins, dtype=float)
    elif isinstance(icir_weights, dict):
        weights = pd.Series({int(k): float(v) for k, v in icir_weights.items()})
        weights = weights.reindex(wins, fill_value=0.0)
    else:
        weights = icir_weights.reindex(wins, fill_value=0.0).astype(float)

    wsum = float(weights.sum())
    if wsum <= 0 or not np.isfinite(wsum):
        weights = pd.Series(1.0, index=wins, dtype=float)
        wsum = float(len(wins))
    norm_w = weights / wsum

    raw = pd.DataFrame(0.0, index=common_dates, columns=close.columns, dtype=float)
    for w in wins:
        raw = raw.add(norm_w.loc[w] * rets[w], fill_value=0.0)

    if gate_type == "none":
        factor = raw
    elif gate_type == "median":
        asc = asc_panel.reindex(index=common_dates, columns=close.columns).astype(float)
        median_asc = asc.median(axis=1)
        mask = asc.gt(median_asc, axis=0)
        factor = (raw * asc).where(mask)
    elif gate_type == "rolling_rank":
        asc = asc_panel.reindex(index=common_dates, columns=close.columns).astype(float)
        rank_asc = asc.rank(axis=1, pct=True, method="average")
        roll_rank_mean = rank_asc.rolling(
            gate_roll, min_periods=max(3, gate_roll // 2)
        ).mean()
        mask = (roll_rank_mean > gate_threshold) & (rank_asc > 0.5)
        factor = (raw * roll_rank_mean).where(mask)
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")

    n_ok = factor.notna().sum(axis=1)
    return factor.where(n_ok >= min_cs_n)


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


# ---- deprecated aliases ----
_inst_ratio_one_day = concentration_one_day
compute_daily_inst_ratio = compute_daily_active_size_concentration
smooth_inst_ratio = smooth_active_size_concentration
