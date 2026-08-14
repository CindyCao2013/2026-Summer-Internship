"""Event / value-cut operators. Within-stock-day or causal-intraday only."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    EVENT_Q_DEFAULT,
)

EXTREME_Z = 2.0
SHOCK_Z = 2.0


def _as_float(x: Sequence) -> np.ndarray:
    return np.asarray(x, dtype=float)


def top_q_mask(values: Sequence, q: float = EVENT_Q_DEFAULT) -> np.ndarray:
    """Top q fraction of finite observations. q is fixed at 0.20 in V1."""
    if abs(float(q) - EVENT_Q_DEFAULT) > 1e-12:
        raise ValueError("V1 forbids q grid search; q must be {}".format(EVENT_Q_DEFAULT))
    arr = _as_float(values)
    finite = np.isfinite(arr)
    n = int(finite.sum())
    out = np.zeros(arr.shape, dtype=bool)
    if n == 0:
        return out
    thresh = np.nanquantile(arr, 1.0 - float(q))
    out[finite] = arr[finite] >= thresh
    return out


def bottom_q_mask(values: Sequence, q: float = EVENT_Q_DEFAULT) -> np.ndarray:
    if abs(float(q) - EVENT_Q_DEFAULT) > 1e-12:
        raise ValueError("V1 forbids q grid search; q must be {}".format(EVENT_Q_DEFAULT))
    arr = _as_float(values)
    finite = np.isfinite(arr)
    n = int(finite.sum())
    out = np.zeros(arr.shape, dtype=bool)
    if n == 0:
        return out
    thresh = np.nanquantile(arr, float(q))
    out[finite] = arr[finite] <= thresh
    return out


def extreme_zscore_mask(
    values: Sequence,
    *,
    z_threshold: float = EXTREME_Z,
) -> np.ndarray:
    """|zscore(X)| above threshold using the same stock-day mean/std."""
    arr = _as_float(values)
    finite = np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=bool)
    if finite.sum() < 3:
        return out
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return out
    z = np.abs((arr - mu) / sd)
    out[finite] = z[finite] > float(z_threshold)
    return out


def shock_mask(
    values: Sequence,
    *,
    z_threshold: float = SHOCK_Z,
    min_history: int = 5,
) -> np.ndarray:
    """Causal shock vs prior-intraday expanding mean/std. No future minutes."""
    arr = _as_float(values)
    n = arr.size
    out = np.zeros(n, dtype=bool)
    csum = 0.0
    csumsq = 0.0
    cnt = 0
    for i in range(n):
        if cnt >= int(min_history):
            mu = csum / cnt
            var = max(csumsq / cnt - mu * mu, 0.0)
            sd = var ** 0.5
            xi = arr[i]
            if np.isfinite(xi) and sd > 0 and xi > mu + float(z_threshold) * sd:
                out[i] = True
        xi = arr[i]
        if np.isfinite(xi):
            csum += float(xi)
            csumsq += float(xi) * float(xi)
            cnt += 1
    return out


def large_trade_event_mask(
    large_order_amount: Sequence,
    *,
    threshold: float = None,
    q: float = EVENT_Q_DEFAULT,
) -> np.ndarray:
    """Top-q minutes of the share-normalized large-order series within one stock-day.

    Absolute CNY ``threshold`` is ignored. q is frozen at EVENT_Q_DEFAULT.
    """
    del threshold
    return top_q_mask(large_order_amount, q=q)


def liquidity_shock_mask(
    *,
    relative_spread: Sequence,
    total_depth: Sequence,
    impact: Sequence = (),
    q: float = EVENT_Q_DEFAULT,
) -> np.ndarray:
    if abs(float(q) - EVENT_Q_DEFAULT) > 1e-12:
        raise ValueError("V1 forbids q grid search; q must be {}".format(EVENT_Q_DEFAULT))
    spread_hi = top_q_mask(relative_spread, q=q)
    depth_lo = bottom_q_mask(total_depth, q=q)
    if len(impact) == 0:
        return spread_hi | depth_lo
    impact_hi = top_q_mask(np.abs(_as_float(impact)), q=q)
    return spread_hi | depth_lo | impact_hi


def event_mask(event_name: str, panel: pd.DataFrame, column: str) -> np.ndarray:
    name = str(event_name).strip().upper()
    if name == "TOP_Q":
        return top_q_mask(panel[column])
    if name == "BOTTOM_Q":
        return bottom_q_mask(panel[column])
    if name == "EXTREME":
        return extreme_zscore_mask(panel[column])
    if name == "SHOCK":
        return shock_mask(panel[column])
    if name == "LARGE_TRADE_EVENT":
        col = "large_order_amount" if "large_order_amount" in panel.columns else column
        return large_trade_event_mask(panel[col])
    if name == "LIQUIDITY_SHOCK":
        impact = panel["impact"] if "impact" in panel.columns else ()
        return liquidity_shock_mask(
            relative_spread=panel["relative_spread"],
            total_depth=panel["total_depth_l5"],
            impact=impact,
        )
    raise KeyError("unknown event {!r}".format(event_name))
