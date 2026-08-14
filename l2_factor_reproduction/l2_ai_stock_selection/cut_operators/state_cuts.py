"""State-conditional cut operators. Within-stock-day thresholds only in V1."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    STATE_DEFINITIONS,
)

LARGE_ORDER_CNY = 200_000.0  # production order-size bucket; not used as a cut threshold


def _as_float(x: Sequence) -> np.ndarray:
    return np.asarray(x, dtype=float)


def within_day_median_mask(
    values: Sequence,
    *,
    high: bool = True,
) -> np.ndarray:
    """high: value > stock-day median; low: value <= median. NaNs stay False."""
    arr = _as_float(values)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros(arr.shape, dtype=bool)
    med = float(np.nanmedian(arr))
    if high:
        return finite & (arr > med)
    return finite & (arr <= med)


def volatility_state_mask(minute_return: Sequence, *, high: bool) -> np.ndarray:
    return within_day_median_mask(np.abs(_as_float(minute_return)), high=high)


def spread_state_mask(relative_spread: Sequence, *, high: bool) -> np.ndarray:
    return within_day_median_mask(relative_spread, high=high)


def depth_state_mask(total_depth: Sequence, *, high: bool) -> np.ndarray:
    return within_day_median_mask(total_depth, high=high)


def price_direction_mask(minute_return: Sequence, *, up: bool) -> np.ndarray:
    arr = _as_float(minute_return)
    finite = np.isfinite(arr)
    if up:
        return finite & (arr > 0)
    return finite & (arr < 0)


def trade_intensity_mask(amount: Sequence, *, high: bool) -> np.ndarray:
    return within_day_median_mask(amount, high=high)


def large_order_state_mask(
    large_order_amount: Sequence,
    *,
    dominated: bool,
    threshold: float = None,
) -> np.ndarray:
    """Within-stock-day median split on the (already share-normalized) series.

    ``threshold`` is accepted for backward compatibility but ignored: V1 does
    not use a cross-sectional 200k CNY cut for large-order state.
    """
    del threshold
    return within_day_median_mask(large_order_amount, high=dominated)


def state_mask(state_name: str, panel: pd.DataFrame) -> np.ndarray:
    """Build a within-day state mask from a minute panel for one stock-day.

    Required columns depend on the state. The panel must already be a single
    TradeDate x Symbol slice; grouping is the caller's responsibility.
    """
    name = str(state_name).strip().lower()
    known = {str(r["state_name"]) for r in STATE_DEFINITIONS}
    if name not in known:
        raise KeyError("unknown state {!r}".format(state_name))
    if name == "high_vol":
        return volatility_state_mask(panel["minute_return"], high=True)
    if name == "low_vol":
        return volatility_state_mask(panel["minute_return"], high=False)
    if name == "high_spread":
        return spread_state_mask(panel["relative_spread"], high=True)
    if name == "low_spread":
        return spread_state_mask(panel["relative_spread"], high=False)
    if name == "high_depth":
        return depth_state_mask(panel["total_depth_l5"], high=True)
    if name == "low_depth":
        return depth_state_mask(panel["total_depth_l5"], high=False)
    if name == "price_up":
        return price_direction_mask(panel["minute_return"], up=True)
    if name == "price_down":
        return price_direction_mask(panel["minute_return"], up=False)
    if name == "high_trade_intensity":
        return trade_intensity_mask(panel["amount"], high=True)
    if name == "low_trade_intensity":
        return trade_intensity_mask(panel["amount"], high=False)
    if name == "large_order_dominated":
        return large_order_state_mask(panel["large_order_amount"], dominated=True)
    if name == "ordinary_order_state":
        return large_order_state_mask(panel["large_order_amount"], dominated=False)
    raise KeyError("unhandled state {!r}".format(state_name))


def grouped_state_mask(
    frame: pd.DataFrame,
    state_name: str,
    *,
    date_col: str = "TradeDate",
    symbol_col: str = "Symbol",
) -> np.ndarray:
    """Vectorized within-group state mask. Never uses other days' values."""
    if frame.empty:
        return np.zeros(0, dtype=bool)
    out = pd.Series(False, index=frame.index)
    grouped = frame.groupby([date_col, symbol_col], sort=False)
    for _, sl in grouped:
        out.loc[sl.index] = state_mask(state_name, sl)
    return out.to_numpy(dtype=bool)
