"""High-frequency realized skewness (P1 extension — not in P0 delivery).

Intended definition (when minute bars are available):

    For each day d, RSKEW_{i,d} = Skew(r_{i,1}, ..., r_{i,M})
    RSKEW20_{i,t} = MA_20(RSKEW_{i,d})

P0 deliberately skips minute RSKEW to keep the delivery focused on the
classic daily / idiosyncratic skew anomaly and its interaction with TGD20.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def daily_realized_skew_from_minute_returns(
    minute_ret: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    ret_col: str = "ret",
    min_bars: int = 30,
) -> pd.DataFrame:
    """Long minute returns → wide daily realized-skew panel.

    Parameters
    ----------
    minute_ret:
        Long-form minute returns with date / symbol / ret columns.
    """
    need = {date_col, symbol_col, ret_col}
    missing = need - set(minute_ret.columns)
    if missing:
        raise ValueError(f"daily_realized_skew missing columns: {sorted(missing)}")

    def _skew(s: pd.Series) -> float:
        x = s.dropna()
        if len(x) < min_bars:
            return float("nan")
        return float(x.skew())

    daily = (
        minute_ret.groupby([date_col, symbol_col], sort=False)[ret_col]
        .apply(_skew)
        .rename("RSKEW")
        .reset_index()
    )
    wide = daily.pivot(index=date_col, columns=symbol_col, values="RSKEW")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def realized_skew_20(
    daily_rskew: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """MA of daily realized skew (RSKEW20)."""
    mp = window if min_periods is None else int(min_periods)
    return daily_rskew.rolling(window, min_periods=mp).mean()
