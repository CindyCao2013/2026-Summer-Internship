"""Unified panel reader — the only data-layer entry for factor research.

Factors and backtests should call ``get_daily_panel`` / ``get_minute_panel``
instead of ad-hoc loaders or parquet paths.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from core.ddb.eod import DAILY_OHLCV_COLUMNS, fetch_eod_long
from core.ddb.minute import MINUTE_OHLCV_COLUMNS, fetch_minute_long

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]


def get_daily_panel(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
    session=None,
) -> pd.DataFrame:
    """Daily OHLCV long table from DDB (``date``, ``symbol``, OHLCV columns)."""
    return fetch_eod_long(start, end, symbols=symbols, session=session)


def get_minute_panel(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    session=None,
    trading_hours_only: bool = False,
    store=None,
) -> pd.DataFrame:
    """Minute OHLCV long table from DDB (``bartime``, ``symbol``, OHLCV columns)."""
    return fetch_minute_long(
        start,
        end,
        symbols=symbols,
        fields=fields,
        session=session,
        trading_hours_only=trading_hours_only,
        store=store,
    )


# Re-export column contracts for downstream tests / factors.
DAILY_PANEL_COLUMNS = DAILY_OHLCV_COLUMNS
MINUTE_PANEL_COLUMNS = MINUTE_OHLCV_COLUMNS
