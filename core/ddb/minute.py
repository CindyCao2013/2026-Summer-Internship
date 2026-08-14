"""Minute-bar reads — delegates to MinuteBarStore (pure DDB, no parquet)."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

MINUTE_OHLCV_COLUMNS = (
    "bartime",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


def fetch_minute_long(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    session=None,
    trading_hours_only: bool = False,
    store=None,
) -> pd.DataFrame:
    """Return canonical minute OHLCV long table (data layer only).

    Normalization is delegated entirely to ``MinuteBarStore``; this module only
    selects the OHLCV column subset.
    """
    from minute_bar_store import MinuteBarStore

    if store is None:
        store = MinuteBarStore(session=session)
    df = store.get_data(
        start,
        end,
        symbols=symbols,
        fields=fields,
        trading_hours_only=trading_hours_only,
    )
    if df.empty:
        return pd.DataFrame(columns=list(MINUTE_OHLCV_COLUMNS))
    keep = [c for c in MINUTE_OHLCV_COLUMNS if c in df.columns]
    return df[keep].sort_values(["symbol", "bartime"]).reset_index(drop=True)
