"""EOD OHLCV reads from WIND distributed tables (long format)."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from core.ddb.connection import get_ddb_session, is_shared_session

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

DDB_EOD_DB = "dfs://WIND.ASHAREEODPRICES"
DDB_EOD_TABLE = "data"

DAILY_OHLCV_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume", "amount")


def _to_ts(value: DateLike) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _filter_a_share_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sym = df["symbol"].astype(str)
    return df.loc[sym.str[0].isin(("0", "3", "6"))].copy()


def build_eod_long_script(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
) -> str:
    s = _to_ts(start).strftime("%Y.%m.%d")
    e = _to_ts(end).strftime("%Y.%m.%d")
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{sym}"' for sym in symbols)
        sym_filter = f"\n  and S_INFO_WINDCODE in ({sym_str})"
    return f"""
t = loadTable('{DDB_EOD_DB}', '{DDB_EOD_TABLE}')
result = select
    TRADE_DT as date,
    S_INFO_WINDCODE as symbol,
    S_DQ_OPEN as open,
    S_DQ_HIGH as high,
    S_DQ_LOW as low,
    S_DQ_CLOSE as close,
    S_DQ_VOLUME as volume,
    S_DQ_AMOUNT as amount
from t
where TRADE_DT between {s} : {e}{sym_filter}
select * from result
"""


def fetch_eod_long(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
    session=None,
) -> pd.DataFrame:
    """Return long-format daily OHLCV (no local parquet)."""
    own = session is None
    s = session or get_ddb_session(reuse=True)
    try:
        raw = s.run(build_eod_long_script(start, end, symbols=symbols))
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=list(DAILY_OHLCV_COLUMNS))
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["symbol"] = df["symbol"].astype(str)
        df = _filter_a_share_long(df)
        keep = [c for c in DAILY_OHLCV_COLUMNS if c in df.columns]
        return df[keep].sort_values(["symbol", "date"]).reset_index(drop=True)
    finally:
        if own and not is_shared_session(s):
            s.close()
