"""Shared Wind panels for the normalized trade-amount single-factor study.

The helpers intentionally return raw daily panels.  Factor construction,
point-in-time universe masking, and residualization remain separate steps so
that the data lineage is explicit.
"""

from __future__ import annotations

from typing import Iterable, Optional

import dolphindb as ddb
import pandas as pd

from COMMON_CONST import DATA_DB_CONN
from Factor_Dev_Lib import get_preheat_ind_data_citics


def _date_literal(value) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def _wind_codes_literal(symbols: Optional[Iterable[str]]) -> str:
    if symbols is None:
        return ""
    values = sorted({str(symbol) for symbol in symbols})
    if not values:
        return " and 1=0"
    quoted = ",".join(f'"{value}"' for value in values)
    return f" and S_INFO_WINDCODE in [{quoted}]"


def load_market_cap_wide(
    start,
    end,
    symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load Wind total market cap ``S_VAL_MV`` with latest-record semantics."""
    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    symbol_filter = _wind_codes_literal(symbols)
    try:
        frame = session.run(
            f"""
            select S_INFO_WINDCODE, TRADE_DT, S_VAL_MV
            from loadTable('dfs://WIND.ASHAREEODDERIVATIVEINDICATOR', 'data')
            where TRADE_DT >= {_date_literal(start)}
              and TRADE_DT <= {_date_literal(end)}
              {symbol_filter}
            context by TRADE_DT, S_INFO_WINDCODE
            csort OPDATE
            limit 1
            """
        )
    finally:
        session.close()
    if frame is None or len(frame) == 0:
        return pd.DataFrame(dtype=float)
    frame = pd.DataFrame(frame)
    frame["S_VAL_MV"] = pd.to_numeric(frame["S_VAL_MV"], errors="coerce")
    wide = frame.pivot_table(
        index="TRADE_DT",
        columns="S_INFO_WINDCODE",
        values="S_VAL_MV",
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.sort_index()


def load_turnover_wide(
    start,
    end,
    symbols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load Wind daily turnover ``S_DQ_TURN`` with latest-record semantics."""
    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    symbol_filter = _wind_codes_literal(symbols)
    try:
        frame = session.run(
            f"""
            select S_INFO_WINDCODE, TRADE_DT, S_DQ_TURN
            from loadTable('dfs://WIND.ASHAREEODDERIVATIVEINDICATOR', 'data')
            where TRADE_DT >= {_date_literal(start)}
              and TRADE_DT <= {_date_literal(end)}
              {symbol_filter}
            context by TRADE_DT, S_INFO_WINDCODE
            csort OPDATE
            limit 1
            """
        )
    finally:
        session.close()
    if frame is None or len(frame) == 0:
        return pd.DataFrame(dtype=float)
    frame = pd.DataFrame(frame)
    frame["S_DQ_TURN"] = pd.to_numeric(frame["S_DQ_TURN"], errors="coerce")
    wide = frame.pivot_table(
        index="TRADE_DT",
        columns="S_INFO_WINDCODE",
        values="S_DQ_TURN",
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.sort_index()


def load_citics_level1_wide(start, end) -> pd.DataFrame:
    """Load the project's preheated daily CITICS level-1 classification."""
    frame = get_preheat_ind_data_citics(
        pd.Timestamp(start).to_pydatetime(),
        pd.Timestamp(end).to_pydatetime(),
    )
    if frame is None or len(frame) == 0:
        return pd.DataFrame(dtype=object)
    frame = pd.DataFrame(frame).set_index("TradingDay")
    frame.index = pd.to_datetime(frame.index).normalize()
    return frame.sort_index()

