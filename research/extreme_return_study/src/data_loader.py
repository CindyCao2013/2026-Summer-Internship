"""Market data + CSI300 index returns for Extreme Return Study."""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Allow importing project-root modules when run as a package under research/
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import Factor_Dev_Lib
import intraday_lib
from factor_data_loaders import connect_ddb, load_eod_wide_tables
from factor_runner import get_universe_mask


@dataclass
class StudyPanels:
    """Aligned panels for the extreme-return study."""

    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    ret_c2c: pd.DataFrame
    ret_o2o: pd.DataFrame
    membership: pd.DataFrame
    df_not_limit: pd.DataFrame
    df_not_st: pd.DataFrame
    df_trade_status: pd.DataFrame
    index_ret: pd.Series
    start: dt.datetime
    end: dt.datetime


def load_csi300_index_return(
    start: dt.datetime,
    end: dt.datetime,
    *,
    index_code: str = "000300.SH",
    session=None,
) -> pd.Series:
    """Load CSI300 close-to-close daily return from Wind AINDEXEODPRICES."""
    own_session = session is None
    if own_session:
        session = connect_ddb()
    try:
        preheat = start - dt.timedelta(days=15)
        t = session.loadTable(dbPath="dfs://WIND.AINDEXEODPRICES", tableName="data")
        t = t.where(
            f"TRADE_DT>={preheat.strftime('%Y.%m.%d')} "
            f"and TRADE_DT<={end.strftime('%Y.%m.%d')} "
            f"and S_INFO_WINDCODE='{index_code}'"
        )
        df = (
            t.select("TRADE_DT as Date, (S_DQ_CLOSE/S_DQ_PRECLOSE-1) as Ret")
            .executeAs("t_csi300_ret")
            .toDF()
        )
        s = df.set_index("Date")["Ret"].sort_index()
        s.index = pd.to_datetime(s.index)
        return s.loc[start:end]
    finally:
        if own_session:
            session.close()


def _open_to_open_return(open_: pd.DataFrame) -> pd.DataFrame:
    """Open-to-open return: open_t / open_{t-1} - 1."""
    return open_ / open_.shift(1) - 1.0


def load_study_panels(
    start: dt.datetime,
    end: dt.datetime,
    *,
    index_code: str = "000300.SH",
    preheat_calendar_days: int = 30,
) -> StudyPanels:
    """
    Load EOD OHLCV, dynamic CSI300 membership, tradability masks, index return.

    Membership uses historical daily weight table (no survivorship bias).
    """
    preheat = start - dt.timedelta(days=preheat_calendar_days)
    eod, session = load_eod_wide_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    membership = get_universe_mask(session, start, end, index_code)
    index_ret = load_csi300_index_return(start, end, index_code=index_code, session=session)

    # Tradability masks (include short preheat for alignment)
    df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(preheat, end)
    df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(preheat, end)
    df_trade_status = Factor_Dev_Lib.get_TradeStatus(preheat, end)

    close = eod.close.copy()
    open_ = eod.open.copy()
    volume = eod.volume.copy()
    amount = eod.amount.copy()

    ret_c2c = close / close.shift(1) - 1.0
    ret_o2o = _open_to_open_return(open_)

    # Align membership columns to price panel
    membership = membership.reindex(index=close.index, columns=close.columns)

    # Restrict study window (keep one prior day for return calc)
    close = close.loc[preheat:end]
    open_ = open_.loc[preheat:end]
    volume = volume.loc[preheat:end]
    amount = amount.loc[preheat:end]
    ret_c2c = ret_c2c.loc[preheat:end]
    ret_o2o = ret_o2o.loc[preheat:end]

    session.close()

    return StudyPanels(
        close=close,
        open=open_,
        volume=volume,
        amount=amount,
        ret_c2c=ret_c2c,
        ret_o2o=ret_o2o,
        membership=membership,
        df_not_limit=df_not_limit,
        df_not_st=df_not_st,
        df_trade_status=df_trade_status,
        index_ret=index_ret,
        start=start,
        end=end,
    )


def clip_end_to_available(end: dt.datetime, last_available: Optional[pd.Timestamp]) -> dt.datetime:
    """Cap configured end date to last available trading day."""
    if last_available is None or pd.isna(last_available):
        return end
    last = pd.Timestamp(last_available).to_pydatetime()
    return min(end, last)
