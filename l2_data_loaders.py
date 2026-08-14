"""L2 data loaders — tick / order book → daily aggregate (not intraday factor zoo).

Primary source: DolphinDB `dfs://QV_Trade_to_MinuteBar.Stock_one_minute`
  (minute bars derived from exchange tick + order-flow labels).

Daily bricks cached under research/cache/l2_daily/ for reuse by validation + backtest.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from factor_data_loaders import connect_ddb

L2_CACHE_DIR = Path("research/cache/l2_daily")
IMBALANCE_VOI_THRESHOLD = 0.10  # minute-level VOI above this → "buy pressure" minute


@dataclass
class L2DailyLongTable:
    """Long-format daily order-flow aggregates (Symbol × Date)."""

    data: pd.DataFrame

    @property
    def columns(self):
        return self.data.columns.tolist()


def _cache_path(start: dt.datetime, end: dt.datetime) -> Path:
    L2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return L2_CACHE_DIR / f"l2_daily_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"


def _ddb_daily_aggregate_script(start: dt.datetime, end: dt.datetime) -> str:
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
select Symbol, Date,
    sum(Active_buy_volume) as active_buy_vol,
    sum(Active_sell_volume) as active_sell_vol,
    sum(Active_buy_amount) as active_buy_amt,
    sum(Active_sell_amount) as active_sell_amt,
    sum(Bid_cancel_volume) as bid_cancel_vol,
    sum(Ask_cancel_volume) as ask_cancel_vol,
    sum(Volume) as volume,
    sum(Amount) as amount
from t
where Date >= {s} and Date <= {e}
group by Symbol, Date
"""


def load_l2_daily_long(
    start: dt.datetime,
    end: dt.datetime,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> L2DailyLongTable:
    """Load daily L2 aggregates from minute bar table (server-side group by)."""
    cache = _cache_path(start, end)
    if use_cache and cache.exists() and not refresh_cache:
        df = pd.read_parquet(cache)
        return L2DailyLongTable(df)

    own = session is None
    s = session or connect_ddb()
    df = s.run(_ddb_daily_aggregate_script(start, end))
    if own:
        s.close()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_parquet(cache, index=False)
    return L2DailyLongTable(df)


def _imbalance_duration_cache_path(start: dt.datetime, end: dt.datetime) -> Path:
    L2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return L2_CACHE_DIR / f"l2_imbalance_duration_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"


def _ddb_imbalance_duration_script(start: dt.datetime, end: dt.datetime) -> str:
    """Daily fraction of minutes with minute-VOI above threshold (duration proxy)."""
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    thr = IMBALANCE_VOI_THRESHOLD
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
m = select Symbol, Date,
    iif(Active_buy_volume + Active_sell_volume > 0,
        (Active_buy_volume - Active_sell_volume) \\ (Active_buy_volume + Active_sell_volume),
        double(NULL)) as m_voi
    from t
    where Date >= {s} and Date <= {e}
select Symbol, Date,
    sum(iif(m_voi > {thr}, 1, 0)) * 1.0 \\ count(*) as imbalance_duration
from m
group by Symbol, Date
"""


def load_imbalance_duration_daily(
    start: dt.datetime,
    end: dt.datetime,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    cache = _imbalance_duration_cache_path(start, end)
    if use_cache and cache.exists() and not refresh_cache:
        long_df = pd.read_parquet(cache)
    else:
        own = session is None
        s = session or connect_ddb()
        long_df = s.run(_ddb_imbalance_duration_script(start, end))
        if own:
            s.close()
        long_df = long_df.copy()
        long_df["Date"] = pd.to_datetime(long_df["Date"])
        long_df.to_parquet(cache, index=False)
    return pivot_l2_metric(long_df, "imbalance_duration")


def pivot_l2_metric(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot long L2 table to Date × Symbol wide panel."""
    wide = long_df.pivot(index="Date", columns="Symbol", values=value_col)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    selected = [c for c in wide.columns if str(c)[0] in ("0", "3", "6")]
    return wide[selected]


def load_l2_daily_wide_panels(
    start: dt.datetime,
    end: dt.datetime,
    session=None,
    use_cache: bool = True,
) -> dict:
    """Return dict of daily wide panels keyed by raw metric name."""
    long_tbl = load_l2_daily_long(start, end, session=session, use_cache=use_cache)
    df = long_tbl.data
    metrics = [
        "active_buy_vol",
        "active_sell_vol",
        "active_buy_amt",
        "active_sell_amt",
        "bid_cancel_vol",
        "ask_cancel_vol",
        "volume",
        "amount",
    ]
    return {m: pivot_l2_metric(df, m) for m in metrics if m in df.columns}


@dataclass
class L2DailyWideCache:
    """Daily wide panels for L2 factor construction."""

    active_buy_vol: pd.DataFrame
    active_sell_vol: pd.DataFrame
    active_buy_amt: pd.DataFrame
    active_sell_amt: pd.DataFrame
    bid_cancel_vol: pd.DataFrame
    ask_cancel_vol: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    imbalance_duration: Optional[pd.DataFrame] = None
    close: Optional[pd.DataFrame] = None


def build_l2_daily_cache(
    start: dt.datetime,
    end: dt.datetime,
    session=None,
    close: Optional[pd.DataFrame] = None,
    use_cache: bool = True,
) -> L2DailyWideCache:
    panels = load_l2_daily_wide_panels(start, end, session=session, use_cache=use_cache)
    imb = load_imbalance_duration_daily(start, end, session=session, use_cache=use_cache)
    return L2DailyWideCache(
        active_buy_vol=panels["active_buy_vol"],
        active_sell_vol=panels["active_sell_vol"],
        active_buy_amt=panels["active_buy_amt"],
        active_sell_amt=panels["active_sell_amt"],
        bid_cancel_vol=panels["bid_cancel_vol"],
        ask_cancel_vol=panels["ask_cancel_vol"],
        volume=panels["volume"],
        amount=panels["amount"],
        imbalance_duration=imb,
        close=close,
    )
