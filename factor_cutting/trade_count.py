"""Daily trade-count panel for ideal-reversal ATS knife.

Source: DolphinDB Stock_one_minute Active_buy_count + Active_sell_count (daily sum).
This is a *daily aggregate* of minute counts — not a minute factor. Separate cache
from existing L2 flow parquet so we do not invalidate prior research caches.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from factor_data_loaders import connect_ddb
from l2_data_loaders import pivot_l2_metric

TRADE_COUNT_CACHE_DIR = Path("research/cache/factor_cutting")


def _cache_path(start: dt.datetime, end: dt.datetime) -> Path:
    TRADE_COUNT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return (
        TRADE_COUNT_CACHE_DIR
        / f"trade_count_daily_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
    )


def _ddb_trade_count_script(start: dt.datetime, end: dt.datetime) -> str:
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
select Symbol, Date,
    sum(Active_buy_count) + sum(Active_sell_count) as trade_count,
    sum(Amount) as amount
from t
where Date >= {s} and Date <= {e}
group by Symbol, Date
"""


def load_trade_count_daily(
    start: dt.datetime,
    end: dt.datetime,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Wide Date × Symbol panel of daily trade counts (active buy+sell prints)."""
    cache = _cache_path(start, end)
    if use_cache and cache.exists() and not refresh_cache:
        long_df = pd.read_parquet(cache)
    else:
        own = session is None
        s = session or connect_ddb()
        long_df = s.run(_ddb_trade_count_script(start, end))
        if own:
            s.close()
        long_df = long_df.copy()
        long_df["Date"] = pd.to_datetime(long_df["Date"])
        long_df.to_parquet(cache, index=False)
    return pivot_l2_metric(long_df, "trade_count")
