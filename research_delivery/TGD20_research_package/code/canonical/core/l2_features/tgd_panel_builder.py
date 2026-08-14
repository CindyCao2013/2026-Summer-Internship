"""Build TGD20 wide panel from L2 minute bars (Stage 4 infrastructure).

Does NOT change Stage-3 formulas in tgd.py. Orchestrates:

  minute Close  →  Gu/Gd + Rū/Rd̄ + R1/R2   (DDB daily aggregate)
  EOD open/close → overnight_return
  Stage 2        → epsilon_u / epsilon_d
  Stage 3        → TGD20 wide panel

Caches daily timing features under research/cache/tgd_timing_daily/.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.l2_features.tgd import build_tgd20, tgd20_to_wide
from core.l2_features.timing_residual import residualize_timing_centers
from factor_data_loaders import connect_ddb

TIMING_CACHE_DIR = Path("research/cache/tgd_timing_daily")
TGD_PANEL_CACHE_DIR = Path("research/cache/tgd_panels")


def _filter_a_share(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    sym = df[symbol_col].astype(str)
    return df.loc[sym.str[0].isin(("6", "0", "3"))].copy()


def _month_starts(start: dt.datetime, end: dt.datetime) -> list[Tuple[dt.datetime, dt.datetime]]:
    chunks = []
    cur = dt.datetime(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = dt.datetime(cur.year + 1, 1, 1)
        else:
            nxt = dt.datetime(cur.year, cur.month + 1, 1)
        c0 = max(cur, start)
        c1 = min(nxt - dt.timedelta(days=1), end)
        if c0 <= c1:
            chunks.append((c0, c1))
        cur = nxt
    return chunks


def _ddb_timing_daily_script(start: dt.datetime, end: dt.datetime) -> str:
    """Server-side Gu/Gd + conditional means + R1/R2 (matches Python primitives)."""
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
m = select Symbol, Date, second(Bartime) as Bartime, Close
from t
where Date >= {s} and Date <= {e}
  and ((second(Bartime) >= 09:31:00 and second(Bartime) <= 11:30:00)
    or (second(Bartime) >= 13:01:00 and second(Bartime) <= 15:00:00))
m = select Symbol, Date, Bartime, Close,
  iif(minute(Bartime) between 09:31m:11:30m,
      minute(Bartime)-09:31m,
      120+(minute(Bartime)-13:01m)) as t_idx,
  ratios(Close)-1 as r
from m
context by Symbol, Date csort Bartime
select Symbol, Date,
  iif(sum(iif(r>0, r, 0.0)) > 0,
      sum(iif(r>0, t_idx*r, 0.0)) / sum(iif(r>0, r, 0.0)),
      double(NULL)) as Gu,
  iif(sum(iif(r<0, -r, 0.0)) > 0,
      sum(iif(r<0, t_idx*(-r), 0.0)) / sum(iif(r<0, -r, 0.0)),
      double(NULL)) as Gd,
  avg(iif(r>0, r, NULL)) as avg_up_return,
  avg(iif(r<0, r, NULL)) as avg_down_return,
  sum(iif(isValid(r) and r==0, 1, 0)) as zero_return_count,
  sum(iif(isValid(r) and r>0, 1, 0)) as n_up,
  sum(iif(isValid(r) and r<0, 1, 0)) as n_down,
  sum(iif(isValid(r) and t_idx<=29, r, 0.0)) as R1,
  sum(iif(isValid(r) and t_idx>29 and t_idx<=59, r, 0.0)) as R2
from m
group by Symbol, Date
"""


def _month_cache_path(start: dt.datetime, end: dt.datetime) -> Path:
    TIMING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TIMING_CACHE_DIR / f"timing_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"


def load_timing_daily_features(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Long daily features: date, symbol, Gu, Gd, avg_up/down, R1, R2, zero_count."""
    chunks = _month_starts(start, end)
    parts = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, c1 in chunks:
            path = _month_cache_path(c0, c1)
            if use_cache and path.exists() and not refresh_cache:
                parts.append(pd.read_parquet(path))
                continue
            df = s.run(_ddb_timing_daily_script(c0, c1))
            df = df.rename(
                columns={
                    "Symbol": "symbol",
                    "Date": "date",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            df = _filter_a_share(df)
            df.to_parquet(path, index=False)
            parts.append(df)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "Gu",
                "Gd",
                "avg_up_return",
                "avg_down_return",
                "zero_return_count",
                "n_up",
                "n_down",
                "R1",
                "R2",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def overnight_return_long(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """overnight_return_t = open_t / close_{t-1} - 1 → long [date, symbol, overnight_return]."""
    open_ = open_.sort_index()
    close = close.reindex_like(open_)
    prev_close = close.shift(1)
    ovn = open_ / prev_close - 1.0
    long = ovn.stack(dropna=False).rename("overnight_return").reset_index()
    long.columns = ["date", "symbol", "overnight_return"]
    long["date"] = pd.to_datetime(long["date"])
    long["symbol"] = long["symbol"].astype(str)
    return long


def assemble_residual_inputs(
    timing: pd.DataFrame,
    overnight: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Stage-1/2.5/session controls into residualize_timing_centers input."""
    left = timing.copy()
    left["date"] = pd.to_datetime(left["date"])
    right = overnight.copy()
    right["date"] = pd.to_datetime(right["date"])
    return left.merge(right, on=["date", "symbol"], how="left")


def build_tgd20_wide_from_eod_l2(
    start: dt.datetime,
    end: dt.datetime,
    *,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 60,
    window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: L2 + EOD → residual long → TGD20 wide.

    Returns (tgd_wide, residual_long_with_tgd).
    Signal formed with full-day minutes of date T — caller must shift(1) for T+1 ret.
    """
    TGD_PANEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wide_path = (
        TGD_PANEL_CACHE_DIR
        / f"TGD20_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_w{window}.parquet"
    )
    long_path = (
        TGD_PANEL_CACHE_DIR
        / f"TGD20_long_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_w{window}.parquet"
    )
    if use_cache and wide_path.exists() and long_path.exists() and not refresh_cache:
        wide = pd.read_parquet(wide_path)
        wide.index = pd.to_datetime(wide.index)
        long = pd.read_parquet(long_path)
        return wide, long

    load_start = start - dt.timedelta(days=preheat_calendar_days)
    timing = load_timing_daily_features(
        load_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    ovn = overnight_return_long(
        open_.loc[load_start:end],
        close.loc[load_start:end],
    )
    daily = assemble_residual_inputs(timing, ovn)
    residual = residualize_timing_centers(daily)
    tgd_long = build_tgd20(residual, window=window)
    # trim to requested window after MA warm-up
    tgd_long = tgd_long[tgd_long["date"] >= pd.Timestamp(start)].copy()
    wide = tgd20_to_wide(tgd_long)
    wide = wide.loc[start:end]
    if use_cache:
        wide.to_parquet(wide_path)
        tgd_long.to_parquet(long_path, index=False)
    return wide, tgd_long
