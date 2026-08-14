"""SmartMoneyActiveV2 panel builder — Active_* large-order concentration.

Cache layout:

  research/cache/smart_money_active_v2/
    minute_raw/minute_YYYYMM.parquet
    daily_brick/daily_YYYYMM.parquet
    factor_panel/SmartMoneyActiveV2_*.parquet

Distinct from research/cache/smart_money/ (Kaiyuan SmartMoney10d).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from factor_cutting.smart_money_active_v2 import (
    EWM_MIN_PERIODS,
    EWM_SPAN,
    FORMULA_VERSION,
    TOP_AMT_PCT,
    apply_minute_qc,
    compute_daily_smart_active_fast,
    ewm_smooth_daily,
)
from factor_data_loaders import connect_ddb

CACHE_ROOT = Path("research/cache/smart_money_active_v2")
MINUTE_RAW_DIR = CACHE_ROOT / "minute_raw"
DAILY_BRICK_DIR = CACHE_ROOT / "daily_brick"
FACTOR_PANEL_DIR = CACHE_ROOT / "factor_panel"


def _filter_a_share(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    sym = df[symbol_col].astype(str)
    return df.loc[sym.str[0].isin(("6", "0", "3"))].copy()


def _month_starts(start: dt.datetime, end: dt.datetime) -> list[Tuple[dt.datetime, dt.datetime]]:
    chunks: list[Tuple[dt.datetime, dt.datetime]] = []
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


def _ym_tag(d: dt.datetime) -> str:
    return d.strftime("%Y%m")


def _ddb_minute_active_script(start: dt.datetime, end: dt.datetime) -> str:
    """Load Active_* + OHLCV + Adjfactor for continuous auction minutes."""
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
select Symbol, Date, second(Bartime) as Bartime,
    Open, High, Low, Close, Volume, Amount, Adjfactor,
    Active_buy_amount, Active_sell_amount,
    Active_buy_count, Active_sell_count,
    Bid_cancel_volume, Ask_cancel_volume
from t
where Date >= {s} and Date <= {e}
  and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
    or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))
"""


def _ddb_daily_smart_active_script(start: dt.datetime, end: dt.datetime) -> str:
    """Server-side top-20% Active_* concentration + day volume/cancel.

    Within-day Adjfactor cancels in the ratio; session filter only.
    Bad-price minute QC is skipped here for throughput (limit/halt masked later).
    """
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
m = select Symbol, Date,
    Active_buy_amount, Active_sell_amount, Volume,
    Bid_cancel_volume, Ask_cancel_volume
from t
where Date >= {s} and Date <= {e}
  and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
    or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))
m = select Symbol, Date, Active_buy_amount, Active_sell_amount, Volume,
    Bid_cancel_volume, Ask_cancel_volume,
    percentile(Active_buy_amount, 80) as thr_buy,
    percentile(Active_sell_amount, 80) as thr_sell
from m
context by Symbol, Date
select Symbol, Date,
    sum(iif(Active_buy_amount >= thr_buy, Active_buy_amount, 0.0))
        \\ sum(Active_buy_amount) as smart_long,
    sum(iif(Active_sell_amount >= thr_sell, Active_sell_amount, 0.0))
        \\ sum(Active_sell_amount) as smart_short,
    sum(Volume) as volume,
    sum(Bid_cancel_volume) as bid_cancel_vol,
    sum(Ask_cancel_volume) as ask_cancel_vol,
    count(*) as n_minutes
from m
group by Symbol, Date
"""


def _normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Date": "date",
            "Bartime": "bartime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
            "Adjfactor": "adjfactor",
            "Active_buy_amount": "active_buy_amt",
            "Active_sell_amount": "active_sell_amt",
            "Active_buy_count": "active_buy_count",
            "Active_sell_count": "active_sell_count",
            "Bid_cancel_volume": "bid_cancel_vol",
            "Ask_cancel_volume": "ask_cancel_vol",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    return _filter_a_share(df)


def load_minute_active_raw(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """L1: monthly Active_* minute bars (session-filtered).

    Fetches via ``MinuteBarStore`` from DolphinDB on demand and applies the
    same continuous-auction filter as the legacy path.
    """
    hist = None
    try:
        import factor_config as cfg

        hist = getattr(cfg, "MINUTE_BAR_HISTORY_START", None)
    except Exception:  # noqa: BLE001
        pass

    from minute_bar_store import MinuteBarStore

    store = MinuteBarStore(
        start_date=hist,
        session=session,
    )
    raw = store.get_data(
        start,
        end,
        force_reload=refresh_cache or (not use_cache),
        trading_hours_only=True,
    )
    if raw.empty:
        return raw
    return raw.sort_values(["symbol", "date", "bartime"]).reset_index(drop=True)


def _daily_volume_and_cancel(minutes: pd.DataFrame) -> pd.DataFrame:
    """Day-level volume sum + cancel volumes (for halt / cancel filter)."""
    agg = {
        "volume": "sum",
        "bid_cancel_vol": "sum",
        "ask_cancel_vol": "sum",
    }
    cols = [c for c in agg if c in minutes.columns]
    if "volume" not in cols:
        return pd.DataFrame(columns=["date", "symbol", "volume"])
    g = minutes.groupby(["symbol", "date"], sort=False)[cols].sum().reset_index()
    return g


def build_daily_brick_from_minutes(
    minutes: pd.DataFrame,
    *,
    top_pct: float = TOP_AMT_PCT,
    require_large_avg_size: bool = False,
    cancel_downweight: bool = False,
    cancel_lookback: int = 20,
    cancel_z: float = 2.0,
) -> pd.DataFrame:
    """QC → daily smart_long/short → optional cancel downweight."""
    qc = apply_minute_qc(minutes)
    day_meta = _daily_volume_and_cancel(qc)
    daily = compute_daily_smart_active_fast(
        qc,
        top_pct=top_pct,
        require_large_avg_size=require_large_avg_size,
    )
    daily = daily.merge(day_meta, on=["symbol", "date"], how="left")
    # halt: volume == 0 → NaN
    if "volume" in daily.columns:
        halted = daily["volume"].fillna(0) <= 0
        daily.loc[halted, ["smart_long", "smart_short"]] = np.nan

    if cancel_downweight and "bid_cancel_vol" in daily.columns:
        daily = daily.sort_values(["symbol", "date"])
        mu = daily.groupby("symbol")["bid_cancel_vol"].transform(
            lambda x: x.rolling(cancel_lookback, min_periods=max(5, cancel_lookback // 2)).mean()
        )
        sd = daily.groupby("symbol")["bid_cancel_vol"].transform(
            lambda x: x.rolling(cancel_lookback, min_periods=max(5, cancel_lookback // 2)).std()
        )
        spike = daily["bid_cancel_vol"] > (mu + cancel_z * sd)
        # downweight: shrink toward 0.5 (neutral concentration) by half
        for col in ("smart_long", "smart_short"):
            daily.loc[spike, col] = 0.5 * daily.loc[spike, col] + 0.5 * 0.5

    return daily


def _normalize_daily_brick(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Date": "date",
            "smart_long": "smart_long",
            "smart_short": "smart_short",
            "Volume": "volume",
            "volume": "volume",
            "bid_cancel_vol": "bid_cancel_vol",
            "ask_cancel_vol": "ask_cancel_vol",
            "Bid_cancel_volume": "bid_cancel_vol",
            "Ask_cancel_volume": "ask_cancel_vol",
            "n_minutes": "n_minutes",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = _filter_a_share(df)
    if "volume" in df.columns:
        halted = df["volume"].fillna(0) <= 0
        df.loc[halted, ["smart_long", "smart_short"]] = np.nan
    # too few continuous-auction minutes → unstable concentration
    if "n_minutes" in df.columns:
        thin = df["n_minutes"].fillna(0) < 30
        df.loc[thin, ["smart_long", "smart_short"]] = np.nan
    return df


def ensure_daily_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    require_large_avg_size: bool = False,
    cancel_downweight: bool = False,
) -> pd.DataFrame:
    """Build / load monthly daily bricks, return concatenated long frame.

    Default path: DolphinDB server-side concentration (fast).
    Enhancement flags fall back to minute → Python knife.
    """
    DAILY_BRICK_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    own = session is None
    s = session or connect_ddb()
    use_ddb_daily = (not require_large_avg_size) and (not cancel_downweight)
    try:
        for c0, _c1 in _month_starts(start, end):
            path = DAILY_BRICK_DIR / f"daily_{_ym_tag(c0)}.parquet"
            if use_cache and path.exists() and not refresh_cache:
                parts.append(pd.read_parquet(path))
                continue
            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)

            if use_ddb_daily:
                print(f"  DDB daily_smart_active {_ym_tag(c0)} ...", flush=True)
                raw = s.run(_ddb_daily_smart_active_script(m0, m1))
                daily = _normalize_daily_brick(raw)
            else:
                minutes = load_minute_active_raw(
                    m0, m1, session=s, use_cache=use_cache, refresh_cache=refresh_cache
                )
                if minutes.empty:
                    continue
                daily = build_daily_brick_from_minutes(
                    minutes,
                    require_large_avg_size=require_large_avg_size,
                    cancel_downweight=cancel_downweight,
                )
            daily.to_parquet(path, index=False)
            print(f"  wrote {path.name} rows={len(daily):,}", flush=True)
            parts.append(daily)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(
            columns=["date", "symbol", "smart_long", "smart_short"]
        )
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def mask_limit_days(
    daily: pd.DataFrame,
    not_limit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Set smart_* to NaN on limit-up/down days (not_limit: 1=ok, NaN=limit)."""
    if not_limit is None or daily.empty:
        return daily
    out = daily.copy()
    nl = not_limit.copy()
    nl.index = pd.to_datetime(nl.index)
    # melt not_limit → long
    long_nl = nl.stack().rename("not_limit").reset_index()
    long_nl.columns = ["date", "symbol", "not_limit"]
    long_nl["date"] = pd.to_datetime(long_nl["date"])
    long_nl["symbol"] = long_nl["symbol"].astype(str)
    merged = out.merge(long_nl, on=["date", "symbol"], how="left")
    bad = merged["not_limit"].isna()
    for col in ("smart_long", "smart_short"):
        if col in merged.columns:
            merged.loc[bad, col] = np.nan
    return merged.drop(columns=["not_limit"], errors="ignore")


def build_smart_money_active_v2_panel(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    span: int = EWM_SPAN,
    min_periods: int = EWM_MIN_PERIODS,
    preheat_calendar_days: int = 40,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    require_large_avg_size: bool = False,
    cancel_downweight: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide + long smart_raw panels.

    Returns (wide, long) where wide index=date, columns=symbol, values=smart_raw.
    """
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    if symbols is not None:
        tag = f"{tag}_n{len(symbols)}"
    if require_large_avg_size:
        tag = f"{tag}_sz"
    if cancel_downweight:
        tag = f"{tag}_cx"
    wide_path = FACTOR_PANEL_DIR / f"SmartMoneyActiveV2_{tag}.parquet"
    long_path = FACTOR_PANEL_DIR / f"SmartMoneyActiveV2_long_{tag}.parquet"
    if use_cache and wide_path.exists() and long_path.exists() and not refresh_cache:
        return pd.read_parquet(wide_path), pd.read_parquet(long_path)

    brick_start = start - dt.timedelta(days=preheat_calendar_days)
    daily = ensure_daily_bricks(
        brick_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        require_large_avg_size=require_large_avg_size,
        cancel_downweight=cancel_downweight,
    )
    if symbols is not None:
        sym_set = set(str(x) for x in symbols)
        daily = daily[daily["symbol"].isin(sym_set)]

    daily = mask_limit_days(daily, not_limit=not_limit)
    smoothed = ewm_smooth_daily(daily, span=span, min_periods=min_periods)
    smoothed = smoothed[
        (smoothed["date"] >= pd.Timestamp(start))
        & (smoothed["date"] <= pd.Timestamp(end))
    ]
    wide = smoothed.pivot(index="date", columns="symbol", values="smart_raw").sort_index()
    wide.to_parquet(wide_path)
    smoothed.to_parquet(long_path, index=False)
    meta = {
        "formula_version": FORMULA_VERSION,
        "top_amt_pct": TOP_AMT_PCT,
        "ewm_span": span,
        "ewm_min_periods": min_periods,
        "require_large_avg_size": require_large_avg_size,
        "cancel_downweight": cancel_downweight,
        "n_dates": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "identity_note": "NOT SmartMoney10d; Active_* concentration long-short",
    }
    (FACTOR_PANEL_DIR / f"SmartMoneyActiveV2_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return wide, smoothed


def coverage_report(wide: pd.DataFrame) -> dict:
    if wide.empty:
        return {"n_days": 0, "n_symbols": 0, "coverage_cell": 0.0}
    finite = np.isfinite(wide.to_numpy(dtype=float))
    return {
        "n_days": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "coverage_cell": float(finite.mean()),
        "mean_daily_coverage": float(finite.mean(axis=1).mean()),
    }


def distribution_report(wide: pd.DataFrame) -> dict:
    vals = wide.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0}
    s = pd.Series(vals)
    return {
        "n": int(len(vals)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p01": float(s.quantile(0.01)),
        "p50": float(s.quantile(0.50)),
        "p99": float(s.quantile(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }
