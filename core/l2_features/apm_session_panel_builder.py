"""APM_SessionResidual Phase1 — session panel builder + cache (no IC).

Cache layout:

  research/cache/apm_session/
    meta/
    calendar/
    stock_overnight/
    stock_pm/
    index_session_proxy/
    residual_panel/

Identity: adapted_replication (EOD index day proxy ≠ true PM).
No Active_* columns. No shift(1) in cache — eval shifts later.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from core.l2_features.tgd_panel_builder import overnight_return_long
from factor_data_loaders import connect_ddb, load_eod_wide_tables

CACHE_ROOT = Path("research/cache/apm_session")
META_DIR = CACHE_ROOT / "meta"
CALENDAR_DIR = CACHE_ROOT / "calendar"
STOCK_OVN_DIR = CACHE_ROOT / "stock_overnight"
STOCK_PM_DIR = CACHE_ROOT / "stock_pm"
INDEX_DIR = CACHE_ROOT / "index_session_proxy"
RESIDUAL_DIR = CACHE_ROOT / "residual_panel"

FORMULA_VERSION = "apm_session_v1_adapted_eod_index"
FACTOR_ID = "APM_SessionResidual"
DEFAULT_INDEX = "000852.SH"
PM_START_RULE = "first_available_bar_after_13_01"
MIN_PM_BARS = 2


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


def _range_tag(start: dt.datetime, end: dt.datetime) -> str:
    return f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


def _filter_a_share(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    sym = df[symbol_col].astype(str)
    return df.loc[sym.str[0].isin(("6", "0", "3"))].copy()


def _refuse_active_columns(df: pd.DataFrame, where: str) -> None:
    bad = [c for c in df.columns if str(c).lower().startswith("active_")]
    if bad:
        raise RuntimeError(f"Active_* columns forbidden in APM cache ({where}): {bad}")


def _ddb_stock_pm_script(start: dt.datetime, end: dt.datetime) -> str:
    """Daily PM session: first/last available bar in [13:01, 15:00]. Close only.

    Step1: context-by csort to order bars within Symbol×Date.
    Step2: group-by to collapse to one row (context-by alone keeps all minute rows).
    """
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
m = select Symbol, Date, second(Bartime) as Bartime, Close
from t
where Date >= {s} and Date <= {e}
  and second(Bartime) >= 13:01:00 and second(Bartime) <= 15:00:00
m = select Symbol, Date, Bartime, Close from m context by Symbol, Date csort Bartime
select Symbol, Date,
  first(Close) as pm_close_first,
  last(Close) as pm_close_last,
  first(Bartime) as pm_bartime_first,
  last(Bartime) as pm_bartime_last,
  count(*) as pm_n_bars
from m
group by Symbol, Date
"""


def _as_time_str(v) -> str:
    """DolphinDB SECOND often arrives as datetime@1970-01-01 — keep HH:MM:SS only."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if hasattr(v, "strftime"):
        try:
            return v.strftime("%H:%M:%S")
        except Exception:
            pass
    s = str(v)
    if " " in s and ":" in s:
        return s.split(" ")[-1]
    return s


def _normalize_pm_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Date": "date",
            "pm_close_first": "pm_close_first",
            "pm_close_last": "pm_close_last",
            "pm_bartime_first": "pm_bartime_first",
            "pm_bartime_last": "pm_bartime_last",
            "pm_n_bars": "pm_n_bars",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = _filter_a_share(df)
    _refuse_active_columns(df, "stock_pm")
    df["pm_bartime_first"] = df["pm_bartime_first"].map(_as_time_str)
    df["pm_bartime_last"] = df["pm_bartime_last"].map(_as_time_str)
    n = pd.to_numeric(df["pm_n_bars"], errors="coerce")
    first = pd.to_numeric(df["pm_close_first"], errors="coerce")
    last = pd.to_numeric(df["pm_close_last"], errors="coerce")
    ok = (n >= MIN_PM_BARS) & first.notna() & last.notna() & (first != 0)
    pm = np.where(ok, last / first - 1.0, np.nan)
    df["pm_return"] = pm
    df["pm_n_bars"] = n.astype("Int64")
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_stock_pm(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """L1: monthly stock PM session returns from minute Close."""
    STOCK_PM_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _month_starts(start, end)
    parts: list[pd.DataFrame] = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, _c1 in chunks:
            path = STOCK_PM_DIR / f"stock_pm_{_ym_tag(c0)}.parquet"
            if use_cache and path.exists() and not refresh_cache:
                part = pd.read_parquet(path)
                _refuse_active_columns(part, f"cached {path.name}")
                parts.append(part)
                continue
            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            print(f"  DDB stock_pm {_ym_tag(c0)} ...", flush=True)
            raw = s.run(_ddb_stock_pm_script(m0, m1))
            df = _normalize_pm_df(raw)
            df.to_parquet(path, index=False)
            print(f"  wrote {path.name} rows={len(df):,}", flush=True)
            parts.append(df)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "pm_close_first",
                "pm_close_last",
                "pm_bartime_first",
                "pm_bartime_last",
                "pm_n_bars",
                "pm_return",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_stock_overnight(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 15,
) -> pd.DataFrame:
    """L2: overnight_return = Open / prev_Close - 1 (EOD)."""
    STOCK_OVN_DIR.mkdir(parents=True, exist_ok=True)
    tag = _range_tag(start, end)
    path = STOCK_OVN_DIR / f"stock_overnight_{tag}.parquet"
    if use_cache and path.exists() and not refresh_cache:
        out = pd.read_parquet(path)
        out["date"] = pd.to_datetime(out["date"])
        return out

    load_start = start - dt.timedelta(days=preheat_calendar_days)
    eod, s = load_eod_wide_tables(load_start, end, session=session)
    try:
        long = overnight_return_long(eod.open, eod.close)
    finally:
        if session is None:
            s.close()

    long = long.rename(columns={"overnight_return": "overnight_return"})
    long = long[(long["date"] >= pd.Timestamp(start)) & (long["date"] <= pd.Timestamp(end))]
    long = long.sort_values(["date", "symbol"]).reset_index(drop=True)
    long.to_parquet(path, index=False)
    return long


def load_index_session_proxy(
    start: dt.datetime,
    end: dt.datetime,
    *,
    index_code: str = DEFAULT_INDEX,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 15,
) -> pd.DataFrame:
    """L3: adapted index overnight + full daytime (not PM-matched)."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    safe_idx = index_code.replace(".", "")
    tag = _range_tag(start, end)
    path = INDEX_DIR / f"index_session_proxy_{safe_idx}_{tag}.parquet"
    if use_cache and path.exists() and not refresh_cache:
        out = pd.read_parquet(path)
        out["date"] = pd.to_datetime(out["date"])
        return out

    load_start = start - dt.timedelta(days=preheat_calendar_days)
    own = session is None
    s = session or connect_ddb()
    try:
        s0 = load_start.strftime("%Y.%m.%d")
        s1 = end.strftime("%Y.%m.%d")
        script = f"""
t = loadTable('dfs://WIND.AINDEXEODPRICES','data')
select TRADE_DT as Date, S_DQ_OPEN as Open, S_DQ_CLOSE as Close, S_DQ_PRECLOSE as PreClose
from t
where S_INFO_WINDCODE = '{index_code}'
  and TRADE_DT >= {s0} and TRADE_DT <= {s1}
"""
        raw = s.run(script)
    finally:
        if own:
            s.close()

    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No index EOD rows for {index_code}")

    df = raw.rename(columns={"Date": "date", "Open": "open", "Close": "close", "PreClose": "preclose"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    # Prefer PreClose when present; else shift Close
    pre = pd.to_numeric(df["preclose"], errors="coerce")
    if pre.isna().all():
        pre = pd.to_numeric(df["close"], errors="coerce").shift(1)
    opn = pd.to_numeric(df["open"], errors="coerce")
    cls = pd.to_numeric(df["close"], errors="coerce")
    df["index_overnight"] = opn / pre - 1.0
    df["index_day"] = cls / opn - 1.0
    out = pd.DataFrame(
        {
            "date": df["date"],
            "index_code": index_code,
            "index_overnight": df["index_overnight"],
            "index_day": df["index_day"],
            "adapted": True,
        }
    )
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    out = out.sort_values("date").reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


def build_trade_calendar(
    start: dt.datetime,
    end: dt.datetime,
    *,
    stock_ovn: pd.DataFrame,
    stock_pm: pd.DataFrame,
    index_proxy: pd.DataFrame,
) -> dict[str, Any]:
    """Persist alignment calendar for session-debug (not factor math)."""
    CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    eod_days = sorted(pd.to_datetime(stock_ovn["date"]).dropna().unique())
    pm_days = set(pd.to_datetime(stock_pm["date"]).dropna().unique())
    idx_days = set(pd.to_datetime(index_proxy["date"]).dropna().unique())
    eod_set = set(eod_days)

    missing_pm_vs_eod = sorted(d for d in eod_days if d not in pm_days)
    missing_idx_vs_eod = sorted(d for d in eod_days if d not in idx_days)
    pm_only = sorted(d for d in pm_days if d not in eod_set)

    cal = {
        "trade_calendar": {
            "source": "WIND.ASHAREEODPRICES dates present in stock_overnight panel",
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "n_eod_days": len(eod_days),
            "n_pm_days": len(pm_days),
            "n_index_days": len(idx_days),
            "missing_days": {
                "eod_without_any_pm_rows": [d.strftime("%Y-%m-%d") for d in missing_pm_vs_eod[:50]],
                "eod_without_index": [d.strftime("%Y-%m-%d") for d in missing_idx_vs_eod[:50]],
                "pm_without_eod": [d.strftime("%Y-%m-%d") for d in pm_only[:50]],
                "n_eod_without_any_pm_rows": len(missing_pm_vs_eod),
                "n_eod_without_index": len(missing_idx_vs_eod),
            },
        }
    }
    path = CALENDAR_DIR / f"trade_calendar_{_range_tag(start, end)}.json"
    path.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")
    return cal


def build_residual_panel(
    stock_ovn: pd.DataFrame,
    stock_pm: pd.DataFrame,
    index_proxy: pd.DataFrame,
) -> pd.DataFrame:
    """L4: join overnight + PM + index proxy → residual legs (no Ret20, no rolling)."""
    ovn = stock_ovn[["date", "symbol", "overnight_return"]].copy()
    ovn["date"] = pd.to_datetime(ovn["date"])
    pm = stock_pm[["date", "symbol", "pm_return"]].copy()
    pm["date"] = pd.to_datetime(pm["date"])
    idx = index_proxy[["date", "index_overnight", "index_day"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])

    panel = ovn.merge(pm, on=["date", "symbol"], how="outer")
    panel = panel.merge(idx, on="date", how="left")
    panel = panel.rename(
        columns={
            "overnight_return": "r_on",
            "pm_return": "r_pm",
            "index_overnight": "r_on_idx",
            "index_day": "r_day_idx",
        }
    )
    panel["alpha_on"] = panel["r_on"] - panel["r_on_idx"]
    panel["alpha_pm"] = panel["r_pm"] - panel["r_day_idx"]
    panel["delta_alpha"] = panel["alpha_on"] - panel["alpha_pm"]
    _refuse_active_columns(panel, "residual_panel")
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def formula_meta(index_code: str = DEFAULT_INDEX) -> dict[str, Any]:
    return {
        "factor_id": FACTOR_ID,
        "formula_version": FORMULA_VERSION,
        "identity_class": "adapted_replication",
        "index_code": index_code,
        "index_pm_matched": False,
        "index_residual_method": "eod_overnight_plus_eod_daytime_proxy",
        "pm_start_rule": PM_START_RULE,
        "cache_shifted_for_backtest": False,
        "active_star_columns": False,
        "proxy_factor_id_untouched": "ActiveTradeProxy",
        "min_pm_bars": MIN_PM_BARS,
    }


def build_apm_session_panel(
    start: dt.datetime,
    end: dt.datetime,
    *,
    index_code: str = DEFAULT_INDEX,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 15,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end Phase1: build caches + residual panel. No IC / no shift."""
    for d in (META_DIR, CALENDAR_DIR, STOCK_OVN_DIR, STOCK_PM_DIR, INDEX_DIR, RESIDUAL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    tag = _range_tag(start, end)
    residual_path = RESIDUAL_DIR / f"apm_residual_panel_{tag}.parquet"
    meta = formula_meta(index_code)

    print("L1 stock_pm ...", flush=True)
    stock_pm = load_stock_pm(
        start, end, session=session, use_cache=use_cache, refresh_cache=refresh_cache
    )
    # convenience concat view
    pm_view = STOCK_PM_DIR / f"stock_pm_{tag}.parquet"
    stock_pm.to_parquet(pm_view, index=False)

    print("L2 stock_overnight ...", flush=True)
    stock_ovn = load_stock_overnight(
        start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        preheat_calendar_days=preheat_calendar_days,
    )

    print("L3 index_session_proxy ...", flush=True)
    index_proxy = load_index_session_proxy(
        start,
        end,
        index_code=index_code,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        preheat_calendar_days=preheat_calendar_days,
    )

    print("calendar ...", flush=True)
    calendar = build_trade_calendar(
        start, end, stock_ovn=stock_ovn, stock_pm=stock_pm, index_proxy=index_proxy
    )

    print("L4 residual_panel ...", flush=True)
    residual = build_residual_panel(stock_ovn, stock_pm, index_proxy)
    residual.to_parquet(residual_path, index=False)
    # alias deliverable names at cache root (optional convenience)
    residual.to_parquet(CACHE_ROOT / "apm_residual_panel.parquet", index=False)
    stock_ovn.to_parquet(CACHE_ROOT / "stock_overnight.parquet", index=False)
    stock_pm.to_parquet(CACHE_ROOT / "stock_pm_return.parquet", index=False)
    index_proxy.to_parquet(CACHE_ROOT / "index_session_proxy.parquet", index=False)

    manifest = {
        **meta,
        "window": {"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d")},
        "paths": {
            "stock_overnight": str(STOCK_OVN_DIR / f"stock_overnight_{tag}.parquet"),
            "stock_pm_months": str(STOCK_PM_DIR / "stock_pm_YYYYMM.parquet"),
            "stock_pm_view": str(pm_view),
            "index_session_proxy": str(
                INDEX_DIR / f"index_session_proxy_{index_code.replace('.', '')}_{tag}.parquet"
            ),
            "residual_panel": str(residual_path),
            "calendar": str(CALENDAR_DIR / f"trade_calendar_{tag}.json"),
        },
        "n_rows_residual": int(len(residual)),
        "n_symbols": int(residual["symbol"].nunique()) if len(residual) else 0,
        "n_dates": int(residual["date"].nunique()) if len(residual) else 0,
        "calendar_summary": calendar.get("trade_calendar", {}),
    }
    (META_DIR / "formula_version.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (META_DIR / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return residual, manifest


def coverage_stats(residual: pd.DataFrame) -> dict[str, Any]:
    if residual.empty:
        return {
            "n_rows": 0,
            "n_dates": 0,
            "n_symbols": 0,
            "pct_nan": {},
        }
    cols = ["r_on", "r_pm", "alpha_on", "alpha_pm", "delta_alpha"]
    pct_nan = {}
    for c in cols:
        if c in residual.columns:
            pct_nan[c] = float(residual[c].isna().mean())
    # daily mean finite symbols for alpha_pm
    g = residual.groupby("date", sort=True)
    daily_pm = g["r_pm"].apply(lambda s: float(np.isfinite(s.to_numpy(dtype=float)).mean()))
    return {
        "n_rows": int(len(residual)),
        "n_dates": int(residual["date"].nunique()),
        "n_symbols": int(residual["symbol"].nunique()),
        "pct_nan": pct_nan,
        "mean_daily_frac_finite_r_pm": float(daily_pm.mean()) if len(daily_pm) else 0.0,
        "min_daily_frac_finite_r_pm": float(daily_pm.min()) if len(daily_pm) else 0.0,
    }


def build_sample_checks(
    stock_ovn: pd.DataFrame,
    stock_pm: pd.DataFrame,
    open_: pd.DataFrame,
    close: pd.DataFrame,
    *,
    n_symbols: int = 20,
    n_dates: int = 5,
) -> pd.DataFrame:
    """Spot-check rows for paper review (no IC)."""
    pm = stock_pm.copy()
    pm["date"] = pd.to_datetime(pm["date"])
    close = close.copy()
    open_ = open_.copy()
    close.index = pd.to_datetime(close.index)
    open_.index = pd.to_datetime(open_.index)

    # Prefer symbols with dense PM coverage
    counts = pm.groupby("symbol")["pm_return"].apply(lambda s: int(s.notna().sum()))
    syms = counts.sort_values(ascending=False).head(n_symbols).index.tolist()
    dates = sorted(pm["date"].unique())[:n_dates]
    rows = []
    for d in dates:
        d_ts = pd.Timestamp(d)
        for sym in syms:
            pm_row = pm[(pm["date"] == d_ts) & (pm["symbol"] == sym)]
            if pm_row.empty:
                continue
            pr = pm_row.iloc[0]
            prev_c = np.nan
            opn = np.nan
            try:
                if sym in close.columns and d_ts in close.index:
                    loc = close.index.get_loc(d_ts)
                    if isinstance(loc, (int, np.integer)) and int(loc) > 0:
                        prev_c = float(close.iloc[int(loc) - 1][sym])
                    if sym in open_.columns:
                        opn = float(open_.loc[d_ts, sym])
            except Exception:
                pass
            rows.append(
                {
                    "date": d_ts.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "prev_close": prev_c,
                    "open": opn,
                    "pm_first_time": str(pr.get("pm_bartime_first", "")),
                    "pm_last_time": str(pr.get("pm_bartime_last", "")),
                    "pm_first_close": float(pr["pm_close_first"])
                    if pd.notna(pr.get("pm_close_first"))
                    else np.nan,
                    "pm_last_close": float(pr["pm_close_last"])
                    if pd.notna(pr.get("pm_close_last"))
                    else np.nan,
                    "pm_return": float(pr["pm_return"])
                    if pd.notna(pr.get("pm_return"))
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)
