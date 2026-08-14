"""SmartMoney10d panel builder — Option B rolling 10d (Kaiyuan).

Cache layout (Phase 1):

  research/cache/smart_money/
    minute_raw/minute_YYYYMM.parquet
    minute_feature/smart_score_YYYYMM.parquet
    factor_panel/SmartMoney10d_*.parquet

L1/L2: DolphinDB month chunks (Close/Volume/Amount only — no Active_*).
L3: Python rolling Q from minute_feature.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from factor_cutting.smart_money import (
    BETA_DEFAULT,
    LOOKBACK_DAYS,
    TOP_CUMVOL_PCT,
)
from core.l2_features.smart_money_q import compute_daily_smart_money_q_fast
from factor_data_loaders import connect_ddb

CACHE_ROOT = Path("research/cache/smart_money")
MINUTE_RAW_DIR = CACHE_ROOT / "minute_raw"
MINUTE_FEATURE_DIR = CACHE_ROOT / "minute_feature"
FACTOR_PANEL_DIR = CACHE_ROOT / "factor_panel"

FORMULA_VERSION = "sm10d_v1_beta0p25_optB"


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


def _ddb_minute_raw_script(start: dt.datetime, end: dt.datetime) -> str:
    """Whitelist columns only — never Active_*."""
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
select Symbol, Date, second(Bartime) as Bartime, Close, Volume, Amount
from t
where Date >= {s} and Date <= {e}
  and ((second(Bartime) >= 09:31:00 and second(Bartime) <= 11:30:00)
    or (second(Bartime) >= 13:01:00 and second(Bartime) <= 15:00:00))
"""


def _ddb_minute_feature_script(start: dt.datetime, end: dt.datetime, beta: float = BETA_DEFAULT) -> str:
    """Server-side ret_1m + smart_score S = abs(ret)/Volume^beta."""
    s = start.strftime("%Y.%m.%d")
    e = end.strftime("%Y.%m.%d")
    return f"""
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
m = select Symbol, Date, second(Bartime) as Bartime, Close, Volume, Amount
from t
where Date >= {s} and Date <= {e}
  and ((second(Bartime) >= 09:31:00 and second(Bartime) <= 11:30:00)
    or (second(Bartime) >= 13:01:00 and second(Bartime) <= 15:00:00))
m = select Symbol, Date, Bartime, Close, Volume, Amount,
  ratios(Close)-1 as ret_1m
from m
context by Symbol, Date csort Bartime
select Symbol, Date, Bartime, Close, Volume, Amount, ret_1m,
  iif(Volume > 0 and isValid(ret_1m),
      abs(ret_1m) / pow(Volume, {beta}),
      double(NULL)) as smart_score
from m
"""


def _normalize_minute_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Symbol": "symbol",
            "Date": "date",
            "Bartime": "bartime",
            "Close": "close",
            "Volume": "volume",
            "Amount": "amount",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = _filter_a_share(df)
    # provenance guard: refuse Active_* if ever leaked
    bad = [c for c in df.columns if c.lower().startswith("active_")]
    if bad:
        raise RuntimeError(f"Active_* columns forbidden in SmartMoney cache: {bad}")
    return df


def load_minute_raw(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """L1: monthly Close/Volume/Amount minute bars."""
    MINUTE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _month_starts(start, end)
    parts: list[pd.DataFrame] = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, c1 in chunks:
            # cache by calendar month of c0
            path = MINUTE_RAW_DIR / f"minute_{_ym_tag(c0)}.parquet"
            if use_cache and path.exists() and not refresh_cache:
                parts.append(pd.read_parquet(path))
                continue
            # full calendar month for stable cache keys
            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            df = s.run(_ddb_minute_raw_script(m0, m1))
            df = _normalize_minute_df(df)
            df.to_parquet(path, index=False)
            parts.append(df)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "bartime", "close", "volume", "amount"])
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["symbol", "date", "bartime"]).reset_index(drop=True)


def load_minute_feature(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    beta: float = BETA_DEFAULT,
) -> pd.DataFrame:
    """L2: minute bars + ret_1m + smart_score (server-side)."""
    MINUTE_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _month_starts(start, end)
    parts: list[pd.DataFrame] = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, c1 in chunks:
            path = MINUTE_FEATURE_DIR / f"smart_score_{_ym_tag(c0)}.parquet"
            if use_cache and path.exists() and not refresh_cache:
                parts.append(pd.read_parquet(path))
                continue
            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            df = s.run(_ddb_minute_feature_script(m0, m1, beta=beta))
            df = _normalize_minute_df(df)
            if "ret_1m" not in df.columns or "smart_score" not in df.columns:
                raise RuntimeError("minute_feature missing ret_1m/smart_score")
            df.to_parquet(path, index=False)
            parts.append(df)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(
            columns=["date", "symbol", "bartime", "close", "volume", "amount", "ret_1m", "smart_score"]
        )
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["symbol", "date", "bartime"]).reset_index(drop=True)


def ensure_minute_feature_months(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    refresh_cache: bool = False,
    beta: float = BETA_DEFAULT,
) -> list:
    """Build missing L2 month caches only — never concat the full history into RAM."""
    MINUTE_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, _c1 in _month_starts(start, end):
            path = MINUTE_FEATURE_DIR / f"smart_score_{_ym_tag(c0)}.parquet"
            paths.append(path)
            if path.exists() and not refresh_cache:
                continue
            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            print(f"  DDB minute_feature {_ym_tag(c0)} ...", flush=True)
            df = s.run(_ddb_minute_feature_script(m0, m1, beta=beta))
            df = _normalize_minute_df(df)
            if "ret_1m" not in df.columns or "smart_score" not in df.columns:
                raise RuntimeError("minute_feature missing ret_1m/smart_score")
            df.to_parquet(path, index=False)
            print(f"  wrote {path.name} rows={len(df):,}", flush=True)
    finally:
        if own:
            s.close()
    return paths


def build_smart_money10d_panel(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    lookback_days: int = LOOKBACK_DAYS,
    top_cumvol_pct: float = TOP_CUMVOL_PCT,
    beta: float = BETA_DEFAULT,
    preheat_calendar_days: int = 25,
    symbols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide + long daily Q panels (Option B).

    Returns (wide, long) where wide index=date, columns=symbol.
    If `symbols` is set, only those names are computed (smoke / subsample).
    """
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    if symbols is not None:
        tag = f"{tag}_n{len(symbols)}"
    wide_path = FACTOR_PANEL_DIR / f"SmartMoney10d_{tag}.parquet"
    long_path = FACTOR_PANEL_DIR / f"SmartMoney10d_long_{tag}.parquet"
    if use_cache and wide_path.exists() and long_path.exists() and not refresh_cache:
        wide = pd.read_parquet(wide_path)
        long = pd.read_parquet(long_path)
        return wide, long

    feat_start = start - dt.timedelta(days=preheat_calendar_days)
    feat = load_minute_feature(
        feat_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        beta=beta,
    )
    if symbols is not None:
        sym_set = set(str(x) for x in symbols)
        feat = feat[feat["symbol"].isin(sym_set)]
    long = compute_daily_smart_money_q_fast(
        feat,
        lookback_days=lookback_days,
        top_cumvol_pct=top_cumvol_pct,
        min_minutes=50,
    )
    long = long[(long["date"] >= pd.Timestamp(start)) & (long["date"] <= pd.Timestamp(end))]
    wide = long.pivot(index="date", columns="symbol", values="Q").sort_index()
    wide.to_parquet(wide_path)
    long.to_parquet(long_path, index=False)
    meta = {
        "formula_version": FORMULA_VERSION,
        "lookback_days": lookback_days,
        "top_cumvol_pct": top_cumvol_pct,
        "beta": beta,
        "window": "option_B_rolling_10d",
        "n_dates": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "symbol_subsample": symbols is not None,
    }
    (FACTOR_PANEL_DIR / f"SmartMoney10d_{tag}_meta.json").write_text(
        __import__("json").dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return wide, long


def coverage_report(wide: pd.DataFrame) -> dict:
    """date/symbol coverage + missing ratio."""
    if wide.empty:
        return {
            "n_days": 0,
            "n_symbols": 0,
            "coverage_cell": 0.0,
            "mean_daily_coverage": 0.0,
            "mean_symbol_coverage": 0.0,
        }
    finite = np.isfinite(wide.to_numpy(dtype=float))
    n_days, n_syms = wide.shape
    cell = float(finite.mean())
    daily = finite.mean(axis=1)
    by_sym = finite.mean(axis=0)
    return {
        "n_days": int(n_days),
        "n_symbols": int(n_syms),
        "coverage_cell": cell,
        "mean_daily_coverage": float(daily.mean()),
        "mean_symbol_coverage": float(by_sym.mean()),
        "min_daily_coverage": float(daily.min()) if len(daily) else 0.0,
    }


def distribution_report(wide: pd.DataFrame) -> dict:
    """Q distribution sanity (expect mass near 1)."""
    vals = wide.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0}
    s = pd.Series(vals)
    near1 = float(((s - 1.0).abs() < 0.01).mean())
    return {
        "n": int(len(vals)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "frac_abs_dev_lt_0p01": near1,
        "frac_abs_dev_lt_0p05": float(((s - 1.0).abs() < 0.05).mean()),
    }
