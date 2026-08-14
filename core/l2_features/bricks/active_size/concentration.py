"""Active buy avg-size concentration (shared brick kernel).

Column name: ``active_size_concentration``
Alias when reading legacy caches: ``inst_ratio`` (deprecated naming).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_SIZE_PCT = 0.20
EWM_SPAN = 5
EWM_MIN_PERIODS = 3
MIN_MINUTES_DEFAULT = 30

ACTIVE_SIZE_COL = "active_size_concentration"
ACTIVE_SIZE_EWM_COL = "active_size_concentration_ewm"
# deprecated — kept only for parquet / API compatibility
_LEGACY_COL = "inst_ratio"
_LEGACY_EWM_COL = "inst_ratio_ewm"

BRICK_VERSION = "active_size_conc_top20_avgbuy"


def concentration_one_day(
    amts: np.ndarray,
    sizes: np.ndarray,
    top_pct: float = TOP_SIZE_PCT,
) -> float:
    """Share of active-buy amount in minutes with top-pct avg buy size."""
    amts = np.asarray(amts, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    valid = np.isfinite(amts) & np.isfinite(sizes) & (amts > 0) & (sizes > 0)
    if not valid.any():
        return np.nan
    amts_v = amts[valid]
    sizes_v = sizes[valid]
    total = float(amts_v.sum())
    if total <= 0 or not np.isfinite(total):
        return np.nan
    thr = float(np.nanquantile(sizes_v, 1.0 - top_pct))
    big = amts_v[sizes_v >= thr]
    if big.size == 0:
        return np.nan
    return float(big.sum() / total)


def normalize_active_size_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy ``inst_ratio`` → ``active_size_concentration``."""
    out = df.copy()
    if ACTIVE_SIZE_COL not in out.columns and _LEGACY_COL in out.columns:
        out = out.rename(columns={_LEGACY_COL: ACTIVE_SIZE_COL})
    if ACTIVE_SIZE_EWM_COL not in out.columns and _LEGACY_EWM_COL in out.columns:
        out = out.rename(columns={_LEGACY_EWM_COL: ACTIVE_SIZE_EWM_COL})
    return out


def compute_daily_active_size_concentration(
    minutes: pd.DataFrame,
    min_minutes: int = MIN_MINUTES_DEFAULT,
    top_pct: float = TOP_SIZE_PCT,
) -> pd.DataFrame:
    """Per (symbol, date) active_size_concentration.

    Required: date, symbol, active_buy_amt, avg_buy_size.
    """
    need = {"date", "symbol", "active_buy_amt", "avg_buy_size"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"active_size minutes missing: {missing}")

    df = minutes
    rows: list[dict] = []
    symbols = df["symbol"].astype(str).unique()
    n_sym = len(symbols)

    for si, (sym, g) in enumerate(df.groupby(df["symbol"].astype(str), sort=False)):
        if si > 0 and si % 500 == 0:
            print(f"  active_size brick {si}/{n_sym} symbols ...", flush=True)
        for d, sub in g.groupby("date", sort=True):
            if len(sub) < min_minutes:
                rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "symbol": sym,
                        ACTIVE_SIZE_COL: np.nan,
                    }
                )
                continue
            val = concentration_one_day(
                sub["active_buy_amt"].to_numpy(dtype=float),
                sub["avg_buy_size"].to_numpy(dtype=float),
                top_pct=top_pct,
            )
            rows.append(
                {
                    "date": pd.Timestamp(d),
                    "symbol": sym,
                    ACTIVE_SIZE_COL: val,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", ACTIVE_SIZE_COL])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def smooth_active_size_concentration(
    daily: pd.DataFrame,
    span: int = EWM_SPAN,
    min_periods: int = EWM_MIN_PERIODS,
) -> pd.DataFrame:
    """EWM-smooth → ``active_size_concentration_ewm``."""
    daily = normalize_active_size_columns(daily)
    if ACTIVE_SIZE_COL not in daily.columns:
        raise ValueError(f"daily missing {ACTIVE_SIZE_COL}")
    out = daily.sort_values(["symbol", "date"]).copy()
    out[ACTIVE_SIZE_EWM_COL] = out.groupby("symbol", sort=False)[
        ACTIVE_SIZE_COL
    ].transform(lambda x: x.ewm(span=span, min_periods=min_periods).mean())
    return out
