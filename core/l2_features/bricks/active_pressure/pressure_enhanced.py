"""Enhanced active_pressure knives: session-weighted, smart-filtered, delta.

Observable extensions on top of baseline amount-weighted APM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.l2_features.bricks.active_pressure.pressure import (
    MIN_MINUTES_PER_DAY,
    PRESSURE_COL,
    PRESSURE_EWM_COL,
    _weighted_mean,
    minute_raw_apm,
    smooth_active_pressure,
)

# Session windows (seconds since midnight) — local copy to avoid factor_cutting import cycle
OPEN_WINDOW_START = 9 * 3600 + 30 * 60
OPEN_WINDOW_END = 10 * 3600
CLOSE_WINDOW_START = 14 * 3600 + 30 * 60
CLOSE_WINDOW_END = 15 * 3600

SESSION_W_OPEN = 0.6
SESSION_W_MID = 1.0
SESSION_W_CLOSE = 1.5

SMART_SIZE_MULT = 1.2
SMART_LOOKBACK_DAYS = 20
SMART_ROLL_MIN_PERIODS = 5
SMART_MIN_MINUTES = 5

DELTA_LAG = 3
BRICK_VERSION_SESSION = "active_pressure_session_v1"
BRICK_VERSION_SMART = "active_pressure_smart_v1"


def bartime_to_seconds(bartime: pd.Series) -> pd.Series:
    if isinstance(bartime, pd.DatetimeIndex):
        bartime = pd.Series(bartime)
    else:
        bartime = pd.Series(bartime)
    if pd.api.types.is_datetime64_any_dtype(bartime):
        return (
            bartime.dt.hour.astype(np.int64) * 3600
            + bartime.dt.minute.astype(np.int64) * 60
            + bartime.dt.second.astype(np.int64)
        )
    if pd.api.types.is_timedelta64_dtype(bartime):
        return pd.to_timedelta(bartime).dt.total_seconds().astype(np.float64)
    return pd.to_numeric(bartime, errors="coerce")


def session_weight_from_seconds(sec: np.ndarray) -> np.ndarray:
    """Map seconds-since-midnight → session weight."""
    sec = np.asarray(sec, dtype=float)
    w = np.full(sec.shape, SESSION_W_MID, dtype=float)
    open_m = (sec >= OPEN_WINDOW_START) & (sec < OPEN_WINDOW_END)
    close_m = (sec >= CLOSE_WINDOW_START) & (sec < CLOSE_WINDOW_END)
    w[open_m] = SESSION_W_OPEN
    w[close_m] = SESSION_W_CLOSE
    return w


def assign_session_weight(bartime: pd.Series) -> np.ndarray:
    """Public helper for tests / callers with bartime series."""
    return session_weight_from_seconds(bartime_to_seconds(bartime).to_numpy(dtype=float))


def ensure_avg_buy_size(minutes: pd.DataFrame) -> pd.DataFrame:
    """Add avg_buy_size = active_buy_amt / active_buy_count when missing."""
    out = minutes
    if "avg_buy_size" in out.columns:
        return out
    if "active_buy_count" not in out.columns:
        raise ValueError("Smart APM needs avg_buy_size or active_buy_count")
    buy = pd.to_numeric(out["active_buy_amt"], errors="coerce").astype(float)
    cnt = pd.to_numeric(out["active_buy_count"], errors="coerce").astype(float)
    size = buy / cnt.where(cnt > 0)
    out = out.copy()
    out["avg_buy_size"] = size
    return out


def compute_daily_apm_session(
    minutes: pd.DataFrame,
    min_minutes: int = MIN_MINUTES_PER_DAY,
) -> pd.DataFrame:
    """Session-weighted daily APM: Σ(w·raw·amt) / Σ(w·amt)."""
    need = {"date", "symbol", "bartime", "active_buy_amt", "active_sell_amt", "amount"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"Session APM missing columns: {missing}")

    df = minutes.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["raw_apm"] = minute_raw_apm(df["active_buy_amt"], df["active_sell_amt"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    df["w"] = session_weight_from_seconds(
        bartime_to_seconds(df["bartime"]).to_numpy(dtype=float)
    )
    df["w_amt"] = df["w"] * df["amount"]

    rows: list[dict] = []
    for (sym, d), g in df.groupby(["symbol", "date"], sort=False):
        if len(g) < min_minutes:
            rows.append({"date": pd.Timestamp(d), "symbol": sym, PRESSURE_COL: np.nan})
            continue
        val = _weighted_mean(
            g["raw_apm"].to_numpy(dtype=float),
            g["w_amt"].to_numpy(dtype=float),
        )
        rows.append({"date": pd.Timestamp(d), "symbol": sym, PRESSURE_COL: val})

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", PRESSURE_COL])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def compute_daily_apm_smart(
    minutes: pd.DataFrame,
    *,
    size_mult: float = SMART_SIZE_MULT,
    lookback_days: int = SMART_LOOKBACK_DAYS,
    roll_min_periods: int = SMART_ROLL_MIN_PERIODS,
    min_minutes: int = SMART_MIN_MINUTES,
) -> pd.DataFrame:
    """Smart APM: keep minutes with avg_buy_size > lagged 20d mean × size_mult."""
    need = {"date", "symbol", "active_buy_amt", "active_sell_amt", "amount"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"Smart APM missing columns: {missing}")

    df = ensure_avg_buy_size(minutes)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["avg_buy_size"] = pd.to_numeric(df["avg_buy_size"], errors="coerce").astype(float)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    df["raw_apm"] = minute_raw_apm(df["active_buy_amt"], df["active_sell_amt"])

    daily_size = (
        df.groupby(["symbol", "date"], sort=False)["avg_buy_size"]
        .mean()
        .rename("day_avg_size")
        .reset_index()
    )
    daily_size = daily_size.sort_values(["symbol", "date"])
    daily_size["roll_mean"] = daily_size.groupby("symbol", sort=False)["day_avg_size"].transform(
        lambda x: x.shift(1).rolling(lookback_days, min_periods=roll_min_periods).mean()
    )
    df = df.merge(
        daily_size[["symbol", "date", "roll_mean"]],
        on=["symbol", "date"],
        how="left",
    )
    thresh = df["roll_mean"] * float(size_mult)
    keep = (
        np.isfinite(df["avg_buy_size"].to_numpy(dtype=float))
        & np.isfinite(thresh.to_numpy(dtype=float))
        & (df["avg_buy_size"].to_numpy(dtype=float) > thresh.to_numpy(dtype=float))
    )
    filtered = df.loc[keep]

    rows: list[dict] = []
    all_keys = df.groupby(["symbol", "date"], sort=False).size().reset_index(name="_n")
    kept_groups = {
        (sym, pd.Timestamp(d)): g
        for (sym, d), g in filtered.groupby(["symbol", "date"], sort=False)
    }
    for _, r in all_keys.iterrows():
        sym, d = str(r["symbol"]), pd.Timestamp(r["date"])
        g = kept_groups.get((sym, d))
        if g is None or len(g) < min_minutes:
            rows.append({"date": d, "symbol": sym, PRESSURE_COL: np.nan})
            continue
        val = _weighted_mean(
            g["raw_apm"].to_numpy(dtype=float),
            g["amount"].to_numpy(dtype=float),
        )
        rows.append({"date": d, "symbol": sym, PRESSURE_COL: val})

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", PRESSURE_COL])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def delta_apm_wide(
    apm_raw_wide: pd.DataFrame,
    lag: int = DELTA_LAG,
) -> pd.DataFrame:
    """Wide panel: APM_t - APM_{t-lag} per symbol (columns)."""
    if apm_raw_wide.empty:
        return apm_raw_wide.copy()
    fac = apm_raw_wide.sort_index().astype(float)
    return fac - fac.shift(lag)


def smooth_delta_wide(
    delta_wide: pd.DataFrame,
    span: int = 5,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Column-wise EWM on delta panel."""
    if delta_wide.empty:
        return delta_wide.copy()
    return delta_wide.sort_index().astype(float).apply(
        lambda s: s.ewm(span=span, min_periods=min_periods).mean()
    )


def long_to_smooth_wide(
    daily: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    span: int = 5,
    min_periods: int = 3,
    value_col: str = PRESSURE_COL,
    out_col: str = PRESSURE_EWM_COL,
) -> pd.DataFrame:
    """EWM-smooth long apm_raw → wide apm_smooth clipped to [start, end]."""
    if daily.empty:
        return pd.DataFrame()
    smoothed = smooth_active_pressure(
        daily, span=span, min_periods=min_periods, value_col=value_col, out_col=out_col
    )
    smoothed = smoothed[
        (smoothed["date"] >= start) & (smoothed["date"] <= end)
    ]
    if smoothed.empty:
        return pd.DataFrame()
    return smoothed.pivot(
        index="date", columns="symbol", values=out_col
    ).sort_index()


# ---------- Smart APM V2 ----------
SMARTV2_LOOKBACK = 20
SMARTV2_QUANTILE = 0.8
SMARTV2_EWM_SPAN = 2
SMARTV2_MIN_PERIODS = 2
SMARTV2_ASC_MIN_RANK = 0.5
BRICK_VERSION_SMARTV2 = "active_pressure_smartv2_q80_span2"

# V2.1 hotfix: relax ASC hard-gate, restore daily EWM span, tighten size quantile
SMARTV2_1_QUANTILE = 0.90
SMARTV2_1_EWM_SPAN = 5
SMARTV2_1_MIN_PERIODS = 3
SMARTV2_1_ASC_MIN_RANK = 0.0  # disable hard gate
BRICK_VERSION_SMARTV2_1 = "active_pressure_smartv2_1_q90_span5"

# Fast ablate: reuse V2 (q80) brick; only change ASC gate + EWM (no recompute minutes)
SMARTV2_1F_QUANTILE = SMARTV2_QUANTILE
SMARTV2_1F_EWM_SPAN = 5
SMARTV2_1F_MIN_PERIODS = 3
SMARTV2_1F_ASC_MIN_RANK = 0.0


def compute_dynamic_size_threshold(
    daily_avg_size: pd.DataFrame,
    *,
    lookback: int = SMARTV2_LOOKBACK,
    quantile: float = SMARTV2_QUANTILE,
    roll_min_periods: int = SMART_ROLL_MIN_PERIODS,
    size_col: str = "avg_buy_size",
) -> pd.Series:
    """Lag-1 rolling quantile of daily avg_buy_size per symbol (no look-ahead)."""
    if size_col not in daily_avg_size.columns:
        raise ValueError(f"daily_avg_size missing {size_col}")
    work = daily_avg_size.sort_values(["symbol", "date"])
    return work.groupby("symbol", sort=False)[size_col].transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=roll_min_periods).quantile(quantile)
    )


def compute_daily_smart_apm_v2(
    minutes: pd.DataFrame,
    *,
    lookback: int = SMARTV2_LOOKBACK,
    quantile: float = SMARTV2_QUANTILE,
    roll_min_periods: int = SMART_ROLL_MIN_PERIODS,
    min_minutes: int = SMART_MIN_MINUTES,
) -> pd.DataFrame:
    """Smart APM V2: dynamic quantile filter + buy/sell intensity split.

    On big-size minutes:
      buy_intensity  = Σ active_buy_amt / Σ amount
      sell_intensity = Σ active_sell_amt / Σ amount
      apm_raw        = buy_intensity - sell_intensity

    ASC gating is applied later in the builder (needs day-level ASC brick).
    Returns [date, symbol, buy_intensity, sell_intensity, apm_raw].
    """
    need = {"date", "symbol", "active_buy_amt", "active_sell_amt", "amount"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"SmartV2 missing columns: {missing}")

    df = ensure_avg_buy_size(minutes).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["avg_buy_size"] = pd.to_numeric(df["avg_buy_size"], errors="coerce").astype(float)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)
    df["active_buy_amt"] = pd.to_numeric(df["active_buy_amt"], errors="coerce").astype(float)
    df["active_sell_amt"] = pd.to_numeric(df["active_sell_amt"], errors="coerce").astype(float)

    daily_size = (
        df.groupby(["symbol", "date"], sort=False)["avg_buy_size"]
        .mean()
        .rename("avg_buy_size")
        .reset_index()
    )
    daily_size = daily_size.sort_values(["symbol", "date"])
    daily_size["threshold"] = compute_dynamic_size_threshold(
        daily_size,
        lookback=lookback,
        quantile=quantile,
        roll_min_periods=roll_min_periods,
    )
    df = df.merge(
        daily_size[["symbol", "date", "threshold"]],
        on=["symbol", "date"],
        how="left",
    )
    keep = (
        np.isfinite(df["avg_buy_size"].to_numpy(dtype=float))
        & np.isfinite(df["threshold"].to_numpy(dtype=float))
        & (df["avg_buy_size"].to_numpy(dtype=float) >= df["threshold"].to_numpy(dtype=float))
    )
    filtered = df.loc[keep]

    rows: list[dict] = []
    all_keys = df.groupby(["symbol", "date"], sort=False).size().reset_index(name="_n")
    kept_groups = {
        (sym, pd.Timestamp(d)): g
        for (sym, d), g in filtered.groupby(["symbol", "date"], sort=False)
    }
    for _, r in all_keys.iterrows():
        sym, d = str(r["symbol"]), pd.Timestamp(r["date"])
        g = kept_groups.get((sym, d))
        if g is None or len(g) < min_minutes:
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "buy_intensity": np.nan,
                    "sell_intensity": np.nan,
                    PRESSURE_COL: np.nan,
                }
            )
            continue
        amt = g["amount"].to_numpy(dtype=float)
        buy = g["active_buy_amt"].to_numpy(dtype=float)
        sell = g["active_sell_amt"].to_numpy(dtype=float)
        mask = np.isfinite(amt) & (amt > 0)
        if not mask.any():
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "buy_intensity": np.nan,
                    "sell_intensity": np.nan,
                    PRESSURE_COL: np.nan,
                }
            )
            continue
        denom = float(amt[mask].sum())
        if denom <= 0:
            bi = si = np.nan
        else:
            bi = float(buy[mask].sum()) / denom
            si = float(sell[mask].sum()) / denom
        rows.append(
            {
                "date": d,
                "symbol": sym,
                "buy_intensity": bi,
                "sell_intensity": si,
                PRESSURE_COL: bi - si if np.isfinite(bi) and np.isfinite(si) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["date", "symbol", "buy_intensity", "sell_intensity", PRESSURE_COL]
        )
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def apply_asc_cs_gate(
    daily: pd.DataFrame,
    asc_wide: pd.DataFrame,
    *,
    min_rank: float = SMARTV2_ASC_MIN_RANK,
    value_col: str = PRESSURE_COL,
) -> pd.DataFrame:
    """Zero-out daily factor where CS ASC rank (pct) < min_rank."""
    if daily.empty or asc_wide is None or asc_wide.empty:
        return daily
    out = daily.copy()
    ranks = asc_wide.sort_index().astype(float).rank(axis=1, pct=True, method="average")
    long_r = ranks.stack().rename("asc_rank").reset_index()
    long_r.columns = ["date", "symbol", "asc_rank"]
    long_r["date"] = pd.to_datetime(long_r["date"])
    long_r["symbol"] = long_r["symbol"].astype(str)
    merged = out.merge(long_r, on=["date", "symbol"], how="left")
    bad = merged["asc_rank"].isna() | (merged["asc_rank"] < float(min_rank))
    merged.loc[bad, value_col] = np.nan
    if "buy_intensity" in merged.columns:
        merged.loc[bad, "buy_intensity"] = np.nan
    if "sell_intensity" in merged.columns:
        merged.loc[bad, "sell_intensity"] = np.nan
    return merged.drop(columns=["asc_rank"], errors="ignore")
