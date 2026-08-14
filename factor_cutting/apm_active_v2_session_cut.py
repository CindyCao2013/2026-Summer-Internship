"""Legacy APM session-cut knife (open/close imbalance + intraday slope).

Kept for research continuity. Default ``APM_ActiveV2`` uses active_pressure brick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OPEN_WINDOW_START = 9 * 3600 + 30 * 60
OPEN_WINDOW_END = 10 * 3600
CLOSE_WINDOW_START = 14 * 3600 + 30 * 60
CLOSE_WINDOW_END = 15 * 3600

MO_WEIGHT = 0.4
MC_WEIGHT = 0.4
SLOPE_WEIGHT = 0.2
N_INTRADAY_SLICES = 6
MIN_MINUTES_PER_DAY = 30
FORMULA_VERSION_SESSION = "apm_active_v2_ewm5_windows_slope"


def bartime_to_seconds(bartime: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(bartime):
        return (
            bartime.dt.hour.astype(np.int64) * 3600
            + bartime.dt.minute.astype(np.int64) * 60
            + bartime.dt.second.astype(np.int64)
        )
    if pd.api.types.is_timedelta64_dtype(bartime):
        return pd.to_timedelta(bartime).dt.total_seconds().astype(np.float64)
    return pd.to_numeric(bartime, errors="coerce")


def _window_imbalance(minutes: pd.DataFrame, start: int, end: int) -> float:
    bt = minutes["bartime_sec"] if "bartime_sec" in minutes.columns else bartime_to_seconds(
        minutes["bartime"]
    )
    mask = (bt >= start) & (bt < end)
    w = minutes.loc[mask]
    if w.empty:
        return np.nan
    net = float(w["active_buy_amt"].sum() - w["active_sell_amt"].sum())
    total = float(w["amount"].sum())
    if total <= 0 or not np.isfinite(total):
        return np.nan
    return float(net / total)


def _intraday_slope(minutes: pd.DataFrame, n_slices: int = N_INTRADAY_SLICES) -> float:
    if len(minutes) < 2:
        return np.nan
    df = minutes.sort_values("bartime")
    bt = (
        df["bartime_sec"].to_numpy(dtype=float)
        if "bartime_sec" in df.columns
        else bartime_to_seconds(df["bartime"]).to_numpy(dtype=float)
    )
    t_min, t_max = float(np.nanmin(bt)), float(np.nanmax(bt))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max - t_min) < 60:
        return np.nan

    bins = np.linspace(t_min, t_max, n_slices + 1)
    buy = df["active_buy_amt"].to_numpy(dtype=float)
    sell = df["active_sell_amt"].to_numpy(dtype=float)
    amt = df["amount"].to_numpy(dtype=float)

    imbalances = np.full(n_slices, np.nan, dtype=float)
    for i in range(n_slices):
        if i < n_slices - 1:
            mask = (bt >= bins[i]) & (bt < bins[i + 1])
        else:
            mask = (bt >= bins[i]) & (bt <= bins[i + 1])
        if not mask.any():
            continue
        total = float(amt[mask].sum())
        if total <= 0 or not np.isfinite(total):
            continue
        net = float(buy[mask].sum() - sell[mask].sum())
        imbalances[i] = net / total

    valid = np.isfinite(imbalances)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(1, n_slices + 1, dtype=float)[valid]
    y = imbalances[valid]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def _weighted_apm_raw(mo: float, mc: float, slope: float) -> float:
    comps = ((MO_WEIGHT, mo), (MC_WEIGHT, mc), (SLOPE_WEIGHT, slope))
    w_sum = 0.0
    v_sum = 0.0
    for w, v in comps:
        if np.isfinite(v):
            w_sum += w
            v_sum += w * float(v)
    if w_sum <= 0:
        return np.nan
    return float(v_sum / w_sum)


def compute_daily_apm_session_cut(
    minutes: pd.DataFrame,
    min_minutes: int = MIN_MINUTES_PER_DAY,
) -> pd.DataFrame:
    need = {"date", "symbol", "bartime", "active_buy_amt", "active_sell_amt", "amount"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"APM session-cut minutes missing columns: {missing}")

    df = minutes.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["bartime_sec"] = bartime_to_seconds(df["bartime"])

    rows: list[tuple] = []
    for (sym, d), g in df.groupby(["symbol", "date"], sort=False):
        if len(g) < min_minutes:
            rows.append((d, sym, np.nan, np.nan, np.nan, np.nan))
            continue
        mo = _window_imbalance(g, OPEN_WINDOW_START, OPEN_WINDOW_END)
        mc = _window_imbalance(g, CLOSE_WINDOW_START, CLOSE_WINDOW_END)
        slope = _intraday_slope(g)
        raw = _weighted_apm_raw(mo, mc, slope)
        rows.append((d, sym, mo, mc, slope, raw))

    out = pd.DataFrame(
        rows,
        columns=["date", "symbol", "mo_imb", "mc_imb", "intraday_slope", "apm_raw"],
    )
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
