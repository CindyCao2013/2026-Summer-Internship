"""active_pressure brick — minute Active_* buy/sell pressure imbalance.

Observable: amount-weighted daily active buy pressure vs sell pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BRICK_VERSION = "active_pressure_v1_amt_weighted"
PRESSURE_COL = "apm_raw"
PRESSURE_EWM_COL = "apm_smooth"
EWM_SPAN = 5
EWM_MIN_PERIODS = 3
MIN_MINUTES_PER_DAY = 30
EPS = 1e-12


def minute_raw_apm(buy: pd.Series, sell: pd.Series) -> pd.Series:
    """(buy - sell) / (buy + sell); NaN when denom ~ 0."""
    buy_f = pd.to_numeric(buy, errors="coerce").astype(float)
    sell_f = pd.to_numeric(sell, errors="coerce").astype(float)
    denom = buy_f + sell_f
    out = (buy_f - sell_f) / denom
    out = out.where(denom > EPS)
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    w = weights[mask]
    v = values[mask]
    w_sum = float(w.sum())
    if w_sum <= 0:
        return np.nan
    return float(np.dot(v, w) / w_sum)


def compute_daily_active_pressure(
    minutes: pd.DataFrame,
    min_minutes: int = MIN_MINUTES_PER_DAY,
) -> pd.DataFrame:
    """Per (symbol, date) amount-weighted active pressure.

    Required: date, symbol, active_buy_amt, active_sell_amt, amount.
    Returns [date, symbol, apm_raw] with apm_raw in [-1, 1] when defined.
    """
    need = {"date", "symbol", "active_buy_amt", "active_sell_amt", "amount"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"active_pressure minutes missing columns: {missing}")

    df = minutes.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["raw_apm"] = minute_raw_apm(df["active_buy_amt"], df["active_sell_amt"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").astype(float)

    rows: list[dict] = []
    for (sym, d), g in df.groupby(["symbol", "date"], sort=False):
        if len(g) < min_minutes:
            rows.append({"date": pd.Timestamp(d), "symbol": sym, PRESSURE_COL: np.nan})
            continue
        val = _weighted_mean(
            g["raw_apm"].to_numpy(dtype=float),
            g["amount"].to_numpy(dtype=float),
        )
        rows.append({"date": pd.Timestamp(d), "symbol": sym, PRESSURE_COL: val})

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", PRESSURE_COL])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def smooth_active_pressure(
    daily: pd.DataFrame,
    span: int = EWM_SPAN,
    min_periods: int = EWM_MIN_PERIODS,
    value_col: str = PRESSURE_COL,
    out_col: str = PRESSURE_EWM_COL,
) -> pd.DataFrame:
    """EWM-smooth daily pressure per symbol."""
    if value_col not in daily.columns:
        raise ValueError(f"daily missing {value_col}")
    out = daily.sort_values(["symbol", "date"]).copy()
    out[out_col] = out.groupby("symbol", sort=False)[value_col].transform(
        lambda x: x.ewm(span=span, min_periods=min_periods).mean()
    )
    return out
