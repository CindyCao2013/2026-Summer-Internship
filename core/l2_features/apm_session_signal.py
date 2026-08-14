"""APM_SessionResidual Phase2 — paper statistic object from residual panel.

Construct only:
  residual_panel.delta_alpha → rolling APM_stat (w=20)
  optional CS residual vs Ret20 (constructability)

No IC / Sharpe / pack. No shift(1) inside outputs.
Does not touch factor_cutting.active_trade.compute_apm().
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 20
DEFAULT_MIN_PERIODS = 10
MIN_CS_NAMES = 50


def _refuse_active_columns(df: pd.DataFrame, where: str) -> None:
    bad = [c for c in df.columns if str(c).lower().startswith("active_")]
    if bad:
        raise RuntimeError(f"Active_* columns forbidden in APM signal ({where}): {bad}")


def build_apm_stat_panel(
    residual_panel: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Rolling t-stat of delta_alpha by symbol (paper-shaped APM_stat).

    APM_stat = mean(delta) / (std(delta) / sqrt(n)) over ``window`` trading rows.
    Output dated T is unshifted (raw economic object).
    """
    need = {"date", "symbol", "delta_alpha"}
    missing = need - set(residual_panel.columns)
    if missing:
        raise ValueError(f"residual_panel missing columns: {sorted(missing)}")
    _refuse_active_columns(residual_panel, "residual_panel")

    df = residual_panel[["date", "symbol", "delta_alpha"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    def _roll_tstat(s: pd.Series) -> pd.Series:
        mu = s.rolling(window, min_periods=min_periods).mean()
        sd = s.rolling(window, min_periods=min_periods).std(ddof=1)
        n = s.rolling(window, min_periods=min_periods).count()
        return mu / (sd / np.sqrt(n.replace(0, np.nan)))

    df["apm_stat"] = df.groupby("symbol", sort=False)["delta_alpha"].transform(_roll_tstat)
    df["n_obs"] = (
        df.groupby("symbol", sort=False)["delta_alpha"]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).count())
        .astype(float)
    )
    _refuse_active_columns(df, "apm_stat")
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def compute_apm_session_residual_signal(
    residual_panel: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> pd.DataFrame:
    """Public API: residual panel → long [date, symbol, APM_stat, ...]."""
    out = build_apm_stat_panel(
        residual_panel, window=window, min_periods=min_periods
    )
    return out.rename(columns={"apm_stat": "APM_stat"})


def build_ret20_long(ret_c2c: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Ret20 = rolling sum of daily c2c returns (wide → long)."""
    wide = ret_c2c.sort_index()
    ret20 = wide.rolling(window, min_periods=max(5, window // 2)).sum()
    long = ret20.stack(dropna=False).rename("ret20").reset_index()
    long.columns = ["date", "symbol", "ret20"]
    long["date"] = pd.to_datetime(long["date"])
    long["symbol"] = long["symbol"].astype(str)
    return long


def cs_residualize_vs_ret20(
    apm_stat_long: pd.DataFrame,
    ret20_long: pd.DataFrame,
    *,
    signal_col: str = "apm_stat",
    min_names: int = MIN_CS_NAMES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-date CS OLS residual of signal on Ret20 + const (constructability).

    Returns (long panel with apm_cs, daily alignment coverage table).
    """
    if signal_col not in apm_stat_long.columns and signal_col == "apm_stat":
        if "APM_stat" in apm_stat_long.columns:
            signal_col = "APM_stat"

    sig = apm_stat_long[["date", "symbol", signal_col]].copy()
    sig["date"] = pd.to_datetime(sig["date"])
    ret = ret20_long[["date", "symbol", "ret20"]].copy()
    ret["date"] = pd.to_datetime(ret["date"])

    merged = sig.merge(ret, on=["date", "symbol"], how="left")
    merged = merged.rename(columns={signal_col: "apm_stat"})

    rows = []
    align_rows = []
    for d, g in merged.groupby("date", sort=True):
        y = g["apm_stat"].to_numpy(dtype=float)
        x = g["ret20"].to_numpy(dtype=float)
        n_apm = int(np.isfinite(y).sum())
        n_ret = int(np.isfinite(x).sum())
        m = np.isfinite(y) & np.isfinite(x)
        n_valid = int(m.sum())
        align_rows.append(
            {
                "date": pd.Timestamp(d),
                "n_APM": n_apm,
                "n_ret20": n_ret,
                "n_valid": n_valid,
                "coverage": float(n_valid / len(g)) if len(g) else 0.0,
            }
        )
        apm_cs = np.full(len(g), np.nan, dtype=float)
        if n_valid >= min_names:
            yy = y[m]
            xx = x[m]
            # OLS: y ~ 1 + x
            X = np.column_stack([np.ones(n_valid), xx])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
                fitted = X @ beta
                resid = yy - fitted
                apm_cs[np.where(m)[0]] = resid
            except Exception:
                pass
        part = g.copy()
        part["apm_cs"] = apm_cs
        rows.append(part)

    out = pd.concat(rows, ignore_index=True) if rows else merged.assign(apm_cs=np.nan)
    align = pd.DataFrame(align_rows)
    _refuse_active_columns(out, "apm_cs")
    return out.sort_values(["date", "symbol"]).reset_index(drop=True), align


def distribution_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One row per column: mean/std/quantiles/pct_nan + mean daily CS std."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        finite = s[np.isfinite(s.to_numpy(dtype=float))]
        # daily CS std
        tmp = df[["date"]].copy()
        tmp["_v"] = s
        daily_std = (
            tmp.groupby("date")["_v"]
            .apply(lambda x: float(np.nanstd(x.to_numpy(dtype=float), ddof=1)) if np.isfinite(x).sum() >= 2 else np.nan)
        )
        rows.append(
            {
                "column": c,
                "n": int(len(s)),
                "n_finite": int(len(finite)),
                "pct_nan": float(1.0 - len(finite) / len(s)) if len(s) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "p01": float(finite.quantile(0.01)) if len(finite) else np.nan,
                "p50": float(finite.quantile(0.50)) if len(finite) else np.nan,
                "p99": float(finite.quantile(0.99)) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "mean_daily_cs_std": float(daily_std.mean()) if len(daily_std) else np.nan,
                "min_daily_cs_std": float(daily_std.min()) if len(daily_std) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def daily_coverage_table(signal_long: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for d, g in signal_long.groupby("date", sort=True):
        row = {"date": pd.Timestamp(d), "n_names": int(g["symbol"].nunique())}
        for c in cols:
            if c not in g.columns:
                continue
            v = pd.to_numeric(g[c], errors="coerce")
            row[f"frac_finite_{c}"] = float(np.isfinite(v.to_numpy(dtype=float)).mean())
            row[f"n_finite_{c}"] = int(np.isfinite(v.to_numpy(dtype=float)).sum())
        rows.append(row)
    return pd.DataFrame(rows)
