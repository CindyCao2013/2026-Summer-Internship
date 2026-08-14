"""Timing residual engine (Temporal Feature Layer Stage 2).

Open Source Securities TGD methodology — interference stripping only.

For each trading day, cross-sectional OLS:

  Gu_i = α + β1·Rū_i + β2·R1_i + β3·R2_i + β4·R_ovn_i + εu_i
  Gd_i = α + β1·Rd̄_i + β2·R1_i + β3·R2_i + β4·R_ovn_i + εd_i

NOT in this module (Stage 3 — see tgd.py):
  - εd ~ εu cross-sectional regression (daily innovation)
  - MA20 smoothing
  - final TGD20 portfolio factor / evaluation

Conventions for session controls (documented; adjustable):
  R1  — cum return over trading-minute indices [0, r1_end]   default 0..29  (≈09:31–10:00)
  R2  — cum return over (r1_end, r2_end]                    default 30..59 (≈10:01–10:30)
  R_overnight — previous close → today's open (passed in or attached)

Avoidance:
  - daily CS fit only (no pooled TS regression)
  - no lookahead beyond same-day known features
  - rows with insufficient peers → NaN residuals
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.l2_features.return_timing import (
    compute_minute_returns,
    trading_minute_index,
)

# Default control column names (accept aliases via normalize)
GU_COL = "Gu"
GD_COL = "Gd"
RU_COL = "avg_up_return"  # Rū / mean_up
RD_COL = "avg_down_return"  # Rd̄ / mean_down
R1_COL = "R1"
R2_COL = "R2"
OVN_COL = "overnight_return"

DEFAULT_MIN_CS = 30  # minimum names with full feature vector per day


def _as_float_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, list(cols)].to_numpy(dtype=float)


def cs_ols_residual(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
    min_obs: int = DEFAULT_MIN_CS,
) -> np.ndarray:
    """Cross-sectional OLS residuals for one day.

    X is (n, k) without intercept; intercept appended if add_intercept.
    Returns length-n residual vector (NaN where y/X invalid or n < min_obs).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    row_ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if int(row_ok.sum()) < min_obs:
        return out

    yy = y[row_ok]
    XX = X[row_ok]
    if add_intercept:
        XX = np.column_stack([np.ones(XX.shape[0]), XX])

    # ridge-less lstsq; rank-deficient → NaN for all that day
    try:
        beta, _, rank, _ = np.linalg.lstsq(XX, yy, rcond=None)
        if rank < XX.shape[1]:
            return out
        fitted = XX @ beta
        resid = yy - fitted
    except np.linalg.LinAlgError:
        return out

    out[row_ok] = resid
    return out


def normalize_timing_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Stage-1 aliases (mean_up/mean_down) onto residual control names."""
    out = df.copy()
    if RU_COL not in out.columns and "mean_up" in out.columns:
        out[RU_COL] = out["mean_up"]
    if RD_COL not in out.columns and "mean_down" in out.columns:
        out[RD_COL] = out["mean_down"]
    return out


def residualize_timing_centers(
    daily: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    min_obs: int = DEFAULT_MIN_CS,
    gu_controls: Optional[Sequence[str]] = None,
    gd_controls: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Daily CS residualization of Gu/Gd → epsilon_u / epsilon_d.

    Required columns after normalize:
      Gu, Gd, avg_up_return, avg_down_return, R1, R2, overnight_return
    """
    df = normalize_timing_feature_columns(daily)
    gu_x = list(gu_controls) if gu_controls is not None else [RU_COL, R1_COL, R2_COL, OVN_COL]
    gd_x = list(gd_controls) if gd_controls is not None else [RD_COL, R1_COL, R2_COL, OVN_COL]

    need = {GU_COL, GD_COL, date_col, symbol_col, *gu_x, *gd_x}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"residualize_timing_centers missing columns: {sorted(missing)}")

    parts = []
    for dt, g in df.groupby(date_col, sort=True):
        g = g.copy()
        eps_u = cs_ols_residual(
            g[GU_COL].to_numpy(dtype=float),
            _as_float_matrix(g, gu_x),
            min_obs=min_obs,
        )
        eps_d = cs_ols_residual(
            g[GD_COL].to_numpy(dtype=float),
            _as_float_matrix(g, gd_x),
            min_obs=min_obs,
        )
        g["epsilon_u"] = eps_u
        g["epsilon_d"] = eps_d
        g["n_cs"] = int(np.isfinite(eps_u).sum())
        parts.append(g)

    if not parts:
        out = df.copy()
        out["epsilon_u"] = np.nan
        out["epsilon_d"] = np.nan
        out["n_cs"] = 0
        return out

    out = pd.concat(parts, ignore_index=True)
    out.attrs["tgd_stage"] = "timing_residual"
    return out


def segment_cum_return(
    returns: np.ndarray,
    time_index: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> float:
    """Sum minute returns with t in (t_lo, t_hi] (or [0,t_hi] if t_lo < 0)."""
    r = np.asarray(returns, dtype=float)
    t = np.asarray(time_index, dtype=float)
    if t_lo < 0:
        mask = np.isfinite(r) & np.isfinite(t) & (t >= 0) & (t <= t_hi)
    else:
        mask = np.isfinite(r) & np.isfinite(t) & (t > t_lo) & (t <= t_hi)
    if not mask.any():
        return float("nan")
    return float(np.nansum(r[mask]))


def attach_session_controls_from_minute(
    minute_df: pd.DataFrame,
    *,
    close_col: str = "close",
    bartime_col: str = "bartime",
    date_col: str = "date",
    symbol_col: str = "symbol",
    open_col: Optional[str] = None,
    prev_close_col: Optional[str] = None,
    overnight: Optional[pd.DataFrame] = None,
    r1_end: int = 29,
    r2_end: int = 59,
) -> pd.DataFrame:
    """Build R1/R2/(optional overnight) controls per date×symbol from minute bars.

    overnight: optional long df [date, symbol, overnight_return], or columns
    open_col + prev_close_col on minute_df (first bar / day-level repeated).
    """
    required = {date_col, symbol_col, bartime_col, close_col}
    missing = required - set(minute_df.columns)
    if missing:
        raise ValueError(f"attach_session_controls_from_minute missing: {sorted(missing)}")

    rows = []
    for (date, symbol), g in minute_df.groupby([date_col, symbol_col], sort=False):
        g = g.sort_values(bartime_col)
        t_idx = trading_minute_index(g[bartime_col])
        rets = compute_minute_returns(g[close_col].to_numpy(dtype=float))
        r1 = segment_cum_return(rets, t_idx, -1, r1_end)
        r2 = segment_cum_return(rets, t_idx, r1_end, r2_end)
        ovn = np.nan
        if open_col and prev_close_col and open_col in g.columns and prev_close_col in g.columns:
            o = float(g[open_col].iloc[0])
            pc = float(g[prev_close_col].iloc[0])
            if np.isfinite(o) and np.isfinite(pc) and pc > 0:
                ovn = o / pc - 1.0
        rows.append(
            {
                date_col: pd.Timestamp(date),
                symbol_col: str(symbol),
                R1_COL: r1,
                R2_COL: r2,
                OVN_COL: ovn,
            }
        )
    ctrl = pd.DataFrame(rows)
    if overnight is not None:
        ov = overnight[[date_col, symbol_col, OVN_COL]].copy()
        ov[date_col] = pd.to_datetime(ov[date_col])
        ctrl = ctrl.drop(columns=[OVN_COL]).merge(ov, on=[date_col, symbol_col], how="left")
    return ctrl


def merge_centers_with_controls(
    centers: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Join Gu/Gd daily centers with R1/R2/overnight (+ keep mean_up/mean_down aliases)."""
    left = normalize_timing_feature_columns(centers)
    left[date_col] = pd.to_datetime(left[date_col])
    right = controls.copy()
    right[date_col] = pd.to_datetime(right[date_col])
    return left.merge(right, on=[date_col, symbol_col], how="left")


# --- Stage-3 hook (stub) ---------------------------------------------------------

def prepare_tgd_from_residuals(residual_df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through for Stage-3 TGD wrapper (εd ~ εu → MA20).

    Does NOT compute TGD — only validates residual columns exist.
    """
    need = {"date", "symbol", "epsilon_u", "epsilon_d"}
    if not need.issubset(residual_df.columns):
        raise ValueError(f"prepare_tgd_from_residuals needs {need}")
    out = residual_df.copy()
    out.attrs["tgd_stage"] = "residuals_ready"
    return out
