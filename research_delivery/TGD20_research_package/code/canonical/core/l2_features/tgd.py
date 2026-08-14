"""TGD factor wrapper (Temporal Feature Layer Stage 3).

Thin module — consumes Stage-2 residuals only.

  Step 1 (daily cross-section):
      εd_i = α + β·εu_i + ε_i

  Step 2 (per symbol, time series):
      TGD20_t = MA_20(ε_t)

Does NOT recompute Gu/Gd, minute returns, R1/R2, or control residualization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.l2_features.timing_residual import DEFAULT_MIN_CS, cs_ols_residual

DEFAULT_TGD_WINDOW = 20


def daily_tgd_innovation(
    residual_df: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    eps_u_col: str = "epsilon_u",
    eps_d_col: str = "epsilon_d",
    min_obs: int = DEFAULT_MIN_CS,
) -> pd.DataFrame:
    """Daily CS: εd ~ εu → innovation ε (column `tgd_eps`)."""
    need = {date_col, symbol_col, eps_u_col, eps_d_col}
    missing = need - set(residual_df.columns)
    if missing:
        raise ValueError(f"daily_tgd_innovation missing columns: {sorted(missing)}")

    parts = []
    for dt, g in residual_df.groupby(date_col, sort=True):
        g = g.copy()
        eps = cs_ols_residual(
            g[eps_d_col].to_numpy(dtype=float),
            g[[eps_u_col]].to_numpy(dtype=float),
            min_obs=min_obs,
        )
        g["tgd_eps"] = eps
        parts.append(g)

    if not parts:
        out = residual_df.copy()
        out["tgd_eps"] = np.nan
        return out

    out = pd.concat(parts, ignore_index=True)
    out.attrs["tgd_stage"] = "tgd_innovation"
    return out


def smooth_tgd(
    innovation_df: pd.DataFrame,
    *,
    window: int = DEFAULT_TGD_WINDOW,
    date_col: str = "date",
    symbol_col: str = "symbol",
    eps_col: str = "tgd_eps",
    out_col: str = "TGD20",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Per-symbol rolling mean of daily TGD innovations."""
    if window <= 0:
        raise ValueError("window must be positive")
    mp = window if min_periods is None else min_periods
    need = {date_col, symbol_col, eps_col}
    missing = need - set(innovation_df.columns)
    if missing:
        raise ValueError(f"smooth_tgd missing columns: {sorted(missing)}")

    df = innovation_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])
    df[out_col] = (
        df.groupby(symbol_col, sort=False)[eps_col]
        .transform(lambda s: s.rolling(window, min_periods=mp).mean())
    )
    df.attrs["tgd_stage"] = "tgd20"
    df.attrs["tgd_window"] = window
    return df


def build_tgd20(
    residual_df: pd.DataFrame,
    *,
    window: int = DEFAULT_TGD_WINDOW,
    min_obs: int = DEFAULT_MIN_CS,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Stage-3 one-shot: residuals → daily ε → TGD20.

    Input columns: date, symbol, epsilon_u, epsilon_d
    Output adds: tgd_eps, TGD20
    """
    innov = daily_tgd_innovation(
        residual_df,
        date_col=date_col,
        symbol_col=symbol_col,
        min_obs=min_obs,
    )
    return smooth_tgd(
        innov,
        window=window,
        date_col=date_col,
        symbol_col=symbol_col,
    )


def tgd20_to_wide(
    tgd_df: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    value_col: str = "TGD20",
) -> pd.DataFrame:
    """Pivot long TGD20 to Date×Symbol wide panel for 10-group evaluation."""
    df = tgd_df[[date_col, symbol_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    wide = df.pivot(index=date_col, columns=symbol_col, values=value_col)
    wide.index.name = "Date"
    return wide.sort_index()
