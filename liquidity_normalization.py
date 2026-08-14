"""Liquidity normalization layer: size-adjusted state variables + cross-sectional orthogonalization.

Level-1 EOD OHLCV collapses amount / volume / turnover into the same latent liquidity state
when not normalized by float market cap. This module defines the normalized liquidity space
and panel residualization used by factor_formulas_liquidity_norm.py.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

_EPS = 1e-6


def rolling_cv(df: pd.DataFrame, window: int = 20, min_periods: int = 10) -> pd.DataFrame:
    """Coefficient of variation: std / mean over rolling window."""
    mean = df.rolling(window, min_periods=min_periods).mean()
    std = df.rolling(window, min_periods=min_periods).std()
    return std / mean.replace(0, np.nan)


def turnover_proxy(amount: pd.DataFrame, float_mktcap: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy daily turnover intensity when S_DQ_TURN is unavailable.

    amount: 千元; float_mktcap: 万元 (Wind convention).
    Absolute scale cancels in CV; ratio captures cross-sectional liquidity intensity.
    """
    return amount / float_mktcap.replace(0, np.nan)


def effective_turnover(
    turnover: Optional[pd.DataFrame],
    amount: pd.DataFrame,
    float_mktcap: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer exchange turnover; fall back to amount / float_mktcap proxy."""
    if turnover is not None:
        return turnover
    return turnover_proxy(amount, float_mktcap)


def panel_cross_sectional_residual(
    y: pd.DataFrame,
    xs: List[pd.DataFrame],
    min_obs: int = 30,
) -> pd.DataFrame:
    """
    Daily cross-sectional OLS: y_t = alpha + sum(beta_k * x_k,t) + epsilon_t.

    Returns epsilon with the same index/columns as y.
    """
    resid = y.copy() * np.nan
    x_names = [f"x{i}" for i in range(len(xs))]

    for dt in y.index:
        row_y = y.loc[dt]
        mat = pd.DataFrame({name: x.loc[dt] for name, x in zip(x_names, xs)})
        panel = pd.concat([row_y.rename("y"), mat], axis=1).dropna()
        if len(panel) < max(min_obs, len(xs) + 5):
            continue

        y_vec = panel["y"].values
        x_mat = panel[x_names].values
        design = np.column_stack([np.ones(len(y_vec)), x_mat])
        coef, _, _, _ = np.linalg.lstsq(design, y_vec, rcond=None)
        pred = design @ coef
        resid.loc[dt, panel.index] = y_vec - pred

    return resid


def rolling_autocorr_1(
    df: pd.DataFrame, window: int = 20, min_periods: int = 10
) -> pd.DataFrame:
    """Vectorized lag-1 autocorrelation over rolling window."""
    lagged = df.shift(1)
    mean = df.rolling(window, min_periods=min_periods).mean()
    mean_lag = lagged.rolling(window, min_periods=min_periods).mean()
    cov = ((df - mean) * (lagged - mean_lag)).rolling(window, min_periods=min_periods).mean()
    std = df.rolling(window, min_periods=min_periods).std()
    std_lag = lagged.rolling(window, min_periods=min_periods).std()
    return cov / (std * std_lag).replace(0, np.nan)


def factor_correlation_matrix(factor_dict: dict, min_overlap: int = 500) -> pd.DataFrame:
    """Pairwise Pearson correlation on stacked (date, stock) observations."""
    stacked = {}
    for name, wide in factor_dict.items():
        s = wide.stack(dropna=True)
        s.name = name
        stacked[name] = s

    panel = pd.concat(stacked.values(), axis=1, join="inner").dropna()
    if len(panel) < min_overlap:
        raise ValueError(f"Insufficient overlap for correlation ({len(panel)} < {min_overlap})")
    return panel.corr()
