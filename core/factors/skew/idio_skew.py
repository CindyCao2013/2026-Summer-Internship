"""Idiosyncratic skewness (CICC-style main research version).

For each stock and date t, form a market-model residual series over window L:

    r_{i,s} = α_{i,t} + β_{i,t} r_{m,s} + ε_{i,s},   s ∈ [t-L+1, t]

then

    IdioSKEW_{i,t}^{(L)} = Skew(ε_{i,s})

Implementation note
-------------------
We use a vectorized rolling market-model residual:

    β_t = Cov_L(r_i, r_m) / Var_L(r_m)
    α_t = Mean_L(r_i) - β_t Mean_L(r_m)
    ε_t = r_{i,t} - α_t - β_t r_{m,t}

then take rolling skewness of ε. This is the standard industry approximation to
per-window OLS residual skew and avoids O(N·T·L) Python loops on the full A-share
panel. Market return defaults to CSI300 daily c2c.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from core.factors.skew.skew import alpha_from_skew

IDIO_SKEW_WINDOWS = (60, 120)

_DEFAULT_MIN_PERIODS: Mapping[int, int] = {
    60: 40,
    120: 80,
}


def rolling_market_residual(
    ret_1d: pd.DataFrame,
    market_ret: pd.Series,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Rolling one-factor market-model residual panel (intercept + beta)."""
    if window <= 2:
        raise ValueError("window must be > 2")
    mp = _DEFAULT_MIN_PERIODS.get(window, max(5, window // 2))
    if min_periods is not None:
        mp = int(min_periods)

    rm = market_ret.reindex(ret_1d.index).astype(float)
    y = ret_1d.astype(float)

    rm_mean = rm.rolling(window, min_periods=mp).mean()
    rm_var = rm.rolling(window, min_periods=mp).var(ddof=0)
    y_mean = y.rolling(window, min_periods=mp).mean()
    xy_mean = y.mul(rm, axis=0).rolling(window, min_periods=mp).mean()
    cov = xy_mean.sub(y_mean.mul(rm_mean, axis=0))
    beta = cov.div(rm_var.replace(0.0, np.nan), axis=0)
    alpha = y_mean.sub(beta.mul(rm_mean, axis=0))
    resid = y.sub(alpha).sub(beta.mul(rm, axis=0))
    return resid


def build_idio_skew(
    ret_1d: pd.DataFrame,
    market_ret: pd.Series,
    windows: Iterable[int] = IDIO_SKEW_WINDOWS,
    *,
    as_alpha: bool = False,
    min_periods: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Build named idiosyncratic-skew panels.

    Keys: ``IdioSKEW{w}`` or ``AlphaIdioSKEW{w}``.
    """
    out: Dict[str, pd.DataFrame] = {}
    for w in windows:
        w = int(w)
        resid = rolling_market_residual(
            ret_1d, market_ret, w, min_periods=min_periods
        )
        mp = _DEFAULT_MIN_PERIODS.get(w, max(5, w // 2))
        if min_periods is not None:
            mp = int(min_periods)
        raw = resid.rolling(w, min_periods=mp).skew()
        key = f"AlphaIdioSKEW{w}" if as_alpha else f"IdioSKEW{w}"
        out[key] = alpha_from_skew(raw) if as_alpha else raw
    return out


def idio_skew_60(
    ret_1d: pd.DataFrame,
    market_ret: pd.Series,
    *,
    as_alpha: bool = False,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    return build_idio_skew(
        ret_1d, market_ret, windows=(60,), as_alpha=as_alpha, min_periods=min_periods
    )["AlphaIdioSKEW60" if as_alpha else "IdioSKEW60"]


def idio_skew_120(
    ret_1d: pd.DataFrame,
    market_ret: pd.Series,
    *,
    as_alpha: bool = False,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    return build_idio_skew(
        ret_1d, market_ret, windows=(120,), as_alpha=as_alpha, min_periods=min_periods
    )["AlphaIdioSKEW120" if as_alpha else "IdioSKEW120"]
