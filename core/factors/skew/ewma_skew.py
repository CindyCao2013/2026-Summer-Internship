"""EWMA-weighted rolling skewness for idiosyncratic residuals.

Keeps formation window length fixed (e.g. 60) but applies exponential decay
weights inside the window so recent residuals dominate the third moment.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from core.factors.skew.idio_skew import rolling_market_residual
from core.factors.skew.skew import alpha_from_skew

_DEFAULT_MIN = {60: 40, 120: 80}


def ewma_window_weights(window: int, half_life: float) -> np.ndarray:
    """Weights for a length-``window`` slice; last index = most recent day."""
    if window < 3:
        raise ValueError("window must be >= 3")
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    # age: oldest = window-1, newest = 0
    ages = np.arange(window - 1, -1, -1, dtype=float)
    w = np.power(0.5, ages / float(half_life))
    return w / w.sum()


def _weighted_skew_1d(col: np.ndarray, w: np.ndarray, min_periods: int) -> np.ndarray:
    L = len(w)
    T = len(col)
    out = np.full(T, np.nan, dtype=float)
    finite = np.isfinite(col)
    if int(finite.sum()) < min_periods:
        return out
    for t in range(L - 1, T):
        sl = slice(t - L + 1, t + 1)
        m = finite[sl]
        n_valid = int(m.sum())
        if n_valid < min_periods:
            continue
        x = col[sl][m]
        ww = w[m]
        s = ww.sum()
        if s <= 0:
            continue
        ww = ww / s
        mu = float(ww @ x)
        d = x - mu
        m2 = float(ww @ (d * d))
        if m2 <= 1e-18:
            continue
        m3 = float(ww @ (d * d * d))
        out[t] = m3 / (m2 ** 1.5)
    return out


def rolling_ewma_skew(
    panel: pd.DataFrame,
    window: int,
    half_life: float,
    *,
    min_periods: Optional[int] = None,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Rolling EWMA-weighted skewness (columns processed in parallel)."""
    mp = _DEFAULT_MIN.get(window, max(5, window // 2))
    if min_periods is not None:
        mp = int(min_periods)
    w = ewma_window_weights(window, half_life)
    values = panel.to_numpy(dtype=float, copy=False)
    cols = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_weighted_skew_1d)(values[:, j], w, mp) for j in range(values.shape[1])
    )
    return pd.DataFrame(np.column_stack(cols), index=panel.index, columns=panel.columns)


def build_idio_skew_ewma(
    ret_1d: pd.DataFrame,
    market_ret: pd.Series,
    *,
    window: int = 60,
    half_lives: Iterable[float] = (10, 15, 20),
    as_alpha: bool = False,
    min_periods: Optional[int] = None,
    n_jobs: int = 8,
) -> Dict[str, pd.DataFrame]:
    """Idio residual → EWMA skew panels.

    Keys: ``IdioSKEW{w}_EWMA{hl}`` or ``AlphaIdioSKEW{w}_EWMA{hl}``.
    """
    resid = rolling_market_residual(ret_1d, market_ret, window, min_periods=min_periods)
    out: Dict[str, pd.DataFrame] = {}
    for hl in half_lives:
        hl = float(hl)
        raw = rolling_ewma_skew(
            resid, window, hl, min_periods=min_periods, n_jobs=n_jobs
        )
        tag = f"IdioSKEW{window}_EWMA{int(hl)}"
        if as_alpha:
            out[f"Alpha{tag}"] = alpha_from_skew(raw)
        else:
            out[tag] = raw
    return out


def build_forward_return(ret_1d: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """On date ``d``: cumulative return over ``[d, d+horizon-1]`` (inclusive)."""
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    if h == 1:
        return ret_1d.copy()
    log1 = np.log1p(ret_1d.astype(float))
    # reverse-rolling sum = sum of current and next h-1 days
    fut = log1.iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]
    return np.expm1(fut)
