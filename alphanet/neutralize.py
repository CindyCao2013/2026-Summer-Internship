"""Five-factor cross-sectional neutralization (guide §4.4).

Factors: industry, log size, momentum(m), volatility(m), turnover(m).
Industry uses dummy OLS; remaining styles are continuous regressors.
Falls back to Citics L1 when Shenwan L1 is not loaded on the panel.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from alphanet.config import EvalConfig


MIN_OBS = 30


def rolling_cs_mean(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    return panel.rolling(int(window), min_periods=max(2, int(window) // 2)).mean()


def rolling_cs_std(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    return panel.rolling(int(window), min_periods=max(3, int(window) // 2)).std()


def style_panels(
    ret_1d: pd.DataFrame,
    turn: pd.DataFrame,
    horizon: int,
) -> Dict[str, pd.DataFrame]:
    m = int(horizon)
    # momentum: past m-day return, shifted 1 so it does not include the forward window
    mom = np.expm1(np.log1p(ret_1d).rolling(m, min_periods=max(2, m // 2)).sum()).shift(1)
    vol = ret_1d.rolling(m, min_periods=max(3, m // 2)).std().shift(1)
    to = rolling_cs_mean(turn, m).shift(1)
    return {"momentum": mom, "volatility": vol, "turnover": to}


def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    resid = np.full(y.shape[0], np.nan, dtype=float)
    ok = np.isfinite(y)
    for j in range(X.shape[1]):
        ok &= np.isfinite(X[:, j])
    if int(ok.sum()) < MIN_OBS:
        return resid
    Y = y[ok]
    design = np.column_stack([np.ones(int(ok.sum())), X[ok]])
    try:
        beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid
    resid[ok] = Y - design @ beta
    return resid


def neutralize_cross_section(
    y: pd.Series,
    industry: Optional[pd.Series],
    extras: Dict[str, pd.Series],
    min_obs: int = MIN_OBS,
) -> pd.Series:
    idx = y.index
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    cols = []
    if industry is not None:
        g = industry.reindex(idx)
        dummies = pd.get_dummies(g, dummy_na=False)
        if dummies.shape[1] >= 2:
            cols.append(dummies.to_numpy(dtype=float)[:, 1:])  # drop one level
        elif dummies.shape[1] == 1:
            cols.append(dummies.to_numpy(dtype=float))
    for name, series in extras.items():
        cols.append(pd.to_numeric(series.reindex(idx), errors="coerce").to_numpy(dtype=float).reshape(-1, 1))
    if not cols:
        return y.copy()
    X = np.column_stack(cols)
    if X.shape[0] < min_obs:
        return pd.Series(np.nan, index=idx)
    resid = _ols_residual(yv, X)
    return pd.Series(resid, index=idx)


def neutralize_panel(
    signal: pd.DataFrame,
    *,
    industry: Optional[pd.DataFrame] = None,
    log_mcap: Optional[pd.DataFrame] = None,
    ret_1d: Optional[pd.DataFrame] = None,
    turn: Optional[pd.DataFrame] = None,
    horizon: int = 10,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    extras_panels: Dict[str, pd.DataFrame] = {}
    if log_mcap is not None:
        extras_panels["size"] = log_mcap.reindex(index=signal.index, columns=signal.columns)
    if ret_1d is not None and turn is not None:
        styles = style_panels(
            ret_1d.reindex(index=signal.index, columns=signal.columns),
            turn.reindex(index=signal.index, columns=signal.columns),
            horizon,
        )
        extras_panels.update(styles)
    elif ret_1d is not None:
        styles = style_panels(
            ret_1d.reindex(index=signal.index, columns=signal.columns),
            pd.DataFrame(np.nan, index=signal.index, columns=signal.columns),
            horizon,
        )
        extras_panels["momentum"] = styles["momentum"]
        extras_panels["volatility"] = styles["volatility"]
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns, dtype=float)
    ind = None if industry is None else industry.reindex(index=signal.index, columns=signal.columns)
    for dt in signal.index:
        extras = {k: v.loc[dt] for k, v in extras_panels.items()}
        out.loc[dt] = neutralize_cross_section(
            signal.loc[dt],
            None if ind is None else ind.loc[dt],
            extras,
            min_obs=min_obs,
        )
    return out
