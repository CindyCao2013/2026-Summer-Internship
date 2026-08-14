"""Performance metrics for equity curves."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    eq = pd.Series(equity).astype(float)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def annualized_return(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    eq = pd.Series(equity).astype(float).dropna()
    if len(eq) < 2:
        return 0.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = len(eq) - 1
    if n <= 0 or eq.iloc[0] <= 0:
        return 0.0
    return float((1.0 + total) ** (periods_per_year / n) - 1.0)


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def performance_summary(equity: pd.Series, periods_per_year: float = 252.0) -> Dict[str, float]:
    eq = pd.Series(equity).astype(float).dropna()
    if eq.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    rets = eq.pct_change().fillna(0.0)
    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "annualized_return": annualized_return(eq, periods_per_year),
        "sharpe": sharpe_ratio(rets, periods_per_year),
        "max_drawdown": max_drawdown(eq),
    }
