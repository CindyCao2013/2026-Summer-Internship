"""Performance statistics used by IC / decile / enhance reports."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from alphanet.config import ANNUALIZATION_DAYS


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    return pd.Series(np.asarray(x, dtype=float))


def annualized_return(ret, n: int = ANNUALIZATION_DAYS) -> float:
    s = _to_series(ret).dropna()
    if s.empty:
        return float("nan")
    return float(s.mean() * n)


def annualized_vol(ret, n: int = ANNUALIZATION_DAYS) -> float:
    s = _to_series(ret).dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1) * np.sqrt(n))


def sharpe(ret, n: int = ANNUALIZATION_DAYS) -> float:
    vol = annualized_vol(ret, n=n)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return float(annualized_return(ret, n=n) / vol)


def max_drawdown(ret) -> float:
    s = _to_series(ret).dropna()
    if s.empty:
        return float("nan")
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def calmar(ret, n: int = ANNUALIZATION_DAYS) -> float:
    mdd = max_drawdown(ret)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(annualized_return(ret, n=n) / abs(mdd))


def hit_rate(ret) -> float:
    s = _to_series(ret).dropna()
    if s.empty:
        return float("nan")
    return float((s > 0).mean())


def information_ratio(excess, n: int = ANNUALIZATION_DAYS) -> float:
    return sharpe(excess, n=n)


def summarize_return_series(ret, n: int = ANNUALIZATION_DAYS) -> Dict[str, float]:
    s = _to_series(ret)
    return {
        "n_obs": int(s.dropna().shape[0]),
        "annu_ret": annualized_return(s, n=n),
        "annu_vol": annualized_vol(s, n=n),
        "sharpe": sharpe(s, n=n),
        "max_drawdown": max_drawdown(s),
        "calmar": calmar(s, n=n),
        "hit_rate": hit_rate(s),
        "mean": float(s.mean()) if s.notna().any() else float("nan"),
        "std": float(s.std(ddof=1)) if s.dropna().shape[0] > 1 else float("nan"),
    }


def rank_ic_daily(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    aligned_ret = ret.reindex(index=signal.index, columns=signal.columns)
    return signal.corrwith(aligned_ret, axis=1, method="spearman")


def summarize_rank_ic(ic: pd.Series) -> Dict[str, float]:
    s = _to_series(ic).dropna()
    if s.empty:
        return {
            "rank_ic_mean": float("nan"),
            "rank_ic_std": float("nan"),
            "icir": float("nan"),
            "ic_positive_frac": float("nan"),
            "n_cs": 0,
        }
    std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    mean = float(s.mean())
    icir = mean / std if std and np.isfinite(std) and std != 0 else float("nan")
    return {
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "icir": icir,
        "ic_positive_frac": float((s > 0).mean()),
        "n_cs": int(len(s)),
    }


def monotonicity_spearman(group_means: Mapping[int, float]) -> float:
    keys = sorted(int(k) for k in group_means)
    if len(keys) < 3:
        return float("nan")
    ranks = np.arange(1, len(keys) + 1, dtype=float)
    vals = np.array([group_means[k] for k in keys], dtype=float)
    s = pd.Series(vals)
    return float(pd.Series(ranks).corr(s, method="spearman"))
