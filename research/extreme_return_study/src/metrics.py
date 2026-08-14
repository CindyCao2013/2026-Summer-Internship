"""Performance metrics, RankIC, regime analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import Factor_Dev_Lib
from alpha_investability import classify_market_regimes

from .backtest import HOLDING_PERIODS, BacktestResult

TRADING_DAYS_YEAR = 250


def performance_stats(pnl: pd.Series, *, name: str = "") -> dict:
    """Annual return, vol, Sharpe, MDD, win rate."""
    s = pnl.dropna()
    if len(s) < 20:
        return {
            "name": name,
            "n_days": int(len(s)),
            "annu_ret": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "mean_daily": np.nan,
        }
    annu = float(Factor_Dev_Lib.calAnnuRet(s))
    sharpe = float(Factor_Dev_Lib.calSharpe(s))
    mdd, _ = Factor_Dev_Lib.calMDD(s)
    vol = float(s.std() * np.sqrt(TRADING_DAYS_YEAR))
    win = float((s > 0).mean())
    return {
        "name": name,
        "n_days": int(len(s)),
        "annu_ret": annu,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(mdd),
        "win_rate": win,
        "mean_daily": float(s.mean()),
    }


def summarize_backtest(
    result: BacktestResult,
    *,
    cost_label: str = "net",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Performance table for bottom10 / top10 / long_short."""
    book = result.net if cost_label == "net" else result.gross
    rows = []
    for name, series in book.items():
        s = series
        if start is not None:
            s = s.loc[start:]
        if end is not None:
            s = s.loc[:end]
        stats = performance_stats(s, name=name)
        to = result.turnover[name]
        if start is not None:
            to = to.loc[start:]
        if end is not None:
            to = to.loc[:end]
        stats["hold_days"] = result.hold_days
        stats["avg_turnover"] = float(to.mean()) if to.notna().any() else np.nan
        stats["annu_turnover"] = (
            float(to.mean() * TRADING_DAYS_YEAR) if to.notna().any() else np.nan
        )
        stats["cost_type"] = cost_label
        rows.append(stats)
    return pd.DataFrame(rows)


def holding_period_comparison(
    results: Dict[int, BacktestResult],
    *,
    cost_label: str = "net",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Wide table: strategy × horizon metrics."""
    frames = [
        summarize_backtest(res, cost_label=cost_label, start=start, end=end)
        for res in results.values()
    ]
    return pd.concat(frames, ignore_index=True)


def cumulative_return(pnl: pd.Series) -> pd.Series:
    return (1.0 + pnl.fillna(0.0)).cumprod() - 1.0


def monthly_returns(pnl: pd.Series) -> pd.Series:
    """Compound daily returns into calendar-month returns."""
    s = pnl.dropna()
    if s.empty:
        return s
    # "M" for older pandas; "ME" on pandas>=2.2
    try:
        return (1.0 + s).resample("ME").prod() - 1.0
    except ValueError:
        return (1.0 + s).resample("M").prod() - 1.0


def rolling_sharpe(pnl: pd.Series, window: int = 60) -> pd.Series:
    s = pnl.dropna()
    mu = s.rolling(window, min_periods=max(20, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(20, window // 2)).std()
    return (mu / sd.replace(0, np.nan)) * np.sqrt(TRADING_DAYS_YEAR)


def daily_rank_ic(
    signal: pd.DataFrame,
    forward: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional Spearman RankIC each day."""
    sig = signal.reindex_like(forward)
    return sig.corrwith(forward, axis=1, method="spearman")


def ic_summary(ic: pd.Series) -> dict:
    s = ic.dropna()
    if len(s) < 20:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "win_rate": np.nan,
            "n_days": int(len(s)),
        }
    mean_ic = float(s.mean())
    std_ic = float(s.std())
    icir = mean_ic / std_ic * np.sqrt(TRADING_DAYS_YEAR) if std_ic > 0 else np.nan
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": float(icir) if pd.notna(icir) else np.nan,
        "win_rate": float((s > 0).mean()),
        "n_days": int(len(s)),
    }


def compute_ic_table(
    signal: pd.DataFrame,
    fwd_map: Dict[int, pd.DataFrame],
    *,
    membership: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """RankIC / ICIR for each forward horizon."""
    rows = []
    sig = signal
    if membership is not None:
        m = membership.reindex_like(sig)
        sig = sig.where(m == 1)
    for h, fwd in fwd_map.items():
        f = fwd
        if membership is not None:
            f = f.where(m.reindex_like(f) == 1)
        ic = daily_rank_ic(sig, f)
        stats = ic_summary(ic)
        stats["horizon"] = h
        rows.append(stats)
    return pd.DataFrame(rows)


def vol_regime(
    index_ret: pd.Series,
    *,
    window: int = 20,
) -> pd.Series:
    """High / low volatility regime via median split of rolling vol."""
    vol = index_ret.rolling(window, min_periods=max(10, window // 2)).std()
    med = vol.median()
    out = pd.Series("low_vol", index=index_ret.index)
    out = out.mask(vol >= med, "high_vol")
    out = out.mask(vol.isna(), np.nan)
    return out


def sign_regime(index_ret: pd.Series) -> pd.Series:
    """Bull / bear by same-day index return sign (simple split)."""
    out = pd.Series("flat", index=index_ret.index)
    out = out.mask(index_ret > 0, "up_day")
    out = out.mask(index_ret < 0, "down_day")
    out = out.mask(index_ret.isna(), np.nan)
    return out


def regime_performance(
    pnl: pd.Series,
    regime: pd.Series,
    *,
    min_days: int = 40,
) -> pd.DataFrame:
    """Performance stats within each regime label."""
    rows = []
    r = regime.reindex(pnl.index)
    for label, sub_idx in r.groupby(r).groups.items():
        if pd.isna(label):
            continue
        sub = pnl.loc[sub_idx]
        stats = performance_stats(sub, name=str(label))
        if stats["n_days"] < min_days:
            stats["sharpe"] = np.nan
            stats["annu_ret"] = np.nan
        rows.append(stats)
    return pd.DataFrame(rows)


def full_regime_pack(
    pnl: pd.Series,
    index_ret: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """Bull/bear (rolling) + up/down day + high/low vol regime tables."""
    trend = classify_market_regimes(index_ret)
    return {
        "trend_regime": regime_performance(pnl, trend.reindex(pnl.index)),
        "day_sign_regime": regime_performance(pnl, sign_regime(index_ret).reindex(pnl.index)),
        "vol_regime": regime_performance(pnl, vol_regime(index_ret).reindex(pnl.index)),
    }
