"""Backtest engine: open execution, costs, regime split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .portfolio import build_strategy_weights, portfolio_turnover
from .signal import formation_returns_in_universe, select_extreme_masks
from .universe import apply_universe_and_tradability


DEFAULT_ONE_WAY_COST = 0.0010  # 10 bps
HOLDING_PERIODS = (1, 5, 10, 20)
# Formation at close t → buy open t+1 → first o2o return is open[t+2]/open[t+1]-1
# When pnl is aligned to ret_o2o index (return ending at that day's open), entry_lag=2.
DEFAULT_ENTRY_LAG_O2O = 2


@dataclass
class BacktestResult:
    hold_days: int
    gross: Dict[str, pd.Series]
    net: Dict[str, pd.Series]
    turnover: Dict[str, pd.Series]
    weights: Dict[str, pd.DataFrame]


def daily_pnl(weights: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Portfolio daily return = sum_i w_{i,t} * r_{i,t}."""
    r = ret.reindex_like(weights)
    return weights.fillna(0.0).mul(r.fillna(0.0)).sum(axis=1)


def apply_cost(
    gross: pd.Series,
    turnover: pd.Series,
    one_way_cost: float = DEFAULT_ONE_WAY_COST,
) -> pd.Series:
    """NetReturn = GrossReturn - Turnover × one_way_cost."""
    return gross - one_way_cost * turnover.reindex(gross.index).fillna(0.0)


def run_holding_backtest(
    ret_formation: pd.DataFrame,
    ret_exec: pd.DataFrame,
    *,
    membership: pd.DataFrame,
    df_not_limit: Optional[pd.DataFrame] = None,
    df_not_st: Optional[pd.DataFrame] = None,
    df_trade_status: Optional[pd.DataFrame] = None,
    close: Optional[pd.DataFrame] = None,
    hold_days: int = 1,
    n_extreme: int = 10,
    entry_lag: int = DEFAULT_ENTRY_LAG_O2O,
    one_way_cost: float = DEFAULT_ONE_WAY_COST,
    apply_tradability: bool = True,
    min_listing_days: int = 60,
) -> BacktestResult:
    """
    Extreme return backtest for one holding period.

    Signal: CSI300 c2c return ranking at close t
    Execution: weights active from day t+entry_lag using ret_exec
              (prefer open-to-open for no look-ahead)

    Tradability applied to formation-day candidates before selection.
    """
    # Formation universe
    form = formation_returns_in_universe(ret_formation, membership)
    if apply_tradability:
        form = apply_universe_and_tradability(
            form,
            membership=membership,
            df_not_limit=df_not_limit,
            df_not_st=df_not_st,
            df_trade_status=df_trade_status,
            close=close,
            min_listing_days=min_listing_days,
        )

    loser_mask, winner_mask = select_extreme_masks(form, n=n_extreme)
    weights = build_strategy_weights(
        loser_mask,
        winner_mask,
        hold_days=hold_days,
        entry_lag=entry_lag,
    )

    # Execution returns aligned to weight index
    r = ret_exec.reindex_like(weights["bottom10"])

    gross: Dict[str, pd.Series] = {}
    net: Dict[str, pd.Series] = {}
    turnover: Dict[str, pd.Series] = {}
    for name, w in weights.items():
        g = daily_pnl(w, r)
        to = portfolio_turnover(w)
        # For long-short, turnover already uses 0.5*L1; cost on that one-way measure
        n = apply_cost(g, to, one_way_cost=one_way_cost)
        gross[name] = g
        net[name] = n
        turnover[name] = to

    return BacktestResult(
        hold_days=hold_days,
        gross=gross,
        net=net,
        turnover=turnover,
        weights=weights,
    )


def run_all_horizons(
    ret_formation: pd.DataFrame,
    ret_exec: pd.DataFrame,
    *,
    membership: pd.DataFrame,
    holding_periods: Sequence[int] = HOLDING_PERIODS,
    **kwargs,
) -> Dict[int, BacktestResult]:
    """Run backtest across holding periods."""
    out: Dict[int, BacktestResult] = {}
    for h in holding_periods:
        out[h] = run_holding_backtest(
            ret_formation,
            ret_exec,
            membership=membership,
            hold_days=h,
            **kwargs,
        )
    return out


def forward_returns(ret: pd.DataFrame, horizons: Sequence[int] = HOLDING_PERIODS) -> Dict[int, pd.DataFrame]:
    """
    Forward cumulative returns over H days starting tomorrow.

    fwd_H[t] = (1+r_{t+1})*...*(1+r_{t+H}) - 1
    """
    out: Dict[int, pd.DataFrame] = {}
    # At date t, next-day gross return factor
    next_log = np.log1p(ret).shift(-1)
    for h in horizons:
        # Forward sum of log returns over [t+1, t+H] via reverse rolling
        fwd_log = next_log.iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]
        out[h] = np.expm1(fwd_log)
    return out
