"""Long-only daily backtester with transaction costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.metrics import performance_summary


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: Dict[str, float]
    buy_hold_metrics: Dict[str, float]


class Backtester:
    """Signal on day t -> trade at next open (already aligned in dataset).

    Long-only:
      flat + UP + tradable -> buy full cash at open_px
      long + DOWN + tradable -> sell all at open_px
    One-way cost: cost_rate.
    """

    def __init__(self, initial_cash: float = 100_000.0, cost_rate: float = 0.001):
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_rate)

    def run(
        self,
        signals: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
    ) -> BacktestResult:
        """
        signals columns required:
          date, next_date, signal (1=UP/0=DOWN), open_px, close_px, tradable_exec
          optional: symbol, proba
        """
        df = signals.copy()
        if symbol is not None and "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        df = df.sort_values("next_date").reset_index(drop=True)
        if df.empty:
            empty = pd.DataFrame()
            return BacktestResult(empty, empty, {}, {})

        cash = self.initial_cash
        shares = 0.0
        position = 0  # 0 flat, 1 long
        records = []
        equity_rows = []

        # Buy & hold benchmark on same execution schedule
        bh_shares = 0.0
        bh_cash = self.initial_cash
        first_buy_done = False

        for _, row in df.iterrows():
            sig = int(row["signal"])
            px = float(row["open_px"])
            tradable = bool(row["tradable_exec"])
            action = "HOLD"
            ret_day = 0.0

            if not first_buy_done and tradable and px > 0:
                # Buy&hold enters on first tradable execution day
                cost = bh_cash * self.cost_rate
                spend = bh_cash - cost
                bh_shares = spend / px
                bh_cash = 0.0
                first_buy_done = True

            if position == 0 and sig == 1 and tradable and px > 0 and cash > 0:
                cost = cash * self.cost_rate
                spend = cash - cost
                shares = spend / px
                cash = 0.0
                position = 1
                action = "BUY"
            elif position == 1 and sig == 0 and tradable and px > 0 and shares > 0:
                proceeds = shares * px
                cost = proceeds * self.cost_rate
                cash = proceeds - cost
                shares = 0.0
                position = 0
                action = "SELL"

            # Mark equity at execution-day close when available
            mtm_px = float(row["next_close_px"]) if "next_close_px" in row and pd.notna(row["next_close_px"]) else px
            signal_close = float(row["close_px"]) if "close_px" in row else mtm_px
            equity = cash + shares * mtm_px
            bh_equity = bh_cash + bh_shares * mtm_px

            records.append(
                {
                    "signal_date": row["date"],
                    "exec_date": row["next_date"],
                    "symbol": row.get("symbol", symbol),
                    "signal": sig,
                    "proba": row.get("proba", np.nan),
                    "action": action,
                    "position": position,
                    "open_px": px,
                    "close_px": signal_close,
                    "mtm_px": mtm_px,
                    "cash": cash,
                    "shares": shares,
                    "equity": equity,
                    "tradable_exec": tradable,
                }
            )
            equity_rows.append(
                {
                    "date": row["next_date"],
                    "equity": equity,
                    "bh_equity": bh_equity,
                    "position": position,
                }
            )

        trades = pd.DataFrame(records)
        equity = pd.DataFrame(equity_rows)
        if not equity.empty:
            equity["date"] = pd.to_datetime(equity["date"])
            equity = equity.drop_duplicates("date", keep="last").set_index("date").sort_index()
            equity["ret"] = equity["equity"].pct_change().fillna(0.0)
            equity["bh_ret"] = equity["bh_equity"].pct_change().fillna(0.0)

        metrics = performance_summary(equity["equity"]) if not equity.empty else {}
        bh_metrics = performance_summary(equity["bh_equity"]) if not equity.empty else {}
        return BacktestResult(trades=trades, equity=equity, metrics=metrics, buy_hold_metrics=bh_metrics)
