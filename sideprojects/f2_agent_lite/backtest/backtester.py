"""Long-short daily backtester with transaction costs and tradability filters."""

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
    """Target-position backtester.

    ``signal`` in {-1, 0, +1}: short / flat / long (full notional).
    Execution at ``open_px`` on ``next_date``; MTM at ``next_close_px``.
    Overnight gap (prev close -> open) is applied when a position is held.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        cost_rate: float = 0.001,
        allow_short: bool = True,
    ):
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_rate)
        self.allow_short = bool(allow_short)

    def run(
        self,
        signals: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
    ) -> BacktestResult:
        df = signals.copy()
        if symbol is not None and "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()
        df = df.sort_values("next_date").reset_index(drop=True)
        if df.empty:
            empty = pd.DataFrame()
            return BacktestResult(empty, empty, {}, {})

        equity = self.initial_cash
        position = 0  # -1 / 0 / +1
        prev_close = None
        records = []
        equity_rows = []

        bh_equity = self.initial_cash
        bh_position = 0
        bh_prev_close = None

        for _, row in df.iterrows():
            raw_sig = int(row["signal"])
            if not self.allow_short:
                desired = 1 if raw_sig > 0 else 0
            else:
                desired = int(np.clip(raw_sig, -1, 1))

            px_open = float(row["open_px"])
            px_close = float(row["next_close_px"]) if "next_close_px" in row and pd.notna(row["next_close_px"]) else px_open
            tradable = bool(row["tradable_exec"])
            action = "HOLD"

            # Overnight gap for strategy
            if position != 0 and prev_close is not None and prev_close > 0 and px_open > 0:
                equity *= 1.0 + position * (px_open / prev_close - 1.0)

            # Overnight gap for buy&hold
            if bh_position != 0 and bh_prev_close is not None and bh_prev_close > 0 and px_open > 0:
                bh_equity *= 1.0 + bh_position * (px_open / bh_prev_close - 1.0)

            # Buy&hold enters once when first tradable
            if bh_position == 0 and tradable and px_open > 0:
                bh_equity *= 1.0 - self.cost_rate
                bh_position = 1
                action_bh = "BH_BUY"
            else:
                action_bh = "BH_HOLD"

            # Rebalance strategy at open
            target = desired if tradable else position
            if target != position and px_open > 0:
                turnover = abs(target - position)
                equity *= 1.0 - self.cost_rate * turnover
                if position == 0 and target == 1:
                    action = "BUY"
                elif position == 0 and target == -1:
                    action = "SHORT"
                elif position == 1 and target == 0:
                    action = "SELL"
                elif position == -1 and target == 0:
                    action = "COVER"
                elif position == 1 and target == -1:
                    action = "FLIP_SHORT"
                elif position == -1 and target == 1:
                    action = "FLIP_LONG"
                else:
                    action = "REBALANCE"
                position = target

            # Open -> close MTM
            if position != 0 and px_open > 0 and px_close > 0:
                equity *= 1.0 + position * (px_close / px_open - 1.0)
            if bh_position != 0 and px_open > 0 and px_close > 0:
                bh_equity *= 1.0 + bh_position * (px_close / px_open - 1.0)

            prev_close = px_close
            bh_prev_close = px_close

            records.append(
                {
                    "signal_date": row["date"],
                    "exec_date": row["next_date"],
                    "symbol": row.get("symbol", symbol),
                    "signal": raw_sig,
                    "proba": row.get("proba", np.nan),
                    "action": action,
                    "bh_action": action_bh,
                    "position": position,
                    "open_px": px_open,
                    "mtm_px": px_close,
                    "equity": equity,
                    "bh_equity": bh_equity,
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
        equity_df = pd.DataFrame(equity_rows)
        if not equity_df.empty:
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.drop_duplicates("date", keep="last").set_index("date").sort_index()
            equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)
            equity_df["bh_ret"] = equity_df["bh_equity"].pct_change().fillna(0.0)

        metrics = performance_summary(equity_df["equity"]) if not equity_df.empty else {}
        bh_metrics = performance_summary(equity_df["bh_equity"]) if not equity_df.empty else {}
        return BacktestResult(
            trades=trades,
            equity=equity_df,
            metrics=metrics,
            buy_hold_metrics=bh_metrics,
        )


# Backward-compatible alias
LongShortBacktester = Backtester
