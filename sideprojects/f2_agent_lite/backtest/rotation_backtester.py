"""Cross-sectional long-short rotation backtester (Scheme A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.metrics import performance_summary


@dataclass
class RotationResult:
    equity: pd.DataFrame
    holdings: pd.DataFrame
    metrics: Dict[str, float]
    equal_weight_bh_metrics: Dict[str, float]
    selection_stats: Dict[str, object]


class RotationBacktester:
    """Daily rank by score: long Top-K, short Bottom-K, equal-weight within side.

    Expected columns in ``signals``:
      date, next_date, symbol, score, open_px, next_close_px, tradable_exec
    Ranking uses ``date`` (signal day); trades execute on ``next_date`` open.
    ``rebalance_every`` keeps weights for N execution days (weekly when =5).
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        cost_rate: float = 0.001,
        top_k: Optional[int] = None,
        bottom_k: Optional[int] = None,
        top_frac: float = 0.2,
        bottom_frac: float = 0.2,
        long_gross: float = 0.5,
        short_gross: float = 0.5,
        rebalance_every: int = 1,
        use_vol_scaling: bool = False,
        vol_scaling_window: int = 20,
        vol_scaling_floor: float = 0.01,
    ):
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_rate)
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.top_frac = float(top_frac)
        self.bottom_frac = float(bottom_frac)
        self.long_gross = float(long_gross)
        self.short_gross = float(short_gross)
        self.rebalance_every = max(1, int(rebalance_every))
        self.use_vol_scaling = bool(use_vol_scaling)
        self.vol_scaling_window = max(5, int(vol_scaling_window))
        self.vol_scaling_floor = float(vol_scaling_floor)

    def _k(self, n: int) -> Tuple[int, int]:
        top_k = self.top_k if self.top_k is not None else max(1, int(n * self.top_frac))
        bottom_k = self.bottom_k if self.bottom_k is not None else max(1, int(n * self.bottom_frac))
        top_k = min(top_k, n)
        bottom_k = min(bottom_k, n)
        # Avoid overlapping when pool is tiny
        if top_k + bottom_k > n:
            bottom_k = max(0, n - top_k)
        return top_k, bottom_k

    def run(self, signals: pd.DataFrame) -> RotationResult:
        df = signals.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["next_date"] = pd.to_datetime(df["next_date"])
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
        if df.empty:
            empty = pd.DataFrame()
            return RotationResult(empty, empty, {}, {}, {})

        symbols = sorted(df["symbol"].unique().tolist())
        top_k, bottom_k = self._k(len(symbols))
        print(
            "[rotation] pool={} top_k={} bottom_k={} long_gross={} short_gross={} rebalance_every={} vol_scaling={}".format(
                len(symbols),
                top_k,
                bottom_k,
                self.long_gross,
                self.short_gross,
                self.rebalance_every,
                self.use_vol_scaling,
            )
        )

        # Realized vol from signal-day close (causal: use history up to signal day)
        vol_by_date_sym: Dict[Tuple[pd.Timestamp, str], float] = {}
        if self.use_vol_scaling:
            close_sig = (
                df.pivot_table("close_px", index="date", columns="symbol", aggfunc="last")
                .sort_index()
            )
            close_sig.index = pd.to_datetime(close_sig.index).normalize()
            rets = close_sig.pct_change()
            roll_vol = rets.rolling(self.vol_scaling_window, min_periods=max(5, self.vol_scaling_window // 2)).std()
            for d in roll_vol.index:
                for s in roll_vol.columns:
                    v = roll_vol.at[d, s]
                    if pd.notna(v):
                        vol_by_date_sym[(pd.Timestamp(d).normalize(), s)] = float(v)

        # Target weights keyed by execution date; only refresh every rebalance_every signals
        target_by_exec: Dict[pd.Timestamp, Dict[str, float]] = {}
        pick_rows: List[dict] = []
        sig_days = list(df.groupby("date", sort=True).groups.keys())
        for i, sig_day in enumerate(sig_days):
            if i % self.rebalance_every != 0:
                continue
            grp = df[df["date"] == sig_day]
            eligible = grp[grp["tradable_exec"].astype(bool)].copy()
            if eligible.empty:
                continue
            top_k, bottom_k = self._k(len(eligible))
            eligible = eligible.sort_values("score", ascending=False)
            longs = eligible.head(top_k)["symbol"].tolist()
            shorts = eligible.tail(bottom_k)["symbol"].tolist() if bottom_k > 0 else []
            # Disjoint
            shorts = [s for s in shorts if s not in longs]
            weights = {s: 0.0 for s in symbols}
            sig_ts = pd.Timestamp(sig_day).normalize()

            def _side_weights(names: List[str], gross: float, long_side: bool) -> Dict[str, float]:
                if not names or gross <= 0:
                    return {}
                if not self.use_vol_scaling:
                    w = gross / len(names)
                    return {s: (w if long_side else -w) for s in names}
                raw = []
                for s in names:
                    sc = float(eligible.loc[eligible["symbol"] == s, "score"].iloc[0])
                    conf = abs(sc)
                    vol = vol_by_date_sym.get((sig_ts, s), np.nan)
                    if not np.isfinite(vol) or vol < self.vol_scaling_floor:
                        vol = self.vol_scaling_floor
                    raw.append(max(conf, 1e-6) / vol)
                total = float(sum(raw))
                if total <= 0:
                    w = gross / len(names)
                    return {s: (w if long_side else -w) for s in names}
                out_w = {}
                for s, r in zip(names, raw):
                    w = gross * (r / total)
                    out_w[s] = w if long_side else -w
                return out_w

            weights.update(_side_weights(longs, self.long_gross, True))
            weights.update(_side_weights(shorts, self.short_gross, False))
            exec_day = pd.Timestamp(eligible["next_date"].iloc[0]).normalize()
            target_by_exec[exec_day] = weights
            for s in longs:
                pick_rows.append(
                    {
                        "signal_date": sig_day,
                        "exec_date": exec_day,
                        "symbol": s,
                        "side": "LONG",
                        "score": float(eligible.loc[eligible["symbol"] == s, "score"].iloc[0]),
                        "weight": float(weights.get(s, 0.0)),
                    }
                )
            for s in shorts:
                pick_rows.append(
                    {
                        "signal_date": sig_day,
                        "exec_date": exec_day,
                        "symbol": s,
                        "side": "SHORT",
                        "score": float(eligible.loc[eligible["symbol"] == s, "score"].iloc[0]),
                        "weight": float(weights.get(s, 0.0)),
                    }
                )

        # Price panels on execution calendar
        open_px = df.pivot_table("open_px", index="next_date", columns="symbol", aggfunc="last").sort_index()
        close_px = df.pivot_table("next_close_px", index="next_date", columns="symbol", aggfunc="last").sort_index()
        open_px.index = pd.to_datetime(open_px.index).normalize()
        close_px.index = pd.to_datetime(close_px.index).normalize()
        dates = open_px.index.intersection(close_px.index).sort_values()

        equity = self.initial_cash
        weights = {s: 0.0 for s in symbols}
        prev_close = {s: np.nan for s in symbols}
        equity_rows = []
        hold_rows = []

        # Equal-weight buy&hold benchmark (long-only all names when available)
        bh_equity = self.initial_cash
        bh_w = {s: 0.0 for s in symbols}
        bh_prev = {s: np.nan for s in symbols}
        bh_entered = False

        for d in dates:
            # Overnight gap for strategy
            overnight = 0.0
            for s in symbols:
                o = open_px.at[d, s] if s in open_px.columns else np.nan
                pc = prev_close[s]
                w = weights[s]
                if w != 0 and pd.notna(o) and pd.notna(pc) and pc > 0:
                    overnight += w * (float(o) / float(pc) - 1.0)
            equity *= 1.0 + overnight

            # Overnight for BH
            bh_on = 0.0
            for s in symbols:
                o = open_px.at[d, s] if s in open_px.columns else np.nan
                pc = bh_prev[s]
                w = bh_w[s]
                if w != 0 and pd.notna(o) and pd.notna(pc) and pc > 0:
                    bh_on += w * (float(o) / float(pc) - 1.0)
            bh_equity *= 1.0 + bh_on

            # Enter BH once: equal weight across names with valid open
            if not bh_entered:
                avail = [s for s in symbols if s in open_px.columns and pd.notna(open_px.at[d, s])]
                if avail:
                    for s in symbols:
                        bh_w[s] = (1.0 / len(avail)) if s in avail else 0.0
                    bh_equity *= 1.0 - self.cost_rate * 1.0  # full deploy once
                    bh_entered = True

            # Rebalance strategy to target for this exec day
            target = target_by_exec.get(pd.Timestamp(d).normalize(), weights)
            # Only trade names with valid open
            target_eff = {}
            for s in symbols:
                o = open_px.at[d, s] if s in open_px.columns else np.nan
                if pd.isna(o):
                    target_eff[s] = weights[s]  # cannot trade; keep
                else:
                    target_eff[s] = float(target.get(s, 0.0))
            turnover = 0.5 * sum(abs(target_eff[s] - weights[s]) for s in symbols)
            equity *= 1.0 - self.cost_rate * turnover
            weights = target_eff

            # Open -> close
            day_ret = 0.0
            for s in symbols:
                o = open_px.at[d, s] if s in open_px.columns else np.nan
                c = close_px.at[d, s] if s in close_px.columns else np.nan
                w = weights[s]
                if w != 0 and pd.notna(o) and pd.notna(c) and o > 0:
                    day_ret += w * (float(c) / float(o) - 1.0)
                if pd.notna(c):
                    prev_close[s] = float(c)
            equity *= 1.0 + day_ret

            bh_day = 0.0
            for s in symbols:
                o = open_px.at[d, s] if s in open_px.columns else np.nan
                c = close_px.at[d, s] if s in close_px.columns else np.nan
                w = bh_w[s]
                if w != 0 and pd.notna(o) and pd.notna(c) and o > 0:
                    bh_day += w * (float(c) / float(o) - 1.0)
                if pd.notna(c):
                    bh_prev[s] = float(c)
            bh_equity *= 1.0 + bh_day

            equity_rows.append({"date": d, "equity": equity, "bh_equity": bh_equity, "turnover": turnover})
            for s in symbols:
                hold_rows.append({"date": d, "symbol": s, "weight": weights[s]})

        equity_df = pd.DataFrame(equity_rows)
        if not equity_df.empty:
            equity_df["date"] = pd.to_datetime(equity_df["date"])
            equity_df = equity_df.set_index("date").sort_index()
            equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)
            equity_df["bh_ret"] = equity_df["bh_equity"].pct_change().fillna(0.0)

        holdings = pd.DataFrame(hold_rows)
        picks = pd.DataFrame(pick_rows)
        selection_stats: Dict[str, object] = {
            "top_k": top_k,
            "bottom_k": bottom_k,
            "n_symbols": len(symbols),
        }
        if not picks.empty:
            long_freq = (
                picks[picks["side"] == "LONG"].groupby("symbol").size().sort_values(ascending=False)
            )
            short_freq = (
                picks[picks["side"] == "SHORT"].groupby("symbol").size().sort_values(ascending=False)
            )
            selection_stats["long_pick_counts"] = long_freq.to_dict()
            selection_stats["short_pick_counts"] = short_freq.to_dict()

        metrics = performance_summary(equity_df["equity"]) if not equity_df.empty else {}
        bh_metrics = performance_summary(equity_df["bh_equity"]) if not equity_df.empty else {}
        return RotationResult(
            equity=equity_df,
            holdings=holdings,
            metrics=metrics,
            equal_weight_bh_metrics=bh_metrics,
            selection_stats=selection_stats,
        )
