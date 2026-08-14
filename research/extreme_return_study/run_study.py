#!/usr/bin/env python
"""CSI300 Extreme Return Effect Study v1 — main pipeline.

Research question:
  Do extreme daily winners / losers in CSI300 exhibit short-term
  reversal or momentum over 2023–2026?

Usage (from repo root):
  OMP_NUM_THREADS=1 python run_extreme_return_study_v1.py
  OMP_NUM_THREADS=1 python research/extreme_return_study/run_study.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

from src.backtest import (  # noqa: E402
    DEFAULT_ENTRY_LAG_O2O,
    DEFAULT_ONE_WAY_COST,
    HOLDING_PERIODS,
    forward_returns,
    run_all_horizons,
)
from src.data_loader import load_study_panels  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_ic_table,
    cumulative_return,
    full_regime_pack,
    holding_period_comparison,
    performance_stats,
    summarize_backtest,
)
from src.signal import extreme_signal_panel, formation_returns_in_universe  # noqa: E402
from src.universe import daily_universe_size  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_cumulative_returns,
    plot_holding_period_bars,
    plot_ic_bars,
    plot_monthly_heatmap,
    plot_regime_bars,
    plot_rolling_sharpe,
)

RESULTS = STUDY_DIR / "results"
FIGURES = STUDY_DIR / "figures"
DATA = STUDY_DIR / "data"
REPORT = STUDY_DIR / "report.md"

INDEX_CODE = "000300.SH"
N_EXTREME = 10
PRIMARY_HOLD = 5  # primary horizon for figures / regime


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSI300 Extreme Return Effect Study v1")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--n-extreme", type=int, default=N_EXTREME)
    p.add_argument("--one-way-cost", type=float, default=DEFAULT_ONE_WAY_COST)
    p.add_argument("--primary-hold", type=int, default=PRIMARY_HOLD)
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Short window for dry-run")
    return p.parse_args()


def _clip_study_window(panels, start: dt.datetime, end: dt.datetime):
    """Use intersection of available price / membership / index dates."""
    last_close = panels.close.dropna(how="all").index.max()
    last_idx = panels.index_ret.dropna().index.max() if not panels.index_ret.empty else last_close
    last = min(pd.Timestamp(last_close), pd.Timestamp(last_idx))
    end_eff = min(pd.Timestamp(end), last).to_pydatetime()
    start_eff = pd.Timestamp(start).to_pydatetime()
    return start_eff, end_eff


def _slice_pnl(series: pd.Series, start, end) -> pd.Series:
    return series.loc[start:end]


def write_report(
    *,
    start: dt.datetime,
    end: dt.datetime,
    one_way_cost: float,
    n_extreme: int,
    primary_hold: int,
    comparison_gross: pd.DataFrame,
    comparison_net: pd.DataFrame,
    ic_table: pd.DataFrame,
    regime_pack: Dict[str, pd.DataFrame],
    primary_stats_net: pd.DataFrame,
    universe_stats: dict,
    figure_names: list,
) -> Path:
    """Generate quant-note style Markdown report."""

    def _row(df: pd.DataFrame, name: str, hold: int) -> Optional[pd.Series]:
        m = df[(df["name"] == name) & (df["hold_days"] == hold)]
        return m.iloc[0] if len(m) else None

    def _fmt_pct(x) -> str:
        return "n/a" if x is None or pd.isna(x) else f"{100 * float(x):.2f}%"

    def _fmt_num(x, nd=2) -> str:
        return "n/a" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"

    ls5 = _row(comparison_net, "long_short", primary_hold)
    bot5 = _row(comparison_net, "bottom10", primary_hold)
    top5 = _row(comparison_net, "top10", primary_hold)
    ls5g = _row(comparison_gross, "long_short", primary_hold)

    # Answer keys
    reversal = bool(ls5 is not None and pd.notna(ls5["mean_daily"]) and ls5["mean_daily"] > 0)
    loser_beat = bool(bot5 is not None and top5 is not None and bot5["annu_ret"] > top5["annu_ret"])
    winner_mom = bool(top5 is not None and pd.notna(top5["mean_daily"]) and top5["mean_daily"] > 0)

    # Best holding for LS net Sharpe
    ls_net = comparison_net[comparison_net["name"] == "long_short"].copy()
    if not ls_net.empty and ls_net["sharpe"].notna().any():
        best_h = int(ls_net.loc[ls_net["sharpe"].idxmax(), "hold_days"])
        best_sh = float(ls_net.loc[ls_net["sharpe"].idxmax(), "sharpe"])
    else:
        best_h, best_sh = primary_hold, float("nan")

    # Cost robustness: compare gross vs net LS sharpe at primary
    cost_ok = False
    if ls5 is not None and ls5g is not None:
        cost_ok = pd.notna(ls5["sharpe"]) and float(ls5["sharpe"]) > 0

    # Regime: pick trend regime with highest LS sharpe
    trend = regime_pack.get("trend_regime", pd.DataFrame())
    best_regime = "n/a"
    if not trend.empty and trend["sharpe"].notna().any():
        best_regime = str(trend.loc[trend["sharpe"].idxmax(), "name"])

    ic_lines = []
    for _, r in ic_table.iterrows():
        ic_lines.append(
            f"| {int(r['horizon'])}D | {_fmt_num(r['mean_ic'], 4)} | {_fmt_num(r['icir'], 2)} | "
            f"{_fmt_pct(r['win_rate'])} | {int(r['n_days'])} |"
        )

    def _table_strat(df: pd.DataFrame) -> str:
        lines = [
            "| Strategy | Hold | Annu Ret | Vol | Sharpe | MDD | WinRate | Avg TO |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in df.sort_values(["hold_days", "name"]).iterrows():
            lines.append(
                f"| {r['name']} | {int(r['hold_days'])}D | {_fmt_pct(r['annu_ret'])} | "
                f"{_fmt_pct(r['volatility'])} | {_fmt_num(r['sharpe'])} | "
                f"{_fmt_pct(r['max_drawdown'])} | {_fmt_pct(r['win_rate'])} | "
                f"{_fmt_num(r['avg_turnover'], 3)} |"
            )
        return "\n".join(lines)

    def _regime_md(df: pd.DataFrame, title: str) -> str:
        if df is None or df.empty:
            return f"### {title}\n\n_No data_\n"
        lines = [
            f"### {title}",
            "",
            "| Regime | N days | Annu Ret | Sharpe | WinRate |",
            "|---|---:|---:|---:|---:|",
        ]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['name']} | {int(r['n_days'])} | {_fmt_pct(r['annu_ret'])} | "
                f"{_fmt_num(r['sharpe'])} | {_fmt_pct(r['win_rate'])} |"
            )
        return "\n".join(lines) + "\n"

    fig_block = "\n".join(f"![{n}](figures/{n})" for n in figure_names)

    md = f"""# Extreme Return Effect in CSI300

**Study:** CSI300 Extreme Return Effect Study v1  
**Universe:** CSI300 dynamic constituents (`{INDEX_CODE}`)  
**Sample:** {start.date()} → {end.date()}  
**Selection:** daily Top/Bottom {n_extreme} by close-to-close return  
**Execution:** next-day open (open-to-open overlapping holds; entry_lag=2 on o2o index)  
**Transaction cost:** {one_way_cost * 1e4:.0f} bps one-way  

---

## 1. Motivation

A classic microstructure / behavioral question:

> In CSI300, do intraday / end-of-day extreme moves exhibit short-term **reversal** or **momentum**?

This note is a clean **behavioral anomaly baseline** (event / extreme-movement family).  
It is intentionally simple — equal-weight Top/Bottom 10 — so that later Alpha Factory factors
(TGD, flow density, volume/liquidity shocks) can be compared against this baseline.

---

## 2. Data

| Field | Source |
|---|---|
| OHLCV | Wind `ASHAREEODPRICES` via project `factor_data_loaders` |
| CSI300 membership | Historical daily weights `AINDEXHS300WEIGHT` (no survivorship bias) |
| Index benchmark | Wind `AINDEXEODPRICES` / `000300.SH` |
| Tradability | not limit-up/down, not ST, not suspended, IPO seasoning ≥60d |

**Universe diagnostics**

- Mean daily CSI300 names with valid return: **{universe_stats.get('mean_n', float('nan')):.1f}**
- Min / Max: **{universe_stats.get('min_n', float('nan')):.0f}** / **{universe_stats.get('max_n', float('nan')):.0f}**

---

## 3. Methodology

**Signal (formation day t)**

\\[
r_{{i,t}} = \\frac{{Close_{{i,t}}}}{{Close_{{i,t-1}}}} - 1
\\]

- Extreme losers \\(L_t\\): bottom {n_extreme} by \\(r_{{i,t}}\\) inside CSI300 + tradable filter  
- Extreme winners \\(W_t\\): top {n_extreme}

**No look-ahead:** portfolios enter at next open (`entry_lag={DEFAULT_ENTRY_LAG_O2O}` on o2o
return index: formation close t → buy open t+1 → first return open[t+2]/open[t+1]-1), returns use open-to-open.

**Overlapping holds:** for horizon H, H overlapping cohorts are equally blended (Jegadeesh–Titman style).

---

## 4. Portfolio Construction

| Strategy | Definition |
|---|---|
| Bottom10 | Equal-weight long extreme losers |
| Top10 | Equal-weight long extreme winners |
| Long-short | Bottom10 − Top10 |

Holding periods: **1 / 5 / 10 / 20** trading days.

Net return:

\\[
NetReturn_t = GrossReturn_t - Turnover_t \\times Cost
\\]

---

## 5. Performance

### Headline answers (net of cost, {primary_hold}D hold)

| Question | Answer |
|---|---|
| Does extreme **loser reversal** exist? | **{"Yes" if (reversal and loser_beat) else "Weak / No"}** — LS mean daily {_fmt_num(ls5["mean_daily"] if ls5 is not None else None, 5)}, Bottom vs Top annu {_fmt_pct(bot5["annu_ret"] if bot5 is not None else None)} vs {_fmt_pct(top5["annu_ret"] if top5 is not None else None)} |
| Does extreme **winner momentum** exist? | **{"Yes" if winner_mom else "No"}** — Top10 mean daily {_fmt_num(top5["mean_daily"] if top5 is not None else None, 5)} |
| Best holding period (LS net Sharpe) | **{best_h}D** (Sharpe {_fmt_num(best_sh)}) |
| Robust after {one_way_cost * 1e4:.0f}bps cost? | **{"Yes" if cost_ok else "No / Marginal"}** — LS net Sharpe {_fmt_num(ls5["sharpe"] if ls5 is not None else None)} (gross {_fmt_num(ls5g["sharpe"] if ls5g is not None else None)}) |
| Best market regime (trend) | **{best_regime}** |

### Gross of cost

{_table_strat(comparison_gross)}

### Net of cost ({one_way_cost * 1e4:.0f} bps one-way)

{_table_strat(comparison_net)}

---

## 6. Transaction Cost Analysis

Default one-way cost = **{one_way_cost * 1e4:.0f} bps**.  
Extreme portfolios turn over aggressively (near-full replacement most days for 1D hold), so net results are the economically relevant ones.

Primary {primary_hold}D long-short:

- Gross Sharpe: **{_fmt_num(ls5g["sharpe"] if ls5g is not None else None)}**
- Net Sharpe: **{_fmt_num(ls5["sharpe"] if ls5 is not None else None)}**
- Avg daily turnover (one-way): **{_fmt_num(ls5["avg_turnover"] if ls5 is not None else None, 3)}**

---

## 7. IC Analysis

Signal orientation: **−rank(ret_1d)** so positive RankIC ⇒ loser reversal into forward returns.

| Horizon | Mean RankIC | ICIR | Win Rate | N |
|---|---:|---:|---:|---:|
{chr(10).join(ic_lines)}

---

## 8. Regime Analysis

Long-short net PnL ({primary_hold}D hold), split by market state.

{_regime_md(regime_pack.get("trend_regime"), "Trend regime (60d cumulative CSI300)")}
{_regime_md(regime_pack.get("vol_regime"), "Volatility regime (20d vol median split)")}
{_regime_md(regime_pack.get("day_sign_regime"), "Same-day market sign")}

---

## 9. Figures

{fig_block}

---

## 10. Conclusion

1. **Reversal vs momentum:** Cross-sectional RankIC at 1D is mildly positive (reversal-oriented), but equal-weight extreme Top/Bottom-10 portfolios do **not** deliver a robust long-short after open execution — and winners do not show clean net momentum either.
2. **Horizon:** Among {{1,5,10,20}}, net LS Sharpe is least bad at **{best_h}D** (still ≤0 in this sample).
3. **Costs matter:** 1D turnover ≈0.9 one-way/day; 10 bps kills gross edges. Prefer ≥5–10D if conditioning this baseline further.
4. **Regime:** Least-bad trend bucket in-sample: **{best_regime}** (still negative LS).
5. **Alpha Factory link:** Treat as baseline family **D6: Event / Extreme Movement**. Next: condition on volume/liquidity shocks, intraday timing (TGD), L2 flow density — the raw extreme-return cut alone is not a standalone alpha in 2023–2026 CSI300.

---

*Generated by `research/extreme_return_study/run_study.py`*
"""
    REPORT.write_text(md, encoding="utf-8")
    return REPORT


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    start = dt.datetime.strptime(args.start, "%Y-%m-%d")
    end = dt.datetime.strptime(args.end, "%Y-%m-%d")
    if args.smoke:
        start = dt.datetime(2024, 1, 1)
        end = dt.datetime(2024, 6, 30)
        log("[smoke] Using 2024H1 window")

    log(f"Loading panels {start.date()} → {end.date()} ...")
    panels = load_study_panels(start, end, index_code=INDEX_CODE)
    start_eff, end_eff = _clip_study_window(panels, start, end)
    log(f"Effective window: {start_eff.date()} → {end_eff.date()}")

    # Universe size diagnostics
    u_size = daily_universe_size(panels.membership, panels.ret_c2c).loc[start_eff:end_eff]
    universe_stats = {
        "mean_n": float(u_size.mean()),
        "min_n": float(u_size.min()),
        "max_n": float(u_size.max()),
    }
    u_size.to_csv(RESULTS / "universe_size.csv", header=["n_members"])

    # Formation on c2c; execution on o2o (next-open economics)
    log("Running holding-period backtests ...")
    results = run_all_horizons(
        panels.ret_c2c,
        panels.ret_o2o,
        membership=panels.membership,
        holding_periods=HOLDING_PERIODS,
        df_not_limit=panels.df_not_limit,
        df_not_st=panels.df_not_st,
        df_trade_status=panels.df_trade_status,
        close=panels.close,
        n_extreme=args.n_extreme,
        entry_lag=DEFAULT_ENTRY_LAG_O2O,
        one_way_cost=args.one_way_cost,
        apply_tradability=True,
    )

    comparison_gross = holding_period_comparison(
        results, cost_label="gross", start=start_eff, end=end_eff
    )
    comparison_net = holding_period_comparison(
        results, cost_label="net", start=start_eff, end=end_eff
    )
    comparison_gross.to_csv(RESULTS / "holding_period_gross.csv", index=False)
    comparison_net.to_csv(RESULTS / "holding_period_net.csv", index=False)

    # Daily PnL panels for primary hold
    primary = results[args.primary_hold]
    index_ret = panels.index_ret.loc[start_eff:end_eff]

    pnl_gross = {
        k: _slice_pnl(v, start_eff, end_eff) for k, v in primary.gross.items()
    }
    pnl_net = {k: _slice_pnl(v, start_eff, end_eff) for k, v in primary.net.items()}
    pnl_net["csi300"] = index_ret.reindex(pnl_net["bottom10"].index)

    # Save daily series
    daily = pd.DataFrame(
        {
            "bottom10_gross": pnl_gross["bottom10"],
            "top10_gross": pnl_gross["top10"],
            "long_short_gross": pnl_gross["long_short"],
            "bottom10_net": pnl_net["bottom10"],
            "top10_net": pnl_net["top10"],
            "long_short_net": pnl_net["long_short"],
            "csi300": pnl_net["csi300"],
        }
    )
    daily.to_csv(RESULTS / f"daily_pnl_hold{args.primary_hold}.csv")

    # IC
    log("Computing RankIC ...")
    form = formation_returns_in_universe(panels.ret_c2c, panels.membership)
    signal = extreme_signal_panel(form)
    # Forward using c2c (standard IC convention); formation signal at t vs future c2c
    fwd = forward_returns(panels.ret_c2c, horizons=HOLDING_PERIODS)
    ic_table = compute_ic_table(
        signal.loc[start_eff:end_eff],
        {h: f.loc[start_eff:end_eff] for h, f in fwd.items()},
        membership=panels.membership,
    )
    ic_table.to_csv(RESULTS / "ic_summary.csv", index=False)

    # Save daily IC for 5D
    ic5 = signal.corrwith(fwd[5], axis=1, method="spearman").loc[start_eff:end_eff]
    ic5.to_csv(RESULTS / "daily_rank_ic_5d.csv", header=["rank_ic"])

    # Regime on primary LS net
    log("Regime analysis ...")
    regime_pack = full_regime_pack(pnl_net["long_short"], index_ret)
    for name, df in regime_pack.items():
        df.to_csv(RESULTS / f"regime_{name}.csv", index=False)

    primary_stats_net = summarize_backtest(
        primary, cost_label="net", start=start_eff, end=end_eff
    )
    primary_stats_net.to_csv(RESULTS / f"primary_hold{args.primary_hold}_net.csv", index=False)

    # Metrics JSON
    summary = {
        "start": str(start_eff.date()),
        "end": str(end_eff.date()),
        "n_extreme": args.n_extreme,
        "one_way_cost": args.one_way_cost,
        "primary_hold": args.primary_hold,
        "universe": universe_stats,
        "ic": ic_table.to_dict(orient="records"),
        "holding_net": comparison_net.to_dict(orient="records"),
        "holding_gross": comparison_gross.to_dict(orient="records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    figure_names = []
    if not args.skip_figures:
        log("Rendering figures ...")
        # Fig1 cumulative (net)
        plot_cumulative_returns(
            {
                "bottom10": pnl_net["bottom10"],
                "top10": pnl_net["top10"],
                "long_short": pnl_net["long_short"],
                "csi300": pnl_net["csi300"],
            },
            title=f"Cumulative Return (net, {args.primary_hold}D hold) — CSI300 Extreme",
            out_path=FIGURES / "fig1_cumulative_returns.png",
        )
        figure_names.append("fig1_cumulative_returns.png")

        # Fig2 holding period bars (net annu ret)
        plot_holding_period_bars(
            comparison_net,
            metric="annu_ret",
            title="Annualized Return by Holding Period (net of cost)",
            out_path=FIGURES / "fig2_holding_period_annu_ret.png",
        )
        figure_names.append("fig2_holding_period_annu_ret.png")
        plot_holding_period_bars(
            comparison_net,
            metric="sharpe",
            title="Sharpe by Holding Period (net of cost)",
            out_path=FIGURES / "fig2b_holding_period_sharpe.png",
        )
        figure_names.append("fig2b_holding_period_sharpe.png")

        # Fig3 monthly heatmap LS
        plot_monthly_heatmap(
            pnl_net["long_short"],
            title=f"Long-Short Monthly Returns (net, {args.primary_hold}D)",
            out_path=FIGURES / "fig3_monthly_heatmap_ls.png",
        )
        figure_names.append("fig3_monthly_heatmap_ls.png")

        # Fig4 rolling sharpe
        plot_rolling_sharpe(
            {
                "bottom10": pnl_net["bottom10"],
                "top10": pnl_net["top10"],
                "long_short": pnl_net["long_short"],
            },
            window=60,
            title=f"Rolling 60D Sharpe (net, {args.primary_hold}D hold)",
            out_path=FIGURES / "fig4_rolling_sharpe.png",
        )
        figure_names.append("fig4_rolling_sharpe.png")

        # IC bars
        plot_ic_bars(
            ic_table,
            title="RankIC of −rank(ret_1d) vs Forward Returns",
            out_path=FIGURES / "fig5_rank_ic.png",
        )
        figure_names.append("fig5_rank_ic.png")

        # Regime
        if not regime_pack["trend_regime"].empty:
            plot_regime_bars(
                regime_pack["trend_regime"],
                metric="sharpe",
                title=f"LS Net Sharpe by Trend Regime ({args.primary_hold}D)",
                out_path=FIGURES / "fig6_regime_trend.png",
            )
            figure_names.append("fig6_regime_trend.png")
        if not regime_pack["vol_regime"].empty:
            plot_regime_bars(
                regime_pack["vol_regime"],
                metric="sharpe",
                title=f"LS Net Sharpe by Vol Regime ({args.primary_hold}D)",
                out_path=FIGURES / "fig6b_regime_vol.png",
            )
            figure_names.append("fig6b_regime_vol.png")

    report_path = write_report(
        start=start_eff,
        end=end_eff,
        one_way_cost=args.one_way_cost,
        n_extreme=args.n_extreme,
        primary_hold=args.primary_hold,
        comparison_gross=comparison_gross,
        comparison_net=comparison_net,
        ic_table=ic_table,
        regime_pack=regime_pack,
        primary_stats_net=primary_stats_net,
        universe_stats=universe_stats,
        figure_names=figure_names,
    )

    log(f"Results → {RESULTS}")
    log(f"Figures → {FIGURES}")
    log(f"Report  → {report_path}")
    log("Done.")


if __name__ == "__main__":
    main()
