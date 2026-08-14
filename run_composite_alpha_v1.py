#!/usr/bin/env python
"""Milestone 2.0 — Composite Alpha Engine v1 (incremental contribution test).

Models (Registry factors only; attribution-approved):
  A: TGD20
  B: TGD20 + D1_LiquidityQuality60d
  C: TGD20 + D1 + FlowDensity20

Method:
  IC-weighted CS ranks (rolling RankIC → non-negative weights, renormalized).
  No equal 50/50. No D4/D5/IdealReversal.

Does NOT modify Registry or factor formulas.

Outputs:
  research/reports/composite_alpha_v1/
    model_comparison.csv
    weights.csv
    incremental_contribution.csv
    composite_report.md
    composite_verdict.json
    charts/

Usage:
  OMP_NUM_THREADS=1 python run_composite_alpha_v1.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from liquidity_normalization import panel_cross_sectional_residual

OUT = Path("research/reports/composite_alpha_v1")
SIGNAL_SHIFT = 1
TOP_FRAC = 0.10  # research decile book (matches TGD/Flow execution research)
IC_LOOKBACK = 60


def log(msg: str) -> None:
    print(msg, flush=True)


def si_neut(panel: pd.DataFrame, ind: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    return cs_zscore(neutralize_size_industry(panel, ind.reindex_like(panel), mkt.reindex_like(panel)))


def align_all(
    panels: Dict[str, pd.DataFrame], ret: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    names = list(panels.keys())
    idx = panels[names[0]].index
    cols = panels[names[0]].columns
    for n in names[1:]:
        idx = idx.intersection(panels[n].index)
        cols = cols.intersection(panels[n].columns)
    idx = idx.intersection(ret.index)
    cols = cols.intersection(ret.columns)
    return {n: panels[n].reindex(index=idx, columns=cols) for n in names}, ret.reindex(
        index=idx, columns=cols
    )


def rolling_ic_weights(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    names: List[str],
    *,
    lookback: int = IC_LOOKBACK,
) -> pd.DataFrame:
    """Day-t weights from lookback RankIC ending at t-1 (no look-ahead). Non-neg, renormalized."""
    ics = {n: daily_rank_ic_series(panels[n], ret, signal_shift=SIGNAL_SHIFT) for n in names}
    rolls = {
        n: ics[n].rolling(lookback, min_periods=max(20, lookback // 3)).mean().shift(1) for n in names
    }
    w = pd.DataFrame({n: rolls[n].clip(lower=0.0) for n in names})
    s = w.sum(axis=1)
    equal = 1.0 / len(names)
    # warm-up / all-zero: fall back to equal weights
    bad = (s <= 0) | s.isna()
    w = w.div(s.replace(0, np.nan), axis=0)
    w.loc[bad] = equal
    return w.fillna(equal)


def ic_weighted_composite(
    panels: Dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    names: List[str],
) -> pd.DataFrame:
    """Combine CS percentile ranks with time-varying IC weights."""
    ranks = {n: panels[n].rank(axis=1, pct=True, method="average") for n in names}
    # Build composite as sum_i w_i,t * rank_i
    out = None
    for n in names:
        # broadcast weight series to panel
        wn = weights[n].reindex(ranks[n].index)
        part = ranks[n].mul(wn, axis=0)
        out = part if out is None else out.add(part, fill_value=0)
    return cs_zscore(out)


def evaluate_signal(name: str, signal: pd.DataFrame, ret: pd.DataFrame) -> dict:
    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    gross, to = daily_hl_pnl_and_turnover(
        signal, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    net = net_pnl_series(gross, to, DEFAULT_ROUND_TRIP_COST)
    # direction-adjust gross for performance reporting
    direction = 1 if gross.mean() >= 0 else -1
    perf_g = series_performance((gross * direction).dropna())
    perf_n = series_performance(net.dropna())
    return {
        "model": name,
        "rank_ic": float(ic.mean()),
        "rank_icir": float(icir_from_daily(ic)),
        "gross_sharpe": perf_g["sharpe"],
        "gross_annu_ret": perf_g["annu_ret"],
        "net_sharpe": perf_n["sharpe"],
        "net_annu_ret": perf_n["annu_ret"],
        "mdd_net": perf_n["max_drawdown"],
        "daily_turnover": float(to.mean()),
        "annu_one_way_turnover": float(to.mean() * 250 / 2.0),
        "n_days": int(ic.dropna().shape[0]),
        "cost_rt": DEFAULT_ROUND_TRIP_COST,
        "top_frac": TOP_FRAC,
        "direction": direction,
    }


def flow_residual_vs_cores(
    flow: pd.DataFrame, tgd: pd.DataFrame, d1: pd.DataFrame, ret: pd.DataFrame
) -> dict:
    """Flow ⊥ (TGD, D1) joint CS residual — marginal information test."""
    raw = daily_rank_ic_series(flow, ret, signal_shift=SIGNAL_SHIFT)
    # residual_ic_stats is single-anchor; do joint residual manually
    f = align_signal(flow, SIGNAL_SHIFT)
    a = align_signal(tgd.reindex_like(flow), SIGNAL_SHIFT)
    b = align_signal(d1.reindex_like(flow), SIGNAL_SHIFT)
    resid = panel_cross_sectional_residual(f, [a, b])
    # resid already shifted space — IC without double-shift: use signal_shift=0 on resid vs ret aligned
    # residual_ic_stats applies align_signal again; for joint we residualized aligned panels,
    # so compute IC of resid vs ret with shift=0
    ic = resid.corrwith(ret.reindex_like(resid), axis=1, method="spearman")
    return {
        "test": "Flow_perp_TGD_D1",
        "raw_icir_flow": float(icir_from_daily(raw)),
        "residual_ic_mean": float(ic.mean()) if ic.notna().any() else np.nan,
        "residual_icir": float(icir_from_daily(ic)),
        "note": "CS residual of Flow on [TGD, D1]; IC on aligned panels",
    }


def plot_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(df))
    axes[0].bar(x, df["rank_icir"], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model"], rotation=15)
    axes[0].set_title("RankICIR")
    axes[1].bar(x - 0.2, df["gross_sharpe"], 0.4, label="gross")
    axes[1].bar(x + 0.2, df["net_sharpe"], 0.4, label="net")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["model"], rotation=15)
    axes[1].legend()
    axes[1].set_title("Sharpe")
    axes[2].bar(x, df["daily_turnover"], color="darkorange")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df["model"], rotation=15)
    axes[2].set_title("Daily turnover")
    fig.suptitle("Composite v1 incremental test (IC-weighted)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    out: Path,
    meta: dict,
    comparison: pd.DataFrame,
    weights_summary: pd.DataFrame,
    incremental: pd.DataFrame,
    flow_resid: dict,
) -> None:
    a = comparison[comparison["model"] == "A_TGD"].iloc[0]
    b = comparison[comparison["model"] == "B_TGD_D1"].iloc[0]
    c = comparison[comparison["model"] == "C_TGD_D1_Flow"].iloc[0]

    def row(r):
        return (
            f"| {r['model']} | {r['rank_ic']:.4f} | {r['rank_icir']:.2f} | "
            f"{r['gross_sharpe']:.2f} | {r['net_sharpe']:.2f} | {r['mdd_net']:.3f} | "
            f"{r['daily_turnover']:.3f} |"
        )

    lines = [
        "# Composite Alpha Engine v1 — Incremental Contribution Test",
        "",
        f"**Window:** {meta['start']} → {meta['end']} ({meta['n_days']}d confirmation)",
        f"**Signal book:** size+industry CS-z · IC-weighted ranks · lookback={IC_LOOKBACK}d · top_frac={TOP_FRAC}",
        f"**Cost:** {DEFAULT_ROUND_TRIP_COST} round-trip",
        "",
        "## Models",
        "",
        "| Model | Spec |",
        "|-------|------|",
        "| A | TGD20 |",
        "| B | TGD20 + D1 (IC-weighted) |",
        "| C | TGD20 + D1 + FlowDensity20 (IC-weighted) |",
        "",
        "## Comparison",
        "",
        "| Model | RankIC | RankICIR | Gross Sharpe | Net Sharpe | MDD net | Daily TO |",
        "|-------|--------|----------|--------------|------------|---------|----------|",
        row(a),
        row(b),
        row(c),
        "",
        "## Mean IC weights (confirmation, after warm-up)",
        "",
        weights_summary.round(4).to_string(index=False),
        "",
        "## Incremental contribution",
        "",
        incremental.round(4).to_string(index=False),
        "",
        "## Flow residual vs cores",
        "",
        f"- `{flow_resid['test']}`: raw Flow ICIR={flow_resid['raw_icir_flow']:.2f}, "
        f"resid ICIR={flow_resid['residual_icir']:.2f}",
        f"- {flow_resid['note']}",
        "",
        "## Interpretation",
        "",
    ]

    d_net_ba = float(b["net_sharpe"] - a["net_sharpe"])
    d_net_cb = float(c["net_sharpe"] - b["net_sharpe"])
    d_icir_ba = float(b["rank_icir"] - a["rank_icir"])
    d_icir_cb = float(c["rank_icir"] - b["rank_icir"])

    lines.append(
        f"- **B vs A (add D1):** ΔNet Sharpe={d_net_ba:+.2f}, ΔRankICIR={d_icir_ba:+.2f}. "
        + (
            "Supports two-core combination."
            if d_net_ba > 0.05 or d_icir_ba > 0.3
            else "Weak/no net lift — review before promoting B."
        )
    )
    lines.append(
        f"- **C vs B (add Flow):** ΔNet Sharpe={d_net_cb:+.2f}, ΔRankICIR={d_icir_cb:+.2f}. "
        + (
            "Flow acts as portfolio enhancer on this book."
            if d_net_cb > 0.05
            else "Flow does **not** clearly enhance TGD+D1 on Net Sharpe — keep as satellite research only."
        )
    )
    lines.append(
        f"- Flow⊥(TGD,D1) resid ICIR={flow_resid['residual_icir']:.2f} "
        "(aligns with Attribution Review: Flow largely D1-overlapping)."
    )
    lines += [
        "",
        "## Explicit exclusions",
        "",
        "- D4, D5, IdealReversal not in Composite v1 (per Attribution Review).",
        "- No Registry writes. No formula changes.",
        "",
        "## Next",
        "",
        "- If B ≫ A: treat TGD+D1 as Composite baseline.",
        "- If C ≯ B: do not add Flow to core composite; optional D1+Flow overlay study later.",
        "- D5 direction validation remains separate.",
        "",
    ]
    (out / "composite_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--ic-lookback", type=int, default=IC_LOOKBACK)
    args = parser.parse_args()
    lookback = args.ic_lookback

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    log("=== Milestone 2.0 Composite Alpha Engine v1 ===")

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2 = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
    pv = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    float_mkt = enriched.float_mktcap.loc[start:end]

    log("Build panels ...")
    tgd, _ = build_tgd20_wide_from_eod_l2(
        start, end, open_=enriched.open, close=enriched.close, use_cache=True, window=20
    )
    tgd = tgd.loc[start:end]
    flow = build_net_active_flow_mktcap(l2, float_mkt, window=20).loc[start:end]
    d1 = build_eod_engine_factor("low_vol_liquidity_quality_60d", pv).loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    raw = {"TGD20": tgd, "D1": d1, "FlowDensity20": flow}
    conf = {}
    for k, v in raw.items():
        _, c = split_discovery_confirmation(v, args.discovery_days)
        conf[k] = c
    _, ret = split_discovery_confirmation(ret_full, args.discovery_days)
    _, ind = split_discovery_confirmation(industry, args.discovery_days)
    _, mkt = split_discovery_confirmation(float_mkt, args.discovery_days)

    panels_raw, ret = align_all(conf, ret)
    ind = ind.reindex_like(ret)
    mkt = mkt.reindex_like(ret)
    panels = {k: si_neut(v, ind, mkt) for k, v in panels_raw.items()}
    log(f"Confirmation aligned: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    # --- Model signals ---
    log("Model A: TGD ...")
    sig_a = panels["TGD20"]

    log("Model B: TGD+D1 IC-weighted ...")
    names_b = ["TGD20", "D1"]
    w_b = rolling_ic_weights(
        {n: panels[n] for n in names_b}, ret, names_b, lookback=lookback
    )
    sig_b = ic_weighted_composite({n: panels[n] for n in names_b}, w_b, names_b)

    log("Model C: TGD+D1+Flow IC-weighted ...")
    names_c = ["TGD20", "D1", "FlowDensity20"]
    w_c = rolling_ic_weights(
        {n: panels[n] for n in names_c}, ret, names_c, lookback=lookback
    )
    sig_c = ic_weighted_composite({n: panels[n] for n in names_c}, w_c, names_c)

    rows = [
        evaluate_signal("A_TGD", sig_a, ret),
        evaluate_signal("B_TGD_D1", sig_b, ret),
        evaluate_signal("C_TGD_D1_Flow", sig_c, ret),
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    log(comparison[["model", "rank_icir", "gross_sharpe", "net_sharpe", "daily_turnover"]].to_string(index=False))

    # Weights summary
    w_b_mean = w_b.dropna(how="all").mean()
    w_c_mean = w_c.dropna(how="all").mean()
    weights_summary = pd.DataFrame(
        [
            {"model": "B_TGD_D1", "factor": "TGD20", "mean_weight": float(w_b_mean.get("TGD20", np.nan))},
            {"model": "B_TGD_D1", "factor": "D1", "mean_weight": float(w_b_mean.get("D1", np.nan))},
            {"model": "C_TGD_D1_Flow", "factor": "TGD20", "mean_weight": float(w_c_mean.get("TGD20", np.nan))},
            {"model": "C_TGD_D1_Flow", "factor": "D1", "mean_weight": float(w_c_mean.get("D1", np.nan))},
            {
                "model": "C_TGD_D1_Flow",
                "factor": "FlowDensity20",
                "mean_weight": float(w_c_mean.get("FlowDensity20", np.nan)),
            },
        ]
    )
    # full daily weights
    w_b.assign(model="B").to_csv(OUT / "weights_B_daily.csv")
    w_c.assign(model="C").to_csv(OUT / "weights_C_daily.csv")
    weights_summary.to_csv(OUT / "weights.csv", index=False)

    a, b, c = rows[0], rows[1], rows[2]
    incremental = pd.DataFrame(
        [
            {
                "contrast": "B_minus_A",
                "add": "D1",
                "delta_rank_icir": b["rank_icir"] - a["rank_icir"],
                "delta_gross_sharpe": b["gross_sharpe"] - a["gross_sharpe"],
                "delta_net_sharpe": b["net_sharpe"] - a["net_sharpe"],
                "delta_daily_turnover": b["daily_turnover"] - a["daily_turnover"],
                "delta_mdd_net": b["mdd_net"] - a["mdd_net"],
            },
            {
                "contrast": "C_minus_B",
                "add": "FlowDensity20",
                "delta_rank_icir": c["rank_icir"] - b["rank_icir"],
                "delta_gross_sharpe": c["gross_sharpe"] - b["gross_sharpe"],
                "delta_net_sharpe": c["net_sharpe"] - b["net_sharpe"],
                "delta_daily_turnover": c["daily_turnover"] - b["daily_turnover"],
                "delta_mdd_net": c["mdd_net"] - b["mdd_net"],
            },
        ]
    )
    incremental.to_csv(OUT / "incremental_contribution.csv", index=False)

    flow_resid = flow_residual_vs_cores(panels["FlowDensity20"], panels["TGD20"], panels["D1"], ret)
    pd.DataFrame([flow_resid]).to_csv(OUT / "flow_residual_vs_cores.csv", index=False)
    log(f"Flow⊥(TGD,D1) resid ICIR={flow_resid['residual_icir']:.2f}")

    plot_comparison(comparison, OUT / "charts" / "model_comparison.png")

    meta = {
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "n_days": int(len(ret)),
        "ic_lookback": lookback,
        "top_frac": TOP_FRAC,
        "method": "rolling_rankic_weighted_cs_ranks",
    }
    write_report(OUT, meta, comparison, weights_summary, incremental, flow_resid)

    verdict = {
        "schema_version": "composite_alpha_v1",
        "meta": meta,
        "models": comparison.to_dict(orient="records"),
        "incremental": incremental.to_dict(orient="records"),
        "flow_residual_vs_cores": flow_resid,
        "recommendation": {
            "promote_B_if": "delta_net_sharpe(B-A) materially > 0",
            "add_Flow_if": "delta_net_sharpe(C-B) materially > 0 AND resid IC Flow⊥cores not absorbed",
            "exclude": ["D4", "D5", "IdealReversal"],
        },
    }
    (OUT / "composite_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    log(f"Wrote {OUT / 'composite_report.md'}")
    log("=== Composite v1 complete ===")


if __name__ == "__main__":
    main()
