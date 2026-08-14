#!/usr/bin/env python
"""C7 combination: C2_D1_0.60 + λ · z(net_active_flow_mktcap_20d) on confirmation.

Portfolio-layer confirmation for P2 enhancer (not factor IC).

Tracks:
  A) additive:  (1-λ)·z(C2) + λ·z(P2)          [P2 already size+ind neut]
  B) size_tight: neutralize combined signal vs industry + ln(mktcap)
                 before ranking — scheme-1 “ALL + tight size”

λ grid: 0.0 (baseline) / 0.1 / 0.2 / 0.3 / 0.4 / 0.5
Rank by ICIR among schemes with net Sharpe > 0 and annu one-way TO ≤ 100%.
Also report aspirational TO ≤ 50% fence from mentor brief.

Usage:
  OMP_NUM_THREADS=1 python run_combination_c7.py
  OMP_NUM_THREADS=1 python run_combination_c7.py --discovery-days 504 --cost 0.0015
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    apply_tradability_mask,
    build_long_short_weights,
    classify_market_regimes,
    daily_hl_pnl_and_turnover,
    evaluate_investability,
    net_pnl_series,
    strip_internal,
    yearly_net_sharpes,
)
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from run_alpha_combination_v1 import blend_base3, load_masks
from run_l2_validation import load_context

OUT = Path("research/reports/l2_flow_density_v1/combination_c7")
D1 = "low_vol_liquidity_quality_60d"
D4 = "winner_sentiment_reversal_5d"
D5 = "upside_fragility_20d"
BASE3 = [D1, D4, D5]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
MAX_NAME_WEIGHT = 0.01
TO_HARD = 100.0
TO_SOFT = 50.0


def log(msg: str) -> None:
    print(msg, flush=True)


def capped_ls_weights(
    signal: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    max_w: float = MAX_NAME_WEIGHT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight L/S with per-name abs weight cap (renormalize within book)."""
    w_long, w_short = build_long_short_weights(signal, top_frac=top_frac, bottom_frac=top_frac)

    def _cap(w: pd.DataFrame) -> pd.DataFrame:
        capped = w.clip(upper=max_w)
        # zero out non-members then renormalize rows that still have mass
        row_sum = capped.sum(axis=1).replace(0, np.nan)
        return capped.div(row_sum, axis=0)

    return _cap(w_long), _cap(w_short)


def size_exposure_ls(
    signal: pd.DataFrame,
    log_size_z: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    signal_shift: int = 1,
) -> pd.Series:
    """Daily long−short exposure to CS-z ln(mktcap)."""
    sig = align_signal(signal, signal_shift)
    sz = log_size_z.reindex_like(sig)
    w_long, w_short = capped_ls_weights(sig, top_frac=top_frac)
    long_exp = w_long.mul(sz).sum(axis=1)
    short_exp = w_short.mul(sz).sum(axis=1)
    return long_exp - short_exp


def industry_active_l1(
    signal: pd.DataFrame,
    industry: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    signal_shift: int = 1,
) -> pd.Series:
    """Daily L1 of industry net weights |w_long_ind − w_short_ind| summed."""
    sig = align_signal(signal, signal_shift)
    ind = industry.reindex_like(sig)
    w_long, w_short = capped_ls_weights(sig, top_frac=top_frac)
    w_ls = w_long.fillna(0) - w_short.fillna(0)

    out = []
    for dt_i in w_ls.index:
        row = w_ls.loc[dt_i]
        ind_row = ind.loc[dt_i]
        valid = row.notna() & ind_row.notna()
        if valid.sum() < 10:
            out.append(np.nan)
            continue
        g = row[valid].groupby(ind_row[valid]).sum()
        out.append(float(g.abs().sum()))
    return pd.Series(out, index=w_ls.index)


def holding_mktcap_distribution(
    signal: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    *,
    top_frac: float = 0.2,
    signal_shift: int = 1,
) -> dict:
    """Median long/short float-mktcap percentiles vs universe (time-averaged)."""
    sig = align_signal(signal, signal_shift)
    mcap = float_mktcap.reindex_like(sig)
    ranks = mcap.rank(axis=1, pct=True)
    w_long, w_short = capped_ls_weights(sig, top_frac=top_frac)
    long_pct = ranks.where(w_long > 0).mean(axis=1)
    short_pct = ranks.where(w_short > 0).mean(axis=1)
    uni_med = ranks.median(axis=1)
    return {
        "long_mktcap_pct_mean": float(long_pct.mean()),
        "short_mktcap_pct_mean": float(short_pct.mean()),
        "universe_mktcap_pct_median_mean": float(uni_med.mean()),
        "long_minus_uni_pct": float((long_pct - uni_med).mean()),
        "short_minus_uni_pct": float((short_pct - uni_med).mean()),
    }


def evaluate_combo(
    label: str,
    track: str,
    lambda_: float,
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    trad_kw: dict,
    log_size_z: pd.DataFrame,
    industry: pd.DataFrame,
    float_mktcap: pd.DataFrame,
    *,
    baseline_net: Optional[float] = None,
    baseline_icir: Optional[float] = None,
    baseline_pnl: Optional[pd.Series] = None,
) -> Tuple[dict, pd.Series]:
    inv = evaluate_investability(signal, ret, **trad_kw)
    size_exp = size_exposure_ls(signal, log_size_z)
    ind_l1 = industry_active_l1(signal, industry)
    mcap_dist = holding_mktcap_distribution(signal, float_mktcap)

    # name weight check on a mid sample day
    sig_a = align_signal(signal, 1)
    mid = sig_a.index[len(sig_a) // 2]
    wl, ws = capped_ls_weights(sig_a.loc[[mid]])
    max_w = float(max(wl.loc[mid].max(), ws.loc[mid].max()))

    row = {
        "label": label,
        "track": track,
        "lambda": lambda_,
        **strip_internal(inv),
        "yearly_net_sharpe": yearly_net_sharpes(inv["_net_pnl"]),
        "size_exposure_mean": float(size_exp.mean()),
        "size_exposure_abs_mean": float(size_exp.abs().mean()),
        "size_exposure_abs_p90": float(size_exp.abs().quantile(0.9)),
        "industry_active_l1_mean": float(ind_l1.mean()),
        "max_name_weight_midday": max_w,
        **mcap_dist,
        "gate_net_sharpe_gt_0": bool(
            pd.notna(inv["net_sharpe_tradable"]) and inv["net_sharpe_tradable"] > 0
        ),
        "gate_to_le_100": bool(
            pd.notna(inv["annu_one_way_turnover"]) and inv["annu_one_way_turnover"] <= TO_HARD
        ),
        "gate_to_le_50": bool(
            pd.notna(inv["annu_one_way_turnover"]) and inv["annu_one_way_turnover"] <= TO_SOFT
        ),
        "gate_size_abs_mean_le_0.2": bool(
            pd.notna(size_exp.abs().mean()) and size_exp.abs().mean() <= 0.2
        ),
    }
    row["research_feasible"] = bool(row["gate_net_sharpe_gt_0"] and row["gate_to_le_100"])
    if baseline_net is not None and pd.notna(inv["net_sharpe_tradable"]):
        row["net_sharpe_delta_vs_c2"] = float(inv["net_sharpe_tradable"] - baseline_net)
    if baseline_icir is not None and pd.notna(inv["icir_tradable"]):
        row["icir_delta_vs_c2"] = float(inv["icir_tradable"] - baseline_icir)
    if baseline_pnl is not None:
        excess = inv["_net_pnl"] - baseline_pnl.reindex(inv["_net_pnl"].index)
        row["excess_annu_ret"] = float(excess.mean() * 250) if excess.notna().any() else np.nan
        row["excess_sharpe"] = (
            float(excess.mean() / excess.std() * np.sqrt(250))
            if excess.std() and excess.std() > 0
            else np.nan
        )
    return row, inv["_net_pnl"]


def save_excess_curve(
    curves: Dict[str, pd.Series],
    baseline: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    base = curves[baseline]
    for label, pnl in curves.items():
        if label == baseline:
            continue
        excess = (pnl - base.reindex(pnl.index)).fillna(0)
        ax.plot(excess.index, excess.cumsum(), label=label, linewidth=1.2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("C7 excess cumulative net PnL vs C2_D1_0.60")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylabel("cum excess (net)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_lambda_heatmap(df: pd.DataFrame, path: Path) -> None:
    """λ × track heatmap of ICIR and net Sharpe."""
    metrics = ["icir_tradable", "net_sharpe_tradable", "annu_one_way_turnover"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        pivot = df.pivot(index="track", columns="lambda", values=metric)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn" if "turnover" not in metric else "RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.1f}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_title(metric)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("C7 λ-grid (ICIR / net Sharpe / annu TO)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_size_hist(size_exp: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(size_exp.dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(size_exp.mean(), color="red", label=f"mean={size_exp.mean():.3f}")
    ax.axvline(0.2, color="orange", linestyle=":", label="±0.2σ")
    ax.axvline(-0.2, color="orange", linestyle=":")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)  # create before any tee/redirect

    log("=== C7: C2_D1_0.60 + λ·P2 (confirmation) ===")
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    ret_full = ctx["ret"]
    _, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    log(f"Confirmation: {ret_conf.index[0].date()} -> {ret_conf.index[-1].date()} ({len(ret_conf)}d)")

    panels = {
        n: ctx["frozen_panels"][n].reindex(index=ret_conf.index, columns=ret_conf.columns)
        for n in BASE3
    }
    masks = load_masks(start, end)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

    float_mkt = enriched.float_mktcap.loc[start:end]
    ind_full = industry.reindex(index=float_mkt.index, columns=float_mkt.columns)
    raw_p2 = build_net_active_flow_mktcap(l2_cache, float_mkt, window=20).loc[start:end]
    p2_neut = cs_zscore(neutralize_size_industry(raw_p2, ind_full, float_mkt))
    p2 = p2_neut.reindex(index=ret_conf.index, columns=ret_conf.columns)

    float_c = float_mkt.reindex(index=ret_conf.index, columns=ret_conf.columns)
    ind_c = ind_full.reindex(index=ret_conf.index, columns=ret_conf.columns)
    log_size = np.log(float_c.replace(0, np.nan))
    log_size_z = cs_zscore(log_size)
    close_c = enriched.close.reindex(index=ret_conf.index, columns=ret_conf.columns)
    amount_c = enriched.amount.reindex(index=ret_conf.index, columns=ret_conf.columns)

    trad_kw = dict(
        df_not_limit=masks["df_not_limit"].reindex(index=ret_conf.index, columns=ret_conf.columns),
        df_not_st=masks["df_not_st"].reindex(index=ret_conf.index, columns=ret_conf.columns),
        df_trade_status=masks["df_trade_status"].reindex(index=ret_conf.index, columns=ret_conf.columns),
        close=close_c,
        amount=amount_c,
        round_trip_cost=args.cost,
        apply_tradability=True,
    )

    c2 = blend_base3(panels, 0.60)
    z_c2 = cs_zscore(c2)
    z_p2 = cs_zscore(p2)

    results: List[dict] = []
    curves: Dict[str, pd.Series] = {}
    size_series: Dict[str, pd.Series] = {}

    # Baseline first
    log("\n--- Baseline C2_D1_0.60 ---")
    row0, pnl0 = evaluate_combo(
        "C2_D1_0.60",
        "baseline",
        0.0,
        c2,
        ret_conf,
        trad_kw,
        log_size_z,
        ind_c,
        float_c,
    )
    results.append(row0)
    curves["C2_D1_0.60"] = pnl0
    size_series["C2_D1_0.60"] = size_exposure_ls(c2, log_size_z)
    base_net = row0["net_sharpe_tradable"]
    base_icir = row0["icir_tradable"]
    log(
        f"  ICIR={base_icir:.3f} net={base_net:.3f} TO={row0['annu_one_way_turnover']:.1f} "
        f"size|μ|={row0['size_exposure_abs_mean']:.3f}"
    )

    for track in ("additive", "size_tight"):
        log(f"\n--- Track {track} ---")
        for lam in LAMBDAS:
            if lam == 0.0:
                continue
            combo = (1.0 - lam) * z_c2 + lam * z_p2
            if track == "size_tight":
                combo = cs_zscore(neutralize_size_industry(combo, ind_c, float_c))
            label = f"C7_{track}_λ{lam:.1f}"
            row, pnl = evaluate_combo(
                label,
                track,
                lam,
                combo,
                ret_conf,
                trad_kw,
                log_size_z,
                ind_c,
                float_c,
                baseline_net=base_net,
                baseline_icir=base_icir,
                baseline_pnl=pnl0,
            )
            results.append(row)
            curves[label] = pnl
            size_series[label] = size_exposure_ls(combo, log_size_z)
            log(
                f"  λ={lam:.1f} ICIR={row['icir_tradable']:.3f} "
                f"ΔICIR={row.get('icir_delta_vs_c2', np.nan):+.3f} "
                f"net={row['net_sharpe_tradable']:.3f} "
                f"Δnet={row.get('net_sharpe_delta_vs_c2', np.nan):+.3f} "
                f"TO={row['annu_one_way_turnover']:.1f} "
                f"size|μ|={row['size_exposure_abs_mean']:.3f} "
                f"feas={row['research_feasible']}"
            )

    df = pd.DataFrame(results)
    # Rank feasible by ICIR
    feas = df[df["research_feasible"]].copy()
    if not feas.empty:
        feas = feas.sort_values(["icir_tradable", "net_sharpe_tradable"], ascending=False)
        best = feas.iloc[0].to_dict()
    else:
        best = df.sort_values("icir_tradable", ascending=False).iloc[0].to_dict()

    # Prefer size_tight if ICIR within 0.1 of best additive and size gate passes
    size_ok = feas[feas["gate_size_abs_mean_le_0.2"]] if not feas.empty else feas
    recommended = best
    if not size_ok.empty:
        # among size-ok, max ICIR; if close to unconstrained best, prefer size_tight
        cand = size_ok.iloc[0].to_dict()
        if cand["track"] == "size_tight" or (
            best.get("size_exposure_abs_mean", 99) > 0.2
            and cand["icir_tradable"] >= best["icir_tradable"] - 0.15
        ):
            recommended = cand

    # Persist
    flat_cols = [
        "label",
        "track",
        "lambda",
        "rank_ic_tradable",
        "icir_tradable",
        "gross_sharpe_tradable",
        "net_sharpe_tradable",
        "net_annu_ret_tradable",
        "net_max_drawdown_tradable",
        "annu_one_way_turnover",
        "icir_delta_vs_c2",
        "net_sharpe_delta_vs_c2",
        "excess_annu_ret",
        "excess_sharpe",
        "size_exposure_mean",
        "size_exposure_abs_mean",
        "industry_active_l1_mean",
        "long_mktcap_pct_mean",
        "short_mktcap_pct_mean",
        "max_name_weight_midday",
        "gate_net_sharpe_gt_0",
        "gate_to_le_100",
        "gate_to_le_50",
        "gate_size_abs_mean_le_0.2",
        "research_feasible",
    ]
    summary = df[[c for c in flat_cols if c in df.columns]].copy()
    summary.to_csv(OUT / "c7_lambda_grid.csv", index=False)

    save_excess_curve(curves, "C2_D1_0.60", OUT / "excess_cum_net_vs_c2.png")
    save_lambda_heatmap(df[df["track"] != "baseline"], OUT / "lambda_heatmap.png")

    best_label = recommended["label"]
    if best_label in size_series:
        save_size_hist(
            size_series[best_label],
            OUT / "best_size_exposure_hist.png",
            f"{best_label} — L/S size exposure (σ)",
        )

    # Cumulative net curves (absolute)
    fig, ax = plt.subplots(figsize=(11, 5))
    for lab in ["C2_D1_0.60", best_label]:
        if lab in curves:
            ax.plot(curves[lab].index, curves[lab].cumsum(), label=lab, linewidth=1.4)
    ax.set_title("Cumulative net PnL — C2 vs recommended C7")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cum_net_c2_vs_best_c7.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    verdict = {
        "period": f"{ret_conf.index[0].date()} -> {ret_conf.index[-1].date()}",
        "n_days": int(len(ret_conf)),
        "base": "C2_D1_0.60",
        "enhancer": "net_active_flow_mktcap_20d",
        "cost_rt": args.cost,
        "constraints_note": {
            "name_weight_cap": MAX_NAME_WEIGHT,
            "size_style": "reported; size_tight track residualizes combo vs industry+ln(mktcap)",
            "industry": "reported via industry active L1; signal-level neut on size_tight",
            "turnover_hard": TO_HARD,
            "turnover_soft_aspirational": TO_SOFT,
        },
        "baseline": {
            k: row0[k]
            for k in [
                "icir_tradable",
                "net_sharpe_tradable",
                "annu_one_way_turnover",
                "size_exposure_abs_mean",
                "long_mktcap_pct_mean",
            ]
            if k in row0
        },
        "recommended": {
            k: recommended.get(k)
            for k in flat_cols
            if k in recommended
        },
        "best_unconstrained_feasible": {
            k: best.get(k) for k in ["label", "track", "lambda", "icir_tradable", "net_sharpe_tradable"]
        },
        "pass_combo_gate": bool(
            recommended.get("research_feasible")
            and recommended.get("icir_delta_vs_c2", -1) > 0
            and recommended.get("net_sharpe_delta_vs_c2", -1) >= 0
        ),
        "minute_heatmap_note": (
            "Intraday bartime×horizon heatmaps live in Intraday_Factor_Test_Process.py / "
            "intraday_lib.create_group_heatmap — for minute-bar factors, not this daily C7 combo."
        ),
        "next": [
            "If pass_combo_gate: update production_stack_v3_design.md with P2 additive enhancer",
            "If size_tight preferred: document ALL + size-residual as production construction",
        ],
    }
    (OUT / "c7_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    # README
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# C7 Combination — C2_D1_0.60 + λ·P2",
                "",
                f"**Period:** {verdict['period']} ({verdict['n_days']}d)",
                f"**Recommended:** `{recommended.get('label')}`",
                f"**Pass combo gate:** {verdict['pass_combo_gate']}",
                "",
                "## Recommended metrics",
                f"- ICIR: {recommended.get('icir_tradable')}",
                f"- ΔICIR vs C2: {recommended.get('icir_delta_vs_c2')}",
                f"- net Sharpe: {recommended.get('net_sharpe_tradable')}",
                f"- Δnet vs C2: {recommended.get('net_sharpe_delta_vs_c2')}",
                f"- annu TO 1-way: {recommended.get('annu_one_way_turnover')}",
                f"- size |exposure| mean: {recommended.get('size_exposure_abs_mean')}",
                "",
                "## Artifacts",
                "- `c7_lambda_grid.csv`",
                "- `c7_verdict.json`",
                "- `excess_cum_net_vs_c2.png`",
                "- `lambda_heatmap.png`",
                "- `cum_net_c2_vs_best_c7.png`",
                "- `best_size_exposure_hist.png`",
                "",
            ]
        )
    )

    log("\n=== C7 VERDICT ===")
    log(f"Recommended: {recommended.get('label')}")
    log(
        f"  ICIR={recommended.get('icir_tradable'):.3f} "
        f"(Δ={recommended.get('icir_delta_vs_c2', float('nan')):+.3f}) "
        f"net={recommended.get('net_sharpe_tradable'):.3f} "
        f"(Δ={recommended.get('net_sharpe_delta_vs_c2', float('nan')):+.3f}) "
        f"TO={recommended.get('annu_one_way_turnover'):.1f} "
        f"size|μ|={recommended.get('size_exposure_abs_mean'):.3f}"
    )
    log(f"pass_combo_gate={verdict['pass_combo_gate']}")
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
