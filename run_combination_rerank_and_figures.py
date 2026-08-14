#!/usr/bin/env python
"""Re-rank combination results with alpha-research metrics + publish decile figures.

Primary sort: ICIR (information quality)
Constraint: annu one-way turnover ≤ 120%, net Sharpe > 0 (feasibility fence)
Tie-break: gross Sharpe, then net Sharpe

Also publishes decile / IC figures for D1, D4, D5, C1, and recommended blend.

Usage:
  OMP_NUM_THREADS=1 python run_combination_rerank_and_figures.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_d4_expansion_stack import daily_rank_ic_series, decile_group_means
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_research_report import monotonicity_score, save_cumulative_decile_figure
from factor_attribution import align_signal, cs_zscore, hl_sharpe_from_composite
from run_l2_validation import load_context

OUT = Path("research/results/alpha_combination_v1")
FIG_ROOT = Path("research/alpha_library_v1/figures")
LIBRARY_PATH = Path("research/alpha_library_v1/alpha_library_v1.0-frozen.json")
POOL_PATH = Path("research/frozen_candidate_pool_v1.json")
RESULTS_CSV = OUT / "combination_results.csv"

D1 = "low_vol_liquidity_quality_60d"
D4 = "winner_sentiment_reversal_5d"
D5 = "upside_fragility_20d"
MAX_TURNOVER = 120.0


def log(msg: str) -> None:
    print(msg, flush=True)


def blend(panels, w_d1: float) -> pd.DataFrame:
    rest = (1.0 - w_d1) / 2.0
    return w_d1 * cs_zscore(panels[D1]) + rest * cs_zscore(panels[D4]) + rest * cs_zscore(panels[D5])


def research_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply corrected ranking: ICIR primary, TO/net as fences."""
    out = df.copy()
    out["gate_turnover_le_120"] = out["annu_one_way_turnover"] <= MAX_TURNOVER
    out["gate_net_sharpe_gt_0"] = out["net_sharpe_tradable"] > 0
    out["research_feasible"] = out["gate_turnover_le_120"] & out["gate_net_sharpe_gt_0"]
    # REF cancel has highest ICIR but fails TO — keep visible
    out = out.sort_values(
        ["research_feasible", "icir_tradable", "gross_sharpe_tradable", "net_sharpe_tradable"],
        ascending=[False, False, False, False],
    )
    return out


def publish_summary_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    cols = [
        "label",
        "rank_ic_tradable",
        "icir_tradable",
        "gross_sharpe_tradable",
        "annu_one_way_turnover",
        "net_sharpe_tradable",
        "research_feasible",
    ]
    summary = df[cols].copy()
    summary.columns = ["scheme", "IC", "ICIR", "gross_Sharpe", "annu_TO_1way", "net_Sharpe", "feasible"]
    summary.to_csv(path, index=False)
    return summary


def build_group_cum_pnl(signal: pd.DataFrame, ret: pd.DataFrame, n: int = 10):
    """Silent groupTest → (cum_pnl, group_pnl, group_to)."""
    import Factor_Dev_Lib as fdl

    sig = align_signal(signal, 1)
    r = ret.reindex_like(sig)
    old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        _, group_pnl_df, group_to_df = fdl.groupTest(sig, r, n=n, fee=0, info="silent")
    finally:
        plt.show = old_show
    return group_pnl_df.cumsum(), group_pnl_df, group_to_df


def publish_signal_figures(name: str, signal: pd.DataFrame, ret: pd.DataFrame) -> dict:
    from factor_runner import compute_group_stats, format_group_stats_title

    fig_dir = FIG_ROOT / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    log(f"  figures -> {fig_dir}")

    cum, group_pnl_df, group_to_df = build_group_cum_pnl(signal, ret)
    sig = align_signal(signal, 1)
    r = ret.reindex_like(sig)
    stats = compute_group_stats(sig, r, group_pnl_df, group_to_df)
    gmeans = decile_group_means(signal, ret)
    mono = monotonicity_score(gmeans)
    ic_daily = daily_rank_ic_series(signal, ret)

    title = format_group_stats_title(stats) + f", mono={mono:.2f}"
    save_cumulative_decile_figure(
        cum, factor_name=name, stats_title=title, out_path=fig_dir / "cumulative_long_short.png"
    )
    save_quantile_bar(gmeans, name, fig_dir / "quantile_return.png")
    save_ic_figure(ic_daily, name, fig_dir / "rank_ic_timeseries.png")

    return {
        "factor": name,
        "rank_ic": stats["rank_ic_mean"],
        "icir": stats["icir"],
        "gross_hl_sharpe": stats["hl_sharpe"],
        "implied_annu_fee": stats["implied_annu_fee"],
        "monotonicity": mono,
        "figures": {
            "cumulative": str(fig_dir / "cumulative_long_short.png"),
            "quantile": str(fig_dir / "quantile_return.png"),
            "ic_ts": str(fig_dir / "rank_ic_timeseries.png"),
        },
    }


def save_ic_figure(ic_daily: pd.Series, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ic_daily.plot(ax=ax, alpha=0.35, linewidth=0.8, label="daily IC")
    ic_daily.rolling(20).mean().plot(ax=ax, color="red", label="20d MA")
    ic_daily.rolling(252, min_periods=60).mean().plot(ax=ax, color="darkblue", label="252d MA")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{name} — rank IC (confirmation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_quantile_bar(group_means: pd.Series, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    group_means.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"{name} — decile mean daily return")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean daily return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_c2_icir_curve(df: pd.DataFrame, path: Path) -> None:
    c2 = df[df["label"].str.startswith("C2_D1_")].copy()
    c2["w1"] = c2["label"].str.extract(r"C2_D1_([\d.]+)").astype(float)
    c2 = c2.sort_values("w1")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(c2["w1"], c2["icir_tradable"], "o-", color="steelblue", label="ICIR")
    ax1.set_xlabel("D1 weight")
    ax1.set_ylabel("ICIR", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(c2["w1"], c2["gross_sharpe_tradable"], "s--", color="darkorange", label="gross Sharpe")
    ax2.set_ylabel("Gross Sharpe", color="darkorange")
    ax1.set_title("C2: D1 weight vs ICIR / gross Sharpe (confirmation)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    c2[["w1", "rank_ic_tradable", "icir_tradable", "gross_sharpe_tradable", "annu_one_way_turnover", "net_sharpe_tradable"]].to_csv(
        OUT / "c2_d1_tilt_icir_curve.csv", index=False
    )


def publish_signal_figures(name: str, signal: pd.DataFrame, ret: pd.DataFrame) -> dict:
    fig_dir = FIG_ROOT / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    log(f"  figures -> {fig_dir}")

    cum = build_group_cum_pnl(signal, ret)
    sharpe, ann_ret, direction = hl_sharpe_from_composite(signal, ret)
    ic_daily = daily_rank_ic_series(signal, ret)
    ic_mean = float(ic_daily.mean())
    icir = float(ic_mean / ic_daily.std() * np.sqrt(250)) if ic_daily.std() > 0 else np.nan
    gmeans = decile_group_means(signal, ret)
    mono = monotonicity_score(gmeans)

    title = (
        f"H-L dir={direction}, AnnuRet={ann_ret:.2%}, Sharpe={sharpe:.2f}, "
        f"IC={ic_mean:.4f}, ICIR={icir:.2f}, mono={mono:.2f}"
    )
    save_cumulative_decile_figure(
        cum, factor_name=name, stats_title=title, out_path=fig_dir / "cumulative_long_short.png"
    )
    save_quantile_bar(gmeans, name, fig_dir / "quantile_return.png")
    save_ic_figure(ic_daily, name, fig_dir / "rank_ic_timeseries.png")

    return {
        "factor": name,
        "rank_ic": ic_mean,
        "icir": icir,
        "gross_hl_sharpe": sharpe,
        "monotonicity": mono,
        "figures": {
            "cumulative": str(fig_dir / "cumulative_long_short.png"),
            "quantile": str(fig_dir / "quantile_return.png"),
            "ic_ts": str(fig_dir / "rank_ic_timeseries.png"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    log("=== Re-rank by ICIR (alpha-research paradigm) ===")
    raw = pd.read_csv(RESULTS_CSV)
    ranked = research_rank(raw)
    summary = publish_summary_table(ranked, OUT / "combination_results_research_rank.csv")
    log("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    feasible = ranked[ranked["research_feasible"]]
    if feasible.empty:
        raise RuntimeError("No feasible schemes under TO≤120 and net Sharpe>0")

    recommended = feasible.iloc[0]
    # Parse w1 if C2
    best_w1 = None
    if str(recommended["label"]).startswith("C2_D1_"):
        best_w1 = float(str(recommended["label"]).split("_")[-1])

    log(
        f"\nRecommended (ICIR-primary): {recommended['label']} | "
        f"IC={recommended['rank_ic_tradable']:.4f} ICIR={recommended['icir_tradable']:.3f} "
        f"gross={recommended['gross_sharpe_tradable']:.3f} TO={recommended['annu_one_way_turnover']:.1f} "
        f"net={recommended['net_sharpe_tradable']:.3f}"
    )

    save_c2_icir_curve(ranked, FIG_ROOT / "c2_d1_weight_vs_icir.png")

    figure_meta = {}
    if not args.skip_figures:
        log("\n=== Decile / IC figures (confirmation) ===")
        ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
        ret_full = ctx["ret"]
        _, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
        panels = {
            n: ctx["frozen_panels"][n].reindex(index=ret_conf.index, columns=ret_conf.columns)
            for n in [D1, D4, D5]
        }
        for name, key in [("D1_low_vol_liquidity", D1), ("D4_winner_sentiment", D4), ("D5_upside_fragility", D5)]:
            figure_meta[name] = publish_signal_figures(name, panels[key], ret_conf)

        sig_c1 = blend(panels, 1 / 3)
        figure_meta["C1_base3_equal"] = publish_signal_figures("C1_base3_equal", sig_c1, ret_conf)

        w1 = best_w1 if best_w1 is not None else 0.60
        sig_rec = blend(panels, w1)
        rec_name = f"C2_D1_{w1:.2f}_recommended"
        figure_meta[rec_name] = publish_signal_figures(rec_name, sig_rec, ret_conf)

    verdict = {
        "version": "alpha_combination_v1",
        "ranking_paradigm": "ICIR_primary_turnover_and_net_sharpe_as_feasibility_fence",
        "frozen_at": "2026-07-10",
        "period": "confirmation_951d",
        "gates": {
            "max_annu_one_way_turnover": MAX_TURNOVER,
            "min_net_sharpe": 0.0,
            "primary_sort": "icir_tradable",
            "tie_break": ["gross_sharpe_tradable", "net_sharpe_tradable"],
        },
        "recommended": {
            "label": recommended["label"],
            "weights": (
                {"D1": best_w1, "D4": (1 - best_w1) / 2, "D5": (1 - best_w1) / 2}
                if best_w1 is not None
                else None
            ),
            "IC": float(recommended["rank_ic_tradable"]),
            "ICIR": float(recommended["icir_tradable"]),
            "gross_Sharpe": float(recommended["gross_sharpe_tradable"]),
            "annu_one_way_turnover": float(recommended["annu_one_way_turnover"]),
            "net_Sharpe": float(recommended["net_sharpe_tradable"]),
            "note": (
                "Selected by highest ICIR among schemes with TO≤120% and net Sharpe>0. "
                "Net Sharpe is a feasibility fence, not the primary ranking metric."
            ),
        },
        "ranking_top_feasible": feasible.head(5)[
            ["label", "rank_ic_tradable", "icir_tradable", "gross_sharpe_tradable", "annu_one_way_turnover", "net_sharpe_tradable"]
        ].to_dict(orient="records"),
        "eliminated_high_icir_examples": ranked[
            (~ranked["research_feasible"]) & (ranked["icir_tradable"] > feasible.iloc[0]["icir_tradable"] - 0.5)
        ][["label", "icir_tradable", "annu_one_way_turnover", "net_sharpe_tradable", "gate_turnover_le_120", "gate_net_sharpe_gt_0"]]
        .head(5)
        .to_dict(orient="records"),
        "figures": figure_meta,
        "artifacts": {
            "research_rank_csv": str(OUT / "combination_results_research_rank.csv"),
            "c2_icir_curve_csv": str(OUT / "c2_d1_tilt_icir_curve.csv"),
            "c2_icir_curve_png": str(FIG_ROOT / "c2_d1_weight_vs_icir.png"),
            "figures_root": str(FIG_ROOT),
        },
    }
    (OUT / "alpha_combination_v1.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    # Update library
    if LIBRARY_PATH.exists():
        lib = json.loads(LIBRARY_PATH.read_text())
        lib["combination_layer_v1"] = {
            "ranking_paradigm": "ICIR_primary; turnover≤120% and net Sharpe>0 as fences",
            "artifact": str(OUT / "alpha_combination_v1.json"),
            "recommended": verdict["recommended"],
            "figures_root": str(FIG_ROOT),
            "note": (
                "Prior C2_D1_0.80 was net-Sharpe-maximizing; research-rank prefers C2 near w1=0.60 "
                "(highest ICIR in feasible set). Net Sharpe remains a production gate only."
            ),
        }
        lib["next_phase"] = "production_stack_v3_on_icir_optimal_tilt"
        LIBRARY_PATH.write_text(json.dumps(lib, indent=2, ensure_ascii=False, default=str) + "\n")

    if POOL_PATH.exists():
        pool = json.loads(POOL_PATH.read_text())
        pool["combination_v1_recommended"] = {
            "label": recommended["label"],
            "weights": verdict["recommended"]["weights"],
            "ICIR": float(recommended["icir_tradable"]),
            "gross_Sharpe": float(recommended["gross_sharpe_tradable"]),
            "net_Sharpe": float(recommended["net_sharpe_tradable"]),
            "ranking": "ICIR_primary",
            "artifact": str(OUT / "alpha_combination_v1.json"),
        }
        pool["next_phase"] = "production_stack_v3_on_icir_optimal_tilt"
        POOL_PATH.write_text(json.dumps(pool, indent=2, ensure_ascii=False, default=str) + "\n")

    log(f"\nPublished -> {OUT / 'alpha_combination_v1.json'}")
    log(f"Figures -> {FIG_ROOT}")


if __name__ == "__main__":
    main()
