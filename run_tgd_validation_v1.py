#!/usr/bin/env python
"""TGD Stage-4 validation — research validation (not formula tuning).

Layers:
  A  Replication sanity — 10-group + RankIC / ICIR / monotonicity
  B  Neutralization — raw / size / industry / size+industry
  C  Period stability — yearly + multi-year blocks (positive IC ratio)
  D  Cost / turnover — Implied AnnuFee + net Sharpe@15bp

Critical:
  signal = TGD20.shift(1)   # day-T Gu/Gd use full session → predict T+1 only
  do NOT modify core/l2_features/tgd.py

Usage:
  OMP_NUM_THREADS=1 python run_tgd_validation_v1.py
  OMP_NUM_THREADS=1 python run_tgd_validation_v1.py --sample-days 252
  OMP_NUM_THREADS=1 python run_tgd_validation_v1.py --refresh-cache
"""

from __future__ import annotations

import argparse
import datetime as dt
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
import factor_config as cfg
from alpha_d4_expansion_stack import daily_rank_ic_series
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    evaluate_investability,
    long_book_excess_performance,
)
from alpha_research_report import save_cumulative_decile_figure
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from factor_runner import compute_group_stats, format_group_stats_title
from industry_neutral import load_citics_industry_panel, panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual

OUT = Path("research/reports/tgd_v1")
FACTOR = "TGD20"
SIGNAL_SHIFT = 1  # mandatory: no same-day lookahead
N_GROUPS = 10

PERIOD_BLOCKS = [
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs(root: Path) -> dict:
    sub = {
        "root": root,
        "ic": root / "ic",
        "portfolio": root / "portfolio",
        "neutralization": root / "neutralization",
        "stability": root / "stability",
        "cost": root / "cost",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub


def build_neut_ladder(
    raw: pd.DataFrame,
    industry: pd.DataFrame,
    float_mkt: pd.DataFrame,
) -> dict:
    log_size = np.log(float_mkt.replace(0, np.nan)).reindex_like(raw)
    ind = industry.reindex_like(raw)
    return {
        "raw": cs_zscore(raw),
        "size": cs_zscore(panel_cross_sectional_residual(raw, [log_size])),
        "industry": cs_zscore(panel_industry_demean(raw, ind)),
        "size_industry": cs_zscore(neutralize_size_industry(raw, ind, float_mkt)),
    }


def decile_mean_returns(group_pnl: pd.DataFrame, n: int = N_GROUPS) -> pd.Series:
    cols = [c for c in group_pnl.columns if c != "H-L"]
    # Factor_Dev_Lib uses group labels 1..n or similar
    means = group_pnl[cols].mean()
    return means


def monotonicity_spearman(decile_means: pd.Series) -> float:
    """Spearman(group_order, mean_ret). High |rho| ⇒ ordered; sign = direction."""
    if len(decile_means) < 3:
        return float("nan")
    # try numeric group ids
    try:
        order = pd.to_numeric(decile_means.index, errors="coerce")
        if order.isna().all():
            order = np.arange(1, len(decile_means) + 1, dtype=float)
        else:
            order = order.to_numpy(dtype=float)
    except Exception:
        order = np.arange(1, len(decile_means) + 1, dtype=float)
    y = decile_means.to_numpy(dtype=float)
    mask = np.isfinite(order) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    return float(pd.Series(order[mask]).corr(pd.Series(y[mask]), method="spearman"))


def plot_decile_bars(means: pd.Series, path: Path, title: str, caption: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(means))
    ax.bar(x, means.values, color="steelblue", edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in means.index], rotation=0)
    ax.set_ylabel("Mean daily return")
    ax.set_title(title)
    ax.set_xlabel(caption, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hml_curve(hml: pd.Series, path: Path, title: str, caption: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    cum = (1.0 + hml.fillna(0)).cumprod() - 1.0
    ax.plot(cum.index, cum.values, color="darkred", lw=1.2)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative H-L")
    ax.set_xlabel(caption, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def eval_panel(
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    masks: dict,
    close: pd.DataFrame,
    amount: pd.DataFrame,
    mode: str,
) -> dict:
    """One neutralization mode: shift-1 groupTest + investability."""
    aligned = panel.reindex_like(ret)
    sig = align_signal(aligned, SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    _, pnl, to = Factor_Dev_Lib.groupTest(sig, r, n=N_GROUPS, fee=0, info="silent")
    stats = compute_group_stats(sig, r, pnl, to)
    excess = long_book_excess_performance(
        aligned,
        ret,
        top_frac=1.0 / N_GROUPS,
        signal_shift=SIGNAL_SHIFT,
        direction=stats["direction"],
    )
    # Keep full close history for IPO seasoning (do not truncate to ret window).
    min_list = 60 if len(ret) >= 120 else 0
    inv = evaluate_investability(
        aligned,
        ret,
        df_not_limit=masks["df_not_limit"].reindex_like(ret),
        df_not_st=masks["df_not_st"].reindex_like(ret),
        df_trade_status=masks["df_trade_status"].reindex_like(ret),
        close=close,
        amount=amount,
        round_trip_cost=DEFAULT_ROUND_TRIP_COST,
        signal_shift=SIGNAL_SHIFT,
        min_listing_days=min_list,
    )
    decile = decile_mean_returns(pnl, N_GROUPS)
    mono = monotonicity_spearman(decile)
    ic = daily_rank_ic_series(aligned, ret, signal_shift=SIGNAL_SHIFT)
    return {
        "mode": mode,
        "rank_ic": stats["rank_ic_mean"],
        "icir": stats["icir"],
        "hl_sharpe": stats["hl_sharpe"],
        "hl_annu_ret": stats["hl_annu_ret"],
        "hl_mdd": stats["hl_mdd"],
        "daily_turnover_hl": stats["hl_avg_turnover"],
        "implied_annu_fee": stats["implied_annu_fee"],
        "gross_sharpe_tradable": inv["gross_sharpe_tradable"],
        "net_sharpe_15bp": inv["net_sharpe_tradable"],
        "long_book_excess_sharpe": excess["excess_sharpe"],
        "long_book_excess_annu_ret": excess["excess_annu_ret"],
        "long_book_excess_mdd": excess["excess_max_drawdown"],
        "long_group": excess["long_group"],
        "selected_count_mean": excess["selected_count_mean"],
        "universe_count_mean": excess["universe_count_mean"],
        "annu_one_way_turnover": inv["annu_one_way_turnover"],
        "direction": stats["direction"],
        "monotonicity_spearman": mono,
        "stats_title": format_group_stats_title(stats),
        "decile_means": decile,
        "hml": pnl["H-L"] * stats["direction"],
        "rank_ic_daily": ic,
        "pnl": pnl,
        "to": to,
        "long_book_ret": excess["_long_ret"],
        "universe_ew_ret": excess["_universe_ew_ret"],
        "long_book_excess_ret": excess["_excess_ret"],
    }


def period_ic_table(panel: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ic_full = daily_rank_ic_series(panel.reindex_like(ret), ret, signal_shift=SIGNAL_SHIFT)
    # yearly
    years = sorted(set(ret.index.year))
    for y in years:
        mask = ret.index.year == y
        if mask.sum() < 40:
            continue
        sub = ic_full.loc[mask].dropna()
        if len(sub) < 20:
            continue
        rows.append(
            {
                "period": str(y),
                "kind": "year",
                "n_days": int(len(sub)),
                "rank_ic": float(sub.mean()),
                "icir": float(sub.mean() / sub.std() * np.sqrt(250)) if sub.std() > 0 else np.nan,
                "pos_ic_frac": float((sub > 0).mean()),
            }
        )
    for name, s, e in PERIOD_BLOCKS:
        mask = (ret.index >= s) & (ret.index <= e)
        sub = ic_full.loc[mask].dropna()
        if len(sub) < 40:
            continue
        rows.append(
            {
                "period": name,
                "kind": "block",
                "n_days": int(len(sub)),
                "rank_ic": float(sub.mean()),
                "icir": float(sub.mean() / sub.std() * np.sqrt(250)) if sub.std() > 0 else np.nan,
                "pos_ic_frac": float((sub > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    paths: dict,
    *,
    raw_eval: dict,
    neut_df: pd.DataFrame,
    stab_df: pd.DataFrame,
    sample_meta: dict,
) -> None:
    yearly = stab_df[stab_df["kind"] == "year"]
    n_pos = int((yearly["rank_ic"] > 0).sum()) if len(yearly) else 0
    n_y = int(len(yearly))
    lines = [
        "# TGD v1 — Stage 4 Validation Summary",
        "",
        f"**Factor:** `{FACTOR}`",
        f"**Signal shift:** `{SIGNAL_SHIFT}` (TGD20 on day T uses full-session Gu/Gd → predicts T+1)",
        f"**Portfolio:** {N_GROUPS}-group + H-L (project standard; not paper 5-group)",
        f"**Sample:** {sample_meta.get('start')} → {sample_meta.get('end')} "
        f"({sample_meta.get('n_days')}d confirmation focus noted below)",
        "",
        "## Pipeline (frozen)",
        "",
        "```",
        "minute → Gu/Gd → residual(εu,εd) → εd~εu → MA20 → TGD20",
        "```",
        "",
        "Formula layer is frozen — this report is research validation only.",
        "",
        "## A. Replication sanity (raw, confirmation)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| RankIC | {raw_eval['rank_ic']:.4f} |",
        f"| ICIR | {raw_eval['icir']:.2f} |",
        f"| Group{raw_eval['long_group']} excess Sharpe vs exact ALL EW | "
        f"{raw_eval['long_book_excess_sharpe']:.2f} |",
        f"| H-L Sharpe | {raw_eval['hl_sharpe']:.2f} |",
        f"| Decile monotonicity (Spearman) | {raw_eval['monotonicity_spearman']:.3f} |",
        f"| Direction | {raw_eval['direction']} |",
        "",
        "Artifacts: `portfolio/cumulative_long_short.png` (10-group + H-L), "
        "`portfolio/decile_return.png`, `portfolio/hml_curve.png`, `ic/rank_ic.csv`",
        "",
        "## B. Neutralization ladder",
        "",
        "| Mode | RankIC | ICIR | Long-book excess Sharpe | H-L Sharpe | Net Sharpe@15bp | Daily TO(H-L) |",
        "|------|--------|------|-------------------------|------------|-----------------|---------------|",
    ]
    for _, r in neut_df.iterrows():
        lines.append(
            f"| {r['mode']} | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{r['long_book_excess_sharpe']:.2f} | {r['hl_sharpe']:.2f} | "
            f"{r['net_sharpe_15bp']:.2f} | {r['daily_turnover_hl']:.2f} |"
        )
    lines += [
        "",
        "Artifact: `neutralization/neut_summary.csv`",
        "",
        "## C. Period stability",
        "",
        f"Yearly positive mean-RankIC: **{n_pos}/{n_y}**",
        "",
        "| Period | Kind | RankIC | ICIR | Pos IC frac | n |",
        "|--------|------|--------|------|-------------|---|",
    ]
    for _, r in stab_df.iterrows():
        lines.append(
            f"| {r['period']} | {r['kind']} | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{r['pos_ic_frac']:.2f} | {int(r['n_days'])} |"
        )
    lines += [
        "",
        "Artifact: `stability/yearly_ic.csv`",
        "",
        "## D. Cost / turnover",
        "",
        f"| Metric | Raw confirmation |",
        f"|--------|------------------|",
        f"| Gross Sharpe (tradable) | {raw_eval['gross_sharpe_tradable']:.2f} |",
        f"| Net Sharpe @15bp RT | {raw_eval['net_sharpe_15bp']:.2f} |",
        f"| Daily H-L turnover | {raw_eval['daily_turnover_hl']:.2f} |",
        f"| Implied AnnuFee(7.5%) | {raw_eval['implied_annu_fee']:.2%} |",
        f"| Annu one-way TO | {raw_eval['annu_one_way_turnover']:.2f} |",
        "",
        "Artifact: `cost/turnover_cost.csv`",
        "",
        "## Notes vs paper",
        "",
        "- Do **not** hard-match paper 5-group IR / RankICIR numbers.",
        "- Compare mechanism (decile ordering) + statistical strength + robustness + cost.",
        "",
        "## Status",
        "",
        "| Stage | Status |",
        "|-------|--------|",
        "| 0–3 formula / info layer | ✅ frozen |",
        "| 4 validation runner | ✅ this report |",
        "",
    ]
    (paths["root"] / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TGD Stage-4 validation")
    parser.add_argument("--sample-days", type=int, default=0, help="If >0, use last N days of cfg window")
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--preheat-days", type=int, default=60)
    args = parser.parse_args()

    paths = ensure_dirs(OUT)
    start, end = cfg.START_DAY, cfg.END_DAY
    if args.sample_days and args.sample_days > 0:
        # approximate: end-fixed, start moved — exact calendar trim after load
        start_hint = end - dt.timedelta(days=int(args.sample_days * 1.7) + args.preheat_days)
    else:
        start_hint = start

    log("=== TGD Stage-4 Validation ===")
    log(f"Config window: {start.date()} → {end.date()} | signal_shift={SIGNAL_SHIFT}")

    enriched, _session = load_eod_enriched_tables(
        start_hint - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS),
        end,
    )
    # drop unused DDB session handle from loader if any
    industry = load_citics_industry_panel(start_hint, end)
    float_mkt = enriched.float_mktcap

    log("Building TGD20 panel (cached monthly timing + residual + MA20)...")
    tgd_wide, _tgd_long = build_tgd20_wide_from_eod_l2(
        start_hint,
        end,
        open_=enriched.open,
        close=enriched.close,
        use_cache=True,
        refresh_cache=args.refresh_cache,
        preheat_calendar_days=args.preheat_days,
    )
    tgd_wide = tgd_wide.loc[start:end]
    if args.sample_days and args.sample_days > 0 and len(tgd_wide) > args.sample_days:
        tgd_wide = tgd_wide.iloc[-args.sample_days :]
    log(f"TGD20 panel: {tgd_wide.index[0].date()} → {tgd_wide.index[-1].date()} "
        f"({len(tgd_wide)}d × {tgd_wide.shape[1]} names)")

    ret_full = Factor_Dev_Lib.get_Ret_Matrix(
        tgd_wide.index[0].to_pydatetime(),
        tgd_wide.index[-1].to_pydatetime(),
        method="c2c",
    )
    ret_full = ret_full.reindex(index=tgd_wide.index, columns=tgd_wide.columns)

    # Confirmation slice (match Flow Density convention) when full sample
    if args.sample_days and args.sample_days > 0:
        ret_eval = ret_full
        panel_eval = tgd_wide
        log(f"Eval sample (sample-days): {ret_eval.index[0].date()} → {ret_eval.index[-1].date()}")
    else:
        _, ret_eval = split_discovery_confirmation(ret_full, args.discovery_days)
        if ret_eval.empty:
            ret_eval = ret_full
        panel_eval = tgd_wide.reindex(index=ret_eval.index, columns=ret_eval.columns)
        log(f"Confirmation: {ret_eval.index[0].date()} → {ret_eval.index[-1].date()} ({len(ret_eval)}d)")

    masks = {
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(
            ret_eval.index[0].to_pydatetime(), ret_eval.index[-1].to_pydatetime()
        ),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(
            ret_eval.index[0].to_pydatetime(), ret_eval.index[-1].to_pydatetime()
        ),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(
            ret_eval.index[0].to_pydatetime(), ret_eval.index[-1].to_pydatetime()
        ),
    }

    ladder = build_neut_ladder(
        panel_eval,
        industry.reindex_like(panel_eval),
        float_mkt.reindex_like(panel_eval),
    )

    # --- A/B/D: neutralization ladder on confirmation ---
    log("\n--- Neutralization ladder (10-group, shift-1) ---")
    evals = []
    for mode, panel in ladder.items():
        ev = eval_panel(
            panel,
            ret_eval,
            masks=masks,
            close=enriched.close,
            amount=enriched.amount,
            mode=mode,
        )
        evals.append(ev)
        log(
            f"  {mode}: RankIC={ev['rank_ic']:.4f} ICIR={ev['icir']:.2f} "
            f"HL={ev['hl_sharpe']:.2f} net={ev['net_sharpe_15bp']:.2f} "
            f"TO={ev['daily_turnover_hl']:.2f} mono={ev['monotonicity_spearman']:.3f}"
        )

    raw_eval = next(e for e in evals if e["mode"] == "raw")
    neut_df = pd.DataFrame(
        [
            {
                "mode": e["mode"],
                "rank_ic": e["rank_ic"],
                "icir": e["icir"],
                "hl_sharpe": e["hl_sharpe"],
                "hl_annu_ret": e["hl_annu_ret"],
                "hl_mdd": e["hl_mdd"],
                "daily_turnover_hl": e["daily_turnover_hl"],
                "implied_annu_fee": e["implied_annu_fee"],
                "gross_sharpe_tradable": e["gross_sharpe_tradable"],
                "net_sharpe_15bp": e["net_sharpe_15bp"],
                "long_book_excess_sharpe": e["long_book_excess_sharpe"],
                "long_book_excess_annu_ret": e["long_book_excess_annu_ret"],
                "long_book_excess_mdd": e["long_book_excess_mdd"],
                "long_group": e["long_group"],
                "selected_count_mean": e["selected_count_mean"],
                "universe_count_mean": e["universe_count_mean"],
                "annu_one_way_turnover": e["annu_one_way_turnover"],
                "direction": e["direction"],
                "monotonicity_spearman": e["monotonicity_spearman"],
            }
            for e in evals
        ]
    )
    neut_df.to_csv(paths["neutralization"] / "neut_summary.csv", index=False)

    # A artifacts
    ic_daily = raw_eval["rank_ic_daily"].rename("rank_ic").to_frame()
    ic_daily.to_csv(paths["ic"] / "rank_ic.csv")
    group_cum = raw_eval["pnl"].cumsum()
    group_cum.to_csv(paths["portfolio"] / "group_cum_pnl.csv")
    raw_eval["to"].to_csv(paths["portfolio"] / "group_turnover.csv")
    for ev in evals:
        pd.concat(
            [
                ev["long_book_ret"].rename("long_book_return"),
                ev["universe_ew_ret"].rename("test_universe_ew_return"),
                ev["long_book_excess_ret"].rename("long_book_excess_return"),
            ],
            axis=1,
        ).to_csv(paths["portfolio"] / f"long_book_excess_daily_{ev['mode']}.csv")
    save_cumulative_decile_figure(
        group_cum,
        factor_name=FACTOR,
        stats_title=raw_eval["stats_title"],
        out_path=paths["portfolio"] / "cumulative_long_short.png",
    )
    plot_decile_bars(
        raw_eval["decile_means"],
        paths["portfolio"] / "decile_return.png",
        f"{FACTOR} decile mean daily return (raw, shift={SIGNAL_SHIFT})",
        raw_eval["stats_title"],
    )
    plot_hml_curve(
        raw_eval["hml"],
        paths["portfolio"] / "hml_curve.png",
        f"{FACTOR} H-L cumulative (direction-adjusted)",
        raw_eval["stats_title"],
    )

    # C: period stability on full available panel (raw zscored)
    log("\n--- Period stability ---")
    stab_panel = cs_zscore(tgd_wide.reindex_like(ret_full))
    stab_df = period_ic_table(stab_panel, ret_full)
    stab_df.to_csv(paths["stability"] / "yearly_ic.csv", index=False)
    log(stab_df.to_string(index=False))

    # D: cost table
    cost_df = neut_df[
        [
            "mode",
            "gross_sharpe_tradable",
            "net_sharpe_15bp",
            "daily_turnover_hl",
            "implied_annu_fee",
            "annu_one_way_turnover",
        ]
    ].copy()
    cost_df.to_csv(paths["cost"] / "turnover_cost.csv", index=False)

    sample_meta = {
        "start": str(panel_eval.index[0].date()),
        "end": str(panel_eval.index[-1].date()),
        "n_days": int(len(panel_eval)),
        "signal_shift": SIGNAL_SHIFT,
        "n_groups": N_GROUPS,
        "sample_days_arg": args.sample_days,
    }
    write_summary(
        paths,
        raw_eval=raw_eval,
        neut_df=neut_df,
        stab_df=stab_df,
        sample_meta=sample_meta,
    )

    payload = {
        "factor": FACTOR,
        "signal_shift": SIGNAL_SHIFT,
        "sample": sample_meta,
        "neutralization": neut_df.to_dict(orient="records"),
        "stability": stab_df.to_dict(orient="records"),
        "note": "Do not hard-match paper 5-group IR; compare mechanism + robustness + cost.",
    }
    (paths["root"] / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    log(f"\nWrote {OUT}/summary.md")


if __name__ == "__main__":
    main()
