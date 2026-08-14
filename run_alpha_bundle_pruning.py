#!/usr/bin/env python
"""Correlation pruning: Priority A anchors vs liquidity pool → non-redundant alpha bundle."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List

import factor_config as cfg
import intraday_lib
import pandas as pd

from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_liquidity_norm import (
    build_liquidity_norm_cache,
    build_liquidity_norm_factor,
)
from factor_selection import compute_alpha_scores, prune_redundant_factors
from liquidity_normalization import factor_correlation_matrix

# Liquidity-family pool + Priority A anchors
LIQUIDITY_POOL = [
    "amount_stability_20d",
    "liquidity_stability_20d",
    "liquidity_shock_20d",
    "amihud_shock_reversal_5d",
    "volume_stability_20d",
    "liquidity_amount_residual_20d",
    "turnover_amount_residual_20d",
]

ANCHOR_FACTORS = [
    "winner_sentiment_reversal_5d",
    "max_daily_return_20d",
]

CANDIDATE_FACTORS = ANCHOR_FACTORS + LIQUIDITY_POOL

LIQUIDITY_NORM_FACTORS = {
    "amount_stability_20d",
    "volume_stability_20d",
    "liquidity_amount_residual_20d",
    "turnover_amount_residual_20d",
}

EOD_ENGINE_FACTORS = {
    "liquidity_stability_20d",
    "liquidity_shock_20d",
    "amihud_shock_reversal_5d",
    "winner_sentiment_reversal_5d",
    "max_daily_return_20d",
}

BATCH_SOURCES = [
    ("result/eod_engine", "core"),
    ("result/eod_liquidity_norm", "core"),
    ("result/eod_engine_priority_a", "priority_a"),
    ("result/eod_pv", "new_eod"),
]


def load_merged_summaries() -> pd.DataFrame:
    frames = []
    for root, tag in BATCH_SOURCES:
        path = Path(root) / f"batch_summary_{tag}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No batch summaries found for liquidity pool")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["factor_name", "universe"], keep="last")
    return merged


def build_factor_panel(
    name: str,
    pv_cache,
    engine_cache,
    norm_cache,
    start_day: dt.datetime,
    end_day: dt.datetime,
) -> pd.DataFrame:
    if name in LIQUIDITY_NORM_FACTORS:
        wide = build_liquidity_norm_factor(name, norm_cache)
    elif name in EOD_ENGINE_FACTORS:
        wide = build_eod_engine_factor(name, engine_cache)
    else:
        wide = build_factor(name, pv_cache)
    return wide.loc[start_day:end_day]


def build_all_panels(start_day, end_day, start_preheat) -> Dict[str, pd.DataFrame]:
    enriched, session = load_eod_enriched_tables(start_preheat, end_day)
    session.run(intraday_lib.ddb_functions)

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )

    panels = {}
    for name in CANDIDATE_FACTORS:
        panels[name] = build_factor_panel(
            name, pv_cache, pv_cache, norm_cache, start_day, end_day
        )
    return panels


def main():
    parser = argparse.ArgumentParser(description="Liquidity pool correlation pruning")
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    args = parser.parse_args()

    start_day = cfg.START_DAY
    end_day = cfg.END_DAY
    start_preheat = start_day - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    print("Building factor panels (EOD + enriched mktcap)...")
    panels = build_all_panels(start_day, end_day, start_preheat)

    corr = factor_correlation_matrix(panels)
    out_dir = cfg.RESEARCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_path = out_dir / "alpha_bundle_correlation.csv"
    corr.to_csv(corr_path)

    summary_all = load_merged_summaries()
    summary_all = summary_all[summary_all["factor_name"].isin(CANDIDATE_FACTORS)]
    scored = compute_alpha_scores(summary_all)
    ranked = scored.sort_values("composite_score", ascending=False)["factor_name"].tolist()
    for a in ANCHOR_FACTORS:
        if a not in ranked:
            ranked.insert(0, a)

    bundle = prune_redundant_factors(panels, ranked, corr_threshold=args.corr_threshold)

    # Pairwise corr vs anchors
    anchor_corr = corr.loc[ANCHOR_FACTORS, CANDIDATE_FACTORS]

    bundle_rows = []
    for name in bundle:
        row = scored[scored["factor_name"] == name]
        meta = {
            "factor_name": name,
            "in_bundle": True,
            "abs_icir_ALL": row["abs_icir_mean"].iloc[0] if len(row) else None,
            "hl_sharpe_ALL": row["hl_sharpe_mean"].iloc[0] if len(row) else None,
            "family": row["family"].iloc[0] if len(row) else "unknown",
        }
        if name not in ANCHOR_FACTORS:
            meta["corr_winner_sentiment"] = corr.loc["winner_sentiment_reversal_5d", name]
            meta["corr_max_daily_return"] = corr.loc["max_daily_return_20d", name]
        bundle_rows.append(meta)

    dropped = [n for n in CANDIDATE_FACTORS if n not in bundle]
    bundle_df = pd.DataFrame(bundle_rows)
    bundle_path = out_dir / "alpha_bundle_non_redundant.csv"
    bundle_df.to_csv(bundle_path, index=False)

    print("\n=== Correlation matrix (stacked obs) ===")
    print(corr.round(3).to_string())
    print(f"\nSaved -> {corr_path}")

    print("\n=== Anchor correlations ===")
    print(anchor_corr.round(3).to_string())

    print(f"\n=== Rank order (by composite score, ALL) ===")
    for i, name in enumerate(ranked, 1):
        mark = "✓" if name in bundle else "✗"
        icir = scored.loc[scored["factor_name"] == name, "abs_icir_mean"]
        icir_v = f"{icir.iloc[0]:.2f}" if len(icir) else "n/a"
        print(f"  {mark} {i:2d}. {name:35s} abs_icir={icir_v}")

    print(f"\n=== Non-redundant bundle (|corr| < {args.corr_threshold}) ===")
    for name in bundle:
        print(f"  + {name}")
    print("\n=== Dropped (redundant) ===")
    for name in dropped:
        reasons = []
        for k in bundle:
            c = corr.loc[name, k]
            if abs(c) >= args.corr_threshold:
                reasons.append(f"{k} ({c:+.3f})")
        print(f"  - {name}: redundant with {', '.join(reasons) or 'lower score'}")

    print(f"\nSaved bundle -> {bundle_path}")


if __name__ == "__main__":
    main()
