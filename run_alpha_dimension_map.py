#!/usr/bin/env python
"""Build Alpha Dimension Map v1: economic return drivers vs PCA variance dimensions."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import factor_config as cfg
import intraday_lib
import pandas as pd

import Factor_Dev_Lib
from alpha_dimension_map import (
    OHLCV_PRODUCTION_DIMENSIONS,
    build_dimension_map_v1,
    format_methodology_verdict,
)
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import FACTOR_REGISTRY, build_factor, build_factor_cache
from factor_formulas_cn_broker import CN_BROKER_REGISTRY, build_cn_broker_factor
from factor_formulas_eod_engine import EOD_ENGINE_REGISTRY, build_eod_engine_factor
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_REGISTRY,
    build_liquidity_norm_cache,
    build_liquidity_norm_factor,
)

OUT = cfg.RESEARCH_DIR


def log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def build_any_factor(name: str, pv_cache, norm_cache) -> pd.DataFrame:
    if name in LIQUIDITY_NORM_REGISTRY:
        return build_liquidity_norm_factor(name, norm_cache)
    if name in EOD_ENGINE_REGISTRY:
        return build_eod_engine_factor(name, pv_cache)
    if name in CN_BROKER_REGISTRY:
        return build_cn_broker_factor(name, pv_cache)
    if name in FACTOR_REGISTRY:
        return build_factor(name, pv_cache)
    raise KeyError(name)


def main():
    clusters_path = OUT / "alpha_latent_clusters.csv"
    residual_path = OUT / "alpha_cluster_residual_ic.csv"
    ranking_path = OUT / "robust_alpha_ranking.csv"
    intrinsic_path = OUT / "alpha_information_space_summary.csv"

    for p in (clusters_path, residual_path, ranking_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — run run_robust_alpha_score.py + run_alpha_information_space.py first")

    clusters = pd.read_csv(clusters_path)
    residual_ic = pd.read_csv(residual_path)
    ranking = pd.read_csv(ranking_path)
    intrinsic = pd.read_csv(intrinsic_path) if intrinsic_path.exists() else None

    log("Building production-representative panels for cross-dimension IC...")
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    enriched, session = load_eod_enriched_tables(preheat, end)
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

    rep_panels = {}
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        name = spec.representative
        try:
            rep_panels[name] = build_any_factor(name, pv_cache, norm_cache).loc[start:end]
            log(f"  OK {name}")
        except Exception as exc:
            log(f"  SKIP {name}: {exc}")

    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    dimension_map, variance_vs_return, cluster_map, ortho_table, expansion = build_dimension_map_v1(
        clusters_df=clusters,
        residual_ic_df=residual_ic,
        ranking_df=ranking,
        intrinsic_summary=intrinsic,
        rep_panels=rep_panels,
        ret_wide=ret,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    dimension_map.to_csv(OUT / "alpha_dimension_map_v1.csv", index=False)
    variance_vs_return.to_csv(OUT / "alpha_variance_vs_return_dims.csv", index=False)
    cluster_map.to_csv(OUT / "alpha_cluster_economic_assignment.csv", index=False)
    ortho_table.to_csv(OUT / "alpha_orthogonal_return_drivers.csv", index=False)
    expansion.to_csv(OUT / "alpha_expansion_roadmap.csv", index=False)

    verdict = format_methodology_verdict(variance_vs_return)
    pd.DataFrame([{"verdict": verdict}]).to_csv(OUT / "alpha_dimension_map_verdict.csv", index=False)

    log("\n=== Variance vs Return-Predictive Dimensions ===")
    log(variance_vs_return.to_string(index=False))
    log(f"\n{verdict}")

    log("\n=== Alpha Dimension Map v1 (OHLCV frozen production stack) ===")
    cols = [
        "dimension_id",
        "dimension_name",
        "representative_factor",
        "n_supporting",
        "n_correlation_clusters",
        "representative_ic_cross_dimension",
        "representative_production_score",
        "missing_information",
        "next_expansion_phase",
    ]
    log(dimension_map[cols].to_string(index=False))

    log("\n=== Orthogonal return drivers (greedy sequential IC) ===")
    if len(ortho_table):
        log(
            ortho_table[
                ["factor_name", "ic_raw", "ic_orthogonal", "accepted_as_return_driver", "orthogonal_rank"]
            ].to_string(index=False)
        )

    log("\n=== Expansion roadmap (next information sources) ===")
    log(expansion[["phase", "dimension_id", "name", "information_source", "target_factors"]].to_string(index=False))

    log(f"\nSaved -> {OUT}/alpha_dimension_map_v1.csv (+ roadmap, orthogonal drivers)")


if __name__ == "__main__":
    main()
