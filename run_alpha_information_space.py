#!/usr/bin/env python
"""Alpha information space analysis: clustering + residual IC → latent dimensions."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_information_space import (
    cluster_summary,
    correlation_matrix,
    dedupe_ranking,
    hierarchical_cluster,
    intrinsic_dimension,
    latent_dimension_verdict,
    residual_ic_table,
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
RANKING_PATH = OUT / "robust_alpha_ranking.csv"


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


def select_factor_universe(ranking: pd.DataFrame, min_production: float = 0.0, top_k: int = 0) -> list:
    r = dedupe_ranking(ranking)
    if min_production > 0:
        r = r[r["production_score"] >= min_production]
    if top_k > 0:
        r = r.head(top_k)
    return r["factor_name"].tolist()


def main():
    import sys

    def log(msg: str) -> None:
        print(msg, flush=True)
        sys.stdout.flush()

    if not RANKING_PATH.exists():
        raise FileNotFoundError(f"Run run_robust_alpha_score.py first -> {RANKING_PATH}")

    ranking = pd.read_csv(RANKING_PATH)
    names = select_factor_universe(ranking, min_production=0.005, top_k=35)
    log(f"Building {len(names)} unique factor panels...")

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

    panels = {}
    skipped = []
    for i, name in enumerate(names, 1):
        try:
            panels[name] = build_any_factor(name, pv_cache, norm_cache).loc[start:end]
            if i % 5 == 0:
                log(f"  built {i}/{len(names)}")
        except Exception as exc:
            skipped.append((name, str(exc)))

    log(f"Built {len(panels)} panels, skipped {len(skipped)}")
    if skipped[:5]:
        for n, e in skipped[:5]:
            print(f"  skip {n}: {e}")

    log("Computing correlation matrix...")
    corr = correlation_matrix(panels)
    intrinsic = intrinsic_dimension(corr)
    labels = hierarchical_cluster(corr, distance_threshold=0.35)
    clusters = cluster_summary(corr, labels, dedupe_ranking(ranking))

    log("Computing forward returns + residual IC...")
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    resid_ic = residual_ic_table(panels, clusters, ret)
    n_incremental = int(resid_ic["has_incremental_alpha"].sum()) if len(resid_ic) else 0

    OUT.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT / "alpha_factor_correlation.csv")
    clusters.to_csv(OUT / "alpha_latent_clusters.csv", index=False)
    resid_ic.to_csv(OUT / "alpha_cluster_residual_ic.csv", index=False)

    # Assign each factor its cluster + incremental flag
    factor_map = resid_ic[["factor_name", "cluster_id", "representative", "ic_raw", "ic_residual", "has_incremental_alpha"]]
    factor_map = factor_map.merge(
        dedupe_ranking(ranking)[["factor_name", "production_score", "universe_stability", "mean_abs_ic"]],
        on="factor_name",
        how="left",
    )
    factor_map.to_csv(OUT / "alpha_information_space_map.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                **intrinsic,
                "n_correlation_clusters": len(clusters),
                "n_incremental_factors": n_incremental,
                "verdict": latent_dimension_verdict(intrinsic, len(clusters), n_incremental),
            }
        ]
    )
    summary.to_csv(OUT / "alpha_information_space_summary.csv", index=False)

    log("\n=== Intrinsic dimension (PCA on correlation) ===")
    log(f"  Factors: {intrinsic['n_factors']}")
    log(f"  Dims for 90% variance: {intrinsic['intrinsic_dim_90']}")
    log(f"  Correlation clusters (dist<0.35): {len(clusters)}")
    log(f"  Factors with incremental residual IC: {n_incremental}")
    log(f"\n{summary['verdict'].iloc[0]}")

    log("\n=== Latent clusters (representative per cluster) ===")
    log(
        clusters[
            ["cluster_id", "n_members", "representative", "dominant_mechanism_layer", "mean_intra_cluster_abs_corr"]
        ].to_string(index=False)
    )

    log("\n=== Incremental alpha (residual IC vs cluster rep) ===")
    inc = resid_ic[resid_ic["has_incremental_alpha"]].sort_values("ic_residual", key=abs, ascending=False)
    if len(inc):
        log(inc[["cluster_id", "factor_name", "representative", "ic_raw", "ic_residual"]].to_string(index=False))
    else:
        log("  None — cluster members are largely redundant measurements of the same latent driver.")

    log("\n=== Recommended production dimensions (cluster reps) ===")
    reps = clusters.merge(
        dedupe_ranking(ranking)[["factor_name", "production_score", "universe_stability", "mean_abs_ic"]],
        left_on="representative",
        right_on="factor_name",
        how="left",
    ).sort_values("production_score", ascending=False)
    log(
        reps[
            ["cluster_id", "representative", "dominant_mechanism_layer", "n_members", "production_score", "universe_stability", "mean_abs_ic"]
        ].head(10).to_string(index=False)
    )

    log(f"\nSaved -> {OUT}/alpha_*.csv")
    log("\nNext step: python run_alpha_dimension_map.py")
    log("  (maps clusters → economic return drivers; separates variance vs predictive dims)")


if __name__ == "__main__":
    main()
