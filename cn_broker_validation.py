"""CN Broker v1 validation — four analyses before L2 expansion.

Stage 1: CN Broker v1 batch
Stage 2: Robust ranking
Stage 3: Information space comparison (Old OHLCV vs CN Broker)
Stage 4: Retain new alpha dimensions
Stage 5: L2 (only if ≥2-3 new clusters + stable residual IC)

Analyses:
  1. Universe stability + IC decay curve (1/5/10/20d)
  2. Residual vs OHLCV frozen dimensions + exposure attribution
  3. Combined clustering (old OHLCV top + CN broker)
  4. Incremental alpha bundle test (OHLCV stack + CN factor)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from alpha_information_space import (
    cluster_summary,
    correlation_matrix,
    hierarchical_cluster,
    intrinsic_dimension,
)
from factor_attribution import (
    OHLCV_FROZEN_REPS,
    build_attribution_row,
    incremental_bundle_test,
    rank_ic_by_horizon,
    universe_ic_table,
)
from factor_taxonomy_cn import CN_FACTOR_TAXONOMY, EOD_CN_BROKER_ALL_LIST

# Re-export for run script
__all__ = [
    "run_full_cn_broker_validation",
    "classify_new_clusters",
    "build_validation_verdict",
]


def classify_new_clusters(
    cluster_summary_df: pd.DataFrame,
    cn_factor_names: List[str],
) -> pd.DataFrame:
    """Tag clusters that are CN-only or mixed — signal of new alpha space."""
    cn_set = set(cn_factor_names)
    rows = []
    for _, crow in cluster_summary_df.iterrows():
        members = crow["members"].split("|") if isinstance(crow["members"], str) else []
        cn_members = [m for m in members if m in cn_set]
        old_members = [m for m in members if m not in cn_set]
        if cn_members and not old_members:
            tag = "cn_only_new_cluster"
        elif cn_members and old_members:
            tag = "cn_ohlcv_mixed"
        elif cn_members:
            tag = "cn_singleton"
        else:
            tag = "ohlcv_only"
        rows.append(
            {
                "cluster_id": crow["cluster_id"],
                "representative": crow["representative"],
                "n_members": crow["n_members"],
                "cn_members": "|".join(cn_members),
                "old_members": "|".join(old_members),
                "cluster_tag": tag,
                "dominant_mechanism_layer": crow.get("dominant_mechanism_layer"),
            }
        )
    return pd.DataFrame(rows)


def build_validation_verdict(
    attribution_df: pd.DataFrame,
    cluster_tags_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
) -> pd.DataFrame:
    """Stage gate using strict HF triage (strict_pass column when present)."""
    if "strict_pass" in attribution_df.columns:
        n_independent = int(attribution_df["strict_pass"].sum())
    else:
        n_independent = int((attribution_df["conclusion"] == "independent_incremental_alpha").sum())
    n_weak = int((attribution_df["conclusion"] == "partial_incremental_alpha").sum())
    n_sign_flip = int((attribution_df["conclusion"] == "sign_flip_after_neutral").sum()) if "conclusion" in attribution_df.columns else 0
    n_cn_only_clusters = int((cluster_tags_df["cluster_tag"] == "cn_only_new_cluster").sum()) if len(cluster_tags_df) else 0
    n_new_singletons = int((cluster_tags_df["cluster_tag"].isin(["cn_only_new_cluster", "cn_singleton"])).sum()) if len(cluster_tags_df) else 0
    n_bundle_positive = int(incremental_df["incremental_bundle_value"].sum()) if len(incremental_df) else 0

    retain_col = attribution_df["strict_pass"] if "strict_pass" in attribution_df.columns else (
        attribution_df["conclusion"] == "independent_incremental_alpha"
    )
    retain = attribution_df.loc[retain_col, "factor_name"].tolist()

    return pd.DataFrame(
        [
            {
                "n_factors_tested": len(attribution_df),
                "n_strict_independent_alpha": n_independent,
                "n_partial_incremental_alpha": n_weak,
                "n_sign_flip_after_neutral": n_sign_flip,
                "n_ohlcv_redundant_proxy": int(
                    (attribution_df["conclusion"] == "ohlcv_redundant_proxy").sum()
                ),
                "n_new_only_clusters": n_cn_only_clusters,
                "n_new_singletons": n_new_singletons,
                "n_bundle_sharpe_improvement": n_bundle_positive,
                "recommended_retain": "|".join(retain),
                "recommended_drop": "|".join(
                    attribution_df.loc[~retain_col, "factor_name"].tolist()
                ),
            }
        ]
    )


def run_full_cn_broker_validation(
    cn_panels: Dict[str, pd.DataFrame],
    ohlcv_top_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    exposure_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    close: pd.DataFrame,
    universe_masks: Optional[Dict[str, pd.DataFrame]] = None,
    ranking_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Run all four analyses; return dict of output DataFrames."""

    # --- Analysis 1: IC decay + universe stability ---
    decay_rows = []
    uni_rows = []
    uni_stab_rows = []
    for fname, panel in cn_panels.items():
        meta = CN_FACTOR_TAXONOMY.get(fname, {})
        decay = rank_ic_by_horizon(panel, close)
        decay["factor_name"] = fname
        decay["cn_family"] = meta.get("cn_family", "")
        decay_rows.append(decay)

        if universe_masks:
            uni = universe_ic_table(panel, ret, universe_masks)
            for _, urow in uni.iterrows():
                uni_rows.append(
                    {
                        "factor_name": fname,
                        "universe": urow["universe"],
                        "rank_ic": urow["rank_ic"],
                        "abs_rank_ic": urow["abs_rank_ic"],
                    }
                )
            uni_stab_rows.append(
                {
                    "factor_name": fname,
                    "universe_stability": uni.attrs.get("universe_stability"),
                    "sign_consistency": uni.attrs.get("sign_consistency"),
                }
            )

    ic_decay_df = pd.concat(decay_rows, ignore_index=True) if decay_rows else pd.DataFrame()
    universe_stability_df = pd.DataFrame(uni_rows)
    universe_stability_summary_df = pd.DataFrame(uni_stab_rows)

    # Best horizon per factor
    if len(ic_decay_df):
        best_h = (
            ic_decay_df.groupby("factor_name")
            .apply(lambda g: g.loc[g["abs_rank_ic"].idxmax(), "horizon_days"])
            .reset_index(name="best_horizon_days")
        )
    else:
        best_h = pd.DataFrame()

    # --- Analysis 2: Factor attribution / residual vs OHLCV ---
    attr_rows = []
    for i, (fname, panel) in enumerate(cn_panels.items(), 1):
        meta = CN_FACTOR_TAXONOMY.get(fname, {})
        row = build_attribution_row(
            factor_name=fname,
            factor=panel,
            ret=ret,
            exposure_panels=exposure_panels,
            frozen_panels=frozen_panels,
            cn_family=meta.get("cn_family", ""),
            hypothesis=meta.get("hypothesis", ""),
        )
        attr_rows.append(row)
        if i % 3 == 0:
            print(f"  attribution {i}/{len(cn_panels)}", flush=True)
    attribution_df = pd.DataFrame(attr_rows)
    if len(attribution_df) and len(best_h):
        attribution_df = attribution_df.merge(best_h, on="factor_name", how="left")
    if len(universe_stability_summary_df):
        attribution_df = attribution_df.merge(
            universe_stability_summary_df, on="factor_name", how="left"
        )

    # --- Analysis 3: Combined clustering ---
    combined = {**ohlcv_top_panels, **cn_panels}
    corr = correlation_matrix(combined)
    intrinsic = intrinsic_dimension(corr)
    labels = hierarchical_cluster(corr, distance_threshold=0.35)
    if ranking_df is not None and len(ranking_df):
        rank_for_cluster = ranking_df.copy()
        rank_for_cluster["production_score"] = pd.to_numeric(
            rank_for_cluster.get("production_score", 0), errors="coerce"
        ).fillna(0)
    else:
        rank_for_cluster = pd.DataFrame(
            {"factor_name": list(combined.keys()), "production_score": 0.0}
        )
    clusters = cluster_summary(corr, labels, rank_for_cluster)
    cluster_tags_df = classify_new_clusters(clusters, list(cn_panels.keys()))

    combined_cluster_df = clusters.merge(cluster_tags_df, on="cluster_id", how="left")

    # --- Analysis 4: Incremental bundle test ---
    baseline_panels = [frozen_panels[r] for r in OHLCV_FROZEN_REPS if r in frozen_panels]
    incr_rows = []
    for fname, panel in cn_panels.items():
        row = incremental_bundle_test(baseline_panels, panel, ret)
        row["factor_name"] = fname
        row["ic_ohlcv_stack_residual"] = attribution_df.loc[
            attribution_df["factor_name"] == fname, "ic_after_ohlcv_stack"
        ].values[0] if fname in attribution_df["factor_name"].values else np.nan
        incr_rows.append(row)
    incremental_df = pd.DataFrame(incr_rows)

    variance_summary = pd.DataFrame(
        [
            {
                "combined_factors": len(combined),
                "ohlcv_top_factors": len(ohlcv_top_panels),
                "cn_broker_factors": len(cn_panels),
                "pca_variance_dims_90pct": intrinsic.get("intrinsic_dim_90"),
                "correlation_clusters": len(clusters),
                "cn_only_new_clusters": int((cluster_tags_df["cluster_tag"] == "cn_only_new_cluster").sum()),
                "cn_singletons": int((cluster_tags_df["cluster_tag"] == "cn_singleton").sum()),
            }
        ]
    )

    verdict_df = build_validation_verdict(attribution_df, cluster_tags_df, incremental_df)

    return {
        "ic_decay": ic_decay_df,
        "universe_stability": universe_stability_df,
        "universe_stability_summary": universe_stability_summary_df,
        "attribution": attribution_df,
        "combined_clusters": combined_cluster_df,
        "cluster_tags": cluster_tags_df,
        "incremental_bundle": incremental_df,
        "variance_summary": variance_summary,
        "verdict": verdict_df,
        "correlation": corr,
    }
