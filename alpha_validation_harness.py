"""Alpha Validation Harness v1 — unified QC for factor candidates.

Consolidates scattered validation from run_*_validation.py into one pipeline:
  IC decay → strict residual vs D1–D5 → clustering → stack enhancement → verdict

Per-factor harness labels:
  production   strict_pass + not ohlcv_redundant
  enhancer     stack IC/Sharpe delta pass (no strict)
  research_only partial / mixed cluster / weak incremental
  drop         ohlcv_redundant_proxy / sign_flip / insufficient
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_information_space import cluster_summary, hierarchical_cluster, intrinsic_dimension
from cn_broker_validation import build_validation_verdict
from factor_attribution import (
    OHLCV_FROZEN_REPS,
    build_attribution_row,
    incremental_bundle_test,
    rank_ic_by_horizon,
)
from l2_stack_enhancement import run_stack_enhancement_test


@dataclass
class HarnessConfig:
    track: str
    candidate_prefix: str = "candidate"
    distance_threshold: float = 0.35
    output_prefix: str = "harness"
    enable_enhancement_gate: bool = True


def classify_candidate_clusters(
    cluster_summary_df: pd.DataFrame,
    candidate_names: List[str],
    *,
    candidate_prefix: str = "candidate",
) -> pd.DataFrame:
    """Generic cluster tagger (CN / L2 / Fundamental share same logic)."""
    cand_set = set(candidate_names)
    rows = []
    for _, crow in cluster_summary_df.iterrows():
        members = crow["members"].split("|") if isinstance(crow["members"], str) else []
        cand_members = [m for m in members if m in cand_set]
        old_members = [m for m in members if m not in cand_set]
        if cand_members and not old_members:
            tag = f"{candidate_prefix}_only_new_cluster"
        elif cand_members and old_members:
            tag = f"{candidate_prefix}_ohlcv_mixed"
        elif cand_members:
            tag = f"{candidate_prefix}_singleton"
        else:
            tag = "ohlcv_only"
        rows.append(
            {
                "cluster_id": crow["cluster_id"],
                "representative": crow["representative"],
                "n_members": crow["n_members"],
                "candidate_members": "|".join(cand_members),
                "old_members": "|".join(old_members),
                "cluster_tag": tag,
                "dominant_mechanism_layer": crow.get("dominant_mechanism_layer"),
            }
        )
    return pd.DataFrame(rows)


def assign_harness_verdict(row: pd.Series, enhancement_pass: bool = False) -> str:
    """Map attribution + enhancement to harness label."""
    conclusion = row.get("conclusion", "")
    strict = bool(row.get("strict_pass", False))

    if strict:
        return "production"
    if enhancement_pass:
        return "enhancer"
    if conclusion in ("ohlcv_redundant_proxy", "size_proxy_remove", "sign_flip_after_neutral"):
        return "drop"
    if conclusion in ("partial_incremental_alpha", "weak_incremental", "marginal"):
        return "research_only"
    if conclusion == "insufficient_data":
        return "drop"
    if conclusion == "amplification_artifact":
        return "drop"
    return "research_only"


def run_attribution_stage(
    candidate_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    close: pd.DataFrame,
    exposure_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    factor_meta: Optional[Dict[str, dict]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = factor_meta or {}
    attr_rows = []
    decay_rows = []
    for fname, panel in candidate_panels.items():
        m = meta.get(fname, {})
        attr_rows.append(
            build_attribution_row(
                fname,
                panel,
                ret,
                exposure_panels,
                frozen_panels,
                cn_family=m.get("family", ""),
                hypothesis=m.get("hypothesis", ""),
            )
        )
        decay = rank_ic_by_horizon(panel, close)
        decay["factor_name"] = fname
        decay_rows.append(decay)
    return pd.DataFrame(attr_rows), pd.concat(decay_rows, ignore_index=True) if decay_rows else pd.DataFrame()


def run_cluster_stage(
    candidate_panels: Dict[str, pd.DataFrame],
    ohlcv_rep_panels: Dict[str, pd.DataFrame],
    candidate_names: List[str],
    *,
    config: HarnessConfig,
    extend_corr_fn: Optional[Callable[[pd.DataFrame, Dict[str, pd.DataFrame]], pd.DataFrame]] = None,
    existing_corr_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_panels = {**ohlcv_rep_panels, **candidate_panels}
    if extend_corr_fn is not None and existing_corr_path is not None and existing_corr_path.exists():
        existing = pd.read_csv(existing_corr_path, index_col=0)
        corr = extend_corr_fn(existing, combined_panels)
    else:
        from alpha_information_space import correlation_matrix

        corr = correlation_matrix(combined_panels)

    labels = hierarchical_cluster(corr, distance_threshold=config.distance_threshold)
    rank = pd.DataFrame({"factor_name": list(corr.index), "production_score": 0.0})
    clusters = cluster_summary(corr, labels, rank)
    tags = classify_candidate_clusters(clusters, candidate_names, candidate_prefix=config.candidate_prefix)
    combined = clusters.merge(tags, on="cluster_id", how="left")
    intrinsic = pd.DataFrame([intrinsic_dimension(corr)])
    return corr, combined, tags


def run_bundle_stage(
    candidate_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    attribution_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline = [frozen_panels[r] for r in OHLCV_FROZEN_REPS if r in frozen_panels]
    rows = []
    for fname, panel in candidate_panels.items():
        row = incremental_bundle_test(baseline, panel, ret)
        row["factor_name"] = fname
        if fname in attribution_df["factor_name"].values:
            sub = attribution_df.loc[attribution_df["factor_name"] == fname]
            row["ic_ohlcv_stack_residual"] = sub["ic_after_ohlcv_stack"].values[0]
            row["strict_pass"] = bool(sub["strict_pass"].values[0])
        rows.append(row)
    return pd.DataFrame(rows)


def run_enhancement_stage(
    candidate_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
) -> pd.DataFrame:
    baseline = [frozen_panels[r] for r in OHLCV_FROZEN_REPS if r in frozen_panels]
    return run_stack_enhancement_test(baseline, candidate_panels, ret)


def build_harness_factor_verdicts(
    attribution_df: pd.DataFrame,
    enhancement_df: Optional[pd.DataFrame] = None,
    cluster_tags_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    enh_map = {}
    if enhancement_df is not None and len(enhancement_df):
        for _, r in enhancement_df.iterrows():
            enh_map[r["factor_name"]] = bool(r.get("stack_enhancement_pass", False))

    rows = []
    for _, row in attribution_df.iterrows():
        fname = row["factor_name"]
        label = assign_harness_verdict(row, enhancement_pass=enh_map.get(fname, False))
        cluster_tag = ""
        if cluster_tags_df is not None and len(cluster_tags_df):
            # factor may appear in multiple clusters — take first match
            for _, ct in cluster_tags_df.iterrows():
                members = ct.get("candidate_members", "")
                if isinstance(members, str) and fname in members.split("|"):
                    cluster_tag = ct.get("cluster_tag", "")
                    break
        rows.append(
            {
                "factor_name": fname,
                "harness_verdict": label,
                "strict_pass": bool(row.get("strict_pass", False)),
                "conclusion": row.get("conclusion", ""),
                "ic_raw": row.get("ic_raw", np.nan),
                "ic_after_ohlcv_stack": row.get("ic_after_ohlcv_stack", np.nan),
                "stack_enhancement_pass": enh_map.get(fname, False),
                "cluster_tag": cluster_tag,
            }
        )
    return pd.DataFrame(rows)


def build_harness_summary_verdict(
    attribution_df: pd.DataFrame,
    cluster_tags_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
    enhancement_df: Optional[pd.DataFrame] = None,
    *,
    config: HarnessConfig,
) -> pd.DataFrame:
    base = build_validation_verdict(attribution_df, cluster_tags_df, incremental_df)
    row = base.iloc[0].to_dict()
    factor_verdicts = build_harness_factor_verdicts(attribution_df, enhancement_df, cluster_tags_df)

    n_production = int((factor_verdicts["harness_verdict"] == "production").sum())
    n_enhancer = int((factor_verdicts["harness_verdict"] == "enhancer").sum())
    n_drop = int((factor_verdicts["harness_verdict"] == "drop").sum())
    n_research = int((factor_verdicts["harness_verdict"] == "research_only").sum())

    n_enhancement = 0
    if enhancement_df is not None and len(enhancement_df) and "stack_enhancement_pass" in enhancement_df.columns:
        n_enhancement = int(enhancement_df["stack_enhancement_pass"].sum())

    only_tag = f"{config.candidate_prefix}_only_new_cluster"
    n_new_clusters = int((cluster_tags_df["cluster_tag"] == only_tag).sum()) if len(cluster_tags_df) else 0

    row.update(
        {
            "track": config.track,
            "harness_version": "v1",
            "n_harness_production": n_production,
            "n_harness_enhancer": n_enhancer,
            "n_harness_research_only": n_research,
            "n_harness_drop": n_drop,
            "n_stack_enhancement_pass": n_enhancement,
            "n_candidate_only_new_clusters": n_new_clusters,
            "recommended_production": "|".join(
                factor_verdicts.loc[factor_verdicts["harness_verdict"] == "production", "factor_name"]
            ),
            "recommended_enhancer": "|".join(
                factor_verdicts.loc[factor_verdicts["harness_verdict"] == "enhancer", "factor_name"]
            ),
        }
    )
    return pd.DataFrame([row]), factor_verdicts


def save_harness_outputs(
    out_dir: Path,
    *,
    config: HarnessConfig,
    attribution_df: pd.DataFrame,
    ic_decay_df: pd.DataFrame,
    corr: pd.DataFrame,
    combined_clusters: pd.DataFrame,
    cluster_tags: pd.DataFrame,
    incremental_df: pd.DataFrame,
    enhancement_df: Optional[pd.DataFrame],
    summary_verdict: pd.DataFrame,
    factor_verdicts: pd.DataFrame,
) -> None:
    prefix = config.output_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    attribution_df.to_csv(out_dir / f"{prefix}_attribution.csv", index=False)
    ic_decay_df.to_csv(out_dir / f"{prefix}_ic_decay.csv", index=False)
    corr.to_csv(out_dir / f"{prefix}_correlation.csv")
    combined_clusters.to_csv(out_dir / f"{prefix}_combined_clusters.csv", index=False)
    cluster_tags.to_csv(out_dir / f"{prefix}_cluster_tags.csv", index=False)
    incremental_df.to_csv(out_dir / f"{prefix}_incremental_bundle.csv", index=False)
    if enhancement_df is not None and len(enhancement_df):
        enhancement_df.to_csv(out_dir / f"{prefix}_stack_enhancement.csv", index=False)
    factor_verdicts.to_csv(out_dir / f"{prefix}_factor_verdicts.csv", index=False)
    summary_verdict.to_csv(out_dir / f"{prefix}_verdict.csv", index=False)
