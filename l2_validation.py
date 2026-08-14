"""L2 validation — dual gates: dimension (strict IC) vs stack enhancement.

v1 finding: level bricks fail dimension gate but may pass enhancement gate.
v2 question: event-driven bricks — dimension OR enhancement on frozen D1–D5.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from cn_broker_validation import build_validation_verdict
from l2_v1_triage import L2_V1_RESEARCH_FINDING, build_l2_v1_triage


def classify_l2_clusters(
    cluster_summary_df: pd.DataFrame,
    l2_factor_names: List[str],
) -> pd.DataFrame:
    """Tag clusters: l2_only_new_cluster vs l2_ohlcv_mixed vs ohlcv_only."""
    l2_set = set(l2_factor_names)
    rows = []
    for _, crow in cluster_summary_df.iterrows():
        members = crow["members"].split("|") if isinstance(crow["members"], str) else []
        l2_members = [m for m in members if m in l2_set]
        old_members = [m for m in members if m not in l2_set]
        if l2_members and not old_members:
            tag = "l2_only_new_cluster"
        elif l2_members and old_members:
            tag = "l2_ohlcv_mixed"
        elif l2_members:
            tag = "l2_singleton"
        else:
            tag = "ohlcv_only"
        rows.append(
            {
                "cluster_id": crow["cluster_id"],
                "representative": crow["representative"],
                "n_members": crow["n_members"],
                "l2_members": "|".join(l2_members),
                "old_members": "|".join(old_members),
                "cluster_tag": tag,
                "dominant_mechanism_layer": crow.get("dominant_mechanism_layer"),
            }
        )
    return pd.DataFrame(rows)


def build_l2_validation_verdict(
    attribution_df: pd.DataFrame,
    cluster_tags_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
    enhancement_df: Optional[pd.DataFrame] = None,
    track: str = "l2_microstructure_v1",
) -> pd.DataFrame:
    """Dual-gate verdict: dimension (strict IC + new cluster) vs stack enhancement."""
    base = build_validation_verdict(attribution_df, cluster_tags_df, incremental_df)
    row = base.iloc[0].to_dict()

    n_strict = int(row.get("n_strict_independent_alpha", 0))
    n_l2_clusters = 0
    n_l2_mixed = 0
    if len(cluster_tags_df):
        n_l2_clusters = int((cluster_tags_df["cluster_tag"] == "l2_only_new_cluster").sum())
        n_l2_mixed = int((cluster_tags_df["cluster_tag"] == "l2_ohlcv_mixed").sum())

    n_enhancement_pass = 0
    if enhancement_df is not None and len(enhancement_df) and "stack_enhancement_pass" in enhancement_df.columns:
        n_enhancement_pass = int(enhancement_df["stack_enhancement_pass"].sum())

    dimension_gate = n_strict >= 1 and n_l2_clusters >= 1
    enhancement_gate = n_enhancement_pass >= 1

    row.update(
        {
            "track": track,
            "n_l2_only_new_clusters": n_l2_clusters,
            "n_l2_ohlcv_mixed_clusters": n_l2_mixed,
            "n_l2_singletons": int(
                (cluster_tags_df["cluster_tag"].isin(["l2_only_new_cluster", "l2_singleton"])).sum()
            )
            if len(cluster_tags_df)
            else 0,
            "n_stack_enhancement_pass": n_enhancement_pass,
            "dimension_gate_pass": dimension_gate,
            "enhancement_gate_pass": enhancement_gate,
            # Legacy alias
            "phase_success": dimension_gate,
            "dimension_success_criteria": "strict_pass>=1 AND l2_only_new_cluster>=1",
            "enhancement_success_criteria": "stack_ic_delta>=30bp OR stack_sharpe_delta>=0.10",
            "research_finding": L2_V1_RESEARCH_FINDING if track == "l2_microstructure_v1" else "",
        }
    )
    return pd.DataFrame([row])


def publish_l2_v1_archive(
    attribution_df: pd.DataFrame,
    cluster_tags_df: pd.DataFrame,
    enhancement_df: pd.DataFrame,
    out_dir,
) -> pd.DataFrame:
    triage = build_l2_v1_triage(attribution_df, cluster_tags_df, enhancement_df)
    triage.to_csv(out_dir / "l2_v1_triage.csv", index=False)
    return triage
