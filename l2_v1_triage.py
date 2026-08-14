"""L2 v1 research triage — level bricks (VOI/OIR/MPB) closed with findings.

Key finding (Jul 2026):
  Simple L2 daily-average level factors re-measure existing OHLCV price-path /
  liquidity dimensions — NOT a new return-predictive manifold.

  Residual IC sign-flip / amplification ≠ alpha; it indicates structural overlap
  with frozen stack D1–D5. L2 may still act as conditional alpha enhancer.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

L2_V1_FACTORS = ["cn_voi_20d", "cn_oir_20d", "cn_mpb_20d"]

L2_V1_RESEARCH_FINDING = (
    "CLOSED: VOI/OIR/MPB daily level bricks — no independent OHLCV dimension; "
    "alpha enhancer when combined with frozen stack D1–D5. "
    "See research/results/l2_v1_closed.md. Next: L2 v2 event-driven."
)

# Per-factor v1 labels (populated from validation; defaults before run)
L2_V1_DEFAULT_TRIAGE: Dict[str, dict] = {
    "cn_voi_20d": {
        "engine": "l2_microstructure_v1",
        "brick_type": "level",
        "dimension_verdict": "no_independent_dimension",
        "conditional_alpha": "possible_enhancer",
        "cluster_tag": "l2_ohlcv_mixed",
        "notes": "Active-flow imbalance ≈ price-path / liquidity pressure; not queue depth.",
    },
    "cn_oir_20d": {
        "engine": "l2_microstructure_v1",
        "brick_type": "level",
        "dimension_verdict": "no_independent_dimension",
        "conditional_alpha": "possible_enhancer",
        "cluster_tag": "l2_ohlcv_mixed",
        "notes": "Cancel-volume proxy ≠ order-book depth OIR; sign-flip after stack neutral.",
    },
    "cn_mpb_20d": {
        "engine": "l2_microstructure_v1",
        "brick_type": "level",
        "dimension_verdict": "no_independent_dimension",
        "conditional_alpha": "possible_enhancer",
        "cluster_tag": "l2_ohlcv_mixed",
        "notes": "Signed amount imbalance overlaps MPB/liquidity EOD proxies.",
    },
}


def build_l2_v1_triage(
    attribution_df: pd.DataFrame,
    cluster_tags_df: pd.DataFrame,
    enhancement_df: pd.DataFrame,
) -> pd.DataFrame:
    """Publish per-factor v1 triage table for research archive."""
    cluster_map = {}
    if len(cluster_tags_df):
        for _, row in cluster_tags_df.iterrows():
            for fname in str(row.get("l2_members", "")).split("|"):
                if fname:
                    cluster_map[fname] = row.get("cluster_tag", "")

    enh_map = {}
    if len(enhancement_df) and "factor_name" in enhancement_df.columns:
        enh_map = enhancement_df.set_index("factor_name").to_dict("index")

    rows: List[dict] = []
    for fname in L2_V1_FACTORS:
        base = dict(L2_V1_DEFAULT_TRIAGE.get(fname, {}))
        base["factor_name"] = fname
        base["v1_status"] = "CLOSED"
        base["role"] = "alpha_enhancer_not_source"
        if fname in attribution_df["factor_name"].values:
            a = attribution_df.loc[attribution_df["factor_name"] == fname].iloc[0]
            base.update(
                {
                    "ic_raw": a.get("ic_raw"),
                    "ic_after_ohlcv_stack": a.get("ic_after_ohlcv_stack"),
                    "strict_pass": bool(a.get("strict_pass", False)),
                    "conclusion": a.get("conclusion"),
                }
            )
        base["cluster_tag"] = cluster_map.get(fname, base.get("cluster_tag", ""))
        e = enh_map.get(fname, {})
        base["stack_ic_delta"] = e.get("stack_ic_delta")
        base["stack_sharpe_delta"] = e.get("stack_sharpe_delta")
        base["stack_enhancement_pass"] = bool(e.get("stack_enhancement_pass", False))
        rows.append(base)

    df = pd.DataFrame(rows)
    df["research_finding"] = L2_V1_RESEARCH_FINDING
    return df
