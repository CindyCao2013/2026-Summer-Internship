"""Feature-selection jury for AI v1.

Statistical + sparse linear + dual tree boosting + permutation + stability
+ residual alpha. Tree gain alone is never sufficient to KEEP a factor.
Weak RankIC + strong nonlinear evidence is REVIEW, never automatic KEEP.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.horizon import consensus_selection


# Expanded jury. TREE (FS-2 RF gain) is optional diagnostic, not a core vote.
EVIDENCE_METHODS = (
    "F_REGRESSION",
    "RANK_IC",
    "MUTUAL_INFO",
    "L1",
    "ELASTICNET",
    "LIGHTGBM",
    "XGBOOST",
    "PERMUTATION",
    "STABILITY",
    "INCREMENTAL_ALPHA",
)

EVIDENCE_GROUPS: Mapping[str, Sequence[str]] = {
    "statistical": ("F_REGRESSION", "RANK_IC", "MUTUAL_INFO"),
    "sparse_linear": ("L1", "ELASTICNET"),
    "nonlinear_trees": ("LIGHTGBM", "XGBOOST"),
    "model_agnostic": ("PERMUTATION",),
    "stability": ("STABILITY",),
    "incremental": ("INCREMENTAL_ALPHA",),
}

LINEAR_METHODS = ("F_REGRESSION", "RANK_IC", "L1", "ELASTICNET")
NONLINEAR_CORE = ("MUTUAL_INFO", "LIGHTGBM", "XGBOOST", "PERMUTATION", "INCREMENTAL_ALPHA")
CONFIRMING_METHODS = (
    "F_REGRESSION",
    "RANK_IC",
    "MUTUAL_INFO",
    "L1",
    "ELASTICNET",
    "PERMUTATION",
    "STABILITY",
    "INCREMENTAL_ALPHA",
)

JURY_STATES = ("DROP", "REVIEW", "KEEP")


def nonlinear_review_override(row: Mapping[str, object]) -> bool:
    """Weak linear + MI + permutation + residual + at least one booster.

    This is REVIEW_NONLINEAR, not automatic KEEP. Tree-gain-only is never KEEP.
    Once both boosters exist, LGBM+XGB agreement may strengthen the review vote
    but still does not auto-promote to KEEP.
    """
    linear_on = sum(float(row.get(m, 0) or 0) >= 0.5 for m in LINEAR_METHODS)
    mi = float(row.get("MUTUAL_INFO", 0) or 0) >= 0.5
    permutation = float(row.get("PERMUTATION", 0) or 0) >= 0.5
    residual = float(row.get("INCREMENTAL_ALPHA", 0) or 0) >= 0.5
    trees_on = sum(float(row.get(m, 0) or 0) >= 0.5 for m in ("LIGHTGBM", "XGBOOST"))
    return linear_on <= 1 and mi and permutation and residual and trees_on >= 1


def nonlinear_keep_override(row: Mapping[str, object]) -> bool:
    """Deprecated: nonlinear evidence never auto-KEEPs. Always False."""
    return False


def tree_gain_without_confirmation(row: Mapping[str, object]) -> bool:
    """Tree votes only — no statistical / linear / permutation / residual confirmation.

    High LightGBM/XGBoost gain with no other jury vote is treated as a
    correlated substitute, not as proof of factor validity.
    """
    trees_on = sum(float(row.get(m, 0) or 0) >= 0.5 for m in ("LIGHTGBM", "XGBOOST"))
    confirm_on = sum(float(row.get(m, 0) or 0) >= 0.5 for m in CONFIRMING_METHODS)
    return trees_on >= 1 and confirm_on == 0


def assign_jury_state(
    selected: bool,
    *,
    nonlinear_review: bool,
    gain_only: bool,
) -> str:
    """DROP / REVIEW / KEEP. Tree gain alone is DROP. Nonlinear override is REVIEW."""
    if gain_only:
        return "DROP"
    if nonlinear_review:
        return "REVIEW"
    if selected:
        return "KEEP"
    return "DROP"


def apply_jury_rules(
    evidence: pd.DataFrame,
    *,
    min_methods: int = 2,
) -> pd.DataFrame:
    out = consensus_selection(evidence, min_methods=min_methods, methods=EVIDENCE_METHODS)
    review_nl = []
    gain_only = []
    states = []
    keep_nl_deprecated = []
    for _, row in out.iterrows():
        nl = bool(nonlinear_review_override(row))
        gain = bool(tree_gain_without_confirmation(row))
        review_nl.append(nl)
        gain_only.append(gain)
        keep_nl_deprecated.append(False)
        # Gain-only never KEEP. Nonlinear override never upgrades to KEEP.
        selected = bool(row["selected"]) and (not gain)
        states.append(assign_jury_state(selected, nonlinear_review=nl, gain_only=gain))
    out["nonlinear_review"] = review_nl
    out["nonlinear_keep_override"] = keep_nl_deprecated
    out["tree_gain_without_confirmation"] = gain_only
    out["jury_state"] = states
    out["selected"] = [s == "KEEP" for s in states]
    return out


def importance_family_share(
    importance: pd.DataFrame,
    *,
    value_col: str = "gain",
    family_col: str = "family",
) -> pd.DataFrame:
    """Share of tree importance by factor family (and optional time_scale)."""
    d = importance.copy()
    val = d[value_col].astype(float).clip(lower=0)
    total = float(val.sum())
    if total <= 0:
        d["share"] = 0.0
        return d.groupby(family_col, dropna=False).size().reset_index(name="n")
    grouped = (
        pd.DataFrame({family_col: d[family_col], "value": val})
        .groupby(family_col, dropna=False)["value"]
        .sum()
        .reset_index()
    )
    grouped["share"] = grouped["value"] / total
    return grouped.sort_values("share", ascending=False).reset_index(drop=True)


def top_k_overlap(a: Sequence[str], b: Sequence[str], k: int = 20) -> float:
    sa = set(list(a)[:k])
    sb = set(list(b)[:k])
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / len(sa | sb))
