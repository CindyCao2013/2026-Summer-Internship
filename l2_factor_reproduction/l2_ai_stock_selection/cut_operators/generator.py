"""Economically constrained cut-candidate generator.

Never builds the Cartesian product of primitives × cuts × states ×
transforms × interactions. Recipes are explicit. Budget is enforced.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CONDITIONAL_WHITELIST,
    MAX_DESCENDANTS_PER_PARENT,
    MAX_WORKERS,
    MODES,
    P0_PRIMITIVE_WHITELIST,
    PREFERRED_DESCENDANTS_PER_PARENT,
    PRIMITIVE_CLASS_AGGREGATORS,
    TC1_RECIPES,
    UNCONTROLLED_CARTESIAN_CAP,
    primitive_spec,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    append_rows,
    candidate_name,
    duplicate_names,
    empty_registry,
)

BUDGET_COLUMNS = (
    "parent",
    "mode",
    "proposed_count",
    "accepted_count",
    "rejected_count",
    "rejection_reason",
)


class CartesianSearchError(ValueError):
    """Raised when a caller requests an uncontrolled operator product."""


def assert_not_cartesian(
    n_primitives: int,
    n_cuts: int,
    n_states: int,
    n_transforms: int,
    n_interactions: int,
) -> None:
    product = (
        int(n_primitives)
        * int(n_cuts)
        * int(n_states)
        * int(n_transforms)
        * int(n_interactions)
    )
    if product > UNCONTROLLED_CARTESIAN_CAP:
        raise CartesianSearchError(
            "refusing uncontrolled Cartesian search of size {} "
            "({}x{}x{}x{}x{}); use explicit recipes".format(
                product, n_primitives, n_cuts, n_states, n_transforms, n_interactions
            )
        )


def assert_max_workers(n_workers: int) -> None:
    if int(n_workers) > MAX_WORKERS:
        raise ValueError("max workers is {}; got {}".format(MAX_WORKERS, n_workers))


def _whitelist_index() -> Dict[str, Dict[str, object]]:
    return {str(r["base_primitive"]): dict(r) for r in P0_PRIMITIVE_WHITELIST}


def _conditional_ok(target: str, condition: str) -> bool:
    if not condition:
        return True
    cond = str(condition).strip().lower()
    for row in CONDITIONAL_WHITELIST:
        if row["target"] == target and row["condition"] == cond:
            return True
    return False


def _reject_reason(spec: Mapping[str, object], seen: set) -> Optional[str]:
    base = str(spec["base_primitive"])
    try:
        meta = primitive_spec(base)
    except KeyError:
        return "base_primitive_not_on_p0_whitelist"
    agg = str(spec.get("aggregation") or "")
    pclass = str(meta["primitive_class"])
    cut_type = str(spec.get("cut_type", "")).lower()
    if cut_type != "contrast":
        allowed = PRIMITIVE_CLASS_AGGREGATORS[pclass]
        if agg and agg not in allowed:
            return "aggregator_not_economic_for_primitive_class"
    reason = str(spec.get("reason") or spec.get("generation_reason") or "").strip()
    if not reason:
        return "missing_economic_mechanism"
    if "mathematically possible" in reason.lower():
        return "rationale_is_only_mathematical"
    cut_name = str(spec.get("cut_name") or "")
    if cut_type == "state" and not _conditional_ok(base, cut_name):
        return "cross_primitive_pair_not_whitelisted"
    name = candidate_name(
        base,
        cut_type,
        cut_name,
        aggregation=agg,
        contrast_operator=str(spec.get("contrast_operator") or ""),
    )
    if name in seen:
        return "duplicate_candidate_name"
    return None


def generate_from_recipes(
    recipes: Sequence[Mapping[str, object]],
    *,
    mode: str = "GENERAL_RESEARCH",
    parent: str = "",
    max_descendants: int = MAX_DESCENDANTS_PER_PARENT,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if mode not in MODES:
        raise ValueError("unknown mode {!r}".format(mode))
    proposed = list(recipes)
    accepted: List[Dict[str, object]] = []
    rejected: List[str] = []
    seen = set()
    for spec in proposed:
        if len(accepted) >= int(max_descendants):
            rejected.append("budget_exceeded")
            continue
        why = _reject_reason(spec, seen)
        if why:
            rejected.append(why)
            continue
        row = dict(spec)
        row["parent_factor_if_rescue"] = parent
        row["generation_reason"] = spec.get("reason", "")
        row["status"] = "PROPOSED"
        name = candidate_name(
            str(row["base_primitive"]),
            str(row["cut_type"]),
            str(row["cut_name"]),
            aggregation=str(row.get("aggregation") or ""),
            contrast_operator=str(row.get("contrast_operator") or ""),
        )
        row["candidate_name"] = name
        seen.add(name)
        accepted.append(row)
    reasons = Counter(rejected)
    budget = {
        "parent": parent or "(whitelist)",
        "mode": mode,
        "proposed_count": len(proposed),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_reason": ";".join(
            "{}={}".format(k, v) for k, v in sorted(reasons.items())
        )
        if reasons
        else "",
    }
    registry = append_rows(empty_registry(), accepted)
    dups = duplicate_names(registry)
    if dups:
        raise RuntimeError("duplicate candidate names after generation: {}".format(dups))
    return registry, budget


def generate_tc1_candidates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """MODE A smoke recipes. Compact, ~30-50 names, no performance tuning."""
    by_parent: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for rec in TC1_RECIPES:
        by_parent[str(rec["base_primitive"])].append(rec)
    frames = []
    budgets = []
    for parent, recs in by_parent.items():
        if len(recs) > MAX_DESCENDANTS_PER_PARENT:
            raise CartesianSearchError(
                "TC-1 recipes for {} exceed descendant budget {}".format(
                    parent, MAX_DESCENDANTS_PER_PARENT
                )
            )
        frame, budget = generate_from_recipes(
            recs,
            mode="GENERAL_RESEARCH",
            parent=parent,
            max_descendants=MAX_DESCENDANTS_PER_PARENT,
        )
        frames.append(frame)
        budgets.append(budget)
    registry = pd.concat(frames, ignore_index=True) if frames else empty_registry()
    budget_df = pd.DataFrame(budgets, columns=list(BUDGET_COLUMNS))
    n = len(registry)
    if n < 30 or n > 50:
        # Keep as a soft diagnostic in the budget frame, not a hard crash during
        # TC-0 skeleton; TC-1 runner will enforce the window.
        pass
    return registry, budget_df


def generate_rescue_candidates(
    parent_factor: str,
    recipes: Sequence[Mapping[str, object]],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if len(recipes) > MAX_DESCENDANTS_PER_PARENT:
        recipes = list(recipes)[:MAX_DESCENDANTS_PER_PARENT]
    return generate_from_recipes(
        recipes,
        mode="NONLINEAR_RESCUE",
        parent=parent_factor,
        max_descendants=MAX_DESCENDANTS_PER_PARENT,
    )


def empty_budget() -> pd.DataFrame:
    return pd.DataFrame(columns=list(BUDGET_COLUMNS))
