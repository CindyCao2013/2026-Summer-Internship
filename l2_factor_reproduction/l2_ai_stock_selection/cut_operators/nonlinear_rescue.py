"""MODE A / MODE B entry for nonlinear rescue via temporal-state cuts.

MODE A — GENERAL_RESEARCH: small approved primitive whitelist (TC-1).
MODE B — NONLINEAR_RESCUE: only NONLINEAR_REVIEW parents from the jury.
Does not generate cuts for all 127 factors.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    MAX_DESCENDANTS_PER_PARENT,
    RESCUE_CLASSES,
    RESCUE_CORE_GATES,
    TC1_RECIPES,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.generator import (
    generate_from_recipes,
    generate_rescue_candidates,
    generate_tc1_candidates,
)

# Map frozen factor names onto a cuttable base primitive. Daily-only
# transforms (20d z-score / shock) are marked CANNOT_CUT.
PARENT_TO_PRIMITIVE: Dict[str, str] = {
    "net_buy_ratio": "net_active_flow",
    "net_buy_count_ratio": "net_active_flow",
    "buy_dominance": "net_active_flow",
    "obi_l5_mean": "obi_5",
    "obi_l1_mean": "obi_5",
    "weighted_obi_mean": "obi_5",
    "relative_spread_mean": "relative_spread",
    "large_order_ratio_20w": "large_order_amount",
    "large_order_direction": "large_order_amount",
    "closing_30m_return": "minute_return",
    "morning_return": "minute_return",
    "afternoon_return": "minute_return",
    "intraday_amihud": "minute_return",
    "signed_amount_impact": "signed_amount_impact",
    "cancel_value_pressure": "cancel_imbalance",
    "cancel_count_pressure": "cancel_imbalance",
    "vwap_close_deviation": "vwap_close_deviation",
    "close_location_value": "close_location_value",
    "return_per_amount": "return_per_amount",
    "impact_asymmetry": "impact_asymmetry",
    "closing_obi_l5": "obi_5",
    "microprice_deviation_mean": "microprice_deviation",
    "large_order_pressure": "large_order_pressure",
}

CANNOT_CUT_PARENTS = (
    "flow_zscore_20d",
    "cancel_pressure_shock_20d",
    "cancel_intensity_shock_20d",
    "net_buy_amount_mcap",
)


def recipes_for_primitive(base_primitive: str) -> List[Dict[str, object]]:
    return [dict(r) for r in TC1_RECIPES if str(r["base_primitive"]) == base_primitive]


def classify_rescue(
    *,
    abs_rank_ic: float,
    hl_sharpe: float,
    monotonicity: float,
    mi: float = float("nan"),
    residual_mi: float = float("nan"),
    parent_abs_rank_ic: float = 0.0,
    parent_hl_sharpe: float = 0.0,
    corr_parent: float = 0.0,
    corr_core: float = 0.0,
    materially_improved: Optional[bool] = None,
) -> str:
    """Multi-class rescue label. Not a single binary rule."""
    ic_ok = np.isfinite(abs_rank_ic) and abs_rank_ic >= RESCUE_CORE_GATES["abs_rank_ic"]
    hl_ok = np.isfinite(hl_sharpe) and hl_sharpe >= RESCUE_CORE_GATES["hl_sharpe"]
    mono_ok = np.isfinite(monotonicity) and monotonicity >= RESCUE_CORE_GATES["monotonicity"]
    if corr_core >= 0.80 and ic_ok:
        return "REDUNDANT_RESCUE"
    if ic_ok and hl_ok and mono_ok:
        return "RESCUED_CORE"
    improved = materially_improved
    if improved is None:
        improved = (
            (np.isfinite(abs_rank_ic) and abs_rank_ic > abs(parent_abs_rank_ic) + 0.005)
            or (np.isfinite(hl_sharpe) and hl_sharpe > parent_hl_sharpe + 0.5)
        )
    if improved:
        return "RESCUED_AUXILIARY"
    nl = (np.isfinite(mi) and mi >= 0.01) or (np.isfinite(residual_mi) and residual_mi >= 0.01)
    if nl and (not ic_ok):
        return "NONLINEAR_ONLY"
    return "FAILED_RESCUE"


def nonlinear_review_parents(jury: pd.DataFrame) -> List[str]:
    """Accept jury_state REVIEW or nonlinear_review True. Never all 127."""
    if jury is None or jury.empty:
        return []
    names = []
    name_col = "factor" if "factor" in jury.columns else "factor_name"
    for _, row in jury.iterrows():
        state = str(row.get("jury_state", "")).upper()
        flag = bool(row.get("nonlinear_review", False))
        if state in ("REVIEW", "NONLINEAR_REVIEW") or flag:
            names.append(str(row[name_col]))
    return names


def rescue_from_jury(
    jury: pd.DataFrame,
    *,
    max_descendants: int = MAX_DESCENDANTS_PER_PARENT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parents = nonlinear_review_parents(jury)
    frames = []
    budgets = []
    for parent in parents:
        if parent in CANNOT_CUT_PARENTS:
            budgets.append(
                {
                    "parent": parent,
                    "mode": "NONLINEAR_RESCUE",
                    "proposed_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 1,
                    "rejection_reason": "parent_is_daily_scalar_cannot_cut",
                }
            )
            continue
        base = PARENT_TO_PRIMITIVE.get(parent)
        if not base:
            budgets.append(
                {
                    "parent": parent,
                    "mode": "NONLINEAR_RESCUE",
                    "proposed_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 1,
                    "rejection_reason": "no_cuttable_sequence_mapping",
                }
            )
            continue
        recipes = recipes_for_primitive(base)[: int(max_descendants)]
        frame, budget = generate_rescue_candidates(parent, recipes)
        frames.append(frame)
        budgets.append(budget)
    registry = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    budget_df = pd.DataFrame(budgets)
    return registry, budget_df


def mode_a_whitelist() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return generate_tc1_candidates()


assert set(RESCUE_CLASSES) == {
    "RESCUED_CORE",
    "RESCUED_AUXILIARY",
    "NONLINEAR_ONLY",
    "FAILED_RESCUE",
    "REDUNDANT_RESCUE",
}
