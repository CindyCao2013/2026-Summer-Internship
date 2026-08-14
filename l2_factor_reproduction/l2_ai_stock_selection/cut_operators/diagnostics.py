"""Coverage / NaN / Inf / denominator / availability diagnostics for cut features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CUT_RESULT_ROOT,
    FAMILY_CUT_FEASIBILITY,
    MIN_COVERAGE_OBS,
    REGISTRY_COLUMNS,
    operator_contract_dict,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contrast_ops import (
    ratio_denominator_diagnostics,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.generator import (
    BUDGET_COLUMNS,
    generate_tc1_candidates,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    snapshot_candidate_pool,
)

PARENT_CHILD_COLUMNS = (
    "parent_rank_ic",
    "child_rank_ic",
    "delta_abs_rank_ic",
    "parent_hl_sharpe",
    "child_hl_sharpe",
    "delta_hl_sharpe",
    "parent_monotonicity",
    "child_monotonicity",
    "delta_monotonicity",
    "parent_mi",
    "child_mi",
    "parent_residual_ic",
    "child_residual_ic",
    "correlation_parent_child",
    "correlation_to_existing_core",
)


def feature_diagnostics(
    name: str,
    values: Sequence,
    *,
    n_dates: int = 0,
    n_symbols: int = 0,
) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float).ravel()
    n = int(arr.size)
    finite = arr[np.isfinite(arr)]
    inf_rate = float(np.mean(np.isinf(arr))) if n else float("nan")
    nan_rate = float(np.mean(~np.isfinite(arr))) if n else float("nan")
    if finite.size:
        cs_var = float(np.var(finite))
        cs_std = float(np.std(finite))
    else:
        cs_var = cs_std = float("nan")
    return {
        "candidate_name": name,
        "n": n,
        "n_finite": int(finite.size),
        "nan_rate": nan_rate,
        "inf_rate": inf_rate,
        "cross_sectional_variance": cs_var,
        "cross_sectional_std": cs_std,
        "coverage_n_dates": int(n_dates),
        "coverage_n_symbols": int(n_symbols),
        "min_coverage_ok": bool(finite.size >= MIN_COVERAGE_OBS),
        "pathological": bool(n and (inf_rate > 0 or nan_rate > 0.95)),
    }


def parent_child_improvement(parent: Mapping[str, float], child: Mapping[str, float]) -> Dict[str, object]:
    def g(d, k):
        try:
            return float(d.get(k, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    p_ic, c_ic = g(parent, "rank_ic"), g(child, "rank_ic")
    p_hl, c_hl = g(parent, "hl_sharpe"), g(child, "hl_sharpe")
    p_m, c_m = g(parent, "monotonicity"), g(child, "monotonicity")
    return {
        "parent_rank_ic": p_ic,
        "child_rank_ic": c_ic,
        "delta_abs_rank_ic": abs(c_ic) - abs(p_ic) if np.isfinite(p_ic) and np.isfinite(c_ic) else float("nan"),
        "parent_hl_sharpe": p_hl,
        "child_hl_sharpe": c_hl,
        "delta_hl_sharpe": c_hl - p_hl if np.isfinite(p_hl) and np.isfinite(c_hl) else float("nan"),
        "parent_monotonicity": p_m,
        "child_monotonicity": c_m,
        "delta_monotonicity": c_m - p_m if np.isfinite(p_m) and np.isfinite(c_m) else float("nan"),
        "parent_mi": g(parent, "mi"),
        "child_mi": g(child, "mi"),
        "parent_residual_ic": g(parent, "residual_ic"),
        "child_residual_ic": g(child, "residual_ic"),
        "correlation_parent_child": g(child, "correlation_parent_child"),
        "correlation_to_existing_core": g(child, "correlation_to_existing_core"),
    }


def multiple_testing_row(
    *,
    parent: str,
    n_attempted: int,
    n_evaluated: int,
    n_pass_core: int = 0,
    n_pass_aux: int = 0,
    n_nonlinear_only: int = 0,
    n_failed: int = 0,
    n_redundant: int = 0,
) -> Dict[str, object]:
    return {
        "parent": parent,
        "n_attempted_descendants": int(n_attempted),
        "n_evaluated": int(n_evaluated),
        "n_rescued_core": int(n_pass_core),
        "n_rescued_auxiliary": int(n_pass_aux),
        "n_nonlinear_only": int(n_nonlinear_only),
        "n_failed_rescue": int(n_failed),
        "n_redundant_rescue": int(n_redundant),
        "note": "Do not promote from one lucky period; require walk-forward later.",
    }


def primitive_feasibility_frame() -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in FAMILY_CUT_FEASIBILITY])


def registry_schema_frame() -> pd.DataFrame:
    roles = {
        "candidate_name": "key",
        "base_primitive": "lineage",
        "base_family": "lineage",
        "cut_type": "operator",
        "cut_definition": "operator",
        "condition_primitive": "operator",
        "aggregation": "operator",
        "contrast_operator": "operator",
        "cut_start_time": "availability",
        "cut_end_time": "availability",
        "availability_timestamp": "availability",
        "contains_close_auction": "availability",
        "contains_1456_1500": "availability",
        "latest_source_timestamp": "availability",
        "factor_available_after": "availability",
        "uses_close_auction": "availability",
        "uses_last_5min": "availability",
        "execution_contract_compatible": "availability",
        "production_execution_compatible": "availability",
        "economic_interpretation": "economics",
        "parent_factor_if_rescue": "lineage",
        "generation_reason": "economics",
        "status": "lifecycle",
    }
    rows = []
    for col in REGISTRY_COLUMNS:
        rows.append(
            {
                "column": col,
                "role": roles.get(col, ""),
                "required": True,
            }
        )
    return pd.DataFrame(rows)


def write_tc0_artifacts(root: Optional[Path] = None) -> Dict[str, str]:
    """Write TC-0 machine-readable artifacts. Does not query CH/DDB."""
    out_root = Path(root or CUT_RESULT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    contract = operator_contract_dict()
    contract_path = out_root / "operator_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=False) + "\n")
    feas = primitive_feasibility_frame()
    feas_path = out_root / "primitive_cut_feasibility.csv"
    feas.to_csv(feas_path, index=False)
    schema = registry_schema_frame()
    schema_path = out_root / "cut_registry_schema.csv"
    schema.to_csv(schema_path, index=False)
    _, budget = generate_tc1_candidates()
    budget_path = out_root / "cut_candidate_budget.csv"
    budget.to_csv(budget_path, index=False)
    pool = snapshot_candidate_pool()
    (out_root / "candidate_pool_sha256_at_tc0.txt").write_text(
        "{}\n{}\n".format(pool["path"], pool["sha256"])
    )
    return {
        "operator_contract": str(contract_path),
        "primitive_cut_feasibility": str(feas_path),
        "cut_registry_schema": str(schema_path),
        "cut_candidate_budget": str(budget_path),
    }
