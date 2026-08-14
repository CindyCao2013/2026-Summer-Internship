"""Output layout under research/results/l2_ai_stock_selection_v1/."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from l2_factor_reproduction.l2_ai_stock_selection.contracts import RESULT_ROOT

PROJECT_DIR = RESULT_ROOT

CANDIDATE_DISCOVERY = PROJECT_DIR / "candidate_discovery"
FEATURE_ENGINEERING = PROJECT_DIR / "feature_engineering"
FEATURE_SELECTION = PROJECT_DIR / "feature_selection"
MODELS = PROJECT_DIR / "models"
ATTRIBUTION = PROJECT_DIR / "attribution"
FREQUENCY = PROJECT_DIR / "frequency"
PORTFOLIO = PROJECT_DIR / "portfolio"
DL_FEASIBILITY = PROJECT_DIR / "dl_feasibility"
REPORTS = PROJECT_DIR / "reports"
TIMING = PROJECT_DIR / "timing"
LABELS = PROJECT_DIR / "labels"
RATIO_SMOKE = FEATURE_ENGINEERING / "ratio_smoke"
CUT_OPERATORS = PROJECT_DIR / "cut_operators"
TC1_OUTPUT = CUT_OPERATORS / "tc1_output"
TC2A_OUTPUT = CUT_OPERATORS / "tc2a_output"
EXECUTION = PROJECT_DIR / "execution"
EXECUTABLE_V2V_LABELS = LABELS / "executable_v2v"
TIMING_DEGRADATION = PROJECT_DIR / "timing_degradation"
FACTOR_QUALIFICATION = PROJECT_DIR / "factor_qualification"


def required_directories() -> Tuple[Path, ...]:
    return (
        PROJECT_DIR,
        CANDIDATE_DISCOVERY,
        FEATURE_ENGINEERING,
        FEATURE_SELECTION,
        MODELS / "baseline_linear",
        MODELS / "lightgbm",
        MODELS / "xgboost",
        MODELS / "random_forest",
        ATTRIBUTION,
        FREQUENCY,
        PORTFOLIO,
        REPORTS,
        TIMING,
        LABELS,
        RATIO_SMOKE,
        CUT_OPERATORS,
        TC1_OUTPUT,
        TC2A_OUTPUT,
        EXECUTION,
        EXECUTABLE_V2V_LABELS,
        TIMING_DEGRADATION,
        FACTOR_QUALIFICATION,
        LABELS / "robustness_o2o",
        LABELS / "robustness_c2c_delayed",
    )


def frozen_artifact_paths() -> Tuple[Path, ...]:
    """Read-only historical artifacts. Phase B1.5 must not rewrite these."""
    l2 = RESULT_ROOT.parent / "l2_reproduction"
    fs3 = l2 / "feature_selection" / "fs3_walkforward_selection"
    fs4 = l2 / "feature_selection" / "fs4_fast_track"
    return (
        l2 / "candidate_pool_v1" / "candidate_registry.csv",
        fs3 / "labels" / "label_contract.json",
        fs3 / "labels" / "horizon=1" / "y_wide.parquet",
        fs3 / "labels" / "horizon=5" / "y_wide.parquet",
        fs3 / "labels" / "horizon=20" / "y_wide.parquet",
        fs4 / "audits" / "holdout_training_audit.csv",
    )


def discovery_label_dir(execution_contract: Optional[str] = None) -> Path:
    from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
        LEGACY_C2C_DIAGNOSTIC,
        PRIMARY_EXECUTION_CONTRACT,
        ROBUSTNESS_C2C_DELAYED,
        ROBUSTNESS_O2O,
        resolve_execution_contract,
    )

    name = resolve_execution_contract(execution_contract)
    if name == PRIMARY_EXECUTION_CONTRACT:
        return EXECUTABLE_V2V_LABELS
    if name == ROBUSTNESS_O2O:
        return LABELS / "robustness_o2o"
    if name == ROBUSTNESS_C2C_DELAYED:
        return LABELS / "robustness_c2c_delayed"
    if name == LEGACY_C2C_DIAGNOSTIC:
        raise ValueError(
            "LEGACY_C2C_DIAGNOSTIC labels are frozen FS-3 / diagnostic c2c; "
            "pass execution_contract explicitly and load those files separately"
        )
    raise ValueError("no label directory for {}".format(name))


def ensure_layout() -> None:
    for path in required_directories():
        path.mkdir(parents=True, exist_ok=True)
