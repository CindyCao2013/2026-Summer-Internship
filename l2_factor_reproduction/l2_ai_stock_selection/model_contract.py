"""Walk-forward model layer contract.

Linear → LightGBM → XGBoost → (optional RF) → DL challenger.

LightGBM is P0, not a fallback. Random Forest is optional diagnostic only.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

CORE_BENCHMARKS: Tuple[Dict[str, object], ...] = (
    {
        "id": "B0",
        "name": "equal_weight_selected_factor",
        "priority": "P0",
        "family": "linear_transparent",
    },
    {
        "id": "B1",
        "name": "ridge_elasticnet",
        "priority": "P0",
        "family": "sparse_linear",
        "variants": ("Ridge", "ElasticNet"),
    },
    {
        "id": "B2",
        "name": "lightgbm",
        "priority": "P0",
        "family": "tree_boosting",
        "role": "daily_research_engine_candidate",
    },
    {
        "id": "B3",
        "name": "xgboost",
        "priority": "P0",
        "family": "tree_boosting",
        "role": "stronger_benchmark",
    },
    {
        "id": "B_RF",
        "name": "random_forest",
        "priority": "optional_diagnostic",
        "family": "bagged_trees",
        "compute_budget": "do_not_match_boosting_unless_incremental_value",
    },
)

# Conservative, compact, ex-ante. Not a large search grid.
# sklearn/LGBMRegressor names; aliases documented for LightGBM native params.
LGBM_PARAMS: Dict[str, object] = {
    "objective": "regression",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 50,  # min_data_in_leaf
    "subsample": 0.7,  # bagging_fraction
    "subsample_freq": 1,  # bagging_freq
    "colsample_bytree": 0.8,  # feature_fraction
    "reg_alpha": 0.0,  # lambda_l1
    "reg_lambda": 1.0,  # lambda_l2
    "random_state": 42,
    "n_jobs": 4,
    "early_stopping_rounds": 30,
}

LGBM_TUNE_KEYS: Tuple[str, ...] = (
    "num_leaves",
    "max_depth",
    "learning_rate",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
)

ELASTICNET_PARAMS: Dict[str, object] = {
    "alpha": 0.001,
    "l1_ratio": 0.5,
    "fit_intercept": True,
    "max_iter": 5000,
    "random_state": 42,
}

COMPARISON_MODELS: Tuple[str, ...] = (
    "explicit_l2_combination",
    "horizon_ridge_elasticnet",
    "horizon_lightgbm",
    "horizon_xgboost",
    "residual_alpha_lightgbm",
    "residual_alpha_xgboost",
    "l2_alphanet_feasibility",
)

COMPARISON_METRICS: Tuple[str, ...] = (
    "RankIC",
    "ICIR",
    "positive_ic_fraction",
    "hl_annu_ret",
    "hl_sharpe",
    "decile_monotonicity",
    "gross_annu_ret",
    "net_annu_ret",
    "gross_sharpe",
    "net_sharpe",
    "turnover",
    "MaxDD",
    "incremental_alpha",
    "style_exposure",
    "feature_stability",
    "training_runtime_sec",
    "prediction_runtime_sec",
    "complexity",
    "interpretability",
)

TREE_IMPORTANCE_CHANNELS: Tuple[str, ...] = (
    "gain",
    "split",
    "permutation",
    "shap",
)

RESEARCH_EFFICIENCY_RULE = (
    "A slightly weaker model with dramatically lower compute cost may be "
    "preferred as the daily research engine. XGBoost remains the stronger "
    "benchmark; do not pick it solely for a tiny Sharpe edge."
)


def assert_lgbm_baseline_consistent(params: Optional[Dict[str, object]] = None) -> None:
    """num_leaves must fit under 2^max_depth (full binary tree bound)."""
    p = dict(params or LGBM_PARAMS)
    leaves = int(p["num_leaves"])
    depth = int(p["max_depth"])
    if depth <= 0:
        raise ValueError("LGBM_FAST_V1 max_depth must be positive, got {}".format(depth))
    max_leaves = (1 << depth) - 1
    if leaves > max_leaves:
        raise ValueError(
            "LGBM_FAST_V1 inconsistent: num_leaves={} > 2^{}-1={}".format(
                leaves, depth, max_leaves
            )
        )
    if leaves != 15 or depth != 4:
        raise ValueError(
            "LGBM_FAST_V1 frozen baseline requires num_leaves=15 and max_depth=4, got {}/{}".format(
                leaves, depth
            )
        )


def model_contract_dict() -> Dict[str, object]:
    return {
        "core_path": "Linear → LightGBM → XGBoost → DL challenger",
        "core_benchmarks": list(CORE_BENCHMARKS),
        "lightgbm_params": dict(LGBM_PARAMS),
        "lightgbm_tune_keys": list(LGBM_TUNE_KEYS),
        "elasticnet_params": dict(ELASTICNET_PARAMS),
        "comparison_models": list(COMPARISON_MODELS),
        "comparison_metrics": list(COMPARISON_METRICS),
        "tree_importance_channels": list(TREE_IMPORTANCE_CHANNELS),
        "research_efficiency_rule": RESEARCH_EFFICIENCY_RULE,
        "random_forest": "optional_diagnostic",
        "hyperparameter_search": "compact only; no huge grid",
        "validation": "purged time-series walk-forward; no random K-fold",
        "jury_states": ["DROP", "REVIEW", "KEEP"],
        "nonlinear_override": "REVIEW_NONLINEAR; never automatic KEEP",
        "tree_gain_alone": "never KEEP",
    }
