"""P0 learners for AI v1: Ridge/ElasticNet, LightGBM, XGBoost.

Random Forest is not implemented here on purpose (optional diagnostic).
Runtime is recorded because LightGBM vs XGBoost is also a research-efficiency
comparison, not a Sharpe-only horse race.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from l2_factor_reproduction.l2_ai_stock_selection.model_contract import (
    ELASTICNET_PARAMS,
    LGBM_PARAMS,
)


def lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except Exception:
        return False


def xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    from sklearn.linear_model import Ridge

    t0 = time.perf_counter()
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(X, y)
    return model, {"training_runtime_sec": time.perf_counter() - t0, "name": "Ridge"}


def fit_elasticnet(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import ElasticNet

    t0 = time.perf_counter()
    model = ElasticNet(
        alpha=float(ELASTICNET_PARAMS["alpha"]),
        l1_ratio=float(ELASTICNET_PARAMS["l1_ratio"]),
        fit_intercept=bool(ELASTICNET_PARAMS["fit_intercept"]),
        max_iter=int(ELASTICNET_PARAMS["max_iter"]),
        random_state=int(ELASTICNET_PARAMS["random_state"]),
    )
    model.fit(X, y)
    return model, {
        "training_runtime_sec": time.perf_counter() - t0,
        "name": "ElasticNet",
        "n_nonzero": int(np.sum(np.abs(model.coef_) > 1e-12)),
    }


def _lgbm_sklearn_params() -> Dict[str, object]:
    params = dict(LGBM_PARAMS)
    params.pop("early_stopping_rounds", None)
    params.pop("objective", None)
    return params


def fit_lightgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> Tuple[Any, Dict[str, object]]:
    import lightgbm as lgb

    early = int(LGBM_PARAMS["early_stopping_rounds"])
    params = _lgbm_sklearn_params()
    model = lgb.LGBMRegressor(**params)
    t0 = time.perf_counter()
    fit_kw = dict(
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
    )
    # LightGBM 3.2: early_stopping_rounds on fit(); 4.x moved to callbacks.
    try:
        model.fit(X_tr, y_tr, early_stopping_rounds=early, verbose=False, **fit_kw)
    except TypeError:
        callbacks = []
        try:
            callbacks.append(lgb.early_stopping(early, verbose=False))
            callbacks.append(lgb.log_evaluation(0))
        except Exception:
            pass
        model.fit(X_tr, y_tr, callbacks=callbacks or None, **fit_kw)
    elapsed = time.perf_counter() - t0
    best = getattr(model, "best_iteration_", None)
    if best is None:
        best = getattr(model, "best_iteration", None)
    meta = {
        "name": "LightGBM",
        "training_runtime_sec": float(elapsed),
        "best_iteration": int(best) if best not in (None, 0) else int(params.get("n_estimators", -1)),
        "n_features": int(X_tr.shape[1]),
        "n_train_rows": int(X_tr.shape[0]),
        "params": params,
    }
    return model, meta


def predict_with_runtime(model: Any, X: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    pred = np.asarray(model.predict(X), dtype=float)
    return pred, float(time.perf_counter() - t0)


def tree_gain_and_split(model: Any, feature_names: Optional[Tuple[str, ...]] = None) -> Dict[str, np.ndarray]:
    """Extract gain/split importances when the booster exposes them."""
    names = list(feature_names) if feature_names is not None else None
    gain = getattr(model, "feature_importances_", None)
    split = None
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
        except Exception:
            pass
    out = {}
    if gain is not None:
        out["gain"] = np.asarray(gain, dtype=float)
    if split is not None:
        out["split"] = np.asarray(split, dtype=float)
    if names is not None:
        out["feature"] = np.asarray(names)
    return out
