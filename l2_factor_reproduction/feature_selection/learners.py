"""FS-4 Fast Track learners (fixed configs; no tuning)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from l2_factor_reproduction.feature_selection.fs4_contract import RIDGE_PARAMS, XGB_PARAMS


def fit_ridge(X: np.ndarray, y: np.ndarray) -> Ridge:
    model = Ridge(
        alpha=float(RIDGE_PARAMS["alpha"]),
        fit_intercept=bool(RIDGE_PARAMS["fit_intercept"]),
        solver=str(RIDGE_PARAMS["solver"]),
    )
    model.fit(X, y)
    return model


def predict_ridge(model: Ridge, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def fit_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> Tuple[Any, Dict[str, object]]:
    import xgboost as xgb

    params = dict(XGB_PARAMS)
    early = int(params.pop("early_stopping_rounds"))
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    # xgboost>=2 uses early_stopping via callbacks or constructor; force stop if supported
    try:
        model.set_params(early_stopping_rounds=early)
    except Exception:
        pass
    # re-fit with early stopping if API supports eval_set early_stopping_rounds
    try:
        model = xgb.XGBRegressor(**{**params, "early_stopping_rounds": early})
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except TypeError:
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    meta = {
        "best_iteration": getattr(model, "best_iteration", None),
        "n_estimators_fitted": int(getattr(model, "best_iteration", params.get("n_estimators", -1)) or -1),
    }
    return model, meta


def predict_xgb(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)
