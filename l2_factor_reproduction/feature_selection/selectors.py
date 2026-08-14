"""FS-2 Huatai-methodology-compatible feature selector engine (regression-first).

Canonical selectors:
  F_REGRESSION_KBEST
  MI_REGRESSION_KBEST
  F_REGRESSION_FPR
  F_REGRESSION_FDR
  L1_REGRESSION
  TREE_IMPORTANCE_REGRESSION

No real forward-return labels. No walk-forward. No alpha optimization.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso

SELECTOR_CONTRACT_VERSION = "fs2_selector_v1"
RESULT_COLUMNS: Tuple[str, ...] = (
    "feature",
    "family",
    "selector_name",
    "score",
    "p_value",
    "coefficient",
    "importance",
    "selected",
    "selection_rank",
    "effective_n",
    "coverage_ratio",
    "selector_parameter",
    "feature_schema_hash",
    "is_constant",
    "status",
)

CANONICAL_SELECTORS: Tuple[str, ...] = (
    "F_REGRESSION_KBEST",
    "MI_REGRESSION_KBEST",
    "F_REGRESSION_FPR",
    "F_REGRESSION_FDR",
    "L1_REGRESSION",
    "TREE_IMPORTANCE_REGRESSION",
)


def ordered_feature_hash(feature_names: Sequence[str]) -> str:
    """Stable hash of the frozen ordered feature list."""
    payload = "\n".join(str(f) for f in feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def feature_schema_hash(
    feature_names: Sequence[str],
    *,
    families: Optional[Mapping[str, str]] = None,
) -> str:
    """Hash ordered features (+ optional family map) for FS-1 compatibility."""
    lines = []
    for f in feature_names:
        fam = "" if families is None else str(families.get(f, ""))
        lines.append(f"{f}\t{fam}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def detect_constant_mask(X: np.ndarray) -> np.ndarray:
    """Boolean mask: True where column is constant / near-constant on finite rows."""
    out = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            out[j] = True
            continue
        v = col[finite]
        out[j] = float(np.nanstd(v)) < 1e-12
    return out


def _stable_rank_descending(
    scores: np.ndarray,
    feature_order: Sequence[str],
) -> np.ndarray:
    """Rank 1..n by score desc, ties broken by frozen feature order. NaN last."""
    n = len(scores)
    order_idx = np.arange(n)
    # primary: -score (NaN -> +inf so last); secondary: original order
    score_key = np.where(np.isfinite(scores), -scores.astype(float), np.inf)
    # lexsort: last key is primary
    order = np.lexsort((order_idx, score_key))
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    return ranks


def benjamini_hochberg_reject(
    p_values: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Benjamini–Hochberg FDR control.

    Given sorted p_(1) <= ... <= p_(m), find largest i with
    p_(i) <= alpha * i / m, reject 1..i.

    NaN p-values are never rejected and do not count in m
    (m = number of finite p-values).
    """
    p = np.asarray(p_values, dtype=float)
    reject = np.zeros(len(p), dtype=bool)
    finite = np.isfinite(p)
    if not finite.any():
        return reject
    idx = np.where(finite)[0]
    p_f = p[idx]
    order = np.argsort(p_f, kind="mergesort")
    p_sorted = p_f[order]
    m = len(p_sorted)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p_sorted <= thresh
    if not below.any():
        return reject
    max_i = int(np.max(np.where(below)[0]))  # 0-based in sorted finite
    selected_sorted = order[: max_i + 1]
    reject[idx[selected_sorted]] = True
    return reject


@dataclass
class SelectorResult:
    """Standardized per-feature selector output."""

    table: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_support(self) -> np.ndarray:
        return self.table["selected"].to_numpy(dtype=bool)

    def get_scores(self) -> np.ndarray:
        return self.table["score"].to_numpy(dtype=float)

    def get_result(self) -> pd.DataFrame:
        return self.table.copy()

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)


class FeatureSelector:
    """Thin wrapper around a fitted selector result."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = dict(params)
        self._result: Optional[SelectorResult] = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        feature_metadata: Optional[pd.DataFrame] = None,
    ) -> "FeatureSelector":
        self._result = run_selector(
            self.name,
            X,
            y,
            feature_names=feature_names,
            feature_metadata=feature_metadata,
            params=self.params,
        )
        self._feature_names = list(self._result.table["feature"])
        return self

    def get_support(self) -> np.ndarray:
        assert self._result is not None
        return self._result.get_support()

    def get_scores(self) -> np.ndarray:
        assert self._result is not None
        return self._result.get_scores()

    def get_result(self) -> pd.DataFrame:
        assert self._result is not None
        return self._result.get_result()

    def get_metadata(self) -> Dict[str, Any]:
        assert self._result is not None
        return self._result.get_metadata()


def build_selector(selector_name: str, params: Optional[Dict[str, Any]] = None) -> FeatureSelector:
    if selector_name not in CANONICAL_SELECTORS:
        raise ValueError(
            f"Unknown selector {selector_name!r}; "
            f"expected one of {CANONICAL_SELECTORS}"
        )
    return FeatureSelector(selector_name, params or {})


def validate_params(selector_name: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and return normalized params; raise ValueError on invalid."""
    p = dict(params)
    if selector_name in ("F_REGRESSION_KBEST", "MI_REGRESSION_KBEST"):
        if "k" not in p:
            raise ValueError(f"{selector_name} requires k")
        k = int(p["k"])
        if k <= 0:
            raise ValueError("k must be > 0")
        p["k"] = k
        if selector_name == "MI_REGRESSION_KBEST":
            if "n_neighbors" not in p:
                raise ValueError("MI_REGRESSION_KBEST requires n_neighbors")
            nn = int(p["n_neighbors"])
            if nn < 1:
                raise ValueError("n_neighbors must be >= 1")
            p["n_neighbors"] = nn
            if "random_state" not in p:
                raise ValueError("MI_REGRESSION_KBEST requires explicit random_state")
            p["random_state"] = int(p["random_state"])
    elif selector_name in ("F_REGRESSION_FPR", "F_REGRESSION_FDR"):
        if "alpha" not in p:
            raise ValueError(f"{selector_name} requires alpha")
        alpha = float(p["alpha"])
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must satisfy 0 < alpha < 1")
        p["alpha"] = alpha
    elif selector_name == "L1_REGRESSION":
        if "alpha" not in p:
            raise ValueError("L1_REGRESSION requires alpha")
        alpha = float(p["alpha"])
        if alpha <= 0:
            raise ValueError("L1 alpha must be > 0")
        p["alpha"] = alpha
        p.setdefault("fit_intercept", True)
        p.setdefault("max_iter", 5000)
        p.setdefault("tol", 1e-4)
        p.setdefault("coefficient_tolerance", 1e-12)
        if float(p["coefficient_tolerance"]) < 0:
            raise ValueError("coefficient_tolerance must be >= 0")
    elif selector_name == "TREE_IMPORTANCE_REGRESSION":
        if "random_state" not in p:
            raise ValueError("TREE_IMPORTANCE_REGRESSION requires explicit random_state")
        p["random_state"] = int(p["random_state"])
        p.setdefault("n_estimators", 50)
        p.setdefault("max_depth", 4)
        p.setdefault("min_samples_leaf", 5)
        p.setdefault("max_features", "sqrt")
        p.setdefault("n_jobs", 1)
        p.setdefault("threshold_mode", "mean_multiple")
        p.setdefault("threshold_value", 1.0)
        if int(p["n_estimators"]) <= 0:
            raise ValueError("n_estimators must be > 0")
        if float(p["threshold_value"]) < 0:
            raise ValueError("threshold_value must be >= 0")
    else:
        raise ValueError(f"Unknown selector {selector_name}")
    return p


def _prepare_xy(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    feature_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if feature_names is None else list(feature_names)
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
        if feature_names is None:
            names = [f"f{i}" for i in range(arr.shape[1])]
        else:
            names = list(feature_names)
    yy = np.asarray(y, dtype=float).ravel()
    if arr.ndim != 2:
        raise ValueError("X must be 2D")
    if arr.shape[0] != yy.shape[0]:
        raise ValueError("X/y length mismatch")
    if len(names) != arr.shape[1]:
        raise ValueError("feature_names length mismatch")

    # row filter: finite y and all-finite X row (synthetic fixtures are complete)
    row_ok = np.isfinite(yy) & np.all(np.isfinite(arr), axis=1)
    effective_n = int(row_ok.sum())
    return arr[row_ok], yy[row_ok], names, effective_n


def _meta_map(
    feature_names: Sequence[str],
    feature_metadata: Optional[pd.DataFrame],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    families: Dict[str, str] = {}
    coverage: Dict[str, float] = {}
    if feature_metadata is None or feature_metadata.empty:
        return families, coverage
    md = feature_metadata.copy()
    if "feature" not in md.columns:
        return families, coverage
    md = md.set_index("feature")
    for f in feature_names:
        if f in md.index:
            if "family" in md.columns:
                families[f] = str(md.loc[f, "family"])
            if "coverage_ratio" in md.columns:
                try:
                    coverage[f] = float(md.loc[f, "coverage_ratio"])
                except (TypeError, ValueError):
                    coverage[f] = float("nan")
    return families, coverage


def _empty_table(
    feature_names: Sequence[str],
    selector_name: str,
    *,
    families: Mapping[str, str],
    coverage: Mapping[str, float],
    effective_n: int,
    schema_hash: str,
    param_str: str,
    const_mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(feature_names):
        rows.append(
            {
                "feature": f,
                "family": families.get(f, ""),
                "selector_name": selector_name,
                "score": np.nan,
                "p_value": np.nan,
                "coefficient": np.nan,
                "importance": np.nan,
                "selected": False,
                "selection_rank": i + 1,
                "effective_n": effective_n,
                "coverage_ratio": coverage.get(f, np.nan),
                "selector_parameter": param_str,
                "feature_schema_hash": schema_hash,
                "is_constant": bool(const_mask[i]) if i < len(const_mask) else False,
                "status": "CONSTANT" if (i < len(const_mask) and const_mask[i]) else "OK",
            }
        )
    return pd.DataFrame(rows)[list(RESULT_COLUMNS)]


def run_selector(
    selector_name: str,
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    feature_metadata: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
) -> SelectorResult:
    """Fit one canonical selector and return standardized result."""
    t0 = time.time()
    params_n = validate_params(selector_name, params or {})
    Xc, yc, names, effective_n = _prepare_xy(X, y, feature_names)
    families, coverage = _meta_map(names, feature_metadata)
    schema_hash = feature_schema_hash(names, families=families)
    order_hash = ordered_feature_hash(names)
    param_str = json.dumps(params_n, sort_keys=True, default=str)
    const_mask = detect_constant_mask(Xc) if effective_n >= 2 else np.ones(len(names), dtype=bool)

    if effective_n < 3:
        table = _empty_table(
            names,
            selector_name,
            families=families,
            coverage=coverage,
            effective_n=effective_n,
            schema_hash=schema_hash,
            param_str=param_str,
            const_mask=const_mask,
        )
        table["status"] = "INSUFFICIENT_SAMPLE"
        meta = {
            "selector_name": selector_name,
            "selector_version": SELECTOR_CONTRACT_VERSION,
            "parameters": params_n,
            "n_samples_input": int(np.asarray(y).shape[0]),
            "n_features_input": len(names),
            "n_features_selected": 0,
            "feature_schema_hash": schema_hash,
            "ordered_feature_hash": order_hash,
            "effective_n": effective_n,
            "runtime_seconds": round(time.time() - t0, 4),
            "status": "INSUFFICIENT_SAMPLE",
        }
        return SelectorResult(table=table, metadata=meta)

    scores = np.full(len(names), np.nan, dtype=float)
    p_values = np.full(len(names), np.nan, dtype=float)
    coefs = np.full(len(names), np.nan, dtype=float)
    importances = np.full(len(names), np.nan, dtype=float)
    selected = np.zeros(len(names), dtype=bool)
    status = np.array(["OK"] * len(names), dtype=object)
    status[const_mask] = "CONSTANT"

    # Valid columns for univariate scores: non-constant
    valid = ~const_mask

    try:
        if selector_name in (
            "F_REGRESSION_KBEST",
            "F_REGRESSION_FPR",
            "F_REGRESSION_FDR",
        ):
            # sklearn f_regression on all columns; constants -> nan/0
            F, p = f_regression(Xc, yc)
            scores = np.asarray(F, dtype=float)
            p_values = np.asarray(p, dtype=float)
            scores[const_mask] = np.nan
            p_values[const_mask] = np.nan

            if selector_name == "F_REGRESSION_KBEST":
                k = min(int(params_n["k"]), int(valid.sum()))
                ranks_tmp = _stable_rank_descending(scores, names)
                # select top-k among non-constant by rank
                order = np.argsort(ranks_tmp)
                picked = []
                for idx in order:
                    if not valid[idx]:
                        continue
                    picked.append(idx)
                    if len(picked) >= k:
                        break
                selected[picked] = True
            elif selector_name == "F_REGRESSION_FPR":
                alpha = float(params_n["alpha"])
                selected = (p_values < alpha) & valid
            else:  # FDR
                alpha = float(params_n["alpha"])
                selected = benjamini_hochberg_reject(p_values, alpha) & valid

        elif selector_name == "MI_REGRESSION_KBEST":
            mi = mutual_info_regression(
                Xc,
                yc,
                n_neighbors=int(params_n["n_neighbors"]),
                random_state=int(params_n["random_state"]),
            )
            scores = np.asarray(mi, dtype=float)
            scores[const_mask] = np.nan
            k = min(int(params_n["k"]), int(valid.sum()))
            ranks_tmp = _stable_rank_descending(scores, names)
            order = np.argsort(ranks_tmp)
            picked = []
            for idx in order:
                if not valid[idx]:
                    continue
                picked.append(idx)
                if len(picked) >= k:
                    break
            selected[picked] = True

        elif selector_name == "L1_REGRESSION":
            model = Lasso(
                alpha=float(params_n["alpha"]),
                fit_intercept=bool(params_n["fit_intercept"]),
                max_iter=int(params_n["max_iter"]),
                tol=float(params_n["tol"]),
                random_state=None,
            )
            model.fit(Xc, yc)
            coefs = np.asarray(model.coef_, dtype=float)
            tol_c = float(params_n["coefficient_tolerance"])
            scores = np.abs(coefs)
            selected = (np.abs(coefs) > tol_c) & valid
            # constants forced unselected
            selected[const_mask] = False
            scores[const_mask] = np.nan
            coefs[const_mask] = np.nan

        elif selector_name == "TREE_IMPORTANCE_REGRESSION":
            model = RandomForestRegressor(
                n_estimators=int(params_n["n_estimators"]),
                max_depth=params_n["max_depth"],
                min_samples_leaf=int(params_n["min_samples_leaf"]),
                max_features=params_n["max_features"],
                random_state=int(params_n["random_state"]),
                n_jobs=int(params_n["n_jobs"]),
            )
            model.fit(Xc, yc)
            imp = np.asarray(model.feature_importances_, dtype=float)
            importances = imp.copy()
            scores = imp.copy()
            mode = str(params_n["threshold_mode"])
            thr_v = float(params_n["threshold_value"])
            if mode == "mean_multiple":
                mean_imp = float(np.mean(imp)) if len(imp) else 0.0
                thr = thr_v * mean_imp
                selected = (imp >= thr) & valid
            else:
                raise ValueError(f"Unsupported threshold_mode={mode}")
            selected[const_mask] = False
            scores[const_mask] = np.nan
            importances[const_mask] = np.nan
        else:
            raise ValueError(f"Unhandled selector {selector_name}")

    except Exception as exc:  # noqa: BLE001
        if selector_name in ("L1_REGRESSION", "TREE_IMPORTANCE_REGRESSION"):
            raise RuntimeError(f"{selector_name} FIT_FAILED: {exc}") from exc
        status[:] = "NUMERICAL_WARNING"
        status[const_mask] = "CONSTANT"

    ranks = _stable_rank_descending(scores, names)
    rows = []
    for i, f in enumerate(names):
        rows.append(
            {
                "feature": f,
                "family": families.get(f, ""),
                "selector_name": selector_name,
                "score": scores[i],
                "p_value": p_values[i],
                "coefficient": coefs[i],
                "importance": importances[i],
                "selected": bool(selected[i]),
                "selection_rank": int(ranks[i]),
                "effective_n": effective_n,
                "coverage_ratio": coverage.get(f, np.nan),
                "selector_parameter": param_str,
                "feature_schema_hash": schema_hash,
                "is_constant": bool(const_mask[i]),
                "status": str(status[i]),
            }
        )
    table = pd.DataFrame(rows)[list(RESULT_COLUMNS)]

    meta = {
        "selector_name": selector_name,
        "selector_version": SELECTOR_CONTRACT_VERSION,
        "parameters": params_n,
        "n_samples_input": int(np.asarray(y).shape[0]),
        "n_features_input": len(names),
        "n_features_selected": int(selected.sum()),
        "feature_schema_hash": schema_hash,
        "ordered_feature_hash": order_hash,
        "effective_n": effective_n,
        "missing_policy": "drop_rows_with_any_nonfinite_X_or_y",
        "constant_feature_policy": "keep_in_output_unselected_status_CONSTANT",
        "random_state": params_n.get("random_state"),
        "runtime_seconds": round(time.time() - t0, 4),
        "real_labels_allowed": False,
    }
    return SelectorResult(table=table, metadata=meta)


SELECTOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "F_REGRESSION_KBEST": {
        "selector_family": "univariate",
        "problem_type": "regression",
        "score_type": "F_STATISTIC",
        "has_p_value": True,
        "selection_rule": "TOP_K",
        "deterministic_given_seed": True,
        "default_test_parameters": {"k": 2},
        "implementation_backend": "sklearn.feature_selection.f_regression",
        "status": "READY",
    },
    "MI_REGRESSION_KBEST": {
        "selector_family": "univariate",
        "problem_type": "regression",
        "score_type": "MUTUAL_INFORMATION",
        "has_p_value": False,
        "selection_rule": "TOP_K",
        "deterministic_given_seed": True,
        "default_test_parameters": {
            "k": 2,
            "n_neighbors": 3,
            "random_state": 42,
        },
        "implementation_backend": "sklearn.feature_selection.mutual_info_regression",
        "status": "READY",
    },
    "F_REGRESSION_FPR": {
        "selector_family": "univariate",
        "problem_type": "regression",
        "score_type": "F_STATISTIC",
        "has_p_value": True,
        "selection_rule": "P_LT_ALPHA",
        "deterministic_given_seed": True,
        "default_test_parameters": {"alpha": 0.05},
        "implementation_backend": "sklearn.feature_selection.f_regression",
        "status": "READY",
    },
    "F_REGRESSION_FDR": {
        "selector_family": "univariate",
        "problem_type": "regression",
        "score_type": "F_STATISTIC",
        "has_p_value": True,
        "selection_rule": "BENJAMINI_HOCHBERG",
        "deterministic_given_seed": True,
        "default_test_parameters": {"alpha": 0.05},
        "implementation_backend": "f_regression+BH",
        "status": "READY",
    },
    "L1_REGRESSION": {
        "selector_family": "multivariate_sparse",
        "problem_type": "regression",
        "score_type": "ABS_COEFFICIENT",
        "has_p_value": False,
        "selection_rule": "ABS_COEF_GT_TOL",
        "deterministic_given_seed": True,
        "default_test_parameters": {
            "alpha": 0.15,
            "fit_intercept": True,
            "max_iter": 5000,
            "tol": 1e-4,
            "coefficient_tolerance": 1e-12,
        },
        "implementation_backend": "sklearn.linear_model.Lasso",
        "status": "READY",
    },
    "TREE_IMPORTANCE_REGRESSION": {
        "selector_family": "multivariate_tree",
        "problem_type": "regression",
        "score_type": "FEATURE_IMPORTANCE",
        "has_p_value": False,
        "selection_rule": "IMPORTANCE_GE_MEAN_MULTIPLE",
        "deterministic_given_seed": True,
        "default_test_parameters": {
            "n_estimators": 50,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,
            "threshold_mode": "mean_multiple",
            "threshold_value": 1.0,
        },
        "implementation_backend": "sklearn.ensemble.RandomForestRegressor",
        "status": "READY",
    },
}
