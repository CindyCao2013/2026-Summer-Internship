"""FS-3 helpers: local eligibility + coverage-aware selector execution."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

from l2_factor_reproduction.feature_selection.selectors import (
    RESULT_COLUMNS,
    _stable_rank_descending,
    benjamini_hochberg_reject,
    detect_constant_mask,
    feature_schema_hash,
    validate_params,
)

MIN_FINITE_OBS = 500
# Ex-ante computational bound for MI (not tuned on selection quality).
MI_MAX_SAMPLES = 50_000
# Ex-ante computational bound for Tree complete-case fit rows.
TREE_MAX_SAMPLES = 100_000


def local_eligibility_mask(
    X_col: np.ndarray,
    y: np.ndarray,
    *,
    min_finite_obs: int = MIN_FINITE_OBS,
) -> Tuple[bool, str, int]:
    """Return (eligible, reason, effective_n) for one feature column."""
    m = np.isfinite(X_col) & np.isfinite(y)
    n = int(m.sum())
    if n < min_finite_obs:
        return False, "INSUFFICIENT_SAMPLE", n
    if float(np.nanstd(X_col[m])) < 1e-12:
        return False, "CONSTANT", n
    return True, "", n


def _scores_f(X_col: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(X_col) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    F, p = f_regression(X_col[m].reshape(-1, 1), y[m])
    return float(F[0]), float(p[0])


def _score_mi(
    X_col: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int,
    max_samples: int = MI_MAX_SAMPLES,
) -> float:
    m = np.isfinite(X_col) & np.isfinite(y)
    idx = np.where(m)[0]
    if idx.size < max(10, n_neighbors + 2):
        return np.nan
    if idx.size > max_samples:
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
    mi = mutual_info_regression(
        X_col[idx].reshape(-1, 1),
        y[idx],
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return float(mi[0])


def run_univariate_selector_on_panel(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Sequence[str],
    families: Dict[str, str],
    coverage: Dict[str, float],
    selector_name: str,
    params: Dict,
    schema_hash: str,
) -> pd.DataFrame:
    """F/MI/FPR/FDR with per-feature effective samples (no universal dropna)."""
    params_n = validate_params(selector_name, params)
    n_feat = len(feature_names)
    scores = np.full(n_feat, np.nan)
    pvals = np.full(n_feat, np.nan)
    eligible = np.zeros(n_feat, dtype=bool)
    reasons = [""] * n_feat
    eff_n = np.zeros(n_feat, dtype=int)
    is_const = np.zeros(n_feat, dtype=bool)

    for j, f in enumerate(feature_names):
        col = X[f].to_numpy(dtype=float) if f in X.columns else np.full(len(y), np.nan)
        ok, reason, n = local_eligibility_mask(col, y)
        eligible[j] = ok
        reasons[j] = reason
        eff_n[j] = n
        is_const[j] = reason == "CONSTANT"
        if not ok:
            continue
        if selector_name.startswith("F_REGRESSION"):
            scores[j], pvals[j] = _scores_f(col, y)
        elif selector_name == "MI_REGRESSION_KBEST":
            scores[j] = _score_mi(
                col,
                y,
                n_neighbors=int(params_n["n_neighbors"]),
                random_state=int(params_n["random_state"]),
                max_samples=int(params_n.get("max_samples", MI_MAX_SAMPLES)),
            )

    selected = np.zeros(n_feat, dtype=bool)
    if selector_name == "F_REGRESSION_KBEST":
        k = min(int(params_n["k"]), int(eligible.sum()))
        ranks = _stable_rank_descending(scores, feature_names)
        order = np.argsort(ranks)
        picked = [idx for idx in order if eligible[idx]][:k]
        selected[picked] = True
    elif selector_name == "MI_REGRESSION_KBEST":
        k = min(int(params_n["k"]), int(eligible.sum()))
        ranks = _stable_rank_descending(scores, feature_names)
        order = np.argsort(ranks)
        picked = [idx for idx in order if eligible[idx]][:k]
        selected[picked] = True
    elif selector_name == "F_REGRESSION_FPR":
        selected = (pvals < float(params_n["alpha"])) & eligible
    elif selector_name == "F_REGRESSION_FDR":
        # BH only among eligible finite p-values; others stay False
        p_work = np.where(eligible, pvals, np.nan)
        selected = benjamini_hochberg_reject(p_work, float(params_n["alpha"])) & eligible
    else:
        raise ValueError(selector_name)

    ranks = _stable_rank_descending(scores, feature_names)
    rows = []
    for j, f in enumerate(feature_names):
        status = "OK" if eligible[j] else (reasons[j] or "INELIGIBLE")
        rows.append(
            {
                "feature": f,
                "family": families.get(f, ""),
                "selector_name": selector_name,
                "score": scores[j],
                "p_value": pvals[j],
                "coefficient": np.nan,
                "importance": np.nan,
                "selected": bool(selected[j]),
                "selection_rank": int(ranks[j]),
                "effective_n": int(eff_n[j]),
                "coverage_ratio": coverage.get(f, np.nan),
                "selector_parameter": str(params_n),
                "feature_schema_hash": schema_hash,
                "is_constant": bool(is_const[j]),
                "status": status,
                "locally_eligible": bool(eligible[j]),
                "local_ineligible_reason": reasons[j],
            }
        )
    return pd.DataFrame(rows)


def run_multivariate_selector_on_panel(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Sequence[str],
    families: Dict[str, str],
    coverage: Dict[str, float],
    selector_name: str,
    params: Dict,
    schema_hash: str,
) -> pd.DataFrame:
    """L1/Tree on locally-eligible features with complete-case among that subset."""
    params_n = validate_params(selector_name, params)
    elig_feats: List[str] = []
    elig_info: Dict[str, Tuple[str, int]] = {}
    for f in feature_names:
        col = X[f].to_numpy(dtype=float) if f in X.columns else np.full(len(y), np.nan)
        ok, reason, n = local_eligibility_mask(col, y)
        elig_info[f] = (reason, n)
        if ok:
            elig_feats.append(f)

    coefs = {f: np.nan for f in feature_names}
    imps = {f: np.nan for f in feature_names}
    scores = {f: np.nan for f in feature_names}
    selected = {f: False for f in feature_names}
    status = {f: (elig_info[f][0] or "INELIGIBLE") for f in feature_names}
    for f in elig_feats:
        status[f] = "OK"

    if len(elig_feats) >= 2:
        Xm = X[elig_feats].to_numpy(dtype=float)
        row_ok = np.isfinite(y) & np.all(np.isfinite(Xm), axis=1)
        eff = int(row_ok.sum())
        if eff >= MIN_FINITE_OBS:
            Xc = Xm[row_ok]
            yc = y[row_ok]
            if selector_name == "L1_REGRESSION":
                model = Lasso(
                    alpha=float(params_n["alpha"]),
                    fit_intercept=bool(params_n["fit_intercept"]),
                    max_iter=int(params_n["max_iter"]),
                    tol=float(params_n["tol"]),
                )
                model.fit(Xc, yc)
                tol_c = float(params_n["coefficient_tolerance"])
                for j, f in enumerate(elig_feats):
                    c = float(model.coef_[j])
                    coefs[f] = c
                    scores[f] = abs(c)
                    selected[f] = abs(c) > tol_c
                    status[f] = "OK"
                    elig_info[f] = (elig_info[f][0], eff)
            elif selector_name == "TREE_IMPORTANCE_REGRESSION":
                max_s = int(params_n.get("max_samples", TREE_MAX_SAMPLES))
                if Xc.shape[0] > max_s:
                    rng = np.random.default_rng(int(params_n["random_state"]))
                    pick = rng.choice(Xc.shape[0], size=max_s, replace=False)
                    pick.sort()
                    Xc = Xc[pick]
                    yc = yc[pick]
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
                thr = float(params_n["threshold_value"]) * float(np.mean(imp))
                for j, f in enumerate(elig_feats):
                    imps[f] = float(imp[j])
                    scores[f] = float(imp[j])
                    selected[f] = float(imp[j]) >= thr
                    status[f] = "OK"
                    elig_info[f] = (elig_info[f][0], eff)
            else:
                raise ValueError(selector_name)
        else:
            for f in elig_feats:
                status[f] = "INSUFFICIENT_SAMPLE_MATRIX"
                elig_info[f] = ("INSUFFICIENT_SAMPLE_MATRIX", eff)

    score_arr = np.array([scores[f] for f in feature_names], dtype=float)
    ranks = _stable_rank_descending(score_arr, feature_names)
    rows = []
    for j, f in enumerate(feature_names):
        reason, n = elig_info[f]
        loc_ok = f in elig_feats and status[f] in ("OK",)
        # locally eligible if passed univariate gate even if matrix failed
        locally = f in elig_feats
        rows.append(
            {
                "feature": f,
                "family": families.get(f, ""),
                "selector_name": selector_name,
                "score": scores[f],
                "p_value": np.nan,
                "coefficient": coefs[f],
                "importance": imps[f],
                "selected": bool(selected[f]),
                "selection_rank": int(ranks[j]),
                "effective_n": int(n),
                "coverage_ratio": coverage.get(f, np.nan),
                "selector_parameter": str(params_n),
                "feature_schema_hash": schema_hash,
                "is_constant": reason == "CONSTANT",
                "status": status[f],
                "locally_eligible": bool(locally),
                "local_ineligible_reason": "" if locally else reason,
            }
        )
    return pd.DataFrame(rows)


def run_fs3_selector(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Sequence[str],
    families: Dict[str, str],
    coverage: Dict[str, float],
    selector_name: str,
    params: Dict,
    schema_hash: str,
) -> pd.DataFrame:
    if selector_name in (
        "F_REGRESSION_KBEST",
        "MI_REGRESSION_KBEST",
        "F_REGRESSION_FPR",
        "F_REGRESSION_FDR",
    ):
        return run_univariate_selector_on_panel(
            X, y, feature_names, families, coverage, selector_name, params, schema_hash
        )
    return run_multivariate_selector_on_panel(
        X, y, feature_names, families, coverage, selector_name, params, schema_hash
    )
