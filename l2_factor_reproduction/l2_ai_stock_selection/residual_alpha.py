"""Train-window residual alpha mining.

Round 0: residualize the label on known systematic exposures using TRAIN dates only.
Round k: residualize on currently selected L2 factors using TRAIN dates only.

New-candidate fitness is then RankIC / MI versus the residual, never versus a
future OOS period used to form the residual itself.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    PRIMARY_EXECUTION_CONTRACT,
    resolve_execution_contract,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import (
    binned_conditional_return,
    rank_ic,
    residual_mutual_information,
)


MIN_CROSS_SECTION = 30


def demean_within_groups(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Industry (or other group) demeaning of a single cross-section."""
    v = pd.to_numeric(values, errors="coerce")
    g = groups.reindex(v.index)
    out = v.copy()
    ok = v.notna() & g.notna()
    if int(ok.sum()) == 0:
        return out
    out.loc[ok] = v.loc[ok] - v.loc[ok].groupby(g.loc[ok]).transform("mean")
    return out


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def cross_section_ols_diagnostics(
    y: np.ndarray,
    X: np.ndarray,
    *,
    min_obs: int = MIN_CROSS_SECTION,
) -> tuple:
    """OLS residual plus n_obs / condition_number / residual_std."""
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = y.shape[0]
    resid = np.full(n, np.nan, dtype=float)
    diag = {
        "n_obs": 0,
        "condition_number": float("nan"),
        "residual_std": float("nan"),
        "residual_mean": float("nan"),
        "rank": 0,
        "ok": False,
    }
    x_cols = [X[:, j] for j in range(X.shape[1])] if X.size else []
    ok = _finite_mask(y, *x_cols)
    n_obs = int(ok.sum())
    diag["n_obs"] = n_obs
    if n_obs < min_obs:
        return resid, diag
    Y = y[ok]
    design = np.column_stack([np.ones(n_obs), X[ok]])
    try:
        cond = float(np.linalg.cond(design))
    except Exception:
        cond = float("nan")
    diag["condition_number"] = cond
    beta, _, rank, _ = np.linalg.lstsq(design, Y, rcond=None)
    diag["rank"] = int(rank)
    if rank < design.shape[1]:
        return resid, diag
    e = Y - design @ beta
    resid[ok] = e
    diag["residual_std"] = float(np.std(e, ddof=1)) if e.size > 1 else float("nan")
    diag["residual_mean"] = float(np.mean(e))
    diag["ok"] = True
    return resid, diag


def cross_section_ols_residual(
    y: np.ndarray,
    X: np.ndarray,
    *,
    min_obs: int = MIN_CROSS_SECTION,
) -> np.ndarray:
    """OLS residual y - Xβ with intercept. NaN where under-determined."""
    resid, _ = cross_section_ols_diagnostics(y, X, min_obs=min_obs)
    return resid


def residualize_panel(
    y: pd.DataFrame,
    controls: Dict[str, pd.DataFrame],
    *,
    train_dates: Optional[Sequence] = None,
    min_obs: int = MIN_CROSS_SECTION,
    execution_contract: Optional[str] = None,
) -> pd.DataFrame:
    """Date-by-date residual of ``y`` on ``controls``.

    Default execution_contract is EXEC_V2V_TPLUS1_V1. Legacy C2C labels
    require execution_contract=LEGACY_C2C_DIAGNOSTIC.
    """
    resolve_execution_contract(execution_contract)
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    dates = list(train_dates) if train_dates is not None else list(y.index)
    names = list(controls)
    for dt in dates:
        if dt not in y.index:
            continue
        y_row = y.loc[dt]
        cols = []
        aligned = [y_row]
        for name in names:
            x = controls[name].reindex(index=y.index, columns=y.columns).loc[dt]
            aligned.append(x)
            cols.append(x.to_numpy(dtype=float))
        if not cols:
            out.loc[dt] = y_row
            continue
        X = np.column_stack(cols)
        resid, _ = cross_section_ols_diagnostics(
            y_row.to_numpy(dtype=float), X, min_obs=min_obs
        )
        out.loc[dt, y.columns] = resid
    return out


def residualize_panel_with_diagnostics(
    y: pd.DataFrame,
    controls: Dict[str, pd.DataFrame],
    *,
    train_dates: Optional[Sequence] = None,
    min_obs: int = MIN_CROSS_SECTION,
    execution_contract: Optional[str] = None,
) -> tuple:
    """Canonical discovery residual A: per-date cross-sectional OLS.

    Default target contract is EXEC_V2V_TPLUS1_V1.
    Dates outside ``train_dates`` stay NaN (no OOS residual target).
    """
    resolve_execution_contract(execution_contract)
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    dates = list(train_dates) if train_dates is not None else list(y.index)
    names = list(controls)
    rows = []
    for dt in dates:
        if dt not in y.index:
            continue
        y_row = y.loc[dt]
        cols = []
        for name in names:
            x = controls[name].reindex(index=y.index, columns=y.columns).loc[dt]
            cols.append(x.to_numpy(dtype=float))
        if not cols:
            out.loc[dt] = y_row
            rows.append(
                {
                    "TradeDate": pd.Timestamp(dt),
                    "n_obs": int(np.isfinite(y_row.to_numpy(dtype=float)).sum()),
                    "condition_number": float("nan"),
                    "residual_std": float("nan"),
                    "residual_mean": float("nan"),
                    "ok": True,
                }
            )
            continue
        X = np.column_stack(cols)
        resid, diag = cross_section_ols_diagnostics(
            y_row.to_numpy(dtype=float), X, min_obs=min_obs
        )
        out.loc[dt, y.columns] = resid
        rec = dict(diag)
        rec["TradeDate"] = pd.Timestamp(dt)
        rows.append(rec)
    return out, pd.DataFrame(rows)


def pooled_train_window_residual(
    y: pd.DataFrame,
    controls: Dict[str, pd.DataFrame],
    *,
    train_dates: Sequence,
    min_obs: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:
    """Definition B: one OLS on stacked train-window rows. Not canonical."""
    idx = [d for d in train_dates if d in y.index]
    if not idx:
        return pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    y_tr = y.loc[idx]
    xv = y_tr.to_numpy(dtype=float).ravel()
    cols = []
    for name in controls:
        x = controls[name].reindex(index=y.index, columns=y.columns).loc[idx]
        cols.append(x.to_numpy(dtype=float).ravel())
    if not cols:
        out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
        out.loc[idx] = y_tr
        return out
    X = np.column_stack(cols)
    resid_flat, diag = cross_section_ols_diagnostics(xv, X, min_obs=min_obs)
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype=float)
    if diag.get("ok"):
        out.loc[idx] = resid_flat.reshape(y_tr.shape)
    return out


def candidate_incremental_metrics(
    factor: pd.DataFrame,
    raw_target: pd.DataFrame,
    residual_target: pd.DataFrame,
    *,
    train_dates: Optional[Sequence] = None,
    execution_contract: Optional[str] = None,
) -> Dict[str, float]:
    """A/B/C/D fitness: raw RankIC, residual RankIC, raw MI, residual MI."""
    contract = resolve_execution_contract(execution_contract)
    if train_dates is not None:
        idx = [d for d in train_dates if d in factor.index]
        factor = factor.loc[idx]
        raw_target = raw_target.reindex(index=idx, columns=factor.columns)
        residual_target = residual_target.reindex(index=idx, columns=factor.columns)
    else:
        raw_target = raw_target.reindex_like(factor)
        residual_target = residual_target.reindex_like(factor)

    ic_raw = rank_ic(factor, raw_target)
    ic_res = rank_ic(factor, residual_target)
    mi_raw = residual_mutual_information(factor, raw_target)
    mi_res = residual_mutual_information(factor, residual_target)
    bins = binned_conditional_return(factor, residual_target)
    return {
        "raw_rank_ic": ic_raw,
        "residual_rank_ic": ic_res,
        "incremental_rank_ic": ic_res,
        "raw_mi": mi_raw,
        "residual_mi": mi_res,
        "incremental_mi": mi_res,
        "residual_bin_spread": float(bins["mean_y"].iloc[-1] - bins["mean_y"].iloc[0])
        if len(bins) >= 2
        else float("nan"),
        "n_train_dates": int(len(factor.index)),
        "execution_contract": contract,
    }


def incremental_table(
    factors: Dict[str, pd.DataFrame],
    raw_target: pd.DataFrame,
    residual_target: pd.DataFrame,
    *,
    train_dates: Optional[Sequence] = None,
    families: Optional[Dict[str, str]] = None,
    execution_contract: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    families = families or {}
    for name, wide in factors.items():
        metrics = candidate_incremental_metrics(
            wide, raw_target, residual_target, train_dates=train_dates,
            execution_contract=execution_contract,
        )
        metrics["factor"] = name
        metrics["family"] = families.get(name, "")
        rows.append(metrics)
    cols = [
        "factor",
        "family",
        "raw_rank_ic",
        "residual_rank_ic",
        "incremental_rank_ic",
        "raw_mi",
        "residual_mi",
        "incremental_mi",
        "residual_bin_spread",
        "n_train_dates",
        "execution_contract",
    ]
    return pd.DataFrame(rows)[cols]
