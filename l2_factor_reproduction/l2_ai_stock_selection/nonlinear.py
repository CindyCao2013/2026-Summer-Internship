"""Nonlinear fitness diagnostics.

A near-zero RankIC must not auto-reject a factor that has stable nonlinear
information. These helpers are train-window only; OOS confirmation is a
separate gate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:  # pragma: no cover
    mutual_info_regression = None


def _stack_pairs(x: pd.DataFrame, y: pd.DataFrame) -> tuple:
    xx = x.reindex_like(y) if y.shape != x.shape else x
    xv = xx.to_numpy(dtype=float).ravel()
    yv = y.to_numpy(dtype=float).ravel()
    ok = np.isfinite(xv) & np.isfinite(yv)
    return xv[ok], yv[ok]


def rank_ic(x: pd.DataFrame, y: pd.DataFrame) -> float:
    """Mean daily Spearman RankIC (cross-section)."""
    common = x.index.intersection(y.index)
    cols = x.columns.intersection(y.columns)
    if len(common) == 0 or len(cols) == 0:
        return float("nan")
    ic = x.loc[common, cols].corrwith(y.loc[common, cols], axis=1, method="spearman")
    return float(ic.mean())


def residual_mutual_information(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_neighbors: int = 3,
    random_state: int = 42,
    max_samples: int = 20000,
) -> float:
    """Pooled MI(x, y) on stacked finite pairs. Subsampled, seeded."""
    if mutual_info_regression is None:
        return float("nan")
    xv, yv = _stack_pairs(x, y)
    if xv.size < 50:
        return float("nan")
    rng = np.random.default_rng(random_state)
    if xv.size > max_samples:
        pick = rng.choice(xv.size, size=max_samples, replace=False)
        xv = xv[pick]
        yv = yv[pick]
    mi = mutual_info_regression(
        xv.reshape(-1, 1),
        yv,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return float(mi[0])


def binned_conditional_return(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Equal-count bins of stacked factor values vs mean target."""
    xv, yv = _stack_pairs(x, y)
    if xv.size < n_bins * 5:
        return pd.DataFrame(columns=["bin", "n", "mean_x", "mean_y"])
    ranks = pd.Series(xv).rank(method="first")
    bins = pd.qcut(ranks, n_bins, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bins, "x": xv, "y": yv})
    g = frame.groupby("bin", observed=True)
    out = g.agg(n=("y", "size"), mean_x=("x", "mean"), mean_y=("y", "mean"))
    return out.reset_index()


def nonlinear_should_review(
    raw_rank_ic: float,
    residual_mi: float,
    *,
    ic_abs_floor: float = 0.008,
    mi_floor: float = 0.01,
) -> bool:
    """True when linear IC is weak but MI is not — review, do not auto-keep."""
    if not np.isfinite(raw_rank_ic) or not np.isfinite(residual_mi):
        return False
    return abs(raw_rank_ic) < ic_abs_floor and residual_mi >= mi_floor
