"""Gate 2A — candidate-to-candidate redundancy.

Reuses ``candidate_pool.mean_daily_cross_sectional_spearman`` and
``redundancy_annotations`` (daily cross-sectional Spearman, |ρ|≥0.80 clusters).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.discovery_lite.contracts import (
    MIN_CROSS_SECTION,
    REDUNDANCY_CORR_THRESHOLD,
)
from l2_factor_reproduction.python.candidate_pool import (
    correlation_pairs,
    mean_daily_cross_sectional_spearman,
    redundancy_annotations,
)


def candidate_correlation(
    panel: pd.DataFrame,
    names: Sequence[str],
    *,
    min_names: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:
    present = [n for n in names if n in panel.columns]
    if len(present) < 2:
        return pd.DataFrame(index=present, columns=present, dtype=float)
    return mean_daily_cross_sectional_spearman(
        panel, present, min_names=min_names
    )


def cluster_candidates(
    corr: pd.DataFrame,
    *,
    threshold: float = REDUNDANCY_CORR_THRESHOLD,
) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "redundancy_cluster_080",
                "max_candidate_corr",
                "max_candidate_corr_peer",
            ]
        )
    anno = redundancy_annotations(corr, threshold=threshold)
    out = anno.rename(
        columns={
            "max_corr_peer": "max_candidate_corr_peer",
            "max_abs_corr": "max_candidate_corr",
        }
    )
    keep = [
        "factor",
        "redundancy_cluster_080",
        "max_candidate_corr",
        "max_candidate_corr_peer",
    ]
    extra = [c for c in ("near_alias_observed",) if c in out.columns]
    return out[keep + extra]


def redundancy_pair_table(corr: pd.DataFrame) -> pd.DataFrame:
    return correlation_pairs(corr)


def singleton_clusters(names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(list(names), start=1):
        rows.append(
            {
                "factor": name,
                "redundancy_cluster_080": f"R{i}",
                "max_candidate_corr": np.nan,
                "max_candidate_corr_peer": None,
            }
        )
    return pd.DataFrame(rows)
