"""Gate 2B — novelty vs the frozen existing factor universe.

Research-priority diagnostic, not an economic truth. Near-alias (≥0.90)
blocks automatic Full Discovery unless an explicit exception is recorded.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.discovery_lite.contracts import (
    MIN_CROSS_SECTION,
    NEAR_ALIAS_THRESHOLD,
    NOVELTY_HIGH,
    NOVELTY_MEDIUM,
)
from l2_factor_reproduction.python.candidate_pool import (
    mean_daily_cross_sectional_spearman,
)


def novelty_bucket(max_abs_corr: float) -> str:
    if not np.isfinite(max_abs_corr):
        return "UNKNOWN"
    c = abs(float(max_abs_corr))
    if c < NOVELTY_HIGH:
        return "HIGH_NOVELTY"
    if c < NOVELTY_MEDIUM:
        return "MEDIUM_NOVELTY"
    if c < NEAR_ALIAS_THRESHOLD:
        return "LOW_NOVELTY"
    return "NEAR_ALIAS"


def novelty_vs_existing(
    panel: pd.DataFrame,
    candidates: Sequence[str],
    existing: Sequence[str],
    existing_family: Optional[Mapping[str, str]] = None,
    *,
    min_names: int = MIN_CROSS_SECTION,
) -> pd.DataFrame:
    """Mean daily cross-sectional Spearman of candidates vs existing factors."""
    existing_family = existing_family or {}
    cand = [c for c in candidates if c in panel.columns]
    ref = [e for e in existing if e in panel.columns and e not in cand]
    rows = []
    if not cand:
        return pd.DataFrame(
            columns=[
                "factor",
                "max_abs_corr_to_existing",
                "closest_existing_factor",
                "closest_existing_family",
                "novelty_bucket",
            ]
        )
    if not ref:
        for name in cand:
            rows.append(
                {
                    "factor": name,
                    "max_abs_corr_to_existing": np.nan,
                    "closest_existing_factor": None,
                    "closest_existing_family": None,
                    "novelty_bucket": "UNKNOWN",
                    "novelty_note": "no_existing_reference_loaded",
                }
            )
        return pd.DataFrame(rows)

    names = cand + ref
    corr = mean_daily_cross_sectional_spearman(panel, names, min_names=min_names)
    for name in cand:
        if name not in corr.index:
            rows.append(
                {
                    "factor": name,
                    "max_abs_corr_to_existing": np.nan,
                    "closest_existing_factor": None,
                    "closest_existing_family": None,
                    "novelty_bucket": "UNKNOWN",
                }
            )
            continue
        peers = corr.loc[name, ref].dropna()
        if peers.empty:
            max_corr, closest = np.nan, None
        else:
            closest = str(peers.abs().idxmax())
            max_corr = float(peers.loc[closest])
        abs_corr = abs(max_corr) if np.isfinite(max_corr) else float("nan")
        rows.append(
            {
                "factor": name,
                "max_abs_corr_to_existing": abs_corr,
                "closest_existing_factor": closest,
                "closest_existing_family": (
                    existing_family.get(closest, "") if closest else None
                ),
                "novelty_bucket": novelty_bucket(abs_corr),
            }
        )
    return pd.DataFrame(rows)
