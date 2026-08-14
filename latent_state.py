"""Latent market state estimation from microstructure bricks (PCA / EWMA)."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LATENT_STATE_NAMES = [
    "liquidity_stress",
    "information_flow_intensity",
    "order_imbalance_regime",
    "volatility_impulse",
]


def _cross_section_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.sub(wide.mean(axis=1), axis=0).div(wide.std(axis=1), axis=0)


def estimate_latent_states_pca(
    bricks: Dict[str, pd.DataFrame],
    n_components: int = 1,
    brick_subset: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each calendar date, PCA on cross-section of standardized bricks.

    Returns one latent state series per PC (mapped to LATENT_STATE_NAMES by index).
    """
    names = brick_subset or list(bricks.keys())
    aligned_index = bricks[names[0]].index
    aligned_cols = bricks[names[0]].columns

    latent: Dict[str, pd.DataFrame] = {
        name: pd.DataFrame(index=aligned_index, columns=aligned_cols, dtype=float)
        for name in LATENT_STATE_NAMES[:n_components]
    }

    for date in aligned_index:
        rows = []
        syms = None
        for bname in names:
            row = bricks[bname].loc[date]
            if syms is None:
                syms = row.dropna().index
            rows.append(row.reindex(syms).values)
        mat = np.column_stack(rows)
        valid = ~np.any(np.isnan(mat), axis=1)
        if valid.sum() < max(30, n_components * 5):
            continue
        mat = mat[valid]
        syms_valid = syms[valid]
        scaled = StandardScaler().fit_transform(mat)
        pca = PCA(n_components=min(n_components, scaled.shape[1]))
        comps = pca.fit_transform(scaled)
        for i in range(comps.shape[1]):
            latent[LATENT_STATE_NAMES[i]].loc[date, syms_valid] = comps[:, i]

    return latent


def estimate_liquidity_stress_simple(bricks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fallback: equal-weight z-scored composite of liquidity-related bricks."""
    keys = ["price_impact_proxy", "spread_proxy", "volume_burst_proxy", "ofi_proxy"]
    keys = [k for k in keys if k in bricks]
    if not keys:
        raise ValueError("No liquidity bricks available")

    composite = sum(_cross_section_zscore(bricks[k]) for k in keys) / len(keys)
    return composite
