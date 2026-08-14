"""Alpha information space analysis: clustering, latent dimensions, residual IC tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from factor_taxonomy import FACTOR_TAXONOMY, mechanism_layer_for
from liquidity_normalization import panel_cross_sectional_residual


def dedupe_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    """One row per factor_name — keep highest production_score."""
    if ranking.empty:
        return ranking
    r = ranking.sort_values("production_score", ascending=False)
    return r.drop_duplicates(subset=["factor_name"], keep="first").reset_index(drop=True)


def stack_panel(wide: pd.DataFrame) -> pd.Series:
    s = wide.stack(dropna=True)
    s.index.names = ["date", "symbol"]
    return s


def correlation_matrix(
    panels: Dict[str, pd.DataFrame],
    min_overlap: int = 5000,
    sample_per_factor: int = 80_000,
) -> pd.DataFrame:
    """Pairwise correlation on subsampled stacked observations (memory-safe)."""
    names = list(panels.keys())
    sampled = {}
    for name, wide in panels.items():
        s = stack_panel(wide).dropna()
        if len(s) > sample_per_factor:
            s = s.sample(n=sample_per_factor, random_state=42)
        sampled[name] = s

    corr = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            aligned = pd.concat([sampled[a], sampled[b]], axis=1, join="inner").dropna()
            if len(aligned) < min_overlap:
                c = np.nan
            else:
                if len(aligned) > sample_per_factor:
                    aligned = aligned.sample(n=sample_per_factor, random_state=42)
                c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            corr.loc[a, b] = c
            corr.loc[b, a] = c
    return corr


def intrinsic_dimension(corr: pd.DataFrame, variance_threshold: float = 0.90) -> dict:
    """PCA-style count of eigenvalues needed to explain variance_threshold of total."""
    c = corr.astype(float).fillna(0.0).values
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 1.0)
    try:
        eigvals = np.sort(np.linalg.eigvalsh(c))[::-1]
    except np.linalg.LinAlgError:
        eigvals = np.sort(np.linalg.svd(c, compute_uv=False)[0])[::-1]
    eigvals = np.maximum(eigvals, 0)
    total = eigvals.sum()
    if total <= 0:
        return {"n_factors": len(corr), "intrinsic_dim_90": len(corr), "eigvals": eigvals.tolist()}
    cum = np.cumsum(eigvals) / total
    k = int(np.searchsorted(cum, variance_threshold) + 1)
    return {
        "n_factors": len(corr),
        "intrinsic_dim_90": k,
        "eigvals_top10": eigvals[:10].tolist(),
        "variance_explained_top10": (eigvals[:10] / total).tolist(),
    }


def hierarchical_cluster(corr: pd.DataFrame, distance_threshold: float = 0.35) -> pd.Series:
    """Cluster factors by |correlation| distance (1 - |rho|)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    c = corr.astype(float).copy()
    c = c.fillna(0.0)
    np.fill_diagonal(c.values, 1.0)

    dist = 1.0 - c.abs().values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    labels = fcluster(z, t=distance_threshold, criterion="distance")
    return pd.Series(labels, index=corr.index, name="cluster_id")


def dominant_family(factor_names: List[str]) -> str:
    layers = [mechanism_layer_for(n) for n in factor_names]
    if not layers:
        return "unknown"
    return pd.Series(layers).mode().iloc[0]


def cluster_summary(
    corr: pd.DataFrame,
    cluster_labels: pd.Series,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rank_idx = ranking.set_index("factor_name")
    for cid, members in cluster_labels.groupby(cluster_labels):
        names = members.index.tolist()
        sub_corr = corr.loc[names, names]
        mean_abs_corr = sub_corr.where(~np.eye(len(names), dtype=bool)).abs().mean().mean()
        rep = None
        rep_score = -np.inf
        for n in names:
            if n in rank_idx.index:
                sc = rank_idx.loc[n, "production_score"]
                if sc > rep_score:
                    rep_score = sc
                    rep = n
        if rep is None:
            rep = names[0]
        rows.append(
            {
                "cluster_id": int(cid),
                "n_members": len(names),
                "representative": rep,
                "representative_production_score": rep_score if rep_score > -np.inf else np.nan,
                "dominant_mechanism_layer": dominant_family(names),
                "mean_intra_cluster_abs_corr": float(mean_abs_corr),
                "members": "|".join(sorted(names)),
            }
        )
    return pd.DataFrame(rows).sort_values("n_members", ascending=False)


def mean_rank_ic(factor_wide: pd.DataFrame, ret_wide: pd.DataFrame, sample_dates: Optional[int] = None) -> float:
    aligned_f = factor_wide.reindex_like(ret_wide)
    aligned_r = ret_wide
    if sample_dates and len(aligned_f) > sample_dates:
        aligned_f = aligned_f.iloc[-sample_dates:]
        aligned_r = aligned_r.iloc[-sample_dates:]
    ic_daily = aligned_f.corrwith(aligned_r, axis=1, method="spearman")
    return float(ic_daily.mean())


def residual_ic_vs_anchor(
    factor_wide: pd.DataFrame,
    anchor_wide: pd.DataFrame,
    ret_wide: pd.DataFrame,
    sample_dates: int = 504,
) -> Tuple[float, float]:
    """Return (IC_raw, IC_residual) after CS orthogonalization to anchor."""
    ic_raw = mean_rank_ic(factor_wide, ret_wide, sample_dates)
    resid = panel_cross_sectional_residual(factor_wide, [anchor_wide])
    ic_resid = mean_rank_ic(resid, ret_wide, sample_dates)
    return ic_raw, ic_resid


def residual_ic_table(
    panels: Dict[str, pd.DataFrame],
    cluster_summary_df: pd.DataFrame,
    ret_wide: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, crow in cluster_summary_df.iterrows():
        rep = crow["representative"]
        if rep not in panels:
            continue
        anchor = panels[rep]
        members = crow["members"].split("|")
        # Representative row
        ic_raw_rep = mean_rank_ic(anchor, ret_wide, 504)
        rows.append(
            {
                "cluster_id": crow["cluster_id"],
                "representative": rep,
                "factor_name": rep,
                "is_representative": True,
                "ic_raw": ic_raw_rep,
                "ic_residual": np.nan,
                "incremental_ic": np.nan,
                "has_incremental_alpha": False,
            }
        )
        for name in members:
            if name == rep or name not in panels:
                continue
            ic_raw, ic_resid = residual_ic_vs_anchor(panels[name], anchor, ret_wide)
            rows.append(
                {
                    "cluster_id": crow["cluster_id"],
                    "representative": rep,
                    "factor_name": name,
                    "is_representative": False,
                    "ic_raw": ic_raw,
                    "ic_residual": ic_resid,
                    "incremental_ic": ic_resid,
                    "has_incremental_alpha": (
                        (abs(ic_resid) >= 0.005) and (abs(ic_resid) > 0.5 * abs(ic_raw))
                    ),
                }
            )
    return pd.DataFrame(rows)


def latent_dimension_verdict(intrinsic: dict, n_clusters: int, n_incremental: int) -> str:
    k = intrinsic.get("intrinsic_dim_90", n_clusters)
    return (
        f"OHLCV library: {intrinsic['n_factors']} factors → "
        f"~{k} PCA variance dims (90% var) [NOT alpha count], "
        f"{n_clusters} correlation clusters (latent measurement families), "
        f"{n_incremental} factors with incremental residual IC vs cluster rep. "
        f"Next: run_alpha_dimension_map.py for economic return-driver map."
    )
