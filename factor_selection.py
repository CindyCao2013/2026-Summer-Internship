"""Alpha Selection Engine v1: rank factors by ICIR, stability, cross-universe consistency."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from factor_taxonomy import FACTOR_TAXONOMY


def load_batch_summaries(result_root: str, batch_tag: str = "core") -> pd.DataFrame:
    path = Path(result_root) / f"batch_summary_{batch_tag}.csv"
    if not path.exists():
        alt = Path(result_root) / "batch_summary_c2c_all_factors.csv"
        if alt.exists():
            return pd.read_csv(alt)
        raise FileNotFoundError(f"No batch summary at {path}")
    return pd.read_csv(path)


def cross_universe_consistency(df: pd.DataFrame) -> pd.Series:
    """Fraction of universes where |IC| >= 2% or |ICIR| >= 0.7."""
    if "factor_name" not in df.columns:
        return pd.Series(dtype=float)

    scores = []
    for fname, grp in df.groupby("factor_name"):
        hit = (
            (grp["abs_rank_ic_mean"] >= 0.02) | (grp["abs_icir"] >= 0.7)
        ).mean()
        sign_consistency = grp["rank_ic_mean"].apply(np.sign).nunique() == 1
        score = hit * (1.0 if sign_consistency else 0.5)
        scores.append((fname, score))
    return pd.Series(dict(scores), name="cross_universe_score")


def compute_alpha_scores(
    summary_df: pd.DataFrame,
    icir_weight: float = 0.4,
    stability_weight: float = 0.2,
    universe_weight: float = 0.2,
    sharpe_weight: float = 0.2,
) -> pd.DataFrame:
    """Aggregate per-factor scores across universes (mean metrics + consistency)."""
    agg = summary_df.groupby("factor_name").agg(
        icir_mean=("icir", "mean"),
        abs_icir_mean=("abs_icir", "mean"),
        rank_ic_mean=("rank_ic_mean", "mean"),
        abs_rank_ic_mean=("abs_rank_ic_mean", "mean"),
        hl_sharpe_mean=("hl_sharpe", "mean"),
        hl_mdd_mean=("hl_mdd", "mean"),
        hl_turnover_mean=("hl_avg_turnover", "mean"),
        n_universes=("universe", "count"),
    )
    consistency = cross_universe_consistency(summary_df)
    agg = agg.join(consistency, how="left")
    agg["cross_universe_score"] = agg["cross_universe_score"].fillna(0)

    # stability proxy: lower turnover variance + moderate sharpe
    agg["stability_score"] = (
        agg["hl_sharpe_mean"].clip(lower=0) / (agg["hl_turnover_mean"] + 0.1)
    ).rank(pct=True)

    agg["composite_score"] = (
        icir_weight * agg["abs_icir_mean"].rank(pct=True)
        + stability_weight * agg["stability_score"]
        + universe_weight * agg["cross_universe_score"]
        + sharpe_weight * agg["hl_sharpe_mean"].clip(lower=0).rank(pct=True)
    )

    if "factor_name" in agg.index.names or True:
        agg = agg.reset_index()
        families = agg["factor_name"].map(
            lambda x: FACTOR_TAXONOMY.get(x, {}).get("family", "unknown")
        )
        agg["family"] = families

    return agg.sort_values("composite_score", ascending=False)


def select_top_factors(
    scored_df: pd.DataFrame,
    top_k: int = 10,
    min_abs_ic: float = 0.02,
    min_abs_icir: float = 0.7,
    min_universes: int = 2,
) -> pd.DataFrame:
    mask = (
        ((scored_df["abs_rank_ic_mean"] >= min_abs_ic) | (scored_df["abs_icir_mean"] >= min_abs_icir))
        & (scored_df["n_universes"] >= min_universes)
    )
    filtered = scored_df[mask].head(top_k)
    return filtered


def prune_redundant_factors(
    factor_values: dict,
    selected_names: List[str],
    corr_threshold: float = 0.7,
) -> List[str]:
    """Greedy correlation pruning on wide factor matrices (Date x Stock)."""
    if len(selected_names) <= 1:
        return selected_names

    kept: List[str] = []
    for name in selected_names:
        if name not in factor_values:
            continue
        f = factor_values[name].stack().dropna()
        redundant = False
        for k in kept:
            g = factor_values[k].stack().dropna()
            aligned = pd.concat([f, g], axis=1, join="inner").dropna()
            if len(aligned) < 100:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if abs(corr) >= corr_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(name)
    return kept


def run_selection_report(
    result_root: str,
    batch_tag: str = "core",
    output_path: Optional[str] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    summary = load_batch_summaries(result_root, batch_tag)
    scored = compute_alpha_scores(summary)
    top = select_top_factors(scored, top_k=top_k)

    out = Path(output_path or Path(result_root) / f"alpha_selection_{batch_tag}.csv")
    scored.to_csv(out.with_name(f"alpha_scores_{batch_tag}.csv"), index=False)
    top.to_csv(out, index=False)
    print(f"Saved alpha scores -> {out.with_name(f'alpha_scores_{batch_tag}.csv')}")
    print(f"Saved top-{top_k} selection -> {out}")
    return top
