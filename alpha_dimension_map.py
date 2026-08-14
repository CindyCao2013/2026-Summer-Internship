"""Alpha Dimension Map — economic return drivers vs statistical variance dimensions.

Key distinction (per research methodology):
  - PCA / correlation clusters → **variance dimensions** (how many ways OHLCV measures co-move)
  - Orthogonal residual IC     → **return-predictive dimensions** (independent alpha drivers)

Clustering/pruning purpose:
  Confirm how many *independent return drivers* each factor family contributes,
  not simply delete weak factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from alpha_information_space import mean_rank_ic
from factor_taxonomy import FACTOR_TAXONOMY, mechanism_layer_for
from liquidity_normalization import panel_cross_sectional_residual


# ---------------------------------------------------------------------------
# OHLCV Phase-1: frozen production dimensions (economic drivers, not PCA axes)
# ---------------------------------------------------------------------------
@dataclass
class EconomicDimensionSpec:
    dimension_id: str
    name: str
    economic_meaning: str
    representative: str
    information_source: str  # ohlcv_frozen | l2 | cn_behavior | fundamental
    missing_information: str
    next_phase: str
    cluster_keywords: List[str] = field(default_factory=list)
    supporting_candidates: List[str] = field(default_factory=list)


OHLCV_PRODUCTION_DIMENSIONS: List[EconomicDimensionSpec] = [
    EconomicDimensionSpec(
        dimension_id="D1",
        name="Stable Liquidity Quality",
        economic_meaning=(
            "Low-volatility stocks with stable, size-adjusted trading activity; "
            "liquidity × risk-state interaction (not raw turnover level)."
        ),
        representative="low_vol_liquidity_quality_60d",
        information_source="ohlcv_frozen",
        missing_information="L2 order flow direction (VOI/OIR), bid-ask pressure",
        next_phase="L2 microstructure",
        cluster_keywords=["liquidity", "stability", "amount", "turnover", "amihud"],
        supporting_candidates=[
            "low_vol_liquidity_quality_20d",
            "liquidity_stability_20d",
            "amount_stability_20d",
            "volume_stability_20d",
            "composite_liquidity_stability_20d",
            "stability_quality_composite_20d",
            "relative_liquidity_strength_20d",
            "turnover_stability_20d",
            "amount_per_mktcap_stability_20d",
        ],
    ),
    EconomicDimensionSpec(
        dimension_id="D2",
        name="Low Volatility / Risk",
        economic_meaning=(
            "Realized return dispersion and range — low-risk anomaly; "
            "one risk dimension (do not keep volatility_20d + volatility_60d + high_low separately)."
        ),
        representative="volatility_60d",
        information_source="ohlcv_frozen",
        missing_information="Downside semivariance, skew/kurtosis, options-implied vol",
        next_phase="tail refinement + options",
        cluster_keywords=["volatility", "risk", "high_low", "range"],
        supporting_candidates=[
            "volatility_20d",
            "volatility_level_20d",
            "high_low_60d",
            "high_low_20d",
            "max_daily_return_20d",
            "range_contraction_20d",
        ],
    ),
    EconomicDimensionSpec(
        dimension_id="D3",
        name="Price Path Asymmetry",
        economic_meaning=(
            "Candle/shadow geometry — lower-shadow support vs upper-shadow pressure; "
            "intraday path asymmetry not captured by close-to-close momentum."
        ),
        representative="lower_shadow_support_20d",
        information_source="ohlcv_frozen",
        missing_information="Intraday path (VWAP deviation, open-drive, close-drive)",
        next_phase="intraday path + L2",
        cluster_keywords=["shadow", "candle", "path", "upper", "lower"],
        supporting_candidates=[
            "upper_shadow_pressure_20d",
            "range_contraction_20d",
            "high_low_20d",
            "volume_price_efficiency_20d",
        ],
    ),
    EconomicDimensionSpec(
        dimension_id="D4",
        name="Behavioral Reversal",
        economic_meaning=(
            "Retail/sentiment overreaction — winners/losers, attention spikes, "
            "Amihud shock reversal; A-share behavioral anomaly."
        ),
        representative="winner_sentiment_reversal_5d",
        information_source="ohlcv_frozen",
        missing_information="Chase/herding (order flow), sentiment/news, limit-up emotion",
        next_phase="CN behavior + L2",
        cluster_keywords=["reversal", "sentiment", "attention", "behavior", "winner", "amihud"],
        supporting_candidates=[
            "amihud_shock_reversal_5d",
            "low_attention_reversal_20d",
            "overheated_turnover_proxy_20d",
            "liquidity_shock_20d",
            "stable_reversal_blend_20d",
            "reversal_20d",
        ],
    ),
    EconomicDimensionSpec(
        dimension_id="D5",
        name="Tail Fragility",
        economic_meaning=(
            "Upside semivariance dominance / asymmetric tail — fragility distinct from "
            "level volatility (measures shape, not scale)."
        ),
        representative="upside_fragility_20d",
        information_source="ohlcv_frozen",
        missing_information="Full tail decomposition, jump detection, limit-down asymmetry",
        next_phase="tail + A-share limit structure",
        cluster_keywords=["tail", "fragility", "asymmetric", "semivar"],
        supporting_candidates=[
            "asymmetric_tail_ratio_20d",
            "tail_risk_min_return_20d",
            "drawup_drawdown_ratio_20d",
        ],
    ),
]


# Phase 2/3: planned new information sources (not yet in OHLCV manifold)
EXPANSION_ROADMAP: List[dict] = [
    {
        "phase": "Phase 2",
        "dimension_id": "L1",
        "name": "Order Flow Imbalance",
        "economic_meaning": "Active buy vs sell pressure (VOI, OIR, MPB)",
        "target_factors": "cn_voi_20d, cn_oir_20d, cn_mpb_20d",
        "information_source": "l2",
        "missing_information": "Tick/order-book data pipeline",
        "reference": "中信建投 高频量价 VOI/OIR/MPB",
    },
    {
        "phase": "Phase 2",
        "dimension_id": "L2",
        "name": "Trade Direction / Price Impact",
        "economic_meaning": "Who initiates trades; temporary vs permanent impact",
        "target_factors": "cn_order_imbalance_20d, price_impact_20d",
        "information_source": "l2",
        "missing_information": "L2 trade direction labels",
        "reference": "中信建投 盘口降频",
    },
    {
        "phase": "Phase 3",
        "dimension_id": "C1",
        "name": "Limit-Up / Board Structure",
        "economic_meaning": "A-share sentiment, 游资, 连板 momentum",
        "target_factors": "cn_limit_up_strength_20d, board_momentum_20d",
        "information_source": "cn_behavior",
        "missing_information": "Limit-up flag database (currently EOD proxy)",
        "reference": "A股涨停行为研究",
    },
    {
        "phase": "Phase 3",
        "dimension_id": "C2",
        "name": "Chase / Herding",
        "economic_meaning": "Retail chase (corr ret,volume), sync order flow",
        "target_factors": "cn_chase_behavior_20d, cn_herding_proxy_20d",
        "information_source": "cn_behavior",
        "missing_information": "Intraday or tick-level for full fidelity",
        "reference": "开源/国盛金工 追涨杀跌/羊群",
    },
    {
        "phase": "Phase 3",
        "dimension_id": "C3",
        "name": "Turnover Structure",
        "economic_meaning": "Turnover percentile, acceleration, concentration",
        "target_factors": "cn_turnover_percentile_20d, cn_turnover_concentration_20d",
        "information_source": "cn_behavior",
        "missing_information": "Already in eod_cn_broker_v1 — validate incremental IC vs D1",
        "reference": "中信/华泰 换手率结构",
    },
    {
        "phase": "Phase 4",
        "dimension_id": "F1",
        "name": "Value",
        "economic_meaning": "PE/PB/PS relative to industry — pricing distortion",
        "target_factors": "pe_cs_zscore, pb_industry_neutral",
        "information_source": "fundamental",
        "missing_information": "Fundamental data + industry classification",
        "reference": "中信证券 多因子体系 Value",
    },
    {
        "phase": "Phase 4",
        "dimension_id": "F2",
        "name": "Quality / Growth",
        "economic_meaning": "Profitability, leverage, growth — orthogonal to price",
        "target_factors": "roe_stability, revenue_growth_cs",
        "information_source": "fundamental",
        "missing_information": "Financial statement pipeline",
        "reference": "中信证券 多因子体系 Quality+Growth",
    },
]


def _factor_family(name: str) -> str:
    meta = FACTOR_TAXONOMY.get(name)
    if meta:
        return meta.get("family", "unknown")
    return mechanism_layer_for(name)


def assign_factor_to_dimension(factor_name: str) -> Optional[str]:
    """Map a factor to the best-matching production dimension (or None)."""
    fn = factor_name.lower()
    best_dim = None
    best_score = 0
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        if factor_name == spec.representative or factor_name in spec.supporting_candidates:
            return spec.dimension_id
        score = sum(1 for kw in spec.cluster_keywords if kw in fn)
        if score > best_score:
            best_score = score
            best_dim = spec.dimension_id
    return best_dim if best_score > 0 else None


def map_clusters_to_dimensions(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Attach economic dimension label to each correlation cluster."""
    rows = []
    for _, crow in clusters_df.iterrows():
        rep = crow["representative"]
        members = crow["members"].split("|") if isinstance(crow["members"], str) else []
        dim_id = assign_factor_to_dimension(rep)
        spec = next((s for s in OHLCV_PRODUCTION_DIMENSIONS if s.dimension_id == dim_id), None)
        rows.append(
            {
                "cluster_id": crow["cluster_id"],
                "n_members": crow["n_members"],
                "representative": rep,
                "dominant_mechanism_layer": crow.get("dominant_mechanism_layer"),
                "mean_intra_cluster_abs_corr": crow.get("mean_intra_cluster_abs_corr"),
                "economic_dimension_id": dim_id,
                "economic_dimension_name": spec.name if spec else "unmapped",
                "members": crow["members"],
                "interpretation": (
                    f"Not {crow['n_members']} independent alphas — "
                    f"noisy measurements of latent driver '{spec.name if spec else 'unknown'}'"
                    if crow["n_members"] > 1
                    else "Singleton — candidate standalone return driver"
                ),
            }
        )
    return pd.DataFrame(rows)


def orthogonal_return_predictive_dimensions(
    rep_panels: Dict[str, pd.DataFrame],
    ret_wide: pd.DataFrame,
    min_abs_ic: float = 0.008,
    sample_dates: int = 504,
) -> pd.DataFrame:
    """
    Greedy sequential orthogonalization: count dimensions that retain |IC| after
    removing all prior accepted drivers.  This is return-predictive, not variance.
    """
    ic_raw = {n: mean_rank_ic(w, ret_wide, sample_dates) for n, w in rep_panels.items()}
    order = sorted(rep_panels.keys(), key=lambda n: abs(ic_raw[n]), reverse=True)

    accepted: List[str] = []
    rows = []
    for name in order:
        if not accepted:
            resid_panel = rep_panels[name]
            ic_resid = ic_raw[name]
        else:
            anchors = [rep_panels[a] for a in accepted]
            resid_panel = panel_cross_sectional_residual(rep_panels[name], anchors)
            ic_resid = mean_rank_ic(resid_panel, ret_wide, sample_dates)

        is_driver = abs(ic_resid) >= min_abs_ic
        if is_driver:
            accepted.append(name)

        rows.append(
            {
                "factor_name": name,
                "ic_raw": ic_raw[name],
                "ic_orthogonal": ic_resid,
                "accepted_as_return_driver": is_driver,
                "orthogonal_rank": len(accepted) if is_driver else np.nan,
                "orthogonalized_against": "|".join(accepted[:-1]) if is_driver and len(accepted) > 1 else (
                    "" if not is_driver else ""
                ),
            }
        )

    out = pd.DataFrame(rows)
    # Fix orthogonalized_against column
    accepted_so_far: List[str] = []
    against = []
    for _, r in out.iterrows():
        if r["accepted_as_return_driver"]:
            against.append("|".join(accepted_so_far))
            accepted_so_far.append(r["factor_name"])
        else:
            against.append("|".join(accepted_so_far))
    out["orthogonalized_against"] = against
    return out


def cross_dimension_incremental_ic(
    rep_panels: Dict[str, pd.DataFrame],
    ret_wide: pd.DataFrame,
    sample_dates: int = 504,
) -> Dict[str, float]:
    """Each rep's IC after orthogonalizing against all OTHER production reps."""
    names = list(rep_panels.keys())
    out = {}
    for name in names:
        others = [rep_panels[n] for n in names if n != name]
        if not others:
            out[name] = mean_rank_ic(rep_panels[name], ret_wide, sample_dates)
        else:
            resid = panel_cross_sectional_residual(rep_panels[name], others)
            out[name] = mean_rank_ic(resid, ret_wide, sample_dates)
    return out


def build_dimension_map_v1(
    clusters_df: pd.DataFrame,
    residual_ic_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    intrinsic_summary: Optional[pd.DataFrame] = None,
    rep_panels: Optional[Dict[str, pd.DataFrame]] = None,
    ret_wide: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      dimension_map      — one row per economic dimension
      variance_vs_return — PCA vs return-predictive comparison
      cluster_assignment — clusters mapped to economic dims
    """
    rank_idx = ranking_df.drop_duplicates("factor_name", keep="first").set_index("factor_name")
    cluster_map = map_clusters_to_dimensions(clusters_df)

    # Supporting factors: from clusters + explicit candidates present in ranking
    all_ranked = set(rank_idx.index)
    incremental_by_factor = {}
    if len(residual_ic_df):
        for _, r in residual_ic_df.iterrows():
            incremental_by_factor[r["factor_name"]] = {
                "ic_raw": r.get("ic_raw"),
                "ic_residual_vs_cluster_rep": r.get("ic_residual"),
                "has_incremental_alpha": r.get("has_incremental_alpha", False),
            }

    cross_dim_ic: Dict[str, float] = {}
    ortho_table = pd.DataFrame()
    if rep_panels and ret_wide is not None:
        cross_dim_ic = cross_dimension_incremental_ic(rep_panels, ret_wide)
        ortho_table = orthogonal_return_predictive_dimensions(rep_panels, ret_wide)

    dim_rows = []
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        # Clusters assigned to this dimension
        assigned = cluster_map[cluster_map["economic_dimension_id"] == spec.dimension_id]
        cluster_members: List[str] = []
        for members_str in assigned["members"].dropna():
            cluster_members.extend(members_str.split("|"))
        cluster_members = sorted(set(cluster_members))

        supporting = sorted(
            set(spec.supporting_candidates) & all_ranked | set(cluster_members)
        )
        supporting = [s for s in supporting if s != spec.representative]

        rep_stats = rank_idx.loc[spec.representative] if spec.representative in rank_idx.index else None
        rep_ic_raw = incremental_by_factor.get(spec.representative, {}).get("ic_raw")
        if rep_ic_raw is None and rep_stats is not None:
            rep_ic_raw = rep_stats.get("mean_ic")

        n_intra_incremental = sum(
            1
            for s in supporting
            if incremental_by_factor.get(s, {}).get("has_incremental_alpha")
        )

        dim_rows.append(
            {
                "dimension_id": spec.dimension_id,
                "dimension_name": spec.name,
                "economic_meaning": spec.economic_meaning,
                "information_source": spec.information_source,
                "status": "frozen_production",
                "representative_factor": spec.representative,
                "supporting_factors": "|".join(supporting),
                "n_supporting": len(supporting),
                "n_correlation_clusters": len(assigned),
                "n_cluster_members_total": len(cluster_members),
                "n_supporting_with_incremental_ic": n_intra_incremental,
                "representative_ic_raw": rep_ic_raw,
                "representative_ic_cross_dimension": cross_dim_ic.get(spec.representative),
                "representative_production_score": (
                    rep_stats["production_score"] if rep_stats is not None else np.nan
                ),
                "representative_universe_stability": (
                    rep_stats["universe_stability"] if rep_stats is not None else np.nan
                ),
                "missing_information": spec.missing_information,
                "next_expansion_phase": spec.next_phase,
                "interpretation": (
                    f"{len(cluster_members)} correlated measurements → 1 return driver; "
                    f"keep `{spec.representative}` only"
                    if len(cluster_members) > 1
                    else f"Singleton driver — `{spec.representative}`"
                ),
            }
        )

    dimension_map = pd.DataFrame(dim_rows)

    # Variance vs return-predictive summary
    n_pca = np.nan
    n_clusters = len(clusters_df)
    n_factors = np.nan
    if intrinsic_summary is not None and len(intrinsic_summary):
        n_pca = intrinsic_summary.iloc[0].get("intrinsic_dim_90", np.nan)
        n_factors = intrinsic_summary.iloc[0].get("n_factors", np.nan)

    n_return_drivers = int(ortho_table["accepted_as_return_driver"].sum()) if len(ortho_table) else len(
        OHLCV_PRODUCTION_DIMENSIONS
    )

    variance_vs_return = pd.DataFrame(
        [
            {
                "metric": "input_factors_analyzed",
                "value": n_factors,
                "type": "input",
                "meaning": "High-quality candidates in information-space run",
            },
            {
                "metric": "pca_variance_dims_90pct",
                "value": n_pca,
                "type": "variance",
                "meaning": "Statistical dimensions (co-movement / redundancy) — NOT alpha count",
            },
            {
                "metric": "correlation_clusters",
                "value": n_clusters,
                "type": "variance",
                "meaning": "Correlation groups — each group ≈ one latent measurement family",
            },
            {
                "metric": "economic_return_drivers_ohlcv",
                "value": len(OHLCV_PRODUCTION_DIMENSIONS),
                "type": "economic",
                "meaning": "Human-defined latent drivers from cluster + IC validation",
            },
            {
                "metric": "orthogonal_return_predictive_dims",
                "value": n_return_drivers,
                "type": "return_predictive",
                "meaning": "Reps retaining |IC| after sequential orthogonalization",
            },
        ]
    )

    expansion = pd.DataFrame(EXPANSION_ROADMAP)
    return dimension_map, variance_vs_return, cluster_map, ortho_table, expansion


def format_methodology_verdict(variance_vs_return: pd.DataFrame) -> str:
    v = variance_vs_return.set_index("metric")["value"]
    return (
        f"OHLCV exploration complete: {v.get('input_factors_analyzed', '?')} factors → "
        f"~{v.get('pca_variance_dims_90pct', '?')} variance dims (PCA) ≠ "
        f"{v.get('orthogonal_return_predictive_dims', '?')} return-predictive dims. "
        f"Freeze {v.get('economic_return_drivers_ohlcv', 5)} OHLCV economic drivers; "
        f"expand via L2 + CN behavior + fundamental (see expansion roadmap)."
    )
