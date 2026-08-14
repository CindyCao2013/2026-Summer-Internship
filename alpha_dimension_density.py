"""Cross-dimension density analysis — OOS split, orthogonal IC tables, stack λ-grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import (
    daily_rank_ic_series,
    evaluate_stack_signal,
    icir_from_daily,
)
from alpha_density_mining import analyze_density_candidate, mean_cross_sectional_corr, save_density_summary
from alpha_frozen_stack_v1 import FROZEN_OHLCV_REPS
from alpha_information_space import mean_rank_ic
from factor_attribution import align_signal, combine_equal_weight, cs_zscore
from liquidity_normalization import panel_cross_sectional_residual

DISCOVERY_DAYS = 504
MIN_RESIDUAL_T = 2.0
MIN_RESIDUAL_ICIR = 0.30
GOOD_CORR = 0.30
ACCEPT_CORR = 0.60

DIM_REP_FACTORS = {
    "D1": "low_vol_liquidity_quality_60d",
    "D2": "volatility_60d",
    "D3": "lower_shadow_support_20d",
    "D4": "winner_sentiment_reversal_5d",
    "D5": "upside_fragility_20d",
}

DEFAULT_STACK_LAMBDAS: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def split_discovery_confirmation(
    ret: pd.DataFrame,
    discovery_days: int = DISCOVERY_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological OOS split: first `discovery_days` vs remainder."""
    if len(ret) <= discovery_days:
        return ret, ret.iloc[0:0]
    return ret.iloc[:discovery_days], ret.iloc[discovery_days:]


def align_panel(panel: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    return panel.reindex(index=ret.index, columns=ret.columns)


def ic_series_stats(ic_daily: pd.Series) -> dict:
    s = ic_daily.dropna()
    n = len(s)
    if n < 20:
        return {"n_days": n, "residual_ic_mean": np.nan, "residual_icir": np.nan, "residual_ic_t": np.nan}
    mean_ic = float(s.mean())
    std_ic = float(s.std())
    icir = float(mean_ic / std_ic * np.sqrt(250)) if std_ic > 1e-12 else np.nan
    t_stat = float(mean_ic / std_ic * np.sqrt(n)) if std_ic > 1e-12 else np.nan
    return {
        "n_days": n,
        "residual_ic_mean": mean_ic,
        "residual_icir": icir,
        "residual_ic_t": t_stat,
    }


def rank_ic_series_corr(
    factor: pd.DataFrame,
    anchor: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    signal_shift: int = 1,
) -> float:
    """Pearson correlation of daily rank-IC time series."""
    ic_f = daily_rank_ic_series(factor, ret, signal_shift=signal_shift)
    ic_a = daily_rank_ic_series(anchor, ret, signal_shift=signal_shift)
    aligned = pd.concat([ic_f, ic_a], axis=1, keys=["f", "a"]).dropna()
    if len(aligned) < 20:
        return np.nan
    return float(aligned["f"].corr(aligned["a"]))


def residual_ic_stats(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    signal_shift: int = 1,
) -> dict:
    f = align_signal(factor, signal_shift)
    a = align_signal(anchor.reindex_like(factor), signal_shift)
    resid = panel_cross_sectional_residual(f, [a])
    ic_daily = daily_rank_ic_series(resid, ret, signal_shift=signal_shift)
    return ic_series_stats(ic_daily)


def classify_cross_dim_independence(row: pd.Series) -> str:
    t = row.get("residual_ic_t", np.nan)
    icir = row.get("residual_icir", np.nan)
    corr = abs(row.get("rank_ic_corr", np.nan))
    if pd.isna(t):
        return "insufficient_data"
    if corr >= ACCEPT_CORR and (pd.isna(t) or abs(t) < MIN_RESIDUAL_T):
        return "redundant"
    if abs(t) >= MIN_RESIDUAL_T and pd.notna(icir) and abs(icir) >= MIN_RESIDUAL_ICIR:
        return "independent"
    if abs(t) >= MIN_RESIDUAL_T:
        return "partial_independent"
    return "absorbed"


def cross_dimension_independence_table(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    anchors: Dict[str, pd.DataFrame],
    *,
    period: str,
    factor_name: str = "",
) -> pd.DataFrame:
    rows = []
    f = align_panel(factor, ret)
    for anchor_name, anchor in anchors.items():
        a = align_panel(anchor, ret)
        panel_corr = mean_cross_sectional_corr(f, a, sample_dates=None)
        ic_corr = rank_ic_series_corr(f, a, ret)
        resid = residual_ic_stats(f, ret, a)
        row = {
            "factor_name": factor_name,
            "period": period,
            "orthogonalize_vs": anchor_name,
            "panel_spearman_corr": panel_corr,
            "rank_ic_corr": ic_corr,
            **resid,
        }
        row["independence_verdict"] = classify_cross_dim_independence(pd.Series(row))
        rows.append(row)
    return pd.DataFrame(rows)


def build_tri_base_stack(frozen_panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight D1 + D4 + D5 (template partial base for D2/D3 stack tests)."""
    names = [DIM_REP_FACTORS["D1"], DIM_REP_FACTORS["D4"], DIM_REP_FACTORS["D5"]]
    parts = [frozen_panels[n] for n in names]
    return combine_equal_weight(parts)


def build_quad_base_without(frozen_panels: Dict[str, pd.DataFrame], exclude_dim: str) -> pd.DataFrame:
    """Four-dim base excluding one OHLCV dimension rep."""
    exclude_factor = DIM_REP_FACTORS[exclude_dim]
    parts = []
    for spec in FROZEN_OHLCV_REPS:
        if spec["factor"] == exclude_factor:
            continue
        parts.append(frozen_panels[spec["factor"]])
    return combine_equal_weight(parts)


def build_dim_lambda_stack(
    base: pd.DataFrame,
    dim_rep: pd.DataFrame,
    lam: float,
    *,
    signal_shift: int = 1,
) -> pd.DataFrame:
    """Blend cs_z(base) with cs_z(dim_rep): (1-λ)·base + λ·dim."""
    if lam <= 0:
        return cs_zscore(align_signal(base, signal_shift))
    if lam >= 1:
        return cs_zscore(align_signal(dim_rep, signal_shift))
    b = cs_zscore(align_signal(base, signal_shift))
    d = cs_zscore(align_signal(dim_rep, signal_shift))
    return cs_zscore((1 - lam) * b + lam * d)


def evaluate_dim_lambda_grid(
    base: pd.DataFrame,
    dim_rep: pd.DataFrame,
    ret: pd.DataFrame,
    lambdas: Sequence[float] = DEFAULT_STACK_LAMBDAS,
    *,
    period: str = "full",
    dim_label: str = "",
) -> pd.DataFrame:
    baseline = evaluate_stack_signal(build_dim_lambda_stack(base, dim_rep, 0.0), ret)
    rows = []
    for lam in lambdas:
        signal = build_dim_lambda_stack(base, dim_rep, lam)
        metrics = evaluate_stack_signal(signal, ret)
        rows.append(
            {
                "period": period,
                "dim": dim_label,
                "lambda": lam,
                **metrics,
                "rank_ic_delta": metrics["rank_ic"] - baseline["rank_ic"],
                "icir_delta": metrics["icir"] - baseline["icir"],
                "hl_sharpe_delta": metrics["hl_sharpe"] - baseline["hl_sharpe"],
                "monotonicity_delta": metrics["monotonicity_score"] - baseline["monotonicity_score"],
            }
        )
    return pd.DataFrame(rows)


def classify_stack_uplift(
    grid_df: pd.DataFrame,
    *,
    sharpe_threshold: float = 0.10,
    uplift_lambdas: Sequence[float] = (0.2, 0.4, 0.6),
) -> dict:
    base = grid_df[grid_df["lambda"] == 0.0]
    if base.empty:
        return {"stack_incremental": False, "recommendation": "insufficient_data"}
    base_sharpe = float(base["hl_sharpe"].iloc[0])
    zone = grid_df[grid_df["lambda"].isin(uplift_lambdas)]
    if zone.empty:
        return {"stack_incremental": False, "recommendation": "no_lambda_zone"}
    sharpe_uplift = (zone["hl_sharpe_delta"] >= sharpe_threshold).sum() >= 2
    stable_ic = (zone["rank_ic_delta"] > 0).mean() >= 0.5
    if sharpe_uplift and stable_ic:
        rec = "confirm_base_slot"
    elif stable_ic:
        rec = "conditional_enhancer"
    else:
        rec = "drop_or_satellite"
    return {
        "stack_incremental": bool(sharpe_uplift),
        "ic_uplift_majority": bool(stable_ic),
        "base_hl_sharpe": base_sharpe,
        "max_hl_sharpe_delta": float(zone["hl_sharpe_delta"].max()),
        "recommendation": rec,
    }


def conditional_ic_by_anchor_tercile(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    anchor_label: str = "anchor",
    signal_shift: int = 1,
) -> pd.DataFrame:
    """Mean rank IC of factor within anchor tercile groups (cross-sectional each day)."""
    f = align_signal(factor, signal_shift)
    a = align_signal(anchor.reindex_like(factor), signal_shift)
    aligned_r = ret.reindex_like(f)
    bucket_ics: Dict[str, List[float]] = {"bottom": [], "mid": [], "top": []}

    for dt in f.index:
        row_f = f.loc[dt]
        row_a = a.loc[dt]
        row_r = aligned_r.loc[dt]
        mask = row_f.notna() & row_a.notna() & row_r.notna()
        if mask.sum() < 90:
            continue
        ranks = row_a[mask].rank(pct=True)
        rets = row_r[mask]
        sigs = row_f[mask]
        for label, lo, hi in [("bottom", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("top", 2 / 3, 1.01)]:
            sel = (ranks >= lo) & (ranks < hi) if hi < 1.01 else (ranks >= lo)
            if sel.sum() < 20:
                continue
            ic = sigs[sel].corr(rets[sel], method="spearman")
            if pd.notna(ic):
                bucket_ics[label].append(float(ic))

    rows = []
    for bucket, vals in bucket_ics.items():
        rows.append(
            {
                "anchor": anchor_label,
                "tercile": bucket,
                "mean_ic": float(np.mean(vals)) if vals else np.nan,
                "n_days": len(vals),
            }
        )
    return pd.DataFrame(rows)


def analyze_candidate_on_period(
    factor_name: str,
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    dim_rep: pd.DataFrame,
    exposure_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    frozen_list: List[pd.DataFrame],
    *,
    period: str,
    hypothesis: str = "",
    dim_label: str,
    anchor_key: str,
    stack_enhancement_pass: Optional[bool] = None,
    monotonicity: Optional[float] = None,
) -> dict:
    row = analyze_density_candidate(
        factor_name,
        align_panel(panel, ret),
        ret,
        align_panel(dim_rep, ret),
        {k: align_panel(v, ret) for k, v in exposure_panels.items()},
        frozen_panels,
        frozen_list,
        hypothesis=hypothesis,
        sample_dates=None,
        stack_enhancement_pass=stack_enhancement_pass,
        monotonicity=monotonicity,
        anchor_key=anchor_key,
        dim_label=dim_label,
    )
    row["period"] = period
    return row


def summarize_dimension_verdict(
    dim: str,
    rep: str,
    density_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    stack_disc: pd.DataFrame,
    stack_conf: pd.DataFrame,
    *,
    extra: Optional[dict] = None,
) -> dict:
    disc = density_df[density_df["period"] == "discovery"] if "period" in density_df.columns else density_df
    rep_disc = disc[disc["factor_name"] == rep]
    independent = disc[disc["density_classification"] == "independent_alpha"]["factor_name"].tolist()
    redundant = disc[disc["density_classification"].isin(["redundant", "amplification_artifact"])]["factor_name"].tolist()

    cross_rep = cross_df[cross_df["factor_name"] == rep] if not cross_df.empty else pd.DataFrame()
    cross_disc = cross_rep[cross_rep["period"] == "discovery"] if "period" in cross_rep.columns else cross_rep
    indep_vs = cross_disc[cross_disc["independence_verdict"] == "independent"]["orthogonalize_vs"].tolist()
    absorbed_vs = cross_disc[cross_disc["independence_verdict"].isin(["absorbed", "redundant"])]["orthogonalize_vs"].tolist()

    stack_d = classify_stack_uplift(stack_disc) if not stack_disc.empty else {}
    stack_c = classify_stack_uplift(stack_conf) if not stack_conf.empty else {}

    role = "base"
    if rep in redundant or (stack_d.get("recommendation") == "drop_or_satellite" and not stack_c.get("stack_incremental")):
        role = "drop"
    elif not stack_d.get("stack_incremental") and stack_c.get("stack_incremental"):
        role = "conditional_base"
    elif stack_d.get("recommendation") == "conditional_enhancer":
        role = "enhancer"
    elif len(indep_vs) >= 3 and stack_d.get("stack_incremental"):
        role = "base"
    elif len(absorbed_vs) >= 2:
        role = "satellite"

    verdict = {
        "dimension": dim,
        "representative": rep,
        "research_status": "density_v1",
        "discovery_days": DISCOVERY_DAYS,
        "internal_independent": [x for x in independent if x != rep],
        "internal_redundant": redundant,
        "cross_dim_independent_vs": indep_vs,
        "cross_dim_absorbed_vs": absorbed_vs,
        "stack_discovery": stack_d,
        "stack_confirmation": stack_c,
        "proposed_role": role,
        "rep_discovery_ic_raw": float(rep_disc["ic_raw"].iloc[0]) if not rep_disc.empty else np.nan,
        "rep_discovery_ic_after_dim": float(rep_disc.iloc[0].get(f"ic_after_{dim.lower()}", np.nan)) if not rep_disc.empty else np.nan,
    }
    if extra:
        verdict.update(extra)
    return verdict


def publish_dimension_density(
    out_dir: Path,
    *,
    prefix: str,
    density_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    stack_df: pd.DataFrame,
    verdict: dict,
    conditional_df: Optional[pd.DataFrame] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_density_summary(density_df, out_dir, filename=f"{prefix}_density_summary.csv")
    cross_df.to_csv(out_dir / f"{prefix}_cross_dimension_independence.csv", index=False)
    stack_df.to_csv(out_dir / f"{prefix}_stack_lambda_grid.csv", index=False)
    if conditional_df is not None and not conditional_df.empty:
        conditional_df.to_csv(out_dir / f"{prefix}_conditional_ic.csv", index=False)
    (out_dir / f"{prefix}_density_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
