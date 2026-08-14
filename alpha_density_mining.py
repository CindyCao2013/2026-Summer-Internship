"""Alpha Density Mining — measure marginal independent signal within a dimension family."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_information_space import mean_rank_ic
from alpha_research_report import monotonicity_score
from factor_attribution import (
    MAX_IC_AMPLIFICATION_RATIO,
    align_signal,
    passes_strict_incremental,
    sequential_neutral_ics,
)
from liquidity_normalization import panel_cross_sectional_residual

MIN_INDEPENDENT_IC = 0.015
MIN_RESIDUAL_WEAK = 0.005
REDUNDANT_CORR = 0.70


def mean_cross_sectional_corr(a: pd.DataFrame, b: pd.DataFrame, sample_dates: int = 504) -> float:
    """Average Spearman correlation between two wide panels."""
    fa = a.iloc[-sample_dates:] if sample_dates and len(a) > sample_dates else a
    fb = b.reindex(index=fa.index, columns=fa.columns)
    corrs = []
    for dt in fa.index:
        x = fa.loc[dt]
        y = fb.loc[dt]
        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue
        c = x[mask].corr(y[mask], method="spearman")
        if pd.notna(c):
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else np.nan


def ic_after_anchor(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    signal_shift: int = 1,
    sample_dates: int = 504,
) -> float:
    f = factor.iloc[-sample_dates:] if sample_dates and len(factor) > sample_dates else factor
    r = ret.reindex(f.index)
    a = anchor.reindex(index=f.index, columns=f.columns)
    signal = align_signal(f, signal_shift)
    anchor_signal = align_signal(a, signal_shift)
    resid = panel_cross_sectional_residual(signal, [anchor_signal])
    return mean_rank_ic(resid, r, sample_dates=None)


def classify_density_signal(row: pd.Series, *, anchor_key: str = "ic_after_anchor") -> str:
    ic_raw = row.get("ic_raw", np.nan)
    ic_anchor = row.get(anchor_key, np.nan)
    ic_stack = row.get("ic_after_ohlcv_stack", np.nan)
    corr = row.get("corr_rep", row.get("corr_d4_rep", np.nan))

    if pd.isna(ic_raw) or abs(ic_raw) < 0.01:
        return "noise"

    if (
        pd.notna(ic_anchor)
        and pd.notna(ic_raw)
        and abs(ic_raw) > 1e-8
        and abs(ic_anchor) > MAX_IC_AMPLIFICATION_RATIO * abs(ic_raw)
    ):
        return "amplification_artifact"

    if pd.notna(corr) and abs(corr) >= REDUNDANT_CORR:
        if passes_strict_incremental(ic_raw, ic_anchor, min_abs_ic=MIN_INDEPENDENT_IC):
            return "partial_independent"
        return "redundant"

    if passes_strict_incremental(ic_raw, ic_anchor, min_abs_ic=MIN_INDEPENDENT_IC):
        return "independent_alpha"

    if pd.notna(ic_anchor) and abs(ic_anchor) >= MIN_RESIDUAL_WEAK and ic_raw * ic_anchor > 0:
        if abs(ic_anchor) >= MIN_INDEPENDENT_IC:
            return "partial_independent"
        return "weak_incremental"

    if row.get("stack_enhancement_pass"):
        return "enhancer"

    if pd.notna(ic_stack) and abs(ic_stack) < MIN_RESIDUAL_WEAK:
        return "ohlcv_redundant"

    return "weak_or_mixed"


def analyze_density_candidate(
    factor_name: str,
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    dim_rep: pd.DataFrame,
    exposure_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    frozen_list: List[pd.DataFrame],
    *,
    hypothesis: str = "",
    sample_dates: int = 504,
    stack_enhancement_pass: Optional[bool] = None,
    monotonicity: Optional[float] = None,
    anchor_key: str = "ic_after_anchor",
    dim_label: str = "dim",
) -> dict:
    ics = sequential_neutral_ics(
        panel, ret, exposure_panels, frozen_list, sample_dates=sample_dates
    )
    ic_anchor = ic_after_anchor(panel, ret, dim_rep, sample_dates=sample_dates)
    corr_rep = mean_cross_sectional_corr(panel, dim_rep, sample_dates=sample_dates)

    row = {
        "factor_name": factor_name,
        "hypothesis": hypothesis,
        "ic_raw": ics.get("ic_raw", np.nan),
        anchor_key: ic_anchor,
        "ic_after_ohlcv_stack": ics.get("ic_after_ohlcv_stack", np.nan),
        "ic_ohlcv_stack_only": ics.get("ic_ohlcv_stack_only", np.nan),
        "corr_rep": corr_rep,
        "abs_corr_rep": abs(corr_rep) if pd.notna(corr_rep) else np.nan,
        f"ic_retention_after_{dim_label}": (
            abs(ic_anchor) / abs(ics["ic_raw"])
            if pd.notna(ics.get("ic_raw")) and abs(ics["ic_raw"]) > 1e-8
            else np.nan
        ),
        "monotonicity_score": monotonicity,
        "stack_enhancement_pass": stack_enhancement_pass,
    }
    # Backward compat for D4/D5 scripts
    row["ic_after_d4"] = ic_anchor if dim_label == "d4" else np.nan
    row["ic_after_d5"] = ic_anchor if dim_label == "d5" else np.nan
    row["ic_after_d2"] = ic_anchor if dim_label == "d2" else np.nan
    row["ic_after_d3"] = ic_anchor if dim_label == "d3" else np.nan
    row["ic_after_d1"] = ic_anchor if dim_label == "d1" else np.nan
    row["corr_d4_rep"] = corr_rep if dim_label == "d4" else np.nan
    row["density_classification"] = classify_density_signal(pd.Series(row), anchor_key=anchor_key)
    return row


def save_density_summary(df: pd.DataFrame, out_dir: Path, filename: str = "density_summary.csv") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path
