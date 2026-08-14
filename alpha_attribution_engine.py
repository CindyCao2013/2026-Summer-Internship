"""Alpha Attribution Engine v1 — exposure neutralization + enhancer × dimension grid.

Scope (intentionally narrow):
  1. Market-structure exposure chain (4 proxies only): size, vol, liquidity, mom/rev
  2. Enhancer attribution: Score = z(D_i) + λ z(E) for each frozen dimension + enhancer

NOT in v1: full Fama-MacBeth platform, 8+ economic exposures, ML weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from alpha_information_space import mean_rank_ic
from factor_attribution import (
    align_signal,
    cs_zscore,
    hl_sharpe_from_composite,
    pct_ic_explained,
)
from liquidity_normalization import panel_cross_sectional_residual

ENGINE_VERSION = "v1"

# Frozen market-structure proxies — NOT the production dimension reps (avoid circularity).
MARKET_STRUCTURE_EXPOSURES: List[Tuple[str, str]] = [
    ("size", "log_float_mktcap"),
    ("volatility", "volatility_60d"),
    ("liquidity", "amount_20d_mean"),
    ("momentum_reversal", "momentum_20d"),
]

DEFAULT_ENHANCER_LAMBDAS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
DEFAULT_ENHANCER_LAMBDA_REPORT = 0.2

FROZEN_DIM_MAP: Dict[str, str] = {
    spec.dimension_id: spec.representative for spec in OHLCV_PRODUCTION_DIMENSIONS
}
FROZEN_REP_TO_DIM: Dict[str, str] = {v: k for k, v in FROZEN_DIM_MAP.items()}


@dataclass
class ExposureSpec:
    name: str
    proxy_key: str


def market_structure_specs() -> List[ExposureSpec]:
    return [ExposureSpec(name=n, proxy_key=k) for n, k in MARKET_STRUCTURE_EXPOSURES]


def _slice_panels(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    exposures: Dict[str, pd.DataFrame],
    sample_dates: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if sample_dates and len(factor) > sample_dates:
        factor = factor.iloc[-sample_dates:]
        ret = ret.reindex(factor.index)
        exposures = {k: v.reindex(factor.index) for k, v in exposures.items()}
    return factor, ret, exposures


def sequential_market_structure_ics(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    exposure_panels: Dict[str, pd.DataFrame],
    *,
    signal_shift: int = 1,
    sample_dates: Optional[int] = 504,
    exclude_exposure: Optional[str] = None,
) -> Dict[str, float]:
    """
    Progressive IC chain vs 4 market-structure exposures (in fixed order).

    exclude_exposure: skip one exposure when candidate IS that proxy (e.g. D2 vs vol).
    """
    factor, ret, exposure_panels = _slice_panels(factor, ret, exposure_panels, sample_dates)
    signal = align_signal(factor, signal_shift)
    out: Dict[str, float] = {"ic_raw": mean_rank_ic(signal, ret, sample_dates=None)}

    anchors: List[pd.DataFrame] = []
    for spec in market_structure_specs():
        if spec.name == exclude_exposure:
            continue
        panel = exposure_panels.get(spec.name)
        if panel is None:
            continue
        anchors.append(panel)
        resid = panel_cross_sectional_residual(signal, anchors)
        out[f"ic_after_{spec.name}"] = mean_rank_ic(resid, ret, sample_dates=None)

    if anchors:
        resid_all = panel_cross_sectional_residual(signal, anchors)
        out["ic_true_residual"] = mean_rank_ic(resid_all, ret, sample_dates=None)
    else:
        out["ic_true_residual"] = out["ic_raw"]

    return out


def ic_after_other_frozen_dims(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    frozen_panels: Dict[str, pd.DataFrame],
    factor_name: str,
    *,
    signal_shift: int = 1,
    sample_dates: Optional[int] = 504,
) -> float:
    """IC after neutralizing all other D1–D5 reps (cross-dimension independence)."""
    factor, ret, _ = _slice_panels(factor, ret, {}, sample_dates)
    signal = align_signal(factor, signal_shift)
    others = [
        frozen_panels[r]
        for r in frozen_panels
        if r != factor_name and r in FROZEN_REP_TO_DIM
    ]
    if not others:
        return mean_rank_ic(signal, ret, sample_dates=None)
    resid = panel_cross_sectional_residual(signal, others)
    return mean_rank_ic(resid, ret, sample_dates=None)


def dominant_ic_drop_step(row: pd.Series) -> str:
    """Which exposure step removed the largest share of |raw IC|."""
    ic_raw = row.get("ic_raw", np.nan)
    if pd.isna(ic_raw) or abs(ic_raw) < 1e-8:
        return "insufficient"
    prev = ic_raw
    best_step = "none"
    best_drop = 0.0
    for spec in market_structure_specs():
        col = f"ic_after_{spec.name}"
        ic = row.get(col, np.nan)
        if pd.isna(ic):
            continue
        drop = abs(prev) - abs(ic)
        if drop > best_drop:
            best_drop = drop
            best_step = spec.name
        prev = ic
    return best_step


def build_exposure_attribution_row(
    factor_name: str,
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    market_exposures: Dict[str, pd.DataFrame],
    frozen_panels: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    dimension_id: str = "",
    role: str = "candidate",
    sample_dates: Optional[int] = 504,
) -> dict:
    """Single-factor exposure attribution row."""
    exclude = None
    if factor_name == "volatility_60d":
        exclude = "volatility"
    elif factor_name == "momentum_20d":
        exclude = "momentum_reversal"

    ics = sequential_market_structure_ics(
        factor,
        ret,
        market_exposures,
        sample_dates=sample_dates,
        exclude_exposure=exclude,
    )
    ic_raw = ics.get("ic_raw", np.nan)
    ic_resid = ics.get("ic_true_residual", np.nan)

    row = {
        "engine_version": ENGINE_VERSION,
        "factor_name": factor_name,
        "dimension_id": dimension_id or FROZEN_REP_TO_DIM.get(factor_name, ""),
        "role": role,
        **ics,
        "pct_ic_retained_after_market_exposures": (
            abs(ic_resid) / abs(ic_raw) if pd.notna(ic_raw) and abs(ic_raw) > 1e-8 else np.nan
        ),
        "pct_ic_explained_by_market_exposures": pct_ic_explained(ic_raw, ic_resid),
    }

    if frozen_panels and factor_name in FROZEN_REP_TO_DIM:
        ic_cross = ic_after_other_frozen_dims(
            factor, ret, frozen_panels, factor_name, sample_dates=sample_dates
        )
        row["ic_after_other_frozen_dims"] = ic_cross
        row["pct_ic_retained_vs_other_dims"] = (
            abs(ic_cross) / abs(ic_raw) if pd.notna(ic_raw) and abs(ic_raw) > 1e-8 else np.nan
        )

    s = pd.Series(row)
    row["dominant_exposure_drop"] = dominant_ic_drop_step(s)
    return row


def additive_enhancer_metrics(
    base_panel: pd.DataFrame,
    enhancer_panel: pd.DataFrame,
    ret: pd.DataFrame,
    lam: float,
    *,
    signal_shift: int = 1,
) -> Dict[str, float]:
    """Score = z(base) + λ z(enhancer); report IC and H-L Sharpe vs base alone."""
    base_z = cs_zscore(align_signal(base_panel, signal_shift))
    enh_z = cs_zscore(align_signal(enhancer_panel, signal_shift))
    combined = cs_zscore(base_z + lam * enh_z) if lam != 0 else base_z

    ic_base = mean_rank_ic(base_z, ret)
    ic_combined = mean_rank_ic(combined, ret)
    sharpe_base, _, _ = hl_sharpe_from_composite(base_z, ret)
    sharpe_combined, _, _ = hl_sharpe_from_composite(combined, ret)

    return {
        "lambda": lam,
        "ic_base": ic_base,
        "ic_combined": ic_combined,
        "ic_delta": ic_combined - ic_base if pd.notna(ic_combined) and pd.notna(ic_base) else np.nan,
        "sharpe_base": sharpe_base,
        "sharpe_combined": sharpe_combined,
        "sharpe_delta": (
            sharpe_combined - sharpe_base
            if pd.notna(sharpe_combined) and pd.notna(sharpe_base)
            else np.nan
        ),
    }


def run_enhancer_dimension_grid(
    frozen_panels: Dict[str, pd.DataFrame],
    enhancer_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    lambdas: Sequence[float] = DEFAULT_ENHANCER_LAMBDAS,
    report_lambda: float = DEFAULT_ENHANCER_LAMBDA_REPORT,
) -> pd.DataFrame:
    """
    For each (D_i, enhancer, λ): additive blend metrics.
    Primary research question: which dimension does each enhancer improve?
    """
    rows = []
    for dim_id, rep in FROZEN_DIM_MAP.items():
        if rep not in frozen_panels:
            continue
        base = frozen_panels[rep]
        for enh_name, enh_panel in enhancer_panels.items():
            for lam in lambdas:
                m = additive_enhancer_metrics(base, enh_panel, ret, lam)
                rows.append(
                    {
                        "engine_version": ENGINE_VERSION,
                        "dimension_id": dim_id,
                        "base_factor": rep,
                        "enhancer_factor": enh_name,
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def summarize_enhancer_targets(
    grid_df: pd.DataFrame,
    report_lambda: float = DEFAULT_ENHANCER_LAMBDA_REPORT,
) -> pd.DataFrame:
    """Best target dimension per enhancer at report_lambda (by Sharpe delta)."""
    if grid_df.empty:
        return pd.DataFrame()
    sub = grid_df[np.isclose(grid_df["lambda"], report_lambda)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for enh in sub["enhancer_factor"].unique():
        g = sub[sub["enhancer_factor"] == enh]
        if g.empty:
            continue
        best = g.loc[g["sharpe_delta"].idxmax()]
        rows.append(
            {
                "enhancer_factor": enh,
                "report_lambda": report_lambda,
                "best_dimension_id": best["dimension_id"],
                "best_base_factor": best["base_factor"],
                "sharpe_delta": best["sharpe_delta"],
                "ic_delta": best["ic_delta"],
                "interpretation": (
                    f"{enh} primarily enhances {best['dimension_id']} ({best['base_factor']})"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_exposure_attribution_batch(
    candidates: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    market_exposures: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    roles: Optional[Dict[str, str]] = None,
    *,
    sample_dates: Optional[int] = 504,
) -> pd.DataFrame:
    roles = roles or {}
    rows = []
    for name, panel in candidates.items():
        dim = FROZEN_REP_TO_DIM.get(name, "")
        rows.append(
            build_exposure_attribution_row(
                name,
                panel,
                ret,
                market_exposures,
                frozen_panels=frozen_panels,
                dimension_id=dim,
                role=roles.get(name, "candidate"),
                sample_dates=sample_dates,
            )
        )
    return pd.DataFrame(rows)


def interpret_dimension_independence(row: pd.Series) -> str:
    """Heuristic label for frozen dimension rows."""
    ic_raw = row.get("ic_raw", np.nan)
    ic_resid = row.get("ic_true_residual", np.nan)
    ic_cross = row.get("ic_after_other_frozen_dims", np.nan)
    if pd.isna(ic_raw):
        return "insufficient_data"
    if pd.notna(ic_cross) and abs(ic_cross) >= 0.02 and ic_raw * ic_cross > 0:
        if pd.notna(ic_resid) and abs(ic_resid) >= 0.015:
            return "independent_return_driver"
        return "partially_independent_dim"
    if pd.notna(ic_resid) and abs(ic_resid) < 0.005:
        return "mostly_market_structure_proxy"
    return "mixed_or_shared_with_other_dims"
