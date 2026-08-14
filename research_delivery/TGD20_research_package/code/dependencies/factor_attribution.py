"""Factor Attribution Layer — diagnose where alpha comes from (exposure vs incremental).

Answers: "Is this factor independent behavioral alpha, or a hidden size/liquidity proxy?"

Pipeline per factor:
  raw IC → size-neutral IC → +liquidity → +volatility → +OHLCV frozen stack residual IC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from alpha_information_space import mean_rank_ic
from liquidity_normalization import panel_cross_sectional_residual

OHLCV_FROZEN_REPS: List[str] = [s.representative for s in OHLCV_PRODUCTION_DIMENSIONS]
DEFAULT_HORIZONS = (1, 5, 10, 20)
MIN_INCREMENTAL_IC = 0.02
MIN_RESIDUAL_IC_WEAK = 0.005
MIN_IC_RETENTION_RATIO = 0.5  # |residual| >= 50% of |raw|
MAX_IC_AMPLIFICATION_RATIO = 1.5  # |residual| <= 150% of |raw| — reject OLS amplification artifacts


def passes_strict_incremental(
    ic_raw: float,
    ic_residual: float,
    min_abs_ic: float = MIN_INCREMENTAL_IC,
    min_retention: float = MIN_IC_RETENTION_RATIO,
    max_amplification: float = MAX_IC_AMPLIFICATION_RATIO,
) -> bool:
    """
    HF strict triage: same-sign residual IC in [min_retention, max_amplification] × |raw|, |residual| >= min_abs_ic.
    Rejects cases where orthogonalization inflates IC (hidden collinearity / exposure leak).
    """
    if pd.isna(ic_raw) or pd.isna(ic_residual):
        return False
    if abs(ic_raw) < 1e-8:
        return False
    if abs(ic_residual) < min_abs_ic:
        return False
    if ic_raw * ic_residual <= 0:
        return False
    if abs(ic_residual) < min_retention * abs(ic_raw):
        return False
    if abs(ic_residual) > max_amplification * abs(ic_raw):
        return False
    return True


def strict_triage_conclusion(row: pd.Series) -> str:
    ic_raw = row.get("ic_raw", np.nan)
    ic_stack = row.get("ic_after_ohlcv_stack", np.nan)
    ic_stack_only = row.get("ic_ohlcv_stack_only", np.nan)
    ic_size = row.get("ic_after_size", np.nan)

    if pd.isna(ic_raw):
        return "insufficient_data"

    if passes_strict_incremental(ic_raw, ic_stack):
        return "independent_incremental_alpha"

    if pd.notna(ic_stack) and ic_raw * ic_stack > 0 and abs(ic_stack) >= MIN_INCREMENTAL_IC:
        if abs(ic_stack) > MAX_IC_AMPLIFICATION_RATIO * abs(ic_raw):
            return "amplification_artifact"

    if passes_strict_incremental(ic_raw, ic_stack_only):
        return "partial_incremental_alpha"

    if pd.notna(ic_stack) and abs(ic_stack) >= MIN_RESIDUAL_IC_WEAK:
        if ic_raw * ic_stack <= 0:
            return "sign_flip_after_neutral"
        return "weak_incremental"

    if pd.notna(ic_size) and abs(ic_size) < MIN_RESIDUAL_IC_WEAK and abs(ic_raw) >= MIN_INCREMENTAL_IC:
        return "size_proxy_remove"

    if pd.notna(ic_stack) and abs(ic_stack) < MIN_RESIDUAL_IC_WEAK:
        return "ohlcv_redundant_proxy"

    return "marginal"


def attribution_conclusion_loose(row: pd.Series) -> str:
    """Pre-strict heuristic (kept for audit comparison)."""
    ic_raw = row.get("ic_raw", np.nan)
    ic_stack = row.get("ic_after_ohlcv_stack", row.get("ic_ohlcv_stack_only", np.nan))
    ic_size = row.get("ic_after_size", np.nan)

    if pd.isna(ic_raw):
        return "insufficient_data"

    if pd.notna(ic_stack) and abs(ic_stack) >= MIN_INCREMENTAL_IC:
        if pct_ic_explained(ic_raw, ic_stack) < 0.7:
            return "loose_independent"
        return "loose_partial"

    if pd.notna(ic_stack) and abs(ic_stack) >= MIN_RESIDUAL_IC_WEAK:
        return "weak_incremental"

    if pd.notna(ic_size) and abs(ic_size) < 0.005 and abs(ic_raw) >= 0.02:
        return "size_proxy_remove"

    if pd.notna(ic_stack) and abs(ic_stack) < MIN_RESIDUAL_IC_WEAK:
        return "ohlcv_redundant_proxy"

    return "marginal"


def attribution_conclusion(row: pd.Series) -> str:
    """Primary conclusion — strict HF triage."""
    return strict_triage_conclusion(row)


@dataclass
class ExposureSpec:
    name: str
    label: str


EXPOSURE_SEQUENCE: List[ExposureSpec] = [
    ExposureSpec("size", "log_float_mktcap"),
    ExposureSpec("liquidity", "liquidity_proxy"),
    ExposureSpec("volatility", "volatility_proxy"),
]


def cs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1).replace(0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0)


def align_signal(factor: pd.DataFrame, shift: int = 1) -> pd.DataFrame:
    """T+1 convention: factor known at t-1 predicts return at t."""
    return factor.shift(shift)


def build_forward_returns(close: pd.DataFrame, horizons: Sequence[int] = DEFAULT_HORIZONS) -> Dict[int, pd.DataFrame]:
    """Forward return from next-day open proxy: close[t+h]/close[t] - 1."""
    out = {}
    for h in horizons:
        out[h] = close.shift(-h) / close - 1
    return out


def rank_ic_by_horizon(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    signal_shift: int = 1,
    sample_dates: Optional[int] = None,
) -> pd.DataFrame:
    """IC decay curve: rank IC at 1d / 5d / 10d / 20d forward horizons."""
    signal = align_signal(factor, signal_shift)
    fwd = build_forward_returns(close, horizons)
    rows = []
    for h in horizons:
        ic = mean_rank_ic(signal, fwd[h], sample_dates)
        rows.append({"horizon_days": h, "rank_ic": ic, "abs_rank_ic": abs(ic)})
    return pd.DataFrame(rows)


def mean_abs_cross_sectional_corr(factor: pd.DataFrame, exposure: pd.DataFrame, sample_dates: int = 504) -> float:
    """Average |cross-sectional correlation| between factor and exposure."""
    f = factor.iloc[-sample_dates:] if sample_dates and len(factor) > sample_dates else factor
    e = exposure.reindex_like(f)
    corrs = []
    for dt in f.index:
        row_f = f.loc[dt]
        row_e = e.loc[dt]
        mask = row_f.notna() & row_e.notna()
        if mask.sum() < 30:
            continue
        c = row_f[mask].corr(row_e[mask], method="spearman")
        if pd.notna(c):
            corrs.append(abs(c))
    return float(np.mean(corrs)) if corrs else np.nan


def sequential_neutral_ics(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    exposure_panels: Dict[str, pd.DataFrame],
    frozen_reps: Optional[List[pd.DataFrame]] = None,
    signal_shift: int = 1,
    sample_dates: int = 504,
) -> Dict[str, float]:
    """
    Progressive neutralization IC chain:
      raw → size → +liquidity → +volatility → +ohlcv_stack
    """
    if sample_dates and len(factor) > sample_dates:
        factor = factor.iloc[-sample_dates:]
        ret = ret.reindex(factor.index)
        exposure_panels = {k: v.reindex(factor.index) for k, v in exposure_panels.items()}
        if frozen_reps:
            frozen_reps = [f.reindex(factor.index) for f in frozen_reps]

    signal = align_signal(factor, signal_shift)
    out = {"ic_raw": mean_rank_ic(signal, ret, sample_dates=None)}

    anchors: List[pd.DataFrame] = []
    for spec in EXPOSURE_SEQUENCE:
        if spec.name not in exposure_panels:
            continue
        anchors.append(exposure_panels[spec.name])
        resid = panel_cross_sectional_residual(signal, anchors)
        out[f"ic_after_{spec.name}"] = mean_rank_ic(resid, ret, sample_dates=None)

    if frozen_reps:
        all_anchors = anchors + list(frozen_reps)
        resid = panel_cross_sectional_residual(signal, all_anchors)
        out["ic_after_ohlcv_stack"] = mean_rank_ic(resid, ret, sample_dates=None)
        out["ic_ohlcv_stack_only"] = mean_rank_ic(
            panel_cross_sectional_residual(signal, list(frozen_reps)), ret, sample_dates=None
        )

    return out


def pct_ic_explained(ic_raw: float, ic_residual: float) -> float:
    if ic_raw is None or pd.isna(ic_raw) or abs(ic_raw) < 1e-8:
        return np.nan
    return float(max(0.0, 1.0 - abs(ic_residual) / abs(ic_raw)))


def build_attribution_row(
    factor_name: str,
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    exposure_panels: Dict[str, pd.DataFrame],
    frozen_panels: Dict[str, pd.DataFrame],
    cn_family: str = "",
    hypothesis: str = "",
) -> dict:
    frozen_list = [frozen_panels[r] for r in OHLCV_FROZEN_REPS if r in frozen_panels]
    ics = sequential_neutral_ics(factor, ret, exposure_panels, frozen_list)

    exposure_corrs = {
        f"exposure_corr_{spec.name}": mean_abs_cross_sectional_corr(factor, exposure_panels[spec.name])
        for spec in EXPOSURE_SEQUENCE
        if spec.name in exposure_panels
    }

    ic_raw = ics.get("ic_raw", np.nan)
    ic_stack = ics.get("ic_after_ohlcv_stack", np.nan)

    row = {
        "factor_name": factor_name,
        "cn_family": cn_family,
        "hypothesis": hypothesis,
        **ics,
        **exposure_corrs,
        "pct_ic_explained_by_ohlcv_stack": pct_ic_explained(ic_raw, ic_stack),
        "pct_ic_explained_by_size": pct_ic_explained(ic_raw, ics.get("ic_after_size", np.nan)),
    }
    s = pd.Series(row)
    row["strict_pass"] = passes_strict_incremental(ic_raw, ic_stack)
    row["conclusion_loose"] = attribution_conclusion_loose(s)
    row["conclusion"] = strict_triage_conclusion(s)
    return row


def hl_sharpe_from_composite(
    composite: pd.DataFrame,
    ret: pd.DataFrame,
    n_groups: int = 10,
    signal_shift: int = 1,
) -> Tuple[float, float, int]:
    """
    Decile H-L Sharpe on composite signal.
    Returns (hl_sharpe, hl_annu_ret_approx, direction).
    """
    signal = align_signal(composite, signal_shift)
    aligned_ret = ret.reindex_like(signal)

    daily_hl = []
    for dt in signal.index:
        sig = signal.loc[dt]
        r = aligned_ret.loc[dt]
        mask = sig.notna() & r.notna()
        if mask.sum() < n_groups * 5:
            continue
        ranks = sig[mask].rank(pct=True)
        top = r[mask][ranks >= 1 - 1 / n_groups].mean()
        bot = r[mask][ranks <= 1 / n_groups].mean()
        daily_hl.append(top - bot)

    if len(daily_hl) < 50:
        return np.nan, np.nan, 1

    s = pd.Series(daily_hl)
    direction = 1 if s.mean() >= 0 else -1
    s_adj = s * direction
    ann_ret = s_adj.mean() * 250
    sharpe = s_adj.mean() / s.std() * np.sqrt(250) if s.std() > 0 else np.nan
    return float(sharpe), float(ann_ret), direction


def combine_equal_weight(panels: List[pd.DataFrame]) -> pd.DataFrame:
    zs = [cs_zscore(p) for p in panels]
    return sum(zs) / len(zs)


def incremental_bundle_test(
    baseline_panels: List[pd.DataFrame],
    candidate_panel: pd.DataFrame,
    ret: pd.DataFrame,
    candidate_weight: float = 1.0,
) -> dict:
    """Compare H-L Sharpe: baseline OHLCV bundle vs bundle + CN factor."""
    baseline = combine_equal_weight(baseline_panels)
    cand_z = cs_zscore(candidate_panel)
    enhanced = cs_zscore(baseline + candidate_weight * cand_z)

    sharpe_base, ret_base, dir_base = hl_sharpe_from_composite(baseline, ret)
    sharpe_enh, ret_enh, dir_enh = hl_sharpe_from_composite(enhanced, ret)
    sharpe_cand, _, _ = hl_sharpe_from_composite(candidate_panel, ret)

    return {
        "baseline_hl_sharpe": sharpe_base,
        "enhanced_hl_sharpe": sharpe_enh,
        "sharpe_delta": sharpe_enh - sharpe_base if pd.notna(sharpe_enh) and pd.notna(sharpe_base) else np.nan,
        "candidate_solo_hl_sharpe": sharpe_cand,
        "baseline_direction": dir_base,
        "incremental_bundle_value": (
            pd.notna(sharpe_enh)
            and pd.notna(sharpe_base)
            and (sharpe_enh - sharpe_base) >= 0.15
        ),
    }


def universe_ic_table(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    universe_masks: Dict[str, pd.DataFrame],
    signal_shift: int = 1,
) -> pd.DataFrame:
    """Per-universe rank IC for universe stability diagnostics."""
    signal = align_signal(factor, signal_shift)
    rows = []
    for uni, mask in universe_masks.items():
        masked = signal * mask.reindex_like(signal)
        ic = mean_rank_ic(masked, ret)
        rows.append({"universe": uni, "rank_ic": ic, "abs_rank_ic": abs(ic)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    ic_vals = df["rank_ic"]
    denom = abs(ic_vals.mean())
    df.attrs["universe_stability"] = (
        float(np.clip(1.0 - ic_vals.std() / denom, 0, 1)) if denom > 1e-8 else np.nan
    )
    df.attrs["sign_consistency"] = float((np.sign(ic_vals) == np.sign(ic_vals.mean())).mean())
    return df
