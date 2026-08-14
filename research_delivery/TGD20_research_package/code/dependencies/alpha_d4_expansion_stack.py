"""D4 Behavioral expansion stack validation — equal-weight composite vs frozen D1–D5 base."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alpha_frozen_stack_v1 import FROZEN_OHLCV_REPS
from alpha_information_space import mean_rank_ic
from alpha_research_report import ic_positive_ratio, monotonicity_score
from factor_attribution import (
    align_signal,
    combine_equal_weight,
    cs_zscore,
    hl_sharpe_from_composite,
    universe_ic_table,
)

D4_REPRESENTATIVE = "winner_sentiment_reversal_5d"
D4_EXPANSION_FACTORS: List[str] = [
    "winner_sentiment_reversal_5d",
    "max_daily_return_20d",
    "d4_consecutive_gain_exhaustion_20d",
]

D4_SATELLITE_FACTORS: List[str] = [
    "max_daily_return_20d",
    "d4_consecutive_gain_exhaustion_20d",
]

DEFAULT_SATELLITE_LAMBDAS: List[float] = [0.0, 0.05, 0.1, 0.2, 0.3]

DEFAULT_REGIME_SLICES: List[Tuple[str, str, str]] = [
    ("2020_2021", "2020-01-01", "2021-12-31"),
    ("2022_bear", "2022-01-01", "2022-12-31"),
    ("2023_2024", "2023-01-01", "2024-12-31"),
    ("2025_ytd", "2025-01-01", "2025-12-31"),
]


@dataclass
class StackEvalRow:
    stack_name: str
    period: str
    n_days: int
    rank_ic: float
    icir: float
    ic_positive_ratio: float
    hl_sharpe: float
    hl_annu_ret: float
    monotonicity_score: float
    universe_stability: float
    sign_consistency: float


def slice_panel(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return panel.loc[(panel.index >= start) & (panel.index <= end)]


def build_d4_composite(panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight cs_z blend of the three D4 density-pass factors."""
    parts = [panels[name] for name in D4_EXPANSION_FACTORS if name in panels]
    if len(parts) != len(D4_EXPANSION_FACTORS):
        missing = [n for n in D4_EXPANSION_FACTORS if n not in panels]
        raise KeyError(f"D4 expansion panels missing: {missing}")
    return combine_equal_weight(parts)


def build_base_stack(
    frozen_panels: Dict[str, pd.DataFrame],
    *,
    d4_composite: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Equal-weight D1–D5 base.

    If d4_composite is provided, the D4 slot uses the 3-factor composite
    instead of winner_sentiment_reversal alone.
    """
    parts: List[pd.DataFrame] = []
    for spec in FROZEN_OHLCV_REPS:
        if spec["dim"] == "D4" and d4_composite is not None:
            parts.append(d4_composite)
            continue
        panel = frozen_panels.get(spec["factor"])
        if panel is not None:
            parts.append(panel)
    if len(parts) != len(FROZEN_OHLCV_REPS):
        raise ValueError(f"Expected {len(FROZEN_OHLCV_REPS)} base panels, got {len(parts)}")
    return combine_equal_weight(parts)


def build_satellite_stack(
    base: pd.DataFrame,
    satellite_panels: Dict[str, pd.DataFrame],
    lam: float,
    *,
    satellite_factors: Optional[List[str]] = None,
    signal_shift: int = 1,
) -> pd.DataFrame:
    """
    Base D1–D5 + λ·z(satellite) for each satellite factor.

    Does not replace dimension base slots in the frozen stack.
    """
    names = satellite_factors if satellite_factors is not None else D4_SATELLITE_FACTORS
    combined = cs_zscore(align_signal(base, signal_shift))
    if lam == 0:
        return combined
    for name in names:
        panel = satellite_panels.get(name)
        if panel is None:
            raise KeyError(f"Missing satellite panel: {name}")
        combined = combined + lam * cs_zscore(align_signal(panel, signal_shift))
    return cs_zscore(combined)


def evaluate_lambda_grid(
    base: pd.DataFrame,
    satellite_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    lambdas: Sequence[float] = DEFAULT_SATELLITE_LAMBDAS,
    regime_slices: Sequence[Tuple[str, str, str]] = DEFAULT_REGIME_SLICES,
    universe_masks: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    satellite_factors: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full-sample + per-regime metrics for each satellite λ."""
    baseline_metrics = evaluate_stack_signal(base, ret, universe_masks)
    full_rows = []
    regime_rows = []

    for lam in lambdas:
        signal = build_satellite_stack(
            base, satellite_panels, lam, satellite_factors=satellite_factors
        )
        metrics = evaluate_stack_signal(signal, ret, universe_masks)
        row = {
            "lambda": lam,
            "period": "full",
            "n_days": len(ret),
            **metrics,
            "rank_ic_delta": metrics["rank_ic"] - baseline_metrics["rank_ic"],
            "icir_delta": metrics["icir"] - baseline_metrics["icir"],
            "hl_sharpe_delta": metrics["hl_sharpe"] - baseline_metrics["hl_sharpe"],
            "monotonicity_delta": metrics["monotonicity_score"] - baseline_metrics["monotonicity_score"],
        }
        full_rows.append(row)

        for label, start_s, end_s in regime_slices:
            start = pd.Timestamp(start_s)
            end = pd.Timestamp(end_s)
            ret_s = slice_panel(ret, start, end)
            if len(ret_s) < 60:
                continue
            base_s = slice_panel(base, start, end)
            sig_s = slice_panel(signal, start, end)
            masks_s = None
            if universe_masks:
                masks_s = {k: slice_panel(v, start, end) for k, v in universe_masks.items()}
            base_m = evaluate_stack_signal(base_s, ret_s, masks_s)
            sig_m = evaluate_stack_signal(sig_s, ret_s, masks_s)
            regime_rows.append(
                {
                    "lambda": lam,
                    "period": label,
                    "n_days": len(ret_s),
                    "rank_ic": sig_m["rank_ic"],
                    "icir": sig_m["icir"],
                    "hl_sharpe": sig_m["hl_sharpe"],
                    "monotonicity_score": sig_m["monotonicity_score"],
                    "rank_ic_delta": sig_m["rank_ic"] - base_m["rank_ic"],
                    "hl_sharpe_delta": sig_m["hl_sharpe"] - base_m["hl_sharpe"],
                }
            )

    return pd.DataFrame(full_rows), pd.DataFrame(regime_rows)


def classify_satellite_uplift(
    full_df: pd.DataFrame,
    *,
    promote_label: str = "promote_satellite_layer",
    conditional_label: str = "conditional_satellite_ic_only",
    reject_label: str = "keep_base_only",
) -> dict:
    """Check whether λ=0.1 and λ=0.2 both improve IC and Sharpe vs baseline."""
    base = full_df[full_df["lambda"] == 0.0]
    if base.empty:
        return {"robust_uplift_zone": False, "recommendation": "insufficient_data"}

    zone = full_df[full_df["lambda"].isin([0.1, 0.2])]
    ic_pass = (zone["rank_ic_delta"] > 0).all()
    sharpe_pass = (zone["hl_sharpe_delta"] > 0).all()
    mono_pass = (zone["monotonicity_delta"] >= 0).all()

    if ic_pass and sharpe_pass:
        rec = promote_label
    elif ic_pass and mono_pass:
        rec = conditional_label
    else:
        rec = reject_label

    return {
        "robust_uplift_zone": bool(ic_pass and sharpe_pass),
        "ic_uplift_at_01_02": bool(ic_pass),
        "sharpe_uplift_at_01_02": bool(sharpe_pass),
        "recommendation": rec,
    }


def publish_satellite_additive_test(
    summary: dict,
    full_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    out_dir: Path,
    *,
    file_prefix: str = "satellite",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{file_prefix}_additive_validation.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    full_df.to_csv(out_dir / f"{file_prefix}_lambda_grid.csv", index=False)
    regime_df.to_csv(out_dir / f"{file_prefix}_regime_by_lambda.csv", index=False)

    pivot = regime_df.pivot_table(
        index="lambda",
        columns="period",
        values=["rank_ic", "hl_sharpe", "hl_sharpe_delta"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot.reset_index().to_csv(out_dir / f"{file_prefix}_regime_pivot.csv", index=False)
    return json_path


def daily_rank_ic_series(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    signal_shift: int = 1,
) -> pd.Series:
    sig = align_signal(signal, signal_shift)
    aligned_r = ret.reindex_like(sig)
    return sig.corrwith(aligned_r, axis=1, method="spearman")


def decile_group_means(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    n_groups: int = 10,
    signal_shift: int = 1,
) -> pd.Series:
    sig = align_signal(signal, signal_shift)
    aligned_r = ret.reindex_like(sig)
    buckets: Dict[int, List[float]] = {i: [] for i in range(1, n_groups + 1)}

    for dt in sig.index:
        row_sig = sig.loc[dt]
        row_ret = aligned_r.loc[dt]
        mask = row_sig.notna() & row_ret.notna()
        if mask.sum() < n_groups * 5:
            continue
        ranks = row_sig[mask].rank(pct=True)
        rets = row_ret[mask]
        for g in range(1, n_groups + 1):
            lo = (g - 1) / n_groups
            hi = g / n_groups
            if g == n_groups:
                sel = ranks >= lo
            else:
                sel = (ranks >= lo) & (ranks < hi)
            if sel.any():
                buckets[g].append(float(rets[sel].mean()))

    return pd.Series({g: np.mean(v) if v else np.nan for g, v in buckets.items()})


def icir_from_daily(ic_daily: pd.Series) -> float:
    s = ic_daily.dropna()
    if len(s) < 20 or s.std() == 0:
        return np.nan
    return float(s.mean() / s.std() * np.sqrt(250))


def evaluate_stack_signal(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    universe_masks: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    signal_shift: int = 1,
) -> dict:
    sig = align_signal(signal, signal_shift)
    ic_daily = daily_rank_ic_series(signal, ret, signal_shift=signal_shift)
    sharpe, ann_ret, _ = hl_sharpe_from_composite(signal, ret, signal_shift=signal_shift)
    group_means = decile_group_means(signal, ret, signal_shift=signal_shift)

    out = {
        "rank_ic": mean_rank_ic(sig, ret, sample_dates=None),
        "icir": icir_from_daily(ic_daily),
        "ic_positive_ratio": ic_positive_ratio(ic_daily),
        "hl_sharpe": sharpe,
        "hl_annu_ret": ann_ret,
        "monotonicity_score": monotonicity_score(group_means),
        "universe_stability": np.nan,
        "sign_consistency": np.nan,
    }

    if universe_masks:
        uni_df = universe_ic_table(signal, ret, universe_masks, signal_shift=signal_shift)
        if not uni_df.empty:
            out["universe_stability"] = uni_df.attrs.get("universe_stability", np.nan)
            out["sign_consistency"] = uni_df.attrs.get("sign_consistency", np.nan)
            out["universe_ic"] = uni_df.set_index("universe")["rank_ic"].to_dict()
    return out


def compare_stacks(
    baseline: pd.DataFrame,
    expanded: pd.DataFrame,
    ret: pd.DataFrame,
    universe_masks: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    period: str = "full",
) -> Tuple[dict, dict, dict]:
    base_m = evaluate_stack_signal(baseline, ret, universe_masks)
    exp_m = evaluate_stack_signal(expanded, ret, universe_masks)
    delta = {
        "rank_ic_delta": exp_m["rank_ic"] - base_m["rank_ic"],
        "icir_delta": exp_m["icir"] - base_m["icir"],
        "hl_sharpe_delta": exp_m["hl_sharpe"] - base_m["hl_sharpe"],
        "monotonicity_delta": exp_m["monotonicity_score"] - base_m["monotonicity_score"],
        "universe_stability_delta": exp_m["universe_stability"] - base_m["universe_stability"],
    }
    delta["period"] = period
    delta["n_days"] = len(ret)
    return base_m, exp_m, delta


def evaluate_regime_comparison(
    baseline: pd.DataFrame,
    expanded: pd.DataFrame,
    ret: pd.DataFrame,
    regime_slices: Sequence[Tuple[str, str, str]] = DEFAULT_REGIME_SLICES,
    universe_masks: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    rows = []
    for label, start_s, end_s in regime_slices:
        start = pd.Timestamp(start_s)
        end = pd.Timestamp(end_s)
        ret_s = slice_panel(ret, start, end)
        if len(ret_s) < 60:
            continue
        base_s = slice_panel(baseline, start, end)
        exp_s = slice_panel(expanded, start, end)
        masks_s = None
        if universe_masks:
            masks_s = {k: slice_panel(v, start, end) for k, v in universe_masks.items()}
        base_m, exp_m, delta = compare_stacks(base_s, exp_s, ret_s, masks_s, period=label)
        rows.append({"stack": "baseline_d1_d5", "period": label, "n_days": len(ret_s), **base_m})
        rows.append({"stack": "d4_expanded_base", "period": label, "n_days": len(ret_s), **exp_m})
        rows.append({"stack": "delta_expanded_minus_baseline", "period": label, **delta})
    return pd.DataFrame(rows)


def publish_d4_stack_validation(
    summary: dict,
    regime_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "d4_density_stack_validation.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    regime_df.to_csv(out_dir / "d4_density_stack_regime.csv", index=False)
    universe_df.to_csv(out_dir / "d4_density_stack_universe.csv", index=False)
    pd.DataFrame([summary.get("full_sample_delta", {})]).to_csv(
        out_dir / "d4_density_stack_delta.csv", index=False
    )
    return json_path
