"""Alpha Production Stack v2 — D1–D5 base + conditional enhancers.

Structure (Jul 2026):
  Base:   equal-weight cs_z(D1..D5 frozen reps)
  Layer:  additive enhancers at attribution-calibrated λ
          - cn_cancel_shock      (L2 primary)
          - quality_composite    (D7 representative)

Score = cs_z( eq_wt(D1..D5) + λ_cancel·z(cancel_shock) + λ_quality·z(quality_composite) )

See alpha_information_space_v1.json + research/results/alpha_production_stack_v2.*
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import factor_config as cfg
from alpha_frozen_stack_v1 import FROZEN_OHLCV_REPS
from factor_attribution import align_signal, combine_equal_weight, cs_zscore, hl_sharpe_from_composite
from alpha_information_space import mean_rank_ic

STACK_VERSION = "v2"
STACK_STATUS = "production_enhancer_layer"

DEFAULT_ENHANCER_LAMBDAS: Dict[str, float] = {
    "cn_cancel_shock": 0.2,
    "quality_composite": 0.2,
}

PRODUCTION_ENHANCERS: List[Dict[str, str]] = [
    {
        "factor": "cn_cancel_shock",
        "source": "l2",
        "role": "cancel_state",
        "tier": "primary_enhancer",
        "default_lambda": "0.2",
        "attribution_target": "D4",
    },
    {
        "factor": "quality_composite",
        "source": "fundamental_quality",
        "role": "quality_state",
        "tier": "d7_representative",
        "default_lambda": "0.2",
        "attribution_target": "D4",
    },
]


@dataclass
class StackV2Metrics:
    ic_base: float
    ic_enhanced: float
    ic_delta: float
    sharpe_base: float
    sharpe_enhanced: float
    sharpe_delta: float
    direction: int


def base_panel_list(frozen_panels: Dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
    panels = []
    for spec in FROZEN_OHLCV_REPS:
        p = frozen_panels.get(spec["factor"])
        if p is not None:
            panels.append(p)
    return panels


def build_production_stack_v2_signal(
    frozen_panels: Dict[str, pd.DataFrame],
    enhancer_panels: Dict[str, pd.DataFrame],
    *,
    lambdas: Optional[Dict[str, float]] = None,
    signal_shift: int = 1,
) -> pd.DataFrame:
    """Equal-weight D1–D5 + additive enhancer blend."""
    lambdas = lambdas or DEFAULT_ENHANCER_LAMBDAS
    base = combine_equal_weight(base_panel_list(frozen_panels))
    combined = cs_zscore(align_signal(base, signal_shift))
    for name, lam in lambdas.items():
        panel = enhancer_panels.get(name)
        if panel is None or lam == 0:
            continue
        combined = combined + lam * cs_zscore(align_signal(panel, signal_shift))
    return cs_zscore(combined)


def evaluate_stack_v2(
    frozen_panels: Dict[str, pd.DataFrame],
    enhancer_panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    lambdas: Optional[Dict[str, float]] = None,
    signal_shift: int = 1,
) -> StackV2Metrics:
    base = combine_equal_weight(base_panel_list(frozen_panels))
    base_sig = align_signal(base, signal_shift)
    enhanced = build_production_stack_v2_signal(
        frozen_panels, enhancer_panels, lambdas=lambdas, signal_shift=signal_shift
    )

    ic_base = mean_rank_ic(base_sig, ret)
    ic_enh = mean_rank_ic(enhanced, ret)
    sharpe_base, _, dir_base = hl_sharpe_from_composite(base_sig, ret)
    sharpe_enh, _, _ = hl_sharpe_from_composite(enhanced, ret)

    return StackV2Metrics(
        ic_base=ic_base,
        ic_enhanced=ic_enh,
        ic_delta=ic_enh - ic_base if pd.notna(ic_enh) and pd.notna(ic_base) else np.nan,
        sharpe_base=sharpe_base,
        sharpe_enhanced=sharpe_enh,
        sharpe_delta=(
            sharpe_enh - sharpe_base
            if pd.notna(sharpe_enh) and pd.notna(sharpe_base)
            else np.nan
        ),
        direction=dir_base,
    )


def load_enhancer_lambdas_from_attribution(
    targets_path: Optional[Path] = None,
    report_lambda: float = 0.2,
) -> Dict[str, float]:
    """Pull λ defaults from attribution summary; fall back to constants."""
    path = targets_path or cfg.RESEARCH_DIR / "alpha_enhancer_targets_v1.csv"
    out = dict(DEFAULT_ENHANCER_LAMBDAS)
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for spec in PRODUCTION_ENHANCERS:
        name = spec["factor"]
        if name in df["enhancer_factor"].values:
            # production uses fixed report_lambda for all enhancers
            out[name] = report_lambda
    return out


def publish_production_stack_v2(
    metrics: StackV2Metrics,
    lambdas: Dict[str, float],
    *,
    out_dir: Optional[Path] = None,
    sample_days: int = 504,
    n_days: int = 0,
) -> Path:
    out_dir = out_dir or cfg.RESEARCH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "stack_version": STACK_VERSION,
        "stack_status": STACK_STATUS,
        "published": pd.Timestamp.now().isoformat(),
        "base": {
            "type": "equal_weight_cs_z",
            "dimensions": FROZEN_OHLCV_REPS,
        },
        "enhancers": [
            {**e, "lambda": lambdas.get(e["factor"], float(e["default_lambda"]))}
            for e in PRODUCTION_ENHANCERS
        ],
        "blend_formula": "cs_z( eq_wt(D1..D5) + Σ λ_i·z(enhancer_i) )",
        "metrics": {
            "sample_days": sample_days,
            "n_days": n_days,
            "ic_base": metrics.ic_base,
            "ic_enhanced": metrics.ic_enhanced,
            "ic_delta": metrics.ic_delta,
            "hl_sharpe_base": metrics.sharpe_base,
            "hl_sharpe_enhanced": metrics.sharpe_enhanced,
            "hl_sharpe_delta": metrics.sharpe_delta,
            "direction": metrics.direction,
        },
    }
    json_path = out_dir / "alpha_production_stack_v2.json"
    json_path.write_text(json.dumps(spec, indent=2, default=str) + "\n")

    row = {
        "stack_version": STACK_VERSION,
        "stack_status": STACK_STATUS,
        **spec["metrics"],
        "lambda_cancel_shock": lambdas.get("cn_cancel_shock"),
        "lambda_quality_composite": lambdas.get("quality_composite"),
    }
    pd.DataFrame([row]).to_csv(out_dir / "alpha_production_stack_v2_metrics.csv", index=False)
    return json_path
