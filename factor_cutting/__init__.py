"""Factor Cutting framework v1 — Object / Knife / Output.

Implements Kaiyuan-style partition operators for daily microstructure alphas.
Minute-dependent factors (APM, Smart Money) are stubbed with clear upgrade hooks.
"""

from __future__ import annotations

from factor_cutting.engine import (
    CuttingSpec,
    KnifeSpec,
    ObjectSpec,
    OutputSpec,
    cut_difference,
    rolling_rank_split_sum,
    rolling_quantile_mean_diff,
)
from factor_cutting.ideal_amplitude import compute_ideal_amplitude
from factor_cutting.ideal_reversal import compute_ideal_reversal
from factor_cutting.registry import CUTTING_FACTOR_LIST, compute_cutting_factor

__all__ = [
    "CuttingSpec",
    "KnifeSpec",
    "ObjectSpec",
    "OutputSpec",
    "cut_difference",
    "rolling_rank_split_sum",
    "rolling_quantile_mean_diff",
    "compute_ideal_reversal",
    "compute_ideal_amplitude",
    "CUTTING_FACTOR_LIST",
    "compute_cutting_factor",
]
