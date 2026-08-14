"""L2 Feature Factory v1 — metacode registry (base × transform × window)."""

from __future__ import annotations

from typing import Dict, List, Tuple

# (transform, window) — window=1 means "current minute primitive only"
TransformSpec = Tuple[str, int]

# DolphinDB-style metacode: base minute metric → derived transforms.
# Base columns come from within-minute avg of snapshot LOB metrics.
L2_BASE_TRANSFORMS: Dict[str, List[TransformSpec]] = {
    "depth_oi": [
        ("mean", 10),
        ("std", 10),
        ("slope", 20),
        ("persistence", 20),
    ],
    "weighted_oi": [
        ("mean", 10),
        ("std", 20),
        ("delta", 5),
        ("delta", 30),
    ],
    "micro_bias": [
        ("mean", 1),  # current-minute primitive alias
        ("delta", 5),
        ("std", 20),
    ],
    "rel_spread": [
        ("mean", 1),
        ("std", 10),
        ("zscore", 20),
    ],
    "cancel_imb": [
        ("mean", 1),
        ("std", 10),
        ("zscore", 20),
    ],
}

# Short name prefixes used in exported factor names.
BASE_NAME_ALIAS = {
    "depth_oi": "depth_imb",
    "weighted_oi": "woi",
    "micro_bias": "micro_bias",
    "rel_spread": "spread",
    "cancel_imb": "cancel",
}

# Cross-sectional ranks applied in Python after CH extract (per bartime).
CS_RANK_SOURCES = (
    "depth_imb_mean10",
    "woi_mean10",
    "spread_mean",
)

SSE_ONLY_BASES = frozenset({"cancel_imb"})


def factor_name(base: str, transform: str, window: int) -> str:
    alias = BASE_NAME_ALIAS[base]
    if transform == "mean" and window == 1:
        return f"{alias}_mean"
    if transform == "delta":
        return f"{alias}_delta{window}"
    if transform == "zscore":
        return f"{alias}_z{window}"
    if transform == "persistence":
        return f"{alias}_persistence{window}"
    if transform == "slope":
        return f"{alias}_slope{window}"
    if transform == "mean":
        return f"{alias}_mean{window}"
    if transform == "std":
        return f"{alias}_std{window}"
    raise ValueError(f"Unknown transform {transform!r}")


def expand_derived_names() -> List[str]:
    """Expand metacode registry into CH-side derived factor names (17)."""
    names: List[str] = []
    for base, specs in L2_BASE_TRANSFORMS.items():
        for transform, window in specs:
            names.append(factor_name(base, transform, window))
    return names


def expand_all_factor_names() -> List[str]:
    """All v1 discovery factors including CS ranks (20)."""
    derived = expand_derived_names()
    ranks = [f"{src}_rank" for src in CS_RANK_SOURCES]
    return derived + ranks


def max_lookback() -> int:
    """Largest rolling window (minutes) required before evaluation bartime."""
    return max(w for specs in L2_BASE_TRANSFORMS.values() for _, w in specs)


L2_FF_DERIVED_COLUMNS = tuple(expand_derived_names())
L2_FF_ALL_COLUMNS = tuple(expand_all_factor_names())

# Assert frozen first-20 contract from the plan.
_EXPECTED = (
    "depth_imb_mean10",
    "depth_imb_std10",
    "depth_imb_slope20",
    "depth_imb_persistence20",
    "woi_mean10",
    "woi_std20",
    "woi_delta5",
    "woi_delta30",
    "micro_bias_mean",
    "micro_bias_delta5",
    "micro_bias_std20",
    "spread_mean",
    "spread_std10",
    "spread_z20",
    "cancel_mean",
    "cancel_std10",
    "cancel_z20",
    "depth_imb_mean10_rank",
    "woi_mean10_rank",
    "spread_mean_rank",
)
assert tuple(expand_all_factor_names()) == _EXPECTED, expand_all_factor_names()
