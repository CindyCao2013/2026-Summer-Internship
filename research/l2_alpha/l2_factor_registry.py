"""Phase 2 L2 discovery factor registry (first four only)."""

from __future__ import annotations

from typing import Dict, List, TypedDict


class L2FactorSpec(TypedDict):
    source: str
    mechanism: str
    base_metric: str
    aggregation: str
    horizons: List[str]
    bartimes: List[str]
    sse_only: bool


# Preheat matrix has Ret_15/30/60… — no Ret_5. Use closest available set.
DEFAULT_HORIZONS = ["Ret_15", "Ret_30", "Ret_60"]
# Must match PREHEAT_RET_MATRIX_ZZ1000 Bartime grid (:29/:59 slots).
# Round clock times like 10:00 / 14:30 do NOT join and break evaluation.
DEFAULT_BARTIMES = [
    "09:59",
    "10:29",
    "10:59",
    "11:29",
    "13:29",
    "13:59",
    "14:29",
]

# First-round discovery set — do not expand until residual alpha is proven.
L2_PHASE2_FACTORS: Dict[str, L2FactorSpec] = {
    "l2_weighted_oi_mean": {
        "source": "SSL2",
        "mechanism": "order_imbalance",
        "base_metric": "weighted_oi",
        "aggregation": "mean",
        "horizons": list(DEFAULT_HORIZONS),
        "bartimes": list(DEFAULT_BARTIMES),
        "sse_only": False,
    },
    "l2_microprice_bias_mean": {
        "source": "SSL2",
        "mechanism": "price_pressure",
        "base_metric": "micro_bias",
        "aggregation": "mean",
        "horizons": list(DEFAULT_HORIZONS),
        "bartimes": list(DEFAULT_BARTIMES),
        "sse_only": False,
    },
    "l2_depth_imbalance_mean": {
        "source": "SSL2",
        "mechanism": "depth_pressure",
        "base_metric": "depth_oi",
        "aggregation": "mean",
        "horizons": list(DEFAULT_HORIZONS),
        "bartimes": list(DEFAULT_BARTIMES),
        "sse_only": False,
    },
    "l2_cancel_pressure_sum": {
        "source": "SSL2",
        "mechanism": "cancel_flow",
        "base_metric": "cancel_flow",
        "aggregation": "sum_ratio",
        "horizons": list(DEFAULT_HORIZONS),
        "bartimes": list(DEFAULT_BARTIMES),
        "sse_only": True,  # SZSE lacks withdraw columns
    },
}

# Optional diagnostic aggregations emitted by SQL but not in Phase-2 gate.
L2_DIAGNOSTIC_COLUMNS = (
    "l2_weighted_oi_max",
    "l2_weighted_oi_std",
)

EXISTING_BASELINE_FACTORS = (
    "realized_volatility",
    "close_vwap_deviation",
    "intraday_amihud",
)

UNIVERSE_INDEX = "000852.SH"  # CSI1000
