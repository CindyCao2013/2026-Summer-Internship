"""L2 Factor x AI Stock Selection System v1.

Track A (P0) research namespace. Reuses candidate_pool_v1, FS-1..FS-5,
Fast Discovery, and Factor_Dev_Lib. Does not replace frozen registries.
"""

from .contracts import (
    AI_CONTRACT_VERSION,
    CANONICAL_HORIZONS,
    COST_SCENARIOS_BPS,
    EXECUTION_CONVENTION,
    LABEL_HORIZON_MAP,
    PROJECT_ROOT,
    RESULT_ROOT,
)
from .model_contract import CORE_BENCHMARKS, LGBM_PARAMS

__all__ = [
    "AI_CONTRACT_VERSION",
    "CANONICAL_HORIZONS",
    "CORE_BENCHMARKS",
    "COST_SCENARIOS_BPS",
    "EXECUTION_CONVENTION",
    "LABEL_HORIZON_MAP",
    "LGBM_PARAMS",
    "PROJECT_ROOT",
    "RESULT_ROOT",
]
