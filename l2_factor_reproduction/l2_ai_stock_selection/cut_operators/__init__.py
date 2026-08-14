"""Temporal / State Cutting Operators v1.

Reusable feature-engineering sidecar for L2 AI Stock Selection v1.
Does not replace candidate_pool_v1, tree models, or AlphaNet.
"""

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CUT_CONTRACT_VERSION,
    CUT_MODULE_ID,
    CUT_RESULT_ROOT,
    MAX_DESCENDANTS_PER_PARENT,
    MAX_WORKERS,
)

__all__ = [
    "CUT_CONTRACT_VERSION",
    "CUT_MODULE_ID",
    "CUT_RESULT_ROOT",
    "MAX_DESCENDANTS_PER_PARENT",
    "MAX_WORKERS",
]
