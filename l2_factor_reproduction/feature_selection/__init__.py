"""L2 Feature Selection layer (FS-1+).

FS-1 builds the ML feature panel / dataset contract only.
FS-2 adds the regression-first selector engine (synthetic validation).
No learners / walk-forward / real forward-return labels in FS-1/FS-2.
"""

from .contracts import (
    FORBIDDEN_OUTPUT_COLUMNS,
    PREPROCESS_CONTRACT_ID,
    FS1_VERDICTS,
)
from .selectors import CANONICAL_SELECTORS, build_selector, run_selector

__all__ = [
    "FORBIDDEN_OUTPUT_COLUMNS",
    "PREPROCESS_CONTRACT_ID",
    "FS1_VERDICTS",
    "CANONICAL_SELECTORS",
    "build_selector",
    "run_selector",
]
