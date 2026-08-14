"""research.l2_alpha — ClickHouse SSL2 feature factory (Sprint 4.4)."""

from research.l2_alpha.clickhouse_ssl2 import (
    extract_minute_agg_wide,
    extract_minute_features,
)
from research.l2_alpha.formulas import compute_all_snapshot_features
from research.l2_alpha.l2_factor_registry import L2_PHASE2_FACTORS
from research.l2_alpha.schema import FACTOR_NAMES, NARROW_COLUMNS

__all__ = [
    "FACTOR_NAMES",
    "L2_PHASE2_FACTORS",
    "NARROW_COLUMNS",
    "compute_all_snapshot_features",
    "extract_minute_agg_wide",
    "extract_minute_features",
]
