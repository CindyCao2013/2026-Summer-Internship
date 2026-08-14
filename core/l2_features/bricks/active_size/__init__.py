"""active_size brick — Active_* average-size concentration.

Observable only. Do **not** call this institutional participation.
Large active buys may be 游资 / 量化 / 散户集中交易 as well as 机构.
"""

from __future__ import annotations

from core.l2_features.bricks.active_size.concentration import (
    ACTIVE_SIZE_COL,
    ACTIVE_SIZE_EWM_COL,
    BRICK_VERSION,
    TOP_SIZE_PCT,
    compute_daily_active_size_concentration,
    concentration_one_day,
    smooth_active_size_concentration,
)
from core.l2_features.bricks.active_size.daily_builder import (
    CACHE_ROOT,
    ensure_active_size_daily_bricks,
)

__all__ = [
    "ACTIVE_SIZE_COL",
    "ACTIVE_SIZE_EWM_COL",
    "BRICK_VERSION",
    "CACHE_ROOT",
    "TOP_SIZE_PCT",
    "compute_daily_active_size_concentration",
    "concentration_one_day",
    "ensure_active_size_daily_bricks",
    "smooth_active_size_concentration",
]
