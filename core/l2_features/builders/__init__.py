"""Factor builders that turn bricks into raw factor panels.

Keep builders thin: load brick(s) + close/EOD → factor wide.
Shared observables live under ``core.l2_features.bricks``.
"""

from __future__ import annotations

from core.l2_features.builders.ideal_reversal_active_v2_builder import (  # noqa: F401
    build_ideal_reversal_active_v2_panel,
    build_ideal_reversal_raw_variants,
    ensure_daily_bricks as ensure_ideal_reversal_daily_bricks,
)
from core.l2_features.builders.apm_active_v2_builder import (  # noqa: F401
    build_apm_active_v2_panel,
    build_apm_enhanced_variants,
    build_apm_raw_variants,
)
from core.l2_features.bricks.active_size import (  # noqa: F401
    ensure_active_size_daily_bricks,
)
from core.l2_features.bricks.active_pressure import (  # noqa: F401
    ensure_active_pressure_daily_bricks,
)
