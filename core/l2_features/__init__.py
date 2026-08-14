"""L2 temporal / microstructure feature primitives.

Stages:
  1  return_timing.py       Gu / Gd
  2  timing_residual.py     εu / εd vs return-structure controls
  2.5 return_distribution.py Rū / Rd̄ / zero counts (controls)
  3  tgd.py                 εd ~ εu → MA20 → TGD20
  4  validation             run_tgd_validation_v1.py (+ tgd_panel_builder)
"""

from core.l2_features.return_distribution import (
    DailyReturnDistribution,
    compute_return_distribution,
    compute_return_distribution_daily,
    enrich_centers_with_distribution,
)
from core.l2_features.return_timing import (
    DailyTimingCenters,
    compute_down_time_center,
    compute_minute_returns,
    compute_timing_centers_daily,
    compute_up_time_center,
    prepare_tgd_inputs,
    trading_minute_index,
)
from core.l2_features.tgd import (
    build_tgd20,
    daily_tgd_innovation,
    smooth_tgd,
    tgd20_to_wide,
)
from core.l2_features.tgd_panel_builder import (
    assemble_residual_inputs,
    build_tgd20_wide_from_eod_l2,
    load_timing_daily_features,
    overnight_return_long,
)
from core.l2_features.timing_residual import (
    attach_session_controls_from_minute,
    cs_ols_residual,
    merge_centers_with_controls,
    prepare_tgd_from_residuals,
    residualize_timing_centers,
)

__all__ = [
    "DailyReturnDistribution",
    "DailyTimingCenters",
    "assemble_residual_inputs",
    "attach_session_controls_from_minute",
    "build_tgd20",
    "build_tgd20_wide_from_eod_l2",
    "compute_down_time_center",
    "compute_minute_returns",
    "compute_return_distribution",
    "compute_return_distribution_daily",
    "compute_timing_centers_daily",
    "compute_up_time_center",
    "cs_ols_residual",
    "daily_tgd_innovation",
    "enrich_centers_with_distribution",
    "load_timing_daily_features",
    "merge_centers_with_controls",
    "overnight_return_long",
    "prepare_tgd_from_residuals",
    "prepare_tgd_inputs",
    "residualize_timing_centers",
    "smooth_tgd",
    "tgd20_to_wide",
    "trading_minute_index",
]
