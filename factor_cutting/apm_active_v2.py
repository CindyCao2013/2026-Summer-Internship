"""APM_ActiveV2 — Active Pressure Metric（主动买卖压力）.

Primary identity (brick ``active_pressure``):
  raw_apm = (active_buy_amt - active_sell_amt) / (active_buy_amt + active_sell_amt)
  daily   = amount-weighted mean of minute raw_apm
  smooth  = EWM(span=5)

Legacy research knife (session open/close + intraday slope) lives in
``apm_active_v2_session_cut`` and is **not** the default factor id.
"""

from __future__ import annotations

from factor_cutting.engine import CuttingSpec, KnifeSpec, ObjectSpec, OutputSpec
from core.l2_features.bricks.active_pressure.pressure import (
    BRICK_VERSION,
    EWM_MIN_PERIODS,
    EWM_SPAN,
    MIN_MINUTES_PER_DAY,
    PRESSURE_COL,
    PRESSURE_EWM_COL,
    compute_daily_active_pressure,
    minute_raw_apm,
    smooth_active_pressure,
)
from factor_cutting.freq_hold import to_weekly_hold, to_weekly_thu_hold
from core.l2_features.bricks.active_pressure.pressure_enhanced import (
    assign_session_weight,
    compute_daily_apm_session,
    compute_daily_apm_smart,
    compute_daily_smart_apm_v2,
    compute_dynamic_size_threshold,
    apply_asc_cs_gate,
    delta_apm_wide,
    smooth_delta_wide,
    SMARTV2_EWM_SPAN,
    SMARTV2_LOOKBACK,
    SMARTV2_QUANTILE,
)

FORMULA_VERSION = "apm_active_v2_pressure_ewm5_v1_smartv2"

APM_ACTIVE_V2_SPEC = CuttingSpec(
    name="apm_active_v2",
    paper="APM因子2.0（Active Pressure Metric·研究版）",
    direction_paper="positive_ic",
    status="implemented_active_pressure",
    object=ObjectSpec(variable="active_buy_sell_amount", additive=False),
    knife=KnifeSpec(
        variable="active_pressure",
        method="amount_weighted_daily_mean",
        window=EWM_SPAN,
        formula="ewm(amt_wmean((buy-sell)/(buy+sell)), span=5)",
    ),
    output=OutputSpec(
        op="ewm",
        formula="ewm(apm_raw, span=5)",
    ),
)

# Public aliases used by builder / tests
compute_daily_apm = compute_daily_active_pressure


def ewm_smooth_daily(
    daily,
    span: int = EWM_SPAN,
    min_periods: int = EWM_MIN_PERIODS,
):
    """EWM-smooth apm_raw → apm_smooth (per symbol)."""
    return smooth_active_pressure(
        daily, span=span, min_periods=min_periods,
        value_col=PRESSURE_COL, out_col=PRESSURE_EWM_COL,
    )


__all__ = [
    "APM_ACTIVE_V2_SPEC",
    "BRICK_VERSION",
    "EWM_MIN_PERIODS",
    "EWM_SPAN",
    "FORMULA_VERSION",
    "MIN_MINUTES_PER_DAY",
    "PRESSURE_COL",
    "PRESSURE_EWM_COL",
    "assign_session_weight",
    "apply_asc_cs_gate",
    "compute_daily_active_pressure",
    "compute_daily_apm",
    "compute_daily_apm_session",
    "compute_daily_apm_smart",
    "compute_daily_smart_apm_v2",
    "compute_dynamic_size_threshold",
    "delta_apm_wide",
    "ewm_smooth_daily",
    "minute_raw_apm",
    "smooth_active_pressure",
    "smooth_delta_wide",
    "SMARTV2_EWM_SPAN",
    "SMARTV2_LOOKBACK",
    "SMARTV2_QUANTILE",
    "to_weekly_hold",
    "to_weekly_thu_hold",
]
