"""Canonical evaluation contracts for intraday factor research."""

from .intraday_metrics import (
    ANNUALIZATION_DAYS,
    build_group_excess_panel,
    build_hl_panel,
    summarize_cross_sectional_metrics,
    summarize_ic_series,
)

__all__ = [
    "ANNUALIZATION_DAYS",
    "build_group_excess_panel",
    "build_hl_panel",
    "summarize_cross_sectional_metrics",
    "summarize_ic_series",
]
