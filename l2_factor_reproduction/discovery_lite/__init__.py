"""Batch Discovery Lite — cheap triage before Full Fast Discovery.

This package does not replace Fast Discovery, mutate frozen factor formulas,
or consume the ML feature panel.
"""

from l2_factor_reproduction.discovery_lite.contracts import (
    BDL_CONTRACT,
    CONTRACT_VERSION,
    lite_trading_dates,
)

__all__ = ["BDL_CONTRACT", "CONTRACT_VERSION", "lite_trading_dates"]
