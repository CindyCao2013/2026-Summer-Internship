"""D1 Liquidity Alpha Density — candidate registry for density mining v1."""

from __future__ import annotations

from typing import List, Tuple

D1_REPRESENTATIVE = "low_vol_liquidity_quality_60d"

D1_SATELLITE_FACTORS: List[str] = [
    "amihud_shock_reversal_5d",
]

# (factor_name, source, hypothesis, family)
D1_LIQUIDITY_DENSITY_CANDIDATES: List[Tuple[str, str, str, str]] = [
    (
        "low_vol_liquidity_quality_60d",
        "eod_engine",
        "Low vol + stable amount participation (D1 rep)",
        "stable_liquidity",
    ),
    (
        "amount_stability_20d",
        "liquidity_norm",
        "Stable dollar volume — liquidity quality without vol filter",
        "stable_liquidity",
    ),
    (
        "relative_liquidity_strength_20d",
        "eod_engine",
        "Stock liquidity vs cross-sectional median",
        "stable_liquidity",
    ),
    (
        "liquidity_amount_residual_20d",
        "liquidity_norm",
        "Amount stability orthogonal to size + liquidity level",
        "residual_liquidity",
    ),
    (
        "amihud_shock_reversal_5d",
        "eod_engine",
        "Amihud spike × short reversal — liquidity stress fade",
        "liquidity_shock",
    ),
    (
        "liquidity_shock_20d",
        "eod_engine",
        "Volume/turnover shock vs recent baseline",
        "liquidity_shock",
    ),
    (
        "cn_cancel_shock",
        "l2",
        "Bid/ask cancel imbalance z-score (L2 conditioning)",
        "liquidity_shock",
    ),
]

D1_CONFIRMATION_FACTORS: List[str] = [name for name, _, _, _ in D1_LIQUIDITY_DENSITY_CANDIDATES]
