"""D3 Price Path / Support Alpha Density — candidate registry for density mining v1."""

from __future__ import annotations

from typing import List, Tuple

D3_REPRESENTATIVE = "lower_shadow_support_20d"

# (factor_name, source, hypothesis, family)
D3_PATH_DENSITY_CANDIDATES: List[Tuple[str, str, str, str]] = [
    (
        "lower_shadow_support_20d",
        "pv",
        "Lower shadow strength — intraday dip absorption (D3 rep)",
        "shadow_support",
    ),
    (
        "upper_shadow_pressure_20d",
        "pv",
        "Upper shadow dominance — rejection / supply pressure",
        "shadow_pressure",
    ),
    (
        "range_contraction_20d",
        "pv",
        "Compressed candle range — path consolidation",
        "path_range",
    ),
    (
        "range_expansion_20d",
        "eod_engine",
        "Range expansion — volatile path breakout",
        "path_range",
    ),
    (
        "high_low_20d",
        "pv",
        "Daily range level — path amplitude",
        "path_range",
    ),
    (
        "drawup_drawdown_ratio_20d",
        "eod_engine",
        "Cumulative gain vs loss path asymmetry",
        "path_asymmetry",
    ),
    (
        "return_stability_20d",
        "eod_engine",
        "Return stability — smooth path vs noisy",
        "path_stability",
    ),
    (
        "volume_price_efficiency_20d",
        "pv",
        "Price move per unit volume — path efficiency",
        "path_efficiency",
    ),
]

D3_CONFIRMATION_FACTORS: List[str] = [name for name, _, _, _ in D3_PATH_DENSITY_CANDIDATES]
