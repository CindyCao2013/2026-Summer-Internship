"""D2 Volatility / Risk Alpha Density — candidate registry for density mining v1."""

from __future__ import annotations

from typing import List, Tuple

D2_REPRESENTATIVE = "volatility_60d"

# (factor_name, source, hypothesis, family)
D2_RISK_DENSITY_CANDIDATES: List[Tuple[str, str, str, str]] = [
    (
        "volatility_60d",
        "pv",
        "60d realized return std — D2 rep (low-vol anomaly)",
        "realized_vol",
    ),
    (
        "volatility_20d",
        "pv",
        "20d realized vol — shorter risk horizon",
        "realized_vol",
    ),
    (
        "volatility_level_20d",
        "eod_engine",
        "Negated 20d vol level — low-vol rank signal",
        "realized_vol",
    ),
    (
        "high_low_60d",
        "pv",
        "60d average daily range — path volatility proxy",
        "range_vol",
    ),
    (
        "high_low_20d",
        "pv",
        "20d average daily range",
        "range_vol",
    ),
    (
        "range_contraction_20d",
        "pv",
        "Compressed range vs baseline — vol regime",
        "range_vol",
    ),
    (
        "range_expansion_20d",
        "eod_engine",
        "Range expansion vs recent baseline",
        "range_vol",
    ),
    (
        "return_stability_20d",
        "eod_engine",
        "Low return dispersion — stability / low-risk",
        "stability",
    ),
    (
        "vol_liquidity_stress_20d",
        "eod_engine",
        "Amihud × vol interaction — illiquid high-vol stress",
        "vol_liquidity",
    ),
    (
        "amount_to_volatility_20d",
        "pv",
        "Liquidity per unit vol — vol-adjusted participation",
        "vol_liquidity",
    ),
    (
        "volatility_regime_change_20d",
        "eod_engine",
        "Short vs long vol ratio — regime shift",
        "regime",
    ),
    (
        "tail_adjusted_momentum_60d",
        "eod_engine",
        "Return per downside vol — tail-adjusted momentum",
        "tail_adj",
    ),
]

D2_CONFIRMATION_FACTORS: List[str] = [name for name, _, _, _ in D2_RISK_DENSITY_CANDIDATES]
