"""D5 Tail Fragility Alpha Density — candidate registry for density mining v1."""

from __future__ import annotations

from typing import List, Tuple

D5_REPRESENTATIVE = "upside_fragility_20d"

# Mechanism buckets: A=lottery, B=fragile_rally, C=downside_tail, D=path_tail
D5_SATELLITE_FACTORS: List[str] = [
    "max_daily_return_20d",
    "drawup_drawdown_ratio_20d",
]

# (factor_name, source, hypothesis, family)
D5_TAIL_DENSITY_CANDIDATES: List[Tuple[str, str, str, str]] = [
    # B — Upside fragility (D5 rep)
    (
        "upside_fragility_20d",
        "eod_engine",
        "Fragile rally: upside peak vs baseline (D5 rep)",
        "fragile_rally",
    ),
    # A — Lottery / extreme winner
    (
        "max_daily_return_20d",
        "eod_engine",
        "Lottery reversal: max daily return in 20d",
        "lottery_reversal",
    ),
    # C — Downside tail risk
    (
        "tail_risk_min_return_20d",
        "eod_engine",
        "Worst daily return in 20d — left tail exposure",
        "downside_tail",
    ),
    (
        "downside_tail_cluster_20d",
        "eod_engine",
        "Count of large negative return days — tail clustering",
        "downside_tail",
    ),
    (
        "asymmetric_tail_ratio_20d",
        "eod_engine",
        "Downside/upside semivariance ratio",
        "downside_tail",
    ),
    (
        "tail_adjusted_momentum_60d",
        "eod_engine",
        "Return per unit downside volatility — tail-adjusted momentum",
        "downside_tail",
    ),
    # D — Path / range tail
    (
        "drawup_drawdown_ratio_20d",
        "eod_engine",
        "Cumulative gain vs loss path asymmetry",
        "path_tail",
    ),
    (
        "high_low_20d",
        "pv",
        "Daily range (high-low) level — range expansion proxy",
        "path_tail",
    ),
    (
        "high_low_60d",
        "pv",
        "60d average range — sustained path instability",
        "path_tail",
    ),
    (
        "upper_shadow_pressure_20d",
        "pv",
        "Upper shadow dominance — rejection tail",
        "path_tail",
    ),
    (
        "range_expansion_20d",
        "eod_engine",
        "Range expansion vs recent baseline",
        "path_tail",
    ),
    (
        "range_contraction_20d",
        "pv",
        "Range contraction — compressed tail state",
        "path_tail",
    ),
    (
        "return_skew_shift_20d",
        "eod_engine",
        "Short vs long window skew shift",
        "skew_regime",
    ),
    (
        "cn_amount_distribution_skew_20d",
        "cn_broker",
        "Flow distribution skew",
        "flow_tail",
    ),
]

D5_CONFIRMATION_FACTORS: List[str] = [name for name, _, _, _ in D5_TAIL_DENSITY_CANDIDATES]
