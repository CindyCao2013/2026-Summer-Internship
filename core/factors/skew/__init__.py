"""SKEW / IdioSKEW research-grade factor family.

Economic intuition
------------------
Positive skewness (lottery-like payoff) is overpriced → lower future return.
Negative skewness is underpriced (crash-risk compensation) → higher future return.

Variants
--------
- Total return skewness (baseline): rolling skew of daily close-to-close returns
- Idiosyncratic skewness (headline): skew of market-model residuals (CICC style)
- Realized / intraday skew (P1 extension): see realized_skew.py

Delivery alpha convention
-------------------------
Raw research quantity SKEW has expected negative RankIC.
Delivery / book construction uses Alpha = -SKEW (long low-skew).
"""

from core.factors.skew.idio_skew import (
    build_idio_skew,
    idio_skew_60,
    idio_skew_120,
    rolling_market_residual,
)
from core.factors.skew.skew import (
    SKEW_WINDOWS,
    alpha_from_skew,
    build_total_skew,
    skew_20d,
    skew_60d,
    skew_120d,
    total_return_skew,
)
from core.factors.skew.skew_v2 import (
    build_skew_v2_panels,
    mad_winsorize_cs,
    max_residual_skew,
    tail_skew,
    tgd_residual_skew,
    vol_adjusted_skew,
    vol_residual_skew,
)

__all__ = [
    "SKEW_WINDOWS",
    "alpha_from_skew",
    "build_idio_skew",
    "build_skew_v2_panels",
    "build_total_skew",
    "idio_skew_60",
    "idio_skew_120",
    "mad_winsorize_cs",
    "max_residual_skew",
    "rolling_market_residual",
    "skew_20d",
    "skew_60d",
    "skew_120d",
    "tail_skew",
    "tgd_residual_skew",
    "total_return_skew",
    "vol_adjusted_skew",
    "vol_residual_skew",
]
