"""Scale-compression helpers for cut operators.

All statistics are stock-day (or causal-intraday). No full-sample future
quantiles. These wrap the module SHARE / location-normalization formulas.
"""

from __future__ import annotations

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    RATIO_EPSILON,
)


def share_ratio(part, full, *, eps: float = RATIO_EPSILON) -> pd.Series:
    """SHARE: part / (|full| + eps). Undefined when |full| <= eps."""
    a = pd.to_numeric(part, errors="coerce")
    b = pd.to_numeric(full, errors="coerce")
    if not isinstance(a, pd.Series):
        a = pd.Series(a)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    den = b.abs()
    zero = den.notna() & (den <= float(eps))
    out = a / (den + float(eps))
    return out.where(a.notna() & b.notna() & ~zero)


def relative_to_group_median(
    frame: pd.DataFrame,
    col: str,
    *,
    keys=("symbol", "TradeDate"),
    eps: float = RATIO_EPSILON,
) -> pd.Series:
    """X / (|stock_day_median(X)| + eps). Location normalization, no future days."""
    val = pd.to_numeric(frame[col], errors="coerce")
    med = frame.groupby(list(keys), sort=False)[col].transform("median")
    med = pd.to_numeric(med, errors="coerce")
    return val / (med.abs() + float(eps))
