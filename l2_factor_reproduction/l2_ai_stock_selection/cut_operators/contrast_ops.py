"""Contrast operators between two cut aggregates."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CONTRAST_OPERATORS,
    RATIO_EPSILON,
)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def denom_ok(den: float, eps: float = RATIO_EPSILON) -> bool:
    return np.isfinite(den) and abs(float(den)) > float(eps)


def contrast_diff(a, b) -> float:
    aa, bb = _f(a), _f(b)
    if not (np.isfinite(aa) and np.isfinite(bb)):
        return float("nan")
    return aa - bb


def contrast_ratio(a, b, *, eps: float = RATIO_EPSILON) -> float:
    aa, bb = _f(a), _f(b)
    den = abs(bb) + float(eps)
    if not np.isfinite(aa) or not np.isfinite(bb):
        return float("nan")
    if abs(bb) <= float(eps):
        return float("nan")
    out = aa / den
    if not np.isfinite(out):
        return float("nan")
    return float(out)


def contrast_normalized_diff(a, b, *, eps: float = RATIO_EPSILON) -> float:
    aa, bb = _f(a), _f(b)
    if not (np.isfinite(aa) and np.isfinite(bb)):
        return float("nan")
    den = abs(aa) + abs(bb) + float(eps)
    if not denom_ok(den, eps):
        return float("nan")
    out = (aa - bb) / den
    if not np.isfinite(out):
        return float("nan")
    return float(out)


def contrast_share(part, full, *, eps: float = RATIO_EPSILON) -> float:
    return contrast_ratio(part, full, eps=eps)


def contrast_acceleration(late, early) -> float:
    """late_state - early_state."""
    return contrast_diff(late, early)


def contrast_reversal(early, late) -> float:
    """Sign-aware early vs late reversal: -sign(early) * late.

    Positive when the late state flips against the early sign.
    """
    e, l = _f(early), _f(late)
    if not (np.isfinite(e) and np.isfinite(l)):
        return float("nan")
    if e == 0:
        return float("nan")
    return float(-np.sign(e) * l)


def contrast_persistence(a, b) -> float:
    return contrast_diff(a, b)


def ratio_denominator_diagnostics(a, b, *, eps: float = RATIO_EPSILON) -> Dict[str, object]:
    aa, bb = _f(a), _f(b)
    zero = bool(np.isfinite(bb) and bb == 0)
    near = bool(np.isfinite(bb) and bb != 0 and abs(bb) <= float(eps))
    used = abs(bb) + float(eps) if np.isfinite(bb) else float("nan")
    return {
        "numerator": aa,
        "denominator_raw": bb,
        "denominator_used": used,
        "eps": float(eps),
        "zero_denominator": zero,
        "near_zero_denominator": near,
        "ratio_defined": bool(np.isfinite(aa) and np.isfinite(bb) and abs(bb) > float(eps)),
    }


_DISPATCH = {
    "DIFF": contrast_diff,
    "RATIO": contrast_ratio,
    "NORMALIZED_DIFF": contrast_normalized_diff,
    "SHARE": contrast_share,
    "ACCELERATION": contrast_acceleration,
    "REVERSAL": contrast_reversal,
    "PERSISTENCE_CONTRAST": contrast_persistence,
}


def apply_contrast(op: str, a, b, *, eps: float = RATIO_EPSILON):
    name = str(op).strip().upper()
    if name not in CONTRAST_OPERATORS:
        raise KeyError("unknown contrast operator {!r}".format(op))
    fn = _DISPATCH[name]
    if name in ("RATIO", "SHARE", "NORMALIZED_DIFF"):
        return fn(a, b, eps=eps)
    if name == "ACCELERATION":
        return fn(a, b)  # late, early — caller passes (late, early)
    if name == "REVERSAL":
        return fn(a, b)  # early, late — caller passes (early, late)
    return fn(a, b)
