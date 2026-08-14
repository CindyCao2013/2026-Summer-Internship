"""Total-return skewness (baseline SKEW anomaly).

Definition (window L):

    r_{i,t} = P_{i,t} / P_{i,t-1} - 1

    SKEW_{i,t}^{(L)} = m3 / m2^{3/2}

where m2, m3 are rolling sample central moments of daily returns.

No lookahead inside the formula; callers must apply signal_shift=1 for
cross-sectional prediction of next-day returns.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import pandas as pd

SKEW_WINDOWS = (20, 60, 120)

_DEFAULT_MIN_PERIODS: Mapping[int, int] = {
    20: 10,
    60: 40,
    120: 80,
}


def total_return_skew(
    ret_1d: pd.DataFrame,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Rolling skewness of daily returns (wide panel).

    Parameters
    ----------
    ret_1d:
        Wide panel of daily close-to-close returns (index=date, columns=symbol).
    window:
        Formation window in trading days (20 / 60 / 120).
    min_periods:
        Minimum valid observations; defaults scale with window.
    """
    if window <= 2:
        raise ValueError("window must be > 2")
    mp = _DEFAULT_MIN_PERIODS.get(window, max(5, window // 2))
    if min_periods is not None:
        mp = int(min_periods)
    return ret_1d.rolling(window, min_periods=mp).skew()


def skew_20d(ret_1d: pd.DataFrame, *, min_periods: Optional[int] = None) -> pd.DataFrame:
    return total_return_skew(ret_1d, 20, min_periods=min_periods)


def skew_60d(ret_1d: pd.DataFrame, *, min_periods: Optional[int] = None) -> pd.DataFrame:
    return total_return_skew(ret_1d, 60, min_periods=min_periods)


def skew_120d(ret_1d: pd.DataFrame, *, min_periods: Optional[int] = None) -> pd.DataFrame:
    return total_return_skew(ret_1d, 120, min_periods=min_periods)


def alpha_from_skew(skew: pd.DataFrame) -> pd.DataFrame:
    """Delivery alpha: Alpha = -SKEW (long low / negative skew)."""
    return -skew


def build_total_skew(
    ret_1d: pd.DataFrame,
    windows: Iterable[int] = SKEW_WINDOWS,
    *,
    as_alpha: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build named total-skew panels.

    Keys are ``SKEW{w}`` (raw) or ``AlphaSKEW{w}`` when ``as_alpha=True``.
    """
    out: Dict[str, pd.DataFrame] = {}
    for w in windows:
        raw = total_return_skew(ret_1d, int(w))
        key = f"AlphaSKEW{int(w)}" if as_alpha else f"SKEW{int(w)}"
        out[key] = alpha_from_skew(raw) if as_alpha else raw
    return out
