"""Daily factor-cache coverage helpers (correctness only).

tradetime values like ``2019-01-02 09:30:00`` must be compared against
trading-date bounds via normalized TradeDate, never as raw timestamps.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import pandas as pd

TimestampLike = Union[str, pd.Timestamp]


def normalized_trade_date(
    tradetime: Union[pd.Series, pd.Timestamp, str],
) -> Union[pd.Series, pd.Timestamp]:
    """Return calendar TradeDate at midnight for coverage comparisons."""
    ts = pd.to_datetime(tradetime)
    if isinstance(ts, pd.Series):
        return ts.dt.normalize()
    return ts.normalize()


def coverage_bounds_ok(
    tradetime: pd.Series,
    *,
    expected_min: TimestampLike,
    expected_max: TimestampLike,
) -> Tuple[bool, Optional[str]]:
    """Gate: min/max normalized TradeDate must cover [expected_min, expected_max]."""
    if tradetime is None or len(tradetime) == 0:
        return False, "empty tradetime"
    dates = normalized_trade_date(tradetime)
    d_min = dates.min()
    d_max = dates.max()
    lo = pd.Timestamp(expected_min).normalize()
    hi = pd.Timestamp(expected_max).normalize()
    if d_min > lo:
        return False, f"coverage starts late: {d_min.date()} > {lo.date()}"
    if d_max < hi:
        return False, f"coverage ends early: {d_max.date()} < {hi.date()}"
    return True, None


def assert_timestamp_covers_date(
    tradetime: TimestampLike,
    expected_date: TimestampLike,
) -> None:
    """Regression helper: ``2019-01-02 09:30:00`` covers date ``2019-01-02``."""
    day = normalized_trade_date(tradetime)
    want = pd.Timestamp(expected_date).normalize()
    if day != want:
        raise AssertionError(f"{tradetime!r} normalized to {day}, expected {want}")
