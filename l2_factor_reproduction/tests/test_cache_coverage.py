"""Regression: daily-cache coverage uses normalized TradeDate."""

from __future__ import annotations

import pandas as pd

from l2_factor_reproduction.python.cache_coverage import (
    assert_timestamp_covers_date,
    coverage_bounds_ok,
    normalized_trade_date,
)


def test_0930_accepted_as_coverage_for_same_calendar_date() -> None:
    assert_timestamp_covers_date("2019-01-02 09:30:00", "2019-01-02")
    assert normalized_trade_date(pd.Timestamp("2019-01-02 09:30:00")) == pd.Timestamp(
        "2019-01-02"
    )


def test_series_coverage_bounds_with_intraday_timestamps() -> None:
    tt = pd.Series(
        [
            "2019-01-02 09:30:00",
            "2026-07-31 09:30:00",
        ]
    )
    ok, err = coverage_bounds_ok(
        tt, expected_min="2019-01-02", expected_max="2026-07-31"
    )
    assert ok and err is None


def test_raw_timestamp_compare_would_be_wrong_but_normalize_fixes() -> None:
    """Document the prior bug: 09:30 > midnight date bound under naive compare."""
    raw = pd.Timestamp("2019-01-02 09:30:00")
    bound = pd.Timestamp("2019-01-02")
    assert raw > bound  # naive compare falsely treats coverage as late
    assert normalized_trade_date(raw) == bound
