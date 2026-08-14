"""Factor-bound wrappers around the shared discovery-v1 implementation."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from factors.intraday.discovery_v1 import (
    align_narrow,
    assert_bartime_alignment,
    assert_standard_bartimes,
    compute_factor as _compute_factor,
    ddb_version as _ddb_version,
    python_version as _python_version,
)
from factors.intraday.discovery_v1 import (
    assert_no_future_leakage_contract as _assert_no_future_leakage_contract,
)
from minute_bar_store import MinuteBarStore

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]
FACTOR_NAME = "bartime_ofi"


def python_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Run the explicit pandas fallback and golden reference."""
    return _python_version(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def ddb_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    bartimes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the production DolphinDB calculation."""
    return _ddb_version(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        bartimes=bartimes,
    )


def compute_factor(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Dispatch by the per-factor flag; full-day requests explicitly use Python."""
    return _compute_factor(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def assert_no_future_leakage_contract() -> None:
    """Validate the shared generated script for this factor."""
    _assert_no_future_leakage_contract(FACTOR_NAME)


__all__ = [
    "FACTOR_NAME",
    "align_narrow",
    "assert_bartime_alignment",
    "assert_no_future_leakage_contract",
    "assert_standard_bartimes",
    "compute_factor",
    "ddb_version",
    "python_version",
]
