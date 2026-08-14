"""Package wrappers for the shared ``ofi_persistence`` implementation."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from factors.intraday import discovery_v1
from minute_bar_store import MinuteBarStore

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]
FACTOR_NAME = "ofi_persistence"

align_narrow = discovery_v1.align_narrow
assert_bartime_alignment = discovery_v1.assert_bartime_alignment
assert_standard_bartimes = discovery_v1.assert_standard_bartimes


def python_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Run the shared pandas reference implementation."""
    return discovery_v1.python_version(
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
    """Run the shared DolphinDB implementation."""
    return discovery_v1.ddb_version(
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
    """Select the configured backend through the shared implementation."""
    return discovery_v1.compute_factor(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def assert_no_future_leakage_contract() -> None:
    """Validate the generated DDB script's ordered trailing-window contract."""
    discovery_v1.assert_no_future_leakage_contract(FACTOR_NAME)
