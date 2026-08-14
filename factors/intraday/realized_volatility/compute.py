"""Factor-local wrappers for the discovery-v1 realized-volatility implementation."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from factors.intraday import discovery_v1
from minute_bar_store import MinuteBarStore

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]
FACTOR_NAME = "realized_volatility"
STANDARD_BARTIMES = discovery_v1.STANDARD_BARTIMES
NARROW_COLUMNS = discovery_v1.NARROW_COLUMNS


def python_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Compute the causal pandas reference implementation."""
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
    """Compute the DolphinDB implementation."""
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
    """Dispatch through the discovery-v1 backend flag."""
    return discovery_v1.compute_factor(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def use_ddb() -> bool:
    return discovery_v1.use_ddb(FACTOR_NAME)


def assert_standard_bartimes(narrow: pd.DataFrame) -> None:
    discovery_v1.assert_standard_bartimes(narrow)


def assert_bartime_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    discovery_v1.assert_bartime_alignment(left, right)


def assert_no_future_leakage_contract() -> None:
    discovery_v1.assert_no_future_leakage_contract(FACTOR_NAME)


def align_narrow(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return discovery_v1.align_narrow(left, right)
