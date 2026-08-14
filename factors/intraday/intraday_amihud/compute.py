"""Factor-bound wrappers around :mod:`factors.intraday.discovery_v1`."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factors.intraday import discovery_v1
from minute_bar_store import MinuteBarStore

DateLike = discovery_v1.DateLike
FACTOR_NAME = "intraday_amihud"
STANDARD_BARTIMES = discovery_v1.STANDARD_BARTIMES


def python_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Compute the pandas reference implementation."""
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
    """Compute the factor in DolphinDB and normalize its narrow output."""
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
    """Select the configured discovery-v1 backend."""
    return discovery_v1.compute_factor(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def ddb_script(
    start_date: DateLike = "2024-05-01",
    end_date: DateLike = "2024-05-31",
    *,
    symbols: Optional[List[str]] = None,
    bartimes: Optional[List[str]] = None,
) -> str:
    """Build the canonical discovery-v1 DolphinDB script."""
    return discovery_v1.discovery_v1_factor_script(
        FACTOR_NAME,
        start_date,
        end_date,
        symbols=symbols,
        bartimes=bartimes,
    )


def assert_standard_bartimes(narrow: pd.DataFrame) -> None:
    discovery_v1.assert_standard_bartimes(narrow)


def assert_bartime_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    discovery_v1.assert_bartime_alignment(left, right)


def assert_no_future_leakage_contract() -> None:
    discovery_v1.assert_no_future_leakage_contract(FACTOR_NAME)
    script = ddb_script()
    required = (
        "close_adj \\ move(close_adj, 1) - 1.0 as minute_ret",
        "msum(abs(minute_ret), 5, 3)",
        "msum(amount_adj, 5, 3)",
        "context by Symbol, Date csort Bartime",
    )
    missing = [fragment for fragment in required if fragment not in script]
    if missing:
        raise AssertionError(f"Amihud no-look-ahead contract missing: {missing}")


def align_narrow(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return discovery_v1.align_narrow(left, right)
