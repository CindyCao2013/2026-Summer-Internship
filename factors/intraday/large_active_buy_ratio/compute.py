"""Wrappers for the discovery-v1 large-active-buy bar-level proxy."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from factors.intraday import discovery_v1

FACTOR_NAME = "large_active_buy_ratio"
NARROW_COLUMNS = discovery_v1.NARROW_COLUMNS
STANDARD_BARTIMES = discovery_v1.STANDARD_BARTIMES


def python_version(
    start_date: discovery_v1.DateLike,
    end_date: discovery_v1.DateLike,
    store: Optional[discovery_v1.MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Compute the pandas reference for the bar-level proxy."""
    return discovery_v1.python_version(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def ddb_version(
    start_date: discovery_v1.DateLike,
    end_date: discovery_v1.DateLike,
    store: Optional[discovery_v1.MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    bartimes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute the DDB-native version for the bar-level proxy."""
    return discovery_v1.ddb_version(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        bartimes=bartimes,
    )


def compute_factor(
    start_date: discovery_v1.DateLike,
    end_date: discovery_v1.DateLike,
    store: Optional[discovery_v1.MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Dispatch through the shared discovery-v1 backend selection."""
    return discovery_v1.compute_factor(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def assert_standard_bartimes(narrow: pd.DataFrame) -> None:
    discovery_v1.assert_standard_bartimes(narrow)


def assert_bartime_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    discovery_v1.assert_bartime_alignment(left, right)


def align_narrow(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return discovery_v1.align_narrow(left, right)


def assert_no_future_leakage_contract() -> None:
    """Require shifted baseline statistics and ordered trailing windows."""
    discovery_v1.assert_no_future_leakage_contract(FACTOR_NAME)
    script = discovery_v1.discovery_v1_factor_script(
        FACTOR_NAME, "2024-05-01", "2024-05-31"
    )
    required = (
        "move(mavg(buy_size, 20, 10), 1)",
        "move(mstd(buy_size, 20, 10), 1)",
        "context by Symbol, Date csort Bartime",
        "msum(large_buy_amt, 20, 10)",
        "msum(buy_amt, 20, 10)",
    )
    missing = [fragment for fragment in required if fragment not in script]
    if missing:
        raise AssertionError(f"No-look-ahead proxy contract missing: {missing}")


def assert_unit_interval(narrow: pd.DataFrame) -> None:
    """The classified buy-amount share must remain in [0, 1]."""
    values = pd.to_numeric(narrow["value"], errors="coerce")
    if values.isna().any() or not values.between(0.0, 1.0).all():
        raise AssertionError("large_active_buy_ratio values must be finite in [0, 1]")
