"""minute_skew package wrappers over the shared discovery implementation."""

from __future__ import annotations

from factors.intraday import discovery_v1 as shared

FACTOR_NAME = "minute_skew"
STANDARD_BARTIMES = shared.STANDARD_BARTIMES
align_narrow = shared.align_narrow
assert_bartime_alignment = shared.assert_bartime_alignment
assert_standard_bartimes = shared.assert_standard_bartimes


def python_version(start_date, end_date, store=None, *, symbols=None, return_full_day=False):
    return shared.python_version(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def ddb_version(start_date, end_date, store=None, *, symbols=None, bartimes=None):
    return shared.ddb_version(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        bartimes=bartimes,
    )


def compute(start_date, end_date, store=None, *, symbols=None, return_full_day=False):
    return shared.compute_factor(
        FACTOR_NAME,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def assert_no_future_leakage_contract() -> None:
    shared.assert_no_future_leakage_contract(FACTOR_NAME)
