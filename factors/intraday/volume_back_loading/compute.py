"""volume_back_loading — dual-path compute (Python reference + DDB-native)."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from core.ddb_intraday_queries import volume_back_loading_script
from minute_bar_store import (
    MinuteBarStore,
    filter_a_share,
    get_default_store,
    to_wind_code,
)

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

FACTOR_NAME = "volume_back_loading"
SIGNAL_TIME = dt.time(9, 59)
NARROW_COLUMNS = ("bartime", "symbol", "factorname", "value")


def _empty_narrow() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NARROW_COLUMNS))


def python_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Reference pandas implementation using prior sessions only."""
    from core.intraday_alphas import _compute_volume_back_loading_python

    return _compute_volume_back_loading_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=False,
        lookback_days=lookback_days,
    )


def _normalize_ddb_narrow(
    raw: pd.DataFrame,
    start_date: DateLike,
    end_date: DateLike,
) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return _empty_narrow()
    df = raw.copy().rename(columns={"Symbol": "symbol", "Date": "date"})
    df["symbol"] = df["symbol"].map(to_wind_code)
    df = filter_a_share(df, include_bj=False)
    # Preserve the reference contract exactly: pandas BDay skips weekends but
    # intentionally does not apply an exchange-holiday calendar.
    source_date = pd.to_datetime(df["date"]).dt.normalize()
    df["bartime"] = source_date + pd.offsets.BDay(1) + pd.Timedelta(
        hours=9, minutes=59
    )
    df["factorname"] = FACTOR_NAME
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    df = df[(df["bartime"] >= start_ts) & (df["bartime"] <= end_ts)]
    return (
        df[list(NARROW_COLUMNS)]
        .dropna(subset=["value"])
        .sort_values(["bartime", "symbol"])
        .reset_index(drop=True)
    )


def ddb_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Compute late-session volume ratio in DDB; return next-day signals."""
    if store is None:
        hist_start = pd.Timestamp(start_date) - pd.Timedelta(
            days=lookback_days + 10
        )
        store = get_default_store(start_date=hist_start)
    raw = store.run_script(
        volume_back_loading_script(
            start_date,
            end_date,
            lookback_days=lookback_days,
            symbols=symbols,
        )
    )
    return _normalize_ddb_narrow(raw, start_date, end_date)


def assert_signal_time(narrow: pd.DataFrame) -> None:
    if narrow.empty:
        return
    times = set(pd.to_datetime(narrow["bartime"]).dt.time.unique())
    if times != {SIGNAL_TIME}:
        raise AssertionError(f"Expected only 09:59 signals, got {sorted(times)}")


def assert_bartime_alignment(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
) -> None:
    assert_signal_time(python_narrow)
    assert_signal_time(ddb_narrow)
    py_dates = set(pd.to_datetime(python_narrow["bartime"]).dt.normalize().unique())
    db_dates = set(pd.to_datetime(ddb_narrow["bartime"]).dt.normalize().unique())
    if py_dates != db_dates:
        raise AssertionError(
            f"Signal-date mismatch: python_only={len(py_dates - db_dates)}, "
            f"ddb_only={len(db_dates - py_dates)}"
        )


def assert_no_future_leakage_contract(
    start_date: DateLike = "2024-05-01",
    end_date: DateLike = "2024-05-31",
    *,
    lookback_days: int = 20,
) -> None:
    script = volume_back_loading_script(
        start_date,
        end_date,
        lookback_days=lookback_days,
    )
    required = (
        "Date between histStart : endDate",
        "context by Symbol csort Date",
        "move(msum(closing_vol, lookback, minP), 1)",
        "move(mcount(closing_vol, lookback, minP), 1)",
    )
    missing = [fragment for fragment in required if fragment not in script]
    if missing:
        raise AssertionError(f"Look-ahead contract missing SQL fragments: {missing}")
    if "move(msum(closing_vol, lookback, minP), -1)" in script:
        raise AssertionError("Forward move detected in rolling history")


def align_narrow(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["bartime", "symbol"]
    a = left[keys + ["value"]].rename(columns={"value": "python"}).copy()
    b = right[keys + ["value"]].rename(columns={"value": "ddb"}).copy()
    a["bartime"] = pd.to_datetime(a["bartime"])
    b["bartime"] = pd.to_datetime(b["bartime"])
    merged = a.merge(b, on=keys, how="inner")
    merged["abs_diff"] = (merged["python"] - merged["ddb"]).abs()
    return merged
