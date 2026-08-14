"""active_buy_sell_imbalance — Python reference and DDB-native compute."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Set, Union

import pandas as pd

from core.ddb_intraday_queries import active_buy_sell_imbalance_script
from minute_bar_store import (
    MinuteBarStore,
    filter_a_share,
    get_default_store,
    to_wind_code,
)

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]
FACTOR_NAME = "active_buy_sell_imbalance"
NARROW_COLUMNS = ("bartime", "symbol", "factorname", "value")
STANDARD_BARTIME_SET: Set[dt.time] = {
    dt.time(9, 59),
    dt.time(10, 29),
    dt.time(11, 29),
    dt.time(13, 29),
    dt.time(14, 29),
}


def _empty_narrow() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NARROW_COLUMNS))


def python_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    from core.intraday_alphas import _compute_active_buy_sell_imbalance_python

    return _compute_active_buy_sell_imbalance_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def _normalize_ddb_narrow(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return _empty_narrow()
    df = raw.copy().rename(
        columns={"Symbol": "symbol", "Bartime": "bartime"}
    )
    df["symbol"] = df["symbol"].map(to_wind_code)
    df = filter_a_share(df, include_bj=False)
    df["bartime"] = pd.to_datetime(df["bartime"])
    df["factorname"] = FACTOR_NAME
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
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
    bartimes: Optional[List[str]] = None,
) -> pd.DataFrame:
    if store is None:
        store = get_default_store(start_date=start_date)
    raw = store.run_script(
        active_buy_sell_imbalance_script(
            start_date,
            end_date,
            symbols=symbols,
            bartimes=bartimes,
        )
    )
    return _normalize_ddb_narrow(raw)


def assert_standard_bartimes(narrow: pd.DataFrame) -> None:
    if narrow.empty:
        return
    times = set(pd.to_datetime(narrow["bartime"]).dt.time.unique())
    extra = times - STANDARD_BARTIME_SET
    if extra:
        raise AssertionError(f"Unexpected signal times: {sorted(extra)}")


def assert_bartime_alignment(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
) -> None:
    assert_standard_bartimes(python_narrow)
    assert_standard_bartimes(ddb_narrow)
    py_times = set(pd.to_datetime(python_narrow["bartime"]).unique())
    db_times = set(pd.to_datetime(ddb_narrow["bartime"]).unique())
    if py_times != db_times:
        raise AssertionError(
            f"Signal-time mismatch: python_only={len(py_times - db_times)}, "
            f"ddb_only={len(db_times - py_times)}"
        )


def assert_no_future_leakage_contract(
    start_date: DateLike = "2024-05-01",
    end_date: DateLike = "2024-05-31",
) -> None:
    script = active_buy_sell_imbalance_script(start_date, end_date)
    required = (
        "context by Symbol, Date csort Bartime",
        "cumsum(buy_amt) as cum_buy",
        "cumsum(sell_amt) as cum_sell",
        "where Bartime in btFilter",
    )
    missing = [fragment for fragment in required if fragment not in script]
    if missing:
        raise AssertionError(f"No-look-ahead SQL contract missing: {missing}")
    if "move(" in script:
        raise AssertionError("Cumulative intraday OFI must not use shifted future state")


def align_narrow(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["bartime", "symbol"]
    a = left[keys + ["value"]].rename(columns={"value": "python"}).copy()
    b = right[keys + ["value"]].rename(columns={"value": "ddb"}).copy()
    a["bartime"] = pd.to_datetime(a["bartime"])
    b["bartime"] = pd.to_datetime(b["bartime"])
    merged = a.merge(b, on=keys, how="inner")
    merged["abs_diff"] = (merged["python"] - merged["ddb"]).abs()
    return merged
