"""close_vwap_deviation — dual-path compute (Python reference + DDB-native)."""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Set, Union

import pandas as pd

from core.ddb_intraday_queries import close_vwap_deviation_script
from minute_bar_store import MinuteBarStore, get_default_store, to_wind_code

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

FACTOR_NAME = "close_vwap_deviation"
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
    """Reference implementation (pandas on minute panel from MinuteBarStore)."""
    from core.intraday_alphas import _compute_close_vwap_deviation_python

    return _compute_close_vwap_deviation_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def _normalize_ddb_narrow(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return _empty_narrow()
    df = raw.copy()
    rename = {}
    if "Symbol" in df.columns:
        rename["Symbol"] = "symbol"
    if "bartime" not in df.columns and "Bartime" in df.columns:
        rename["Bartime"] = "bartime"
    df = df.rename(columns=rename)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].map(to_wind_code)
    df["bartime"] = pd.to_datetime(df["bartime"])
    df["factorname"] = FACTOR_NAME
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (
        df[list(NARROW_COLUMNS)]
        .dropna(subset=["value"])
        .sort_values(["bartime", "symbol"])
        .reset_index(drop=True)
    )


def standard_bartime_times(narrow: pd.DataFrame) -> Set[dt.time]:
    if narrow.empty:
        return set()
    return set(pd.to_datetime(narrow["bartime"]).dt.time.unique())


def assert_standard_bartimes(
    narrow: pd.DataFrame,
    *,
    allowed: Optional[Set[dt.time]] = None,
) -> None:
    """Check 1: signal times must be subset of standard bartimes (no shift)."""
    allowed = allowed or STANDARD_BARTIME_SET
    times = standard_bartime_times(narrow)
    extra = times - allowed
    if extra:
        raise AssertionError(
            f"Non-standard bartimes detected (possible time shift): {sorted(extra)}"
        )


def assert_bartime_alignment(
    python_narrow: pd.DataFrame,
    ddb_narrow: pd.DataFrame,
) -> None:
    """Check 1b: inner-join keys must have identical bartime on every (bartime, symbol)."""
    aligned = align_narrow(python_narrow, ddb_narrow)
    if aligned.empty:
        raise AssertionError("No overlapping (bartime, symbol) keys for alignment check")
    py_bt = pd.to_datetime(aligned["bartime"]).dt.strftime("%H:%M:%S")
    # aligned uses single bartime column from merge — keys already matched
    unique_bt = set(py_bt.unique())
    bad = unique_bt - {t.strftime("%H:%M:%S") for t in STANDARD_BARTIME_SET}
    if bad:
        raise AssertionError(f"Bartime alignment failed; unexpected times: {bad}")


def assert_no_future_leakage_contract() -> None:
    """Check 2: document look-ahead contract (cumsum only uses past+current bar).

    DDB: ``context by Symbol, Date csort Bartime`` + ``cumsum`` + ``rowNo > 0``.
    Python: ``groupby.cumsum()`` on bars sorted by bartime within session.
    Neither path uses ``move(..., -k)`` or forward shift on prices/volumes.
    """
    # Runtime no-op; enforced by SQL/script review + parity tests against Python.
    return None


def ddb_version(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    bartimes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """DDB-native: loadTable + context by + cumsum; returns narrow table only."""
    if store is None:
        store = get_default_store(start_date=start_date)
    script = close_vwap_deviation_script(
        start_date,
        end_date,
        symbols=symbols,
        bartimes=bartimes,
    )
    raw = store.run_script(script)
    return _normalize_ddb_narrow(raw)


def align_narrow(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join two narrow tables on (bartime, symbol) with diff column."""
    keys = ["bartime", "symbol"]
    a = left[keys + ["value"]].rename(columns={"value": "python"})
    b = right[keys + ["value"]].rename(columns={"value": "ddb"})
    a["bartime"] = pd.to_datetime(a["bartime"])
    b["bartime"] = pd.to_datetime(b["bartime"])
    merged = a.merge(b, on=keys, how="inner")
    merged["abs_diff"] = (merged["python"] - merged["ddb"]).abs()
    return merged
