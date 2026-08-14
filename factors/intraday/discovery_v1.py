"""Shared implementation for Intraday Alpha Expansion v1 candidates.

The eight factors share one normalization, timestamp and parity contract while
retaining per-factor packages as their research records. Python is the golden
reference; DolphinDB is the production computation path after parity passes.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from core.ddb_intraday_queries import (
    DISCOVERY_V1_FACTORS,
    discovery_v1_factor_script,
)
from minute_bar_store import (
    MinuteBarStore,
    apply_trading_hours,
    filter_a_share,
    get_default_store,
    to_wind_code,
)

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]
NARROW_COLUMNS = ("bartime", "symbol", "factorname", "value")
STANDARD_BARTIMES: Sequence[dt.time] = (
    dt.time(9, 59),
    dt.time(10, 29),
    dt.time(11, 29),
    dt.time(13, 29),
    dt.time(14, 29),
)
WINDOW = 20
MIN_PERIODS = 10


def _empty_narrow() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NARROW_COLUMNS))


def _rolling(
    values: pd.Series,
    group_keys: list[pd.Series],
    *,
    window: int,
    min_periods: int,
    method: str,
) -> pd.Series:
    grouped = values.groupby(group_keys, sort=False)
    roller = grouped.rolling(window, min_periods=min_periods)
    result = getattr(roller, method)()
    return result.reset_index(level=list(range(len(group_keys))), drop=True)


def _rolling_sample_std(
    values: pd.Series,
    group_keys: list[pd.Series],
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Two-pass sample std matching DolphinDB ``mstd`` numerics."""
    result = (
        values.groupby(group_keys, sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(lambda x: np.nanstd(x, ddof=1), raw=True)
    )
    return result.reset_index(level=list(range(len(group_keys))), drop=True)


def _rolling_mean_exact(
    values: pd.Series,
    group_keys: list[pd.Series],
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Two-pass-compatible mean without cross-group rolling state residue."""
    result = (
        values.groupby(group_keys, sort=False)
        .rolling(window, min_periods=min_periods)
        .apply(np.nanmean, raw=True)
    )
    return result.reset_index(level=list(range(len(group_keys))), drop=True)


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = apply_trading_hours(raw)
    df = df.sort_values(["symbol", "date", "bartime"]).reset_index(drop=True)
    for col in (
        "active_buy_amt",
        "active_sell_amt",
        "active_buy_count",
        "active_sell_count",
        "amount",
        "close",
    ):
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[["active_buy_amt", "active_sell_amt"]] = df[
        ["active_buy_amt", "active_sell_amt"]
    ].fillna(0.0).clip(lower=0.0)
    df[["active_buy_count", "active_sell_count", "amount"]] = df[
        ["active_buy_count", "active_sell_count", "amount"]
    ].fillna(0.0).clip(lower=0.0)
    keys = [df["symbol"], df["date"]]
    flow_total = df["active_buy_amt"] + df["active_sell_amt"]
    df["bar_ofi"] = np.where(
        flow_total > 0,
        (df["active_buy_amt"] - df["active_sell_amt"]) / flow_total,
        np.nan,
    )
    df["buy_size"] = np.where(
        df["active_buy_count"] > 0,
        df["active_buy_amt"] / df["active_buy_count"],
        np.nan,
    )
    df["minute_ret"] = df.groupby(
        ["symbol", "date"], sort=False
    )["close"].pct_change(fill_method=None)
    df.attrs["group_keys"] = keys
    return df


def _factor_values(df: pd.DataFrame, factor_name: str) -> pd.Series:
    if factor_name not in DISCOVERY_V1_FACTORS:
        raise KeyError(f"Unknown discovery factor: {factor_name}")
    keys = [df["symbol"], df["date"]]
    if factor_name == "bartime_ofi":
        return df["bar_ofi"]
    if factor_name == "ofi_persistence":
        positive = df["bar_ofi"].gt(0).astype(float).where(df["bar_ofi"].notna())
        numerator = _rolling(
            positive,
            keys,
            window=WINDOW,
            min_periods=5,
            method="sum",
        )
        denominator = _rolling(
            df["bar_ofi"].notna().astype(float),
            keys,
            window=WINDOW,
            min_periods=5,
            method="sum",
        )
        return numerator / denominator.replace(0, np.nan)
    if factor_name == "active_buy_shock":
        hist_mean = _rolling_mean_exact(
            df["active_buy_amt"],
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
        ).groupby(keys, sort=False).shift(1)
        hist_std = _rolling_sample_std(
            df["active_buy_amt"],
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
        ).groupby(keys, sort=False).shift(1)
        stable = hist_std > np.maximum(hist_mean.abs() * 1e-8, 1.0)
        return ((df["active_buy_amt"] - hist_mean) / hist_std).where(stable)
    if factor_name == "average_active_trade_size":
        hist_mean = _rolling_mean_exact(
            df["buy_size"],
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
        ).groupby(keys, sort=False).shift(1)
        return df["buy_size"] / hist_mean.replace(0, np.nan) - 1.0
    if factor_name == "large_active_buy_ratio":
        hist_mean = _rolling_mean_exact(
            df["buy_size"],
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
        ).groupby(keys, sort=False).shift(1)
        hist_std = _rolling_sample_std(
            df["buy_size"],
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
        ).groupby(keys, sort=False).shift(1)
        valid = hist_mean.notna() & hist_std.notna()
        large = valid & df["buy_size"].gt(hist_mean + hist_std)
        large_amount = df["active_buy_amt"].where(large, 0.0)
        numerator = _rolling(
            large_amount,
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
            method="sum",
        )
        denominator = _rolling(
            df["active_buy_amt"],
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
            method="sum",
        )
        valid_count = _rolling(
            valid.astype(float).where(valid),
            keys,
            window=WINDOW,
            min_periods=MIN_PERIODS,
            method="count",
        )
        return (numerator / denominator).where(
            (valid_count >= MIN_PERIODS) & (denominator > 1.0)
        )
    if factor_name == "intraday_amihud":
        abs_ret = _rolling(
            df["minute_ret"].abs(),
            keys,
            window=5,
            min_periods=3,
            method="sum",
        )
        amount = _rolling(
            df["amount"],
            keys,
            window=5,
            min_periods=3,
            method="sum",
        )
        return (abs_ret / amount).where(amount > 1.0)
    if factor_name == "realized_volatility":
        valid_count = df["minute_ret"].notna().groupby(keys, sort=False).cumsum()
        rv = (
            df["minute_ret"]
            .fillna(0.0)
            .pow(2)
            .groupby(keys, sort=False)
            .cumsum()
            .pow(0.5)
        )
        return rv.where(valid_count >= 5)
    valid = df["minute_ret"].notna()
    ret = df["minute_ret"].fillna(0.0)
    n = valid.astype(float).groupby(keys, sort=False).cumsum()
    s1 = ret.groupby(keys, sort=False).cumsum()
    s2 = ret.pow(2).groupby(keys, sort=False).cumsum()
    s3 = ret.pow(3).groupby(keys, sort=False).cumsum()
    m2 = s2 - s1.pow(2) / n.replace(0, np.nan)
    m3 = s3 - 3.0 * s1 * s2 / n.replace(0, np.nan)
    m3 += 2.0 * s1.pow(3) / n.pow(2).replace(0, np.nan)
    skew = (
        n
        * np.sqrt(n - 1.0)
        / (n - 2.0)
        * m3
        / m2.pow(1.5)
    )
    return skew.where((n >= 3) & (m2 > 0))


def python_version(
    factor_name: str,
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    if store is None:
        store = get_default_store(start_date=start_date)
    raw = store.get_data(start_date, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()
    df = _prepare(raw)
    df["value"] = _factor_values(df, factor_name)
    if not return_full_day:
        times = pd.to_datetime(df["bartime"]).dt.time
        df = df[times.isin(STANDARD_BARTIMES)]
    out = df[["bartime", "symbol", "value"]].copy()
    out["factorname"] = factor_name
    return (
        out[list(NARROW_COLUMNS)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["value"])
        .sort_values(["bartime", "symbol"])
        .reset_index(drop=True)
    )


def _normalize_ddb_narrow(raw: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return _empty_narrow()
    df = raw.copy().rename(columns={"Symbol": "symbol", "Bartime": "bartime"})
    df["symbol"] = df["symbol"].map(to_wind_code)
    df = filter_a_share(df, include_bj=False)
    df["bartime"] = pd.to_datetime(df["bartime"])
    df["factorname"] = factor_name
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return (
        df[list(NARROW_COLUMNS)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["value"])
        .sort_values(["bartime", "symbol"])
        .reset_index(drop=True)
    )


def ddb_version(
    factor_name: str,
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
        discovery_v1_factor_script(
            factor_name,
            start_date,
            end_date,
            symbols=symbols,
            bartimes=bartimes,
        )
    )
    return _normalize_ddb_narrow(raw, factor_name)


def use_ddb(factor_name: str) -> bool:
    stem = factor_name.upper()
    key = f"{stem}_USE_DDB" if stem.startswith("INTRADAY_") else f"INTRADAY_{stem}_USE_DDB"
    env = os.environ.get(key)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import factor_config as cfg

        return bool(getattr(cfg, key, False))
    except Exception:  # noqa: BLE001
        return False


def compute_factor(
    factor_name: str,
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    if use_ddb(factor_name) and not return_full_day:
        return ddb_version(
            factor_name,
            start_date,
            end_date,
            store=store,
            symbols=symbols,
        )
    return python_version(
        factor_name,
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def assert_standard_bartimes(narrow: pd.DataFrame) -> None:
    if narrow.empty:
        return
    actual = set(pd.to_datetime(narrow["bartime"]).dt.time.unique())
    extra = actual - set(STANDARD_BARTIMES)
    if extra:
        raise AssertionError(f"Unexpected signal times: {sorted(extra)}")


def assert_bartime_alignment(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert_standard_bartimes(left)
    assert_standard_bartimes(right)
    left_times = set(pd.to_datetime(left["bartime"]).unique())
    right_times = set(pd.to_datetime(right["bartime"]).unique())
    if left_times != right_times:
        raise AssertionError(
            f"Signal-time mismatch: python_only={len(left_times - right_times)}, "
            f"ddb_only={len(right_times - left_times)}"
        )


def align_narrow(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["bartime", "symbol"]
    a = left[keys + ["value"]].rename(columns={"value": "python"}).copy()
    b = right[keys + ["value"]].rename(columns={"value": "ddb"}).copy()
    a["bartime"] = pd.to_datetime(a["bartime"])
    b["bartime"] = pd.to_datetime(b["bartime"])
    merged = a.merge(b, on=keys, how="inner")
    merged["abs_diff"] = (merged["python"] - merged["ddb"]).abs()
    return merged


def assert_no_future_leakage_contract(factor_name: str) -> None:
    script = discovery_v1_factor_script(
        factor_name, "2024-05-01", "2024-05-31"
    )
    required = ["context by Symbol, Date csort Bartime", "where Bartime in btFilter"]
    if factor_name in {
        "active_buy_shock",
        "average_active_trade_size",
        "large_active_buy_ratio",
    }:
        required.append("move(m")
    if factor_name in {"realized_volatility", "minute_skew"}:
        required.append("cumsum(")
    missing = [fragment for fragment in required if fragment not in script]
    if missing:
        raise AssertionError(f"No-look-ahead SQL contract missing: {missing}")
    forbidden = ("move(mavg(buy_amt, 20, 10), -", "move(mstd(buy_amt, 20, 10), -")
    if any(fragment in script for fragment in forbidden):
        raise AssertionError("Negative/forward move is forbidden")
