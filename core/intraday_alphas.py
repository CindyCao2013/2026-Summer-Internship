"""True intraday alpha factors — Phase 2 (first batch).

Each factor reads canonical minute bars via MinuteBarStore and returns a
narrow-format DataFrame::

    bartime | symbol | factorname | value

Prices/amounts are multiplied by ``adjfactor`` before aggregation.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from minute_bar_store import MinuteBarStore, apply_trading_hours, get_default_store

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

STANDARD_BARTIMES: List[dt.time] = [
    dt.time(9, 59),
    dt.time(10, 29),
    dt.time(11, 29),
    dt.time(13, 29),
    dt.time(14, 29),
]


def _default_store(start_date: DateLike) -> MinuteBarStore:
    return get_default_store(start_date=pd.Timestamp(start_date))


def _safe_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Multiply price/amount columns by adjfactor (copy)."""
    out = df.copy()
    if "adjfactor" not in out.columns:
        return out
    adj = out["adjfactor"].replace(0, np.nan).fillna(1.0)
    for col in (
        "open",
        "high",
        "low",
        "close",
        "amount",
        "active_buy_amt",
        "active_sell_amt",
    ):
        if col in out.columns:
            out[col] = out[col] * adj
    return out


def _cum_vwap_until(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from session start up to each row."""
    df = df.sort_values(["symbol", "date", "bartime"])
    grp = df.groupby(["symbol", "date"], sort=False)
    cum_amt = grp["amount"].cumsum()
    cum_vol = grp["volume"].cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return cum_amt / cum_vol.replace(0, np.nan)


def _filter_standard_bartimes(
    df: pd.DataFrame,
    bartimes: Sequence[dt.time] = STANDARD_BARTIMES,
) -> pd.DataFrame:
    times = pd.to_datetime(df["bartime"]).dt.time
    return df.loc[times.isin(list(bartimes))].copy()


def _empty_narrow() -> pd.DataFrame:
    return pd.DataFrame(columns=["bartime", "symbol", "factorname", "value"])


def _close_vwap_use_ddb() -> bool:
    import os

    env = os.environ.get("INTRADAY_CLOSE_VWAP_USE_DDB")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import factor_config as cfg

        return bool(getattr(cfg, "INTRADAY_CLOSE_VWAP_USE_DDB", False))
    except Exception:  # noqa: BLE001
        return False


def _compute_close_vwap_deviation_python(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Python reference: (Close − cumulative VWAP) / VWAP up to each bartime."""
    if store is None:
        store = _default_store(start_date)
    raw = store.get_data(start_date, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()

    df = apply_trading_hours(raw)
    df = _safe_adjust(df)
    df = df.sort_values(["symbol", "date", "bartime"])
    df["cum_vwap"] = _cum_vwap_until(df)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["value"] = (df["close"] - df["cum_vwap"]) / df["cum_vwap"]
    bad_vol = df["volume"].fillna(0) <= 0
    df.loc[bad_vol, "value"] = np.nan
    first = df.groupby(["symbol", "date"], sort=False).cumcount() == 0
    df.loc[first, "value"] = np.nan

    if not return_full_day:
        df = _filter_standard_bartimes(df)

    out = df[["bartime", "symbol"]].copy()
    out["factorname"] = "close_vwap_deviation"
    out["value"] = df["value"].astype(float)
    return out.dropna(subset=["value"]).reset_index(drop=True)


def compute_close_vwap_deviation(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """(Close − cumulative VWAP) / VWAP — dispatches Python or DDB via config flag."""
    if _close_vwap_use_ddb() and not return_full_day:
        from factors.intraday.close_vwap_deviation.compute import ddb_version

        return ddb_version(
            start_date,
            end_date,
            store=store,
            symbols=symbols,
        )
    return _compute_close_vwap_deviation_python(
        start_date,
        end_date,
        store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def _active_buy_sell_imbalance_use_ddb() -> bool:
    import os

    env = os.environ.get("INTRADAY_ACTIVE_BUY_SELL_IMBALANCE_USE_DDB")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import factor_config as cfg

        return bool(
            getattr(cfg, "INTRADAY_ACTIVE_BUY_SELL_IMBALANCE_USE_DDB", False)
        )
    except Exception:  # noqa: BLE001
        return False


def _compute_active_buy_sell_imbalance_python(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Cumulative aggressive-flow imbalance available at each signal time."""
    if store is None:
        store = _default_store(start_date)
    raw = store.get_data(start_date, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()

    df = _safe_adjust(apply_trading_hours(raw))
    df = df.sort_values(["symbol", "date", "bartime"])
    df["active_buy_amt"] = pd.to_numeric(
        df["active_buy_amt"], errors="coerce"
    ).fillna(0.0)
    df["active_sell_amt"] = pd.to_numeric(
        df["active_sell_amt"], errors="coerce"
    ).fillna(0.0)
    grp = df.groupby(["symbol", "date"], sort=False)
    df["cum_buy"] = grp["active_buy_amt"].cumsum()
    df["cum_sell"] = grp["active_sell_amt"].cumsum()
    total = df["cum_buy"] + df["cum_sell"]
    with np.errstate(invalid="ignore", divide="ignore"):
        df["value"] = (df["cum_buy"] - df["cum_sell"]) / total.replace(0, np.nan)

    if not return_full_day:
        df = _filter_standard_bartimes(df)
    out = df[["bartime", "symbol", "value"]].copy()
    out["factorname"] = "active_buy_sell_imbalance"
    return out[
        ["bartime", "symbol", "factorname", "value"]
    ].dropna(subset=["value"]).reset_index(drop=True)


def compute_active_buy_sell_imbalance(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Dispatch cumulative L2 aggressive-flow imbalance."""
    if _active_buy_sell_imbalance_use_ddb() and not return_full_day:
        from factors.intraday.active_buy_sell_imbalance.compute import ddb_version

        return ddb_version(
            start_date,
            end_date,
            store=store,
            symbols=symbols,
        )
    return _compute_active_buy_sell_imbalance_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def _late_session_strength_use_ddb() -> bool:
    import os

    env = os.environ.get("INTRADAY_LATE_SESSION_STRENGTH_USE_DDB")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import factor_config as cfg

        return bool(
            getattr(cfg, "INTRADAY_LATE_SESSION_STRENGTH_USE_DDB", False)
        )
    except Exception:  # noqa: BLE001
        return False


def _compute_late_session_strength_python(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Active-buy share in 14:30–15:00, stamped at *next* trading day 09:59.

    Uses only day-T close-session data; predicts returns from day T+1 09:59 onward
    (no same-day look-ahead). ``return_full_day`` kept for API compat (ignored).
    """
    del return_full_day
    # Need prior session tails so T+1 stamps cover [start_date, end_date]
    raw_start = pd.Timestamp(start_date) - pd.Timedelta(days=10)
    if store is None:
        store = _default_store(raw_start)
    raw = store.get_data(raw_start, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()

    df = apply_trading_hours(raw)
    df = _safe_adjust(df)
    t = pd.to_datetime(df["bartime"]).dt.time
    tail = df.loc[(t >= dt.time(14, 30)) & (t <= dt.time(15, 0))]
    if tail.empty:
        return _empty_narrow()

    grp = tail.groupby(["symbol", "date"], sort=False)
    buy = grp["active_buy_amt"].sum()
    sell = grp["active_sell_amt"].sum()
    total = buy + sell
    with np.errstate(invalid="ignore", divide="ignore"):
        strength = (buy / total.replace(0, np.nan)).rename("value")

    strength = strength.reset_index()
    strength["date"] = pd.to_datetime(strength["date"])
    # Next business day (holiday gaps ≈ ok for research; exact calendar optional later)
    strength["date"] = strength["date"] + pd.offsets.BDay(1)
    strength["bartime"] = strength["date"] + pd.Timedelta(hours=9, minutes=59)
    strength["factorname"] = "late_session_strength"
    out = strength[["bartime", "symbol", "factorname", "value"]].dropna(subset=["value"])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    out = out[(out["bartime"] >= start_ts) & (out["bartime"] <= end_ts)]
    return out.reset_index(drop=True)


def compute_late_session_strength(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """Dispatch late-session active-buy share to DDB or Python reference."""
    if _late_session_strength_use_ddb():
        from factors.intraday.late_session_strength.compute import ddb_version

        return ddb_version(
            start_date,
            end_date,
            store=store,
            symbols=symbols,
        )
    return _compute_late_session_strength_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
    )


def _volume_front_use_ddb() -> bool:
    import os

    env = os.environ.get("INTRADAY_VOLUME_FRONT_USE_DDB")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import factor_config as cfg

        return bool(getattr(cfg, "INTRADAY_VOLUME_FRONT_USE_DDB", False))
    except Exception:  # noqa: BLE001
        return False


def _compute_volume_front_loading_python(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """早盘量比（信号打在 10:29，与 PREHEAT 标准时点对齐）::

        volume(09:30–10:00) / mean(volume(09:30–10:00) over prior N trading days)

    因子在 10:00 已知；打点 10:29 便于与 ret_matrix / 标准 bartime 对齐。
    仅用每日前 30 分钟 bar，避免全日 groupby-rolling。``return_full_day`` 保留接口兼容，忽略。
    """
    del return_full_day  # API compat with other computers
    hist_start = pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days + 10)
    if store is None:
        store = _default_store(hist_start)

    raw = store.get_data(hist_start, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()

    df = apply_trading_hours(raw)
    t = pd.to_datetime(df["bartime"]).dt.time
    mask = (t >= dt.time(9, 30)) & (t <= dt.time(10, 0))
    df = df.loc[mask].copy()
    if df.empty:
        return _empty_narrow()

    daily_vol = (
        df.groupby(["symbol", "date"], sort=False)["volume"]
        .sum()
        .reset_index()
    )
    daily_vol["date"] = pd.to_datetime(daily_vol["date"])
    daily_vol = daily_vol.sort_values(["symbol", "date"])

    min_p = max(5, lookback_days // 2)
    daily_vol["hist_avg"] = daily_vol.groupby("symbol", sort=False)["volume"].transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=min_p).mean()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_vol["value"] = daily_vol["volume"] / daily_vol["hist_avg"].replace(0, np.nan)

    daily_vol = daily_vol[
        (daily_vol["date"] >= pd.Timestamp(start_date))
        & (daily_vol["date"] <= pd.Timestamp(end_date))
    ]
    daily_vol["bartime"] = daily_vol["date"] + pd.Timedelta(hours=10, minutes=29)
    out = daily_vol[["bartime", "symbol", "value"]].copy()
    out["factorname"] = "volume_front_loading"
    return out.dropna(subset=["value"]).reset_index(drop=True)


def compute_volume_front_loading(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Dispatch early-session volume ratio to DDB-native or Python reference."""
    if _volume_front_use_ddb():
        from factors.intraday.volume_front_loading.compute import ddb_version

        return ddb_version(
            start_date,
            end_date,
            store=store,
            symbols=symbols,
            lookback_days=lookback_days,
        )
    return _compute_volume_front_loading_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
        lookback_days=lookback_days,
    )


def _volume_back_use_ddb() -> bool:
    import os

    env = os.environ.get("INTRADAY_VOLUME_BACK_USE_DDB")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import factor_config as cfg

        return bool(getattr(cfg, "INTRADAY_VOLUME_BACK_USE_DDB", False))
    except Exception:  # noqa: BLE001
        return False


def _compute_volume_back_loading_python(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """尾盘量比（信号打在次日 09:59）::

        volume(14:30–15:00) / mean(volume(14:30–15:00) over prior N trading days)

    因子在当日 15:00 已知；打点 T+1 09:59 避免同日 look-ahead。
    ``return_full_day`` 保留接口兼容，忽略。
    """
    del return_full_day
    hist_start = pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days + 10)
    if store is None:
        store = _default_store(hist_start)

    raw = store.get_data(hist_start, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()

    df = apply_trading_hours(raw)
    t = pd.to_datetime(df["bartime"]).dt.time
    mask = (t >= dt.time(14, 30)) & (t <= dt.time(15, 0))
    df = df.loc[mask].copy()
    if df.empty:
        return _empty_narrow()

    daily_vol = (
        df.groupby(["symbol", "date"], sort=False)["volume"]
        .sum()
        .reset_index()
    )
    daily_vol["date"] = pd.to_datetime(daily_vol["date"])
    daily_vol = daily_vol.sort_values(["symbol", "date"])

    min_p = max(5, lookback_days // 2)
    daily_vol["hist_avg"] = daily_vol.groupby("symbol", sort=False)["volume"].transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=min_p).mean()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        daily_vol["value"] = daily_vol["volume"] / daily_vol["hist_avg"].replace(0, np.nan)

    daily_vol = daily_vol[
        (daily_vol["date"] >= pd.Timestamp(start_date))
        & (daily_vol["date"] <= pd.Timestamp(end_date))
    ]
    daily_vol["bartime"] = daily_vol["date"] + pd.offsets.BDay(1) + pd.Timedelta(
        hours=9, minutes=59
    )
    out = daily_vol[["bartime", "symbol", "value"]].copy()
    out["factorname"] = "volume_back_loading"
    out = out.dropna(subset=["value"])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
    out = out[(out["bartime"] >= start_ts) & (out["bartime"] <= end_ts)]
    return out.reset_index(drop=True)


def compute_volume_back_loading(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Dispatch late-session volume ratio to DDB-native or Python reference."""
    if _volume_back_use_ddb():
        from factors.intraday.volume_back_loading.compute import ddb_version

        return ddb_version(
            start_date,
            end_date,
            store=store,
            symbols=symbols,
            lookback_days=lookback_days,
        )
    return _compute_volume_back_loading_python(
        start_date,
        end_date,
        store=store,
        symbols=symbols,
        return_full_day=return_full_day,
        lookback_days=lookback_days,
    )


def compute_morning_reversal_pressure(
    start_date: DateLike,
    end_date: DateLike,
    store: Optional[MinuteBarStore] = None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """早盘动量（信号打在 10:29）::

        (close_10:00 − open_first_bar) / open_first_bar

    正值 = 早盘上涨。短样本上原「下跌=正」定义 HML 为负，故改为早盘收益方向。
    ``return_full_day`` 保留接口兼容。
    """
    del return_full_day
    if store is None:
        store = _default_store(start_date)
    raw = store.get_data(start_date, end_date, symbols=symbols)
    if raw.empty:
        return _empty_narrow()

    df = apply_trading_hours(raw)
    df = _safe_adjust(df)
    df = df.sort_values(["symbol", "date", "bartime"])

    first_bar = (
        df.groupby(["symbol", "date"], sort=False)
        .first()[["open"]]
        .reset_index()
        .rename(columns={"open": "open_price"})
    )
    ten_oclock = df.loc[
        pd.to_datetime(df["bartime"]).dt.time == dt.time(10, 0),
        ["symbol", "date", "close"],
    ].rename(columns={"close": "close_10"})

    merged = first_bar.merge(ten_oclock, on=["symbol", "date"], how="inner")
    with np.errstate(invalid="ignore", divide="ignore"):
        merged["value"] = (merged["close_10"] - merged["open_price"]) / merged[
            "open_price"
        ].replace(0, np.nan)

    merged["bartime"] = pd.to_datetime(merged["date"]) + pd.Timedelta(hours=10, minutes=29)
    merged["factorname"] = "morning_reversal_pressure"
    return (
        merged.dropna(subset=["value"])[
            ["bartime", "symbol", "factorname", "value"]
        ].reset_index(drop=True)
    )


# --------------------- 时点固定版单因子（读日频面板缓存） ---------------------

def _load_panel(
    factor_name: str,
    start: DateLike,
    end: DateLike,
    shift_days: int = 1,
) -> pd.DataFrame:
    """加载日频宽表并做 shift，输出 Date×Symbol。"""
    from intraday_heatmap_lib import load_factor_panel_from_cache

    panel = load_factor_panel_from_cache(factor_name, start, end)
    panel = panel.sort_index()
    if shift_days:
        panel = panel.shift(shift_days)
    return panel


def _narrow_from_panel(
    panel: pd.DataFrame,
    factor_name: str,
    hour: int,
    minute: int,
    start: DateLike,
    end: DateLike,
) -> pd.DataFrame:
    """将面板在指定时点展开为窄表（bartime|symbol|factorname|value）。"""
    dates = panel.index[
        (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    ]
    if len(dates) == 0:
        return _empty_narrow()
    sliced = panel.loc[dates]
    stacked = sliced.stack().reset_index()
    stacked.columns = ["Date", "symbol", "value"]
    stacked["bartime"] = pd.to_datetime(stacked["Date"]) + pd.Timedelta(
        hours=hour, minutes=minute
    )
    stacked["factorname"] = factor_name
    stacked = stacked.dropna(subset=["value"])
    stacked["value"] = stacked["value"].astype(float)
    sym = stacked["symbol"].astype(str)
    stacked = stacked[sym.str[0].isin(("6", "0", "3"))]
    return stacked[["bartime", "symbol", "factorname", "value"]].reset_index(drop=True)


def compute_tgd20_1429(
    start_date: DateLike,
    end_date: DateLike,
    store=None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """TGD20 仅在 14:29 取值（面板已 shift 1 日，避免同日偷看）。"""
    panel = _load_panel("TGD20", start_date, end_date, shift_days=1)
    if symbols is not None:
        keep = [c for c in panel.columns if str(c) in set(map(str, symbols))]
        panel = panel[keep]
    return _narrow_from_panel(panel, "TGD20_1429", 14, 29, start_date, end_date)


def compute_smartmoney_1129_rev(
    start_date: DateLike,
    end_date: DateLike,
    store=None,
    *,
    symbols: Optional[List[str]] = None,
    return_full_day: bool = False,
) -> pd.DataFrame:
    """SmartMoneyActiveV2 在 11:29 取负（面板 shift 1 日）。"""
    panel = _load_panel("SmartMoneyActiveV2", start_date, end_date, shift_days=1)
    if symbols is not None:
        keep = [c for c in panel.columns if str(c) in set(map(str, symbols))]
        panel = panel[keep]
    narrow = _narrow_from_panel(
        panel, "SmartMoney_1129_Rev", 11, 29, start_date, end_date
    )
    if not narrow.empty:
        narrow = narrow.copy()
        narrow["value"] = -narrow["value"]
    return narrow


def _discovery_v1_computer(factor_name: str):
    """Bind a discovery factor to the existing intraday computer signature."""
    from factors.intraday.discovery_v1 import compute_factor

    def _compute(
        start_date: DateLike,
        end_date: DateLike,
        store=None,
        *,
        symbols: Optional[List[str]] = None,
        return_full_day: bool = False,
    ) -> pd.DataFrame:
        return compute_factor(
            factor_name,
            start_date,
            end_date,
            store=store,
            symbols=symbols,
            return_full_day=return_full_day,
        )

    _compute.__name__ = f"compute_{factor_name}"
    return _compute


compute_bartime_ofi = _discovery_v1_computer("bartime_ofi")
compute_ofi_persistence = _discovery_v1_computer("ofi_persistence")
compute_active_buy_shock = _discovery_v1_computer("active_buy_shock")
compute_average_active_trade_size = _discovery_v1_computer(
    "average_active_trade_size"
)
compute_large_active_buy_ratio = _discovery_v1_computer(
    "large_active_buy_ratio"
)
compute_intraday_amihud = _discovery_v1_computer("intraday_amihud")
compute_realized_volatility = _discovery_v1_computer("realized_volatility")
compute_minute_skew = _discovery_v1_computer("minute_skew")


# Registry used by intraday_formulas
INTRADAY_ALPHA_COMPUTERS = {
    "close_vwap_deviation": compute_close_vwap_deviation,
    "active_buy_sell_imbalance": compute_active_buy_sell_imbalance,
    "late_session_strength": compute_late_session_strength,
    "volume_front_loading": compute_volume_front_loading,
    "volume_back_loading": compute_volume_back_loading,
    "morning_reversal_pressure": compute_morning_reversal_pressure,
    "TGD20_1429": compute_tgd20_1429,
    "SmartMoney_1129_Rev": compute_smartmoney_1129_rev,
    "bartime_ofi": compute_bartime_ofi,
    "ofi_persistence": compute_ofi_persistence,
    "active_buy_shock": compute_active_buy_shock,
    "average_active_trade_size": compute_average_active_trade_size,
    "large_active_buy_ratio": compute_large_active_buy_ratio,
    "intraday_amihud": compute_intraday_amihud,
    "realized_volatility": compute_realized_volatility,
    "minute_skew": compute_minute_skew,
}

# Single source of truth for factor execution backends.
INTRADAY_FACTOR_BACKEND = {
    "close_vwap_deviation": "ddb",
    "active_buy_sell_imbalance": "ddb",
    "late_session_strength": "ddb",
    "volume_front_loading": "ddb",
    "volume_back_loading": "ddb",
    "morning_reversal_pressure": "python",
    "TGD20_1429": "panel",
    "SmartMoney_1129_Rev": "panel",
    "bartime_ofi": "ddb",
    "ofi_persistence": "ddb",
    "active_buy_shock": "ddb",
    "average_active_trade_size": "ddb",
    "large_active_buy_ratio": "ddb",
    "intraday_amihud": "ddb",
    "realized_volatility": "ddb",
    "minute_skew": "ddb",
}

# Backward-compatible derived view: keeps Intraday_Factor_Test_Process.py unchanged.
PANEL_BASED_INTRADAY_FACTORS = frozenset(
    name for name, backend in INTRADAY_FACTOR_BACKEND.items() if backend == "panel"
)


def narrow_for_ddb(df: pd.DataFrame) -> pd.DataFrame:
    """Rename bartime→tradetime for Intraday_Factor_Test_Process / DDB upload."""
    if df.empty:
        return pd.DataFrame(columns=["tradetime", "symbol", "factorname", "value"])
    out = df.rename(columns={"bartime": "tradetime"})
    return out[["tradetime", "symbol", "factorname", "value"]].copy()
