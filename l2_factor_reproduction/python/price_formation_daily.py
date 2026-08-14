"""Canonical minute bars -> reusable Price Formation daily primitive.

The production extractor aggregates in DolphinDB and returns symbol-day rows.
``compute_price_formation_daily`` is the small-sample Python reference used by
tests and source validation. Both paths freeze:

- continuous grid: [09:30, 11:30) and [13:00, 15:00), 240 labels;
- 15:00 close auction stored separately;
- no price fill across lunch;
- at most three consecutive in-session price-state fills;
- amount and volume are never filled;
- realized moments use observed-to-observed returns only.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from core.ddb.connection import get_ddb_session, is_shared_session
from minute_bar_store import filter_a_share, normalize_bartime, to_wind_code


DateLike = Union[str, date, datetime, pd.Timestamp]

SCHEMA_VERSION = "l2_primitive_price_formation_daily_v1"
FORMULA_VERSION = "price_formation_level_formulas_v1"
CANONICAL_SOURCE = "dfs://QV_Trade_to_MinuteBar/Stock_one_minute"
EXPECTED_CONTINUOUS_MINUTES = 240
COVERAGE_THRESHOLD = 0.80
MAX_CONSECUTIVE_PRICE_GAP = 3
VARIANCE_RATIO_HORIZON = 5
TAIL_QUANTILE = 0.95

KEY_COLUMNS = ("symbol", "TradeDate")
AUDIT_COLUMNS = (
    "source_exchange",
    "valid_minute_count",
    "expected_minute_count",
    "coverage_ratio",
    "imputed_price_minute_count",
    "valid_return_minute_count",
    "valid_amihud_minute_count",
    "daily_volume",
    "daily_amount",
    "adjfactor",
)
PRICE_PATH_COLUMNS = (
    "open_price",
    "close_price",
    "high_price",
    "low_price",
    "daily_vwap",
    "continuous_close",
    "close_auction_price",
    "overnight_gap",
    "open_to_close_return",
    "open_to_30m_return",
    "morning_return",
    "afternoon_return",
    "closing_30m_return",
    "lunch_gap_return",
    "close_auction_return",
)
PATH_GEOMETRY_COLUMNS = (
    "close_location_value",
    "path_efficiency",
    "intraday_return_sign_persistence",
    "minute_return_autocorr1",
    "variance_ratio_5m",
    "max_drawup",
    "max_drawdown_intraday",
)
REALIZED_MOMENT_COLUMNS = (
    "realized_variance",
    "upside_semivariance",
    "downside_semivariance",
    "downside_semivariance_share",
    "realized_skewness",
    "realized_kurtosis",
    "bipower_variation",
    "jump_variation",
    "jump_share",
    "max_abs_minute_return",
    "tail_return_share",
)
VOLUME_TIMING_COLUMNS = (
    "opening_30m_amount_share",
    "closing_30m_amount_share",
    "morning_amount_share",
    "afternoon_amount_share",
    "volume_concentration_hhi",
    "amount_time_center",
    "volume_return_corr",
    "volume_abs_return_corr",
)
IMPACT_COLUMNS = (
    "intraday_amihud",
    "return_per_amount",
    "vwap_close_deviation",
    "high_low_range",
    "range_per_amount",
)
PRICE_FORMATION_DAILY_COLUMNS = (
    *KEY_COLUMNS,
    *AUDIT_COLUMNS,
    *PRICE_PATH_COLUMNS,
    *PATH_GEOMETRY_COLUMNS,
    *REALIZED_MOMENT_COLUMNS,
    *VOLUME_TIMING_COLUMNS,
    *IMPACT_COLUMNS,
)


def _as_day(value: DateLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.normalize()


def _ddb_day(value: DateLike) -> str:
    return _as_day(value).strftime("%Y.%m.%d")


def _safe_divide(numerator, denominator):
    try:
        denominator_value = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(denominator_value) or abs(denominator_value) <= 1e-30:
        return np.nan
    value = float(numerator) / denominator_value
    return value if np.isfinite(value) else np.nan


def _safe_log_ratio(numerator, denominator) -> float:
    try:
        left, right = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(left) or not np.isfinite(right) or left <= 0 or right <= 0:
        return np.nan
    return float(np.log(left / right))


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _continuous_grid(trade_date: pd.Timestamp) -> pd.DatetimeIndex:
    day = pd.Timestamp(trade_date).normalize()
    morning = pd.date_range(
        day + pd.Timedelta(hours=9, minutes=30),
        day + pd.Timedelta(hours=11, minutes=29),
        freq="min",
    )
    afternoon = pd.date_range(
        day + pd.Timedelta(hours=13),
        day + pd.Timedelta(hours=14, minutes=59),
        freq="min",
    )
    return morning.append(afternoon)


def _continuous_mask(times: pd.Series) -> pd.Series:
    values = pd.to_datetime(times)
    minute = values.dt.hour * 60 + values.dt.minute
    return minute.between(570, 689) | minute.between(780, 899)


def _normalize_minute_input(minute: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Symbol": "symbol",
        "Date": "date",
        "Bartime": "bartime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Amount": "amount",
        "Adjfactor": "adjfactor",
    }
    frame = minute.rename(
        columns={column: aliases[column] for column in minute if column in aliases}
    ).copy()
    required = {
        "symbol",
        "date",
        "bartime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adjfactor",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"minute input missing columns: {missing}")
    frame["symbol"] = frame["symbol"].map(to_wind_code)
    frame = filter_a_share(frame)
    frame = normalize_bartime(frame)
    for column in ("open", "high", "low", "close", "volume", "amount", "adjfactor"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["symbol", "bartime"]).any():
        raise ValueError("minute input contains duplicate symbol-minute keys")
    return frame.sort_values(["symbol", "bartime"], kind="stable").reset_index(
        drop=True
    )


def _fill_price_state(
    continuous: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fill price state inside each session only; preserve raw flow columns."""
    frame = continuous.copy()
    observed = frame["close"].gt(0) & frame["close"].notna()
    adjusted = pd.DataFrame(index=frame.index)
    for column in ("open", "high", "low", "close"):
        adjusted[column] = frame[column].where(frame[column] > 0) * frame[
            "adjfactor"
        ]
    state_close = adjusted["close"].copy()
    for start, stop in ((0, 120), (120, 240)):
        session_close = state_close.iloc[start:stop].ffill(
            limit=MAX_CONSECUTIVE_PRICE_GAP
        )
        state_close.iloc[start:stop] = session_close
    imputed = ~observed & state_close.notna()
    for column in ("open", "high", "low", "close"):
        adjusted.loc[imputed, column] = state_close.loc[imputed]
    adjusted["close"] = state_close
    return adjusted, observed, imputed


def _compute_symbol_day(
    symbol: str,
    trade_date: pd.Timestamp,
    block: pd.DataFrame,
) -> Dict[str, object]:
    day = pd.Timestamp(trade_date).normalize()
    grid = _continuous_grid(day)
    indexed = block.set_index("bartime").sort_index()
    continuous = indexed.reindex(grid)
    continuous["symbol"] = symbol
    continuous["date"] = day
    prices, observed, imputed = _fill_price_state(continuous)
    close = prices["close"]
    open_px = prices["open"]
    high = prices["high"]
    low = prices["low"]
    amount = pd.to_numeric(continuous["amount"], errors="coerce")
    volume = pd.to_numeric(continuous["volume"], errors="coerce")
    adjfactor_series = pd.to_numeric(
        continuous["adjfactor"], errors="coerce"
    )

    valid_return = observed & observed.shift(1, fill_value=False)
    returns = np.log(close / close.shift(1)).where(valid_return)
    abs_returns = returns.abs()
    lagged_returns = returns.shift(1)
    five_returns = np.log(close / close.shift(VARIANCE_RATIO_HORIZON)).where(
        observed & observed.shift(VARIANCE_RATIO_HORIZON, fill_value=False)
    )

    valid_minute_count = int(observed.sum())
    daily_amount = float(amount.fillna(0).sum())
    daily_volume = float(volume.fillna(0).sum())
    adjfactor = float(adjfactor_series.dropna().iloc[0]) if adjfactor_series.notna().any() else np.nan
    daily_vwap = _safe_divide(
        (amount * adjfactor_series).fillna(0).sum(),
        daily_volume,
    )
    open_price = float(open_px.iloc[0]) if pd.notna(open_px.iloc[0]) else np.nan
    continuous_close = (
        float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else np.nan
    )
    high_price = float(high.max()) if high.notna().any() else np.nan
    low_price = float(low.min()) if low.notna().any() else np.nan

    auction = indexed.loc[
        (indexed.index.hour == 15) & (indexed.index.minute == 0)
    ]
    if len(auction):
        auction_raw = pd.to_numeric(auction["close"], errors="coerce").iloc[-1]
        auction_adj = pd.to_numeric(
            auction["adjfactor"], errors="coerce"
        ).iloc[-1]
        close_auction_price = (
            float(auction_raw * auction_adj)
            if auction_raw > 0 and pd.notna(auction_adj)
            else np.nan
        )
    else:
        close_auction_price = np.nan
    close_price = (
        close_auction_price
        if np.isfinite(close_auction_price)
        else continuous_close
    )

    first_close = close.iloc[0]
    if close.notna().all() and pd.notna(first_close):
        path_length = abs(first_close - open_price)
        path_length += float(close.diff().abs().sum(skipna=True))
        path_efficiency = _safe_divide(
            abs(continuous_close - open_price), path_length
        )
    else:
        path_efficiency = np.nan

    running_max = close.cummax()
    running_min = close.cummin()
    max_drawdown = (1.0 - close / running_max).max()
    max_drawup = (close / running_min - 1.0).max()

    rv = float((returns**2).sum(min_count=1))
    upside = float(((returns.where(returns > 0)) ** 2).sum(min_count=1))
    downside = float(((returns.where(returns < 0)) ** 2).sum(min_count=1))
    return_count = int(returns.notna().sum())
    if return_count > 0 and np.isfinite(rv) and rv > 0:
        realized_skewness = (
            np.sqrt(return_count)
            * float((returns**3).sum())
            / (rv ** 1.5)
        )
        realized_kurtosis = (
            return_count * float((returns**4).sum()) / (rv**2)
        )
        tail_cut = float(abs_returns.quantile(TAIL_QUANTILE))
        tail_return_share = float(
            (returns.where(abs_returns >= tail_cut) ** 2).sum() / rv
        )
    else:
        realized_skewness = np.nan
        realized_kurtosis = np.nan
        tail_return_share = np.nan
    bipower = float(
        np.pi
        / 2.0
        * (abs_returns * lagged_returns.abs()).sum(min_count=1)
    )
    jump = max(rv - bipower, 0.0) if np.isfinite(rv) and np.isfinite(bipower) else np.nan

    positive_amount = amount > 0
    amihud_values = (abs_returns / amount).where(
        positive_amount & returns.notna()
    )
    minute_index = pd.Series(np.arange(EXPECTED_CONTINUOUS_MINUTES), index=grid)
    amount_weights = amount.fillna(0)
    amount_share = amount_weights / daily_amount if daily_amount > 0 else amount_weights * np.nan

    high_low_range = _safe_divide(high_price - low_price, open_price)
    row: Dict[str, object] = {
        "symbol": symbol,
        "TradeDate": day,
        "source_exchange": "SSE" if symbol.endswith(".SH") else "SZSE",
        "valid_minute_count": valid_minute_count,
        "expected_minute_count": EXPECTED_CONTINUOUS_MINUTES,
        "coverage_ratio": valid_minute_count / EXPECTED_CONTINUOUS_MINUTES,
        "imputed_price_minute_count": int(imputed.sum()),
        "valid_return_minute_count": return_count,
        "valid_amihud_minute_count": int(amihud_values.notna().sum()),
        "daily_volume": daily_volume,
        "daily_amount": daily_amount,
        "adjfactor": adjfactor,
        "open_price": open_price,
        "close_price": close_price,
        "high_price": high_price,
        "low_price": low_price,
        "daily_vwap": daily_vwap,
        "continuous_close": continuous_close,
        "close_auction_price": close_auction_price,
        "overnight_gap": np.nan,
        "open_to_close_return": _safe_log_ratio(
            continuous_close, open_price
        ),
        "open_to_30m_return": _safe_log_ratio(close.iloc[29], open_px.iloc[0]),
        "morning_return": _safe_log_ratio(close.iloc[119], open_px.iloc[0]),
        "afternoon_return": _safe_log_ratio(close.iloc[239], open_px.iloc[120]),
        "closing_30m_return": _safe_log_ratio(
            close.iloc[239], open_px.iloc[210]
        ),
        "lunch_gap_return": _safe_log_ratio(open_px.iloc[120], close.iloc[119]),
        "close_auction_return": _safe_log_ratio(
            close_auction_price, continuous_close
        ),
        "close_location_value": _safe_divide(
            2.0 * continuous_close - high_price - low_price,
            high_price - low_price,
        ),
        "path_efficiency": path_efficiency,
        "intraday_return_sign_persistence": float(
            (np.sign(returns) == np.sign(lagged_returns))[
                returns.notna() & lagged_returns.notna()
            ].mean()
        ),
        "minute_return_autocorr1": _safe_corr(returns, lagged_returns),
        "variance_ratio_5m": _safe_divide(
            float((five_returns**2).mean()),
            VARIANCE_RATIO_HORIZON * float((returns**2).mean()),
        ),
        "max_drawup": float(max_drawup),
        "max_drawdown_intraday": float(max_drawdown),
        "realized_variance": rv,
        "upside_semivariance": upside,
        "downside_semivariance": downside,
        "downside_semivariance_share": _safe_divide(downside, rv),
        "realized_skewness": realized_skewness,
        "realized_kurtosis": realized_kurtosis,
        "bipower_variation": bipower,
        "jump_variation": jump,
        "jump_share": _safe_divide(jump, rv),
        "max_abs_minute_return": float(abs_returns.max()),
        "tail_return_share": tail_return_share,
        "opening_30m_amount_share": float(amount_share.iloc[:30].sum()),
        "closing_30m_amount_share": float(amount_share.iloc[210:].sum()),
        "morning_amount_share": float(amount_share.iloc[:120].sum()),
        "afternoon_amount_share": float(amount_share.iloc[120:].sum()),
        "volume_concentration_hhi": float((amount_share**2).sum()),
        "amount_time_center": _safe_divide(
            float((minute_index * amount_weights).sum()),
            239.0 * daily_amount,
        ),
        "volume_return_corr": _safe_corr(volume, returns),
        "volume_abs_return_corr": _safe_corr(volume, abs_returns),
        "intraday_amihud": float(amihud_values.mean()),
        "return_per_amount": _safe_divide(
            _safe_log_ratio(continuous_close, open_price), daily_amount
        ),
        "vwap_close_deviation": _safe_divide(
            continuous_close - daily_vwap, daily_vwap
        ),
        "high_low_range": high_low_range,
        "range_per_amount": _safe_divide(high_low_range, daily_amount),
    }
    return row


def _attach_overnight_gap(
    frame: pd.DataFrame,
    previous_close: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    out = frame.sort_values(["symbol", "TradeDate"], kind="stable").copy()
    state: Dict[str, float] = dict(previous_close or {})
    gaps = pd.Series(np.nan, index=out.index, dtype=float)
    for index, row in out.iterrows():
        symbol = str(row["symbol"])
        prior = state.get(symbol, np.nan)
        gaps.at[index] = _safe_log_ratio(row["open_price"], prior)
        close = row["continuous_close"]
        if pd.notna(close) and np.isfinite(float(close)) and float(close) > 0:
            state[symbol] = float(close)
    out["overnight_gap"] = gaps
    return out, state


def compute_price_formation_daily(
    minute: pd.DataFrame,
    *,
    previous_close: Optional[Dict[str, float]] = None,
    return_state: bool = False,
):
    """Compute the reference primitive from a small canonical minute frame."""
    frame = _normalize_minute_input(minute)
    rows: List[Dict[str, object]] = []
    for (symbol, trade_date), block in frame.groupby(
        ["symbol", "date"], sort=True, observed=True
    ):
        rows.append(
            _compute_symbol_day(
                str(symbol), pd.Timestamp(trade_date), block.copy()
            )
        )
    daily = prepare_price_formation_daily(pd.DataFrame(rows))
    daily, state = _attach_overnight_gap(daily, previous_close)
    daily = prepare_price_formation_daily(daily)
    if return_state:
        return daily, state
    return daily


def price_formation_daily_sql(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """DolphinDB server-side continuous-path aggregation query."""
    symbol_clause = ""
    if symbols:
        values = ", ".join(f'"{to_wind_code(symbol)}"' for symbol in symbols)
        symbol_clause = f"\n  and Symbol in ({values})"
    return f"""
t=loadTable("dfs://QV_Trade_to_MinuteBar","Stock_one_minute")
m0=select Symbol,Date,Bartime,
    iif(
      hour(Bartime)<12,
      hour(Bartime)*60+minuteOfHour(Bartime)-570,
      120+hour(Bartime)*60+minuteOfHour(Bartime)-780
    ) as minuteIndex,
    Open*Adjfactor as openPx,
    High*Adjfactor as highPx,
    Low*Adjfactor as lowPx,
    Close*Adjfactor as closePx,
    Adjfactor as adj,
    Volume as minuteVolume,
    Amount as minuteAmount
  from t
  where Date between {_ddb_day(start)} : {_ddb_day(end)}
    and (
      (
        second(Bartime)>=09:30:00
        and second(Bartime)<=11:29:00
      )
      or (
        second(Bartime)>=13:00:00
        and second(Bartime)<=14:59:00
      )
    ){symbol_clause}
m1=select *,
    prev(minuteIndex) as minuteIndexLag1,
    move(minuteIndex,5) as minuteIndexLag5,
    prev(closePx) as closeLag1,
    move(closePx,5) as closeLag5,
    cummax(closePx) as runningMax,
    cummin(closePx) as runningMin
  from m0
  context by Symbol,Date csort minuteIndex
m2=select *,
    iif(
      closePx>0 and closeLag1>0 and minuteIndex==minuteIndexLag1+1,
      log(closePx/closeLag1),
      double(NULL)
    ) as minuteReturn,
    iif(
      closePx>0 and closeLag5>0 and minuteIndex==minuteIndexLag5+5,
      log(closePx/closeLag5),
      double(NULL)
    ) as return5m,
    iif(
      closePx>0 and closeLag1>0 and minuteIndex==minuteIndexLag1+1,
      abs(closePx-closeLag1),
      double(NULL)
    ) as absPriceChange
  from m1
m3=select *,
    prev(minuteReturn) as minuteReturnLag1,
    prev(minuteIndex) as returnIndexLag1
  from m2
  context by Symbol,Date csort minuteIndex
m4=select *,
    abs(minuteReturn) as absMinuteReturn,
    sum(minuteAmount) as dayAmount,
    quantile(abs(minuteReturn),{TAIL_QUANTILE}) as tailCut
  from m3
  context by Symbol,Date
select
    Symbol,
    Date,
    sum(iif(closePx>0,1,0)) as valid_minute_count,
    3*iif(
      lastNot(
        iif(minuteIndex==236,closePx,double(NULL))
      )>0,
      1,
      0
    ) as imputed_price_minute_count,
    sum(iif(isValid(minuteReturn),1,0)) as valid_return_minute_count,
    sum(iif(
      minuteAmount>0 and isValid(minuteReturn),1,0
    )) as valid_amihud_minute_count,
    sum(minuteVolume) as daily_volume,
    sum(minuteAmount) as daily_amount,
    firstNot(adj) as adjfactor,
    firstNot(openPx) as open_price,
    lastNot(closePx) as continuous_close,
    max(highPx) as high_price,
    min(lowPx) as low_price,
    sum(minuteAmount*adj)/sum(minuteVolume) as daily_vwap,
    lastNot(
      iif(minuteIndex<30,closePx,double(NULL))
    ) as open_30m_close,
    lastNot(
      iif(minuteIndex<120,closePx,double(NULL))
    ) as morning_close,
    firstNot(
      iif(minuteIndex>=120,openPx,double(NULL))
    ) as afternoon_open,
    firstNot(
      iif(minuteIndex>=210,openPx,double(NULL))
    ) as closing_30m_open,
    abs(lastNot(closePx)-firstNot(openPx)) / (
        abs(firstNot(closePx)-firstNot(openPx))
        + sum(absPriceChange)
      ) as path_efficiency,
    avg(iif(
      isValid(minuteReturn)
        and isValid(minuteReturnLag1)
        and minuteIndex==returnIndexLag1+1,
      double(sign(minuteReturn)==sign(minuteReturnLag1)),
      double(NULL)
    )) as intraday_return_sign_persistence,
    corr(minuteReturn,minuteReturnLag1) as minute_return_autocorr1,
    avg(return5m*return5m) / (
      {VARIANCE_RATIO_HORIZON}*avg(minuteReturn*minuteReturn)
    )
      as variance_ratio_5m,
    max(1-closePx/runningMax) as max_drawdown_intraday,
    max(closePx/runningMin-1) as max_drawup,
    sum(minuteReturn*minuteReturn) as realized_variance,
    sum(iif(
      minuteReturn>0,minuteReturn*minuteReturn,0
    )) as upside_semivariance,
    sum(iif(
      minuteReturn<0,minuteReturn*minuteReturn,0
    )) as downside_semivariance,
    count(minuteReturn) as return_n,
    sum(minuteReturn*minuteReturn*minuteReturn) as return_sum3,
    sum(
      minuteReturn*minuteReturn*minuteReturn*minuteReturn
    ) as return_sum4,
    3.141592653589793/2*sum(iif(
      isValid(minuteReturn) and isValid(minuteReturnLag1),
      abs(minuteReturn)*abs(minuteReturnLag1),
      0
    )) as bipower_variation,
    max(absMinuteReturn) as max_abs_minute_return,
    sum(iif(
      absMinuteReturn>=tailCut,
      minuteReturn*minuteReturn,
      0
    ))/sum(minuteReturn*minuteReturn) as tail_return_share,
    sum(iif(minuteIndex<30,minuteAmount,0)) / sum(minuteAmount)
      as opening_30m_amount_share,
    sum(iif(minuteIndex>=210,minuteAmount,0)) / sum(minuteAmount)
      as closing_30m_amount_share,
    sum(iif(minuteIndex<120,minuteAmount,0)) / sum(minuteAmount)
      as morning_amount_share,
    sum(iif(minuteIndex>=120,minuteAmount,0)) / sum(minuteAmount)
      as afternoon_amount_share,
    sum(
      (minuteAmount/dayAmount)*(minuteAmount/dayAmount)
    ) as volume_concentration_hhi,
    sum(minuteIndex*minuteAmount) / (239*sum(minuteAmount))
      as amount_time_center,
    corr(minuteVolume,minuteReturn) as volume_return_corr,
    corr(minuteVolume,absMinuteReturn) as volume_abs_return_corr,
    avg(iif(
      minuteAmount>0 and isValid(minuteReturn),
      absMinuteReturn/minuteAmount,
      double(NULL)
    )) as intraday_amihud
  from m4
  group by Symbol,Date
  order by Date,Symbol
"""


def close_auction_daily_sql(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    symbol_clause = ""
    if symbols:
        values = ", ".join(f'"{to_wind_code(symbol)}"' for symbol in symbols)
        symbol_clause = f"\n  and Symbol in ({values})"
    return f"""
t=loadTable("dfs://QV_Trade_to_MinuteBar","Stock_one_minute")
select
    Symbol,
    Date,
    last(Close*Adjfactor) as close_auction_price
  from t
  where Date between {_ddb_day(start)} : {_ddb_day(end)}
    and second(Bartime)==15:00:00{symbol_clause}
  group by Symbol,Date
  order by Date,Symbol
"""


def _finalize_server_daily(
    continuous: pd.DataFrame,
    auction: pd.DataFrame,
    previous_close: Optional[Dict[str, float]],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frame = continuous.rename(columns={"Symbol": "symbol", "Date": "TradeDate"})
    auction_frame = auction.rename(
        columns={"Symbol": "symbol", "Date": "TradeDate"}
    )
    frame["symbol"] = frame["symbol"].astype(str).map(to_wind_code)
    auction_frame["symbol"] = auction_frame["symbol"].astype(str).map(
        to_wind_code
    )
    frame = filter_a_share(frame)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
    auction_frame["TradeDate"] = pd.to_datetime(
        auction_frame["TradeDate"]
    ).dt.normalize()
    frame = frame.merge(
        auction_frame[["symbol", "TradeDate", "close_auction_price"]],
        on=["symbol", "TradeDate"],
        how="left",
        validate="one_to_one",
    )
    frame["expected_minute_count"] = EXPECTED_CONTINUOUS_MINUTES
    frame["coverage_ratio"] = (
        frame["valid_minute_count"] / EXPECTED_CONTINUOUS_MINUTES
    )
    frame["source_exchange"] = np.where(
        frame["symbol"].str.endswith(".SH"), "SSE", "SZSE"
    )
    frame["close_price"] = frame["close_auction_price"].where(
        frame["close_auction_price"] > 0,
        frame["continuous_close"],
    )
    frame["overnight_gap"] = np.nan
    frame["open_to_close_return"] = np.log(
        frame["continuous_close"] / frame["open_price"]
    )
    frame["open_to_30m_return"] = np.log(
        frame["open_30m_close"] / frame["open_price"]
    )
    frame["morning_return"] = np.log(
        frame["morning_close"] / frame["open_price"]
    )
    frame["afternoon_return"] = np.log(
        frame["continuous_close"] / frame["afternoon_open"]
    )
    frame["closing_30m_return"] = np.log(
        frame["continuous_close"] / frame["closing_30m_open"]
    )
    frame["lunch_gap_return"] = np.log(
        frame["afternoon_open"] / frame["morning_close"]
    )
    frame["close_auction_return"] = np.log(
        frame["close_auction_price"] / frame["continuous_close"]
    )
    price_range = frame["high_price"] - frame["low_price"]
    frame["close_location_value"] = (
        2.0 * frame["continuous_close"]
        - frame["high_price"]
        - frame["low_price"]
    ) / price_range.replace(0, np.nan)
    rv = frame["realized_variance"]
    frame["downside_semivariance_share"] = (
        frame["downside_semivariance"] / rv.replace(0, np.nan)
    )
    frame["realized_skewness"] = (
        np.sqrt(frame["return_n"])
        * frame["return_sum3"]
        / rv.pow(1.5).replace(0, np.nan)
    )
    frame["realized_kurtosis"] = (
        frame["return_n"] * frame["return_sum4"] / rv.pow(2).replace(0, np.nan)
    )
    frame["jump_variation"] = (
        rv - frame["bipower_variation"]
    ).clip(lower=0)
    frame["jump_share"] = (
        frame["jump_variation"] / rv.replace(0, np.nan)
    )
    frame["return_per_amount"] = (
        frame["open_to_close_return"] / frame["daily_amount"].replace(0, np.nan)
    )
    frame["vwap_close_deviation"] = (
        frame["continuous_close"] - frame["daily_vwap"]
    ) / frame["daily_vwap"].replace(0, np.nan)
    frame["high_low_range"] = (
        price_range / frame["open_price"].replace(0, np.nan)
    )
    frame["range_per_amount"] = (
        frame["high_low_range"] / frame["daily_amount"].replace(0, np.nan)
    )
    frame = frame.drop(
        columns=[
            "open_30m_close",
            "morning_close",
            "afternoon_open",
            "closing_30m_open",
            "return_n",
            "return_sum3",
            "return_sum4",
        ]
    )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = prepare_price_formation_daily(frame)
    frame, state = _attach_overnight_gap(frame, previous_close)
    return prepare_price_formation_daily(frame), state


def fetch_price_formation_daily(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    session=None,
    previous_close: Optional[Dict[str, float]] = None,
    return_state: bool = False,
):
    """Fetch one inclusive date chunk as symbol-day primitive rows."""
    own_session = session is None
    session = session or get_ddb_session(reuse=False)
    try:
        continuous = pd.DataFrame(
            session.run(
                price_formation_daily_sql(start, end, symbols=symbols)
            )
        )
        auction = pd.DataFrame(
            session.run(close_auction_daily_sql(start, end, symbols=symbols))
        )
    finally:
        if own_session and not is_shared_session(session):
            session.close()
    if continuous.empty:
        empty = pd.DataFrame(columns=list(PRICE_FORMATION_DAILY_COLUMNS))
        return (empty, dict(previous_close or {})) if return_state else empty
    daily, state = _finalize_server_daily(
        continuous, auction, previous_close
    )
    return (daily, state) if return_state else daily


def prepare_price_formation_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize schema and enforce hard primitive invariants."""
    if frame.empty:
        return pd.DataFrame(columns=list(PRICE_FORMATION_DAILY_COLUMNS))
    missing = sorted(set(PRICE_FORMATION_DAILY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            f"Price Formation primitive missing columns: {missing}"
        )
    out = frame.loc[:, PRICE_FORMATION_DAILY_COLUMNS].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["TradeDate"] = pd.to_datetime(out["TradeDate"]).dt.normalize()
    out["source_exchange"] = out["source_exchange"].astype(str)
    integer_columns = (
        "valid_minute_count",
        "expected_minute_count",
        "imputed_price_minute_count",
        "valid_return_minute_count",
        "valid_amihud_minute_count",
    )
    for column in integer_columns:
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    numeric_columns = [
        column
        for column in PRICE_FORMATION_DAILY_COLUMNS
        if column not in {"symbol", "TradeDate", "source_exchange", *integer_columns}
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Duplicate symbol/TradeDate in Price Formation primitive")
    numeric_values = out[numeric_columns].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("Price Formation primitive contains infinite values")
    if not out["valid_minute_count"].between(
        0, EXPECTED_CONTINUOUS_MINUTES
    ).all():
        raise ValueError("valid_minute_count outside [0, 240]")
    expected_coverage = (
        out["valid_minute_count"] / EXPECTED_CONTINUOUS_MINUTES
    )
    if not (
        out["coverage_ratio"] - expected_coverage
    ).abs().le(1e-12).all():
        raise ValueError("coverage_ratio is inconsistent with valid minutes")
    bounds = {
        "close_location_value": (-1.0, 1.0),
        "path_efficiency": (0.0, 1.0),
        "intraday_return_sign_persistence": (0.0, 1.0),
        "downside_semivariance_share": (0.0, 1.0),
        "jump_share": (0.0, 1.0),
        "tail_return_share": (0.0, 1.0),
        "opening_30m_amount_share": (0.0, 1.0),
        "closing_30m_amount_share": (0.0, 1.0),
        "morning_amount_share": (0.0, 1.0),
        "afternoon_amount_share": (0.0, 1.0),
        "volume_concentration_hhi": (0.0, 1.0),
        "amount_time_center": (0.0, 1.0),
    }
    for column, (lower, upper) in bounds.items():
        values = out[column].dropna()
        if not values.between(lower - 1e-10, upper + 1e-10).all():
            raise ValueError(f"{column} outside [{lower}, {upper}]")
    nonnegative = (
        "daily_volume",
        "daily_amount",
        "realized_variance",
        "upside_semivariance",
        "downside_semivariance",
        "bipower_variation",
        "jump_variation",
        "max_abs_minute_return",
        "max_drawup",
        "max_drawdown_intraday",
        "intraday_amihud",
        "high_low_range",
        "range_per_amount",
    )
    for column in nonnegative:
        if (out[column].dropna() < -1e-12).any():
            raise ValueError(f"{column} contains negative values")
    return out.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )
