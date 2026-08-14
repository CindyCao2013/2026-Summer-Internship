"""DolphinDB server-side intraday factor SQL builders.

Batch-mode equivalents of streaming patterns in:
https://docs.dolphindb.com/zh/tutorials/str_comp_fin_quant.html

Use ``MinuteBarStore.run_script()`` to execute these scripts; only narrow
results are returned to Python.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

DateLike = Union[str, pd.Timestamp]


def _ddb_date(value: DateLike) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def volume_front_loading_script(
    start_date: DateLike,
    end_date: DateLike,
    *,
    lookback_days: int = 20,
    symbols: Optional[list[str]] = None,
) -> str:
    """早盘量比 — prior-N-session rolling mean, signal at 10:29."""
    start = _ddb_date(start_date)
    end = _ddb_date(end_date)
    hist_start = _ddb_date(
        pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days + 10)
    )
    min_periods = max(5, lookback_days // 2)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{s}"' for s in symbols)
        sym_filter = f" and Symbol in ({sym_str})"

    return f"""
startDate = {start}
endDate = {end}
histStart = {hist_start}
lookback = {lookback_days}
minP = {min_periods}

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
morning = select nullFill(sum(Volume), 0) as morning_vol
    from t
    where Date between histStart : endDate
      and second(Bartime) between 09:30:00 : 10:00:00
      {sym_filter}
    group by Symbol, Date

morning = select Symbol, Date, morning_vol,
    // move(..., 1) excludes today's morning volume: no look-ahead.
    move(msum(morning_vol, lookback, minP), 1) \\ move(mcount(morning_vol, lookback, minP), 1) as hist_avg
    from morning
    context by Symbol csort Date

result = select Symbol, Date,
    morning_vol \\ hist_avg as value,
    concatDateTime(Date, 10:29:00) as bartime
    from morning
    where Date between startDate : endDate
      and isValid(hist_avg) and hist_avg > 0
select Symbol, Date, bartime, value from result
"""


def volume_back_loading_script(
    start_date: DateLike,
    end_date: DateLike,
    *,
    lookback_days: int = 20,
    symbols: Optional[list[str]] = None,
) -> str:
    """尾盘量比 — prior-N-session rolling mean, computed at day-T close."""
    start_ts = pd.Timestamp(start_date)
    start = _ddb_date(start_ts)
    end = _ddb_date(end_date)
    hist_start = _ddb_date(start_ts - pd.Timedelta(days=lookback_days + 10))
    min_periods = max(5, lookback_days // 2)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{s}"' for s in symbols)
        sym_filter = f" and Symbol in ({sym_str})"

    return f"""
startDate = {start}
endDate = {end}
histStart = {hist_start}
lookback = {lookback_days}
minP = {min_periods}

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
closing = select nullFill(sum(Volume), 0) as closing_vol
    from t
    where Date between histStart : endDate
      and second(Bartime) between 14:30:00 : 15:00:00
      {sym_filter}
    group by Symbol, Date

closing = select Symbol, Date, closing_vol,
    // move(..., 1) excludes today's closing volume from its history.
    move(msum(closing_vol, lookback, minP), 1) \\ move(mcount(closing_vol, lookback, minP), 1) as hist_avg
    from closing
    context by Symbol csort Date

result = select Symbol, Date,
    closing_vol \\ hist_avg as value
    from closing
    where Date between startDate : endDate
      and isValid(hist_avg) and hist_avg > 0
select Symbol, Date, value from result
"""


def late_session_strength_script(
    start_date: DateLike,
    end_date: DateLike,
    *,
    symbols: Optional[list[str]] = None,
) -> str:
    """尾盘主动买入金额占比 — day-T close state for T+1 signal."""
    start_ts = pd.Timestamp(start_date)
    raw_start = _ddb_date(start_ts - pd.Timedelta(days=10))
    end = _ddb_date(end_date)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{s}"' for s in symbols)
        sym_filter = f" and Symbol in ({sym_str})"

    return f"""
rawStart = {raw_start}
endDate = {end}

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date,
    Active_buy_amount * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as buy_amt,
    Active_sell_amount * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as sell_amt
    from t
    where Date between rawStart : endDate
      and second(Bartime) between 14:30:00 : 15:00:00
      {sym_filter}

closingFlow = select
    nullFill(sum(buy_amt), 0) as buy_amt,
    nullFill(sum(sell_amt), 0) as sell_amt
    from bars
    group by Symbol, Date

result = select Symbol, Date,
    buy_amt \\ (buy_amt + sell_amt) as value
    from closingFlow
    where buy_amt + sell_amt > 0
select Symbol, Date, value from result
"""


def active_buy_sell_imbalance_script(
    start_date: DateLike,
    end_date: DateLike,
    *,
    symbols: Optional[list[str]] = None,
    bartimes: Optional[list[str]] = None,
) -> str:
    """Session-cumulative aggressive buy/sell amount imbalance."""
    start = _ddb_date(start_date)
    end = _ddb_date(end_date)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{s}"' for s in symbols)
        sym_filter = f" and Symbol in ({sym_str})"
    bt_list = bartimes or [
        "09:59:00",
        "10:29:00",
        "11:29:00",
        "13:29:00",
        "14:29:00",
    ]
    bt_str = ", ".join(bt_list)

    return f"""
startDate = {start}
endDate = {end}
btFilter = [{bt_str}]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date, second(Bartime) as Bartime,
    nullFill(Active_buy_amount, 0)
        * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as buy_amt,
    nullFill(Active_sell_amount, 0)
        * iif(isNull(Adjfactor) or Adjfactor == 0, 1.0, Adjfactor) as sell_amt
    from t
    where Date between startDate : endDate
      {sym_filter}
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

flows = select Symbol, Date, Bartime,
    cumsum(buy_amt) as cum_buy,
    cumsum(sell_amt) as cum_sell
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol,
    concatDateTime(Date, Bartime) as bartime,
    (cum_buy - cum_sell) \\ (cum_buy + cum_sell) as value
    from flows
    where Bartime in btFilter
      and cum_buy + cum_sell > 0
select Symbol, bartime, value from result
"""


DISCOVERY_V1_FACTORS = frozenset(
    {
        "bartime_ofi",
        "ofi_persistence",
        "active_buy_shock",
        "average_active_trade_size",
        "large_active_buy_ratio",
        "intraday_amihud",
        "realized_volatility",
        "minute_skew",
    }
)


def discovery_v1_factor_script(
    factor_name: str,
    start_date: DateLike,
    end_date: DateLike,
    *,
    symbols: Optional[list[str]] = None,
    bartimes: Optional[list[str]] = None,
) -> str:
    """Build a no-look-ahead DDB script for Phase 4 discovery factors.

    ``large_active_buy_ratio`` is explicitly a bar-level proxy because the
    minute table has no large-order amount bucket. A bar is classified as large
    when its average active-buy ticket exceeds the prior-20-bar mean plus one
    prior-20-bar standard deviation.
    """
    if factor_name not in DISCOVERY_V1_FACTORS:
        raise KeyError(f"Unknown discovery factor: {factor_name}")
    start = _ddb_date(start_date)
    end = _ddb_date(end_date)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{s}"' for s in symbols)
        sym_filter = f" and Symbol in ({sym_str})"
    bt_list = bartimes or [
        "09:59:00",
        "10:29:00",
        "11:29:00",
        "13:29:00",
        "14:29:00",
    ]
    bt_str = ", ".join(bt_list)

    common = f"""
startDate = {start}
endDate = {end}
btFilter = [{bt_str}]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars0 = select Symbol, Date, second(Bartime) as Bartime,
    iif(isValid(Active_buy_amount) and Active_buy_amount > 0,
        Active_buy_amount, 0.0) as buy_amt,
    iif(isValid(Active_sell_amount) and Active_sell_amount > 0,
        Active_sell_amount, 0.0) as sell_amt,
    iif(isValid(Active_buy_count) and Active_buy_count > 0,
        Active_buy_count, 0) as buy_count,
    iif(isValid(Active_sell_count) and Active_sell_count > 0,
        Active_sell_count, 0) as sell_count,
    Close as close_adj,
    iif(isValid(Amount) and Amount > 0, Amount, 0.0) as amount_adj
    from t
    where Date between startDate : endDate
      {sym_filter}
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime, buy_amt, sell_amt, buy_count, sell_count,
    close_adj, amount_adj,
    iif(buy_amt + sell_amt > 0,
        (buy_amt - sell_amt) \\ (buy_amt + sell_amt), NULL) as bar_ofi,
    iif(buy_count > 0, buy_amt \\ buy_count, NULL) as buy_size,
    close_adj \\ move(close_adj, 1) - 1.0 as minute_ret
    from bars0
    context by Symbol, Date csort Bartime
"""

    if factor_name == "bartime_ofi":
        body = """
result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    bar_ofi as value
    from bars
    where Bartime in btFilter and isValid(bar_ofi)
"""
    elif factor_name == "ofi_persistence":
        body = """
features = select Symbol, Date, Bartime,
    msum(iif(isValid(bar_ofi), iif(bar_ofi > 0, 1.0, 0.0), NULL), 20, 5)
        \\ mcount(bar_ofi, 20, 5) as value
    from bars
    context by Symbol, Date csort Bartime
result = select Symbol, concatDateTime(Date, Bartime) as bartime, value
    from features
    where Bartime in btFilter and isValid(value)
"""
    elif factor_name == "active_buy_shock":
        body = """
features = select Symbol, Date, Bartime, buy_amt,
    move(mavg(buy_amt, 20, 10), 1) as hist_mean,
    move(mstd(buy_amt, 20, 10), 1) as hist_std
    from bars
    context by Symbol, Date csort Bartime
result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    (buy_amt - hist_mean) \\ hist_std as value
    from features
    where Bartime in btFilter and isValid(hist_std)
      and hist_std > iif(abs(hist_mean) * 0.00000001 > 1.0,
        abs(hist_mean) * 0.00000001, 1.0)
"""
    elif factor_name == "average_active_trade_size":
        body = """
features = select Symbol, Date, Bartime, buy_size,
    move(mavg(buy_size, 20, 10), 1) as hist_mean
    from bars
    context by Symbol, Date csort Bartime
result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    buy_size \\ hist_mean - 1.0 as value
    from features
    where Bartime in btFilter and isValid(buy_size)
      and isValid(hist_mean) and hist_mean > 0
"""
    elif factor_name == "large_active_buy_ratio":
        body = """
baseline = select Symbol, Date, Bartime, buy_amt, buy_size,
    move(mavg(buy_size, 20, 10), 1) as hist_mean,
    move(mstd(buy_size, 20, 10), 1) as hist_std
    from bars
    context by Symbol, Date csort Bartime
classified = select Symbol, Date, Bartime, buy_amt,
    iif(isValid(buy_size) and isValid(hist_mean) and isValid(hist_std)
        and buy_size > hist_mean + hist_std, buy_amt, 0.0) as large_buy_amt,
    iif(isValid(hist_mean) and isValid(hist_std), 1.0, NULL) as baseline_valid
    from baseline
features = select Symbol, Date, Bartime,
    msum(large_buy_amt, 20, 10) as large_buy_sum,
    msum(buy_amt, 20, 10) as buy_sum,
    mcount(baseline_valid, 20, 10) as valid_count
    from classified
    context by Symbol, Date csort Bartime
result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    large_buy_sum \\ buy_sum as value
    from features
    where Bartime in btFilter and valid_count >= 10
      and buy_sum > 1.0
"""
    elif factor_name == "intraday_amihud":
        body = """
features = select Symbol, Date, Bartime,
    msum(abs(minute_ret), 5, 3) as abs_ret_sum,
    msum(amount_adj, 5, 3) as amount_sum
    from bars
    context by Symbol, Date csort Bartime
result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    abs_ret_sum \\ amount_sum as value
    from features
    where Bartime in btFilter and amount_sum > 1.0
"""
    elif factor_name == "realized_volatility":
        body = """
features = select Symbol, Date, Bartime,
    sqrt(cumsum(nullFill(minute_ret * minute_ret, 0.0))) as value,
    cumsum(iif(isValid(minute_ret), 1, 0)) as obs_count
    from bars
    context by Symbol, Date csort Bartime
result = select Symbol, concatDateTime(Date, Bartime) as bartime, value
    from features
    where Bartime in btFilter and obs_count >= 5 and isValid(value)
"""
    else:
        body = """
moments = select Symbol, Date, Bartime,
    cumsum(iif(isValid(minute_ret), 1.0, 0.0)) as n,
    cumsum(nullFill(minute_ret, 0.0)) as s1,
    cumsum(nullFill(minute_ret * minute_ret, 0.0)) as s2,
    cumsum(nullFill(minute_ret * minute_ret * minute_ret, 0.0)) as s3
    from bars
    context by Symbol, Date csort Bartime
central = select Symbol, Date, Bartime, n,
    s2 - s1 * s1 \\ n as m2,
    s3 - 3.0 * s1 * s2 \\ n + 2.0 * s1 * s1 * s1 \\ (n * n) as m3
    from moments
    where n >= 3
result = select Symbol, concatDateTime(Date, Bartime) as bartime,
    n * sqrt(n - 1.0) \\ (n - 2.0) * m3 \\ pow(m2, 1.5) as value
    from central
    where Bartime in btFilter and m2 > 0
"""

    return common + body + "\nselect Symbol, bartime, value from result\n"


def close_vwap_deviation_script(
    start_date: DateLike,
    end_date: DateLike,
    *,
    symbols: Optional[list[str]] = None,
    bartimes: Optional[list[str]] = None,
) -> str:
    """收盘价相对累计 VWAP 偏离 — ``cumsum`` + ``context by``."""
    start = _ddb_date(start_date)
    end = _ddb_date(end_date)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{s}"' for s in symbols)
        sym_filter = f" and Symbol in ({sym_str})"
    bt_list = bartimes or ["09:59:00", "10:29:00", "11:29:00", "13:29:00", "14:29:00"]
    bt_str = ", ".join(f"{bt}" for bt in bt_list)

    return f"""
startDate = {start}
endDate = {end}
btFilter = [{bt_str}]

t = loadTable('dfs://QV_Trade_to_MinuteBar', 'Stock_one_minute')
bars = select Symbol, Date, second(Bartime) as Bartime, Close, Volume, Amount, Adjfactor
    from t
    where Date between startDate : endDate
      {sym_filter}
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))

bars = select Symbol, Date, Bartime,
    Close * Adjfactor as close_adj,
    Volume,
    cumsum(Amount * Adjfactor) \\ cumsum(Volume) as cum_vwap,
    rowNo(Bartime) as bar_idx
    from bars
    context by Symbol, Date csort Bartime

result = select Symbol,
    concatDateTime(Date, Bartime) as bartime,
    (close_adj - cum_vwap) \\ cum_vwap as value
    from bars
    where Bartime in btFilter
      and Volume > 0
      and isValid(cum_vwap) and cum_vwap != 0
      and bar_idx > 0
select Symbol, bartime, value from result
"""
