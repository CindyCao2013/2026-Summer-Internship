"""ClickHouse SSL2 -> symbol-day Order Book primitive.

All Snapshot filtering, array math, minute-last sampling, and daily
aggregation stay in ClickHouse. Python receives daily rows only.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client


DateLike = Union[str, date, datetime, pd.Timestamp]

SCHEMA_VERSION = "l2_primitive_order_book_daily_v1"
FORMULA_VERSION = "order_book_snapshot_metrics_v1"
EXPECTED_MINUTE_COUNT = 240
COVERAGE_THRESHOLD = 0.80

ORDER_BOOK_TABLES: Tuple[Tuple[str, str, str], ...] = (
    ("SSE_AL_SSL2_EXG", ".SH", "SSE"),
    ("SZSE_AL_SSL2_EXG", ".SZ", "SZSE"),
)

KEY_COLUMNS = ("symbol", "TradeDate")
AUDIT_COLUMNS = (
    "source_exchange",
    "valid_snapshot_count",
    "valid_minute_count",
    "expected_minute_count",
    "coverage_ratio",
)
MEAN_COLUMNS = (
    "obi_1_mean",
    "obi_5_mean",
    "obi_10_mean",
    "weighted_obi_mean",
    "relative_spread_mean",
    "microprice_deviation_mean",
    "near_far_imbalance_mean",
    "bid_depth_hhi_mean",
    "ask_depth_hhi_mean",
    "depth_concentration_asymmetry_mean",
    "bid_depth_slope_mean",
    "ask_depth_slope_mean",
    "depth_slope_asymmetry_mean",
    "book_vwap_gap_mean",
    "log_total_depth_mean",
)
STD_COLUMNS = (
    "obi_1_std",
    "obi_5_std",
    "weighted_obi_std",
    "relative_spread_std",
    "microprice_deviation_std",
    "log_total_depth_std",
)
SEGMENT_COLUMNS = (
    "opening_30m_obi_5",
    "closing_30m_obi_5",
    "opening_30m_relative_spread",
    "closing_30m_relative_spread",
    "opening_30m_log_depth",
    "closing_30m_log_depth",
)
TREND_COLUMNS = (
    "obi_5_intraday_slope",
    "relative_spread_intraday_slope",
    "depth_intraday_slope",
    "obi_5_sign_persistence",
    "spread_widening_share",
)
EXTREME_COLUMNS = (
    "obi_5_p10",
    "obi_5_p90",
    "relative_spread_p90",
    "log_total_depth_p10",
)
CLOSE_COLUMNS = (
    "close_auction_valid",
    "close_auction_obi_5",
    "close_auction_relative_spread",
    "close_auction_log_depth",
)
ORDER_BOOK_DAILY_COLUMNS = (
    *KEY_COLUMNS,
    *AUDIT_COLUMNS,
    *MEAN_COLUMNS,
    *STD_COLUMNS,
    *SEGMENT_COLUMNS,
    *TREND_COLUMNS,
    *EXTREME_COLUMNS,
    *CLOSE_COLUMNS,
)


def _as_day(value: DateLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.normalize()


def _symbol_filter_sql(
    exchange: str,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    if exchange == "SSE":
        stock_filter = "startsWith(Symbol, '6')"
    elif exchange == "SZSE":
        stock_filter = (
            "(startsWith(Symbol, '000') OR startsWith(Symbol, '001') "
            "OR startsWith(Symbol, '002') OR startsWith(Symbol, '003') "
            "OR startsWith(Symbol, '300') OR startsWith(Symbol, '301') "
            "OR startsWith(Symbol, '302'))"
        )
    else:
        raise ValueError(f"Unknown exchange: {exchange}")
    if not symbols:
        return stock_filter
    bare = sorted({str(symbol).split(".")[0] for symbol in symbols})
    values = ", ".join(repr(symbol) for symbol in bare)
    return f"{stock_filter} AND Symbol IN ({values})"


def _snapshot_metric_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start: DateLike,
    end: DateLike,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """Valid Snapshot metrics, including the separate 15:00 minute."""
    start_day = _as_day(start)
    end_exclusive = _as_day(end) + pd.Timedelta(days=1)
    start_text = start_day.strftime("%Y-%m-%d")
    end_text = end_exclusive.strftime("%Y-%m-%d")
    symbol_filter = _symbol_filter_sql(exchange, symbols)
    type_filter = "AND Type = '010'" if exchange == "SZSE" else ""
    return f"""
WITH
  arrayMap(
    x -> ifNull(toFloat64(x), 0.),
    arraySlice(BidVolumes, 1, 10)
  ) AS bid_vol,
  arrayMap(
    x -> ifNull(toFloat64(x), 0.),
    arraySlice(AskVolumes, 1, 10)
  ) AS ask_vol,
  arrayMap(
    x -> ifNull(toFloat64(x), 0.),
    arraySlice(BidPrices, 1, 10)
  ) AS bid_px,
  arrayMap(
    x -> ifNull(toFloat64(x), 0.),
    arraySlice(AskPrices, 1, 10)
  ) AS ask_px,
  bid_px[1] AS bid1,
  ask_px[1] AS ask1,
  bid_vol[1] AS bid_depth_1,
  ask_vol[1] AS ask_depth_1,
  arraySum(arraySlice(bid_vol, 1, 5)) AS bid_depth_5,
  arraySum(arraySlice(ask_vol, 1, 5)) AS ask_depth_5,
  arraySum(bid_vol) AS bid_depth_10,
  arraySum(ask_vol) AS ask_depth_10,
  (bid1 + ask1) / 2. AS mid_price,
  arrayMap(i -> 1. / toFloat64(i), range(1, 11)) AS level_weights,
  arraySum(arrayMap(
    (volume, weight) -> volume * weight,
    bid_vol, level_weights
  )) AS weighted_bid_depth,
  arraySum(arrayMap(
    (volume, weight) -> volume * weight,
    ask_vol, level_weights
  )) AS weighted_ask_depth,
  arraySum(arraySlice(bid_vol, 1, 3)) AS near_bid,
  arraySum(arraySlice(ask_vol, 1, 3)) AS near_ask,
  arrayCumSum(bid_vol) AS cumulative_bid,
  arrayCumSum(ask_vol) AS cumulative_ask,
  arrayMap(value -> value / bid_depth_10, cumulative_bid)
    AS cumulative_bid_share,
  arrayMap(value -> value / ask_depth_10, cumulative_ask)
    AS cumulative_ask_share,
  arrayMap(
    price -> if(price > 0, abs(price - mid_price) / mid_price, 0.),
    bid_px
  ) AS bid_distance,
  arrayMap(
    price -> if(price > 0, abs(price - mid_price) / mid_price, 0.),
    ask_px
  ) AS ask_distance,
  arrayCount(price -> price > 0, bid_px) AS bid_slope_n,
  arrayCount(price -> price > 0, ask_px) AS ask_slope_n,
  arraySum(arrayMap(
    (price, x) -> if(price > 0, x, 0.),
    bid_px, bid_distance
  )) AS bid_sx,
  arraySum(arrayMap(
    (price, y) -> if(price > 0, y, 0.),
    bid_px, cumulative_bid_share
  )) AS bid_sy,
  arraySum(arrayMap(
    (price, x) -> if(price > 0, x * x, 0.),
    bid_px, bid_distance
  )) AS bid_sxx,
  arraySum(arrayMap(
    (price, x, y) -> if(price > 0, x * y, 0.),
    bid_px, bid_distance, cumulative_bid_share
  )) AS bid_sxy,
  arraySum(arrayMap(
    (price, x) -> if(price > 0, x, 0.),
    ask_px, ask_distance
  )) AS ask_sx,
  arraySum(arrayMap(
    (price, y) -> if(price > 0, y, 0.),
    ask_px, cumulative_ask_share
  )) AS ask_sy,
  arraySum(arrayMap(
    (price, x) -> if(price > 0, x * x, 0.),
    ask_px, ask_distance
  )) AS ask_sxx,
  arraySum(arrayMap(
    (price, x, y) -> if(price > 0, x * y, 0.),
    ask_px, ask_distance, cumulative_ask_share
  )) AS ask_sxy,
  bid_slope_n * bid_sxx - bid_sx * bid_sx AS bid_slope_den,
  ask_slope_n * ask_sxx - ask_sx * ask_sx AS ask_slope_den,
  arraySum(arrayMap(
    (price, volume) -> if(price > 0 AND volume > 0, volume, 0.),
    bid_px, bid_vol
  )) AS bid_vwap_den,
  arraySum(arrayMap(
    (price, volume) -> if(price > 0 AND volume > 0, volume, 0.),
    ask_px, ask_vol
  )) AS ask_vwap_den,
  arraySum(arrayMap(
    (price, volume) ->
      if(price > 0 AND volume > 0, price * volume, 0.),
    bid_px, bid_vol
  )) AS bid_vwap_num,
  arraySum(arrayMap(
    (price, volume) ->
      if(price > 0 AND volume > 0, price * volume, 0.),
    ask_px, ask_vol
  )) AS ask_vwap_num,
  if(
    bid_slope_n >= 2 AND abs(bid_slope_den) > 1e-20,
    (
      bid_slope_n * bid_sxy - bid_sx * bid_sy
    ) / bid_slope_den,
    CAST(NULL AS Nullable(Float64))
  ) AS bid_depth_slope,
  if(
    ask_slope_n >= 2 AND abs(ask_slope_den) > 1e-20,
    (
      ask_slope_n * ask_sxy - ask_sx * ask_sy
    ) / ask_slope_den,
    CAST(NULL AS Nullable(Float64))
  ) AS ask_depth_slope
SELECT
  ExchTime AS exch_time,
  toDate(ExchTime) AS TradeDate,
  '{exchange}' AS source_exchange,
  concat(Symbol, '{exchange_suffix}') AS symbol,
  toUInt8(toHour(ExchTime) = 15) AS is_close_auction,
  multiIf(
    toHour(ExchTime) < 12,
      toHour(ExchTime) * 60 + toMinute(ExchTime) - 570,
    toHour(ExchTime) < 15,
      120 + toHour(ExchTime) * 60 + toMinute(ExchTime) - 780,
    240
  ) AS minute_index,
  if(
    bid_depth_1 + ask_depth_1 > 0,
    (bid_depth_1 - ask_depth_1) / (bid_depth_1 + ask_depth_1),
    CAST(NULL AS Nullable(Float64))
  ) AS obi_1,
  if(
    bid_depth_5 + ask_depth_5 > 0,
    (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5),
    CAST(NULL AS Nullable(Float64))
  ) AS obi_5,
  (bid_depth_10 - ask_depth_10)
    / (bid_depth_10 + ask_depth_10) AS obi_10,
  if(
    weighted_bid_depth + weighted_ask_depth > 0,
    (weighted_bid_depth - weighted_ask_depth)
      / (weighted_bid_depth + weighted_ask_depth),
    CAST(NULL AS Nullable(Float64))
  ) AS weighted_obi,
  (ask1 - bid1) / mid_price AS relative_spread,
  if(
    bid_depth_1 + ask_depth_1 > 0,
    (
      (
        ask1 * bid_depth_1 + bid1 * ask_depth_1
      ) / (bid_depth_1 + ask_depth_1)
      - mid_price
    ) / mid_price,
    CAST(NULL AS Nullable(Float64))
  ) AS microprice_deviation,
  if(
    bid_depth_10 > 0 AND ask_depth_10 > 0,
    near_bid / bid_depth_10 - near_ask / ask_depth_10,
    CAST(NULL AS Nullable(Float64))
  ) AS near_far_imbalance,
  if(
    bid_depth_10 > 0,
    arraySum(arrayMap(
      volume -> (volume / bid_depth_10) * (volume / bid_depth_10),
      bid_vol
    )),
    CAST(NULL AS Nullable(Float64))
  ) AS bid_depth_hhi,
  if(
    ask_depth_10 > 0,
    arraySum(arrayMap(
      volume -> (volume / ask_depth_10) * (volume / ask_depth_10),
      ask_vol
    )),
    CAST(NULL AS Nullable(Float64))
  ) AS ask_depth_hhi,
  bid_depth_slope,
  ask_depth_slope,
  if(
    bid_vwap_den > 0 AND ask_vwap_den > 0,
    (
      ask_vwap_num / ask_vwap_den - bid_vwap_num / bid_vwap_den
    ) / mid_price,
    CAST(NULL AS Nullable(Float64))
  ) AS book_vwap_gap,
  log(1. + bid_depth_10 + ask_depth_10) AS log_total_depth
FROM cmds.`{table}`
WHERE ExchTime >= toDateTime64(
    '{start_text} 00:00:00', 6, 'Asia/Shanghai'
  )
  AND ExchTime < toDateTime64(
    '{end_text} 00:00:00', 6, 'Asia/Shanghai'
  )
  AND {symbol_filter}
  {type_filter}
  AND (
    (toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30)
    OR toHour(ExchTime) IN (10, 13, 14)
    OR (toHour(ExchTime) = 11 AND toMinute(ExchTime) < 30)
    OR (toHour(ExchTime) = 15 AND toMinute(ExchTime) = 0)
  )
  AND length(BidPrices) >= 10
  AND length(AskPrices) >= 10
  AND length(BidVolumes) >= 10
  AND length(AskVolumes) >= 10
  AND toFloat64(BidPrices[1]) > 0
  AND toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])
  AND arrayAll(
    value -> ifNull(toFloat64(value), 0.) >= 0,
    arraySlice(BidVolumes, 1, 10)
  )
  AND arrayAll(
    value -> ifNull(toFloat64(value), 0.) >= 0,
    arraySlice(AskVolumes, 1, 10)
  )
  AND (
    arraySum(arrayMap(
      value -> ifNull(toFloat64(value), 0.),
      arraySlice(BidVolumes, 1, 10)
    ))
    + arraySum(arrayMap(
      value -> ifNull(toFloat64(value), 0.),
      arraySlice(AskVolumes, 1, 10)
    ))
  ) > 0
"""


def order_book_daily_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start: DateLike,
    end: DateLike,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """Build the server-side minute-last and daily aggregate query."""
    snapshot_sql = _snapshot_metric_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        exchange=exchange,
        start=start,
        end=end,
        symbols=symbols,
    )
    minute_sql = f"""
SELECT
  TradeDate,
  source_exchange,
  symbol,
  toStartOfMinute(exch_time) AS minute_time,
  minute_index,
  is_close_auction,
  count() AS minute_snapshot_count,
  argMax(obi_1, exch_time) AS obi_1,
  argMax(obi_5, exch_time) AS obi_5,
  argMax(obi_10, exch_time) AS obi_10,
  argMax(weighted_obi, exch_time) AS weighted_obi,
  argMax(relative_spread, exch_time) AS relative_spread,
  argMax(microprice_deviation, exch_time) AS microprice_deviation,
  argMax(near_far_imbalance, exch_time) AS near_far_imbalance,
  argMax(bid_depth_hhi, exch_time) AS bid_depth_hhi,
  argMax(ask_depth_hhi, exch_time) AS ask_depth_hhi,
  argMax(bid_depth_slope, exch_time) AS bid_depth_slope,
  argMax(ask_depth_slope, exch_time) AS ask_depth_slope,
  argMax(book_vwap_gap, exch_time) AS book_vwap_gap,
  argMax(log_total_depth, exch_time) AS log_total_depth
FROM (
{snapshot_sql}
)
GROUP BY
  TradeDate, source_exchange, symbol, minute_time,
  minute_index, is_close_auction
"""
    lagged_sql = f"""
SELECT
  *,
  lagInFrame(
    minute_index, 1, -999
  ) OVER (
    PARTITION BY symbol, TradeDate
    ORDER BY minute_index
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS previous_minute_index,
  lagInFrame(
    toNullable(obi_5), 1, CAST(NULL AS Nullable(Float64))
  ) OVER (
    PARTITION BY symbol, TradeDate
    ORDER BY minute_index
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS previous_obi_5,
  lagInFrame(
    toNullable(relative_spread), 1, CAST(NULL AS Nullable(Float64))
  ) OVER (
    PARTITION BY symbol, TradeDate
    ORDER BY minute_index
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS previous_relative_spread
FROM (
{minute_sql}
)
"""
    return f"""
SELECT
  symbol,
  TradeDate,
  any(source_exchange) AS source_exchange,
  toInt64(sumIf(minute_snapshot_count, is_close_auction = 0))
    AS valid_snapshot_count,
  toInt64(countIf(is_close_auction = 0)) AS valid_minute_count,
  toInt64({EXPECTED_MINUTE_COUNT}) AS expected_minute_count,
  countIf(is_close_auction = 0) / {EXPECTED_MINUTE_COUNT}.
    AS coverage_ratio,
  avgIf(obi_1, is_close_auction = 0) AS obi_1_mean,
  avgIf(obi_5, is_close_auction = 0) AS obi_5_mean,
  avgIf(obi_10, is_close_auction = 0) AS obi_10_mean,
  avgIf(weighted_obi, is_close_auction = 0) AS weighted_obi_mean,
  avgIf(relative_spread, is_close_auction = 0)
    AS relative_spread_mean,
  avgIf(microprice_deviation, is_close_auction = 0)
    AS microprice_deviation_mean,
  avgIf(near_far_imbalance, is_close_auction = 0)
    AS near_far_imbalance_mean,
  avgIf(bid_depth_hhi, is_close_auction = 0) AS bid_depth_hhi_mean,
  avgIf(ask_depth_hhi, is_close_auction = 0) AS ask_depth_hhi_mean,
  avgIf(
    bid_depth_hhi - ask_depth_hhi,
    is_close_auction = 0
  ) AS depth_concentration_asymmetry_mean,
  avgIf(bid_depth_slope, is_close_auction = 0)
    AS bid_depth_slope_mean,
  avgIf(ask_depth_slope, is_close_auction = 0)
    AS ask_depth_slope_mean,
  avgIf(
    bid_depth_slope - ask_depth_slope,
    is_close_auction = 0
  ) AS depth_slope_asymmetry_mean,
  avgIf(book_vwap_gap, is_close_auction = 0) AS book_vwap_gap_mean,
  avgIf(log_total_depth, is_close_auction = 0)
    AS log_total_depth_mean,
  stddevPopIf(obi_1, is_close_auction = 0) AS obi_1_std,
  stddevPopIf(obi_5, is_close_auction = 0) AS obi_5_std,
  stddevPopIf(weighted_obi, is_close_auction = 0) AS weighted_obi_std,
  stddevPopIf(relative_spread, is_close_auction = 0)
    AS relative_spread_std,
  stddevPopIf(microprice_deviation, is_close_auction = 0)
    AS microprice_deviation_std,
  stddevPopIf(log_total_depth, is_close_auction = 0)
    AS log_total_depth_std,
  avgIf(obi_5, minute_index >= 0 AND minute_index < 30)
    AS opening_30m_obi_5,
  avgIf(obi_5, minute_index >= 210 AND minute_index < 240)
    AS closing_30m_obi_5,
  avgIf(
    relative_spread,
    minute_index >= 0 AND minute_index < 30
  ) AS opening_30m_relative_spread,
  avgIf(
    relative_spread,
    minute_index >= 210 AND minute_index < 240
  ) AS closing_30m_relative_spread,
  avgIf(log_total_depth, minute_index >= 0 AND minute_index < 30)
    AS opening_30m_log_depth,
  avgIf(log_total_depth, minute_index >= 210 AND minute_index < 240)
    AS closing_30m_log_depth,
  covarPopIf(
    obi_5, toFloat64(minute_index),
    is_close_auction = 0 AND isNotNull(obi_5)
  ) / nullIf(
    varPopIf(
      toFloat64(minute_index),
      is_close_auction = 0 AND isNotNull(obi_5)
    ),
    0
  ) AS obi_5_intraday_slope,
  covarPopIf(
    relative_spread, toFloat64(minute_index),
    is_close_auction = 0 AND isNotNull(relative_spread)
  ) / nullIf(
    varPopIf(
      toFloat64(minute_index),
      is_close_auction = 0 AND isNotNull(relative_spread)
    ),
    0
  ) AS relative_spread_intraday_slope,
  covarPopIf(
    log_total_depth, toFloat64(minute_index),
    is_close_auction = 0 AND isNotNull(log_total_depth)
  ) / nullIf(
    varPopIf(
      toFloat64(minute_index),
      is_close_auction = 0 AND isNotNull(log_total_depth)
    ),
    0
  ) AS depth_intraday_slope,
  avgIf(
    toFloat64(sign(obi_5) = sign(previous_obi_5)),
    is_close_auction = 0
      AND previous_minute_index = minute_index - 1
      AND isNotNull(obi_5)
      AND isNotNull(previous_obi_5)
  ) AS obi_5_sign_persistence,
  avgIf(
    toFloat64(relative_spread > previous_relative_spread),
    is_close_auction = 0
      AND previous_minute_index = minute_index - 1
      AND isNotNull(relative_spread)
      AND isNotNull(previous_relative_spread)
  ) AS spread_widening_share,
  quantileExactIf(0.10)(obi_5, is_close_auction = 0) AS obi_5_p10,
  quantileExactIf(0.90)(obi_5, is_close_auction = 0) AS obi_5_p90,
  quantileExactIf(0.90)(
    relative_spread, is_close_auction = 0
  ) AS relative_spread_p90,
  quantileExactIf(0.10)(
    log_total_depth, is_close_auction = 0
  ) AS log_total_depth_p10,
  toUInt8(countIf(is_close_auction = 1) > 0) AS close_auction_valid,
  argMaxIf(obi_5, minute_time, is_close_auction = 1)
    AS close_auction_obi_5,
  argMaxIf(relative_spread, minute_time, is_close_auction = 1)
    AS close_auction_relative_spread,
  argMaxIf(log_total_depth, minute_time, is_close_auction = 1)
    AS close_auction_log_depth
FROM (
{lagged_sql}
)
GROUP BY symbol, TradeDate
ORDER BY TradeDate, symbol
"""


def raw_row_daily_reference_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start: DateLike,
    end: DateLike,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """Daily raw-row mean used only to quantify update-frequency bias."""
    snapshot_sql = _snapshot_metric_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        exchange=exchange,
        start=start,
        end=end,
        symbols=symbols,
    )
    return f"""
SELECT
  symbol,
  TradeDate,
  countIf(is_close_auction = 0) AS raw_snapshot_count,
  avgIf(obi_5, is_close_auction = 0) AS raw_obi_5_mean,
  avgIf(relative_spread, is_close_auction = 0)
    AS raw_relative_spread_mean
FROM (
{snapshot_sql}
)
GROUP BY symbol, TradeDate
ORDER BY TradeDate, symbol
"""


def prepare_order_book_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a daily primitive returned by ClickHouse."""
    if frame.empty:
        return pd.DataFrame(columns=list(ORDER_BOOK_DAILY_COLUMNS))
    missing = set(ORDER_BOOK_DAILY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Order Book primitive missing columns: {sorted(missing)}")
    out = frame[list(ORDER_BOOK_DAILY_COLUMNS)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["TradeDate"] = pd.to_datetime(out["TradeDate"]).dt.normalize()
    out["source_exchange"] = out["source_exchange"].astype(str)
    integer_columns = (
        "valid_snapshot_count",
        "valid_minute_count",
        "expected_minute_count",
        "close_auction_valid",
    )
    for column in integer_columns:
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    numeric_columns = set(ORDER_BOOK_DAILY_COLUMNS) - {
        "symbol",
        "TradeDate",
        "source_exchange",
        *integer_columns,
    }
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    numeric_values = out[list(numeric_columns)].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("Order Book primitive contains infinite values")
    if out.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Duplicate symbol/TradeDate in Order Book primitive")
    if not out["valid_minute_count"].between(0, EXPECTED_MINUTE_COUNT).all():
        raise ValueError("valid_minute_count outside [0, 240]")
    expected_coverage = out["valid_minute_count"] / EXPECTED_MINUTE_COUNT
    if not (out["coverage_ratio"] - expected_coverage).abs().le(1e-12).all():
        raise ValueError("coverage_ratio is inconsistent with minute counts")
    for column in (
        "obi_1_mean",
        "obi_5_mean",
        "obi_10_mean",
        "weighted_obi_mean",
        "near_far_imbalance_mean",
    ):
        values = out[column].dropna()
        if not values.between(-1.0 - 1e-12, 1.0 + 1e-12).all():
            raise ValueError(f"{column} outside [-1, 1]")
    for column in ("relative_spread_mean", "relative_spread_p90"):
        if (out[column].dropna() < -1e-12).any():
            raise ValueError(f"{column} contains negative spread")
    for column in ("bid_depth_hhi_mean", "ask_depth_hhi_mean"):
        values = out[column].dropna()
        if not values.between(0.1 - 1e-10, 1.0 + 1e-10).all():
            raise ValueError(f"{column} outside [0.1, 1]")
    return out.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def fetch_order_book_daily(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    tables: Iterable[Tuple[str, str, str]] = ORDER_BOOK_TABLES,
    client=None,
) -> pd.DataFrame:
    """Fetch daily primitive rows for an inclusive date interval."""
    own_client = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, exchange in tables:
            query = order_book_daily_sql(
                table=table,
                exchange_suffix=suffix,
                exchange=exchange,
                start=start,
                end=end,
                symbols=symbols,
            )
            frame = client.query_df(query)
            if not frame.empty:
                frames.append(frame)
    finally:
        if own_client:
            client.close()
    if not frames:
        return pd.DataFrame(columns=list(ORDER_BOOK_DAILY_COLUMNS))
    return prepare_order_book_daily(pd.concat(frames, ignore_index=True))


def fetch_raw_row_daily_reference(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    tables: Iterable[Tuple[str, str, str]] = ORDER_BOOK_TABLES,
    client=None,
) -> pd.DataFrame:
    """Fetch daily raw-row means without transferring Snapshot-level data."""
    own_client = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, exchange in tables:
            query = raw_row_daily_reference_sql(
                table=table,
                exchange_suffix=suffix,
                exchange=exchange,
                start=start,
                end=end,
                symbols=symbols,
            )
            frame = client.query_df(query)
            if not frame.empty:
                frames.append(frame)
    finally:
        if own_client:
            client.close()
    columns = (
        "symbol",
        "TradeDate",
        "raw_snapshot_count",
        "raw_obi_5_mean",
        "raw_relative_spread_mean",
    )
    if not frames:
        return pd.DataFrame(columns=list(columns))
    out = pd.concat(frames, ignore_index=True)[list(columns)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["TradeDate"] = pd.to_datetime(out["TradeDate"]).dt.normalize()
    for column in columns[2:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError("Duplicate raw-row reference key")
    return out.sort_values(["symbol", "TradeDate"]).reset_index(drop=True)
