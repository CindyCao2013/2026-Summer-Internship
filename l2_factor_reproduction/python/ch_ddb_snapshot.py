"""ClickHouse SSL2 -> symbol-day DolphinDB reference snapshot factors.

Faithful replication of the five official DolphinDB L2 snapshot formulas
(docs.dolphindb.cn/zh/tutorials/l2_stk_data_proc_2.html, section 3.1) with
all computation pushed down to ClickHouse. Python receives symbol-day rows
only; raw snapshots are never persisted.

Frozen formulas (state resets per symbol x TradeDate, ordered by ExchTime):

- time_weighted_order_slope
    (log(ask_eff)-log(bid_eff)) /
    nullFill(mavg(ffill(log(askQty1)-log(bidQty1)), 20, 1), 0)
    where ask_eff = ask1 if ask1 != 0 else bid1 (and symmetrically).
- wavg_soir
    imbalance = rowWavg(level imbalances, weights 10..1) dropping NULL
    levels with weight renormalization, ffill, nullFill(0);
    mean/std over prev(imbalance) window of 19 (minPeriods=2, pop std);
    value = (imb-mean)/std if std >= 1e-7 else NULL, then ffill/nullFill(0).
- tra_price_weighted_net_buy_quote_volume_ratio
    level-1 bid/ask change decomposition x inter-snapshot average trade
    price from AccAmount/AccVolume deltas; msum(20,1) ratio, nullFill(0).
- level10_diff_buy
    rowAlign("bid") price-level alignment between current and previous
    ten-level bid arrays; per-price qty diff x price summed over the
    aligned grid; msum(20,1), nullFill(0).
- level10_infer_price_trend
    amount-weighted ten-level implied price, ffill, linearTimeTrend(60)
    slope (full window required), nullFill(0), mavg(20,1), nullFill(0).

DolphinDB engine semantics replicated (probed on company 2.00.17):
log of non-positive -> NULL; division by zero -> NULL (never inf);
moving aggregations skip NULLs and require >= minPeriods non-null values;
iif(NULL condition) takes the else branch (same as ClickHouse if);
ffill fills from history only; linearTimeTrend needs a full window;
rowAt is 0-based with -1 -> NULL; rowWavg drops NULL elements and
renormalizes weights; CH arraySum cannot take NULL elements, so all
array aggregations here map invalid elements to 0 with separate
denominator guards.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client


DateLike = Union[str, date, datetime, pd.Timestamp]

SCHEMA_VERSION = "ddb_reference_snapshot_daily_v1"
FORMULA_VERSION = "ddb_official_snapshot_formulas_v1"
EXPECTED_MINUTE_COUNT = 240
COVERAGE_THRESHOLD = 0.80

DDB_SNAPSHOT_TABLES: Tuple[Tuple[str, str, str], ...] = (
    ("SSE_AL_SSL2_EXG", ".SH", "SSE"),
    ("SZSE_AL_SSL2_EXG", ".SZ", "SZSE"),
)

FACTOR_NAMES: Tuple[str, ...] = (
    "time_weighted_order_slope",
    "wavg_soir",
    "tra_price_weighted_net_buy_quote_volume_ratio",
    "level10_diff_buy",
    "level10_infer_price_trend",
)

# Per-factor clip-guard thresholds (monitoring only; values never modified).
CLIP_GUARDS: Dict[str, float] = {
    "time_weighted_order_slope": 1e6,
    "wavg_soir": 1e6,
    "tra_price_weighted_net_buy_quote_volume_ratio": 1e6,
    "level10_diff_buy": 1e12,
    "level10_infer_price_trend": 1e6,
}

SMALL_DENOMINATOR_EPS = 1e-6
SOIR_STD_FLOOR = 1e-7

KEY_COLUMNS = ("symbol", "TradeDate")
SHARED_AUDIT_COLUMNS = (
    "source_exchange",
    "valid_snapshot_count",
    "valid_minute_count",
    "expected_minute_count",
    "coverage_ratio",
    "exact_tie_row_count",
)
MEAN_COLUMNS = tuple(f"{name}_mean" for name in FACTOR_NAMES)
VALID_VALUE_COLUMNS = tuple(
    f"{name}_valid_minute_count" for name in FACTOR_NAMES
)
DIAG_COLUMNS = tuple(
    f"{name}_{diag}"
    for name in FACTOR_NAMES
    for diag in (
        "small_denominator_count",
        "ffill_count",
        "inf_count",
        "clipped_count",
        "null_snapshot_count",
    )
)
DAILY_COLUMNS = (
    *KEY_COLUMNS,
    *SHARED_AUDIT_COLUMNS,
    *MEAN_COLUMNS,
    *VALID_VALUE_COLUMNS,
    *DIAG_COLUMNS,
)

# Snapshot-level intermediate columns exposed by the debug series query
# (used by the golden-reference comparison against the DolphinDB engine).
SERIES_COLUMNS = (
    "symbol",
    "exch_time",
    "minute_index",
    "is_close_auction",
    "rn",
    "bid1",
    "ask1",
    "bid_qty1",
    "ask_qty1",
    "acc_amount",
    "acc_volume",
    "twos_num",
    "twos_den_raw",
    "twos_den_ffill",
    "twos_den_final",
    "soir_imb_raw",
    "soir_imb",
    "soir_prev_imb",
    "soir_mean",
    "soir_std",
    "wavg_soir_raw",
    "bid_chg",
    "offer_chg",
    "avg_price",
    "tpw_fv",
    "tpw_fv_sum",
    "tpw_ap_sum",
    "l10_amt_diff",
    "l10_amt_sum",
    "infer_price_raw",
    "infer_ffill",
    "ltt_slope0",
    "time_weighted_order_slope",
    "wavg_soir",
    "tra_price_weighted_net_buy_quote_volume_ratio",
    "level10_diff_buy",
    "level10_infer_price_trend",
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


def _base_extract_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start: DateLike,
    end: DateLike,
    symbols: Optional[Sequence[str]] = None,
    raw_source_sql: Optional[str] = None,
) -> str:
    """Filtered snapshot rows + plain per-row expressions (no windows).

    The WHERE clause mirrors ch_order_book._snapshot_metric_sql exactly
    (Phase-0 filtering: complete ten levels; continuous auction plus the
    separately flagged 15:00 close-auction row; crossed-book exclusion;
    SZSE Type='010'; non-negative volumes; positive total volume).

    ``raw_source_sql`` is a test-only hook replacing the raw table source
    (must expose Symbol/ExchTime/BidPrices/AskPrices/BidVolumes/AskVolumes/
    AccAmount/AccVolume); production paths always read cmds.<table>.
    """
    start_day = _as_day(start)
    end_exclusive = _as_day(end) + pd.Timedelta(days=1)
    start_text = start_day.strftime("%Y-%m-%d")
    end_text = end_exclusive.strftime("%Y-%m-%d")
    symbol_filter = _symbol_filter_sql(exchange, symbols)
    type_filter = "AND Type = '010'" if exchange == "SZSE" else ""
    raw_source = raw_source_sql or f"cmds.`{table}`"
    return f"""
SELECT
  exch_time,
  toDate(exch_time) AS TradeDate,
  '{exchange}' AS source_exchange,
  concat(Symbol, '{exchange_suffix}') AS symbol,
  toUInt8(toHour(exch_time) = 15) AS is_close_auction,
  multiIf(
    toHour(exch_time) < 12,
      toHour(exch_time) * 60 + toMinute(exch_time) - 570,
    toHour(exch_time) < 15,
      120 + toHour(exch_time) * 60 + toMinute(exch_time) - 780,
    240
  ) AS minute_index,
  bid_px[1] AS bid1,
  ask_px[1] AS ask1,
  bid_vol[1] AS bid_qty1,
  ask_vol[1] AS ask_qty1,
  bid_px,
  ask_px,
  bid_vol,
  ask_vol,
  acc_amount,
  acc_volume,
  if(
    ask1_eff > 0 AND bid1_eff > 0,
    log(ask1_eff) - log(bid1_eff),
    CAST(NULL AS Nullable(Float64))
  ) AS twos_num,
  if(
    bid_qty1 > 0 AND ask_qty1 > 0,
    log(ask_qty1) - log(bid_qty1),
    CAST(NULL AS Nullable(Float64))
  ) AS twos_den_raw,
  arraySum(arrayMap(
    (b, a, w) -> if(b + a > 0, w * (b - a) / (b + a), 0.),
    bid_vol, ask_vol, soir_weights
  )) / nullIf(arraySum(arrayMap(
    (b, a, w) -> if(b + a > 0, w, 0.),
    bid_vol, ask_vol, soir_weights
  )), 0.) AS soir_imb_raw,
  if(
    bid1 > 0 AND ask1 > 0,
    (arraySum(arrayMap((p, q) -> p * q, bid_px, bid_vol))
     + arraySum(arrayMap((p, q) -> p * q, ask_px, ask_vol)))
      / nullIf(arraySum(bid_vol) + arraySum(ask_vol), 0.),
    CAST(NULL AS Nullable(Float64))
  ) AS infer_price_raw
FROM (
  SELECT
    Symbol,
    ExchTime AS exch_time,
    arrayMap(
      x -> ifNull(toFloat64(x), 0.),
      arraySlice(BidPrices, 1, 10)
    ) AS bid_px,
    arrayMap(
      x -> ifNull(toFloat64(x), 0.),
      arraySlice(AskPrices, 1, 10)
    ) AS ask_px,
    arrayMap(
      x -> ifNull(toFloat64(x), 0.),
      arraySlice(BidVolumes, 1, 10)
    ) AS bid_vol,
    arrayMap(
      x -> ifNull(toFloat64(x), 0.),
      arraySlice(AskVolumes, 1, 10)
    ) AS ask_vol,
    arrayMap(i -> toFloat64(11 - i), range(1, 11)) AS soir_weights,
    toFloat64(AccAmount) AS acc_amount,
    toFloat64(AccVolume) AS acc_volume,
    if(
      toFloat64(AskPrices[1]) = 0,
      toFloat64(BidPrices[1]),
      toFloat64(AskPrices[1])
    ) AS ask1_eff,
    if(
      toFloat64(BidPrices[1]) = 0,
      toFloat64(AskPrices[1]),
      toFloat64(BidPrices[1])
    ) AS bid1_eff
  FROM {raw_source}
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
  ORDER BY Symbol, exch_time
)
"""


def _window_layer_sql(base_sql: str) -> str:
    """Window layer: row number, lags, running ffill (anyLast skips NULLs)."""
    return f"""
SELECT
  *,
  row_number() OVER wd AS rn,
  lagInFrame(toNullable(bid1), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS prev_bid1,
  lagInFrame(toNullable(bid_qty1), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS prev_bid_qty1,
  lagInFrame(toNullable(ask1), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS prev_ask1,
  lagInFrame(toNullable(ask_qty1), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS prev_ask_qty1,
  lagInFrame(toNullable(acc_amount), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS prev_acc_amount,
  lagInFrame(toNullable(acc_volume), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS prev_acc_volume,
  lagInFrame(bid_px, 1, []) OVER wd AS prev_bid_px,
  lagInFrame(bid_vol, 1, []) OVER wd AS prev_bid_vol,
  lagInFrame(
    toNullable(exch_time), 1,
    CAST(NULL AS Nullable(DateTime64(6, 'Asia/Shanghai')))
  ) OVER wd AS prev_exch_time,
  anyLast(twos_den_raw) OVER run AS twos_den_ffill,
  anyLast(soir_imb_raw) OVER run AS soir_imb_ffill,
  anyLast(infer_price_raw) OVER run AS infer_ffill
FROM (
{base_sql}
)
WINDOW
  wd AS (PARTITION BY symbol, TradeDate ORDER BY exch_time),
  run AS (
    PARTITION BY symbol, TradeDate ORDER BY exch_time
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )
"""


def _compose_layer_sql(window_sql: str) -> str:
    """Plain compositions needing lagged values (change decomposition etc.)."""
    return f"""
SELECT
  *,
  if(
    round(bid1 - prev_bid1, 2) > 0,
    bid_qty1,
    if(
      round(bid1 - prev_bid1, 2) < 0,
      -prev_bid_qty1,
      bid_qty1 - prev_bid_qty1
    )
  ) AS bid_chg,
  if(
    if(ask1 = 0, if(prev_ask1 > 0, 1, 0), ask1 - prev_ask1) > 0,
    prev_ask_qty1,
    if(
      if(
        prev_ask1 = 0,
        if(ask1 > 0, -1, 0),
        if(ask1 > 0, ask1 - prev_ask1, 1)
      ) < 0,
      ask_qty1,
      ask_qty1 - prev_ask_qty1
    )
  ) AS offer_chg,
  (acc_amount - prev_acc_amount)
    / nullIf(acc_volume - prev_acc_volume, 0.) AS avg_price,
  ifNull(soir_imb_ffill, 0.) AS soir_imb,
  arraySum(arrayMap(
    g -> g * (
      if(indexOf(bid_px, g) > 0, bid_vol[indexOf(bid_px, g)], 0.)
      - if(
        indexOf(prev_bid_px, g) > 0,
        prev_bid_vol[indexOf(prev_bid_px, g)],
        0.
      )
    ),
    arraySort(
      x -> -x,
      arrayDistinct(arrayFilter(
        x ->
          x >= greatest(arrayMin(bid_px), arrayMin(prev_bid_px))
          AND x <= greatest(arrayMax(bid_px), arrayMax(prev_bid_px)),
        arrayConcat(bid_px, prev_bid_px)
      )))
    )
  ) AS l10_amt_diff,
  toUInt8(ifNull(prev_exch_time = exch_time, false)) AS tie_flag
FROM (
{window_sql}
)
"""


def _rolling_layer_sql(compose_sql: str) -> str:
    """Rolling-window aggregates over the 3s state series."""
    return f"""
SELECT
  *,
  lagInFrame(toNullable(soir_imb), 1, CAST(NULL AS Nullable(Float64)))
    OVER wd AS soir_prev_imb,
  avg(twos_den_ffill) OVER w19 AS twos_den_mavg,
  count(twos_den_ffill) OVER w19 AS twos_den_cnt,
  sum(tpw_fv) OVER w19 AS tpw_fv_sum,
  count(tpw_fv) OVER w19 AS tpw_fv_cnt,
  sum(avg_price) OVER w19 AS tpw_ap_sum,
  count(avg_price) OVER w19 AS tpw_ap_cnt,
  sum(l10_amt_diff) OVER w19 AS l10_amt_sum,
  count(l10_amt_diff) OVER w19 AS l10_amt_cnt,
  covarPop(toFloat64(rn), infer_ffill) OVER w59 AS ltt_cov,
  varPop(toFloat64(rn)) OVER w59 AS ltt_var,
  count(infer_ffill) OVER w59 AS ltt_cnt
FROM (
  SELECT
    *,
    (bid_chg - offer_chg)
      / nullIf(abs(bid_chg) + abs(offer_chg), 0.) * avg_price AS tpw_fv
  FROM (
{compose_sql}
  )
)
WINDOW
  wd AS (PARTITION BY symbol, TradeDate ORDER BY exch_time),
  w19 AS (
    PARTITION BY symbol, TradeDate ORDER BY exch_time
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
  ),
  w59 AS (
    PARTITION BY symbol, TradeDate ORDER BY exch_time
    ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
  )
"""


def _soir_stats_layer_sql(rolling_sql: str) -> str:
    """prev(imbalance) normalization window stats (19 obs, minPeriods=2)."""
    return f"""
SELECT
  *,
  avg(soir_prev_imb) OVER w18 AS soir_mean,
  stddevPop(soir_prev_imb) OVER w18 AS soir_std,
  count(soir_prev_imb) OVER w18 AS soir_cnt
FROM (
{rolling_sql}
)
WINDOW
  w18 AS (
    PARTITION BY symbol, TradeDate ORDER BY exch_time
    ROWS BETWEEN 18 PRECEDING AND CURRENT ROW
  )
"""


def _final_value_layer_sql(soir_sql: str) -> str:
    """Snapshot-level final values for all five formulas + diagnostics."""
    return f"""
SELECT
  *,
  twos_num / nullIf(twos_den_final, 0.) AS time_weighted_order_slope,
  if(
    soir_std >= {SOIR_STD_FLOOR},
    (soir_imb - soir_mean) / soir_std,
    CAST(NULL AS Nullable(Float64))
  ) AS wavg_soir_raw,
  ifNull(
    if(
      tpw_fv_cnt >= 1 AND tpw_ap_cnt >= 1,
      tpw_fv_sum / nullIf(tpw_ap_sum, 0.),
      CAST(NULL AS Nullable(Float64))
    ),
    0.
  ) AS tra_price_raw,
  ifNull(if(l10_amt_cnt >= 1, l10_amt_sum, NULL), 0.) AS level10_diff_buy,
  if(
    ltt_cnt = 60 AND ltt_var > 0,
    ltt_cov / ltt_var,
    0.
  ) AS ltt_slope0
FROM (
  SELECT
    *,
    ifNull(
      if(twos_den_cnt >= 1, twos_den_mavg, CAST(NULL AS Nullable(Float64))),
      0.
    ) AS twos_den_final
  FROM (
{soir_sql}
  )
)
"""


def _soir_final_layer_sql(final_sql: str) -> str:
    """wavgSOIR output ffill + infer-price-trend mavg (second window pass)."""
    return f"""
SELECT
  *,
  anyLast(wavg_soir_raw) OVER run AS soir_ffill_val,
  ifNull(anyLast(wavg_soir_raw) OVER run, 0.) AS wavg_soir,
  avg(ltt_slope0) OVER w19 AS level10_infer_price_trend
FROM (
{final_sql}
)
WINDOW
  run AS (
    PARTITION BY symbol, TradeDate ORDER BY exch_time
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ),
  w19 AS (
    PARTITION BY symbol, TradeDate ORDER BY exch_time
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
  )
"""


def _diagnostic_layer_sql(final2_sql: str) -> str:
    """Per-snapshot diagnostic flags on final values (no new windows)."""
    diag_exprs = []
    for name in FACTOR_NAMES:
        guard = CLIP_GUARDS[name]
        # flags must stay non-NULL even when the factor value is NULL,
        # otherwise all-NULL groups (e.g. locked-book suspension days)
        # propagate NULL through the minute-level sumIf aggregates
        diag_exprs.append(
            f"  toUInt32(ifNull(isInfinite({name}), false))"
            f" AS {name}__inf_flag"
        )
        diag_exprs.append(
            f"  toUInt32(ifNull(abs({name}) > {guard!r}, false))"
            f" AS {name}__clip_flag"
        )
        diag_exprs.append(
            f"  toUInt32(isNull({name})) AS {name}__null_flag"
        )
    diag_block = ",\n".join(diag_exprs)
    return f"""
SELECT
  *,
  tra_price_raw AS tra_price_weighted_net_buy_quote_volume_ratio,
  toUInt32(
    twos_den_final != 0 AND abs(twos_den_final) < {SMALL_DENOMINATOR_EPS!r}
  ) AS time_weighted_order_slope__small_den_flag,
  toUInt32(isNull(twos_den_raw) AND isNotNull(twos_den_ffill))
  AS time_weighted_order_slope__ffill_flag,
  toUInt32(
    isNotNull(soir_std) AND soir_std < {SOIR_STD_FLOOR}
  ) AS wavg_soir__small_den_flag,
  toUInt32(isNull(wavg_soir_raw) AND isNotNull(soir_ffill_val))
    AS wavg_soir__ffill_flag,
  toUInt32(
    isNotNull(bid_chg) AND isNotNull(offer_chg)
    AND abs(bid_chg) + abs(offer_chg) = 0
  ) AS tra_price_weighted_net_buy_quote_volume_ratio__small_den_flag,
  toUInt32(0) AS tra_price_weighted_net_buy_quote_volume_ratio__ffill_flag,
  toUInt32(0) AS level10_diff_buy__small_den_flag,
  toUInt32(0) AS level10_diff_buy__ffill_flag,
  toUInt32(0) AS level10_infer_price_trend__small_den_flag,
  toUInt32(
    isNull(infer_price_raw) AND isNotNull(infer_ffill)
  ) AS level10_infer_price_trend__ffill_flag,
{diag_block}
FROM (
{final2_sql}
)
"""


def ddb_snapshot_series_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start: DateLike,
    end: DateLike,
    symbols: Optional[Sequence[str]] = None,
    raw_source_sql: Optional[str] = None,
) -> str:
    """Full 3s-series with all intermediate values (debug/golden tests)."""
    base = _base_extract_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        exchange=exchange,
        start=start,
        end=end,
        symbols=symbols,
        raw_source_sql=raw_source_sql,
    )
    query = _diagnostic_layer_sql(
        _soir_final_layer_sql(
            _final_value_layer_sql(
                _soir_stats_layer_sql(
                    _rolling_layer_sql(_compose_layer_sql(_window_layer_sql(base)))
                )
            )
        )
    )
    keep = ",\n  ".join(SERIES_COLUMNS)
    return f"""
SELECT
  {keep}
FROM (
{query}
)
ORDER BY symbol, exch_time
"""


def ddb_snapshot_daily_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start: DateLike,
    end: DateLike,
    symbols: Optional[Sequence[str]] = None,
    raw_source_sql: Optional[str] = None,
) -> str:
    """Symbol-day aggregate: minute-last on the 240-minute grid + daily mean."""
    base = _base_extract_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        exchange=exchange,
        start=start,
        end=end,
        symbols=symbols,
        raw_source_sql=raw_source_sql,
    )
    series = _diagnostic_layer_sql(
        _soir_final_layer_sql(
            _final_value_layer_sql(
                _soir_stats_layer_sql(
                    _rolling_layer_sql(_compose_layer_sql(_window_layer_sql(base)))
                )
            )
        )
    )
    factor_minute_cols = ",\n".join(
        f"  argMax({name}, exch_time) AS {name}" for name in FACTOR_NAMES
    )
    flag_sum_cols = ",\n".join(
        f"  toUInt32(sum({name}__{flag}_flag)) AS {name}__{flag}_sum"
        for name in FACTOR_NAMES
        for flag in ("small_den", "ffill", "inf", "clip", "null")
    )
    minute_sql = f"""
SELECT
  symbol,
  TradeDate,
  any(source_exchange) AS source_exchange,
  minute_index,
  is_close_auction,
  toUInt32(count()) AS minute_snapshot_count,
  toUInt32(sum(tie_flag)) AS minute_tie_count,
{factor_minute_cols},
{flag_sum_cols}
FROM (
{series}
)
GROUP BY symbol, TradeDate, minute_index, is_close_auction
"""
    factor_mean_cols = ",\n".join(
        f"  avgIf({name}, is_close_auction = 0) AS {name}_mean"
        for name in FACTOR_NAMES
    )
    factor_valid_cols = ",\n".join(
        f"  toInt64(countIf(is_close_auction = 0 AND isNotNull({name})))"
        f" AS {name}_valid_minute_count"
        for name in FACTOR_NAMES
    )
    diag_cols = ",\n".join(
        f"  toInt64(sumIf({name}__{flag}_sum, is_close_auction = 0))"
        f" AS {name}_{diag}"
        for name in FACTOR_NAMES
        for flag, diag in (
            ("small_den", "small_denominator_count"),
            ("ffill", "ffill_count"),
            ("inf", "inf_count"),
            ("clip", "clipped_count"),
            ("null", "null_snapshot_count"),
        )
    )
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
  toInt64(sumIf(minute_tie_count, is_close_auction = 0))
    AS exact_tie_row_count,
{factor_mean_cols},
{factor_valid_cols},
{diag_cols}
FROM (
{minute_sql}
)
GROUP BY symbol, TradeDate
ORDER BY TradeDate, symbol
"""


INTEGER_NA_FILL_WHITELIST_SUFFIXES = (
    "_clipped_count",
    "_small_denominator_count",
    "_ffill_count",
    "_tie_count",
)


def _integer_na_policy(column: str) -> str:
    """fill0 only for whitelisted diagnostic counters; anything else fails."""
    if column.endswith(INTEGER_NA_FILL_WHITELIST_SUFFIXES):
        return "fill0"
    return "hard_fail"


def prepare_ddb_snapshot_daily(
    frame: pd.DataFrame,
    na_report: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """Normalize and validate a daily factor frame returned by ClickHouse.

    NA policy (frozen): integer NA -> 0 is allowed ONLY for whitelisted
    diagnostic counters (*_clipped_count / *_small_denominator_count /
    *_ffill_count / *_tie_count). NA in base counters
    (valid_snapshot_count / valid_minute_count / expected_minute_count /
    exact_tie_row_count / per-factor *_valid_count), key columns
    (symbol / TradeDate) or any non-whitelisted column is a hard failure.
    Factor *_mean NULLs are by design (low-coverage days) and are never
    filled. When `na_report` is provided, every NA event is recorded with
    date / column / na_count / na_share / policy / affected_symbols.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(DAILY_COLUMNS))
    missing = set(DAILY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"DDB snapshot daily frame missing columns: {sorted(missing)}"
        )
    out = frame[list(DAILY_COLUMNS)].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["TradeDate"] = pd.to_datetime(out["TradeDate"]).dt.normalize()
    out["source_exchange"] = out["source_exchange"].astype(str)
    if out["symbol"].isna().any() or out["TradeDate"].isna().any():
        raise ValueError("NA in key columns symbol/TradeDate")
    integer_columns = [
        "valid_snapshot_count",
        "valid_minute_count",
        "expected_minute_count",
        "exact_tie_row_count",
        *VALID_VALUE_COLUMNS,
        *DIAG_COLUMNS,
    ]
    for column in integer_columns:
        series = pd.to_numeric(out[column], errors="coerce")
        na_mask = series.isna()
        na_count = int(na_mask.sum())
        if na_count:
            policy = _integer_na_policy(column)
            event = {
                "date": str(out["TradeDate"].min().date()),
                "column": column,
                "na_count": na_count,
                "na_share": na_count / len(series),
                "policy": policy,
                "affected_symbols": ",".join(
                    out.loc[na_mask, "symbol"].astype(str).head(20)
                ),
            }
            if na_report is not None:
                na_report.append(event)
            if policy == "hard_fail":
                raise ValueError(
                    f"{column}: {na_count}/{len(series)} NA on "
                    f"{event['date']} not in fill whitelist "
                    f"{INTEGER_NA_FILL_WHITELIST_SUFFIXES}; "
                    f"affected symbols sample: {event['affected_symbols']}"
                )
            print(
                f"  [warn] {column}: filling {na_count} NA -> 0 "
                f"({event['na_share']:.4%} of rows, whitelisted)",
                flush=True,
            )
            series = series.fillna(0)
        out[column] = series.astype("int64")
    numeric_columns = [
        column
        for column in DAILY_COLUMNS
        if column
        not in {"symbol", "TradeDate", "source_exchange", *integer_columns}
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    numeric_values = out[numeric_columns].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("DDB snapshot daily frame contains infinite values")
    if out.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Duplicate symbol/TradeDate in DDB snapshot daily")
    if not out["valid_minute_count"].between(0, EXPECTED_MINUTE_COUNT).all():
        raise ValueError("valid_minute_count outside [0, 240]")
    expected_coverage = out["valid_minute_count"] / EXPECTED_MINUTE_COUNT
    if not (out["coverage_ratio"] - expected_coverage).abs().le(1e-12).all():
        raise ValueError("coverage_ratio is inconsistent with minute counts")
    soir_values = out["wavg_soir_mean"].dropna()
    if not soir_values.between(-1e9, 1e9).all():
        raise ValueError("wavg_soir_mean contains absurd magnitudes")
    tpw_values = out[
        "tra_price_weighted_net_buy_quote_volume_ratio_mean"
    ].dropna()
    if not tpw_values.between(-1e9, 1e9).all():
        raise ValueError("tra price weighted ratio contains absurd magnitudes")
    for name in FACTOR_NAMES:
        inf_total = int(out[f"{name}_inf_count"].sum())
        if inf_total != 0:
            raise ValueError(f"{name}: inf_count={inf_total} must be 0")
    return out.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def fetch_ddb_snapshot_daily(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    tables: Iterable[Tuple[str, str, str]] = DDB_SNAPSHOT_TABLES,
    client=None,
    na_report: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """Fetch symbol-day factor rows for an inclusive date interval."""
    own_client = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, exchange in tables:
            query = ddb_snapshot_daily_sql(
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
        return pd.DataFrame(columns=list(DAILY_COLUMNS))
    return prepare_ddb_snapshot_daily(
        pd.concat(frames, ignore_index=True), na_report=na_report
    )


def fetch_ddb_snapshot_series(
    start: DateLike,
    end: DateLike,
    *,
    symbols: Sequence[str],
    tables: Iterable[Tuple[str, str, str]] = DDB_SNAPSHOT_TABLES,
    client=None,
) -> pd.DataFrame:
    """Fetch the full 3s series (debug/golden tests; small scopes only)."""
    if not symbols:
        raise ValueError("series fetch requires explicit symbols")
    own_client = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, exchange in tables:
            query = ddb_snapshot_series_sql(
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
        return pd.DataFrame(columns=list(SERIES_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    return out.sort_values(["symbol", "exch_time"]).reset_index(drop=True)
