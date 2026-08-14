"""Liquidity / Price Impact daily primitive from ClickHouse Tick + SSL2.

Design (Sprint 7 frozen):
- sources: cmds.LOCAL_SSE_AL_TICK_EXG / LOCAL_SZSE_AL_TICK_EXG (trades)
  joined server-side with cmds.LOCAL_SSE_AL_SSL2_EXG /
  LOCAL_SZSE_AL_SSL2_EXG (minute-last book state) on the frozen continuous
  auction grid 09:30-11:29 + 13:00-14:59 (240 minute labels).
- LOCAL MergeTree tables are queried directly; the Distributed tables are
  never joined (no distributed double JOIN, no raw Tick/Snapshot pull into
  pandas).
- direction rules (frozen):
  * SSE: BSFlag 'B' / 'S' / 'N' on Type='T' records.
  * SZSE: Type='011' AND Category='F' records; BidOrderNo > AskOrderNo ->
    active buy, '<' -> active sell, '=' -> neutral (0 observed in audit).
  * neutral amount is aggregated separately and reported daily.
- minute approximation: effective_spread_proxy and permanent impact use the
  minute-level signed direction and the minute-last midquote, NOT exact
  trade-quote matching; fields are named *_proxy accordingly.
- size buckets (frozen, CNY per trade): small <= 1e4; mid in (4e4, 2e5];
  large > 2e5; super_large > 1e6 (large overlaps super_large, matching the
  frozen order-size boundaries; buckets are evaluated independently).
- high-impact minutes: abs(minute_return) >= per-symbol-day 90th percentile
  (frozen top-10% definition; no grid over 5/10/20%).
- forward mid returns are computed strictly inside the symbol-day (no
  cross-day leakage); the backtest layer applies signal.shift(1) later.
- prices are unadjusted; all returns are within-day ratios.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DATABASE = "cmds"
EXPECTED_CONTINUOUS_MINUTES = 240
COVERAGE_THRESHOLD = 0.80
DEPTH_EPSILON_CNY = 1.0
CANONICAL_SOURCE = (
    "cmds.LOCAL_SSE_AL_TICK_EXG+cmds.LOCAL_SSE_AL_SSL2_EXG | "
    "cmds.LOCAL_SZSE_AL_TICK_EXG+cmds.LOCAL_SZSE_AL_SSL2_EXG"
)

EXCHANGES: Dict[str, Dict[str, str]] = {
    "sse": {
        "suffix": ".SH",
        "tick_table": f"{DATABASE}.LOCAL_SSE_AL_TICK_EXG",
        "book_table": f"{DATABASE}.LOCAL_SSE_AL_SSL2_EXG",
        "symbol_filter": "Symbol LIKE '6%'",
        "trade_filter": "Type = 'T'",
        "amount_expr": "toFloat64(Amount)",
        "buy_cond": "BSFlag = 'B'",
        "sell_cond": "BSFlag = 'S'",
        "neutral_cond": "BSFlag = 'N'",
    },
    "szse": {
        "suffix": ".SZ",
        "tick_table": f"{DATABASE}.LOCAL_SZSE_AL_TICK_EXG",
        "book_table": f"{DATABASE}.LOCAL_SZSE_AL_SSL2_EXG",
        "symbol_filter": (
            "(Symbol LIKE '000%' OR Symbol LIKE '001%' OR "
            "Symbol LIKE '002%' OR Symbol LIKE '003%' OR "
            "Symbol LIKE '300%' OR Symbol LIKE '301%')"
        ),
        "trade_filter": "Type = '011' AND Category = 'F'",
        "amount_expr": "toFloat64(Price) * toFloat64(Volume)",
        "buy_cond": "BidOrderNo > AskOrderNo",
        "sell_cond": "BidOrderNo < AskOrderNo",
        "neutral_cond": "BidOrderNo = AskOrderNo",
    },
}

SIZE_BUCKETS: Dict[str, str] = {
    "small": "amt <= 10000",
    "mid": "amt > 40000 AND amt <= 200000",
    "large": "amt > 200000",
    "super_large": "amt > 1000000",
}

DAILY_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "TradeDate",
    "expected_continuous_minutes",
    "coverage_ratio",
    "matched_minute_count",
    "trade_minute_count",
    "valid_book_minute_count",
    "directional_trade_share",
    "neutral_trade_share",
    "daily_amount",
    "daily_volume",
    "trade_count",
    "active_buy_amount",
    "active_sell_amount",
    "neutral_amount",
    "active_buy_count",
    "active_sell_count",
    "spread_per_depth",
    "depth_per_amount",
    "amount_to_depth",
    "depth_turnover",
    "liquidity_cost_state",
    "signed_amount_impact",
    "signed_sqrt_amount_impact",
    "impact_per_trade",
    "buy_price_impact",
    "sell_price_impact",
    "effective_spread_proxy",
    "permanent_impact_1m",
    "permanent_impact_5m",
    "small_trade_impact",
    "mid_trade_impact",
    "large_trade_impact",
    "super_large_trade_impact",
    "high_impact_threshold",
    "high_impact_minute_count",
    "spread_recovery_5m",
    "depth_recovery_5m",
    "exchange",
)


def _session_filter() -> str:
    return (
        "((toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30) "
        "OR toHour(ExchTime) = 10 "
        "OR (toHour(ExchTime) = 11 AND toMinute(ExchTime) <= 29) "
        "OR toHour(ExchTime) = 13 OR toHour(ExchTime) = 14)"
    )


def _dt64(day: str) -> str:
    return f"toDateTime64('{day} 00:00:00', 6, 'Asia/Shanghai')"


def minute_trade_sql(exchange: str, start: str, end: str) -> str:
    """Per symbol x minute trade aggregate, computed in ClickHouse."""
    cfg = EXCHANGES[exchange]
    bucket_signed = []
    for name, cond in SIZE_BUCKETS.items():
        bucket_signed.append(
            "sumIf(amt, {buy} AND {cond}) - sumIf(amt, {sell} AND {cond})"
            " AS signed_{name}_amount".format(
                buy=cfg["buy_cond"], sell=cfg["sell_cond"], cond=cond,
                name=name,
            )
        )
    bucket_clause = ",\n        ".join(bucket_signed)
    return f"""
SELECT
    Symbol AS symbol_raw,
    toDate(ExchTime) AS TradeDate,
    toHour(ExchTime) * 60 + toMinute(ExchTime) AS mkey,
    count() AS trade_count,
    sum(amt) AS trade_amount,
    sum(toFloat64(Volume)) AS trade_volume,
    sumIf(amt, {cfg['buy_cond']}) AS active_buy_amount,
    sumIf(amt, {cfg['sell_cond']}) AS active_sell_amount,
    sumIf(amt, {cfg['neutral_cond']}) AS neutral_amount,
    countIf({cfg['buy_cond']}) AS active_buy_count,
    countIf({cfg['sell_cond']}) AS active_sell_count,
        {bucket_clause}
FROM (
    SELECT Symbol, ExchTime, Volume, BidOrderNo, AskOrderNo,
        {cfg['amount_expr']} AS amt
        {', BSFlag' if exchange == 'sse' else ''}
    FROM {cfg['tick_table']}
    WHERE ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
        AND {cfg['trade_filter']}
        AND {cfg['symbol_filter']}
        AND {_session_filter()}
)
GROUP BY symbol_raw, TradeDate, mkey
"""


def minute_book_sql(exchange: str, start: str, end: str) -> str:
    """Per symbol x minute last book state, computed in ClickHouse."""
    cfg = EXCHANGES[exchange]
    return f"""
SELECT
    symbol_raw,
    TradeDate,
    mkey,
    ifNull(toFloat64(book.1[1]), 0.0) AS bid1,
    ifNull(toFloat64(book.3[1]), 0.0) AS ask1,
    toFloat64(arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.0),
        arraySlice(book.2, 1, 1)))) AS bid_depth_1,
    toFloat64(arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.0),
        arraySlice(book.4, 1, 1)))) AS ask_depth_1,
    toFloat64(arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.0),
        arraySlice(book.2, 1, 5)))) AS bid_depth_5,
    toFloat64(arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.0),
        arraySlice(book.4, 1, 5)))) AS ask_depth_5
FROM (
    SELECT
        Symbol AS symbol_raw,
        toDate(ExchTime) AS TradeDate,
        toHour(ExchTime) * 60 + toMinute(ExchTime) AS mkey,
        argMax((BidPrices, BidVolumes, AskPrices, AskVolumes), ExchTime) AS book
    FROM {cfg['book_table']}
    WHERE ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
        AND {cfg['symbol_filter']}
        AND {_session_filter()}
        AND length(BidPrices) > 0 AND length(AskPrices) > 0
    GROUP BY symbol_raw, TradeDate, mkey
)
"""


def joined_minute_sql(exchange: str, start: str, end: str) -> str:
    """Server-side minute join of trade aggregate and last book state."""
    trade = minute_trade_sql(exchange, start, end)
    book = minute_book_sql(exchange, start, end)
    return f"""
SELECT
    multiIf(t.symbol_raw != '', t.symbol_raw, b.symbol_raw) AS symbol_raw,
    multiIf(t.TradeDate != toDate(0), t.TradeDate, b.TradeDate) AS TradeDate,
    multiIf(t.mkey != 0, t.mkey, b.mkey) AS mkey,
    ifNull(t.trade_count, 0) > 0 AS has_trade,
    ifNull(b.bid1, 0.0) > 0 AND ifNull(b.ask1, 0.0) > 0
        AND ifNull(b.ask1, 0.0) >= ifNull(b.bid1, 0.0) AS has_book,
    ifNull(t.trade_count, 0) AS trade_count,
    ifNull(t.trade_amount, 0.0) AS trade_amount,
    ifNull(t.trade_volume, 0.0) AS trade_volume,
    ifNull(t.active_buy_amount, 0.0) AS active_buy_amount,
    ifNull(t.active_sell_amount, 0.0) AS active_sell_amount,
    ifNull(t.neutral_amount, 0.0) AS neutral_amount,
    ifNull(t.active_buy_count, 0) AS active_buy_count,
    ifNull(t.active_sell_count, 0) AS active_sell_count,
    ifNull(t.signed_small_amount, 0.0) AS signed_small_amount,
    ifNull(t.signed_mid_amount, 0.0) AS signed_mid_amount,
    ifNull(t.signed_large_amount, 0.0) AS signed_large_amount,
    ifNull(t.signed_super_large_amount, 0.0) AS signed_super_large_amount,
    if(trade_volume > 0, trade_amount / trade_volume, nan) AS trade_vwap,
    ifNull(b.bid1, 0.0) AS bid1,
    ifNull(b.ask1, 0.0) AS ask1,
    ifNull(b.bid_depth_1, 0.0) AS bid_depth_1,
    ifNull(b.ask_depth_1, 0.0) AS ask_depth_1,
    ifNull(b.bid_depth_5, 0.0) AS bid_depth_5,
    ifNull(b.ask_depth_5, 0.0) AS ask_depth_5
FROM (
    {trade}
) AS t
FULL OUTER JOIN (
    {book}
) AS b
ON t.symbol_raw = b.symbol_raw AND t.TradeDate = b.TradeDate
    AND t.mkey = b.mkey
"""


def daily_sql(exchange: str, start: str, end: str) -> str:
    """Full server-side Symbol x TradeDate liquidity-impact primitive.

    Layer 1 joins minute trades with minute-last book state; layer 2 packs
    per-day sorted arrays plus plain aggregates; layer 3 derives
    forward-return / high-impact-minute fields from the arrays. Forward
    references never cross the day boundary because arrays are per day and
    lead conditions require exact consecutive minute keys.
    """
    joined = joined_minute_sql(exchange, start, end)
    suffix = EXCHANGES[exchange]["suffix"]
    return f"""
WITH daily AS (
    SELECT
        symbol_raw,
        TradeDate,
        countIf(has_trade AND has_book) AS matched_minute_count,
        countIf(has_trade) AS trade_minute_count,
        countIf(has_book) AS valid_book_minute_count,
        sum(trade_amount) AS d_daily_amount,
        sum(trade_volume) AS d_daily_volume,
        sum(trade_count) AS d_trade_count,
        sum(active_buy_amount) AS d_active_buy_amount,
        sum(active_sell_amount) AS d_active_sell_amount,
        sum(neutral_amount) AS d_neutral_amount,
        sum(active_buy_count) AS d_active_buy_count,
        sum(active_sell_count) AS d_active_sell_count,
        avgIf(spread / log1p(depth5), has_book AND depth5 > 0)
            AS spread_per_depth,
        avgIf(depth5 / (trade_amount + {DEPTH_EPSILON_CNY}),
            has_trade AND depth5 > 0) AS depth_per_amount,
        sum(trade_amount) / avgIf(depth5, has_book AND depth5 > 0)
            AS amount_to_depth,
        sum(trade_volume) / avgIf(depth5, has_book AND depth5 > 0)
            AS depth_turnover,
        avgIf(spread * trade_amount / depth5,
            has_trade AND has_book AND depth5 > 0) AS liquidity_cost_state,
        arraySort(x -> x.1, groupArray((
            mkey, mq, spread, depth5, signed_amt, signed_cnt,
            active_buy_amount, active_sell_amount, trade_vwap,
            signed_small_amount, signed_mid_amount, signed_large_amount,
            signed_super_large_amount
        ))) AS arr
    FROM (
        SELECT
            symbol_raw, TradeDate, mkey, has_trade, has_book,
            trade_count, trade_amount, trade_volume,
            active_buy_amount, active_sell_amount, neutral_amount,
            active_buy_count, active_sell_count,
            signed_small_amount, signed_mid_amount,
            signed_large_amount, signed_super_large_amount,
            trade_vwap,
            if(has_book, (bid1 + ask1) / 2, nan) AS mq,
            if(has_book, (ask1 - bid1) / ((bid1 + ask1) / 2), nan) AS spread,
            bid_depth_5 + ask_depth_5 AS depth5,
            active_buy_amount - active_sell_amount AS signed_amt,
            toFloat64(active_buy_count) - toFloat64(active_sell_count)
                AS signed_cnt
        FROM (
            {joined}
        )
    )
    GROUP BY symbol_raw, TradeDate
)
SELECT
    symbol_raw,
    TradeDate,
    {EXPECTED_CONTINUOUS_MINUTES} AS expected_continuous_minutes,
    matched_minute_count / {EXPECTED_CONTINUOUS_MINUTES} AS coverage_ratio,
    matched_minute_count,
    trade_minute_count,
    valid_book_minute_count,
    if(d_daily_amount > 0,
        (d_active_buy_amount + d_active_sell_amount) / d_daily_amount, nan)
        AS directional_trade_share,
    if(d_daily_amount > 0, d_neutral_amount / d_daily_amount, nan)
        AS neutral_trade_share,
    d_daily_amount AS daily_amount,
    d_daily_volume AS daily_volume,
    d_trade_count AS trade_count,
    d_active_buy_amount AS active_buy_amount,
    d_active_sell_amount AS active_sell_amount,
    d_neutral_amount AS neutral_amount,
    d_active_buy_count AS active_buy_count,
    d_active_sell_count AS active_sell_count,
    spread_per_depth,
    depth_per_amount,
    amount_to_depth,
    depth_turnover,
    liquidity_cost_state,
    arrayMap(x -> x.1, arr) AS mi,
    arrayMap(x -> x.2, arr) AS mq,
    arrayMap(x -> x.3, arr) AS sp,
    arrayMap(x -> x.4, arr) AS dp,
    arrayMap(x -> x.5, arr) AS sa,
    arrayMap(x -> x.6, arr) AS sc,
    arrayMap(x -> x.7, arr) AS ab,
    arrayMap(x -> x.8, arr) AS asl,
    arrayMap(x -> x.9, arr) AS vw,
    arrayMap(i -> if(i > 1 AND mi[i] = mi[i - 1] + 1
        AND mq[i] > 0 AND mq[i - 1] > 0,
        log(mq[i] / mq[i - 1]), nan), arrayEnumerate(mi)) AS r,
    arrayMap(i -> if(mi[i + 1] = mi[i] + 1
        AND mq[i] > 0 AND mq[i + 1] > 0,
        mq[i + 1] / mq[i] - 1, nan), arrayEnumerate(mi)) AS fwd1,
    arrayMap(i -> if(mi[i + 5] = mi[i] + 5
        AND mq[i] > 0 AND mq[i + 5] > 0,
        mq[i + 5] / mq[i] - 1, nan), arrayEnumerate(mi)) AS fwd5,
    arraySum(arrayMap((ri, si) -> if(NOT isNaN(ri), ri * si, 0), r, sa))
        / nullIf(arraySum(arrayMap((ri, si) ->
            if(NOT isNaN(ri), abs(si), 0), r, sa)), 0)
        AS signed_amount_impact,
    arraySum(arrayMap((ri, si) -> if(NOT isNaN(ri),
        ri * sign(si) * sqrt(abs(si)), 0), r, sa))
        / nullIf(arraySum(arrayMap((ri, si) ->
            if(NOT isNaN(ri), sqrt(abs(si)), 0), r, sa)), 0)
        AS signed_sqrt_amount_impact,
    arraySum(arrayMap((ri, si) -> if(NOT isNaN(ri), ri * si, 0), r, sc))
        / nullIf(arraySum(arrayMap((ri, si) ->
            if(NOT isNaN(ri), abs(si), 0), r, sc)), 0)
        AS impact_per_trade,
    arraySum(arrayMap((fi, ai) ->
        if(NOT isNaN(fi) AND ai > 0, fi * ai, 0), fwd1, ab))
        / nullIf(arraySum(arrayMap((fi, ai) ->
            if(NOT isNaN(fi) AND ai > 0, ai, 0), fwd1, ab)), 0)
        AS buy_price_impact,
    arraySum(arrayMap((fi, ai) ->
        if(NOT isNaN(fi) AND ai > 0, fi * ai, 0), fwd1, asl))
        / nullIf(arraySum(arrayMap((fi, ai) ->
            if(NOT isNaN(fi) AND ai > 0, ai, 0), fwd1, asl)), 0)
        AS sell_price_impact,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap((si, vi, mqi) ->
        if(si != 0 AND NOT isNaN(vi) AND mqi > 0,
        2 * sign(si) * (vi - mqi) / mqi, nan), sa, vw, mq)))
        AS effective_spread_proxy,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap((si, fi) ->
        if(si != 0 AND NOT isNaN(fi), 2 * sign(si) * fi, nan), sa, fwd1)))
        AS permanent_impact_1m,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap((si, fi) ->
        if(si != 0 AND NOT isNaN(fi), 2 * sign(si) * fi, nan), sa, fwd5)))
        AS permanent_impact_5m,
    arraySum(arrayMap((fi, si) ->
        if(NOT isNaN(fi), fi * si, 0), fwd1, arrayMap(x -> x.10, arr)))
        / nullIf(arraySum(arrayMap((fi, si) ->
            if(NOT isNaN(fi), abs(si), 0), fwd1, arrayMap(x -> x.10, arr))), 0)
        AS small_trade_impact,
    arraySum(arrayMap((fi, si) ->
        if(NOT isNaN(fi), fi * si, 0), fwd1, arrayMap(x -> x.11, arr)))
        / nullIf(arraySum(arrayMap((fi, si) ->
            if(NOT isNaN(fi), abs(si), 0), fwd1, arrayMap(x -> x.11, arr))), 0)
        AS mid_trade_impact,
    arraySum(arrayMap((fi, si) ->
        if(NOT isNaN(fi), fi * si, 0), fwd1, arrayMap(x -> x.12, arr)))
        / nullIf(arraySum(arrayMap((fi, si) ->
            if(NOT isNaN(fi), abs(si), 0), fwd1, arrayMap(x -> x.12, arr))), 0)
        AS large_trade_impact,
    arraySum(arrayMap((fi, si) ->
        if(NOT isNaN(fi), fi * si, 0), fwd1, arrayMap(x -> x.13, arr)))
        / nullIf(arraySum(arrayMap((fi, si) ->
            if(NOT isNaN(fi), abs(si), 0), fwd1, arrayMap(x -> x.13, arr))), 0)
        AS super_large_trade_impact,
    arrayReduce('quantile(0.9)', arrayFilter(x -> NOT isNaN(x),
        arrayMap(x -> abs(x), r))) AS high_impact_threshold,
    arrayMap(i -> if(NOT isNaN(r[i]) AND abs(r[i]) >= high_impact_threshold
        AND high_impact_threshold > 0, 1, 0), arrayEnumerate(mi)) AS hi,
    arraySum(hi) AS high_impact_minute_count,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(hi[i] = 1 AND mi[i + 5] = mi[i] + 5 AND sp[i] > 0
            AND NOT isNaN(sp[i + 5]),
            (sp[i] - sp[i + 5]) / sp[i], nan), arrayEnumerate(mi))))
        AS spread_recovery_5m,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(hi[i] = 1 AND mi[i + 5] = mi[i] + 5 AND dp[i] > 0
            AND NOT isNaN(dp[i + 5]),
            (dp[i + 5] - dp[i]) / dp[i], nan), arrayEnumerate(mi))))
        AS depth_recovery_5m,
    '{suffix}' AS exchange
FROM daily
"""


def query_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def finalize_daily(
    frames: List[pd.DataFrame], *, start: str, end: str
) -> pd.DataFrame:
    """Concatenate exchange frames, attach WindCode symbols, sort, validate."""
    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol_raw"].astype(str) + frame["exchange"]
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
    frame = frame.drop(columns=["symbol_raw"])
    frame = frame.sort_values(["TradeDate", "symbol"], kind="stable")
    frame = frame.reset_index(drop=True)
    return frame[list(DAILY_COLUMNS)]


def prepare_liquidity_impact_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate primitive schema and hard invariants (fail fast)."""
    missing = [c for c in DAILY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing primitive columns: {missing}")
    key = ["symbol", "TradeDate"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate symbol-day rows in primitive")
    numeric = frame.drop(columns=key + ["exchange"])
    if np.isinf(numeric.select_dtypes("number").to_numpy()).any():
        raise ValueError("inf values in primitive")
    checks = {
        "coverage_ratio": (0.0, 1.0),
        "directional_trade_share": (0.0, 1.0),
        "neutral_trade_share": (0.0, 1.0),
    }
    tolerance = 1e-9
    for column, (lo, hi) in checks.items():
        values = frame[column].dropna()
        if len(values) and (
            (values < lo - tolerance) | (values > hi + tolerance)
        ).any():
            raise ValueError(f"{column} outside [{lo}, {hi}]")
        frame[column] = frame[column].clip(lo, hi)
    for column in ("daily_amount", "daily_volume", "trade_count"):
        if (frame[column] < 0).any():
            raise ValueError(f"negative {column}")
    return frame


__all__ = [
    "CANONICAL_SOURCE",
    "COVERAGE_THRESHOLD",
    "DAILY_COLUMNS",
    "EXPECTED_CONTINUOUS_MINUTES",
    "EXCHANGES",
    "SIZE_BUCKETS",
    "daily_sql",
    "finalize_daily",
    "joined_minute_sql",
    "minute_book_sql",
    "minute_trade_sql",
    "prepare_liquidity_impact_daily",
    "query_sha256",
]
