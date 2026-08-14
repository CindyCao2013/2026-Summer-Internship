"""Directional refill asymmetry daily primitive (Sprint 13).

Reuses liquidity_impact Tick+SSL2 minute join. Does NOT mutate frozen
liquidity_impact_daily_v1. Packs bid_depth_5 / ask_depth_5 separately and
aggregates side-conditioned post-shock refill.

Frozen shock / recovery definitions (see sprint13 audit + schema):

  hi_t     = 1{|r_t| >= symbol-day 90th pct of |r|}   # same as depth_recovery_5m
  sell_shock_t = hi_t AND active_sell_amount_t > active_buy_amount_t
                 AND active_sell_amount_t > 0
  buy_shock_t  = hi_t AND active_buy_amount_t  > active_sell_amount_t
                 AND active_buy_amount_t  > 0

  bid_recovery_5m = mean( (bid5_{t+5}-bid5_t)/bid5_t | sell_shock )
  ask_recovery_5m = mean( (ask5_{t+5}-ask5_t)/ask5_t | buy_shock )
  directional_refill_asymmetry = bid_recovery_5m - ask_recovery_5m

Forward windows never cross the day boundary (exact consecutive minute keys).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python import liquidity_impact_daily as lid

DATABASE = lid.DATABASE
EXPECTED_CONTINUOUS_MINUTES = lid.EXPECTED_CONTINUOUS_MINUTES
COVERAGE_THRESHOLD = lid.COVERAGE_THRESHOLD
CANONICAL_SOURCE = lid.CANONICAL_SOURCE
EXCHANGES = lid.EXCHANGES

SCHEMA_VERSION = "directional_refill_daily_v1"
FORMULA_VERSION = "frozen_v1"

DAILY_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "TradeDate",
    "expected_continuous_minutes",
    "coverage_ratio",
    "matched_minute_count",
    "trade_minute_count",
    "valid_book_minute_count",
    "high_impact_threshold",
    "high_impact_minute_count",
    "sell_shock_minute_count",
    "buy_shock_minute_count",
    "sell_shock_event_count",
    "buy_shock_event_count",
    "depth_recovery_5m",
    "bid_recovery_5m",
    "ask_recovery_5m",
    "directional_refill_asymmetry",
    "active_buy_amount",
    "active_sell_amount",
    "exchange",
)


def daily_sql(exchange: str, start: str, end: str) -> str:
    """Server-side Symbol x TradeDate directional refill primitive."""
    joined = lid.joined_minute_sql(exchange, start, end)
    suffix = EXCHANGES[exchange]["suffix"]
    return f"""
WITH daily AS (
    SELECT
        symbol_raw,
        TradeDate,
        countIf(has_trade AND has_book) AS matched_minute_count,
        countIf(has_trade) AS trade_minute_count,
        countIf(has_book) AS valid_book_minute_count,
        sum(active_buy_amount) AS d_active_buy_amount,
        sum(active_sell_amount) AS d_active_sell_amount,
        arraySort(x -> x.1, groupArray((
            mkey, mq, depth5, bid5, ask5, active_buy_amount, active_sell_amount
        ))) AS arr
    FROM (
        SELECT
            symbol_raw, TradeDate, mkey, has_trade, has_book,
            active_buy_amount, active_sell_amount,
            if(has_book, (bid1 + ask1) / 2, nan) AS mq,
            bid_depth_5 AS bid5,
            ask_depth_5 AS ask5,
            bid_depth_5 + ask_depth_5 AS depth5
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
    arrayMap(x -> x.1, arr) AS mi,
    arrayMap(x -> x.2, arr) AS mq,
    arrayMap(x -> x.3, arr) AS dp,
    arrayMap(x -> x.4, arr) AS bd,
    arrayMap(x -> x.5, arr) AS ad,
    arrayMap(x -> x.6, arr) AS ab,
    arrayMap(x -> x.7, arr) AS asl,
    arrayMap(i -> if(i > 1 AND mi[i] = mi[i - 1] + 1
        AND mq[i] > 0 AND mq[i - 1] > 0,
        log(mq[i] / mq[i - 1]), nan), arrayEnumerate(mi)) AS r,
    arrayReduce('quantile(0.9)', arrayFilter(x -> NOT isNaN(x),
        arrayMap(x -> abs(x), r))) AS high_impact_threshold,
    arrayMap(i -> if(NOT isNaN(r[i]) AND abs(r[i]) >= high_impact_threshold
        AND high_impact_threshold > 0, 1, 0), arrayEnumerate(mi)) AS hi,
    arraySum(hi) AS high_impact_minute_count,
    arrayMap(i -> if(hi[i] = 1 AND asl[i] > ab[i] AND asl[i] > 0, 1, 0),
        arrayEnumerate(mi)) AS sell_shock,
    arrayMap(i -> if(hi[i] = 1 AND ab[i] > asl[i] AND ab[i] > 0, 1, 0),
        arrayEnumerate(mi)) AS buy_shock,
    arraySum(sell_shock) AS sell_shock_minute_count,
    arraySum(buy_shock) AS buy_shock_minute_count,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(hi[i] = 1 AND mi[i + 5] = mi[i] + 5 AND dp[i] > 0
            AND NOT isNaN(dp[i + 5]),
            (dp[i + 5] - dp[i]) / dp[i], nan), arrayEnumerate(mi))))
        AS depth_recovery_5m,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(sell_shock[i] = 1 AND mi[i + 5] = mi[i] + 5 AND bd[i] > 0
            AND NOT isNaN(bd[i + 5]),
            (bd[i + 5] - bd[i]) / bd[i], nan), arrayEnumerate(mi))))
        AS bid_recovery_5m,
    length(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(sell_shock[i] = 1 AND mi[i + 5] = mi[i] + 5 AND bd[i] > 0
            AND NOT isNaN(bd[i + 5]),
            (bd[i + 5] - bd[i]) / bd[i], nan), arrayEnumerate(mi))))
        AS sell_shock_event_count,
    arrayAvg(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(buy_shock[i] = 1 AND mi[i + 5] = mi[i] + 5 AND ad[i] > 0
            AND NOT isNaN(ad[i + 5]),
            (ad[i + 5] - ad[i]) / ad[i], nan), arrayEnumerate(mi))))
        AS ask_recovery_5m,
    length(arrayFilter(x -> NOT isNaN(x), arrayMap(i ->
        if(buy_shock[i] = 1 AND mi[i + 5] = mi[i] + 5 AND ad[i] > 0
            AND NOT isNaN(ad[i + 5]),
            (ad[i + 5] - ad[i]) / ad[i], nan), arrayEnumerate(mi))))
        AS buy_shock_event_count,
    if(isNaN(bid_recovery_5m) OR isNaN(ask_recovery_5m), nan,
        bid_recovery_5m - ask_recovery_5m) AS directional_refill_asymmetry,
    d_active_buy_amount AS active_buy_amount,
    d_active_sell_amount AS active_sell_amount,
    '{suffix}' AS exchange
FROM daily
"""


def query_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def finalize_daily(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol_raw"].astype(str) + frame["exchange"]
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
    frame = frame.drop(columns=["symbol_raw"])
    # ClickHouse arrayAvg over an empty filter returns 0; treat no-event days as NA.
    bid = frame["bid_recovery_5m"].astype(float).to_numpy().copy()
    ask = frame["ask_recovery_5m"].astype(float).to_numpy().copy()
    bid[frame["sell_shock_event_count"].to_numpy() <= 0] = np.nan
    ask[frame["buy_shock_event_count"].to_numpy() <= 0] = np.nan
    frame["bid_recovery_5m"] = bid
    frame["ask_recovery_5m"] = ask
    frame["directional_refill_asymmetry"] = bid - ask
    depth = frame["depth_recovery_5m"].astype(float).to_numpy().copy()
    depth[frame["high_impact_minute_count"].to_numpy() <= 0] = np.nan
    frame["depth_recovery_5m"] = depth
    frame = frame.sort_values(["TradeDate", "symbol"], kind="stable")
    frame = frame.reset_index(drop=True)
    return frame[list(DAILY_COLUMNS)]


def prepare_directional_refill_daily(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in DAILY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing primitive columns: {missing}")
    key = ["symbol", "TradeDate"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate symbol/TradeDate in directional_refill_daily")
    if (frame["coverage_ratio"] < 0).any() or (frame["coverage_ratio"] > 1.01).any():
        raise ValueError("coverage_ratio out of [0, 1]")
    for col in (
        "bid_recovery_5m",
        "ask_recovery_5m",
        "directional_refill_asymmetry",
        "depth_recovery_5m",
    ):
        if np.isinf(frame[col].to_numpy(dtype=float)).any():
            raise ValueError(f"inf values in {col}")
    return frame


PRIMITIVE_FORMULAS: Dict[str, str] = {
    "high_impact": (
        "hi_t = 1{|minute_log_mid_return_t| >= symbol-day quantile_0.9(|r|)}"
    ),
    "sell_shock": (
        "sell_shock_t = hi_t AND active_sell_amount_t > active_buy_amount_t "
        "AND active_sell_amount_t > 0"
    ),
    "buy_shock": (
        "buy_shock_t = hi_t AND active_buy_amount_t > active_sell_amount_t "
        "AND active_buy_amount_t > 0"
    ),
    "bid_recovery_5m": (
        "mean((bid_depth5_{t+5}-bid_depth5_t)/bid_depth5_t | sell_shock_t, "
        "mi_{t+5}=mi_t+5, bid_depth5_t>0)"
    ),
    "ask_recovery_5m": (
        "mean((ask_depth5_{t+5}-ask_depth5_t)/ask_depth5_t | buy_shock_t, "
        "mi_{t+5}=mi_t+5, ask_depth5_t>0)"
    ),
    "directional_refill_asymmetry": "bid_recovery_5m - ask_recovery_5m",
    "depth_recovery_5m": (
        "mean((depth5_{t+5}-depth5_t)/depth5_t | hi_t, mi_{t+5}=mi_t+5, "
        "depth5_t>0); depth5=bid5+ask5; parity vs liquidity_impact_daily"
    ),
}
