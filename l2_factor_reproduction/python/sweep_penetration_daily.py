"""Conservative sweep / book-penetration daily primitive (Sprint 14B).

Contract frozen from Sprint 14A (PARTIAL):

  reference_book(t) = latest VALID SSL2 snapshot with ExchTime
                      STRICTLY BEFORE trade ExchTime
  BUY  -> ASK ladder only
  SELL -> BID ladder only

All level metrics are *estimated* vs a ~3s-stale book.
Trade-print unit only (no parent-order recombination).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python import liquidity_impact_daily as lid

DATABASE = lid.DATABASE
EXCHANGES = lid.EXCHANGES
CANONICAL_SOURCE = lid.CANONICAL_SOURCE
SCHEMA_VERSION = "sweep_penetration_daily_v1"
FORMULA_VERSION = "frozen_conservative_v1_from_14a"

# Lag beyond this (ms) is flagged stale_alignment ( > ~1.5 snapshot intervals ).
STALE_LAG_MS = 5000

DAILY_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "TradeDate",
    "total_event_count",
    "usable_event_count",
    "ambiguous_event_share",
    "sweep_2plus_count",
    "sweep_2plus_share",
    "sweep_notional_share",
    "mean_estimated_levels_penetrated",
    "mean_depth_consumed_ratio",
    "mean_penetration_price_distance",
    "buy_sweep_share",
    "sell_sweep_share",
    "sweep_directional_asymmetry",
    "median_alignment_lag_ms",
    "mean_trade_amount_usable",
    "exchange",
)


def _session_filter(alias: str = "ExchTime") -> str:
    """Continuous auction only; exclude open/close auction endpoints."""
    return (
        f"((toHour({alias}) = 9 AND toMinute({alias}) >= 30) "
        f"OR toHour({alias}) = 10 "
        f"OR (toHour({alias}) = 11 AND toMinute({alias}) < 30) "
        f"OR toHour({alias}) = 13 "
        f"OR (toHour({alias}) = 14 AND toMinute({alias}) < 57))"
    )


def _dt64(day: str) -> str:
    return f"toDateTime64('{day} 00:00:00', 6, 'Asia/Shanghai')"


def daily_sql(exchange: str, start: str, end: str) -> str:
    """Symbol x TradeDate conservative sweep aggregates via ASOF book join."""
    cfg = EXCHANGES[exchange]
    suffix = cfg["suffix"]
    if exchange == "sse":
        trade_select = f"""
        SELECT
            Symbol,
            toDate(ExchTime) AS TradeDate,
            ExchTime AS trade_time,
            toFloat64(Price) AS trade_price,
            toFloat64(Volume) AS trade_volume,
            toFloat64(Amount) AS trade_amount,
            if(BSFlag = 'B', 1, -1) AS trade_direction
        FROM {cfg['tick_table']}
        WHERE ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
            AND {cfg['symbol_filter']}
            AND Type = 'T' AND BSFlag IN ('B', 'S')
            AND {_session_filter('ExchTime')}
        """
    else:
        trade_select = f"""
        SELECT
            Symbol,
            toDate(ExchTime) AS TradeDate,
            ExchTime AS trade_time,
            toFloat64(Price) AS trade_price,
            toFloat64(Volume) AS trade_volume,
            toFloat64(Price) * toFloat64(Volume) AS trade_amount,
            if(BidOrderNo > AskOrderNo, 1, -1) AS trade_direction
        FROM {cfg['tick_table']}
        WHERE ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
            AND {cfg['symbol_filter']}
            AND Type = '011' AND Category = 'F'
            AND BidOrderNo != AskOrderNo
            AND {_session_filter('ExchTime')}
        """

    # Books: include a few minutes before session for early trades.
    book_select = f"""
        SELECT
            Symbol,
            ExchTime AS book_time,
            arrayMap(x -> ifNull(toFloat64(x), 0.), AskPrices) AS ask_px,
            arrayMap(x -> ifNull(toFloat64(x), 0.), AskVolumes) AS ask_vol,
            arrayMap(x -> ifNull(toFloat64(x), 0.), BidPrices) AS bid_px,
            arrayMap(x -> ifNull(toFloat64(x), 0.), BidVolumes) AS bid_vol
        FROM {cfg['book_table']}
        WHERE ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
            AND {cfg['symbol_filter']}
            AND length(BidPrices) > 0 AND length(AskPrices) > 0
            AND toFloat64(AskPrices[1]) > 0 AND toFloat64(BidPrices[1]) > 0
            AND toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])
            AND (
                (toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 25)
                OR toHour(ExchTime) = 10
                OR (toHour(ExchTime) = 11 AND toMinute(ExchTime) < 30)
                OR (toHour(ExchTime) = 12 AND toMinute(ExchTime) >= 55)
                OR toHour(ExchTime) = 13
                OR (toHour(ExchTime) = 14 AND toMinute(ExchTime) < 57)
            )
    """

    return f"""
WITH
trades AS ({trade_select}),
books AS ({book_select}),
joined AS (
    SELECT
        t.Symbol AS symbol_raw,
        t.TradeDate AS TradeDate,
        t.trade_direction AS trade_direction,
        t.trade_amount AS trade_amount,
        t.trade_price AS trade_price,
        t.trade_volume AS trade_volume,
        b.book_time AS book_time,
        b.ask_px AS ask_px,
        b.ask_vol AS ask_vol,
        b.bid_px AS bid_px,
        b.bid_vol AS bid_vol,
        toUnixTimestamp64Milli(t.trade_time)
            - toUnixTimestamp64Milli(b.book_time) AS alignment_lag_ms,
        if(t.trade_direction = 1,
            toUInt32(length(arrayFilter(
                p -> p > 0 AND p <= t.trade_price, b.ask_px))),
            toUInt32(length(arrayFilter(
                p -> p > 0 AND p >= t.trade_price, b.bid_px)))
        ) AS estimated_levels_penetrated,
        if(t.trade_direction = 1,
            t.trade_volume / nullIf(arraySum(b.ask_vol), 0),
            t.trade_volume / nullIf(arraySum(b.bid_vol), 0)
        ) AS estimated_depth_consumed_ratio,
        if(t.trade_direction = 1,
            (t.trade_price - b.ask_px[1]) / nullIf(b.ask_px[1], 0),
            (b.bid_px[1] - t.trade_price) / nullIf(b.bid_px[1], 0)
        ) AS penetration_price_distance
    FROM trades AS t
    ASOF LEFT JOIN books AS b
        ON t.Symbol = b.Symbol AND t.trade_time > b.book_time
),
flagged AS (
    SELECT
        symbol_raw,
        TradeDate,
        trade_direction,
        trade_amount,
        estimated_levels_penetrated,
        estimated_depth_consumed_ratio,
        penetration_price_distance,
        alignment_lag_ms,
        multiIf(
            book_time < toDateTime64('2018-01-01 00:00:00', 6, 'Asia/Shanghai'),
                'missing_reference_book',
            trade_direction = 1 AND (
                length(ask_px) = 0 OR trade_price + 0.000001 < ask_px[1]
            ), 'price_not_on_ladder',
            trade_direction = -1 AND (
                length(bid_px) = 0 OR trade_price - 0.000001 > bid_px[1]
            ), 'price_not_on_ladder',
            estimated_levels_penetrated = 0, 'price_not_on_ladder',
            alignment_lag_ms > {STALE_LAG_MS} OR alignment_lag_ms < 0,
                'stale_alignment',
            'ok'
        ) AS quality_flag
    FROM joined
),
marked AS (
    SELECT
        *,
        quality_flag = 'ok' AS usable,
        (quality_flag = 'ok') AND (estimated_levels_penetrated >= 2) AS is_sweep
    FROM flagged
)
SELECT
    symbol_raw,
    TradeDate,
    count() AS total_event_count,
    countIf(usable) AS usable_event_count,
    countIf(NOT usable) / count() AS ambiguous_event_share,
    countIf(is_sweep) AS sweep_2plus_count,
    countIf(is_sweep) / nullIf(countIf(usable), 0) AS sweep_2plus_share,
    sumIf(trade_amount, is_sweep)
        / nullIf(sumIf(trade_amount, usable), 0) AS sweep_notional_share,
    avgIf(estimated_levels_penetrated, usable)
        AS mean_estimated_levels_penetrated,
    avgIf(estimated_depth_consumed_ratio, usable)
        AS mean_depth_consumed_ratio,
    avgIf(penetration_price_distance, usable)
        AS mean_penetration_price_distance,
    countIf(usable AND trade_direction = 1 AND estimated_levels_penetrated >= 2)
        / nullIf(countIf(usable AND trade_direction = 1), 0)
        AS buy_sweep_share,
    countIf(usable AND trade_direction = -1 AND estimated_levels_penetrated >= 2)
        / nullIf(countIf(usable AND trade_direction = -1), 0)
        AS sell_sweep_share,
    buy_sweep_share - sell_sweep_share AS sweep_directional_asymmetry,
    quantileIf(0.5)(alignment_lag_ms, usable) AS median_alignment_lag_ms,
    avgIf(trade_amount, usable) AS mean_trade_amount_usable,
    '{suffix}' AS exchange
FROM marked
GROUP BY symbol_raw, TradeDate
"""


def query_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def finalize_daily(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol_raw"].astype(str) + frame["exchange"]
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
    frame = frame.drop(columns=["symbol_raw"])
    # CH may return NaN shares when usable=0; keep as NA
    frame = frame.sort_values(["TradeDate", "symbol"], kind="stable")
    frame = frame.reset_index(drop=True)
    return frame[list(DAILY_COLUMNS)]


def prepare_sweep_penetration_daily(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in DAILY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError("duplicate symbol/TradeDate")
    for col in (
        "sweep_2plus_share",
        "sweep_notional_share",
        "mean_estimated_levels_penetrated",
        "mean_depth_consumed_ratio",
        "sweep_directional_asymmetry",
    ):
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        if np.isinf(arr).any():
            raise ValueError(f"inf in {col}")
    return frame


PRIMITIVE_FORMULAS: Dict[str, str] = {
    "reference_book": (
        "latest VALID SSL2 snapshot with ExchTime STRICTLY BEFORE trade; "
        "same-timestamp NOT used"
    ),
    "estimated_levels_penetrated": (
        "BUY: count AskPrices<=trade_price; SELL: count BidPrices>=trade_price "
        "on reference book; NA/flag if price_not_on_ladder"
    ),
    "usable": "quality_flag=='ok' (not missing/stale/off-ladder)",
    "sweep_2plus_share": (
        "count(usable & estimated_levels_penetrated>=2) / usable_event_count"
    ),
    "sweep_notional_share": (
        "sum(trade_amount | usable & levels>=2) / sum(trade_amount | usable)"
    ),
    "sweep_directional_asymmetry": "buy_sweep_share - sell_sweep_share",
    "stale_alignment": f"alignment_lag_ms > {STALE_LAG_MS} or < 0",
}
