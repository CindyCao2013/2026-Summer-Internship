#!/usr/bin/env python
"""Sprint 6B Phase 1 — Cancellation / Order Lifecycle daily primitive.

Implements the frozen contract
(primitives/cancel_lifecycle_daily/implementation_contract.json):

- SSE: direct aggregation of valid D records (Type='D', Price>0, Volume>0,
  BSFlag in B/S, non-null order key). D rows carry residual qty and the
  original order price, so no A/T/D lifecycle join is built in production.
- SZSE: single-scan order-key grouping (TradeDate x Symbol x Channel x
  normalized_order_no; order rows Category 1/2 -> SeqNo, cancel rows
  Category 4 -> BidOrderNo/AskOrderNo whichever > 0) using
  argMaxIf/sumIf/countIf. No distributed full self join.
- Zero-price (market order) cancels are extracted as a small side table and
  filled with the backward ASOF nearest same-symbol trade price; fill
  failure -> zero value + invalid_cancel_count.
- Trading window: continuous auction 09:30-11:29 + 13:00-14:59 (auction
  periods excluded), matching the contract.

Count semantics: cancel_event_count is the official count field;
cancelled_unique_order_count is retained for robustness (empirically equal
on the 2024-06-28 full-market audit).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    _as_day,
    _symbol_filter_sql,
)

PRIMITIVE_VERSION = "cancel_lifecycle_v1_phase0_amended"

SSE_TABLE = "cmds.SSE_AL_TICK_EXG"
SZSE_TABLE = "cmds.SZSE_AL_TICK_EXG"

PRIMITIVE_COLUMNS = [
    "buy_cancel_value", "sell_cancel_value",
    "buy_cancel_qty", "sell_cancel_qty",
    "buy_cancel_event_count", "sell_cancel_event_count",
    "buy_cancelled_unique_order_count",
    "sell_cancelled_unique_order_count",
    "total_trade_value", "total_trade_qty", "total_trade_count",
    "join_coverage", "zero_price_cancel_count",
    "market_order_price_fill_count", "invalid_cancel_count",
]

# continuous auction window in minutes-of-day: [09:30, 11:30) + [13:00, 15:00)
_WINDOW_SQL = (
    "((toHour(ExchTime) * 60 + toMinute(ExchTime)) >= 570"
    " AND (toHour(ExchTime) * 60 + toMinute(ExchTime)) < 690)"
    " OR ((toHour(ExchTime) * 60 + toMinute(ExchTime)) >= 780"
    " AND (toHour(ExchTime) * 60 + toMinute(ExchTime)) < 900)"
)


def _day_pred(day: pd.Timestamp) -> str:
    return (
        f"ExchTime >= toDateTime64('{day:%Y-%m-%d} 00:00:00', 6,"
        " 'Asia/Shanghai') AND ExchTime < toDateTime64"
        f"('{day:%Y-%m-%d} 00:00:00', 6, 'Asia/Shanghai') + toIntervalDay(1)"
    )


# ---------------------------------------------------------------------------
# SSE: direct valid-D aggregation
# ---------------------------------------------------------------------------

_VALID_D = (
    "Type = 'D' AND Price > 0 AND Volume > 0"
    " AND BSFlag IN ('B', 'S')"
    " AND if(BSFlag = 'B', BidOrderNo, AskOrderNo) IS NOT NULL"
)


def sse_daily_sql(day: pd.Timestamp,
                  symbols: Optional[Sequence[str]] = None) -> str:
    """One GROUP BY Symbol pass over Type D/T rows for one trading day."""
    where = (
        f"{_day_pred(day)} AND ({_WINDOW_SQL})"
        f" AND {_symbol_filter_sql('SSE', symbols)}"
        " AND Type IN ('D', 'T')"
    )
    return f"""
SELECT
    Symbol,
    sumIf(toFloat64(Price) * toFloat64(Volume),
          {_VALID_D} AND BSFlag = 'B') AS buy_cancel_value,
    sumIf(toFloat64(Price) * toFloat64(Volume),
          {_VALID_D} AND BSFlag = 'S') AS sell_cancel_value,
    sumIf(toFloat64(Volume),
          {_VALID_D} AND BSFlag = 'B') AS buy_cancel_qty,
    sumIf(toFloat64(Volume),
          {_VALID_D} AND BSFlag = 'S') AS sell_cancel_qty,
    countIf({_VALID_D} AND BSFlag = 'B') AS buy_cancel_event_count,
    countIf({_VALID_D} AND BSFlag = 'S') AS sell_cancel_event_count,
    uniqExactIf((Channel, BidOrderNo),
                {_VALID_D} AND BSFlag = 'B')
        AS buy_cancelled_unique_order_count,
    uniqExactIf((Channel, AskOrderNo),
                {_VALID_D} AND BSFlag = 'S')
        AS sell_cancelled_unique_order_count,
    sumIf(toFloat64(Price) * toFloat64(Volume), Type = 'T')
        AS total_trade_value,
    sumIf(toFloat64(Volume), Type = 'T') AS total_trade_qty,
    countIf(Type = 'T') AS total_trade_count,
    countIf(Type = 'D' AND NOT ({_VALID_D})) AS invalid_cancel_count
FROM {SSE_TABLE}
WHERE {where}
GROUP BY Symbol
"""


# ---------------------------------------------------------------------------
# SZSE: single-scan order-key grouping + small zero-price side table
# ---------------------------------------------------------------------------

def _szse_grouped_scan(day: pd.Timestamp,
                       symbols: Optional[Sequence[str]] = None) -> str:
    """Inner single scan grouped by the canonical order key."""
    # cancel events are counted in the continuous-auction window only, but
    # order rows must cover the full day: orders placed in the opening
    # auction (or lunch break) can be cancelled in-window, and dropping them
    # would break the order-key match (join_coverage < 1).
    where = (
        f"{_day_pred(day)}"
        f" AND {_symbol_filter_sql('SZSE', symbols)}"
        " AND Type = '011'"
        f" AND (Category IN ('1', '2')"
        f"      OR (Category = '4' AND ({_WINDOW_SQL})))"
    )
    return f"""
SELECT
    Symbol,
    Channel,
    multiIf(Category IN ('1', '2'), SeqNo,
            BidOrderNo > 0, BidOrderNo, AskOrderNo) AS norm_no,
    maxIf(toFloat64(Price), Category IN ('1', '2')) AS order_price,
    countIf(Category IN ('1', '2')) AS n_order_rows,
    sumIf(toFloat64(Volume), Category = '4') AS cancel_qty,
    countIf(Category = '4') AS cancel_events,
    minIf(ExchTime, Category = '4') AS first_cancel_time,
    maxIf(if(BidOrderNo > 0, 1, 0), Category = '4') AS cancel_is_buy
FROM {SZSE_TABLE}
WHERE {where}
GROUP BY Symbol, Channel, norm_no
HAVING cancel_events > 0
"""


def szse_rollup_sql(day: pd.Timestamp,
                    symbols: Optional[Sequence[str]] = None) -> str:
    """Symbol-day rollup of the grouped scan (limit-price cancels valued at
    the original order price; zero-price cancels counted separately and
    valued later via the ASOF side table)."""
    inner = _szse_grouped_scan(day, symbols)
    return f"""
SELECT
    Symbol,
    sumIf(cancel_qty * order_price,
          order_price > 0 AND cancel_is_buy = 1) AS buy_cancel_value_limit,
    sumIf(cancel_qty * order_price,
          order_price > 0 AND cancel_is_buy = 0) AS sell_cancel_value_limit,
    sumIf(cancel_qty, cancel_is_buy = 1) AS buy_cancel_qty,
    sumIf(cancel_qty, cancel_is_buy = 0) AS sell_cancel_qty,
    sumIf(cancel_events, cancel_is_buy = 1) AS buy_cancel_event_count,
    sumIf(cancel_events, cancel_is_buy = 0) AS sell_cancel_event_count,
    countIf(cancel_is_buy = 1) AS buy_cancelled_unique_order_count,
    countIf(cancel_is_buy = 0) AS sell_cancelled_unique_order_count,
    sumIf(cancel_events, order_price = 0) AS zero_price_cancel_count,
    countIf(n_order_rows = 0) AS unmatched_cancel_order_count,
    sum(cancel_events) AS total_cancel_events,
    sumIf(cancel_events, n_order_rows > 0) AS matched_cancel_events
FROM ({inner})
GROUP BY Symbol
"""


def szse_zero_price_side_sql(day: pd.Timestamp,
                             symbols: Optional[Sequence[str]] = None) -> str:
    """Small side table: zero-price (market order) cancel groups only, with
    backward ASOF fill against same-symbol trade prices (Category F)."""
    inner = _szse_grouped_scan(day, symbols)
    trade_where = (
        f"{_day_pred(day)} AND ({_WINDOW_SQL})"
        f" AND {_symbol_filter_sql('SZSE', symbols)}"
        " AND Type = '011' AND Category = 'F' AND Price > 0"
    )
    return f"""
SELECT
    z.Symbol AS Symbol,
    z.cancel_is_buy AS cancel_is_buy,
    z.cancel_qty AS cancel_qty,
    t.Price AS fill_price
FROM (
    SELECT Symbol, norm_no, cancel_is_buy, cancel_qty, first_cancel_time
    FROM ({inner})
    WHERE order_price = 0
) z
ASOF LEFT JOIN (
    SELECT Symbol, ExchTime, toFloat64(Price) AS Price
    FROM {SZSE_TABLE}
    WHERE {trade_where}
) t ON z.Symbol = t.Symbol AND z.first_cancel_time >= t.ExchTime
"""


def szse_trade_totals_sql(day: pd.Timestamp,
                          symbols: Optional[Sequence[str]] = None) -> str:
    where = (
        f"{_day_pred(day)} AND ({_WINDOW_SQL})"
        f" AND {_symbol_filter_sql('SZSE', symbols)}"
        " AND Type = '011' AND Category = 'F'"
    )
    return f"""
SELECT
    Symbol,
    sum(toFloat64(Price) * toFloat64(Volume)) AS total_trade_value,
    sum(toFloat64(Volume)) AS total_trade_qty,
    count() AS total_trade_count
FROM {SZSE_TABLE}
WHERE {where}
GROUP BY Symbol
"""


# ---------------------------------------------------------------------------
# pandas assembly (pure functions, golden-testable)
# ---------------------------------------------------------------------------

def szse_assemble(rollup: pd.DataFrame,
                  fills: pd.DataFrame,
                  totals: pd.DataFrame) -> pd.DataFrame:
    """Combine the SQL rollup, the ASOF price fills and trade totals into
    the symbol-day primitive."""
    frame = rollup.copy()
    for column in [
        "buy_cancel_value_limit", "sell_cancel_value_limit",
        "buy_cancel_qty", "sell_cancel_qty",
        "buy_cancel_event_count", "sell_cancel_event_count",
        "buy_cancelled_unique_order_count",
        "sell_cancelled_unique_order_count",
        "zero_price_cancel_count", "unmatched_cancel_order_count",
        "total_cancel_events", "matched_cancel_events",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    fill_stats = fill_rollup(fills)
    frame = frame.merge(fill_stats, on="Symbol", how="left")
    for column in ["buy_mkt_cancel_value", "sell_mkt_cancel_value",
                   "market_order_price_fill_count", "fill_failure_count"]:
        frame[column] = frame[column].fillna(0.0)

    frame["buy_cancel_value"] = (
        frame["buy_cancel_value_limit"] + frame["buy_mkt_cancel_value"]
    )
    frame["sell_cancel_value"] = (
        frame["sell_cancel_value_limit"] + frame["sell_mkt_cancel_value"]
    )
    frame = frame.merge(totals, on="Symbol", how="left")
    # Cancels with no continuous-auction trades → trade totals are 0, not NA.
    # (intensity candidates then correctly become NaN via /0 guards.)
    for column in ["total_trade_value", "total_trade_qty", "total_trade_count"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["join_coverage"] = np.where(
        frame["total_cancel_events"] > 0,
        frame["matched_cancel_events"] / frame["total_cancel_events"],
        np.nan,
    )
    frame["invalid_cancel_count"] = (
        frame["unmatched_cancel_order_count"] + frame["fill_failure_count"]
    )
    keep = ["Symbol"] + [
        "buy_cancel_value", "sell_cancel_value",
        "buy_cancel_qty", "sell_cancel_qty",
        "buy_cancel_event_count", "sell_cancel_event_count",
        "buy_cancelled_unique_order_count",
        "sell_cancelled_unique_order_count",
        "total_trade_value", "total_trade_qty", "total_trade_count",
        "join_coverage", "zero_price_cancel_count",
        "market_order_price_fill_count", "invalid_cancel_count",
    ]
    return frame[keep]


def fill_rollup(fills: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ASOF-filled zero-price cancels to symbol-day value by side.

    Fill failure (no prior trade) contributes zero value and is counted in
    fill_failure_count per the frozen contract.
    """
    columns = ["Symbol", "buy_mkt_cancel_value", "sell_mkt_cancel_value",
               "market_order_price_fill_count", "fill_failure_count"]
    if fills is None or len(fills) == 0:
        return pd.DataFrame(columns=columns)
    work = fills.copy()
    work["fill_price"] = pd.to_numeric(work["fill_price"], errors="coerce")
    work["cancel_qty"] = pd.to_numeric(work["cancel_qty"], errors="coerce")
    work["filled_value"] = work["cancel_qty"] * work["fill_price"].fillna(0)
    work["filled_ok"] = work["fill_price"].notna()
    grouped = work.groupby("Symbol")
    out = pd.DataFrame({
        "buy_mkt_cancel_value": grouped.apply(
            lambda g: g.loc[g["cancel_is_buy"] == 1, "filled_value"].sum()
        ),
        "sell_mkt_cancel_value": grouped.apply(
            lambda g: g.loc[g["cancel_is_buy"] == 0, "filled_value"].sum()
        ),
        "market_order_price_fill_count": grouped["filled_ok"].sum(),
        "fill_failure_count": grouped["filled_ok"].size()
        - grouped["filled_ok"].sum(),
    }).reset_index()
    return out[columns]


def sse_assemble(frame: pd.DataFrame) -> pd.DataFrame:
    """Add constant SSE columns required by the contract schema."""
    out = frame.copy()
    out["join_coverage"] = 1.0
    out["zero_price_cancel_count"] = 0
    out["market_order_price_fill_count"] = 0
    return out


# ---------------------------------------------------------------------------
# daily fetch (ClickHouse)
# ---------------------------------------------------------------------------

def fetch_cancel_daily(
    client,
    day,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Symbol-day cancel-lifecycle primitive for one trading day, both
    exchanges combined. `symbol` carries the .SH/.SZ suffix."""
    day_ts = _as_day(day)

    sse = client.query_df(sse_daily_sql(day_ts, symbols))
    sse_part = sse_assemble(sse) if len(sse) else sse
    if len(sse_part):
        sse_part["symbol"] = sse_part["Symbol"] + ".SH"

    rollup = client.query_df(szse_rollup_sql(day_ts, symbols))
    if len(rollup):
        fills = client.query_df(szse_zero_price_side_sql(day_ts, symbols))
        totals = client.query_df(szse_trade_totals_sql(day_ts, symbols))
        szse_part = szse_assemble(rollup, fills, totals)
        szse_part["symbol"] = szse_part["Symbol"] + ".SZ"
    else:
        szse_part = rollup

    parts = [p for p in (sse_part, szse_part) if len(p)]
    if not parts:
        return pd.DataFrame(columns=["symbol", "TradeDate"]
                            + PRIMITIVE_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    out["TradeDate"] = day_ts
    return out[["symbol", "TradeDate"] + PRIMITIVE_COLUMNS]


# ---------------------------------------------------------------------------
# frozen candidates (registry: cancel_lifecycle_daily/candidate_registry_v1)
# ---------------------------------------------------------------------------

def build_candidates(primitive: pd.DataFrame) -> pd.DataFrame:
    """The 7 frozen v1 candidates from the symbol-day primitive. Shock
    candidates need history and are NOT built here (they are applied on the
    multi-day panel with shift(1).rolling(20), excluding the current day)."""
    frame = primitive.copy()
    for column in PRIMITIVE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype("float64")
    total_cancel_value = frame["buy_cancel_value"] + frame["sell_cancel_value"]
    total_cancel_events = (
        frame["buy_cancel_event_count"] + frame["sell_cancel_event_count"]
    )
    out = pd.DataFrame({
        "symbol": frame["symbol"],
        "TradeDate": frame["TradeDate"],
    })
    out["cancel_value_pressure"] = (
        (frame["buy_cancel_value"] - frame["sell_cancel_value"])
        / total_cancel_value.where(total_cancel_value > 0)
    )
    out["cancel_count_pressure"] = (
        (frame["buy_cancel_event_count"] - frame["sell_cancel_event_count"])
        / total_cancel_events.where(total_cancel_events > 0)
    )
    out["cancel_value_intensity"] = (
        total_cancel_value / frame["total_trade_value"].where(
            frame["total_trade_value"] > 0)
    )
    out["cancel_qty_intensity"] = (
        (frame["buy_cancel_qty"] + frame["sell_cancel_qty"])
        / frame["total_trade_qty"].where(frame["total_trade_qty"] > 0)
    )
    avg_cancel_value = total_cancel_value / total_cancel_events.where(
        total_cancel_events > 0)
    avg_trade_value = frame["total_trade_value"] / frame[
        "total_trade_count"].where(frame["total_trade_count"] > 0)
    out["relative_cancel_order_size"] = avg_cancel_value / avg_trade_value
    return out


def shock_20d(series: pd.Series) -> pd.Series:
    """Frozen shock standardization: history window excludes the current
    day (inclusive/exclusive variants are not both kept)."""
    mean_prev = series.shift(1).rolling(20).mean()
    std_prev = series.shift(1).rolling(20).std(ddof=0)
    return (series - mean_prev) / std_prev.where(std_prev > 0)
