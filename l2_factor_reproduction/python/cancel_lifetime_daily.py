"""Conservative cancel-lifetime / order-commitment daily primitive (Sprint 15B).

Primary: cancel_age_ms = cancel_time - order_add_time
Universe: continuous auction posted orders (SSE Type=A; SZSE Cat1/2)

Gate cancels with countIf>0 (minIf default is epoch, not NULL).
"""

from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python import liquidity_impact_daily as lid

DATABASE = lid.DATABASE
EXCHANGES = lid.EXCHANGES
CANONICAL_SOURCE = (
    "cmds.LOCAL_SSE_AL_TICK_EXG | cmds.LOCAL_SZSE_AL_TICK_EXG"
)
SCHEMA_VERSION = "cancel_lifetime_daily_v1"
FORMULA_VERSION = "frozen_conservative_cancel_age_v1_from_15a"
SYMBOL_BATCH_SIZE = 150

_SESSION_SQL = (
    "("
    "((toHour(ExchTime) * 60 + toMinute(ExchTime)) >= 570"
    " AND (toHour(ExchTime) * 60 + toMinute(ExchTime)) < 690)"
    " OR ((toHour(ExchTime) * 60 + toMinute(ExchTime)) >= 780"
    " AND (toHour(ExchTime) * 60 + toMinute(ExchTime)) < 900)"
    ")"
)

DAILY_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "TradeDate",
    "source_exchange",
    "eligible_order_count",
    "cancelled_order_count",
    "censored_order_count",
    "full_fill_order_count",
    "partial_fill_then_cancel_count",
    "cancel_age_median_ms",
    "cancel_age_p25_ms",
    "cancel_age_p75_ms",
    "bid_cancel_age_median_ms",
    "ask_cancel_age_median_ms",
    "cancel_age_asymmetry_ms",
    "partial_fill_then_cancel_share",
    "censored_order_share",
    "negative_lifetime_count",
    "invalid_sequence_count",
)


def _dt64(day: str) -> str:
    return f"toDateTime64('{day} 00:00:00', 6, 'Asia/Shanghai')"


def query_sha256(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _in_list(codes: Sequence[str]) -> str:
    return ",".join(f"'{c}'" for c in codes)


def _age_ms() -> str:
    return (
        "toInt64(toUnixTimestamp64Milli(cancel_time))"
        " - toInt64(toUnixTimestamp64Milli(add_time))"
    )


def sse_daily_sql(start: str, end: str, codes: Sequence[str]) -> str:
    table = EXCHANGES["sse"]["tick_table"]
    age = _age_ms()
    return f"""
WITH life AS (
    SELECT
        Symbol,
        Channel,
        if(BSFlag = 'B', BidOrderNo, AskOrderNo) AS order_no,
        BSFlag AS side,
        minIf(ExchTime, Type = 'A') AS add_time,
        argMinIf(toFloat64(Volume), ExchTime, Type = 'A') AS order_size,
        minIf(ExchTime, Type = 'D') AS cancel_time,
        sumIf(toFloat64(Volume), Type = 'D') AS cancel_qty,
        countIf(Type = 'D') AS n_cancel
    FROM {table}
    WHERE Symbol IN ({_in_list(codes)})
      AND ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
      AND {_SESSION_SQL}
      AND Type IN ('A', 'D')
      AND BSFlag IN ('B', 'S')
      AND if(BSFlag = 'B', BidOrderNo, AskOrderNo) > 0
    GROUP BY Symbol, Channel, order_no, side
    HAVING countIf(Type = 'A') > 0
)
SELECT
    Symbol,
    toDate('{start}') AS TradeDate,
    'SSE' AS source_exchange,
    toUInt64(count()) AS eligible_order_count,
    toUInt64(countIf(n_cancel > 0)) AS cancelled_order_count,
    toUInt64(countIf(n_cancel = 0)) AS non_cancel_order_count,
    toUInt64(countIf(
        n_cancel > 0 AND cancel_qty + 1e-6 < order_size AND order_size > 0
    )) AS partial_fill_then_cancel_count,
    quantileExactIf(0.5)({age}, n_cancel > 0) AS cancel_age_median_ms,
    quantileExactIf(0.25)({age}, n_cancel > 0) AS cancel_age_p25_ms,
    quantileExactIf(0.75)({age}, n_cancel > 0) AS cancel_age_p75_ms,
    quantileExactIf(0.5)({age}, n_cancel > 0 AND side = 'B')
        AS bid_cancel_age_median_ms,
    quantileExactIf(0.5)({age}, n_cancel > 0 AND side = 'S')
        AS ask_cancel_age_median_ms,
    toUInt64(countIf(n_cancel > 0 AND {age} < 0)) AS negative_lifetime_count
FROM life
GROUP BY Symbol
"""


def szse_daily_sql(start: str, end: str, codes: Sequence[str]) -> str:
    table = EXCHANGES["szse"]["tick_table"]
    age = _age_ms()
    return f"""
WITH life AS (
    SELECT
        Symbol,
        Channel,
        multiIf(
            Category IN ('1', '2'), SeqNo,
            BidOrderNo > 0, BidOrderNo,
            AskOrderNo
        ) AS order_no,
        multiIf(
            Category = '1', 'B',
            Category = '2', 'S',
            BidOrderNo > 0, 'B',
            'S'
        ) AS side,
        minIf(ExchTime, Category IN ('1', '2')) AS add_time,
        argMinIf(toFloat64(Volume), ExchTime, Category IN ('1', '2')) AS order_size,
        minIf(ExchTime, Category = '4') AS cancel_time,
        sumIf(toFloat64(Volume), Category = '4') AS cancel_qty,
        countIf(Category = '4') AS n_cancel
    FROM {table}
    WHERE Symbol IN ({_in_list(codes)})
      AND ExchTime >= {_dt64(start)} AND ExchTime < {_dt64(end)}
      AND Type = '011'
      AND {_SESSION_SQL}
      AND Category IN ('1', '2', '4')
    GROUP BY Symbol, Channel, order_no, side
    HAVING countIf(Category IN ('1', '2')) > 0
)
SELECT
    Symbol,
    toDate('{start}') AS TradeDate,
    'SZSE' AS source_exchange,
    toUInt64(count()) AS eligible_order_count,
    toUInt64(countIf(n_cancel > 0)) AS cancelled_order_count,
    toUInt64(countIf(n_cancel = 0)) AS non_cancel_order_count,
    toUInt64(countIf(
        n_cancel > 0 AND cancel_qty + 1e-6 < order_size AND order_size > 0
    )) AS partial_fill_then_cancel_count,
    quantileExactIf(0.5)({age}, n_cancel > 0) AS cancel_age_median_ms,
    quantileExactIf(0.25)({age}, n_cancel > 0) AS cancel_age_p25_ms,
    quantileExactIf(0.75)({age}, n_cancel > 0) AS cancel_age_p75_ms,
    quantileExactIf(0.5)({age}, n_cancel > 0 AND side = 'B')
        AS bid_cancel_age_median_ms,
    quantileExactIf(0.5)({age}, n_cancel > 0 AND side = 'S')
        AS ask_cancel_age_median_ms,
    toUInt64(countIf(n_cancel > 0 AND {age} < 0)) AS negative_lifetime_count
FROM life
GROUP BY Symbol
"""


def daily_sql(exchange: str, start: str, end: str, codes: Sequence[str]) -> str:
    if exchange == "sse":
        return sse_daily_sql(start, end, codes)
    if exchange == "szse":
        return szse_daily_sql(start, end, codes)
    raise KeyError(exchange)


def chunked(seq: Sequence[str], n: int) -> List[List[str]]:
    return [list(seq[i : i + n]) for i in range(0, len(seq), n)]


def finalize_daily(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(DAILY_COLUMNS))
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df["symbol"] = df["Symbol"].astype(str)
    sse = df["source_exchange"] == "SSE"
    df.loc[sse, "symbol"] = df.loc[sse, "symbol"].str.zfill(6) + ".SH"
    df.loc[~sse, "symbol"] = df.loc[~sse, "symbol"].str.zfill(6) + ".SZ"

    for col in df.columns:
        if col in ("Symbol", "TradeDate", "source_exchange", "symbol"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["censored_order_count"] = df["non_cancel_order_count"].astype("Int64")
    df["full_fill_order_count"] = 0
    df["invalid_sequence_count"] = df["negative_lifetime_count"]
    df["cancel_age_asymmetry_ms"] = (
        df["bid_cancel_age_median_ms"] - df["ask_cancel_age_median_ms"]
    )
    elig = df["eligible_order_count"].astype(float).replace(0, np.nan)
    df["partial_fill_then_cancel_share"] = (
        df["partial_fill_then_cancel_count"].astype(float) / elig
    )
    df["censored_order_share"] = df["censored_order_count"].astype(float) / elig

    # Drop negative-age pollution from residual epoch edge cases
    neg_mask = df["negative_lifetime_count"].fillna(0) > 0
    # Keep row but ages should already exclude via quantileExactIf; zero-out count after filter
    no_cancel = df["cancelled_order_count"].fillna(0).astype(int) <= 0
    for col in [
        "cancel_age_median_ms",
        "cancel_age_p25_ms",
        "cancel_age_p75_ms",
        "bid_cancel_age_median_ms",
        "ask_cancel_age_median_ms",
        "cancel_age_asymmetry_ms",
    ]:
        df.loc[no_cancel, col] = np.nan
        # if median age absurdly negative, null it
        bad = df[col].notna() & (df[col] < 0)
        df.loc[bad, col] = np.nan

    out = df[list(DAILY_COLUMNS)].copy()
    out = out.drop_duplicates(
        ["symbol", "TradeDate", "source_exchange"], keep="last"
    )
    return out.sort_values(
        ["TradeDate", "source_exchange", "symbol"], kind="stable"
    ).reset_index(drop=True)


def prepare_cancel_lifetime_daily(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in DAILY_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[list(DAILY_COLUMNS)]
