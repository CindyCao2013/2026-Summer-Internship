"""2024-06 sequence loaders for TC-1. Sidecar only; no production mutation.

DDB: Stock_one_minute continuous bars (minute_index 0-239, 15:00 excluded).
CH SSL2: minute-last OBI/spread/depth, is_close_auction=0.
CH tick: optional large-order minute sums; falls back to DDB avg trade size.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CUT_RESULT_ROOT,
    EVENT_Q_DEFAULT,
    EXPECTED_CONTINUOUS_MINUTES,
    RATIO_EPSILON,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.normalize import (
    relative_to_group_median,
    share_ratio,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.time_cuts import (
    mkey_from_minute_index,
)
from l2_factor_reproduction.python.ch_order_book import (
    ORDER_BOOK_TABLES,
    _as_day,
    _symbol_filter_sql,
)
from l2_factor_reproduction.python.ch_tick import (
    _regular_session_filter_sql,
)
from l2_factor_reproduction.python.liquidity_impact_daily import (
    EXCHANGES as LID_EXCHANGES,
    _dt64,
)
from minute_bar_store import filter_a_share, to_wind_code

TC1_START = pd.Timestamp("2024-06-01")
TC1_END = pd.Timestamp("2024-06-30")
CACHE_DIR = CUT_RESULT_ROOT / "tc1_output" / "cache"
CH_TICK_PROBE_DAY = pd.Timestamp("2024-06-03")
CH_TICK_MONTH_BUDGET_SEC = 30 * 60
PROBE_SCALE_DAYS = 19.0
LARGE_ORDER_DEFINITION = "within_stock_day_top20_notional_share"
LARGE_TRADE_Q = 1.0 - EVENT_Q_DEFAULT  # 80th percentile of trade size
SSL2_TABLES_LOCAL = (
    ("LOCAL_SSE_AL_SSL2_EXG", ".SH", "SSE"),
    ("LOCAL_SZSE_AL_SSL2_EXG", ".SZ", "SZSE"),
)


def _week_chunks(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    chunks = []
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=6), end)
        chunks.append((cur, stop))
        cur = stop + pd.Timedelta(days=1)
    return chunks


def _ddb_day(value) -> str:
    return pd.Timestamp(value).normalize().strftime("%Y.%m.%d")


def ddb_minutes_sql(start, end, *, symbols: Optional[Sequence[str]] = None) -> str:
    symbol_clause = ""
    if symbols:
        values = ", ".join('"{}"'.format(to_wind_code(s)) for s in symbols)
        symbol_clause = "\n    and Symbol in ({})".format(values)
    return """
t=loadTable("dfs://QV_Trade_to_MinuteBar","Stock_one_minute")
select
    Symbol,
    Date,
    iif(
      hour(Bartime)<12,
      hour(Bartime)*60+minuteOfHour(Bartime)-570,
      120+hour(Bartime)*60+minuteOfHour(Bartime)-780
    ) as minute_index,
    Close,
    Open,
    High,
    Low,
    Amount,
    Volume,
    Active_buy_amount,
    Active_sell_amount,
    Active_buy_count,
    Active_sell_count,
    Bid_cancel_volume,
    Ask_cancel_volume
  from t
  where Date between {start} : {end}
    and (
      (second(Bartime)>=09:30:00 and second(Bartime)<=11:29:00)
      or (second(Bartime)>=13:00:00 and second(Bartime)<=14:59:00)
    ){symbol_clause}
""".format(start=_ddb_day(start), end=_ddb_day(end), symbol_clause=symbol_clause)


def _normalize_ddb(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    out = frame.rename(columns={"Symbol": "symbol", "Date": "TradeDate"}).copy()
    out["symbol"] = out["symbol"].astype(str).map(to_wind_code)
    out = filter_a_share(out)
    out["TradeDate"] = pd.to_datetime(out["TradeDate"]).dt.normalize()
    out["minute_index"] = pd.to_numeric(out["minute_index"], errors="coerce")
    out = out[out["minute_index"].notna()].copy()
    out["minute_index"] = out["minute_index"].astype(np.int32)
    if (out["minute_index"] >= EXPECTED_CONTINUOUS_MINUTES).any():
        out = out.loc[out["minute_index"] < EXPECTED_CONTINUOUS_MINUTES].copy()
    out["mkey"] = mkey_from_minute_index(out["minute_index"].to_numpy())
    for col in (
        "Close",
        "Open",
        "High",
        "Low",
        "Amount",
        "Volume",
        "Active_buy_amount",
        "Active_sell_amount",
        "Active_buy_count",
        "Active_sell_count",
        "Bid_cancel_volume",
        "Ask_cancel_volume",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    buy = out["Active_buy_amount"].to_numpy(dtype=float)
    sell = out["Active_sell_amount"].to_numpy(dtype=float)
    out["net_active_flow"] = buy - sell
    out["cancel_imbalance"] = (
        out["Bid_cancel_volume"].to_numpy(dtype=float)
        - out["Ask_cancel_volume"].to_numpy(dtype=float)
    )
    cnt = (
        out["Active_buy_count"].to_numpy(dtype=float)
        + out["Active_sell_count"].to_numpy(dtype=float)
    )
    amt = out["Amount"].to_numpy(dtype=float)
    avg = np.full(amt.shape, np.nan)
    ok = np.isfinite(amt) & np.isfinite(cnt) & (cnt > 0)
    avg[ok] = amt[ok] / cnt[ok]
    out["avg_trade_size"] = avg
    out["amount"] = out["Amount"]
    return out


def load_ddb_minutes_202406(
    *,
    start=TC1_START,
    end=TC1_END,
    symbols: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, float]:
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ddb_minutes_202406.parquet"
    t0 = time.time()
    if cache_path.exists() and not force:
        frame = pd.read_parquet(cache_path)
        return frame, time.time() - t0
    from core.ddb.connection import get_ddb_session, is_shared_session

    session = get_ddb_session(reuse=True)
    frames = []
    try:
        for chunk_start, chunk_end in _week_chunks(start, end):
            raw = pd.DataFrame(
                session.run(ddb_minutes_sql(chunk_start, chunk_end, symbols=symbols))
            )
            if raw is not None and len(raw):
                frames.append(_normalize_ddb(raw))
    finally:
        if not is_shared_session(session):
            session.close()
    if not frames:
        frame = pd.DataFrame()
    else:
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates(["symbol", "TradeDate", "minute_index"])
        frame = _attach_minute_return(frame)
        frame.to_parquet(cache_path, index=False)
    return frame, time.time() - t0


def _attach_minute_return(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        frame["minute_return"] = []
        return frame
    out = frame.sort_values(["symbol", "TradeDate", "mkey"]).copy()
    g = out.groupby(["symbol", "TradeDate"], sort=False)
    prev_close = g["Close"].shift(1)
    prev_mkey = g["mkey"].shift(1)
    close = out["Close"].to_numpy(dtype=float)
    prev = prev_close.to_numpy(dtype=float)
    mkey = out["mkey"].to_numpy(dtype=np.int32)
    pmkey = prev_mkey.to_numpy(dtype=float)
    consec = np.isfinite(pmkey) & (mkey == pmkey.astype(np.int32) + 1)
    same_session = (
        ((mkey <= 689) & (pmkey <= 689))
        | ((mkey >= 780) & (pmkey >= 780))
    )
    ok = consec & same_session & (close > 0) & (prev > 0)
    ret = np.full(close.shape, np.nan)
    ret[ok] = np.log(close[ok] / prev[ok])
    out["minute_return"] = ret
    return out


def ch_ssl2_minute_sql(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start,
    end,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """Thin minute-last OBI/spread/depth. Continuous auction only (no 15:00)."""
    start_day = _as_day(start)
    end_exclusive = _as_day(end) + pd.Timedelta(days=1)
    start_text = start_day.strftime("%Y-%m-%d")
    end_text = end_exclusive.strftime("%Y-%m-%d")
    symbol_filter = _symbol_filter_sql(exchange, symbols)
    type_filter = "AND Type = '010'" if exchange == "SZSE" else ""
    return """
SELECT
  concat(Symbol, '{suffix}') AS symbol,
  toDate(ExchTime) AS TradeDate,
  multiIf(
    toHour(ExchTime) < 12,
      toHour(ExchTime) * 60 + toMinute(ExchTime) - 570,
    toHour(ExchTime) < 15,
      120 + toHour(ExchTime) * 60 + toMinute(ExchTime) - 780,
    240
  ) AS minute_index,
  argMax(
    if(
      bid_depth_5 + ask_depth_5 > 0,
      (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5),
      CAST(NULL AS Nullable(Float64))
    ),
    ExchTime
  ) AS obi_5,
  argMax(
    if(mid_price > 0, (ask1 - bid1) / mid_price, CAST(NULL AS Nullable(Float64))),
    ExchTime
  ) AS relative_spread,
  argMax(bid_depth_5 + ask_depth_5, ExchTime) AS total_depth_l5
FROM (
  SELECT
    Symbol,
    ExchTime,
    toFloat64(BidPrices[1]) AS bid1,
    toFloat64(AskPrices[1]) AS ask1,
    (toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2. AS mid_price,
    arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.), arraySlice(BidVolumes, 1, 5)))
      AS bid_depth_5,
    arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.), arraySlice(AskVolumes, 1, 5)))
      AS ask_depth_5
  FROM cmds.`{table}`
  WHERE ExchTime >= toDateTime64('{start_text} 00:00:00', 6, 'Asia/Shanghai')
    AND ExchTime < toDateTime64('{end_text} 00:00:00', 6, 'Asia/Shanghai')
    AND {symbol_filter}
    {type_filter}
    AND (
      (toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30)
      OR toHour(ExchTime) IN (10, 13, 14)
      OR (toHour(ExchTime) = 11 AND toMinute(ExchTime) < 30)
    )
    AND length(BidPrices) >= 5
    AND length(AskPrices) >= 5
    AND length(BidVolumes) >= 5
    AND length(AskVolumes) >= 5
    AND toFloat64(BidPrices[1]) > 0
    AND toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])
    AND toUInt8(toHour(ExchTime) = 15) = 0
)
GROUP BY symbol, TradeDate, minute_index
HAVING minute_index >= 0 AND minute_index < 240
""".format(
        suffix=exchange_suffix,
        table=table,
        start_text=start_text,
        end_text=end_text,
        symbol_filter=symbol_filter,
        type_filter=type_filter,
    )


def load_ch_ssl2_202406(
    *,
    start=TC1_START,
    end=TC1_END,
    symbols: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, float]:
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ch_ssl2_202406.parquet"
    t0 = time.time()
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path), time.time() - t0
    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client

    client = connect_hf_client()
    frames = []
    table_sets = (list(ORDER_BOOK_TABLES), list(SSL2_TABLES_LOCAL))
    try:
        for tables in table_sets:
            frames = []
            for table, suffix, exchange in tables:
                for chunk_start, chunk_end in _week_chunks(start, end):
                    sql = ch_ssl2_minute_sql(
                        table=table,
                        exchange_suffix=suffix,
                        exchange=exchange,
                        start=chunk_start,
                        end=chunk_end,
                        symbols=symbols,
                    )
                    part = client.query_df(sql)
                    if part is not None and len(part):
                        frames.append(part)
            n_sym = 0
            if frames:
                n_sym = int(pd.concat([f[["symbol"]] for f in frames], ignore_index=True)["symbol"].nunique())
            if n_sym >= 3000:
                break
            if tables is table_sets[-1]:
                break
    finally:
        client.close()
    if not frames:
        frame = pd.DataFrame(
            columns=["symbol", "TradeDate", "minute_index", "mkey", "obi_5", "relative_spread", "total_depth_l5"]
        )
    else:
        frame = pd.concat(frames, ignore_index=True)
        frame["symbol"] = frame["symbol"].astype(str)
        frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
        frame["minute_index"] = pd.to_numeric(frame["minute_index"], errors="coerce").astype("int32")
        if (frame["minute_index"] >= EXPECTED_CONTINUOUS_MINUTES).any():
            raise RuntimeError("CH SSL2 loader leaked close-auction minute_index>=240")
        frame["mkey"] = mkey_from_minute_index(frame["minute_index"].to_numpy())
        for col in ("obi_5", "relative_spread", "total_depth_l5"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.to_parquet(cache_path, index=False)
    return frame, time.time() - t0


def _tick_minute_sql(exchange: str, start: str, end: str) -> str:
    """Relative large prints: trades above the stock-day 80th percentile of size.

    Absolute 200k CNY is not used. Minute SHARE is applied after the query.
    """
    cfg = LID_EXCHANGES[exchange]
    tick_table = cfg["tick_table"].replace("LOCAL_", "")
    return """
WITH
trades AS (
    SELECT Symbol, ExchTime, {amount_expr} AS amt
    FROM {table}
    WHERE ExchTime >= {start_dt} AND ExchTime < {end_dt}
      AND {trade_filter}
      AND {symbol_filter}
      {session_filter}
),
day_q AS (
    SELECT
        Symbol,
        toDate(ExchTime) AS TradeDate,
        quantileTDigest({q})(amt) AS q80
    FROM trades
    GROUP BY Symbol, TradeDate
)
SELECT
    concat(t.Symbol, '{suffix}') AS symbol,
    toDate(t.ExchTime) AS TradeDate,
    multiIf(
      toHour(t.ExchTime) < 12,
        toHour(t.ExchTime) * 60 + toMinute(t.ExchTime) - 570,
      toHour(t.ExchTime) < 15,
        120 + toHour(t.ExchTime) * 60 + toMinute(t.ExchTime) - 780,
      240
    ) AS minute_index,
    sumIf(t.amt, t.amt >= q.q80) AS large_order_amount,
    sum(t.amt) AS tick_amount
FROM trades t
INNER JOIN day_q q
    ON t.Symbol = q.Symbol AND toDate(t.ExchTime) = q.TradeDate
GROUP BY symbol, TradeDate, minute_index
HAVING minute_index >= 0 AND minute_index < 240
""".format(
        suffix=cfg["suffix"],
        q=float(LARGE_TRADE_Q),
        amount_expr=cfg["amount_expr"],
        table=tick_table,
        start_dt=_dt64(start),
        end_dt=_dt64(end),
        trade_filter=cfg["trade_filter"],
        symbol_filter=cfg["symbol_filter"],
        session_filter=_regular_session_filter_sql("ExchTime"),
    )


def probe_ch_tick_cost_sec() -> float:
    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client

    day = CH_TICK_PROBE_DAY.strftime("%Y-%m-%d")
    nxt = (CH_TICK_PROBE_DAY + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    client = connect_hf_client()
    t0 = time.time()
    try:
        client.query_df(_tick_minute_sql("sse", day, nxt))
    finally:
        client.close()
    return time.time() - t0


def _write_json(path: Path, payload: dict) -> None:
    serial = {}
    for key, val in payload.items():
        if isinstance(val, (np.floating, float)):
            serial[key] = float(val)
        elif isinstance(val, (np.integer, int)):
            serial[key] = int(val)
        elif isinstance(val, (np.bool_, bool)):
            serial[key] = bool(val)
        else:
            serial[key] = val
    path.write_text(json.dumps(serial, indent=2) + "\n")


def load_ch_tick_large_order_202406(
    *,
    start=TC1_START,
    end=TC1_END,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    use_proxy_if_slow: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ch_tick_large_order_202406.parquet"
    meta_path = cache_dir / "ch_tick_large_order_202406.meta.json"
    meta = {
        "source_used": "ch_tick_distributed",
        "requires_ch_tick": False,
        "proxy_source": "",
        "probe_sec": float("nan"),
        "load_sec": float("nan"),
        "skipped_reason": "",
        "large_order_definition": LARGE_ORDER_DEFINITION,
    }
    t0 = time.time()
    if not force and meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved.get("large_order_definition") == LARGE_ORDER_DEFINITION:
            meta.update(saved)
            if cache_path.exists() and not meta.get("requires_ch_tick"):
                frame = pd.read_parquet(cache_path)
                meta["load_sec"] = time.time() - t0
                meta["source_used"] = "ch_tick_distributed_cache"
                return frame, meta
            if meta.get("requires_ch_tick"):
                meta["load_sec"] = time.time() - t0
                return pd.DataFrame(), meta
    probe = float("nan")
    if use_proxy_if_slow:
        probe = probe_ch_tick_cost_sec()
        meta["probe_sec"] = probe
        estimated = probe * PROBE_SCALE_DAYS * 2.0
        if estimated > CH_TICK_MONTH_BUDGET_SEC:
            meta["source_used"] = "ddb_avg_trade_size_proxy"
            meta["requires_ch_tick"] = True
            meta["proxy_source"] = "DDB_AvgTradeSize"
            meta["skipped_reason"] = (
                "CH tick 1-day SSE probe {:.1f}s; month estimate {:.0f}s > 30min"
                .format(probe, estimated)
            )
            meta["load_sec"] = time.time() - t0
            _write_json(meta_path, meta)
            return pd.DataFrame(), meta
    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client

    client = connect_hf_client()
    frames = []
    try:
        for exchange in ("sse", "szse"):
            for chunk_start, chunk_end in _week_chunks(start, end):
                start_s = pd.Timestamp(chunk_start).strftime("%Y-%m-%d")
                end_excl = (pd.Timestamp(chunk_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                part = client.query_df(_tick_minute_sql(exchange, start_s, end_excl))
                if part is not None and len(part):
                    frames.append(part)
    finally:
        client.close()
    if not frames:
        meta["source_used"] = "ddb_avg_trade_size_proxy"
        meta["requires_ch_tick"] = True
        meta["proxy_source"] = "DDB_AvgTradeSize"
        meta["skipped_reason"] = "CH tick returned empty"
        meta["load_sec"] = time.time() - t0
        _write_json(meta_path, meta)
        return pd.DataFrame(), meta
    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
    frame["minute_index"] = pd.to_numeric(frame["minute_index"], errors="coerce").astype("int32")
    if (frame["minute_index"] >= EXPECTED_CONTINUOUS_MINUTES).any():
        raise RuntimeError("CH tick loader leaked close-auction minute_index>=240")
    frame["mkey"] = mkey_from_minute_index(frame["minute_index"].to_numpy())
    frame["large_order_amount"] = pd.to_numeric(frame["large_order_amount"], errors="coerce")
    frame["tick_amount"] = pd.to_numeric(frame["tick_amount"], errors="coerce")
    frame.to_parquet(cache_path, index=False)
    meta["large_order_definition"] = LARGE_ORDER_DEFINITION
    meta["load_sec"] = time.time() - t0
    _write_json(meta_path, meta)
    return frame, meta


def build_tc1_panel(
    ddb: pd.DataFrame,
    ssl2: pd.DataFrame,
    tick: pd.DataFrame,
    tick_meta: dict,
) -> pd.DataFrame:
    """Left-join CH fields onto the DDB continuous-minute spine."""
    if ddb.empty:
        raise RuntimeError("DDB minute panel is empty; cannot run TC-1")
    keys = ["symbol", "TradeDate", "minute_index"]
    panel = ddb.copy()
    if ssl2 is not None and len(ssl2):
        book_cols = keys + [c for c in ("obi_5", "relative_spread", "total_depth_l5") if c in ssl2.columns]
        panel = panel.merge(ssl2[book_cols].drop_duplicates(keys), on=keys, how="left")
    else:
        panel["obi_5"] = np.nan
        panel["relative_spread"] = np.nan
        panel["total_depth_l5"] = np.nan
    if tick is not None and len(tick) and not tick_meta.get("requires_ch_tick"):
        tick_cols = keys + [
            c for c in ("large_order_amount", "tick_amount") if c in tick.columns
        ]
        panel = panel.merge(tick[tick_cols].drop_duplicates(keys), on=keys, how="left")
        notional = pd.to_numeric(panel["large_order_amount"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(panel["tick_amount"], errors="coerce").fillna(0.0)
        share = share_ratio(notional, total)
        panel["large_order_notional"] = notional
        panel["tick_amount"] = total
        panel["large_order_amount"] = share.fillna(0.0)
        tick_meta["large_order_definition"] = LARGE_ORDER_DEFINITION
    else:
        panel["large_order_notional"] = panel["avg_trade_size"]
        panel["tick_amount"] = panel["amount"]
        panel["large_order_amount"] = relative_to_group_median(panel, "avg_trade_size")
        tick_meta["large_order_definition"] = "ddb_avg_trade_size_over_stock_day_median"
    if int((panel["mkey"] == 900).sum()) > 0:
        raise RuntimeError("combined panel contains 15:00 auction bars")
    if int((panel["minute_index"] >= 240).sum()) > 0:
        raise RuntimeError("combined panel contains minute_index>=240")
    return panel


def month_windows(start, end) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    out = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    while cur <= end:
        nxt = cur + pd.offsets.MonthEnd(0)
        lo = max(cur, start)
        hi = min(pd.Timestamp(nxt), end)
        out.append((lo, hi))
        cur = (nxt + pd.Timedelta(days=1)).normalize()
    return out


def ch_ssl2_minute_sql_tc2a(
    *,
    table: str,
    exchange_suffix: str,
    exchange: str,
    start,
    end,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """SSL2 minute-last OBI/spread/depth/microprice. Continuous only."""
    start_day = _as_day(start)
    end_exclusive = _as_day(end) + pd.Timedelta(days=1)
    start_text = start_day.strftime("%Y-%m-%d")
    end_text = end_exclusive.strftime("%Y-%m-%d")
    symbol_filter = _symbol_filter_sql(exchange, symbols)
    type_filter = "AND Type = '010'" if exchange == "SZSE" else ""
    return """
SELECT
  concat(Symbol, '{suffix}') AS symbol,
  toDate(ExchTime) AS TradeDate,
  multiIf(
    toHour(ExchTime) < 12,
      toHour(ExchTime) * 60 + toMinute(ExchTime) - 570,
    toHour(ExchTime) < 15,
      120 + toHour(ExchTime) * 60 + toMinute(ExchTime) - 780,
    240
  ) AS minute_index,
  argMax(
    if(
      bid_depth_5 + ask_depth_5 > 0,
      (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5),
      CAST(NULL AS Nullable(Float64))
    ),
    ExchTime
  ) AS obi_5,
  argMax(
    if(mid_price > 0, (ask1 - bid1) / mid_price, CAST(NULL AS Nullable(Float64))),
    ExchTime
  ) AS relative_spread,
  argMax(bid_depth_5 + ask_depth_5, ExchTime) AS total_depth_l5,
  argMax(microprice_deviation, ExchTime) AS microprice_deviation
FROM (
  SELECT
    Symbol,
    ExchTime,
    toFloat64(BidPrices[1]) AS bid1,
    toFloat64(AskPrices[1]) AS ask1,
    (toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2. AS mid_price,
    arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.), arraySlice(BidVolumes, 1, 5)))
      AS bid_depth_5,
    arraySum(arrayMap(x -> ifNull(toFloat64(x), 0.), arraySlice(AskVolumes, 1, 5)))
      AS ask_depth_5,
    toFloat64(BidVolumes[1]) AS bid_vol_1,
    toFloat64(AskVolumes[1]) AS ask_vol_1,
    if(
      toFloat64(BidVolumes[1]) + toFloat64(AskVolumes[1]) > 0
      AND (toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2. > 0,
      (
        (
          toFloat64(AskPrices[1]) * toFloat64(BidVolumes[1])
          + toFloat64(BidPrices[1]) * toFloat64(AskVolumes[1])
        ) / (toFloat64(BidVolumes[1]) + toFloat64(AskVolumes[1]))
        - (toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2.
      ) / ((toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2.),
      CAST(NULL AS Nullable(Float64))
    ) AS microprice_deviation
  FROM cmds.`{table}`
  WHERE ExchTime >= toDateTime64('{start_text} 00:00:00', 6, 'Asia/Shanghai')
    AND ExchTime < toDateTime64('{end_text} 00:00:00', 6, 'Asia/Shanghai')
    AND {symbol_filter}
    {type_filter}
    AND (
      (toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30)
      OR toHour(ExchTime) IN (10, 13, 14)
      OR (toHour(ExchTime) = 11 AND toMinute(ExchTime) < 30)
    )
    AND length(BidPrices) >= 5
    AND length(AskPrices) >= 5
    AND length(BidVolumes) >= 5
    AND length(AskVolumes) >= 5
    AND toFloat64(BidPrices[1]) > 0
    AND toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])
    AND toUInt8(toHour(ExchTime) = 15) = 0
)
GROUP BY symbol, TradeDate, minute_index
HAVING minute_index >= 0 AND minute_index < 240
""".format(
        suffix=exchange_suffix,
        table=table,
        start_text=start_text,
        end_text=end_text,
        symbol_filter=symbol_filter,
        type_filter=type_filter,
    )


def _tick_minute_sql_signed(exchange: str, start: str, end: str) -> str:
    cfg = LID_EXCHANGES[exchange]
    tick_table = cfg["tick_table"].replace("LOCAL_", "")
    return """
WITH
trades AS (
    SELECT
        Symbol,
        ExchTime,
        {amount_expr} AS amt,
        ({buy_cond}) AS is_buy,
        ({sell_cond}) AS is_sell
    FROM {table}
    WHERE ExchTime >= {start_dt} AND ExchTime < {end_dt}
      AND {trade_filter}
      AND {symbol_filter}
      {session_filter}
),
day_q AS (
    SELECT
        Symbol,
        toDate(ExchTime) AS TradeDate,
        quantileTDigest({q})(amt) AS q80
    FROM trades
    GROUP BY Symbol, TradeDate
)
SELECT
    concat(t.Symbol, '{suffix}') AS symbol,
    toDate(t.ExchTime) AS TradeDate,
    multiIf(
      toHour(t.ExchTime) < 12,
        toHour(t.ExchTime) * 60 + toMinute(t.ExchTime) - 570,
      toHour(t.ExchTime) < 15,
        120 + toHour(t.ExchTime) * 60 + toMinute(t.ExchTime) - 780,
      240
    ) AS minute_index,
    sumIf(t.amt, t.amt >= q.q80) AS large_order_amount,
    sumIf(t.amt, t.amt >= q.q80 AND t.is_buy) AS large_buy_amount,
    sumIf(t.amt, t.amt >= q.q80 AND t.is_sell) AS large_sell_amount,
    sum(t.amt) AS tick_amount
FROM trades t
INNER JOIN day_q q
    ON t.Symbol = q.Symbol AND toDate(t.ExchTime) = q.TradeDate
GROUP BY symbol, TradeDate, minute_index
HAVING minute_index >= 0 AND minute_index < 240
""".format(
        suffix=cfg["suffix"],
        q=float(LARGE_TRADE_Q),
        amount_expr=cfg["amount_expr"],
        buy_cond=cfg["buy_cond"],
        sell_cond=cfg["sell_cond"],
        table=tick_table,
        start_dt=_dt64(start),
        end_dt=_dt64(end),
        trade_filter=cfg["trade_filter"],
        symbol_filter=cfg["symbol_filter"],
        session_filter=_regular_session_filter_sql("ExchTime"),
    )


def _month_tag(start) -> str:
    return pd.Timestamp(start).strftime("%Y%m")


def load_ddb_minutes_month(
    start,
    end,
    *,
    cache_dir: Path,
    force: bool = False,
) -> Tuple[pd.DataFrame, float]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ddb_minutes_{}.parquet".format(_month_tag(start))
    t0 = time.time()
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path), time.time() - t0
    from core.ddb.connection import get_ddb_session, is_shared_session

    session = get_ddb_session(reuse=True)
    frames = []
    try:
        for chunk_start, chunk_end in _week_chunks(start, end):
            raw = pd.DataFrame(
                session.run(ddb_minutes_sql(chunk_start, chunk_end))
            )
            if raw is not None and len(raw):
                frames.append(_normalize_ddb(raw))
    finally:
        if not is_shared_session(session):
            session.close()
    if not frames:
        frame = pd.DataFrame()
    else:
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates(["symbol", "TradeDate", "minute_index"])
        frame = _attach_minute_return(frame)
        frame.to_parquet(cache_path, index=False)
    return frame, time.time() - t0


def load_ch_ssl2_month(
    start,
    end,
    *,
    cache_dir: Path,
    force: bool = False,
) -> Tuple[pd.DataFrame, float]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ch_ssl2_{}.parquet".format(_month_tag(start))
    t0 = time.time()
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path), time.time() - t0
    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client

    client = connect_hf_client()
    frames = []
    table_sets = (list(ORDER_BOOK_TABLES), list(SSL2_TABLES_LOCAL))
    try:
        for tables in table_sets:
            frames = []
            for table, suffix, exchange in tables:
                for chunk_start, chunk_end in _week_chunks(start, end):
                    sql = ch_ssl2_minute_sql_tc2a(
                        table=table,
                        exchange_suffix=suffix,
                        exchange=exchange,
                        start=chunk_start,
                        end=chunk_end,
                    )
                    part = client.query_df(sql)
                    if part is not None and len(part):
                        frames.append(part)
            n_sym = 0
            if frames:
                n_sym = int(
                    pd.concat([f[["symbol"]] for f in frames], ignore_index=True)["symbol"].nunique()
                )
            if n_sym >= 3000:
                break
            if tables is table_sets[-1]:
                break
    finally:
        client.close()
    if not frames:
        frame = pd.DataFrame(
            columns=[
                "symbol",
                "TradeDate",
                "minute_index",
                "mkey",
                "obi_5",
                "relative_spread",
                "total_depth_l5",
                "microprice_deviation",
            ]
        )
    else:
        frame = pd.concat(frames, ignore_index=True)
        frame["symbol"] = frame["symbol"].astype(str)
        frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
        frame["minute_index"] = pd.to_numeric(frame["minute_index"], errors="coerce").astype("int32")
        if (frame["minute_index"] >= EXPECTED_CONTINUOUS_MINUTES).any():
            raise RuntimeError("CH SSL2 loader leaked close-auction minute_index>=240")
        frame["mkey"] = mkey_from_minute_index(frame["minute_index"].to_numpy())
        for col in ("obi_5", "relative_spread", "total_depth_l5", "microprice_deviation"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.to_parquet(cache_path, index=False)
    return frame, time.time() - t0


def load_ch_tick_signed_month(
    start,
    end,
    *,
    cache_dir: Path,
    force: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _month_tag(start)
    cache_path = cache_dir / "ch_tick_signed_{}.parquet".format(tag)
    meta_path = cache_dir / "ch_tick_signed_{}.meta.json".format(tag)
    meta = {
        "source_used": "ch_tick_distributed",
        "requires_ch_tick": False,
        "proxy_source": "",
        "load_sec": float("nan"),
        "skipped_reason": "",
        "large_order_definition": LARGE_ORDER_DEFINITION,
        "zero_activity_filled_with_zero": False,
    }
    t0 = time.time()
    if not force and meta_path.exists() and cache_path.exists():
        saved = json.loads(meta_path.read_text())
        if saved.get("large_order_definition") == LARGE_ORDER_DEFINITION:
            meta.update(saved)
            frame = pd.read_parquet(cache_path)
            meta["load_sec"] = time.time() - t0
            meta["source_used"] = "ch_tick_distributed_cache"
            return frame, meta
    from research.l2_alpha.clickhouse_ssl2 import connect_hf_client

    client = connect_hf_client()
    frames = []
    try:
        for exchange in ("sse", "szse"):
            for chunk_start, chunk_end in _week_chunks(start, end):
                start_s = pd.Timestamp(chunk_start).strftime("%Y-%m-%d")
                end_excl = (pd.Timestamp(chunk_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                part = client.query_df(_tick_minute_sql_signed(exchange, start_s, end_excl))
                if part is not None and len(part):
                    frames.append(part)
    finally:
        client.close()
    if not frames:
        meta["source_used"] = "empty"
        meta["requires_ch_tick"] = True
        meta["skipped_reason"] = "CH tick returned empty"
        meta["load_sec"] = time.time() - t0
        _write_json(meta_path, meta)
        return pd.DataFrame(), meta
    frame = pd.concat(frames, ignore_index=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
    frame["minute_index"] = pd.to_numeric(frame["minute_index"], errors="coerce").astype("int32")
    if (frame["minute_index"] >= EXPECTED_CONTINUOUS_MINUTES).any():
        raise RuntimeError("CH tick loader leaked close-auction minute_index>=240")
    frame["mkey"] = mkey_from_minute_index(frame["minute_index"].to_numpy())
    for col in ("large_order_amount", "large_buy_amount", "large_sell_amount", "tick_amount"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.to_parquet(cache_path, index=False)
    meta["load_sec"] = time.time() - t0
    _write_json(meta_path, meta)
    return frame, meta


def build_tc2a_panel(
    ddb: pd.DataFrame,
    ssl2: pd.DataFrame,
    tick: pd.DataFrame,
    tick_meta: dict,
) -> pd.DataFrame:
    """TC-2A panel. Large-order zero denominators stay missing (not filled with 0)."""
    if ddb.empty:
        raise RuntimeError("DDB minute panel is empty; cannot run TC-2A")
    keys = ["symbol", "TradeDate", "minute_index"]
    panel = ddb.copy()
    book_keep = [
        c for c in ("obi_5", "relative_spread", "total_depth_l5", "microprice_deviation")
        if ssl2 is not None and c in ssl2.columns
    ]
    if ssl2 is not None and len(ssl2) and book_keep:
        panel = panel.merge(ssl2[keys + book_keep].drop_duplicates(keys), on=keys, how="left")
    else:
        for c in ("obi_5", "relative_spread", "total_depth_l5", "microprice_deviation"):
            if c not in panel.columns:
                panel[c] = np.nan
    if tick is not None and len(tick) and not tick_meta.get("requires_ch_tick"):
        tick_cols = keys + [
            c
            for c in (
                "large_order_amount",
                "large_buy_amount",
                "large_sell_amount",
                "tick_amount",
            )
            if c in tick.columns
        ]
        panel = panel.merge(tick[tick_cols].drop_duplicates(keys), on=keys, how="left")
        buy = pd.to_numeric(panel.get("large_buy_amount"), errors="coerce")
        sell = pd.to_numeric(panel.get("large_sell_amount"), errors="coerce")
        total = pd.to_numeric(panel.get("tick_amount"), errors="coerce")
        notional = pd.to_numeric(panel.get("large_order_amount"), errors="coerce")
        panel["large_order_notional"] = notional
        panel["tick_amount"] = total
        panel["large_order_amount"] = share_ratio(notional, total)
        activity = (buy.fillna(0.0) + sell.fillna(0.0)) > RATIO_EPSILON
        denom_ok = total.notna() & (total.abs() > RATIO_EPSILON)
        pressure = (buy.fillna(0.0) - sell.fillna(0.0)) / total
        panel["large_order_pressure"] = pressure.where(denom_ok)
        panel["large_order_activity"] = activity & denom_ok
        tick_meta["large_order_definition"] = LARGE_ORDER_DEFINITION
        tick_meta["zero_activity_filled_with_zero"] = False
    else:
        panel["large_order_notional"] = np.nan
        panel["tick_amount"] = panel.get("amount")
        panel["large_order_amount"] = np.nan
        panel["large_order_pressure"] = np.nan
        panel["large_order_activity"] = False
        tick_meta["zero_activity_filled_with_zero"] = False
    if int((panel["mkey"] == 900).sum()) > 0:
        raise RuntimeError("combined panel contains 15:00 auction bars")
    if int((panel["minute_index"] >= 240).sum()) > 0:
        raise RuntimeError("combined panel contains minute_index>=240")
    return panel
