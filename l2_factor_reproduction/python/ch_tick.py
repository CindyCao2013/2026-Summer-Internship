"""ClickHouse Tick 订单规模聚合（SSE/SZSE）——在 CH 内 GROUP BY，Python 只读窄结果。

口径：
- SSE：``Type='T'``，金额 = ifNull(Amount, Price*Volume)
- SZSE：``Type='011'`` 且买卖订单号均有效，金额 = Price*Volume
- 每个交易日仅保留 ``09:30:00 <= ExchTime < 15:00:01``
- 中单：Amount > 4万 且 Amount <= 20万
- 小单：Amount <= 4万
- Symbol 输出 Wind 后缀（.SH / .SZ）
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime]

THRESH_MEDIUM = 40_000.0
THRESH_LARGE = 200_000.0
THRESH_SUPER = 1_000_000.0  # 保留，供扩展大单占比
# Sprint 4 frozen boundaries. 40k preserves the existing mid-order definition;
# 10k/50k/200k/1m define the canonical five-bucket distribution.
ORDER_SIZE_BOUNDARIES = (10_000.0, 40_000.0, 50_000.0, 200_000.0, 1_000_000.0)


def _to_date_str(d: DateLike) -> str:
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


def _bare_codes(symbols: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not symbols:
        return None
    return sorted({str(s).split(".")[0] for s in symbols})


def _get_ch_client():
    import clickhouse_connect
    from COMMON_CONST import DATA_DB_HFDATA

    return clickhouse_connect.get_client(**DATA_DB_HFDATA)


def _sym_filter_sql(bare: Optional[List[str]], exchange_suffix: str) -> str:
    if bare is None:
        return ""
    if exchange_suffix == ".SH":
        codes = [c for c in bare if c.startswith(("5", "6", "9"))]
    else:
        codes = [c for c in bare if c.startswith(("0", "1", "2", "3"))]
    if not codes:
        return "AND 1=0"
    in_list = ", ".join(f"'{c}'" for c in codes)
    return f"AND Symbol IN ({in_list})"


def _a_share_filter_sql(exchange_suffix: str) -> str:
    """Conservative exchange-specific A-share code predicate."""
    if exchange_suffix == ".SH":
        return "AND startsWith(Symbol, '6')"
    return (
        "AND (startsWith(Symbol, '000') OR startsWith(Symbol, '001') "
        "OR startsWith(Symbol, '002') OR startsWith(Symbol, '003') "
        "OR startsWith(Symbol, '300') OR startsWith(Symbol, '301') "
        "OR startsWith(Symbol, '302'))"
    )


def _regular_session_filter_sql(time_col: str = "ExchTime") -> str:
    """Return a per-row regular-session predicate for every date in a range.

    Range endpoint predicates alone only constrain the first and last dates;
    without this condition, opening-auction/pre-open records on intermediate
    dates enter multi-day queries.
    """
    return (
        "AND ("
        f"(toHour({time_col}) = 9 AND toMinute({time_col}) >= 30) "
        f"OR (toHour({time_col}) > 9 AND toHour({time_col}) < 15) "
        f"OR (toHour({time_col}) = 15 AND toMinute({time_col}) = 0 "
        f"AND toSecond({time_col}) = 0)"
        ")"
    )


def fetch_tick_agg_by_date_range(
    start_date: DateLike,
    end_date: DateLike,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """CH 内按 (Symbol, TradeDate) 聚合，返回含 mid/small 两列的宽中间表。

    列：symbol, TradeDate, TotalAmount, MediumAmount, SmallAmount
    """
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)
    bare = _bare_codes(symbols)
    client = _get_ch_client()

    # 与逐日版 classify_order_size 对齐：medium = (40k, 200k]，small = (0, 40k]
    med_cond = f"amt > {THRESH_MEDIUM} AND amt <= {THRESH_LARGE}"
    small_cond = f"amt > 0 AND amt <= {THRESH_MEDIUM}"

    sql_sse = f"""
    SELECT
        concat(Symbol, '.SH') AS symbol,
        toDate(ExchTime) AS TradeDate,
        sum(amt) AS TotalAmount,
        sum(if({med_cond}, amt, 0)) AS MediumAmount,
        sum(if({small_cond}, amt, 0)) AS SmallAmount
    FROM (
        SELECT
            Symbol,
            ExchTime,
            toFloat64(ifNull(Amount, Price * Volume)) AS amt
        FROM cmds.SSE_AL_TICK_EXG
        WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
          AND ExchTime <  toDateTime64('{end} 15:00:01', 6, 'Asia/Shanghai')
          AND toDate(ExchTime) BETWEEN toDate('{start}') AND toDate('{end}')
          {_regular_session_filter_sql()}
          AND Type = 'T'
          AND Price > 0 AND Volume > 0
          {_sym_filter_sql(bare, '.SH')}
    )
    GROUP BY Symbol, TradeDate
    """

    sql_szse = f"""
    SELECT
        concat(Symbol, '.SZ') AS symbol,
        toDate(ExchTime) AS TradeDate,
        sum(amt) AS TotalAmount,
        sum(if({med_cond}, amt, 0)) AS MediumAmount,
        sum(if({small_cond}, amt, 0)) AS SmallAmount
    FROM (
        SELECT
            Symbol,
            ExchTime,
            toFloat64(Price * Volume) AS amt
        FROM cmds.SZSE_AL_TICK_EXG
        WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
          AND ExchTime <  toDateTime64('{end} 15:00:01', 6, 'Asia/Shanghai')
          AND toDate(ExchTime) BETWEEN toDate('{start}') AND toDate('{end}')
          {_regular_session_filter_sql()}
          AND Type = '011'
          AND BidOrderNo > 0 AND AskOrderNo > 0
          AND Price > 0 AND Volume > 0
          {_sym_filter_sql(bare, '.SZ')}
    )
    GROUP BY Symbol, TradeDate
    """

    logger.info(
        "CH tick AGG %s~%s symbols=%s (server-side GROUP BY)",
        start,
        end,
        len(bare) if bare else "ALL",
    )
    try:
        df_sse = client.query_df(sql_sse)
        df_szse = client.query_df(sql_szse)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    frames = [x for x in (df_sse, df_szse) if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame(
            columns=["symbol", "TradeDate", "TotalAmount", "MediumAmount", "SmallAmount"]
        )
    out = pd.concat(frames, ignore_index=True)
    logger.info("CH agg rows=%d (symbol-days)", len(out))
    return out


def fetch_tick_bucketed(
    start_date: DateLike,
    end_date: DateLike,
    boundaries: Sequence[float],
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """CH 内一次查询返回各金额边界的**累计**成交额分桶（用于阈值网格扫描）。

    列：symbol, TradeDate, TotalAmount, cum_<b>（amt <= b 的累计成交额）。
    Medium(L, H] 金额 = cum_H - cum_L，可在 pandas 内拼装任意 (L, H) 组合，
    避免每个阈值组合单独扫一遍 Tick 全表。
    """
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)
    bare = _bare_codes(symbols)
    client = _get_ch_client()
    bounds = sorted({float(b) for b in boundaries})

    def _selects() -> str:
        parts = ["sum(amt) AS TotalAmount"]
        for b in bounds:
            parts.append(f"sum(if(amt > 0 AND amt <= {b}, amt, 0)) AS `cum_{int(b)}`")
        return ",\n        ".join(parts)

    def _sql(table: str, suffix: str, amt_expr: str, extra_where: str) -> str:
        return f"""
        SELECT
            concat(Symbol, '{suffix}') AS symbol,
            toDate(ExchTime) AS TradeDate,
            {_selects()}
        FROM (
            SELECT Symbol, ExchTime, toFloat64({amt_expr}) AS amt
            FROM {table}
            WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
              AND ExchTime <  toDateTime64('{end} 15:00:01', 6, 'Asia/Shanghai')
              AND toDate(ExchTime) BETWEEN toDate('{start}') AND toDate('{end}')
              {_regular_session_filter_sql()}
              {extra_where}
              AND Price > 0 AND Volume > 0
              {_sym_filter_sql(bare, suffix)}
        )
        GROUP BY Symbol, TradeDate
        """

    sql_sse = _sql("cmds.SSE_AL_TICK_EXG", ".SH", "ifNull(Amount, Price * Volume)", "AND Type = 'T'")
    sql_szse = _sql(
        "cmds.SZSE_AL_TICK_EXG",
        ".SZ",
        "Price * Volume",
        "AND Type = '011' AND BidOrderNo > 0 AND AskOrderNo > 0",
    )

    logger.info("CH tick BUCKETED %s~%s bounds=%s", start, end, bounds)
    try:
        df_sse = client.query_df(sql_sse)
        df_szse = client.query_df(sql_szse)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    frames = [x for x in (df_sse, df_szse) if x is not None and not x.empty]
    cols = ["symbol", "TradeDate", "TotalAmount"] + [f"cum_{int(b)}" for b in bounds]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def fetch_order_size_distribution_daily(
    start_date: DateLike,
    end_date: DateLike,
    boundaries: Sequence[float] = ORDER_SIZE_BOUNDARIES,
    symbols: Optional[Sequence[str]] = None,
    stock_only: bool = True,
) -> pd.DataFrame:
    """Build the reusable daily order-size distribution primitive in CH.

    One raw-Tick scan per exchange returns symbol-day cumulative amount buckets,
    cumulative trade counts, and aggressor-side cumulative amounts. The factor
    layer can derive both the canonical five buckets
    ``<=1w, (1w,5w], (5w,20w], (20w,100w], >100w`` and the frozen
    ``mid_order_ratio=(4w,20w]`` without scanning Tick again.
    ``stock_only=True`` applies conservative exchange-code A-share predicates
    before aggregation, excluding funds, bonds, B-shares and other securities.

    Columns
    -------
    symbol, TradeDate, total_amt, trade_cnt, active_buy_amt, active_sell_amt,
    plus for every boundary ``b``:
    ``cum_amt_b``, ``cum_cnt_b``, ``buy_cum_amt_b``, ``sell_cum_amt_b``.
    """
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)
    bare = _bare_codes(symbols)
    bounds = sorted({float(boundary) for boundary in boundaries})
    if not bounds or any(boundary <= 0 for boundary in bounds):
        raise ValueError("order-size boundaries must be non-empty and positive")
    client = _get_ch_client()

    def _selects(buy_cond: str, sell_cond: str) -> str:
        parts = [
            "sum(amt) AS total_amt",
            "count() AS trade_cnt",
            f"sum(if({buy_cond}, amt, 0)) AS active_buy_amt",
            f"sum(if({sell_cond}, amt, 0)) AS active_sell_amt",
        ]
        for boundary in bounds:
            label = int(boundary)
            bucket_cond = f"amt > 0 AND amt <= {boundary}"
            parts.extend(
                [
                    f"sum(if({bucket_cond}, amt, 0)) AS `cum_amt_{label}`",
                    f"countIf({bucket_cond}) AS `cum_cnt_{label}`",
                    (
                        f"sum(if(({buy_cond}) AND {bucket_cond}, amt, 0)) "
                        f"AS `buy_cum_amt_{label}`"
                    ),
                    (
                        f"sum(if(({sell_cond}) AND {bucket_cond}, amt, 0)) "
                        f"AS `sell_cum_amt_{label}`"
                    ),
                ]
            )
        return ",\n            ".join(parts)

    def _sql(
        table: str,
        suffix: str,
        amt_expr: str,
        trade_where: str,
        buy_cond: str,
        sell_cond: str,
        extra_cols: str = "",
    ) -> str:
        return f"""
        SELECT
            concat(Symbol, '{suffix}') AS symbol,
            toDate(ExchTime) AS TradeDate,
            {_selects(buy_cond, sell_cond)}
        FROM (
            SELECT Symbol, ExchTime, BidOrderNo, AskOrderNo{extra_cols},
                   toFloat64({amt_expr}) AS amt
            FROM {table}
            WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
              AND ExchTime <  toDateTime64('{end} 15:00:01', 6, 'Asia/Shanghai')
              AND toDate(ExchTime) BETWEEN toDate('{start}') AND toDate('{end}')
              {_regular_session_filter_sql()}
              AND Price > 0 AND Volume > 0
              {trade_where}
              {_a_share_filter_sql(suffix) if stock_only else ""}
              {_sym_filter_sql(bare, suffix)}
        )
        GROUP BY Symbol, TradeDate
        """

    sql_sse = _sql(
        "cmds.SSE_AL_TICK_EXG",
        ".SH",
        "ifNull(Amount, Price * Volume)",
        "AND Type = 'T'",
        "BSFlag = 'B'",
        "BSFlag = 'S'",
        extra_cols=", BSFlag",
    )
    sql_szse = _sql(
        "cmds.SZSE_AL_TICK_EXG",
        ".SZ",
        "Price * Volume",
        "AND Type = '011' AND BidOrderNo > 0 AND AskOrderNo > 0",
        "BidOrderNo > AskOrderNo",
        "BidOrderNo < AskOrderNo",
    )

    logger.info(
        "CH order-size distribution AGG %s~%s bounds=%s",
        start,
        end,
        bounds,
    )
    try:
        df_sse = client.query_df(sql_sse)
        df_szse = client.query_df(sql_szse)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    cols = [
        "symbol",
        "TradeDate",
        "total_amt",
        "trade_cnt",
        "active_buy_amt",
        "active_sell_amt",
    ]
    for boundary in bounds:
        label = int(boundary)
        cols.extend(
            [
                f"cum_amt_{label}",
                f"cum_cnt_{label}",
                f"buy_cum_amt_{label}",
                f"sell_cum_amt_{label}",
            ]
        )
    frames = [
        frame
        for frame in (df_sse, df_szse)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def fetch_trade_flow_daily(
    start_date: DateLike,
    end_date: DateLike,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """CH 内聚合主买/主卖成交（L2 Primitive Layer: trade_flow_daily）。

    方向口径（已通过逐笔样本验证）：
    - SSE：``Type='T'``，``BSFlag='B'/'S'`` 直接给出方向，'N' 计入 total 但不计方向
    - SZSE：``Type='011'`` 且 ``BidOrderNo>0 AND AskOrderNo>0``（双单号=成交），
      晚到单为发起方：``BidOrderNo > AskOrderNo`` ⇒ 主动买，反之主动卖

    列：symbol, TradeDate, active_buy_amt, active_sell_amt, total_amt,
        active_buy_cnt, active_sell_cnt, trade_cnt
    """
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)
    bare = _bare_codes(symbols)
    client = _get_ch_client()

    def _sql(table: str, suffix: str, amt_expr: str, trade_where: str,
             buy_cond: str, sell_cond: str, extra_cols: str = "") -> str:
        return f"""
        SELECT
            concat(Symbol, '{suffix}') AS symbol,
            toDate(ExchTime) AS TradeDate,
            sum(if({buy_cond}, amt, 0)) AS active_buy_amt,
            sum(if({sell_cond}, amt, 0)) AS active_sell_amt,
            sum(amt) AS total_amt,
            countIf({buy_cond}) AS active_buy_cnt,
            countIf({sell_cond}) AS active_sell_cnt,
            count() AS trade_cnt
        FROM (
            SELECT Symbol, ExchTime, BidOrderNo, AskOrderNo{extra_cols},
                   toFloat64({amt_expr}) AS amt
            FROM {table}
            WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
              AND ExchTime <  toDateTime64('{end} 15:00:01', 6, 'Asia/Shanghai')
              AND toDate(ExchTime) BETWEEN toDate('{start}') AND toDate('{end}')
              {_regular_session_filter_sql()}
              AND Price > 0 AND Volume > 0
              {trade_where}
              {_sym_filter_sql(bare, suffix)}
        )
        GROUP BY Symbol, TradeDate
        """

    sql_sse = _sql(
        "cmds.SSE_AL_TICK_EXG", ".SH", "ifNull(Amount, Price * Volume)",
        "AND Type = 'T'",
        "BSFlag = 'B'", "BSFlag = 'S'",
        extra_cols=", BSFlag",
    )
    sql_szse = _sql(
        "cmds.SZSE_AL_TICK_EXG", ".SZ", "Price * Volume",
        "AND Type = '011' AND BidOrderNo > 0 AND AskOrderNo > 0",
        "BidOrderNo > AskOrderNo", "BidOrderNo < AskOrderNo",
    )

    logger.info("CH trade_flow AGG %s~%s", start, end)
    try:
        df_sse = client.query_df(sql_sse)
        df_szse = client.query_df(sql_szse)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass

    cols = ["symbol", "TradeDate", "active_buy_amt", "active_sell_amt", "total_amt",
            "active_buy_cnt", "active_sell_cnt", "trade_cnt"]
    frames = [x for x in (df_sse, df_szse) if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def aggregate_wide_to_narrow(wide: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """宽中间表 → 单因子窄表。"""
    empty = pd.DataFrame(columns=["symbol", "tradetime", "factorname", "value"])
    if wide is None or wide.empty:
        return empty
    if factor_name not in ("mid_order_ratio", "small_order_ratio"):
        raise ValueError(f"unsupported factor: {factor_name}")

    df = wide.copy()
    total = pd.to_numeric(df["TotalAmount"], errors="coerce")
    if factor_name == "mid_order_ratio":
        part = pd.to_numeric(df["MediumAmount"], errors="coerce")
    else:
        part = pd.to_numeric(df["SmallAmount"], errors="coerce")
    value = part / total.replace(0, float("nan"))

    tradetime = pd.to_datetime(df["TradeDate"]) + pd.Timedelta(hours=9, minutes=30)
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str),
            "tradetime": tradetime,
            "factorname": factor_name,
            "value": value.astype(float),
        }
    )
    return out.dropna(subset=["value"])


def fetch_order_size_narrow(
    start_date: DateLike,
    end_date: DateLike,
    factor_name: str,
    symbols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """一步：CH 聚合 + 转指定因子窄表。"""
    wide = fetch_tick_agg_by_date_range(start_date, end_date, symbols=symbols)
    return aggregate_wide_to_narrow(wide, factor_name)


# ---------- 兼容旧接口（已弃用逐日拉全量） ----------

def fetch_tick_day(*args, **kwargs):  # noqa: ANN001
    raise RuntimeError(
        "fetch_tick_day 已弃用：请使用 fetch_tick_agg_by_date_range / fetch_order_size_narrow（CH 内聚合）"
    )


def aggregate_order_size_factors(*args, **kwargs):  # noqa: ANN001
    raise RuntimeError(
        "aggregate_order_size_factors 已弃用：请使用 aggregate_wide_to_narrow"
    )


def classify_order_size(amount: pd.Series) -> pd.Series:
    """保留供单测：与 CH SQL 分档一致。"""
    a = pd.to_numeric(amount, errors="coerce")
    labels = pd.Series("small", index=a.index, dtype=object)
    labels = labels.mask(a > THRESH_MEDIUM, "medium")
    labels = labels.mask(a > THRESH_LARGE, "large")
    labels = labels.mask(a > THRESH_SUPER, "super_large")
    return labels
