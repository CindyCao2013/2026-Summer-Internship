"""Company DB connectors for F² Agent Lite.

OHLCV / tradability: DolphinDB Wind EOD.
Valuation / ROE / northbound / market risk: DolphinDB Wind.
News title + sentiment: Datayes (通联).
Minute factors: ClickHouse cmds KLIN (1MIN).
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Ensure project root is importable when running as a side project.
_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from COMMON_CONST import DATA_DB_CONN, DATA_DB_DATAYES, DATA_DB_HFDATA  # noqa: E402

# Confirmed live tables (probed 2026-08)
DDB_EOD = "dfs://WIND.ASHAREEODPRICES"
DDB_DERIV = "dfs://WIND.ASHAREEODDERIVATIVEINDICATOR"
DDB_TTMHIS = "dfs://WIND.ASHARETTMHIS"
DDB_NORTH = "dfs://WIND.SHSCCHANNELHOLDINGS"
DDB_INDEX = "dfs://WIND.AINDEXEODPRICES"
MKT_INDEX_CODE = "000300.SH"  # CSI300; China IVIX (000188.*) ends 2018 — use realized vol proxy

# ClickHouse KLIN (probed 2026-08-12)
CH_SSE_KLIN = "SSE_AL_KLIN_EXG"
CH_SZSE_KLIN = "SZSE_AL_KLIN_CMD"


def _to_datetime(value) -> dt.datetime:
    ts = pd.Timestamp(value)
    return dt.datetime(ts.year, ts.month, ts.day)


def _ddb_date_literal(value) -> str:
    return _to_datetime(value).strftime("%Y.%m.%d")


def bare_ticker(windcode: str) -> str:
    return str(windcode).split(".")[0]


def connect_ddb():
    import dolphindb as ddb

    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    return session


def connect_datayes():
    import pymysql

    return pymysql.connect(**DATA_DB_DATAYES)


def connect_clickhouse(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    secure: bool = False,
):
    """ClickHouse HTTP client (clickhouse-connect). Defaults = DATA_DB_HFDATA."""
    import clickhouse_connect

    cfg = dict(DATA_DB_HFDATA)
    if host:
        cfg["host"] = host
    if port is not None:
        cfg["port"] = int(port)
    if username:
        cfg["username"] = username
    if password is not None and password != "":
        cfg["password"] = password
    if database:
        cfg["database"] = database
    if secure:
        cfg["secure"] = True
    return clickhouse_connect.get_client(**cfg)


def get_clickhouse_client(config=None):
    """Convenience wrapper reading optional overrides from Config."""
    if config is None:
        return connect_clickhouse()
    return connect_clickhouse(
        host=getattr(config, "clickhouse_host", None),
        port=getattr(config, "clickhouse_port", None),
        username=getattr(config, "clickhouse_user", None),
        password=getattr(config, "clickhouse_password", None),
        database=getattr(config, "clickhouse_database", None) or "cmds",
        secure=bool(getattr(config, "clickhouse_secure", False)),
    )


def resolve_party_id(symbol: str) -> int:
    """Map WindCode -> Datayes PARTY_ID via md_security."""
    ticker = bare_ticker(symbol)
    exch = "XSHG" if str(symbol).endswith(".SH") else "XSHE"
    sql = """
        SELECT PARTY_ID
        FROM md_security
        WHERE TICKER_SYMBOL = %s
          AND EXCHANGE_CD = %s
          AND ASSET_CLASS = 'E'
        ORDER BY LIST_STATUS_CD DESC
        LIMIT 1
    """
    with connect_datayes() as conn:
        df = pd.read_sql(sql, conn, params=[ticker, exch])
    if df.empty or pd.isna(df.iloc[0]["PARTY_ID"]):
        raise RuntimeError(f"Cannot resolve Datayes PARTY_ID for {symbol}")
    return int(df.iloc[0]["PARTY_ID"])


def get_ohlcv(symbol: str, start, end) -> pd.DataFrame:
    """Daily OHLCV for one WindCode from DolphinDB Wind EOD."""
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        script = f"""
        select TRADE_DT as date,
               S_DQ_OPEN as open,
               S_DQ_HIGH as high,
               S_DQ_LOW as low,
               S_DQ_CLOSE as close,
               S_DQ_VOLUME as volume,
               S_DQ_AMOUNT as amount,
               S_DQ_PRECLOSE as preclose,
               S_DQ_LIMIT as up_limit,
               S_DQ_STOPPING as down_limit,
               S_DQ_TRADESTATUS as trade_status
        from loadTable("dfs://WIND.ASHAREEODPRICES", "data")
        where S_INFO_WINDCODE = "{symbol}"
          and TRADE_DT >= {start_s}
          and TRADE_DT <= {end_s}
        order by TRADE_DT
        """
        df = session.run(script)
    finally:
        session.close()

    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "preclose",
                "up_limit",
                "down_limit",
                "trade_status",
            ]
        )

    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for col in ["open", "high", "low", "close", "volume", "amount", "preclose", "up_limit", "down_limit"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").reset_index(drop=True)


def compute_technical_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI / MACD / SMA / Bollinger from an OHLCV frame."""
    import talib

    if ohlcv.empty:
        return pd.DataFrame(
            columns=["date", "rsi", "macd", "macd_signal", "macd_hist", "sma", "bb_upper", "bb_lower"]
        )

    close = ohlcv["close"].astype(float).values
    rsi = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    sma = talib.SMA(close, timeperiod=20)
    bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    return pd.DataFrame(
        {
            "date": ohlcv["date"].values,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "sma": sma,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
        }
    )


def compute_tradability_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """not-limit and not-suspended flags aligned with Factor_Dev_Lib semantics."""
    if ohlcv.empty:
        return pd.DataFrame(columns=["date", "not_limit", "not_suspended", "tradable"])

    not_limit = (
        (ohlcv["close"] < ohlcv["up_limit"]) & (ohlcv["close"] > ohlcv["down_limit"])
    ).astype(float)
    status = ohlcv["trade_status"].astype(str)
    not_suspended = (~status.isin(["停牌", "", "nan", "None"])).astype(float)
    not_limit = not_limit.where(not_limit > 0, np.nan)
    not_suspended = not_suspended.where(not_suspended > 0, np.nan)
    tradable = (not_limit.fillna(0) * not_suspended.fillna(0)).replace(0, np.nan)

    return pd.DataFrame(
        {
            "date": ohlcv["date"].values,
            "not_limit": not_limit.values,
            "not_suspended": not_suspended.values,
            "tradable": tradable.values,
        }
    )


def get_technical_indicators(symbol: str, start, end) -> pd.DataFrame:
    """Compute RSI / MACD / SMA / Bollinger from OHLCV via talib."""
    return compute_technical_from_ohlcv(get_ohlcv(symbol, start, end))


def get_tradability(symbol: str, start, end) -> pd.DataFrame:
    """Reuse EOD fields: not-limit and not-suspended flags (1=tradable)."""
    return compute_tradability_from_ohlcv(get_ohlcv(symbol, start, end))


def get_news_sentiment(
    symbol: str,
    start,
    end,
    *,
    party_id: Optional[int] = None,
    max_titles: int = 3,
    fetch_titles: bool = False,
) -> pd.DataFrame:
    """Daily mean SENTIMENT_SCORE (+ optional titles) from Datayes.

    Sentiment: news_company_score via RELATED_COMPANY_ID = PARTY_ID (fast).
    Titles: news_content_flash join is slow; disabled by default for multi-name runs.
    """
    ticker = bare_ticker(symbol)
    start_ts = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_ts = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if party_id is None:
        party_id = resolve_party_id(symbol)

    sent_sql = """
        SELECT DATE(EFFECTIVE_TIME) AS trade_date,
               AVG(SENTIMENT_SCORE) AS sentiment_score,
               AVG(SENTIMENT) AS sentiment_label,
               COUNT(*) AS news_count
        FROM news_company_score
        WHERE RELATED_COMPANY_ID = %s
          AND EFFECTIVE_TIME >= %s
          AND EFFECTIVE_TIME < %s
        GROUP BY DATE(EFFECTIVE_TIME)
        ORDER BY trade_date
    """
    with connect_datayes() as conn:
        sent = pd.read_sql(sent_sql, conn, params=[party_id, start_ts, end_ts])
        titles = pd.DataFrame()
        if fetch_titles:
            title_sql = """
                SELECT DATE(s.EFFECTIVE_TIME) AS trade_date,
                       s.EFFECTIVE_TIME AS effective_time,
                       s.RELATED_SCORE AS related_score,
                       f.NEWS_TITLE AS news_title
                FROM news_security_score s
                INNER JOIN news_content_flash f
                  ON s.NEWS_ID = f.NEWS_ID
                WHERE s.TICKER_SYMBOL = %s
                  AND s.EFFECTIVE_TIME >= %s
                  AND s.EFFECTIVE_TIME < %s
                  AND f.NEWS_TITLE IS NOT NULL
                  AND f.NEWS_TITLE <> ''
                ORDER BY s.EFFECTIVE_TIME
            """
            titles = pd.read_sql(title_sql, conn, params=[ticker, start_ts, end_ts])

    if not sent.empty:
        sent["trade_date"] = pd.to_datetime(sent["trade_date"]).dt.normalize()
        sent["sentiment_score"] = pd.to_numeric(sent["sentiment_score"], errors="coerce")
        # Datayes SENTIMENT_SCORE is typically in [0,1]; map to [-1,1] for agent
        sent["sentiment_score"] = sent["sentiment_score"] * 2.0 - 1.0

    news_summary_map: Dict[pd.Timestamp, str] = {}
    if not titles.empty:
        titles["trade_date"] = pd.to_datetime(titles["trade_date"]).dt.normalize()
        titles["related_score"] = pd.to_numeric(titles["related_score"], errors="coerce").fillna(0.0)
        titles = titles.drop_duplicates(subset=["trade_date", "news_title"], keep="first")
        for day, grp in titles.groupby("trade_date"):
            top = grp.sort_values(["related_score", "effective_time"], ascending=[False, False]).head(
                max_titles
            )
            news_summary_map[pd.Timestamp(day)] = " || ".join(top["news_title"].astype(str).tolist())

    if sent.empty:
        # Still emit title-only days if any
        days = sorted(news_summary_map.keys())
        if not days:
            return pd.DataFrame(
                columns=["date", "news_summary", "sentiment_score", "news_count"]
            )
        return pd.DataFrame(
            {
                "date": days,
                "news_summary": [news_summary_map[d] for d in days],
                "sentiment_score": 0.0,
                "news_count": 0,
            }
        )

    sent = sent.rename(columns={"trade_date": "date"})
    sent["news_summary"] = sent["date"].map(lambda d: news_summary_map.get(pd.Timestamp(d), ""))
    # Attach title-only days missing from sentiment aggregate
    extra_days = [d for d in news_summary_map if d not in set(sent["date"])]
    if extra_days:
        extra = pd.DataFrame(
            {
                "date": extra_days,
                "sentiment_score": 0.0,
                "sentiment_label": np.nan,
                "news_count": 0,
                "news_summary": [news_summary_map[d] for d in extra_days],
            }
        )
        sent = pd.concat([sent, extra], ignore_index=True)

    return (
        sent[["date", "news_summary", "sentiment_score", "news_count"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def get_news_titles(symbol: str, start, end) -> pd.DataFrame:
    """Per-title rows from Datayes ``news_security_score`` ⨝ ``news_content_flash``.

    Columns: ``date``, ``effective_time``, ``related_score``, ``title``.
    Used by FinBERT scoring (one inference per title, then daily mean).
    """
    ticker = bare_ticker(symbol)
    start_ts = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_ts = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    title_sql = """
        SELECT DATE(s.EFFECTIVE_TIME) AS trade_date,
               s.EFFECTIVE_TIME AS effective_time,
               s.RELATED_SCORE AS related_score,
               f.NEWS_TITLE AS news_title
        FROM news_security_score s
        INNER JOIN news_content_flash f
          ON s.NEWS_ID = f.NEWS_ID
        WHERE s.TICKER_SYMBOL = %s
          AND s.EFFECTIVE_TIME >= %s
          AND s.EFFECTIVE_TIME < %s
          AND f.NEWS_TITLE IS NOT NULL
          AND f.NEWS_TITLE <> ''
        ORDER BY s.EFFECTIVE_TIME
    """
    with connect_datayes() as conn:
        titles = pd.read_sql(title_sql, conn, params=[ticker, start_ts, end_ts])
    if titles.empty:
        return pd.DataFrame(columns=["date", "effective_time", "related_score", "title"])
    titles["date"] = pd.to_datetime(titles["trade_date"]).dt.normalize()
    titles["effective_time"] = pd.to_datetime(titles["effective_time"])
    titles["related_score"] = pd.to_numeric(titles["related_score"], errors="coerce").fillna(0.0)
    titles["title"] = titles["news_title"].astype(str)
    titles = titles.drop_duplicates(subset=["date", "title"], keep="first")
    return (
        titles[["date", "effective_time", "related_score", "title"]]
        .sort_values(["date", "effective_time"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Alpha feature loaders (north / fundamentals / advanced / market risk)
# ---------------------------------------------------------------------------


def get_northbound(symbol: str, start, end) -> pd.DataFrame:
    """陆股通持仓占比及日变化 — Wind SHSCCHANNELHOLDINGS."""
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        script = f"""
        select TRADE_DT as date,
               S_RATIO as north_share_ratio,
               S_QUANTITY as north_share_qty
        from loadTable("{DDB_NORTH}", "data")
        where S_INFO_WINDCODE = "{symbol}"
          and TRADE_DT >= {start_s}
          and TRADE_DT <= {end_s}
        order by TRADE_DT
        """
        df = session.run(script)
    finally:
        session.close()

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "north_share_ratio", "north_share_chg"])

    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["north_share_ratio"] = pd.to_numeric(out["north_share_ratio"], errors="coerce")
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out["north_share_chg"] = out["north_share_ratio"].diff()
    return out[["date", "north_share_ratio", "north_share_chg"]].reset_index(drop=True)


def get_valuation(symbol: str, start, end) -> pd.DataFrame:
    """日频估值：EP_TTM / BP / 换手 / log市值 — ASHAREEODDERIVATIVEINDICATOR."""
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        script = f"""
        select TRADE_DT as date,
               S_VAL_PE_TTM as pe_ttm,
               S_VAL_PB_NEW as pb,
               S_DQ_TURN as turnover,
               S_VAL_MV as mkt_cap
        from loadTable("{DDB_DERIV}", "data")
        where S_INFO_WINDCODE = "{symbol}"
          and TRADE_DT >= {start_s}
          and TRADE_DT <= {end_s}
        order by TRADE_DT
        """
        df = session.run(script)
    finally:
        session.close()

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "ep_ttm", "bp", "turnover", "log_mktcap"])

    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for col in ["pe_ttm", "pb", "turnover", "mkt_cap"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ep_ttm"] = np.where(out["pe_ttm"].abs() > 1e-8, 1.0 / out["pe_ttm"], np.nan)
    out["bp"] = np.where(out["pb"].abs() > 1e-8, 1.0 / out["pb"], np.nan)
    out["log_mktcap"] = np.log(out["mkt_cap"].where(out["mkt_cap"] > 0))
    return out[["date", "ep_ttm", "bp", "turnover", "log_mktcap"]].sort_values("date").reset_index(drop=True)


def get_fundamentals_pit(symbol: str, start, end) -> pd.DataFrame:
    """PIT 基本面：ROE_TTM + 营收同比增速，按 ANN_DT 对齐后向前填充到日频。"""
    # Pull extra history so YoY match and early ffill work
    hist_start = pd.Timestamp(start) - pd.Timedelta(days=800)
    start_s = _ddb_date_literal(hist_start)
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        script = f"""
        select ANN_DT as ann_date,
               REPORT_PERIOD as report_period,
               S_FA_ROE_TTM as roe,
               TOT_OPER_REV_TTM as revenue_ttm
        from loadTable("{DDB_TTMHIS}", "data")
        where S_INFO_WINDCODE = "{symbol}"
          and STATEMENT_TYPE = "合并报表"
          and ANN_DT >= {start_s}
          and ANN_DT <= {end_s}
        order by ANN_DT, REPORT_PERIOD
        """
        df = session.run(script)
    finally:
        session.close()

    empty = pd.DataFrame(columns=["date", "roe", "revenue_growth_yoy"])
    if df is None or len(df) == 0:
        return empty

    fin = pd.DataFrame(df)
    fin["ann_date"] = pd.to_datetime(fin["ann_date"]).dt.normalize()
    fin["report_period"] = pd.to_datetime(fin["report_period"]).dt.normalize()
    fin["roe"] = pd.to_numeric(fin["roe"], errors="coerce")
    fin["revenue_ttm"] = pd.to_numeric(fin["revenue_ttm"], errors="coerce")
    fin = fin.dropna(subset=["ann_date", "report_period"])
    fin = fin.sort_values(["ann_date", "report_period"]).drop_duplicates(
        subset=["ann_date", "report_period"], keep="last"
    )
    # YoY vs same report month/day one year earlier
    fin["rp_key"] = fin["report_period"].dt.strftime("%m-%d")
    fin["rp_year"] = fin["report_period"].dt.year
    prev = fin[["rp_key", "rp_year", "revenue_ttm", "ann_date"]].rename(
        columns={"rp_year": "prev_year", "revenue_ttm": "revenue_prev", "ann_date": "ann_prev"}
    )
    fin["prev_year"] = fin["rp_year"] - 1
    fin = fin.merge(prev, on=["rp_key", "prev_year"], how="left")
    # Only use prior-year row that was already announced (no look-ahead)
    ok = fin["ann_prev"].notna() & (fin["ann_prev"] <= fin["ann_date"])
    fin["revenue_growth_yoy"] = np.where(
        ok & fin["revenue_prev"].abs() > 1e-8,
        fin["revenue_ttm"] / fin["revenue_prev"] - 1.0,
        np.nan,
    )
    # Available next calendar day after announcement (conservative PIT)
    fin["available_date"] = fin["ann_date"] + pd.Timedelta(days=1)
    fin = fin.sort_values("available_date").drop_duplicates("available_date", keep="last")

    cal_start = pd.Timestamp(start).normalize()
    cal_end = pd.Timestamp(end).normalize()
    days = pd.date_range(cal_start, cal_end, freq="D")
    daily = pd.DataFrame({"date": days})
    daily = pd.merge_asof(
        daily.sort_values("date"),
        fin[["available_date", "roe", "revenue_growth_yoy"]].sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    return daily[["date", "roe", "revenue_growth_yoy"]].reset_index(drop=True)


def get_market_risk(start, end, index_code: str = MKT_INDEX_CODE) -> pd.DataFrame:
    """市场情绪/风险：指数实现波动率 + 全市场涨跌停占比 + 截面波动。

    China IVIX (000188.SH/CSI) 仅覆盖到 2018，测试期用 CSI300 20日波动率代替。
    """
    start_s = _ddb_date_literal(pd.Timestamp(start) - pd.Timedelta(days=60))
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        idx = session.run(
            f"""
            select TRADE_DT as date, S_DQ_CLOSE as close, S_DQ_PCTCHANGE as pct
            from loadTable("{DDB_INDEX}", "data")
            where S_INFO_WINDCODE = "{index_code}"
              and TRADE_DT >= {start_s}
              and TRADE_DT <= {end_s}
            order by TRADE_DT
            """
        )
        try:
            lim = session.run(
                f"""
                select TRADE_DT,
                       sum(iif(S_DQ_CLOSE >= S_DQ_LIMIT, 1, 0)) * 1.0 / count(*) as limit_up_ratio,
                       sum(iif(S_DQ_CLOSE <= S_DQ_STOPPING, 1, 0)) * 1.0 / count(*) as limit_down_ratio,
                       std(S_DQ_PCTCHANGE) as cross_sec_vol
                from loadTable("{DDB_EOD}", "data")
                where TRADE_DT >= {start_s}
                  and TRADE_DT <= {end_s}
                group by TRADE_DT
                order by TRADE_DT
                """
            )
        except Exception as lim_exc:
            print("[db] market limit-ratio query failed:", lim_exc)
            lim = None
    finally:
        session.close()

    cols = ["date", "mkt_vol_20d", "limit_up_ratio", "limit_down_ratio", "cross_sec_vol"]
    if idx is None or len(idx) == 0:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(idx)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    ret = out["close"].pct_change()
    out["mkt_vol_20d"] = ret.rolling(20, min_periods=10).std()

    if lim is not None and len(lim) > 0:
        lim_df = pd.DataFrame(lim)
        if "date" not in lim_df.columns and "TRADE_DT" in lim_df.columns:
            lim_df = lim_df.rename(columns={"TRADE_DT": "date"})
        lim_df["date"] = pd.to_datetime(lim_df["date"]).dt.normalize()
        for c in ["limit_up_ratio", "limit_down_ratio", "cross_sec_vol"]:
            lim_df[c] = pd.to_numeric(lim_df[c], errors="coerce")
        # cross_sec_vol from Wind is in percent units (~2.0); scale to decimal
        lim_df["cross_sec_vol"] = lim_df["cross_sec_vol"] / 100.0
        out = out.merge(
            lim_df[["date", "limit_up_ratio", "limit_down_ratio", "cross_sec_vol"]],
            on="date",
            how="left",
        )
    else:
        out["limit_up_ratio"] = np.nan
        out["limit_down_ratio"] = np.nan
        out["cross_sec_vol"] = np.nan

    return out[cols].sort_values("date").reset_index(drop=True)


def get_index_close(index_code: str, start, end) -> pd.Series:
    """Daily close series for a Wind index code (e.g. 000300.SH)."""
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        df = session.run(
            f"""
            select TRADE_DT as date, S_DQ_CLOSE as close
            from loadTable("{DDB_INDEX}", "data")
            where S_INFO_WINDCODE = "{index_code}"
              and TRADE_DT >= {start_s}
              and TRADE_DT <= {end_s}
            order by TRADE_DT
            """
        )
    finally:
        session.close()
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.set_index("date")["close"].sort_index()


# Wind daily weight tables (PIT membership) — same mapping as Factor_Dev_Lib
_INDEX_WEIGHT_DB = {
    "000300.SH": "dfs://WIND.AINDEXHS300WEIGHT",
    "000905.SH": "dfs://WIND.AINDEXCSI500WEIGHT",
    "000852.SH": "dfs://WIND.AINDEXCSI1000WEIGHT",
}


def get_index_member_mask(index_code: str, start, end) -> pd.DataFrame:
    """Daily index membership wide mask: 1.0 = in index, NaN = out (PIT).

    CSI300/500/1000 use Wind daily weight tables. Other indices fall back to
    ``AINDEXMEMBERS`` in/out-date expansion.
    """
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    session = connect_ddb()
    try:
        if index_code in _INDEX_WEIGHT_DB:
            db_path = _INDEX_WEIGHT_DB[index_code]
            df = session.run(
                f"""
                select TRADE_DT as date, S_CON_WINDCODE as symbol
                from loadTable("{db_path}", "data")
                where TRADE_DT >= {start_s} and TRADE_DT <= {end_s}
                """
            )
            if df is None or len(df) == 0:
                return pd.DataFrame(dtype=float)
            out = pd.DataFrame(df)
            out["date"] = pd.to_datetime(out["date"]).dt.normalize()
            out["symbol"] = out["symbol"].astype(str)
            out["flag"] = 1.0
            wide = out.pivot_table(
                index="date", columns="symbol", values="flag", aggfunc="last"
            )
            return wide.sort_index()

        # Generic members table
        script = f"""
        t = loadTable("dfs://WIND.AINDEXMEMBERS", "data")
        select S_CON_WINDCODE as symbol,
               S_CON_INDATE as indate,
               nullFill(temporalParse(S_CON_OUTDATE, "yyyyMMdd"), 2100.01.01) as outdate
        from t
        where S_INFO_WINDCODE = "{index_code}"
        context by S_CON_WINDCODE, S_CON_INDATE
        csort OPDATE desc
        limit 1
        """
        members = session.run(script)
        cal = session.run(
            f"""
            select distinct TRADE_DT as date
            from loadTable("{DDB_EOD}", "data")
            where TRADE_DT >= {start_s} and TRADE_DT <= {end_s}
            order by TRADE_DT
            """
        )
    finally:
        session.close()

    if members is None or len(members) == 0 or cal is None or len(cal) == 0:
        return pd.DataFrame(dtype=float)
    members = pd.DataFrame(members)
    members["indate"] = pd.to_datetime(members["indate"])
    members["outdate"] = pd.to_datetime(members["outdate"])
    trading = pd.to_datetime(pd.DataFrame(cal)["date"]).dt.normalize().sort_values()
    trade_arr = trading.to_numpy(dtype="datetime64[ns]")
    left = np.searchsorted(
        trade_arr, members["indate"].to_numpy(dtype="datetime64[ns]"), side="left"
    )
    right = np.searchsorted(
        trade_arr, members["outdate"].to_numpy(dtype="datetime64[ns]"), side="right"
    )
    parts = []
    for code, lo, hi in zip(members["symbol"].to_numpy(), left, right):
        if lo >= hi:
            continue
        parts.append(
            pd.DataFrame(
                {"date": trading.iloc[lo:hi].values, "symbol": code, "flag": 1.0}
            )
        )
    if not parts:
        return pd.DataFrame(index=trading, dtype=float)
    narrow = pd.concat(parts, ignore_index=True)
    wide = narrow.pivot_table(
        index="date", columns="symbol", values="flag", aggfunc="last"
    )
    return wide.reindex(trading).sort_index()


def get_index_components(index_code: str, trade_date) -> list:
    """Constituent WindCodes for ``index_code`` on ``trade_date`` (PIT, with backfill)."""
    d = pd.Timestamp(trade_date).normalize()
    # Small window lookback for holiday / missing weight rows
    start = (d - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = d.strftime("%Y-%m-%d")
    mask = get_index_member_mask(index_code, start, end)
    if mask.empty:
        return []
    # Prefer exact date; else last available <= d
    if d in mask.index:
        row = mask.loc[d]
    else:
        avail = mask.index[mask.index <= d]
        if len(avail) == 0:
            return []
        row = mask.loc[avail[-1]]
    return sorted([str(c) for c in row.index if pd.notna(row[c])])


def _ddb_symbol_vector_literal(symbols: Sequence[str]) -> str:
    cleaned = [str(s).replace('"', "") for s in symbols]
    return "[" + ",".join('"{}"'.format(s) for s in cleaned) + "]"


def get_ohlcv_bulk(symbols: Sequence[str], start, end) -> pd.DataFrame:
    """Bulk daily OHLCV for many WindCodes (adds ``symbol`` column)."""
    if not symbols:
        return pd.DataFrame()
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    sym_lit = _ddb_symbol_vector_literal(symbols)
    session = connect_ddb()
    try:
        script = f"""
        syms = {sym_lit}
        select TRADE_DT as date,
               S_INFO_WINDCODE as symbol,
               S_DQ_OPEN as open,
               S_DQ_HIGH as high,
               S_DQ_LOW as low,
               S_DQ_CLOSE as close,
               S_DQ_VOLUME as volume,
               S_DQ_AMOUNT as amount,
               S_DQ_PRECLOSE as preclose,
               S_DQ_LIMIT as up_limit,
               S_DQ_STOPPING as down_limit,
               S_DQ_TRADESTATUS as trade_status
        from loadTable("{DDB_EOD}", "data")
        where S_INFO_WINDCODE in syms
          and TRADE_DT >= {start_s}
          and TRADE_DT <= {end_s}
        """
        df = session.run(script)
    finally:
        session.close()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    for col in ["open", "high", "low", "close", "volume", "amount", "preclose", "up_limit", "down_limit"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def get_valuation_bulk(symbols: Sequence[str], start, end) -> pd.DataFrame:
    """Bulk EP/BP/turnover/log_mktcap for many WindCodes."""
    if not symbols:
        return pd.DataFrame()
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    sym_lit = _ddb_symbol_vector_literal(symbols)
    session = connect_ddb()
    try:
        script = f"""
        syms = {sym_lit}
        select TRADE_DT as date,
               S_INFO_WINDCODE as symbol,
               S_VAL_PE_TTM as pe_ttm,
               S_VAL_PB_NEW as pb,
               S_DQ_TURN as turnover,
               S_VAL_MV as mkt_cap
        from loadTable("{DDB_DERIV}", "data")
        where S_INFO_WINDCODE in syms
          and TRADE_DT >= {start_s}
          and TRADE_DT <= {end_s}
        """
        df = session.run(script)
    finally:
        session.close()
    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=["date", "symbol", "ep_ttm", "bp", "turnover", "log_mktcap"]
        )
    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    for col in ["pe_ttm", "pb", "turnover", "mkt_cap"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["ep_ttm"] = np.where(out["pe_ttm"].abs() > 1e-8, 1.0 / out["pe_ttm"], np.nan)
    out["bp"] = np.where(out["pb"].abs() > 1e-8, 1.0 / out["pb"], np.nan)
    out["log_mktcap"] = np.log(out["mkt_cap"].where(out["mkt_cap"] > 0))
    return (
        out[["date", "symbol", "ep_ttm", "bp", "turnover", "log_mktcap"]]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def get_roe_bulk(symbols: Sequence[str], start, end) -> pd.DataFrame:
    """Bulk PIT ROE_TTM: announcement-date aligned, ffilled to daily calendar per symbol."""
    if not symbols:
        return pd.DataFrame(columns=["date", "symbol", "roe"])
    hist_start = pd.Timestamp(start) - pd.Timedelta(days=800)
    start_s = _ddb_date_literal(hist_start)
    end_s = _ddb_date_literal(end)
    sym_lit = _ddb_symbol_vector_literal(symbols)
    session = connect_ddb()
    try:
        script = f"""
        syms = {sym_lit}
        select ANN_DT as ann_date,
               S_INFO_WINDCODE as symbol,
               S_FA_ROE_TTM as roe
        from loadTable("{DDB_TTMHIS}", "data")
        where S_INFO_WINDCODE in syms
          and STATEMENT_TYPE = "合并报表"
          and ANN_DT >= {start_s}
          and ANN_DT <= {end_s}
        """
        df = session.run(script)
    finally:
        session.close()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "roe"])
    fin = pd.DataFrame(df)
    fin["ann_date"] = pd.to_datetime(fin["ann_date"]).dt.normalize()
    fin["symbol"] = fin["symbol"].astype(str)
    fin["roe"] = pd.to_numeric(fin["roe"], errors="coerce")
    fin["available_date"] = fin["ann_date"] + pd.Timedelta(days=1)
    fin = fin.dropna(subset=["available_date"]).sort_values(["symbol", "available_date"])
    fin = fin.drop_duplicates(["symbol", "available_date"], keep="last")

    cal_start = pd.Timestamp(start).normalize()
    cal_end = pd.Timestamp(end).normalize()
    days = pd.date_range(cal_start, cal_end, freq="D")
    parts = []
    for sym, g in fin.groupby("symbol", sort=False):
        daily = pd.DataFrame({"date": days})
        daily = pd.merge_asof(
            daily,
            g[["available_date", "roe"]].sort_values("available_date"),
            left_on="date",
            right_on="available_date",
            direction="backward",
        )
        daily["symbol"] = sym
        parts.append(daily[["date", "symbol", "roe"]])
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", "roe"])
    return pd.concat(parts, ignore_index=True)


def get_northbound_bulk(symbols: Sequence[str], start, end) -> pd.DataFrame:
    """Bulk northbound share ratio for many WindCodes."""
    if not symbols:
        return pd.DataFrame()
    start_s = _ddb_date_literal(start)
    end_s = _ddb_date_literal(end)
    sym_lit = _ddb_symbol_vector_literal(symbols)
    session = connect_ddb()
    try:
        script = f"""
        syms = {sym_lit}
        select TRADE_DT as date,
               S_INFO_WINDCODE as symbol,
               S_RATIO as north_share_ratio
        from loadTable("{DDB_NORTH}", "data")
        where S_INFO_WINDCODE in syms
          and TRADE_DT >= {start_s}
          and TRADE_DT <= {end_s}
        """
        df = session.run(script)
    finally:
        session.close()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "symbol", "north_share_ratio", "north_share_chg"])
    out = pd.DataFrame(df)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    out["north_share_ratio"] = pd.to_numeric(out["north_share_ratio"], errors="coerce")
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["date", "symbol"], keep="last")
    out["north_share_chg"] = out.groupby("symbol")["north_share_ratio"].diff()
    return out[["date", "symbol", "north_share_ratio", "north_share_chg"]].reset_index(drop=True)


def get_citics_l1_industry_map(
    symbols: Optional[Sequence[str]] = None,
    asof=None,
) -> Dict[str, str]:
    """Point-in-time Citics L1 industry code map (prefix of CITICS_IND_CODE).

    Returns ``{WindCode: l1_code}`` where ``l1_code`` is the first 4 chars of
    ``CITICS_IND_CODE`` (repo standard for sector neutralization).
    """
    asof_ts = pd.Timestamp(asof or "2100-01-01").normalize()
    session = connect_ddb()
    try:
        if symbols:
            sym_lit = _ddb_symbol_vector_literal(symbols)
            script = f"""
            syms = {sym_lit}
            select S_INFO_WINDCODE as symbol,
                   CITICS_IND_CODE as ind,
                   ENTRY_DT as entry,
                   REMOVE_DT as remove
            from loadTable("dfs://WIND.ASHAREINDUSTRIESCLASSCITICS", "data")
            where S_INFO_WINDCODE in syms
            """
        else:
            script = """
            select S_INFO_WINDCODE as symbol,
                   CITICS_IND_CODE as ind,
                   ENTRY_DT as entry,
                   REMOVE_DT as remove
            from loadTable("dfs://WIND.ASHAREINDUSTRIESCLASSCITICS", "data")
            """
        df = session.run(script)
    finally:
        session.close()
    if df is None or len(df) == 0:
        return {}
    out = pd.DataFrame(df)
    out["symbol"] = out["symbol"].astype(str)
    out["ind"] = out["ind"].astype(str)
    out["entry"] = pd.to_datetime(out["entry"], errors="coerce")
    out["remove"] = pd.to_datetime(out["remove"], errors="coerce")
    # active on asof: entry <= asof and (remove is null or remove > asof)
    active = out[(out["entry"].isna() | (out["entry"] <= asof_ts))].copy()
    active = active[active["remove"].isna() | (active["remove"] > asof_ts)]
    if active.empty:
        active = out.sort_values("entry").groupby("symbol", as_index=False).tail(1)
    else:
        active = active.sort_values("entry").groupby("symbol", as_index=False).tail(1)
    # L1 = first 4 chars of Citics code (e.g. b10j...)
    mapping = {}
    for _, row in active.iterrows():
        code = str(row["ind"])
        mapping[str(row["symbol"])] = code[:4] if len(code) >= 4 else code
    return mapping


def get_l2_daily_factors(
    symbols: Sequence[str],
    start,
    end,
    *,
    client=None,
) -> pd.DataFrame:
    """Daily L2 book features from ClickHouse SSL2 (server-side aggregation).

    Columns: date, symbol, l2_obi_l1, l2_depth_oi, l2_rel_spread, l2_micro_bias, n_snap
    """
    if not symbols:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "l2_obi_l1",
                "l2_depth_oi",
                "l2_rel_spread",
                "l2_micro_bias",
                "n_snap",
            ]
        )

    from sideprojects.f2_agent_lite.factors.factor_minute import CONTINUOUS_SESSION_SQL

    own = client is None
    client = client or get_clickhouse_client()
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    sh = sorted({str(s).split(".")[0] for s in symbols if str(s).endswith(".SH")})
    sz = sorted(
        {
            str(s).split(".")[0]
            for s in symbols
            if str(s).endswith(".SZ") or str(s).split(".")[0].startswith(("0", "3"))
        }
    )
    # avoid double-count SH bare codes that start with 6
    sz = [c for c in sz if c not in sh]

    frames = []
    try:
        for table, suffix, bare_list in (
            ("SSE_AL_SSL2_EXG", ".SH", sh),
            ("SZSE_AL_SSL2_EXG", ".SZ", sz),
        ):
            if not bare_list:
                continue
            in_list = ", ".join("'{}'".format(c) for c in bare_list)
            sql = f"""
            SELECT
              toDate(ExchTime) AS date,
              concat(Symbol, '{suffix}') AS symbol,
              avg(
                if(length(BidVolumes) >= 1 AND length(AskVolumes) >= 1
                   AND (toFloat64(BidVolumes[1]) + toFloat64(AskVolumes[1])) != 0,
                   (toFloat64(BidVolumes[1]) - toFloat64(AskVolumes[1]))
                    / (toFloat64(BidVolumes[1]) + toFloat64(AskVolumes[1])),
                   NULL)
              ) AS l2_obi_l1,
              avg(
                if(isFinite(toFloat64(TotalBidVolume)) AND isFinite(toFloat64(TotalAskVolume))
                   AND (toFloat64(TotalBidVolume) + toFloat64(TotalAskVolume)) != 0,
                   (toFloat64(TotalBidVolume) - toFloat64(TotalAskVolume))
                    / (toFloat64(TotalBidVolume) + toFloat64(TotalAskVolume)),
                   NULL)
              ) AS l2_depth_oi,
              avg(
                if(length(BidPrices) >= 1 AND length(AskPrices) >= 1
                   AND BidPrices[1] > 0 AND AskPrices[1] > 0,
                   (toFloat64(AskPrices[1]) - toFloat64(BidPrices[1]))
                    / ((toFloat64(AskPrices[1]) + toFloat64(BidPrices[1])) / 2),
                   NULL)
              ) AS l2_rel_spread,
              avg(
                if(length(BidPrices) >= 1 AND length(AskPrices) >= 1
                   AND length(BidVolumes) >= 1 AND length(AskVolumes) >= 1
                   AND (toFloat64(BidVolumes[1]) + toFloat64(AskVolumes[1])) != 0
                   AND BidPrices[1] > 0 AND AskPrices[1] > 0,
                   (
                     (toFloat64(BidPrices[1]) * toFloat64(AskVolumes[1])
                      + toFloat64(AskPrices[1]) * toFloat64(BidVolumes[1]))
                     / (toFloat64(BidVolumes[1]) + toFloat64(AskVolumes[1]))
                     - (toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2
                   ) / ((toFloat64(BidPrices[1]) + toFloat64(AskPrices[1])) / 2),
                   NULL)
              ) AS l2_micro_bias,
              count() AS n_snap
            FROM cmds.`{table}`
            WHERE Symbol IN ({in_list})
              AND ExchTime >= toDateTime64('{start_s} 00:00:00', 6, 'Asia/Shanghai')
              AND ExchTime <  toDateTime64('{end_excl} 00:00:00', 6, 'Asia/Shanghai')
              AND {CONTINUOUS_SESSION_SQL}
              AND length(BidPrices) >= 1 AND length(AskPrices) >= 1
              AND length(BidVolumes) >= 1 AND length(AskVolumes) >= 1
            GROUP BY date, symbol
            """
            result = client.query(sql)
            if not result.result_rows:
                continue
            frames.append(pd.DataFrame(result.result_rows, columns=result.column_names))
    finally:
        if own and client is not None:
            try:
                client.close()
            except Exception:
                pass

    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "l2_obi_l1",
                "l2_depth_oi",
                "l2_rel_spread",
                "l2_micro_bias",
                "n_snap",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    for c in ["l2_obi_l1", "l2_depth_oi", "l2_rel_spread", "l2_micro_bias", "n_snap"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def compute_advanced_alpha(
    ohlcv: pd.DataFrame,
    market_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """WorldQuant-style 高阶量价：动量 / Amihud 非流动性 / 残差波动(RSQR proxy)。"""
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(
            columns=["date", "ret_20d", "ret_60d", "illiquidity", "resid_vol_20d"]
        )

    df = ohlcv.sort_values("date").copy()
    dates = pd.to_datetime(df["date"]).dt.normalize()
    close = pd.to_numeric(df["close"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce") if "amount" in df.columns else None
    ret = close.pct_change()

    ret_20d = close / close.shift(20) - 1.0
    ret_60d = close / close.shift(60) - 1.0
    if amount is not None:
        illiq = (ret.abs() / amount.replace(0, np.nan)) * 1e8
        illiquidity = illiq.rolling(20, min_periods=10).mean()
    else:
        illiquidity = pd.Series(np.nan, index=df.index)

    resid_vol = ret.rolling(20, min_periods=10).std()
    if market_close is not None and len(market_close) > 0:
        mkt_ret = market_close.pct_change()
        mkt_aligned = dates.map(lambda d: mkt_ret.get(pd.Timestamp(d), np.nan))
        y = ret.to_numpy(dtype=float)
        x = pd.to_numeric(mkt_aligned, errors="coerce").to_numpy(dtype=float)
        w = 20
        vals = np.full(len(df), np.nan)
        for i in range(w - 1, len(df)):
            yy = y[i - w + 1 : i + 1]
            xx = x[i - w + 1 : i + 1]
            mask = np.isfinite(yy) & np.isfinite(xx)
            if mask.sum() < 10:
                continue
            yy, xx = yy[mask], xx[mask]
            xb = np.column_stack([np.ones(len(xx)), xx])
            try:
                beta, *_ = np.linalg.lstsq(xb, yy, rcond=None)
                resid = yy - xb @ beta
                vals[i] = float(np.std(resid, ddof=1))
            except Exception:
                continue
        resid_vol = pd.Series(vals, index=df.index)

    return pd.DataFrame(
        {
            "date": dates.values,
            "ret_20d": ret_20d.values,
            "ret_60d": ret_60d.values,
            "illiquidity": illiquidity.values,
            "resid_vol_20d": resid_vol.values,
        }
    )


# ---------------------------------------------------------------------------
# Minute factors (ClickHouse KLIN → daily)
# ---------------------------------------------------------------------------


def get_minute_factors(
    symbols: Sequence[str],
    start,
    end,
    *,
    lookback: int = 10,
    use_local_tables: bool = False,
    config=None,
    client=None,
) -> pd.DataFrame:
    """Fetch 1MIN KLIN bars and return daily minute_amplitude / price_jump.

    Output columns: date, symbol, minute_amplitude, price_jump.
    """
    from sideprojects.f2_agent_lite.factors.factor_minute import (
        compute_minute_daily_factors,
        fetch_minute_data_from_clickhouse,
    )

    own = client is None
    client = client or get_clickhouse_client(config)
    try:
        # Extra calendar buffer so lookback rolling has history
        hist_start = (pd.Timestamp(start) - pd.Timedelta(days=max(40, int(lookback) * 3))).strftime(
            "%Y-%m-%d"
        )
        end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
        minute = fetch_minute_data_from_clickhouse(
            client,
            symbols,
            hist_start,
            end_s,
            use_local_tables=use_local_tables,
        )
    finally:
        if own and client is not None:
            try:
                client.close()
            except Exception:
                pass

    return compute_minute_daily_factors(minute, lookback=int(lookback))


def get_minute_factors_one(
    symbol: str,
    start,
    end,
    *,
    lookback: int = 10,
    use_local_tables: bool = False,
    config=None,
    client=None,
) -> pd.DataFrame:
    """Single-symbol wrapper returning date-indexed daily minute factors."""
    frame = get_minute_factors(
        [symbol],
        start,
        end,
        lookback=lookback,
        use_local_tables=use_local_tables,
        config=config,
        client=client,
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "minute_amplitude", "price_jump"])
    out = frame[frame["symbol"] == symbol][["date", "minute_amplitude", "price_jump"]].copy()
    return out.sort_values("date").reset_index(drop=True)
