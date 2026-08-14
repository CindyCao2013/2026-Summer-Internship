"""EOD / 衍生指标 / 财报公告宽表加载。公式层与 runner 不直接连库。"""

import datetime as dt
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import dolphindb as ddb
import pandas as pd

from COMMON_CONST import DATA_DB_CONN, DATA_DB_WIND
from factor_finance import normalize_finance_long


def _filter_a_share_cols(df: pd.DataFrame) -> pd.DataFrame:
    selected = [x for x in df.columns if x[0] in ("6", "0", "3")]
    return df[selected]


def _pivot_table(t_table, value_col: str, alias: str) -> pd.DataFrame:
    df = t_table.select(
        f"TRADE_DT as Date, S_INFO_WINDCODE as WindCode, {value_col} as {alias}"
    ).executeAs(f"df_{alias.lower()}")
    return df.select(alias).pivotby("Date", "WindCode").toDF().set_index("Date")


@dataclass
class EODWideTables:
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    turnover: Optional[pd.DataFrame] = None


@dataclass
class EODEnrichedTables:
    """EOD OHLCV + derivative market cap (Level 2 liquidity normalization)."""

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    total_mktcap: pd.DataFrame
    float_mktcap: pd.DataFrame
    turnover: Optional[pd.DataFrame] = None


@dataclass
class DerivativeWideTables:
    total_mktcap: pd.DataFrame
    float_mktcap: pd.DataFrame
    pb: Optional[pd.DataFrame] = None
    pe_ttm: Optional[pd.DataFrame] = None
    ps_ttm: Optional[pd.DataFrame] = None


def connect_ddb():
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    return s


def load_eod_wide_tables(
    start_preheat,
    end_day,
    session=None,
) -> Tuple[EODWideTables, object]:
    """从 WIND.ASHAREEODPRICES 加载价量宽表。"""
    own_session = session is None
    s = session or connect_ddb()

    t_eod = s.loadTable(dbPath="dfs://WIND.ASHAREEODPRICES", tableName="data")
    t_eod = t_eod.where(
        f"TRADE_DT>= {start_preheat.strftime('%Y.%m.%d')} "
        f"and TRADE_DT <= {end_day.strftime('%Y.%m.%d')} "
    )

    close = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_CLOSE", "Close"))
    open_ = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_OPEN", "Open"))
    high = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_HIGH", "High"))
    low = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_LOW", "Low"))
    volume = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_VOLUME", "Volume"))
    amount = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_AMOUNT", "Amount"))

    turnover = None
    try:
        turnover = _filter_a_share_cols(_pivot_table(t_eod, "S_DQ_TURN", "Turnover"))
        print("Loaded turnover field: S_DQ_TURN")
    except Exception as exc:
        print(f"WARNING: turnover field not available ({exc})")

    tables = EODWideTables(
        close=close,
        open=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        turnover=turnover,
    )
    return tables, s


def _month_starts(start: dt.datetime, end: dt.datetime):
    cur = dt.datetime(start.year, start.month, 1)
    last = dt.datetime(end.year, end.month, 1)
    while cur <= last:
        yield cur
        if cur.month == 12:
            cur = dt.datetime(cur.year + 1, 1, 1)
        else:
            cur = dt.datetime(cur.year, cur.month + 1, 1)


def _oracle_month_end(month_start: dt.datetime) -> dt.datetime:
    if month_start.month == 12:
        return dt.datetime(month_start.year, 12, 31)
    return dt.datetime(month_start.year, month_start.month + 1, 1) - dt.timedelta(days=1)


def load_eod_wide_tables_from_wind_oracle(
    start_preheat,
    end_day,
    *,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> Tuple[EODWideTables, pd.DataFrame]:
    """从 Wind Oracle `ASHAREEODPRICES` 拉价量宽表（覆盖 DDB 2018 之前区间）。

    按月落本地 parquet 缓存后 pivot；默认跑完删除缓存。
    同时返回 c2c 收益宽表：``S_DQ_CLOSE / S_DQ_PRECLOSE - 1``（与 Factor_Dev_Lib 一致）。
    """
    import oracledb

    oracledb.init_oracle_client(lib_dir=None)

    own_cache = cache_dir is None
    cache_root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="wind_eod_"))
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Wind Oracle EOD cache: {cache_root}")

    # S_DQ_TURN lives on derivative table in Wind Oracle, not ASHAREEODPRICES
    fields = (
        "TRADE_DT",
        "S_INFO_WINDCODE",
        "S_DQ_OPEN",
        "S_DQ_HIGH",
        "S_DQ_LOW",
        "S_DQ_CLOSE",
        "S_DQ_PRECLOSE",
        "S_DQ_VOLUME",
        "S_DQ_AMOUNT",
    )
    sql = f"""
        SELECT {", ".join(fields)}
        FROM wind.ASHAREEODPRICES
        WHERE TRADE_DT >= :d0 AND TRADE_DT <= :d1
          AND (S_INFO_WINDCODE LIKE '0%'
            OR S_INFO_WINDCODE LIKE '3%'
            OR S_INFO_WINDCODE LIKE '6%')
    """

    parts = []
    try:
        with oracledb.connect(**DATA_DB_WIND) as conn:
            for ms in _month_starts(start_preheat, end_day):
                me = min(_oracle_month_end(ms), end_day)
                d0 = max(start_preheat, ms).strftime("%Y%m%d")
                d1 = me.strftime("%Y%m%d")
                part_path = cache_root / f"eod_{d0}_{d1}.parquet"
                if part_path.exists():
                    print(f"  cache hit {part_path.name}")
                    parts.append(pd.read_parquet(part_path))
                    continue
                print(f"  pull {d0}->{d1} ...", flush=True)
                df = pd.read_sql(sql, conn, params={"d0": d0, "d1": d1})
                df.columns = [c.upper() for c in df.columns]
                if len(df) == 0:
                    print(f"  empty {d0}->{d1}")
                    continue
                df.to_parquet(part_path, index=False)
                parts.append(df)
                print(f"  rows={len(df):,} -> {part_path.name}", flush=True)

        if not parts:
            empty = pd.DataFrame()
            tables = EODWideTables(
                close=empty, open=empty, high=empty, low=empty,
                volume=empty, amount=empty, turnover=None,
            )
            return tables, empty

        long = pd.concat(parts, ignore_index=True)
        long["TRADE_DT"] = pd.to_datetime(long["TRADE_DT"].astype(str), format="%Y%m%d")
        long = long.sort_values(["TRADE_DT", "S_INFO_WINDCODE"]).drop_duplicates(
            subset=["TRADE_DT", "S_INFO_WINDCODE"], keep="last"
        )

        def _pivot(col: str) -> pd.DataFrame:
            wide = long.pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values=col)
            return _filter_a_share_cols(wide)

        close = _pivot("S_DQ_CLOSE")
        open_ = _pivot("S_DQ_OPEN").reindex(index=close.index, columns=close.columns)
        high = _pivot("S_DQ_HIGH").reindex(index=close.index, columns=close.columns)
        low = _pivot("S_DQ_LOW").reindex(index=close.index, columns=close.columns)
        volume = _pivot("S_DQ_VOLUME").reindex(index=close.index, columns=close.columns)
        amount = _pivot("S_DQ_AMOUNT").reindex(index=close.index, columns=close.columns)
        preclose = _pivot("S_DQ_PRECLOSE").reindex(index=close.index, columns=close.columns)
        ret_c2c = close / preclose - 1.0

        tables = EODWideTables(
            close=close,
            open=open_,
            high=high,
            low=low,
            volume=volume,
            amount=amount,
            turnover=None,
        )
        print(
            f"Wind Oracle EOD wide: {close.shape[0]}d x {close.shape[1]} names "
            f"({close.index.min().date()} -> {close.index.max().date()})"
        )
        return tables, ret_c2c
    finally:
        if own_cache and not keep_cache and cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
            print(f"Cleared Wind Oracle EOD cache: {cache_root}")


def _align_to_close(reference: pd.DataFrame, *frames: pd.DataFrame) -> tuple:
    """Reindex derivative panels to EOD close grid."""
    aligned = [reference]
    for df in frames:
        aligned.append(df.reindex(index=reference.index, columns=reference.columns))
    return tuple(aligned)


def load_eod_enriched_tables(
    start_preheat,
    end_day,
    session=None,
) -> Tuple[EODEnrichedTables, object]:
    """EOD OHLCV + float/total mktcap for size-adjusted liquidity factors."""
    eod, s = load_eod_wide_tables(start_preheat, end_day, session=session)
    der, _ = load_derivative_wide_tables(start_preheat, end_day, session=s)

    close, total_mktcap, float_mktcap = _align_to_close(
        eod.close, der.total_mktcap, der.float_mktcap
    )
    turnover = eod.turnover
    if turnover is not None:
        turnover = turnover.reindex(index=close.index, columns=close.columns)

    tables = EODEnrichedTables(
        close=close,
        open=eod.open.reindex(index=close.index, columns=close.columns),
        high=eod.high.reindex(index=close.index, columns=close.columns),
        low=eod.low.reindex(index=close.index, columns=close.columns),
        volume=eod.volume.reindex(index=close.index, columns=close.columns),
        amount=eod.amount.reindex(index=close.index, columns=close.columns),
        turnover=turnover,
        total_mktcap=total_mktcap,
        float_mktcap=float_mktcap,
    )
    if turnover is None:
        print("Using turnover proxy: amount / float_mktcap")
    return tables, s


def _try_pivot_derivative(t_table, value_col: str, alias: str) -> Optional[pd.DataFrame]:
    try:
        return _filter_a_share_cols(_pivot_table(t_table, value_col, alias))
    except Exception as exc:
        print(f"WARNING: derivative field {value_col} not available ({exc})")
        return None


def load_derivative_wide_tables(
    start_preheat,
    end_day,
    session=None,
) -> Tuple[DerivativeWideTables, object]:
    """从 WIND.ASHAREEODDERIVATIVEINDICATOR 加载估值 / 市值宽表。"""
    own_session = session is None
    s = session or connect_ddb()

    t_der = s.loadTable(
        dbPath="dfs://WIND.ASHAREEODDERIVATIVEINDICATOR", tableName="data"
    )
    t_der = t_der.where(
        f"TRADE_DT>= {start_preheat.strftime('%Y.%m.%d')} "
        f"and TRADE_DT <= {end_day.strftime('%Y.%m.%d')} "
    )

    total_mktcap = _filter_a_share_cols(_pivot_table(t_der, "S_VAL_MV", "TotalMV"))
    float_mktcap = _filter_a_share_cols(_pivot_table(t_der, "S_DQ_MV", "FloatMV"))
    pb = _try_pivot_derivative(t_der, "S_VAL_PB_NEW", "PB")
    pe_ttm = _try_pivot_derivative(t_der, "S_VAL_PE_TTM", "PE")
    ps_ttm = _try_pivot_derivative(t_der, "S_VAL_PS_TTM", "PS")

    tables = DerivativeWideTables(
        total_mktcap=total_mktcap,
        float_mktcap=float_mktcap,
        pb=pb,
        pe_ttm=pe_ttm,
        ps_ttm=ps_ttm,
    )
    return tables, s


def load_financial_ttmhis_long(
    start_preheat: dt.datetime,
    end_day: dt.datetime,
    session=None,
    *,
    history_years: int = 3,
    statement_type: str = "合并报表",
) -> Tuple[pd.DataFrame, object]:
    """
    从 WIND.ASHARETTMHIS 加载 ann_date 财报面板（ROE + D7 Quality 字段）。

    history_years: 为 roe_stability 滚动窗口预留更早公告历史。
    """
    own_session = session is None
    s = session or connect_ddb()
    ann_start = start_preheat - dt.timedelta(days=365 * history_years)

    t_fin = s.loadTable(dbPath="dfs://WIND.ASHARETTMHIS", tableName="data")
    t_fin = t_fin.where(
        f"ANN_DT>= {ann_start.strftime('%Y.%m.%d')} "
        f"and ANN_DT <= {end_day.strftime('%Y.%m.%d')} "
        f"and STATEMENT_TYPE=`{statement_type}"
    )
    df = t_fin.select(
        "S_INFO_WINDCODE as symbol, ANN_DT as ann_date, "
        "REPORT_PERIOD as report_period, S_FA_ROE_TTM as roe, "
        "S_FA_GROSSMARGIN_TTM as gross_profit_ttm, "
        "S_FA_ASSET_MRQ as total_assets_mrq, "
        "NET_CASH_FLOWS_OPER_ACT_TTM as cfo_ttm, "
        "NET_PROFIT_PARENT_COMP_TTM as net_profit_ttm"
    ).executeAs("df_ttmhis_ann")
    if hasattr(df, "toDF"):
        df = df.toDF()

    if df is None or len(df) == 0:
        long_df = pd.DataFrame(
            columns=[
                "symbol",
                "ann_date",
                "report_period",
                "roe",
                "gross_profit_ttm",
                "total_assets_mrq",
                "cfo_ttm",
                "net_profit_ttm",
            ]
        )
    else:
        long_df = normalize_finance_long(df)
        long_df = long_df[long_df["symbol"].str[0].isin(("6", "0", "3"))]
        long_df = long_df.dropna(subset=["ann_date"])
        long_df = long_df.sort_values(["symbol", "ann_date"]).drop_duplicates(
            subset=["symbol", "ann_date"], keep="last"
        )

    if own_session:
        s.close()
    return long_df, s


def load_financial_roe_long(
    start_preheat: dt.datetime,
    end_day: dt.datetime,
    session=None,
    *,
    history_years: int = 3,
    statement_type: str = "合并报表",
) -> Tuple[pd.DataFrame, object]:
    """Backward-compatible alias — loads full TTMHIS panel (ROE + quality fields)."""
    return load_financial_ttmhis_long(
        start_preheat,
        end_day,
        session=session,
        history_years=history_years,
        statement_type=statement_type,
    )
