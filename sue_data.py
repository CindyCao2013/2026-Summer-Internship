"""SUE / earnings-event data loaders (Wind Oracle).

Builds a point-in-time **earliest-known** earnings timeline:
  known_date = min(profit_notice, profit_express, formal_income)
for each (symbol, report_period).

Units:
  - ASHAREINCOME / ASHAREPROFITEXPRESS net profit: yuan
  - ASHAREPROFITNOTICE net profit min/max: 万元 → converted ×1e4
  - ASHARECONSENSUSDATA NET_PROFIT_AVG: 万元 → converted ×1e4
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from COMMON_CONST import DATA_DB_WIND

STATEMENT_CONSOLIDATED = "408001000"
# Wind consensus cycle: FY1 / rolling FY — prefer 263001000 (FY1)
CONSENSUS_FY1 = "263001000"
NOTICE_NP_TO_YUAN = 1e4
CONSENSUS_NP_TO_YUAN = 1e4


def _init_oracle():
    import oracledb

    oracledb.init_oracle_client(lib_dir=None)
    return oracledb


def _month_starts(start: dt.datetime, end: dt.datetime):
    cur = dt.datetime(start.year, start.month, 1)
    last = dt.datetime(end.year, end.month, 1)
    while cur <= last:
        yield cur
        cur = (
            dt.datetime(cur.year + 1, 1, 1)
            if cur.month == 12
            else dt.datetime(cur.year, cur.month + 1, 1)
        )


def _month_end(ms: dt.datetime) -> dt.datetime:
    if ms.month == 12:
        return dt.datetime(ms.year, 12, 31)
    return dt.datetime(ms.year, ms.month + 1, 1) - dt.timedelta(days=1)


def _a_share_mask(code: pd.Series) -> pd.Series:
    s = code.astype(str)
    return s.str[0].isin(("0", "3", "6"))


def _read_sql_monthly(
    sql: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    date_bind_start: str,
    date_bind_end: str,
    cache_dir: Path,
    label: str,
) -> pd.DataFrame:
    oracledb = _init_oracle()
    parts = []
    with oracledb.connect(**DATA_DB_WIND) as conn:
        for ms in _month_starts(start, end):
            me = min(_month_end(ms), end)
            d0 = max(start, ms).strftime("%Y%m%d")
            d1 = me.strftime("%Y%m%d")
            path = cache_dir / f"{label}_{d0}_{d1}.parquet"
            if path.exists():
                parts.append(pd.read_parquet(path))
                continue
            df = pd.read_sql(
                sql,
                conn,
                params={date_bind_start: d0, date_bind_end: d1},
            )
            df.columns = [c.upper() for c in df.columns]
            if len(df):
                df.to_parquet(path, index=False)
                parts.append(df)
            print(f"  [{label}] {d0}->{d1} rows={len(df):,}", flush=True)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_profit_notice_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """业绩预告 long: symbol, report_period, known_date, np_mid, source=notice."""
    own = cache_dir is None
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="sue_notice_"))
    root.mkdir(parents=True, exist_ok=True)
    sql = """
        SELECT S_INFO_WINDCODE, S_PROFITNOTICE_PERIOD, S_PROFITNOTICE_DATE,
               S_PROFITNOTICE_FIRSTANNDATE,
               S_PROFITNOTICE_NETPROFITMIN, S_PROFITNOTICE_NETPROFITMAX
        FROM wind.ASHAREPROFITNOTICE
        WHERE NVL(S_PROFITNOTICE_FIRSTANNDATE, S_PROFITNOTICE_DATE) >= :d0
          AND NVL(S_PROFITNOTICE_FIRSTANNDATE, S_PROFITNOTICE_DATE) <= :d1
    """
    try:
        raw = _read_sql_monthly(
            sql, start, end, date_bind_start="d0", date_bind_end="d1",
            cache_dir=root, label="notice",
        )
        if raw.empty:
            return pd.DataFrame(
                columns=["symbol", "report_period", "known_date", "np_mid", "source"]
            )
        raw = raw[_a_share_mask(raw["S_INFO_WINDCODE"])].copy()
        known = raw["S_PROFITNOTICE_FIRSTANNDATE"].fillna(raw["S_PROFITNOTICE_DATE"])
        mn = pd.to_numeric(raw["S_PROFITNOTICE_NETPROFITMIN"], errors="coerce")
        mx = pd.to_numeric(raw["S_PROFITNOTICE_NETPROFITMAX"], errors="coerce")
        mid = (mn + mx) / 2.0
        mid = mid.fillna(mn).fillna(mx) * NOTICE_NP_TO_YUAN
        out = pd.DataFrame(
            {
                "symbol": raw["S_INFO_WINDCODE"].astype(str),
                "report_period": raw["S_PROFITNOTICE_PERIOD"].astype(str),
                "known_date": pd.to_datetime(known.astype(str), format="%Y%m%d", errors="coerce"),
                "np_mid": mid,
                "source": "notice",
            }
        )
        return out.dropna(subset=["known_date", "report_period"]).sort_values(
            ["symbol", "report_period", "known_date"]
        )
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)


def load_profit_express_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """业绩快报 long."""
    own = cache_dir is None
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="sue_express_"))
    root.mkdir(parents=True, exist_ok=True)
    sql = """
        SELECT S_INFO_WINDCODE, REPORT_PERIOD, ANN_DT, ACTUAL_ANN_DT,
               NET_PROFIT_EXCL_MIN_INT_INC, EPS_DILUTED
        FROM wind.ASHAREPROFITEXPRESS
        WHERE NVL(ACTUAL_ANN_DT, ANN_DT) >= :d0
          AND NVL(ACTUAL_ANN_DT, ANN_DT) <= :d1
    """
    try:
        raw = _read_sql_monthly(
            sql, start, end, date_bind_start="d0", date_bind_end="d1",
            cache_dir=root, label="express",
        )
        if raw.empty:
            return pd.DataFrame(
                columns=["symbol", "report_period", "known_date", "np_mid", "eps", "source"]
            )
        raw = raw[_a_share_mask(raw["S_INFO_WINDCODE"])].copy()
        known = raw["ACTUAL_ANN_DT"].fillna(raw["ANN_DT"])
        out = pd.DataFrame(
            {
                "symbol": raw["S_INFO_WINDCODE"].astype(str),
                "report_period": raw["REPORT_PERIOD"].astype(str),
                "known_date": pd.to_datetime(known.astype(str), format="%Y%m%d", errors="coerce"),
                "np_mid": pd.to_numeric(raw["NET_PROFIT_EXCL_MIN_INT_INC"], errors="coerce"),
                "eps": pd.to_numeric(raw["EPS_DILUTED"], errors="coerce"),
                "source": "express",
            }
        )
        return out.dropna(subset=["known_date", "report_period"]).sort_values(
            ["symbol", "report_period", "known_date"]
        )
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)


def load_income_ann_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """正式报告 long (合并报表)."""
    own = cache_dir is None
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="sue_income_"))
    root.mkdir(parents=True, exist_ok=True)
    sql = f"""
        SELECT S_INFO_WINDCODE, REPORT_PERIOD, ANN_DT, ACTUAL_ANN_DT,
               NET_PROFIT_EXCL_MIN_INT_INC, S_FA_EPS_BASIC
        FROM wind.ASHAREINCOME
        WHERE STATEMENT_TYPE = '{STATEMENT_CONSOLIDATED}'
          AND NVL(ACTUAL_ANN_DT, ANN_DT) >= :d0
          AND NVL(ACTUAL_ANN_DT, ANN_DT) <= :d1
    """
    try:
        raw = _read_sql_monthly(
            sql, start, end, date_bind_start="d0", date_bind_end="d1",
            cache_dir=root, label="income",
        )
        if raw.empty:
            return pd.DataFrame(
                columns=["symbol", "report_period", "known_date", "np_mid", "eps", "source"]
            )
        raw = raw[_a_share_mask(raw["S_INFO_WINDCODE"])].copy()
        known = raw["ACTUAL_ANN_DT"].fillna(raw["ANN_DT"])
        out = pd.DataFrame(
            {
                "symbol": raw["S_INFO_WINDCODE"].astype(str),
                "report_period": raw["REPORT_PERIOD"].astype(str),
                "known_date": pd.to_datetime(known.astype(str), format="%Y%m%d", errors="coerce"),
                "np_mid": pd.to_numeric(raw["NET_PROFIT_EXCL_MIN_INT_INC"], errors="coerce"),
                "eps": pd.to_numeric(raw["S_FA_EPS_BASIC"], errors="coerce"),
                "source": "income",
            }
        )
        return out.dropna(subset=["known_date", "report_period"]).sort_values(
            ["symbol", "report_period", "known_date"]
        )
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)


def load_consensus_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cycle_typ: str = CONSENSUS_FY1,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """分析师一致预期 (FY1) long: symbol, est_dt, report_period, eps_avg, np_avg."""
    own = cache_dir is None
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="sue_cons_"))
    root.mkdir(parents=True, exist_ok=True)
    sql = f"""
        SELECT S_INFO_WINDCODE, EST_DT, EST_REPORT_DT, EPS_AVG, NET_PROFIT_AVG
        FROM wind.ASHARECONSENSUSDATA
        WHERE CONSEN_DATA_CYCLE_TYP = '{cycle_typ}'
          AND EST_DT >= :d0 AND EST_DT <= :d1
    """
    try:
        raw = _read_sql_monthly(
            sql, start, end, date_bind_start="d0", date_bind_end="d1",
            cache_dir=root, label="consensus",
        )
        if raw.empty:
            return pd.DataFrame(
                columns=["symbol", "est_dt", "report_period", "eps_avg", "np_avg"]
            )
        raw = raw[_a_share_mask(raw["S_INFO_WINDCODE"])].copy()
        out = pd.DataFrame(
            {
                "symbol": raw["S_INFO_WINDCODE"].astype(str),
                "est_dt": pd.to_datetime(raw["EST_DT"].astype(str), format="%Y%m%d", errors="coerce"),
                "report_period": raw["EST_REPORT_DT"].astype(str),
                "eps_avg": pd.to_numeric(raw["EPS_AVG"], errors="coerce"),
                "np_avg": pd.to_numeric(raw["NET_PROFIT_AVG"], errors="coerce")
                * CONSENSUS_NP_TO_YUAN,
            }
        )
        return out.dropna(subset=["est_dt", "report_period"]).sort_values(
            ["symbol", "report_period", "est_dt"]
        )
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)


def build_earliest_known_timeline(
    notice: pd.DataFrame,
    express: pd.DataFrame,
    income: pd.DataFrame,
) -> pd.DataFrame:
    """Per (symbol, report_period): earliest known_date + best available NP/EPS.

    Priority for levels once known: income > express > notice (more accurate),
    but **known_date** is always the earliest public disclosure among the three.
    """
    frames = []
    if notice is not None and not notice.empty:
        n = notice.copy()
        n["eps"] = np.nan
        frames.append(n[["symbol", "report_period", "known_date", "np_mid", "eps", "source"]])
    if express is not None and not express.empty:
        frames.append(
            express[["symbol", "report_period", "known_date", "np_mid", "eps", "source"]]
        )
    if income is not None and not income.empty:
        frames.append(
            income[["symbol", "report_period", "known_date", "np_mid", "eps", "source"]]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "report_period",
                "known_date",
                "np_mid",
                "eps",
                "first_source",
                "best_source",
            ]
        )

    all_ev = pd.concat(frames, ignore_index=True)
    all_ev = all_ev.dropna(subset=["symbol", "report_period", "known_date"])
    all_ev["report_period"] = all_ev["report_period"].astype(str)

    # earliest disclosure date + first source
    first = (
        all_ev.sort_values(["symbol", "report_period", "known_date"])
        .groupby(["symbol", "report_period"], as_index=False)
        .first()
        .rename(columns={"known_date": "known_date", "source": "first_source"})
    )

    # best NP/EPS as of / after earliest: prefer income, then express, then notice
    rank = {"income": 3, "express": 2, "notice": 1}
    all_ev["src_rank"] = all_ev["source"].map(rank).fillna(0)
    best = (
        all_ev.sort_values(["symbol", "report_period", "src_rank", "known_date"])
        .groupby(["symbol", "report_period"], as_index=False)
        .last()
        .rename(columns={"source": "best_source", "np_mid": "np_best", "eps": "eps_best"})
    )

    out = first.merge(
        best[["symbol", "report_period", "np_best", "eps_best", "best_source"]],
        on=["symbol", "report_period"],
        how="left",
    )
    # At earliest known date, use the best figure available *on that date*
    # (may still be notice mid); upgrade NP when better source arrives via event stream.
    # For PIT daily panels we rebuild from event stream; timeline stores first known + best final.
    out["np_mid"] = out["np_best"]
    out["eps"] = out["eps_best"]
    return out[
        [
            "symbol",
            "report_period",
            "known_date",
            "np_mid",
            "eps",
            "first_source",
            "best_source",
        ]
    ].sort_values(["symbol", "known_date"])


def build_pit_event_stream(
    notice: pd.DataFrame,
    express: pd.DataFrame,
    income: pd.DataFrame,
) -> pd.DataFrame:
    """Chronological disclosure events with running best NP/EPS per period.

    Each row: symbol, report_period, known_date, np_mid, eps, source
    sorted by known_date. Later better sources overwrite levels for that period.
    """
    parts = []
    if notice is not None and not notice.empty:
        n = notice.copy()
        n["eps"] = np.nan
        parts.append(n[["symbol", "report_period", "known_date", "np_mid", "eps", "source"]])
    if express is not None and not express.empty:
        parts.append(
            express[["symbol", "report_period", "known_date", "np_mid", "eps", "source"]]
        )
    if income is not None and not income.empty:
        parts.append(
            income[["symbol", "report_period", "known_date", "np_mid", "eps", "source"]]
        )
    if not parts:
        return pd.DataFrame(
            columns=["symbol", "report_period", "known_date", "np_mid", "eps", "source"]
        )
    ev = pd.concat(parts, ignore_index=True)
    ev["report_period"] = ev["report_period"].astype(str)
    ev = ev.dropna(subset=["known_date"]).sort_values(
        ["symbol", "report_period", "known_date", "source"]
    )
    return ev


def load_sue_raw_bundle(
    start: dt.datetime,
    end: dt.datetime,
    *,
    history_start: Optional[dt.datetime] = None,
    cache_root: Optional[Path] = None,
    keep_cache: bool = False,
) -> dict:
    """Load notice/express/income/consensus for SUE construction."""
    hist = history_start or (start - dt.timedelta(days=800))
    own = cache_root is None
    root = Path(cache_root) if cache_root else Path(tempfile.mkdtemp(prefix="sue_bundle_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"SUE Oracle cache: {root} | hist {hist.date()} -> {end.date()}", flush=True)
    try:
        notice = load_profit_notice_long(hist, end, cache_dir=root / "notice", keep_cache=True)
        express = load_profit_express_long(hist, end, cache_dir=root / "express", keep_cache=True)
        income = load_income_ann_long(hist, end, cache_dir=root / "income", keep_cache=True)
        consensus = load_consensus_long(hist, end, cache_dir=root / "consensus", keep_cache=True)
        timeline = build_earliest_known_timeline(notice, express, income)
        events = build_pit_event_stream(notice, express, income)
        print(
            f"SUE bundle: notice={len(notice):,} express={len(express):,} "
            f"income={len(income):,} consensus={len(consensus):,} "
            f"timeline={len(timeline):,} events={len(events):,}",
            flush=True,
        )
        return {
            "notice": notice,
            "express": express,
            "income": income,
            "consensus": consensus,
            "timeline": timeline,
            "events": events,
            "history_start": hist,
            "end": end,
        }
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)
            print(f"Cleared SUE cache: {root}", flush=True)
