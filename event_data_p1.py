"""P1 corporate-event loaders (Wind Oracle).

Primary sources (coverage OK):
  - ASHAREMJRHOLDERTRADE — 大股东增持/减持
  - ASHAREINSIDERTRADE — 董监高增减持

Blocked / incomplete:
  - ASHARESTOCKINCENTIVEIMPLEMENT — only ~444 rows (2024-01..02) in current Wind feed
"""

from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from COMMON_CONST import DATA_DB_WIND


def _init_oracle():
    import oracledb

    oracledb.init_oracle_client(lib_dir=None)
    return oracledb


def _a_share(code: pd.Series) -> pd.Series:
    return code.astype(str).str[0].isin(("0", "3", "6"))


def load_major_holder_trade_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """大股东交易：symbol, known_date, transact_type, qty, qty_ratio."""
    oracledb = _init_oracle()
    own = cache_dir is None
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="p1_mjr_"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"mjr_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    try:
        if path.exists():
            return pd.read_parquet(path)
        sql = """
            SELECT S_INFO_WINDCODE, ANN_DT, TRANSACT_TYPE,
                   TRANSACT_QUANTITY, TRANSACT_QUANTITY_RATIO
            FROM wind.ASHAREMJRHOLDERTRADE
            WHERE ANN_DT >= :d0 AND ANN_DT <= :d1
        """
        with oracledb.connect(**DATA_DB_WIND) as conn:
            df = pd.read_sql(
                sql, conn, params={"d0": start.strftime("%Y%m%d"), "d1": end.strftime("%Y%m%d")}
            )
        df.columns = [c.upper() for c in df.columns]
        if df.empty:
            return pd.DataFrame(
                columns=["symbol", "known_date", "transact_type", "qty", "qty_ratio"]
            )
        df = df[_a_share(df["S_INFO_WINDCODE"])].copy()
        out = pd.DataFrame(
            {
                "symbol": df["S_INFO_WINDCODE"].astype(str),
                "known_date": pd.to_datetime(
                    df["ANN_DT"].astype(str), format="%Y%m%d", errors="coerce"
                ),
                "transact_type": df["TRANSACT_TYPE"].astype(str),
                "qty": pd.to_numeric(df["TRANSACT_QUANTITY"], errors="coerce"),
                "qty_ratio": pd.to_numeric(df["TRANSACT_QUANTITY_RATIO"], errors="coerce"),
            }
        ).dropna(subset=["known_date"])
        out.to_parquet(path, index=False)
        print(f"P1 major-holder trades: {len(out):,}", flush=True)
        return out
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)


def load_insider_trade_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_dir: Optional[Path] = None,
    keep_cache: bool = False,
) -> pd.DataFrame:
    """董监高交易：symbol, known_date, change_volume (signed)."""
    oracledb = _init_oracle()
    own = cache_dir is None
    root = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="p1_ins_"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"insider_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    try:
        if path.exists():
            return pd.read_parquet(path)
        sql = """
            SELECT S_INFO_WINDCODE, NVL(ACTUAL_ANN_DT, ANN_DT) AS KNOWN_DT,
                   CHANGE_VOLUME
            FROM wind.ASHAREINSIDERTRADE
            WHERE NVL(ACTUAL_ANN_DT, ANN_DT) >= :d0
              AND NVL(ACTUAL_ANN_DT, ANN_DT) <= :d1
        """
        with oracledb.connect(**DATA_DB_WIND) as conn:
            df = pd.read_sql(
                sql, conn, params={"d0": start.strftime("%Y%m%d"), "d1": end.strftime("%Y%m%d")}
            )
        df.columns = [c.upper() for c in df.columns]
        if df.empty:
            return pd.DataFrame(columns=["symbol", "known_date", "change_volume"])
        df = df[_a_share(df["S_INFO_WINDCODE"])].copy()
        out = pd.DataFrame(
            {
                "symbol": df["S_INFO_WINDCODE"].astype(str),
                "known_date": pd.to_datetime(
                    df["KNOWN_DT"].astype(str), format="%Y%m%d", errors="coerce"
                ),
                "change_volume": pd.to_numeric(df["CHANGE_VOLUME"], errors="coerce"),
            }
        ).dropna(subset=["known_date"])
        out.to_parquet(path, index=False)
        print(f"P1 insider trades: {len(out):,}", flush=True)
        return out
    finally:
        if own and not keep_cache and root.exists():
            shutil.rmtree(root, ignore_errors=True)


def aggregate_daily_holder_signal(mjr: pd.DataFrame) -> pd.DataFrame:
    """Daily net increase ratio: sum(增持 ratio) - sum(减持 ratio)."""
    if mjr is None or mjr.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])
    df = mjr.copy()
    sign = np.where(df["transact_type"].str.contains("增持", na=False), 1.0, np.nan)
    sign = np.where(df["transact_type"].str.contains("减持", na=False), -1.0, sign)
    df["signed_ratio"] = sign * df["qty_ratio"].fillna(0.0)
    # if ratio missing, use sign only
    miss = df["qty_ratio"].isna() & np.isfinite(sign)
    df.loc[miss, "signed_ratio"] = sign[miss]
    g = (
        df.dropna(subset=["signed_ratio"])
        .groupby(["symbol", "known_date"], as_index=False)["signed_ratio"]
        .sum()
        .rename(columns={"signed_ratio": "surprise"})
    )
    return g


def aggregate_daily_insider_signal(ins: pd.DataFrame) -> pd.DataFrame:
    """Daily net insider volume change (positive = buy)."""
    if ins is None or ins.empty:
        return pd.DataFrame(columns=["symbol", "known_date", "surprise"])
    g = (
        ins.dropna(subset=["change_volume"])
        .groupby(["symbol", "known_date"], as_index=False)["change_volume"]
        .sum()
        .rename(columns={"change_volume": "surprise"})
    )
    return g
