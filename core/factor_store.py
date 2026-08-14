"""DolphinDB factor store — narrow vertical + optional wide TSDB panels.

Implements storage/query patterns from DolphinDB *因子计算最佳实践* §5:
https://docs.dolphindb.com/zh/tutorials/best_practice_for_factor_calculation.html

Storage tiers (official recommendation summary):
  - **Vertical TSDB** (default write path): flexible schema, new factors without DDL.
    Schema: tradetime, symbol, factorname, value
    Partition: VALUE(month) + VALUE(factorname), sortKey: symbol+tradetime
  - **Wide TSDB** (hot read path): best panel/backtest query; one row per (tradetime, factorname)
    with symbols as columns. Materialize from vertical for stable factor universes.

Query patterns (§5.2–5.3):
  - Point:   where symbol=X and factorname=Y and tradetime=Z
  - Series:  where symbol=X and factorname=Y and tradetime between ...
  - Panel:   pivot by tradetime, symbol  OR  select from wide table
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional, Union

import pandas as pd

from core.ddb.connection import get_ddb_session

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

NARROW_COLUMNS = ("tradetime", "symbol", "factorname", "value")

# Run once on cluster (adjust db/table names via factor_config).
DDL_VERTICAL_TSDB = """
// Minute-factor vertical store (TSDB) — §5.1 narrow table pattern
dbPath = "dfs://Factor_DB_DEV"
tableName = "intraday_factor_vertical"

if(existsDatabase(dbPath)){
    dropDatabase(dbPath)
}
db = database(dbPath, VALUE, 2020.01M..2030.12M)

schema = table(
    1:0,
    `tradetime`symbol`factorname`value,
    [TIMESTAMP, SYMBOL, SYMBOL, DOUBLE]
)

db.createPartitionedTable(
    schema,
    tableName,
    `tradetime`factorname,
    sortColumns=`symbol`tradetime,
    keepDuplicates=LAST
)
"""

DDL_WIDE_TSDB_TEMPLATE = """
// Wide panel per factor (TSDB) — §5.1 wide table; columns = symbol universe
// Generate column list from your universe before running.
dbPath = "dfs://Factor_DB_DEV"
tableName = "intraday_factor_wide"
// ... factor-specific wide tables are usually materialized via pivot + append
"""


def _to_ddb_date(value: DateLike) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def _resolve_store_paths() -> tuple[str, str]:
    try:
        import factor_config as cfg

        db = getattr(cfg, "FACTOR_STORE_DB", "dfs://Factor_DB_DEV")
        table = getattr(cfg, "FACTOR_STORE_TABLE_VERTICAL", "intraday_factor_vertical")
        return str(db), str(table)
    except Exception:  # noqa: BLE001
        return "dfs://Factor_DB_DEV", "intraday_factor_vertical"


def normalize_narrow(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical narrow schema for DDB append."""
    if df.empty:
        return pd.DataFrame(columns=list(NARROW_COLUMNS))
    out = df.copy()
    if "bartime" in out.columns and "tradetime" not in out.columns:
        out = out.rename(columns={"bartime": "tradetime"})
    missing = [c for c in NARROW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"narrow factor table missing columns: {missing}")
    out = out[list(NARROW_COLUMNS)].copy()
    out["tradetime"] = pd.to_datetime(out["tradetime"])
    out["symbol"] = out["symbol"].astype(str)
    out["factorname"] = out["factorname"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).reset_index(drop=True)


def append_narrow(
    df: pd.DataFrame,
    *,
    session=None,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
) -> int:
    """Append narrow factor rows to distributed vertical table (``append!``)."""
    narrow = normalize_narrow(df)
    if narrow.empty:
        return 0

    db, table = _resolve_store_paths()
    if db_path is not None:
        db = db_path
    if table_name is not None:
        table = table_name
    own = session is None
    s = session or get_ddb_session()
    try:
        s.upload({"narrow_upload": narrow})
        script = f"""
t = loadTable('{db}', '{table}')
append!(t, narrow_upload)
nrow(narrow_upload)
"""
        n = s.run(script)
        return int(n) if n is not None else len(narrow)
    finally:
        if own:
            s.close()


def query_narrow_script(
    factor_name: str,
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
) -> str:
    """Build §5.2 point/series query for one factor (vertical TSDB)."""
    db, table = db_path or _resolve_store_paths()[0], table_name or _resolve_store_paths()[1]
    s, e = _to_ddb_date(start), _to_ddb_date(end)
    sym_filter = ""
    if symbols:
        sym_str = ", ".join(f'"{sym}"' for sym in symbols)
        sym_filter = f" and symbol in ({sym_str})"
    return f"""
t = loadTable('{db}', '{table}')
result = select tradetime, symbol, factorname, value
from t
where factorname = "{factor_name}"
  and tradetime between {s} : {e}
  {sym_filter}
select * from result
"""


def query_narrow(
    factor_name: str,
    start: DateLike,
    end: DateLike,
    *,
    symbols: Optional[List[str]] = None,
    session=None,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load narrow factor rows from vertical store."""
    own = session is None
    s = session or get_ddb_session()
    try:
        raw = s.run(
            query_narrow_script(
                factor_name, start, end, symbols=symbols, db_path=db_path, table_name=table_name
            )
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=list(NARROW_COLUMNS))
        return pd.DataFrame(raw)
    finally:
        if own:
            s.close()


def query_panel_script(
    factor_name: str,
    start: DateLike,
    end: DateLike,
    *,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
) -> str:
    """Build §5.3 panel query via ``pivot by`` (vertical → Date×Symbol wide)."""
    db, table = db_path or _resolve_store_paths()[0], table_name or _resolve_store_paths()[1]
    s, e = _to_ddb_date(start), _to_ddb_date(end)
    return f"""
t = loadTable('{db}', '{table}')
result = select value
from t
where factorname = "{factor_name}"
  and tradetime between {s} : {e}
pivot by tradetime, symbol
"""


def query_panel(
    factor_name: str,
    start: DateLike,
    end: DateLike,
    *,
    session=None,
    db_path: Optional[str] = None,
    table_name: Optional[str] = None,
) -> pd.DataFrame:
    """Return tradetime × symbol panel for one factor (pivot on server)."""
    own = session is None
    s = session or get_ddb_session()
    try:
        raw = s.run(
            query_panel_script(
                factor_name, start, end, db_path=db_path, table_name=table_name
            )
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        panel = pd.DataFrame(raw)
        if "tradetime" in panel.columns:
            panel = panel.set_index("tradetime")
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()
    finally:
        if own:
            s.close()
