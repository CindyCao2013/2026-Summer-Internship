"""Unified minute-bar store — pure DolphinDB version (no local disk cache).

All data is fetched on-demand from DDB. A lightweight in-memory LRU cache avoids
repeated queries for the same date-range/columns within a session.

DDB query patterns follow project conventions (``intraday_lib``, ``l2_data_loaders``)
and DolphinDB financial quant docs:
  - loadTable + partition filter (``Date between start : end``)
  - server-side column pruning and trading-hours filter
  - ``context by Symbol csort Date, Bartime`` for window/rolling (factor pushdown)
  - assign query result to a variable before returning to client

Streaming factor patterns (reactive state engine, m-series) are documented at:
https://docs.dolphindb.com/zh/tutorials/str_comp_fin_quant.html
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from COMMON_CONST import DATA_DB_CONN
from core.ddb.connection import get_ddb_session

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_HISTORY_START = pd.Timestamp("2020-01-01")
DDB_TABLE = ("dfs://QV_Trade_to_MinuteBar", "Stock_one_minute")
MAX_RETRIES = 3
RETRY_SLEEP_SEC = 2.0
DEFAULT_MEM_CACHE_SIZE = 50

DateLike = Union[str, pd.Timestamp, dt.datetime, dt.date]

COLUMN_MAP = {
    "Symbol": "symbol",
    "Date": "date",
    "Bartime": "bartime",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
    "Adjfactor": "adjfactor",
    "Active_buy_volume": "active_buy_vol",
    "Active_sell_volume": "active_sell_vol",
    "Active_buy_amount": "active_buy_amt",
    "Active_sell_amount": "active_sell_amt",
    "Active_buy_count": "active_buy_count",
    "Active_sell_count": "active_sell_count",
    "Bid_cancel_volume": "bid_cancel_vol",
    "Ask_cancel_volume": "ask_cancel_vol",
    "Bid_cancel_count": "bid_cancel_count",
    "Ask_cancel_count": "ask_cancel_count",
}

FIELD_ALIASES = {
    "active_buy_cnt": "active_buy_count",
    "active_sell_cnt": "active_sell_count",
    "bid_cancel_cnt": "bid_cancel_count",
    "ask_cancel_cnt": "ask_cancel_count",
    "Symbol": "symbol",
    "Date": "date",
    "Bartime": "bartime",
}

CANONICAL_COLUMNS: List[str] = list(COLUMN_MAP.values())
REV_MAP = {v: k for k, v in COLUMN_MAP.items()}

DDB_SELECT_COLUMNS = [
    "Symbol",
    "Date",
    "Bartime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Amount",
    "Active_buy_volume",
    "Active_sell_volume",
    "Active_buy_amount",
    "Active_sell_amount",
    "Active_buy_count",
    "Active_sell_count",
    "Bid_cancel_volume",
    "Bid_cancel_count",
    "Ask_cancel_volume",
    "Ask_cancel_count",
    "Adjfactor",
]

# Push continuous-auction filter to DDB (same as smart_money_active_v2_builder).
TRADING_HOURS_WHERE = """
  and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
    or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))"""


def _ddb_retry_settings() -> tuple[int, float]:
    try:
        import factor_config as cfg

        return (
            int(getattr(cfg, "DDB_MAX_RETRIES", MAX_RETRIES)),
            float(getattr(cfg, "DDB_QUERY_TIMEOUT", 120)),
        )
    except Exception:  # noqa: BLE001
        return MAX_RETRIES, 120.0


def _env_history_start() -> pd.Timestamp:
    raw = os.environ.get("MINUTE_BAR_HISTORY_START") or os.environ.get(
        "MINUTE_BAR_STORE_START"
    )
    if raw:
        return pd.Timestamp(raw)
    return DEFAULT_HISTORY_START


def _to_ts(value: DateLike) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def to_wind_code(symbol) -> str:
    """Normalize DDB / Wind symbol to Wind code (e.g. 600000.SH)."""
    s = str(symbol).strip()
    if not s or s.lower() == "nan":
        return s
    if "." in s:
        return s
    if s.startswith(("5", "6", "9")):
        return f"{s}.SH"
    return f"{s}.SZ"


def filter_a_share(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    *,
    include_bj: bool = False,
) -> pd.DataFrame:
    """Keep A-shares (0/3/6). BJ optional."""
    if df.empty:
        return df
    sym = df[symbol_col].astype(str)
    mask = sym.str[0].isin(("0", "3", "6"))
    if include_bj:
        mask = mask | sym.str.contains(r"\.BJ$|BJ$", regex=True, na=False)
    return df.loc[mask].copy()


def normalize_bartime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine date + bartime into a full timestamp on ``bartime``."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    bt = pd.to_datetime(out["bartime"])
    out["bartime"] = out["date"] + (bt - bt.dt.normalize())
    return out


def apply_trading_hours(
    df: pd.DataFrame,
    *,
    am_start: dt.time = dt.time(9, 30),
    am_end: dt.time = dt.time(11, 30),
    pm_start: dt.time = dt.time(13, 0),
    pm_end: dt.time = dt.time(15, 0),
    bartime_col: str = "bartime",
) -> pd.DataFrame:
    """Filter to continuous-auction-like windows (defaults match ActiveV2 builders)."""
    if df.empty:
        return df
    t = pd.to_datetime(df[bartime_col]).dt.time
    am = (t >= am_start) & (t <= am_end)
    pm = (t >= pm_start) & (t <= pm_end)
    return df.loc[am | pm].copy()


def resolve_fields(fields: Optional[Sequence[str]]) -> Optional[List[str]]:
    if fields is None:
        return None
    out: List[str] = []
    for f in fields:
        out.append(FIELD_ALIASES.get(f, f))
    return out


def _resolve_ddb_columns(fields: Optional[List[str]]) -> List[str]:
    if fields is None:
        return list(DDB_SELECT_COLUMNS)
    cols: List[str] = []
    for f in fields:
        if f in REV_MAP:
            cols.append(REV_MAP[f])
        elif f in COLUMN_MAP:
            cols.append(f)
        else:
            cols.append(f)
    for key in ("Symbol", "Date", "Bartime"):
        if key not in cols:
            cols.insert(0, key)
    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class MinuteBarStore:
    """On-demand DDB minute bar fetcher with in-memory LRU cache.

    No local Parquet files are ever written or read.
    """

    def __init__(
        self,
        ddb_conn: Optional[dict] = None,
        start_date: Optional[DateLike] = None,
        *,
        include_bj: bool = False,
        session=None,
        memory_cache_size: int = DEFAULT_MEM_CACHE_SIZE,
        # Accepted for backward compatibility; ignored (no local cache).
        cache_root=None,
    ):
        if cache_root is not None:
            print(
                "[MinuteBarStore] cache_root is deprecated and ignored "
                "(pure DDB mode)",
                flush=True,
            )
        self.ddb_conn = ddb_conn or dict(DATA_DB_CONN)
        self.history_start = (
            _to_ts(start_date) if start_date is not None else _env_history_start()
        )
        self.include_bj = include_bj
        self._session = session
        self._own_session = False
        self._mem_cache: OrderedDict[
            Tuple[
                pd.Timestamp,
                pd.Timestamp,
                Optional[Tuple[str, ...]],
                Optional[Tuple[str, ...]],
                bool,
            ],
            pd.DataFrame,
        ] = OrderedDict()
        self._mem_cache_max = max(1, memory_cache_size)

    def get_data(
        self,
        start_date: DateLike,
        end_date: DateLike,
        symbols: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        force_reload: bool = False,
        *,
        trading_hours_only: bool = False,
    ) -> pd.DataFrame:
        """Return canonical minute DataFrame for ``[start_date, end_date]``.

        Always queries DDB directly; caches results in memory unless force_reload=True.

        When ``trading_hours_only=True``, the continuous-auction filter is pushed
        to DolphinDB (``second(Bartime)``), reducing network transfer.
        """
        start = _to_ts(start_date)
        end = _to_ts(end_date)
        if end < start:
            raise ValueError(f"end_date {end} < start_date {start}")
        if start < self.history_start:
            print(
                f"[MinuteBarStore] clamp start {start.date()} → "
                f"history_start {self.history_start.date()}",
                flush=True,
            )
            start = self.history_start

        field_list = resolve_fields(fields)
        sym_key = tuple(sorted(to_wind_code(s) for s in symbols)) if symbols else None
        cache_key = (
            start,
            end,
            tuple(field_list) if field_list else None,
            sym_key,
            trading_hours_only,
        )

        if not force_reload and cache_key in self._mem_cache:
            self._mem_cache.move_to_end(cache_key)
            return self._mem_cache[cache_key].copy()

        t0 = time.perf_counter()
        script = self._build_ddb_script(
            start,
            end,
            symbols,
            fields=field_list,
            trading_hours_only=trading_hours_only,
        )
        raw = self._run_with_retry(script)
        df = self.normalize_raw(raw)
        print(
            f"[MinuteBarStore] pulled {start.date()}→{end.date()} "
            f"rows={len(df):,} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

        while len(self._mem_cache) >= self._mem_cache_max:
            self._mem_cache.popitem(last=False)
        self._mem_cache[cache_key] = df.copy()
        return df

    def run_script(self, script: str) -> pd.DataFrame:
        """Execute arbitrary DolphinDB script (factor pushdown / aggregation).

        Use for server-side ``context by`` / m-series / cum-series computations
        so only narrow results cross the network. See intraday_lib.get_ret_matrix
        and https://docs.dolphindb.com/zh/tutorials/str_comp_fin_quant.html
        """
        raw = self._run_with_retry(script)
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        return pd.DataFrame(raw)

    def normalize_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map DDB columns → canonical schema + Wind codes + A-share filter."""
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df = raw.copy()
        rename = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
        df = df.rename(columns=rename)
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        df = df[keep].copy()

        df["symbol"] = df["symbol"].map(to_wind_code)
        df = filter_a_share(df, include_bj=self.include_bj)
        df = normalize_bartime(df)
        df = df.sort_values(["symbol", "bartime"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _connect(self):
        import dolphindb as ddb

        s = ddb.session()
        s.connect(**self.ddb_conn)
        return s

    def _get_session(self):
        if self._session is not None:
            return self._session
        if self.ddb_conn == DATA_DB_CONN:
            self._session = get_ddb_session(reuse=True)
            self._own_session = False
        else:
            self._session = self._connect()
            self._own_session = True
        return self._session

    def _close_if_owned(self) -> None:
        if self._own_session and self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None
            self._own_session = False

    def _run_with_retry(self, script: str) -> pd.DataFrame:
        max_retries, _timeout = _ddb_retry_settings()
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1 and self._own_session:
                    self._close_if_owned()
                sess = self._get_session()
                df = sess.run(script)
                if df is None:
                    return pd.DataFrame()
                return pd.DataFrame(df)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(
                    f"[MinuteBarStore] DDB attempt {attempt}/{max_retries} failed: {exc}",
                    flush=True,
                )
                if attempt < max_retries:
                    time.sleep(RETRY_SLEEP_SEC * attempt)
        raise RuntimeError(
            f"DolphinDB pull failed after {max_retries} retries"
        ) from last_exc

    def _build_ddb_script(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        *,
        trading_hours_only: bool = False,
    ) -> str:
        """Construct DDB SQL with column pruning and partition filter."""
        s = start.strftime("%Y.%m.%d")
        e = end.strftime("%Y.%m.%d")
        db, table = DDB_TABLE
        col_str = ", ".join(_resolve_ddb_columns(fields))

        script = f"""
t = loadTable('{db}', '{table}')
result = select {col_str}
from t
where Date between {s} : {e}
"""
        if trading_hours_only:
            script += TRADING_HOURS_WHERE
        if symbols:
            sym_str = ", ".join(f'"{sym}"' for sym in symbols)
            script += f"\n  and Symbol in ({sym_str})"
        script += "\nresult"
        return script

    def __enter__(self) -> "MinuteBarStore":
        return self

    def __exit__(self, *exc) -> None:
        self._close_if_owned()


def get_default_store(**kwargs) -> MinuteBarStore:
    """Factory using factor_config / env defaults when available."""
    kwargs.pop("cache_root", None)  # deprecated
    start_date = kwargs.pop("start_date", None)
    try:
        import factor_config as cfg

        if start_date is None and getattr(cfg, "MINUTE_BAR_HISTORY_START", None):
            start_date = cfg.MINUTE_BAR_HISTORY_START
    except Exception:  # noqa: BLE001
        pass
    return MinuteBarStore(start_date=start_date, **kwargs)
