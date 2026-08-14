"""ClickHouse extraction for normalized per-trade-amount factors.

The module deliberately keeps raw Tick rows inside ClickHouse.  It exposes two
server-side aggregation stages:

1. a symbol-day scale primitive (amount totals, counts, mean and quantiles);
2. a symbol-day dynamic aggregation joined to ``scale_rows`` supplied through
   clickhouse-connect ``ExternalData``.

Both stages use the strict trade definition frozen in :mod:`ch_tick`:

* SSE: ``Type = 'T'`` and ``ifNull(Amount, Price * Volume)``;
* SZSE: ``Type = '011'``, positive bid/ask order numbers, and
  ``Price * Volume``;
* every date: ``09:30:00 <= ExchTime < 15:00:01``;
* conservative SSE/SZSE A-share code predicates.

All bucket boundaries are lower-exclusive and upper-inclusive.  There is no
return, direction, backtest-fee, or portfolio logic in this data-access layer.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import pandas as pd


DateLike = Union[str, date, datetime, pd.Timestamp]
Grid = Sequence[Tuple[float, float]]

A0_LOWER_RMB = 40_000.0
A0_UPPER_RMB = 200_000.0
A1_ADV_BPS_CANDIDATE_GRID: Tuple[Tuple[float, float], ...] = tuple(
    (lower, upper)
    for lower in (0.5, 1.0, 2.0)
    for upper in (5.0, 10.0, 20.0)
    if lower < upper
)
A2_ATS_MULTIPLE_CANDIDATE_GRID: Tuple[Tuple[float, float], ...] = tuple(
    (lower, upper)
    for lower in (0.25, 0.5, 0.75)
    for upper in (1.5, 2.0, 3.0)
    if lower < upper
)

# Short aliases are useful to callers constructing frozen configurations.
DEFAULT_A1_GRID = A1_ADV_BPS_CANDIDATE_GRID
DEFAULT_A2_GRID = A2_ATS_MULTIPLE_CANDIDATE_GRID

KEY_COLUMNS: Tuple[str, str] = ("symbol", "TradeDate")
DAILY_SCALE_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "TradeDate",
    "total_amount",
    "positive_trade_count",
    "daily_mean_trade_amount",
    "q20",
    "q30",
    "q50",
    "q70",
    "q80",
)
SCALE_ROWS_COLUMNS: Tuple[str, ...] = (
    "symbol",
    "TradeDate",
    "ADV20_lag1",
    "ATS20_lag1",
    "q20",
    "q80",
)
SCALE_ROWS_EXTERNAL_FILE = "scale_rows.csv"
SCALE_ROWS_EXTERNAL_TABLE = "scale_rows"
SCALE_ROWS_EXTERNAL_FORMAT = "CSVWithNames"
SCALE_ROWS_EXTERNAL_STRUCTURE = (
    "symbol String, TradeDate Date, "
    "ADV20_lag1 Nullable(Float64), ATS20_lag1 Nullable(Float64), "
    "q20 Nullable(Float64), q80 Nullable(Float64)"
)

_EXCHANGE_SPECS: Dict[str, Dict[str, str]] = {
    "SSE": {
        "table": "cmds.SSE_AL_TICK_EXG",
        "suffix": ".SH",
        "amount": "ifNull(Amount, Price * Volume)",
        "trade_filter": "AND Type = 'T'",
    },
    "SZSE": {
        "table": "cmds.SZSE_AL_TICK_EXG",
        "suffix": ".SZ",
        "amount": "Price * Volume",
        "trade_filter": (
            "AND Type = '011' "
            "AND BidOrderNo > 0 AND AskOrderNo > 0"
        ),
    },
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SSE_A_SHARE_RE = re.compile(r"^6[0-9]{5}$")
_SZSE_A_SHARE_RE = re.compile(
    r"^(?:000|001|002|003|300|301|302)[0-9]{3}$"
)


def _get_ch_client():
    """Create the production client lazily so unit tests need no database."""
    import clickhouse_connect
    from COMMON_CONST import DATA_DB_HFDATA

    return clickhouse_connect.get_client(**DATA_DB_HFDATA)


def _exchange_name(exchange: str) -> str:
    value = str(exchange).strip().upper()
    aliases = {
        "SH": "SSE",
        ".SH": "SSE",
        "SSE": "SSE",
        "SZ": "SZSE",
        ".SZ": "SZSE",
        "SZSE": "SZSE",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError("exchange must be SSE/.SH or SZSE/.SZ") from exc


def _to_date_str(value: DateLike) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("date cannot be NaT")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.strftime("%Y-%m-%d")


def _date_range(start_date: DateLike, end_date: DateLike) -> Tuple[str, str]:
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    return start, end


def _sql_number(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("SQL boundary must be finite")
    return format(number, ".15g")


def _number_label(value: float) -> str:
    """Return a deterministic SQL-column-safe numeric label."""
    return (
        _sql_number(value)
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )


def _normalize_grid(grid: Grid, name: str) -> Tuple[Tuple[float, float], ...]:
    normalized: List[Tuple[float, float]] = []
    seen = set()
    for pair in grid:
        if len(pair) != 2:
            raise ValueError(f"{name} entries must be (lower, upper) pairs")
        lower, upper = float(pair[0]), float(pair[1])
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"{name} boundaries must be finite")
        if lower < 0 or lower >= upper:
            raise ValueError(
                f"{name} requires nonnegative lower < upper; got {pair}"
            )
        key = (lower, upper)
        if key in seen:
            raise ValueError(f"{name} contains duplicate candidate {pair}")
        seen.add(key)
        normalized.append(key)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return tuple(sorted(normalized))


def a1_selected_amount_column(lower_bps: float, upper_bps: float) -> str:
    """Column name for an A1 ADV20-bps candidate numerator."""
    return (
        f"a1_adv20_bps_l{_number_label(lower_bps)}"
        f"_h{_number_label(upper_bps)}_selected_amount"
    )


def a2_selected_amount_column(lower: float, upper: float) -> str:
    """Column name for an A2 ATS20-multiple candidate numerator."""
    return (
        f"a2_ats20_l{_number_label(lower)}"
        f"_h{_number_label(upper)}_selected_amount"
    )


def dynamic_result_columns(
    a1_grid: Grid = DEFAULT_A1_GRID,
    a2_grid: Grid = DEFAULT_A2_GRID,
) -> Tuple[str, ...]:
    """Return the exact dynamic-query result schema."""
    a1 = _normalize_grid(a1_grid, "a1_grid")
    a2 = _normalize_grid(a2_grid, "a2_grid")
    return (
        "symbol",
        "TradeDate",
        "total_amount",
        "a0_abs_4w20w_selected_amount",
        *(a1_selected_amount_column(lower, upper) for lower, upper in a1),
        *(a2_selected_amount_column(lower, upper) for lower, upper in a2),
        "a3_q20_q80_selected_amount",
    )


def _regular_session_filter_sql(time_col: str = "ExchTime") -> str:
    """Per-row session predicate equivalent to ``09:30 <= t < 15:00:01``."""
    return (
        "AND ("
        f"(toHour({time_col}) = 9 AND toMinute({time_col}) >= 30) "
        f"OR (toHour({time_col}) > 9 AND toHour({time_col}) < 15) "
        f"OR (toHour({time_col}) = 15 AND toMinute({time_col}) = 0 "
        f"AND toSecond({time_col}) = 0)"
        ")"
    )


def _a_share_filter_sql(exchange: str) -> str:
    name = _exchange_name(exchange)
    if name == "SSE":
        return "AND startsWith(Symbol, '6')"
    return (
        "AND (startsWith(Symbol, '000') OR startsWith(Symbol, '001') "
        "OR startsWith(Symbol, '002') OR startsWith(Symbol, '003') "
        "OR startsWith(Symbol, '300') OR startsWith(Symbol, '301') "
        "OR startsWith(Symbol, '302'))"
    )


def _split_symbol(value: object) -> Tuple[str, Optional[str]]:
    text = str(value).strip().upper()
    suffix: Optional[str] = None
    if text.endswith(".SH"):
        text, suffix = text[:-3], "SSE"
    elif text.endswith(".SZ"):
        text, suffix = text[:-3], "SZSE"
    if not re.fullmatch(r"[0-9]{6}", text):
        return text, None
    if suffix is not None:
        return text, suffix
    if _SSE_A_SHARE_RE.fullmatch(text):
        return text, "SSE"
    if _SZSE_A_SHARE_RE.fullmatch(text):
        return text, "SZSE"
    return text, None


def _is_a_share(code: str, exchange: str) -> bool:
    if exchange == "SSE":
        return _SSE_A_SHARE_RE.fullmatch(code) is not None
    return _SZSE_A_SHARE_RE.fullmatch(code) is not None


def _symbol_filter_sql(
    symbols: Optional[Sequence[str]], exchange: str
) -> str:
    if not symbols:
        return ""
    name = _exchange_name(exchange)
    codes = sorted(
        {
            code
            for symbol in symbols
            for code, symbol_exchange in [_split_symbol(symbol)]
            if symbol_exchange == name and _is_a_share(code, name)
        }
    )
    if not codes:
        return "AND 1 = 0"
    values = ", ".join(f"'{code}'" for code in codes)
    return f"AND Symbol IN ({values})"


def _strict_tick_sql(
    start_date: DateLike,
    end_date: DateLike,
    exchange: str,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """Build the strict Tick subquery shared by both aggregation stages."""
    start, end = _date_range(start_date, end_date)
    name = _exchange_name(exchange)
    spec = _EXCHANGE_SPECS[name]
    return f"""
    SELECT
        Symbol,
        ExchTime,
        toFloat64({spec["amount"]}) AS amt
    FROM {spec["table"]}
    WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
      AND ExchTime <  toDateTime64('{end} 15:00:01', 6, 'Asia/Shanghai')
      AND toDate(ExchTime) BETWEEN toDate('{start}') AND toDate('{end}')
      {_regular_session_filter_sql()}
      {spec["trade_filter"]}
      AND Price > 0 AND Volume > 0
      {_a_share_filter_sql(name)}
      {_symbol_filter_sql(symbols, name)}
    """


def build_daily_scale_sql(
    start_date: DateLike,
    end_date: DateLike,
    exchange: str,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> str:
    """Build one exchange's symbol-day trade-size scale query.

    The five quantiles share one ``quantilesExactIf`` aggregate state.  Only
    positive trade amounts contribute to count, mean, and quantiles; the total
    keeps the existing strict A0 ``sum(amt)`` convention.
    """
    name = _exchange_name(exchange)
    suffix = _EXCHANGE_SPECS[name]["suffix"]
    strict_sql = _strict_tick_sql(start_date, end_date, name, symbols)
    return f"""
    WITH strict_ticks AS
    (
        {strict_sql}
    ),
    daily AS
    (
        SELECT
            Symbol,
            toDate(ExchTime) AS TradeDate,
            sum(amt) AS total_amount,
            countIf(amt > 0) AS positive_trade_count,
            avgIf(amt, amt > 0) AS daily_mean_trade_amount,
            quantilesExactIf(0.20, 0.30, 0.50, 0.70, 0.80)(
                amt, amt > 0
            ) AS amount_quantiles
        FROM strict_ticks
        GROUP BY Symbol, TradeDate
    )
    SELECT
        concat(Symbol, '{suffix}') AS symbol,
        TradeDate,
        total_amount,
        positive_trade_count,
        daily_mean_trade_amount,
        amount_quantiles[1] AS q20,
        amount_quantiles[2] AS q30,
        amount_quantiles[3] AS q50,
        amount_quantiles[4] AS q70,
        amount_quantiles[5] AS q80
    FROM daily
    ORDER BY TradeDate, symbol
    """


def build_daily_scale_queries(
    start_date: DateLike,
    end_date: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Build the SSE and SZSE daily-scale queries."""
    return {
        exchange: build_daily_scale_sql(
            start_date, end_date, exchange, symbols=symbols
        )
        for exchange in ("SSE", "SZSE")
    }


def build_dynamic_factor_sql(
    start_date: DateLike,
    end_date: DateLike,
    exchange: str,
    *,
    a1_grid: Grid = DEFAULT_A1_GRID,
    a2_grid: Grid = DEFAULT_A2_GRID,
    symbols: Optional[Sequence[str]] = None,
    external_table: str = SCALE_ROWS_EXTERNAL_TABLE,
) -> str:
    """Build one exchange's single-scan dynamic factor aggregation query."""
    if not _IDENTIFIER_RE.fullmatch(str(external_table)):
        raise ValueError("external_table must be a simple ClickHouse identifier")
    name = _exchange_name(exchange)
    suffix = _EXCHANGE_SPECS[name]["suffix"]
    a1 = _normalize_grid(a1_grid, "a1_grid")
    a2 = _normalize_grid(a2_grid, "a2_grid")
    strict_sql = _strict_tick_sql(start_date, end_date, name, symbols)

    selects = [
        "sum(amt) AS total_amount",
        (
            "ifNull(sumIf(amt, "
            f"amt > {_sql_number(A0_LOWER_RMB)} "
            f"AND amt <= {_sql_number(A0_UPPER_RMB)}), 0) "
            "AS `a0_abs_4w20w_selected_amount`"
        ),
    ]
    for lower, upper in a1:
        condition = (
            "isNotNull(ADV20_lag1) AND ADV20_lag1 > 0 "
            f"AND (amt / ADV20_lag1 * 10000) > {_sql_number(lower)} "
            f"AND (amt / ADV20_lag1 * 10000) <= {_sql_number(upper)}"
        )
        selects.append(
            f"ifNull(sumIf(amt, {condition}), 0) AS "
            f"`{a1_selected_amount_column(lower, upper)}`"
        )
    for lower, upper in a2:
        condition = (
            "isNotNull(ATS20_lag1) AND ATS20_lag1 > 0 "
            f"AND (amt / ATS20_lag1) > {_sql_number(lower)} "
            f"AND (amt / ATS20_lag1) <= {_sql_number(upper)}"
        )
        selects.append(
            f"ifNull(sumIf(amt, {condition}), 0) AS "
            f"`{a2_selected_amount_column(lower, upper)}`"
        )
    selects.append(
        "ifNull(sumIf(amt, isNotNull(q20) AND isNotNull(q80) "
        "AND q20 >= 0 AND q80 >= q20 "
        "AND amt > q20 AND amt <= q80), 0) "
        "AS `a3_q20_q80_selected_amount`"
    )
    aggregate_selects = ",\n        ".join(selects)

    return f"""
    WITH strict_ticks AS
    (
        {strict_sql}
    ),
    joined_ticks AS
    (
        SELECT
            t.Symbol AS Symbol,
            toDate(t.ExchTime) AS TradeDate,
            t.amt AS amt,
            s.ADV20_lag1 AS ADV20_lag1,
            s.ATS20_lag1 AS ATS20_lag1,
            s.q20 AS q20,
            s.q80 AS q80
        FROM strict_ticks AS t
        INNER JOIN {external_table} AS s
            ON t.Symbol = s.symbol
           AND toDate(t.ExchTime) = s.TradeDate
    )
    SELECT
        concat(Symbol, '{suffix}') AS symbol,
        TradeDate,
        {aggregate_selects}
    FROM joined_ticks
    GROUP BY Symbol, TradeDate
    ORDER BY TradeDate, symbol
    """


def build_dynamic_factor_queries(
    start_date: DateLike,
    end_date: DateLike,
    *,
    a1_grid: Grid = DEFAULT_A1_GRID,
    a2_grid: Grid = DEFAULT_A2_GRID,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Build the SSE and SZSE dynamic aggregation queries."""
    return {
        exchange: build_dynamic_factor_sql(
            start_date,
            end_date,
            exchange,
            a1_grid=a1_grid,
            a2_grid=a2_grid,
            symbols=symbols,
        )
        for exchange in ("SSE", "SZSE")
    }


def _rename_scale_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Symbol": "symbol",
        "adv20_lag1": "ADV20_lag1",
        "adv20": "ADV20_lag1",
        "ats20_lag1": "ATS20_lag1",
        "ats20": "ATS20_lag1",
        "daily_q20": "q20",
        "daily_q80": "q80",
    }
    out = frame.copy()
    for source, target in aliases.items():
        if source not in out.columns:
            continue
        if target in out.columns and source != target:
            raise ValueError(
                f"scale_rows contains both {source!r} and {target!r}"
            )
        out = out.rename(columns={source: target})
    return out


def _normalize_scale_rows(scale_rows: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(scale_rows, pd.DataFrame):
        raise TypeError("scale_rows must be a pandas DataFrame")
    frame = _rename_scale_columns(scale_rows)
    missing = sorted(set(SCALE_ROWS_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"scale_rows missing columns: {missing}")
    frame = frame.loc[:, SCALE_ROWS_COLUMNS].copy()
    if frame.empty:
        return frame

    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    dates = pd.to_datetime(frame["TradeDate"], errors="raise")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    frame["TradeDate"] = dates.dt.normalize()

    parsed = frame["symbol"].map(_split_symbol)
    frame["_bare_symbol"] = parsed.map(lambda item: item[0])
    frame["_exchange"] = parsed.map(lambda item: item[1])
    for column in ("ADV20_lag1", "ATS20_lag1", "q20", "q80"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        finite = frame[column].dropna().map(math.isfinite)
        if not finite.all():
            raise ValueError(f"scale_rows {column} contains infinite values")
        if (frame[column].dropna() < 0).any():
            raise ValueError(f"scale_rows {column} contains negative values")
    quantiles = frame[["q20", "q80"]].dropna()
    if (quantiles["q20"] > quantiles["q80"]).any():
        raise ValueError("scale_rows contains q20 > q80")

    # Check both the supplied key and the normalized bare-code key.  The latter
    # catches inputs such as both ``600000`` and ``600000.SH``.
    if frame.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError("scale_rows contains duplicate symbol/TradeDate keys")
    if frame.duplicated(
        ["_bare_symbol", "_exchange", "TradeDate"]
    ).any():
        raise ValueError(
            "scale_rows contains duplicate normalized symbol/TradeDate keys"
        )
    return frame


def prepare_scale_rows_for_exchange(
    scale_rows: pd.DataFrame,
    exchange: str,
    *,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
) -> pd.DataFrame:
    """Validate, date-filter, and convert one exchange to bare CH symbols."""
    name = _exchange_name(exchange)
    frame = _normalize_scale_rows(scale_rows)
    if frame.empty:
        return pd.DataFrame(columns=list(SCALE_ROWS_COLUMNS))
    frame = frame.loc[
        frame["_exchange"].eq(name)
        & frame["_bare_symbol"].map(lambda code: _is_a_share(code, name))
    ].copy()
    if start_date is not None:
        start = pd.Timestamp(_to_date_str(start_date))
        frame = frame.loc[frame["TradeDate"] >= start]
    if end_date is not None:
        end = pd.Timestamp(_to_date_str(end_date))
        frame = frame.loc[frame["TradeDate"] <= end]
    frame["symbol"] = frame["_bare_symbol"]
    frame = frame.loc[:, SCALE_ROWS_COLUMNS]
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(
            "exchange scale_rows contains duplicate symbol/TradeDate keys"
        )
    return frame.sort_values(
        ["TradeDate", "symbol"], kind="stable"
    ).reset_index(drop=True)


def scale_rows_csv_payload(prepared_scale_rows: pd.DataFrame) -> bytes:
    """Serialize validated bare-symbol rows as ClickHouse CSVWithNames."""
    missing = sorted(
        set(SCALE_ROWS_COLUMNS).difference(prepared_scale_rows.columns)
    )
    if missing:
        raise ValueError(f"prepared scale_rows missing columns: {missing}")
    frame = prepared_scale_rows.loc[:, SCALE_ROWS_COLUMNS].copy()
    return frame.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        na_rep=r"\N",
    ).encode("utf-8")


def build_scale_rows_external_data(
    scale_rows: pd.DataFrame,
    exchange: str,
    *,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    external_data_cls: Optional[Type[object]] = None,
):
    """Create clickhouse-connect 0.8.17 ``ExternalData`` for one exchange."""
    prepared = prepare_scale_rows_for_exchange(
        scale_rows,
        exchange,
        start_date=start_date,
        end_date=end_date,
    )
    if prepared.empty:
        raise ValueError(f"no scale_rows for {_exchange_name(exchange)}")
    if external_data_cls is None:
        from clickhouse_connect.driver.external import ExternalData

        external_data_cls = ExternalData
    return external_data_cls(
        data=scale_rows_csv_payload(prepared),
        file_name=SCALE_ROWS_EXTERNAL_FILE,
        fmt=SCALE_ROWS_EXTERNAL_FORMAT,
        structure=SCALE_ROWS_EXTERNAL_STRUCTURE,
    )


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _audit_exact_schema(
    frame: pd.DataFrame, expected_columns: Sequence[str], name: str
) -> pd.DataFrame:
    expected = list(expected_columns)
    missing = sorted(set(expected).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(expected))
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch; missing={missing}, extra={extra}"
        )
    out = frame.loc[:, expected].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["TradeDate"] = pd.to_datetime(
        out["TradeDate"], errors="raise"
    ).dt.normalize()
    if out[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{name} contains null symbol/TradeDate keys")
    if out.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{name} contains duplicate symbol/TradeDate keys")
    return out


def audit_daily_scale_result(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and hard-audit a daily scale primitive result."""
    if frame is None or frame.empty:
        return _empty(DAILY_SCALE_COLUMNS)
    out = _audit_exact_schema(frame, DAILY_SCALE_COLUMNS, "daily scale result")
    numeric = [
        column
        for column in DAILY_SCALE_COLUMNS
        if column not in KEY_COLUMNS
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="raise")
    count = out["positive_trade_count"]
    if count.isna().any() or (count < 0).any() or (count % 1 != 0).any():
        raise ValueError("positive_trade_count must be a nonnegative integer")
    out["positive_trade_count"] = count.astype("int64")
    if (
        ~out["total_amount"].map(math.isfinite)
    ).any() or (out["total_amount"] < -1e-10).any():
        raise ValueError("daily scale total_amount must be finite/nonnegative")
    for column in (
        "daily_mean_trade_amount",
        "q20",
        "q30",
        "q50",
        "q70",
        "q80",
    ):
        values = out[column].dropna()
        if (~values.map(math.isfinite)).any() or (values < -1e-10).any():
            raise ValueError(f"daily scale {column} is invalid")
    complete_quantiles = out.loc[
        out[["q20", "q30", "q50", "q70", "q80"]].notna().all(axis=1),
        ["q20", "q30", "q50", "q70", "q80"],
    ]
    if not complete_quantiles.empty:
        differences = complete_quantiles.diff(axis=1).iloc[:, 1:]
        if (differences < -1e-10).any().any():
            raise ValueError("daily scale quantiles are not monotonic")
    return out.sort_values(
        ["TradeDate", "symbol"], kind="stable"
    ).reset_index(drop=True)


def audit_dynamic_factor_result(
    frame: pd.DataFrame,
    *,
    a1_grid: Grid = DEFAULT_A1_GRID,
    a2_grid: Grid = DEFAULT_A2_GRID,
) -> pd.DataFrame:
    """Normalize schema and enforce selected-amount conservation."""
    columns = dynamic_result_columns(a1_grid, a2_grid)
    if frame is None or frame.empty:
        return _empty(columns)
    out = _audit_exact_schema(frame, columns, "dynamic factor result")
    amount_columns = list(columns[2:])
    for column in amount_columns:
        out[column] = pd.to_numeric(out[column], errors="raise")
        if out[column].isna().any():
            raise ValueError(f"dynamic factor result {column} contains nulls")
        if (~out[column].map(math.isfinite)).any():
            raise ValueError(
                f"dynamic factor result {column} contains non-finite values"
            )
        if (out[column] < -1e-10).any():
            raise ValueError(
                f"dynamic factor result {column} contains negative values"
            )
    total = out["total_amount"]
    tolerance = total.abs().mul(1e-10).clip(lower=1e-8)
    for column in amount_columns[1:]:
        if (out[column] > total + tolerance).any():
            raise ValueError(f"{column} exceeds total_amount")
    return out.sort_values(
        ["TradeDate", "symbol"], kind="stable"
    ).reset_index(drop=True)


def _close_owned_client(client: object) -> None:
    try:
        client.close()
    except Exception:  # noqa: BLE001
        pass


def fetch_daily_scale_primitive(
    start_date: DateLike,
    end_date: DateLike,
    *,
    symbols: Optional[Sequence[str]] = None,
    client=None,
) -> pd.DataFrame:
    """Fetch daily scale rows; each source exchange is scanned once."""
    _date_range(start_date, end_date)
    own_client = client is None
    query_client = client or _get_ch_client()
    frames: List[pd.DataFrame] = []
    try:
        for exchange in ("SSE", "SZSE"):
            sql = build_daily_scale_sql(
                start_date, end_date, exchange, symbols=symbols
            )
            frame = query_client.query_df(sql)
            if frame is not None and not frame.empty:
                frames.append(frame)
    finally:
        if own_client:
            _close_owned_client(query_client)
    if not frames:
        return _empty(DAILY_SCALE_COLUMNS)
    return audit_daily_scale_result(pd.concat(frames, ignore_index=True))


def fetch_dynamic_factor_aggregates(
    start_date: DateLike,
    end_date: DateLike,
    scale_rows: pd.DataFrame,
    *,
    a1_grid: Grid = DEFAULT_A1_GRID,
    a2_grid: Grid = DEFAULT_A2_GRID,
    symbols: Optional[Sequence[str]] = None,
    client=None,
    external_data_cls: Optional[Type[object]] = None,
) -> pd.DataFrame:
    """Fetch A0/A1/A2/A3 symbol-day numerators in one scan per exchange.

    ``scale_rows`` may contain both Wind-suffixed exchanges.  Before each query
    it is filtered to that source exchange and serialized as a separate
    ``CSVWithNames`` external table.  Thus an SSE query can never join SZSE
    scale rows (or vice versa).
    """
    start, end = _date_range(start_date, end_date)
    a1 = _normalize_grid(a1_grid, "a1_grid")
    a2 = _normalize_grid(a2_grid, "a2_grid")
    # Validate the whole input before exchange filtering so duplicate keys
    # cannot be hidden by a date or exchange subset.
    _normalize_scale_rows(scale_rows)
    prepared = {
        exchange: prepare_scale_rows_for_exchange(
            scale_rows, exchange, start_date=start, end_date=end
        )
        for exchange in ("SSE", "SZSE")
    }
    if not any(not frame.empty for frame in prepared.values()):
        return _empty(dynamic_result_columns(a1, a2))

    own_client = client is None
    query_client = client or _get_ch_client()
    frames: List[pd.DataFrame] = []
    try:
        for exchange in ("SSE", "SZSE"):
            exchange_rows = prepared[exchange]
            if exchange_rows.empty:
                continue
            external_data = build_scale_rows_external_data(
                exchange_rows,
                exchange,
                start_date=start,
                end_date=end,
                external_data_cls=external_data_cls,
            )
            sql = build_dynamic_factor_sql(
                start,
                end,
                exchange,
                a1_grid=a1,
                a2_grid=a2,
                symbols=symbols,
            )
            frame = query_client.query_df(
                sql, external_data=external_data
            )
            if frame is not None and not frame.empty:
                frames.append(frame)
    finally:
        if own_client:
            _close_owned_client(query_client)
    if not frames:
        return _empty(dynamic_result_columns(a1, a2))
    return audit_dynamic_factor_result(
        pd.concat(frames, ignore_index=True),
        a1_grid=a1,
        a2_grid=a2,
    )


# Explicit aliases keep naming discoverable for orchestration scripts.
daily_scale_primitive_sql = build_daily_scale_sql
dynamic_factor_aggregate_sql = build_dynamic_factor_sql
fetch_daily_trade_size_scale = fetch_daily_scale_primitive
fetch_normalized_trade_amount_aggregates = fetch_dynamic_factor_aggregates

