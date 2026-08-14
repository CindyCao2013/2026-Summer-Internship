"""Company-data adapter for the isolated GNN feature validity audit.

The adapter reuses the repository's shared DolphinDB session and reads only
verified Wind tables.  It does not contain connection settings or credentials.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.ddb.connection import get_ddb_session


class DataUnavailableError(RuntimeError):
    """Raised when a required verified source or field is unavailable."""


def _date_literal(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def _validate_identifier(value: str, kind: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_./:-]+", str(value)):
        raise ValueError("Unsafe {} identifier: {!r}".format(kind, value))
    return str(value)


def _ddb_string_vector(values: Iterable[str]) -> str:
    clean = []
    for value in sorted(set(str(v) for v in values)):
        if '"' in value or "\\" in value:
            raise ValueError("Unsafe symbol value: {!r}".format(value))
        clean.append('"{}"'.format(value))
    return "[{}]".format(",".join(clean))


def _as_frame(raw, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=list(columns or []))
    frame = pd.DataFrame(raw)
    if frame.empty and columns is not None:
        return pd.DataFrame(columns=list(columns))
    return frame


def _normalise_date(frame: pd.DataFrame, column: str) -> None:
    if column in frame.columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def _normalise_symbol(frame: pd.DataFrame, column: str = "symbol") -> None:
    if column in frame.columns:
        frame[column] = frame[column].astype(str)


def _first_recorded_version(
    frame: pd.DataFrame,
    keys: Sequence[str],
    opdate: str = "opdate",
) -> pd.DataFrame:
    """Keep the earliest stored duplicate; never backfill a later revision."""
    if frame.empty:
        return frame
    out = frame.copy()
    if opdate in out.columns:
        out[opdate] = pd.to_datetime(out[opdate], errors="coerce")
        out = out.sort_values(list(keys) + [opdate], na_position="last")
    else:
        out = out.sort_values(list(keys))
    return out.drop_duplicates(subset=list(keys), keep="first").reset_index(drop=True)


def _pivot(
    frame: pd.DataFrame,
    value: str,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    if frame.empty or value not in frame.columns:
        return pd.DataFrame(index=dates, columns=list(symbols), dtype=float)
    wide = frame.pivot_table(
        index=date_column,
        columns=symbol_column,
        values=value,
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index)
    return wide.reindex(index=dates, columns=list(symbols))


def _expand_intervals(
    intervals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    value_column: str,
    start_column: str,
    end_column: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=dates, columns=list(symbols), dtype=object)
    if intervals.empty:
        return out
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    symbol_set = set(symbols)
    ordered = intervals.sort_values(["symbol", start_column, end_column])
    for row in ordered.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        if symbol not in symbol_set:
            continue
        start = pd.Timestamp(getattr(row, start_column))
        end_value = getattr(row, end_column)
        end = pd.Timestamp(end_value) if pd.notna(end_value) else dates[-1]
        lo = int(np.searchsorted(date_values, start.to_datetime64(), side="left"))
        hi = int(np.searchsorted(date_values, end.to_datetime64(), side="right"))
        if lo < hi:
            out.iloc[lo:hi, out.columns.get_loc(symbol)] = getattr(row, value_column)
    return out


def _map_to_calendar(
    values: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    strictly_after: bool,
) -> pd.Series:
    arr = calendar.to_numpy(dtype="datetime64[ns]")
    side = "right" if strictly_after else "left"
    mapped: List[pd.Timestamp] = []
    for value in pd.to_datetime(values, errors="coerce"):
        if pd.isna(value):
            mapped.append(pd.NaT)
            continue
        pos = int(np.searchsorted(arr, pd.Timestamp(value).to_datetime64(), side=side))
        mapped.append(calendar[pos] if pos < len(calendar) else pd.NaT)
    return pd.Series(mapped, index=values.index, dtype="datetime64[ns]")


@dataclass
class DataBundle:
    calendar: pd.DatetimeIndex
    sample_dates: pd.DatetimeIndex
    symbols: List[str]
    market: pd.DataFrame
    derivative: pd.DataFrame
    financial: pd.DataFrame
    industry: pd.DataFrame
    universe_mask: pd.DataFrame
    tradable_mask: pd.DataFrame
    eligible_mask: pd.DataFrame
    stock_returns: pd.DataFrame
    index_returns: pd.Series
    audit: Dict[str, object]

    def market_wide(self, field: str) -> pd.DataFrame:
        return _pivot(self.market, field, self.calendar, self.symbols)

    def derivative_wide(self, field: str) -> pd.DataFrame:
        return _pivot(self.derivative, field, self.calendar, self.symbols)

    @property
    def market_cap(self) -> pd.DataFrame:
        return self.derivative_wide("market_cap")


class CompanyDataAdapter:
    """Read verified company data through the shared DolphinDB session."""

    REQUIRED_SCHEMAS: Mapping[str, Tuple[str, ...]] = {
        "eod": (
            "S_INFO_WINDCODE",
            "TRADE_DT",
            "S_DQ_PRECLOSE",
            "S_DQ_OPEN",
            "S_DQ_HIGH",
            "S_DQ_LOW",
            "S_DQ_CLOSE",
            "S_DQ_VOLUME",
            "S_DQ_AMOUNT",
            "S_DQ_TRADESTATUS",
            "S_DQ_LIMIT",
            "S_DQ_STOPPING",
            "OPDATE",
        ),
        "derivative": (
            "S_INFO_WINDCODE",
            "TRADE_DT",
            "S_VAL_MV",
            "S_VAL_PE_TTM",
            "S_VAL_PB_NEW",
            "S_DQ_TURN",
            "OPDATE",
        ),
        "financial": (
            "S_INFO_WINDCODE",
            "ANN_DT",
            "REPORT_PERIOD",
            "STATEMENT_TYPE",
            "S_FA_ROE_TTM",
            "TOT_OPER_REV_TTM",
            "NET_PROFIT_PARENT_COMP_TTM",
            "S_FA_DEBTTOASSETS_MRQ",
            "OPDATE",
        ),
        "industry": (
            "S_INFO_WINDCODE",
            "CITICS_IND_CODE",
            "ENTRY_DT",
            "REMOVE_DT",
            "OPDATE",
        ),
        "universe": ("TRADE_DT", "S_CON_WINDCODE"),
        "index_return": (
            "S_INFO_WINDCODE",
            "TRADE_DT",
            "S_DQ_CLOSE",
            "S_DQ_PRECLOSE",
            "OPDATE",
        ),
        "calendar": ("TRADE_DAYS", "S_INFO_EXCHMARKET"),
        "previous_name": (
            "S_INFO_WINDCODE",
            "BEGINDATE",
            "ENDDATE",
            "S_INFO_NAME",
        ),
    }

    def __init__(self, config: Mapping[str, object], session=None) -> None:
        self.config = config
        self.session = session or get_ddb_session()
        data_cfg = config["data"]
        self.paths = {
            "eod": data_cfg["eod_db"],
            "derivative": data_cfg["derivative_db"],
            "financial": data_cfg["financial_db"],
            "index_return": data_cfg["index_return_db"],
            "universe": data_cfg["index_weight_db"],
            "industry": data_cfg["industry_db"],
            "calendar": data_cfg["calendar_db"],
            "previous_name": data_cfg["previous_name_db"],
        }
        for key, value in self.paths.items():
            _validate_identifier(str(value), key)

    def _run(self, script: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        return _as_frame(self.session.run(script), columns=columns)

    def schema_audit(self) -> pd.DataFrame:
        rows = []
        for kind, required in self.REQUIRED_SCHEMAS.items():
            db_path = self.paths[kind]
            raw = self.session.run(
                "schema(loadTable('{}', 'data')).colDefs.name".format(db_path)
            )
            actual = set(str(x) for x in np.asarray(raw).ravel())
            missing = sorted(set(required) - actual)
            rows.append(
                {
                    "source": kind,
                    "db_path": db_path,
                    "required_fields": "|".join(required),
                    "missing_fields": "|".join(missing),
                    "available": not missing,
                }
            )
        return pd.DataFrame(rows)

    def _load_calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        script = """
select TRADE_DAYS as date
from loadTable('{db}', 'data')
where TRADE_DAYS >= {start} and TRADE_DAYS <= {end}
  and S_INFO_EXCHMARKET = 'SSE'
order by TRADE_DAYS
""".format(
            db=self.paths["calendar"],
            start=_date_literal(start),
            end=_date_literal(end),
        )
        frame = self._run(script, ["date"])
        _normalise_date(frame, "date")
        dates = pd.DatetimeIndex(frame["date"].dropna().drop_duplicates().sort_values())
        if dates.empty:
            raise DataUnavailableError("Company trading calendar returned no rows")
        return dates

    def _load_universe(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        script = """
select TRADE_DT as date, S_CON_WINDCODE as symbol
from loadTable('{db}', 'data')
where TRADE_DT >= {start} and TRADE_DT <= {end}
""".format(
            db=self.paths["universe"],
            start=_date_literal(start),
            end=_date_literal(end),
        )
        frame = self._run(script, ["date", "symbol"])
        _normalise_date(frame, "date")
        _normalise_symbol(frame)
        frame = frame.dropna(subset=["date", "symbol"]).drop_duplicates()
        if frame.empty:
            raise DataUnavailableError("PIT universe returned no rows")
        return frame

    def _load_market(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Sequence[str],
    ) -> pd.DataFrame:
        script = """
select
  TRADE_DT as date,
  S_INFO_WINDCODE as symbol,
  S_DQ_PRECLOSE as pre_close,
  S_DQ_OPEN as open,
  S_DQ_HIGH as high,
  S_DQ_LOW as low,
  S_DQ_CLOSE as close,
  S_DQ_VOLUME as volume,
  S_DQ_AMOUNT as amount,
  S_DQ_TRADESTATUS as trade_status,
  S_DQ_LIMIT as limit_up,
  S_DQ_STOPPING as limit_down,
  OPDATE as opdate
from loadTable('{db}', 'data')
where TRADE_DT >= {start} and TRADE_DT <= {end}
  and S_INFO_WINDCODE in {symbols}
""".format(
            db=self.paths["eod"],
            start=_date_literal(start),
            end=_date_literal(end),
            symbols=_ddb_string_vector(symbols),
        )
        columns = [
            "date",
            "symbol",
            "pre_close",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "trade_status",
            "limit_up",
            "limit_down",
            "opdate",
        ]
        frame = self._run(script, columns)
        _normalise_date(frame, "date")
        _normalise_symbol(frame)
        numeric = [
            "pre_close",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "limit_up",
            "limit_down",
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return _first_recorded_version(frame, ["date", "symbol"])

    def _load_derivative(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Sequence[str],
    ) -> pd.DataFrame:
        script = """
select
  TRADE_DT as date,
  S_INFO_WINDCODE as symbol,
  S_DQ_TURN as turnover,
  S_VAL_PE_TTM as pe_ttm,
  S_VAL_PB_NEW as pb,
  S_VAL_MV as market_cap,
  OPDATE as opdate
from loadTable('{db}', 'data')
where TRADE_DT >= {start} and TRADE_DT <= {end}
  and S_INFO_WINDCODE in {symbols}
""".format(
            db=self.paths["derivative"],
            start=_date_literal(start),
            end=_date_literal(end),
            symbols=_ddb_string_vector(symbols),
        )
        columns = ["date", "symbol", "turnover", "pe_ttm", "pb", "market_cap", "opdate"]
        frame = self._run(script, columns)
        _normalise_date(frame, "date")
        _normalise_symbol(frame)
        for column in ("turnover", "pe_ttm", "pb", "market_cap"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return _first_recorded_version(frame, ["date", "symbol"])

    def _load_financial(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        symbols: Sequence[str],
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        script = """
select
  S_INFO_WINDCODE as symbol,
  ANN_DT as ann_date,
  REPORT_PERIOD as report_period,
  STATEMENT_TYPE as statement_type,
  S_FA_ROE_TTM as roe_ttm,
  TOT_OPER_REV_TTM as revenue_ttm,
  NET_PROFIT_PARENT_COMP_TTM as profit_ttm,
  S_FA_DEBTTOASSETS_MRQ as debt_ratio,
  OPDATE as opdate
from loadTable('{db}', 'data')
where ANN_DT >= {start} and ANN_DT <= {end}
  and S_INFO_WINDCODE in {symbols}
""".format(
            db=self.paths["financial"],
            start=_date_literal(start),
            end=_date_literal(end),
            symbols=_ddb_string_vector(symbols),
        )
        columns = [
            "symbol",
            "ann_date",
            "report_period",
            "statement_type",
            "roe_ttm",
            "revenue_ttm",
            "profit_ttm",
            "debt_ratio",
            "opdate",
        ]
        frame = self._run(script, columns)
        _normalise_symbol(frame)
        _normalise_date(frame, "ann_date")
        _normalise_date(frame, "report_period")
        _normalise_date(frame, "opdate")
        statement_type = str(self.config["data"]["financial_statement_type"])
        if "statement_type" in frame.columns:
            exact = frame["statement_type"].astype(str).eq(statement_type)
            frame = frame.loc[exact].copy()
        if frame.empty:
            raise DataUnavailableError(
                "Financial table has no rows for statement type {!r}".format(
                    statement_type
                )
            )
        for column in ("roe_ttm", "revenue_ttm", "profit_ttm", "debt_ratio"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["symbol", "ann_date", "report_period"])
        frame = _first_recorded_version(
            frame,
            ["symbol", "ann_date", "report_period"],
        )

        # ANN_DT is the verified public-availability field. OPDATE in these
        # live vendor tables is a warehouse-maintenance timestamp (often years
        # later), not first availability, so it is audit-only. Conservatively
        # make date-only announcements available on the next trading day.
        frame["available_date"] = _map_to_calendar(
            frame["ann_date"], calendar, strictly_after=True
        )
        frame = frame.dropna(subset=["available_date"])
        return frame.sort_values(["symbol", "available_date", "report_period"])

    def _load_industry(
        self,
        symbols: Sequence[str],
        calendar: pd.DatetimeIndex,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        script = """
select
  S_INFO_WINDCODE as symbol,
  CITICS_IND_CODE as industry_code,
  ENTRY_DT as entry_date,
  REMOVE_DT as remove_date,
  OPDATE as opdate
from loadTable('{db}', 'data')
where S_INFO_WINDCODE in {symbols}
""".format(
            db=self.paths["industry"],
            symbols=_ddb_string_vector(symbols),
        )
        frame = self._run(
            script,
            ["symbol", "industry_code", "entry_date", "remove_date", "opdate"],
        )
        _normalise_symbol(frame)
        for column in ("entry_date", "remove_date", "opdate"):
            _normalise_date(frame, column)
        frame = frame.dropna(subset=["symbol", "industry_code", "entry_date"])
        frame["industry_l1"] = frame["industry_code"].astype(str).str[:4]
        frame = _first_recorded_version(
            frame, ["symbol", "entry_date", "industry_l1"]
        )
        panel = _expand_intervals(
            frame,
            calendar,
            symbols,
            value_column="industry_l1",
            start_column="entry_date",
            end_column="remove_date",
        )
        return frame, panel

    def _load_st_mask(
        self,
        symbols: Sequence[str],
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        script = """
select
  S_INFO_WINDCODE as symbol,
  BEGINDATE as begin_date,
  ENDDATE as end_date,
  S_INFO_NAME as stock_name
from loadTable('{db}', 'data')
where S_INFO_WINDCODE in {symbols}
""".format(
            db=self.paths["previous_name"],
            symbols=_ddb_string_vector(symbols),
        )
        frame = self._run(
            script, ["symbol", "begin_date", "end_date", "stock_name"]
        )
        _normalise_symbol(frame)
        _normalise_date(frame, "begin_date")
        _normalise_date(frame, "end_date")
        frame = frame[
            frame["stock_name"].astype(str).str.contains("ST|退", regex=True, na=False)
        ].copy()
        frame["st_flag"] = 1.0
        expanded = _expand_intervals(
            frame,
            calendar,
            symbols,
            value_column="st_flag",
            start_column="begin_date",
            end_column="end_date",
        )
        return expanded.isna()

    def _load_index_returns(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        index_code = _validate_identifier(
            str(self.config["universe"]["index_code"]), "index code"
        )
        script = """
select
  TRADE_DT as date,
  S_DQ_CLOSE as close,
  S_DQ_PRECLOSE as pre_close,
  OPDATE as opdate
from loadTable('{db}', 'data')
where TRADE_DT >= {start} and TRADE_DT <= {end}
  and S_INFO_WINDCODE = '{index_code}'
""".format(
            db=self.paths["index_return"],
            start=_date_literal(start),
            end=_date_literal(end),
            index_code=index_code,
        )
        frame = self._run(script, ["date", "close", "pre_close", "opdate"])
        _normalise_date(frame, "date")
        for column in ("close", "pre_close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = _first_recorded_version(frame, ["date"])
        ret = frame.set_index("date")["close"] / frame.set_index("date")["pre_close"] - 1.0
        ret.name = "index_return"
        return ret.sort_index()

    def load(self) -> DataBundle:
        sample_cfg = self.config["sample"]
        start = pd.Timestamp(sample_cfg["start_date"])
        end = pd.Timestamp(sample_cfg["end_date"])
        preheat = start - pd.Timedelta(
            days=int(sample_cfg["market_preheat_calendar_days"])
        )
        financial_start = start - pd.DateOffset(
            years=int(sample_cfg["financial_history_years"])
        )

        schemas = self.schema_audit()
        missing = schemas.loc[~schemas["available"]]
        if not missing.empty:
            detail = "; ".join(
                "{}:{}".format(row.source, row.missing_fields)
                for row in missing.itertuples(index=False)
            )
            raise DataUnavailableError("Required schema fields missing: " + detail)

        calendar = self._load_calendar(financial_start, end + pd.Timedelta(days=10))
        market_calendar = calendar[(calendar >= preheat) & (calendar <= end)]
        sample_dates = market_calendar[(market_calendar >= start) & (market_calendar <= end)]
        universe_long = self._load_universe(start, end)
        symbols = sorted(universe_long["symbol"].dropna().unique().tolist())

        market = self._load_market(preheat, end, symbols)
        derivative = self._load_derivative(preheat, end, symbols)
        financial = self._load_financial(
            financial_start, end, symbols, calendar
        )
        industry_long, industry_panel_full = self._load_industry(
            symbols, market_calendar
        )
        not_st = self._load_st_mask(symbols, market_calendar)
        index_returns = self._load_index_returns(preheat, end)

        universe_mask = (
            universe_long.assign(member=True)
            .pivot_table(
                index="date",
                columns="symbol",
                values="member",
                aggfunc="last",
            )
            .reindex(index=sample_dates, columns=symbols)
            .fillna(False)
            .astype(bool)
        )

        close = _pivot(market, "close", market_calendar, symbols)
        pre_close = _pivot(market, "pre_close", market_calendar, symbols)
        limit_up = _pivot(market, "limit_up", market_calendar, symbols)
        limit_down = _pivot(market, "limit_down", market_calendar, symbols)
        trade_status = _pivot(
            market,
            "trade_status",
            market_calendar,
            symbols,
        )
        status_text = trade_status.apply(lambda col: col.astype(str))
        not_suspended = trade_status.notna() & ~status_text.isin(["停牌", ""])
        not_at_limit = (
            close.notna()
            & limit_up.notna()
            & limit_down.notna()
            & (close < limit_up)
            & (close > limit_down)
        )
        listing_count = close.notna().astype(int).cumsum()
        seasoned = listing_count >= int(
            self.config["universe"]["min_listing_observations"]
        )
        tradable_full = (
            not_suspended
            & not_at_limit
            & not_st.reindex_like(close).fillna(False)
            & seasoned
        )
        tradable = tradable_full.reindex(index=sample_dates, columns=symbols).fillna(False)
        eligible = universe_mask & tradable
        signal_lag = int(self.config["timing"]["signal_lag_trading_rows"])
        execution_eligible = (
            eligible
            & tradable.shift(-(signal_lag - 1)).fillna(False)
            & tradable.shift(-signal_lag).fillna(False)
        )

        # Retain the preheat window for the first PIT relation snapshot.
        # Backtests explicitly reindex this panel to sample_dates.
        stock_returns = close / pre_close - 1.0
        industry = industry_panel_full.reindex(
            index=market_calendar, columns=symbols
        )

        eod_future_opdate = int(
            (
                pd.to_datetime(market["opdate"], errors="coerce").dt.normalize()
                > market["date"]
            ).sum()
        )
        derivative_future_opdate = int(
            (
                pd.to_datetime(derivative["opdate"], errors="coerce").dt.normalize()
                > derivative["date"]
            ).sum()
        )
        finance_opdate_after_end = int(
            (
                pd.to_datetime(financial["opdate"], errors="coerce").dt.normalize()
                > end
            ).sum()
        )
        audit: Dict[str, object] = {
            "schema_audit": schemas,
            "sample_start": sample_dates.min(),
            "sample_end": sample_dates.max(),
            "n_sample_dates": int(len(sample_dates)),
            "n_union_symbols": int(len(symbols)),
            "median_universe_size": float(universe_mask.sum(axis=1).median()),
            "median_eligible_size": float(eligible.sum(axis=1).median()),
            "median_execution_eligible_size": float(
                execution_eligible.sum(axis=1).median()
            ),
            "market_rows": int(len(market)),
            "derivative_rows": int(len(derivative)),
            "financial_rows": int(len(financial)),
            "industry_interval_rows": int(len(industry_long)),
            "eod_opdate_after_trade_date_rows": eod_future_opdate,
            "derivative_opdate_after_trade_date_rows": derivative_future_opdate,
            "financial_rows_with_opdate_after_sample_end": finance_opdate_after_end,
            "duplicate_version_policy": "earliest_recorded_opdate",
            "financial_availability_field": "next_trading_day_after_ANN_DT",
            "opdate_used_as_first_availability": False,
            "unavailable_sources": dict(self.config["data"]["unavailable_sources"]),
        }
        return DataBundle(
            calendar=market_calendar,
            sample_dates=sample_dates,
            symbols=symbols,
            market=market,
            derivative=derivative,
            financial=financial,
            industry=industry,
            universe_mask=universe_mask,
            tradable_mask=tradable,
            eligible_mask=eligible,
            stock_returns=stock_returns,
            index_returns=index_returns.reindex(sample_dates),
            audit=audit,
        )


def assert_factor_inputs_pit(
    frame: pd.DataFrame,
    *,
    factor_date_column: str = "factor_date",
    available_at_column: str = "available_at",
) -> None:
    """Assert that no input becomes available after its factor date."""
    required = {factor_date_column, available_at_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("PIT check missing columns: {}".format(sorted(missing)))
    factor_date = pd.to_datetime(frame[factor_date_column], errors="coerce")
    available_at = pd.to_datetime(frame[available_at_column], errors="coerce")
    bad = available_at.notna() & factor_date.notna() & (available_at > factor_date)
    if bad.any():
        raise AssertionError(
            "{} factor inputs are available after factor date".format(int(bad.sum()))
        )
