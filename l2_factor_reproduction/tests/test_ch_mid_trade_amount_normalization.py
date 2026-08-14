"""Unit tests for normalized trade-amount ClickHouse extraction."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from l2_factor_reproduction.python import (
    ch_mid_trade_amount_normalization as ch_norm,
)


class _FakeClient:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def query_df(self, sql: str, external_data=None) -> pd.DataFrame:
        self.calls.append((sql, external_data))
        return pd.DataFrame()

    def close(self) -> None:
        self.closed = True


class _RecordingExternalData:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def _scale_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "600000.SH",
                "000001.SZ",
                "510300.SH",  # ETF: must not enter either A-share query.
            ],
            "TradeDate": ["2024-01-02"] * 3,
            "ADV20_lag1": [20_000_000.0, 2_000_000_000.0, 1.0],
            "ATS20_lag1": [50_000.0, 80_000.0, 1.0],
            "q20": [10_000.0, 20_000.0, 0.5],
            "q80": [200_000.0, 300_000.0, 2.0],
        }
    )


def test_daily_scale_sql_freezes_strict_trade_and_amount_rules() -> None:
    sse = ch_norm.build_daily_scale_sql(
        "2024-01-01",
        "2024-03-31",
        "SSE",
        symbols=["600000.SH", "000001.SZ", "510300.SH"],
    )
    szse = ch_norm.build_daily_scale_sql(
        "2024-01-01",
        "2024-03-31",
        "SZSE",
        symbols=["600000.SH", "000001.SZ"],
    )

    for sql in (sse, szse):
        assert "09:30:00" in sql
        assert "15:00:01" in sql
        assert "toDate(ExchTime) BETWEEN" in sql
        assert "toHour(ExchTime)" in sql
        assert "toMinute(ExchTime) >= 30" in sql
        assert "toSecond(ExchTime) = 0" in sql
        assert "sum(amt) AS total_amount" in sql
        assert "countIf(amt > 0) AS positive_trade_count" in sql
        assert "avgIf(amt, amt > 0) AS daily_mean_trade_amount" in sql
        assert sql.count("quantilesExactIf(") == 1
        assert "0.20, 0.30, 0.50, 0.70, 0.80" in sql
        assert "GROUP BY Symbol, TradeDate" in sql
        assert "sum(Volume)" not in sql
        assert "fee" not in sql.lower()

    assert "cmds.SSE_AL_TICK_EXG" in sse
    assert "Type = 'T'" in sse
    assert "ifNull(Amount, Price * Volume)" in sse
    assert "startsWith(Symbol, '6')" in sse
    assert "Symbol IN ('600000')" in sse
    assert "510300" not in sse

    assert "cmds.SZSE_AL_TICK_EXG" in szse
    assert "Type = '011'" in szse
    assert "BidOrderNo > 0 AND AskOrderNo > 0" in szse
    assert "toFloat64(Price * Volume) AS amt" in szse
    assert "ifNull(Amount" not in szse
    assert "startsWith(Symbol, '000')" in szse
    assert "Symbol IN ('000001')" in szse


def test_daily_scale_fetch_uses_two_aggregated_queries(monkeypatch) -> None:
    fake = _FakeClient()
    monkeypatch.setattr(ch_norm, "_get_ch_client", lambda: fake)

    result = ch_norm.fetch_daily_scale_primitive(
        "2024-01-02", "2024-01-03"
    )

    assert fake.closed
    assert len(fake.calls) == 2
    assert list(result.columns) == list(ch_norm.DAILY_SCALE_COLUMNS)
    assert result.empty
    for sql, external_data in fake.calls:
        assert external_data is None
        assert "GROUP BY Symbol, TradeDate" in sql
        assert "ExchTime" not in sql.split("FROM daily", 1)[1]


def test_dynamic_sql_joins_scales_and_uses_open_closed_boundaries() -> None:
    a1_grid = ((1.0, 10.0),)
    a2_grid = ((0.5, 2.0),)
    sql = ch_norm.build_dynamic_factor_sql(
        "2024-01-02",
        "2024-01-31",
        "SSE",
        a1_grid=a1_grid,
        a2_grid=a2_grid,
    )

    assert sql.count("cmds.SSE_AL_TICK_EXG") == 1
    assert "INNER JOIN scale_rows AS s" in sql
    assert "t.Symbol = s.symbol" in sql
    assert "toDate(t.ExchTime) = s.TradeDate" in sql
    assert "sum(amt) AS total_amount" in sql
    assert "sumIf(amt, amt > 40000 AND amt <= 200000)" in sql
    assert sql.count("ifNull(sumIf(amt,") == 4
    assert "(amt / ADV20_lag1 * 10000) > 1" in sql
    assert "(amt / ADV20_lag1 * 10000) <= 10" in sql
    assert "(amt / ATS20_lag1) > 0.5" in sql
    assert "(amt / ATS20_lag1) <= 2" in sql
    assert "amt > q20 AND amt <= q80" in sql
    assert "sumIf(Volume" not in sql
    assert "sum(Volume)" not in sql
    assert "GROUP BY Symbol, TradeDate" in sql
    assert "fee" not in sql.lower()


def test_external_data_is_csv_with_names_and_exchange_filtered() -> None:
    external = ch_norm.build_scale_rows_external_data(
        _scale_rows(), "SSE", start_date="2024-01-02", end_date="2024-01-02"
    )

    assert len(external.files) == 1
    file = external.files[0]
    assert file.name == "scale_rows"
    assert file.file_name == "scale_rows.csv"
    assert file.fmt == "CSVWithNames"
    assert file.structure == ch_norm.SCALE_ROWS_EXTERNAL_STRUCTURE
    payload = file.data.decode("utf-8")
    assert payload.startswith(
        "symbol,TradeDate,ADV20_lag1,ATS20_lag1,q20,q80\n"
    )
    assert "600000,2024-01-02" in payload
    assert "000001" not in payload
    assert "510300" not in payload
    assert external.query_params == {
        "scale_rows_format": "CSVWithNames",
        "scale_rows_structure": ch_norm.SCALE_ROWS_EXTERNAL_STRUCTURE,
    }


def test_fake_client_checks_dynamic_join_external_rows_and_no_fee(
    monkeypatch,
) -> None:
    fake = _FakeClient()
    _RecordingExternalData.instances = []
    monkeypatch.setattr(ch_norm, "_get_ch_client", lambda: fake)
    a1_grid = ((0.5, 5.0),)
    a2_grid = ((0.25, 1.5),)

    result = ch_norm.fetch_dynamic_factor_aggregates(
        "2024-01-02",
        "2024-01-02",
        _scale_rows(),
        a1_grid=a1_grid,
        a2_grid=a2_grid,
        external_data_cls=_RecordingExternalData,
    )

    assert fake.closed
    assert len(fake.calls) == 2
    assert len(_RecordingExternalData.instances) == 2
    assert list(result.columns) == list(
        ch_norm.dynamic_result_columns(a1_grid, a2_grid)
    )
    for sql, external_data in fake.calls:
        assert "INNER JOIN scale_rows AS s" in sql
        assert "GROUP BY Symbol, TradeDate" in sql
        assert "sum(amt) AS total_amount" in sql
        assert "sum(Volume)" not in sql
        assert "sumIf(Volume" not in sql
        assert "> 0.5" in sql or "> 0.25" in sql
        assert "<= 5" in sql or "<= 1.5" in sql
        assert "fee" not in sql.lower()
        assert external_data.kwargs["file_name"] == "scale_rows.csv"
        assert external_data.kwargs["fmt"] == "CSVWithNames"
        assert (
            external_data.kwargs["structure"]
            == ch_norm.SCALE_ROWS_EXTERNAL_STRUCTURE
        )

    calls_by_exchange = {
        "SSE": next(
            call for call in fake.calls if "SSE_AL_TICK_EXG" in call[0]
        ),
        "SZSE": next(
            call for call in fake.calls if "SZSE_AL_TICK_EXG" in call[0]
        ),
    }
    sse_payload = calls_by_exchange["SSE"][1].kwargs["data"].decode()
    szse_payload = calls_by_exchange["SZSE"][1].kwargs["data"].decode()
    assert "600000,2024-01-02" in sse_payload
    assert "000001" not in sse_payload
    assert "000001,2024-01-02" in szse_payload
    assert "600000" not in szse_payload


def test_scale_rows_reject_duplicate_normalized_join_keys() -> None:
    rows = _scale_rows().iloc[[0]].copy()
    duplicate = rows.copy()
    duplicate["symbol"] = "600000"
    with pytest.raises(ValueError, match="duplicate normalized"):
        ch_norm.prepare_scale_rows_for_exchange(
            pd.concat([rows, duplicate], ignore_index=True), "SSE"
        )


def test_daily_scale_audit_checks_schema_duplicates_and_quantiles() -> None:
    row = {
        "symbol": "600000.SH",
        "TradeDate": "2024-01-02",
        "total_amount": 1_000_000.0,
        "positive_trade_count": 5,
        "daily_mean_trade_amount": 200_000.0,
        "q20": 10_000.0,
        "q30": 20_000.0,
        "q50": 50_000.0,
        "q70": 100_000.0,
        "q80": 200_000.0,
    }
    valid = ch_norm.audit_daily_scale_result(pd.DataFrame([row]))
    assert list(valid.columns) == list(ch_norm.DAILY_SCALE_COLUMNS)
    assert valid.loc[0, "positive_trade_count"] == 5

    with pytest.raises(ValueError, match="duplicate"):
        ch_norm.audit_daily_scale_result(pd.DataFrame([row, row]))
    bad_quantile = dict(row)
    bad_quantile["q70"] = 5_000.0
    with pytest.raises(ValueError, match="not monotonic"):
        ch_norm.audit_daily_scale_result(pd.DataFrame([bad_quantile]))
    with pytest.raises(ValueError, match="schema mismatch"):
        ch_norm.audit_daily_scale_result(
            pd.DataFrame([{**row, "unexpected": 1}])
        )


def test_dynamic_audit_rejects_duplicate_and_selected_over_total() -> None:
    a1_grid = ((1.0, 10.0),)
    a2_grid = ((0.5, 2.0),)
    columns = ch_norm.dynamic_result_columns(a1_grid, a2_grid)
    row = {column: 20.0 for column in columns[2:]}
    row.update(
        {
            "symbol": "600000.SH",
            "TradeDate": "2024-01-02",
            "total_amount": 100.0,
        }
    )
    frame = pd.DataFrame([row], columns=columns)
    valid = ch_norm.audit_dynamic_factor_result(
        frame, a1_grid=a1_grid, a2_grid=a2_grid
    )
    assert valid.loc[0, "total_amount"] == 100.0

    with pytest.raises(ValueError, match="duplicate"):
        ch_norm.audit_dynamic_factor_result(
            pd.concat([frame, frame], ignore_index=True),
            a1_grid=a1_grid,
            a2_grid=a2_grid,
        )
    selected_column = ch_norm.a1_selected_amount_column(1.0, 10.0)
    invalid = frame.copy()
    invalid.loc[0, selected_column] = 100.01
    with pytest.raises(ValueError, match="exceeds total_amount"):
        ch_norm.audit_dynamic_factor_result(
            invalid, a1_grid=a1_grid, a2_grid=a2_grid
        )


def test_grid_validation_and_fee_are_outside_data_layer() -> None:
    with pytest.raises(ValueError, match="lower < upper"):
        ch_norm.build_dynamic_factor_sql(
            "2024-01-02",
            "2024-01-02",
            "SSE",
            a1_grid=((2.0, 2.0),),
        )
    with pytest.raises(ValueError, match="duplicate"):
        ch_norm.dynamic_result_columns(
            ((1.0, 10.0), (1.0, 10.0)), ((0.5, 2.0),)
        )

    assert "fee" not in inspect.signature(
        ch_norm.fetch_daily_scale_primitive
    ).parameters
    assert "fee" not in inspect.signature(
        ch_norm.fetch_dynamic_factor_aggregates
    ).parameters

