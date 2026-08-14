from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python import liquidity_impact_daily as lid


def test_session_grid_is_continuous_only_and_no_lookahead_window() -> None:
    session = lid._session_filter()
    assert "toHour(ExchTime) = 15" not in session
    assert "toHour(ExchTime) = 12" not in session
    assert "toHour(ExchTime) = 9" in session
    assert lid.EXPECTED_CONTINUOUS_MINUTES == 240


def test_exchange_rules_freeze_direction_and_symbol_filters() -> None:
    sse = lid.EXCHANGES["sse"]
    assert sse["tick_table"].endswith("LOCAL_SSE_AL_TICK_EXG")
    assert sse["book_table"].endswith("LOCAL_SSE_AL_SSL2_EXG")
    assert "LOCAL" in sse["tick_table"] and "Distributed" not in sse["tick_table"]
    assert sse["trade_filter"] == "Type = 'T'"
    assert sse["buy_cond"] == "BSFlag = 'B'"
    assert sse["neutral_cond"] == "BSFlag = 'N'"
    szse = lid.EXCHANGES["szse"]
    assert szse["trade_filter"] == "Type = '011' AND Category = 'F'"
    assert szse["buy_cond"] == "BidOrderNo > AskOrderNo"
    assert szse["sell_cond"] == "BidOrderNo < AskOrderNo"
    assert szse["neutral_cond"] == "BidOrderNo = AskOrderNo"
    assert "Price" in szse["amount_expr"] and "Volume" in szse["amount_expr"]


def test_size_buckets_are_frozen_boundaries() -> None:
    assert lid.SIZE_BUCKETS["small"] == "amt <= 10000"
    assert lid.SIZE_BUCKETS["mid"] == "amt > 40000 AND amt <= 200000"
    assert lid.SIZE_BUCKETS["large"] == "amt > 200000"
    assert lid.SIZE_BUCKETS["super_large"] == "amt > 1000000"


def test_sql_queries_reference_local_tables_and_book_array_filter() -> None:
    for exchange in lid.EXCHANGES:
        trade = lid.minute_trade_sql(exchange, "2024-06-01", "2024-07-01")
        book = lid.minute_book_sql(exchange, "2024-06-01", "2024-07-01")
        joined = lid.joined_minute_sql(exchange, "2024-06-01", "2024-07-01")
        daily = lid.daily_sql(exchange, "2024-06-01", "2024-07-01")
        for sql in (trade, book, joined, daily):
            assert "LOCAL_" in sql
            assert "Distributed" not in sql
        assert "length(BidPrices) > 0" in book
        assert "ifNull" in joined
        assert "arrayMap" in daily or "arrayReduce" in daily
        assert lid.query_sha256(sql) == lid.query_sha256(sql)


def _frame(exchange: str, symbol_raw: str, suffix: str) -> pd.DataFrame:
    dates = pd.date_range("2024-06-03", periods=3, freq="B")
    rows = []
    for index, trade_date in enumerate(dates):
        row = {column: 0.0 for column in lid.DAILY_COLUMNS}
        row.update(
            {
                "symbol": None,
                "TradeDate": trade_date,
                "expected_continuous_minutes": lid.EXPECTED_CONTINUOUS_MINUTES,
                "coverage_ratio": 0.98,
                "matched_minute_count": 236.0 + index,
                "trade_minute_count": 220.0,
                "valid_book_minute_count": 236.0,
                "directional_trade_share": 0.99,
                "neutral_trade_share": 0.01,
                "daily_amount": 1.0e8 * (index + 1),
                "daily_volume": 1.0e6,
                "trade_count": 500.0,
                "exchange": suffix,
            }
        )
        row["symbol_raw"] = symbol_raw
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["exchange"] = suffix
    return frame


def test_finalize_daily_attaches_suffix_and_orders_columns() -> None:
    sse = _frame("sse", "600000", ".SH")
    szse = _frame("szse", "000001", ".SZ")
    daily = lid.finalize_daily(
        [sse, szse], start="2024-06-01", end="2024-07-01"
    )
    assert list(daily.columns) == list(lid.DAILY_COLUMNS)
    assert set(daily["symbol"]) == {"600000.SH", "000001.SZ"}
    prepared = lid.prepare_liquidity_impact_daily(daily)
    assert not prepared.duplicated(["symbol", "TradeDate"]).any()


def test_prepare_rejects_inf_duplicates_and_out_of_range_shares() -> None:
    daily = lid.finalize_daily(
        [_frame("sse", "600000", ".SH")], start="2024-06-01", end="2024-07-01"
    )
    bad = daily.copy()
    bad.loc[0, "coverage_ratio"] = 1.5
    with pytest.raises(ValueError, match="coverage_ratio"):
        lid.prepare_liquidity_impact_daily(bad)
    bad = daily.copy()
    bad.loc[0, "daily_amount"] = np.inf
    with pytest.raises(ValueError, match="inf"):
        lid.prepare_liquidity_impact_daily(bad)
    duplicated = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        lid.prepare_liquidity_impact_daily(duplicated)
    missing = daily.drop(columns=["daily_amount"])
    with pytest.raises(ValueError, match="missing"):
        lid.prepare_liquidity_impact_daily(missing)


def test_prepare_clips_tiny_fp_overshoot_on_shares() -> None:
    daily = lid.finalize_daily(
        [_frame("sse", "600000", ".SH")], start="2024-06-01", end="2024-07-01"
    )
    daily.loc[0, "directional_trade_share"] = 1.0 + 1e-16
    prepared = lid.prepare_liquidity_impact_daily(daily)
    assert prepared["directional_trade_share"].max() <= 1.0
