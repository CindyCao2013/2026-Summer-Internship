from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.ch_order_book import (
    EXPECTED_MINUTE_COUNT,
    ORDER_BOOK_DAILY_COLUMNS,
    ORDER_BOOK_TABLES,
    order_book_daily_sql,
    prepare_order_book_daily,
)
from l2_factor_reproduction.python.backtest import prepare_factor_signal


def _sql(exchange: str) -> str:
    table, suffix, name = next(
        item for item in ORDER_BOOK_TABLES if item[2] == exchange
    )
    return order_book_daily_sql(
        table=table,
        exchange_suffix=suffix,
        exchange=name,
        start="2024-06-28",
        end="2024-06-28",
    )


def _daily_fixture() -> pd.DataFrame:
    row = {column: 0.0 for column in ORDER_BOOK_DAILY_COLUMNS}
    row.update(
        {
            "symbol": "600000.SH",
            "TradeDate": pd.Timestamp("2024-06-28"),
            "source_exchange": "SSE",
            "valid_snapshot_count": 4800,
            "valid_minute_count": EXPECTED_MINUTE_COUNT,
            "expected_minute_count": EXPECTED_MINUTE_COUNT,
            "coverage_ratio": 1.0,
            "bid_depth_hhi_mean": 0.1,
            "ask_depth_hhi_mean": 0.1,
            "relative_spread_mean": 0.001,
            "relative_spread_p90": 0.002,
            "close_auction_valid": 1,
        }
    )
    return pd.DataFrame([row])


def test_sse_szse_schema_mapping_and_filters() -> None:
    sse = _sql("SSE")
    szse = _sql("SZSE")
    assert "cmds.`SSE_AL_SSL2_EXG`" in sse
    assert "cmds.`SZSE_AL_SSL2_EXG`" in szse
    assert "concat(Symbol, '.SH')" in sse
    assert "concat(Symbol, '.SZ')" in szse
    assert "Type = '010'" not in sse
    assert "Type = '010'" in szse
    assert "startsWith(Symbol, '6')" in sse
    assert "startsWith(Symbol, '300')" in szse


def test_sql_freezes_ten_levels_minute_last_and_daily_only() -> None:
    sql = _sql("SSE")
    assert "length(BidPrices) >= 10" in sql
    assert "length(AskVolumes) >= 10" in sql
    assert "arraySlice(BidVolumes, 1, 10)" in sql
    assert "argMax(obi_5, exch_time)" in sql
    assert "toStartOfMinute(exch_time)" in sql
    assert "expected_minute_count" in sql
    assert "countIf(is_close_auction = 0) / 240." in sql
    assert "toHour(ExchTime) = 15 AND toMinute(ExchTime) = 0" in sql
    assert "BidVWAP" not in sql
    assert "AskVWAP" not in sql
    assert "bid_vwap_num / bid_vwap_den" in sql
    assert "toFloat64(AskPrices[1]) >= toFloat64(BidPrices[1])" in sql
    assert "arrayAll(" in sql
    assert "bid_depth_1 + ask_depth_1 > 0" in sql


def test_snapshot_formulas_manual_bounds() -> None:
    bid_prices = np.arange(99.0, 89.0, -1.0)
    ask_prices = np.arange(101.0, 111.0, 1.0)
    bid_volume = np.full(10, 100.0)
    ask_volume = np.full(10, 50.0)
    bid_total = bid_volume.sum()
    ask_total = ask_volume.sum()
    obi_10 = (bid_total - ask_total) / (bid_total + ask_total)
    assert obi_10 == pytest.approx(1.0 / 3.0)
    assert -1 <= obi_10 <= 1

    weights = 1.0 / np.arange(1.0, 11.0)
    weighted_denominator = (bid_volume * weights).sum() + (
        ask_volume * weights
    ).sum()
    assert weighted_denominator > 0

    bid1, ask1 = bid_prices[0], ask_prices[0]
    mid = (bid1 + ask1) / 2.0
    spread = (ask1 - bid1) / mid
    assert spread >= 0
    microprice = (
        ask1 * bid_volume[0] + bid1 * ask_volume[0]
    ) / (bid_volume[0] + ask_volume[0])
    assert bid1 <= microprice <= ask1

    bid_hhi = np.square(bid_volume / bid_total).sum()
    ask_hhi = np.square(ask_volume / ask_total).sum()
    assert bid_hhi == pytest.approx(0.1)
    assert ask_hhi == pytest.approx(0.1)

    bid_x = np.abs(bid_prices - mid) / mid
    bid_y = np.cumsum(bid_volume) / bid_total
    assert bid_y[-1] == pytest.approx(1.0)
    expected_slope = np.polyfit(bid_x, bid_y, 1)[0]
    n = len(bid_x)
    manual_slope = (
        n * np.sum(bid_x * bid_y) - bid_x.sum() * bid_y.sum()
    ) / (n * np.sum(bid_x**2) - bid_x.sum() ** 2)
    assert manual_slope == pytest.approx(expected_slope)


def test_prepare_daily_validates_keys_ranges_and_coverage() -> None:
    frame = prepare_order_book_daily(_daily_fixture())
    assert list(frame.columns) == list(ORDER_BOOK_DAILY_COLUMNS)
    assert frame.loc[0, "coverage_ratio"] == 1.0

    duplicated = pd.concat([_daily_fixture(), _daily_fixture()])
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_order_book_daily(duplicated)

    bad_obi = _daily_fixture()
    bad_obi.loc[0, "obi_5_mean"] = 1.01
    with pytest.raises(ValueError, match="obi_5_mean"):
        prepare_order_book_daily(bad_obi)

    bad_coverage = _daily_fixture()
    bad_coverage.loc[0, "coverage_ratio"] = 0.9
    with pytest.raises(ValueError, match="coverage_ratio"):
        prepare_order_book_daily(bad_coverage)

    infinite = _daily_fixture()
    infinite.loc[0, "bid_depth_slope_mean"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        prepare_order_book_daily(infinite)


def test_minute_last_sampling_is_unique_and_not_raw_row_mean() -> None:
    first_minute = pd.DataFrame(
        {
            "symbol": "600000.SH",
            "time": pd.date_range(
                "2024-06-28 09:30:00", periods=100, freq="500ms"
            ),
            "metric": 1.0,
        }
    )
    second_minute = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "time": [pd.Timestamp("2024-06-28 09:31:10")],
            "metric": [3.0],
        }
    )
    raw = pd.concat([first_minute, second_minute], ignore_index=True)
    raw["minute"] = raw["time"].dt.floor("min")
    minute_last = (
        raw.sort_values("time")
        .groupby(["symbol", "minute"], as_index=False)
        .last()
    )
    assert not minute_last.duplicated(["symbol", "minute"]).any()
    assert minute_last["metric"].mean() == pytest.approx(2.0)
    assert raw["metric"].mean() != pytest.approx(2.0)


def test_opening_and_closing_windows_do_not_overlap() -> None:
    minute_index = np.arange(EXPECTED_MINUTE_COUNT)
    opening = (minute_index >= 0) & (minute_index < 30)
    closing = (minute_index >= 210) & (minute_index < 240)
    assert opening.sum() == 30
    assert closing.sum() == 30
    assert not np.any(opening & closing)


def test_backtest_signal_is_shifted_exactly_one_day() -> None:
    dates = pd.date_range("2024-06-24", periods=3, freq="B")
    factor = pd.DataFrame(
        {"600000.SH": [1.0, 2.0, 3.0]},
        index=dates,
    )
    returns = pd.DataFrame(
        {"600000.SH": [0.01, 0.02, 0.03]},
        index=dates,
    )
    mask = pd.DataFrame(1.0, index=dates, columns=["600000.SH"])
    signal, aligned_returns = prepare_factor_signal(
        factor,
        start=dates.min(),
        end=dates.max(),
        mask=mask,
        signal_shift=1,
        ret_matrix=returns,
    )
    assert list(signal.index) == list(dates[1:])
    assert signal.iloc[0, 0] == 1.0
    assert signal.iloc[1, 0] == 2.0
    pd.testing.assert_frame_equal(
        aligned_returns,
        returns.loc[dates[1:]],
    )
