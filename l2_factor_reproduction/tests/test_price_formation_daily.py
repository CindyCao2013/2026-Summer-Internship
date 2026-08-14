from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.backtest import prepare_factor_signal
from l2_factor_reproduction.python.price_formation_daily import (
    EXPECTED_CONTINUOUS_MINUTES,
    PRICE_FORMATION_DAILY_COLUMNS,
    _continuous_grid,
    close_auction_daily_sql,
    compute_price_formation_daily,
    prepare_price_formation_daily,
    price_formation_daily_sql,
)


def _minute_fixture(
    *,
    trade_date: str = "2024-06-03",
    symbol: str = "600000.SH",
    adjfactor: float = 2.0,
    constant_price: bool = False,
    zero_amount: bool = False,
    omit_structural_close_minutes: bool = True,
) -> pd.DataFrame:
    day = pd.Timestamp(trade_date)
    grid = _continuous_grid(day)
    index = np.arange(len(grid), dtype=float)
    if constant_price:
        close = np.full(len(grid), 100.0)
    else:
        close = 100.0 * np.exp(
            0.0002 * index + 0.001 * np.sin(index / 7.0)
        )
    open_px = np.r_[close[0], close[:-1]]
    high = np.maximum(open_px, close) + (0 if constant_price else 0.01)
    low = np.minimum(open_px, close) - (0 if constant_price else 0.01)
    volume = 1000.0 + (index % 17) * 10
    amount = volume * close
    if zero_amount:
        volume[:] = 0
        amount[:] = 0
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "date": day,
            "bartime": grid,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "adjfactor": adjfactor,
        }
    )
    if omit_structural_close_minutes:
        frame = frame.loc[
            ~frame["bartime"].dt.strftime("%H:%M").isin(
                ["14:57", "14:58", "14:59"]
            )
        ]
    auction = pd.DataFrame(
        {
            "symbol": [symbol],
            "date": [day],
            "bartime": [day + pd.Timedelta(hours=15)],
            "open": [close[-1] * 1.001],
            "high": [close[-1] * 1.001],
            "low": [close[-1] * 1.001],
            "close": [close[-1] * 1.001],
            "volume": [5000.0 if not zero_amount else 0.0],
            "amount": [
                close[-1] * 1.001 * 5000.0 if not zero_amount else 0.0
            ],
            "adjfactor": [adjfactor],
        }
    )
    return pd.concat([frame, auction], ignore_index=True)


def test_minute_unique_key_is_required() -> None:
    frame = _minute_fixture()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate symbol-minute"):
        compute_price_formation_daily(duplicated)


def test_continuous_grid_boundaries_are_frozen() -> None:
    grid = _continuous_grid(pd.Timestamp("2024-06-03"))
    assert len(grid) == EXPECTED_CONTINUOUS_MINUTES
    assert grid[0].strftime("%H:%M") == "09:30"
    assert grid[119].strftime("%H:%M") == "11:29"
    assert grid[120].strftime("%H:%M") == "13:00"
    assert grid[-1].strftime("%H:%M") == "14:59"
    assert not any(timestamp.strftime("%H:%M") == "11:30" for timestamp in grid)
    assert not any(timestamp.strftime("%H:%M") == "15:00" for timestamp in grid)


def test_no_forward_fill_across_lunch() -> None:
    frame = _minute_fixture()
    frame = frame.loc[
        frame["bartime"].dt.strftime("%H:%M") != "13:00"
    ]
    daily = compute_price_formation_daily(frame)
    assert np.isnan(daily.loc[0, "afternoon_return"])
    assert np.isnan(daily.loc[0, "lunch_gap_return"])


def test_structural_close_minutes_fill_price_state_only() -> None:
    daily = compute_price_formation_daily(_minute_fixture())
    assert daily.loc[0, "valid_minute_count"] == 237
    assert daily.loc[0, "imputed_price_minute_count"] == 3
    assert daily.loc[0, "coverage_ratio"] == pytest.approx(237 / 240)
    assert daily.loc[0, "valid_return_minute_count"] == 236


def test_amount_volume_vwap_units() -> None:
    frame = _minute_fixture(adjfactor=2.5)
    continuous = frame.loc[
        frame["bartime"].dt.strftime("%H:%M") != "15:00"
    ]
    expected = (
        (continuous["amount"] * continuous["adjfactor"]).sum()
        / continuous["volume"].sum()
    )
    daily = compute_price_formation_daily(frame)
    assert daily.loc[0, "daily_vwap"] == pytest.approx(expected)
    raw_vwap = continuous["amount"].sum() / continuous["volume"].sum()
    assert daily.loc[0, "daily_vwap"] == pytest.approx(raw_vwap * 2.5)


def test_adjusted_overnight_gap_uses_previous_available_close() -> None:
    first = _minute_fixture(trade_date="2024-06-03", adjfactor=2.0)
    second = _minute_fixture(trade_date="2024-06-04", adjfactor=2.1)
    daily = compute_price_formation_daily(
        pd.concat([first, second], ignore_index=True)
    )
    expected = np.log(
        daily.loc[1, "open_price"] / daily.loc[0, "continuous_close"]
    )
    assert np.isnan(daily.loc[0, "overnight_gap"])
    assert daily.loc[1, "overnight_gap"] == pytest.approx(expected)


def test_t_plus_one_signal_shift_is_exact() -> None:
    dates = pd.date_range("2024-06-03", periods=3, freq="B")
    factor = pd.DataFrame(
        {"600000.SH": [1.0, 2.0, 3.0]}, index=dates
    )
    returns = pd.DataFrame(
        {"600000.SH": [0.01, 0.02, 0.03]}, index=dates
    )
    mask = pd.DataFrame(1.0, index=dates, columns=["600000.SH"])
    signal, aligned = prepare_factor_signal(
        factor,
        start=dates.min(),
        end=dates.max(),
        mask=mask,
        signal_shift=1,
        ret_matrix=returns,
    )
    assert signal.iloc[:, 0].tolist() == [1.0, 2.0]
    pd.testing.assert_frame_equal(aligned, returns.loc[dates[1:]])


def test_query_uses_only_current_or_lagged_minutes() -> None:
    sql = price_formation_daily_sql("2024-06-03", "2024-06-03")
    assert "prev(closePx)" in sql
    assert "move(closePx,5)" in sql
    assert "move(closePx,-" not in sql
    assert "09:30:00" in sql
    assert "11:29:00" in sql
    assert "14:59:00" in sql
    auction_sql = close_auction_daily_sql(
        "2024-06-03", "2024-06-03"
    )
    assert "second(Bartime)==15:00:00" in auction_sql


def test_realized_moments_match_manual_calculation() -> None:
    frame = _minute_fixture()
    daily = compute_price_formation_daily(frame)
    continuous = frame.loc[
        frame["bartime"].dt.strftime("%H:%M").isin(
            [timestamp.strftime("%H:%M") for timestamp in _continuous_grid(pd.Timestamp("2024-06-03"))]
        )
    ].sort_values("bartime")
    close = continuous["close"] * continuous["adjfactor"]
    returns = np.log(close / close.shift(1)).dropna()
    rv = float((returns**2).sum())
    n = len(returns)
    skew = np.sqrt(n) * float((returns**3).sum()) / rv**1.5
    kurtosis = n * float((returns**4).sum()) / rv**2
    bipower = (
        np.pi
        / 2
        * float((returns.abs() * returns.abs().shift(1)).sum())
    )
    jump = max(rv - bipower, 0)
    assert daily.loc[0, "realized_variance"] == pytest.approx(rv)
    assert daily.loc[0, "realized_skewness"] == pytest.approx(skew)
    assert daily.loc[0, "realized_kurtosis"] == pytest.approx(kurtosis)
    assert daily.loc[0, "bipower_variation"] == pytest.approx(bipower)
    assert daily.loc[0, "jump_variation"] == pytest.approx(jump)


def test_clv_constant_range_is_safe() -> None:
    daily = compute_price_formation_daily(
        _minute_fixture(constant_price=True)
    )
    assert np.isnan(daily.loc[0, "close_location_value"])


def test_path_efficiency_is_bounded() -> None:
    daily = compute_price_formation_daily(_minute_fixture())
    assert 0 <= daily.loc[0, "path_efficiency"] <= 1


def test_hhi_and_amount_time_center_are_bounded() -> None:
    daily = compute_price_formation_daily(_minute_fixture())
    assert 0 < daily.loc[0, "volume_concentration_hhi"] <= 1
    assert 0 <= daily.loc[0, "amount_time_center"] <= 1


def test_zero_amount_is_safe_and_not_infinite() -> None:
    daily = compute_price_formation_daily(
        _minute_fixture(zero_amount=True)
    )
    assert np.isnan(daily.loc[0, "intraday_amihud"])
    assert daily.loc[0, "valid_amihud_minute_count"] == 0
    assert np.isnan(daily.loc[0, "return_per_amount"])
    assert np.isnan(daily.loc[0, "range_per_amount"])
    numeric = daily.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_prepare_rejects_duplicate_and_infinite_daily_rows() -> None:
    daily = compute_price_formation_daily(_minute_fixture())
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_price_formation_daily(
            pd.concat([daily, daily], ignore_index=True)
        )
    infinite = daily.copy()
    infinite.loc[0, "realized_variance"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        prepare_price_formation_daily(infinite)


def test_daily_primitive_schema_is_complete() -> None:
    daily = compute_price_formation_daily(_minute_fixture())
    assert list(daily.columns) == list(PRICE_FORMATION_DAILY_COLUMNS)
    assert daily.loc[0, "close_auction_price"] > 0
    assert daily.loc[0, "close_auction_return"] == pytest.approx(
        np.log(
            daily.loc[0, "close_auction_price"]
            / daily.loc[0, "continuous_close"]
        )
    )
