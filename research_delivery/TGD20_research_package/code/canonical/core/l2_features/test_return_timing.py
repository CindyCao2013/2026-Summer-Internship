"""Unit tests for Temporal Feature Layer v1 — return timing (Gu/Gd)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.l2_features.return_timing import (
    compute_down_time_center,
    compute_minute_returns,
    compute_timing_centers_daily,
    compute_timing_centers_from_arrays,
    compute_up_time_center,
    prepare_tgd_inputs,
    trading_minute_index,
)


def test_trading_minute_index_skips_lunch():
    times = [
        "2024-01-02 09:31:00",
        "2024-01-02 11:30:00",
        "2024-01-02 13:01:00",
        "2024-01-02 15:00:00",
        "2024-01-02 12:00:00",  # lunch → NaN
    ]
    idx = trading_minute_index(times)
    assert idx[0] == 0
    assert idx[1] == 119  # 09:31..11:30 = 120 bars, last index 119
    assert idx[2] == 120  # first PM bar
    assert idx[3] == 239
    assert np.isnan(idx[4])


def test_minute_returns_basic_and_bad_prices():
    close = np.array([100.0, 101.0, 100.0, 0.0, 102.0])
    r = compute_minute_returns(close)
    assert np.isnan(r[0])
    assert pytest.approx(r[1], rel=1e-9) == 0.01
    assert pytest.approx(r[2], rel=1e-9) == 100.0 / 101.0 - 1.0
    assert np.isnan(r[3])  # curr<=0
    assert np.isnan(r[4])  # prev<=0


def test_up_center_simple():
    # equal up returns at t=0 and t=10 → Gu=5
    r = np.array([0.01, np.nan, 0.01])
    t = np.array([0.0, 5.0, 10.0])
    assert compute_up_time_center(r, t) == pytest.approx(5.0)
    assert np.isnan(compute_down_time_center(r, t))


def test_down_center_weighted():
    # down -0.01 at t=0, -0.03 at t=10 → weights 1:3 → Gd=7.5
    r = np.array([-0.01, 0.02, -0.03])
    t = np.array([0.0, 5.0, 10.0])
    assert compute_down_time_center(r, t) == pytest.approx(7.5)
    assert compute_up_time_center(r, t) == pytest.approx(5.0)


def test_flat_day_both_nan():
    r = np.zeros(10)
    t = np.arange(10, dtype=float)
    assert np.isnan(compute_up_time_center(r, t))
    assert np.isnan(compute_down_time_center(r, t))
    ctr = compute_timing_centers_from_arrays(r, t, symbol="FLAT")
    assert np.isnan(ctr.Gu) and np.isnan(ctr.Gd)
    assert ctr.n_flat == 10
    assert ctr.n_up == 0 and ctr.n_down == 0


def test_all_positive_gd_nan():
    r = np.array([0.01, 0.02, 0.01])
    t = np.array([1.0, 2.0, 3.0])
    assert np.isfinite(compute_up_time_center(r, t))
    assert np.isnan(compute_down_time_center(r, t))


def test_all_negative_gu_nan():
    r = np.array([-0.01, -0.02, -0.01])
    t = np.array([1.0, 2.0, 3.0])
    assert np.isnan(compute_up_time_center(r, t))
    assert np.isfinite(compute_down_time_center(r, t))


def test_missing_bars_ignored_in_weights():
    r = np.array([0.01, np.nan, 0.01, -0.02, np.nan])
    t = np.array([0.0, 1.0, 10.0, 20.0, 30.0])
    # Gu: equal weight 0 and 10 → 5; missing t=1 dropped
    assert compute_up_time_center(r, t) == pytest.approx(5.0)
    assert compute_down_time_center(r, t) == pytest.approx(20.0)
    ctr = compute_timing_centers_from_arrays(r, t)
    assert ctr.n_missing == 2


def test_compute_timing_centers_daily_long():
    # two symbols, synthetic closes on valid session bars
    schedule = ["09:31:00", "10:00:00", "13:01:00", "14:00:00"]
    rows = []
    for sym, path in [("AAA", [100, 101, 102, 101]), ("BBB", [50, 49, 48, 49])]:
        for clock, c in zip(schedule, path):
            rows.append(
                {
                    "date": "2024-06-03",
                    "symbol": sym,
                    "bartime": f"2024-06-03 {clock}",
                    "close": float(c),
                }
            )
    df = pd.DataFrame(rows)
    out = compute_timing_centers_daily(df)
    assert set(out["symbol"]) == {"AAA", "BBB"}
    aaa = out[out["symbol"] == "AAA"].iloc[0]
    bbb = out[out["symbol"] == "BBB"].iloc[0]
    assert np.isfinite(aaa["Gu"])
    assert np.isfinite(aaa["Gd"])  # last bar down
    assert np.isfinite(bbb["Gd"])
    assert np.isfinite(bbb["Gu"])  # last bar up


def test_prepare_tgd_inputs_passthrough():
    daily = pd.DataFrame(
        {"date": ["2024-01-02"], "symbol": ["X"], "Gu": [10.0], "Gd": [20.0]}
    )
    out = prepare_tgd_inputs(daily)
    assert list(out.columns) == list(daily.columns)
    assert out.attrs.get("tgd_stage") == "centers_only"
