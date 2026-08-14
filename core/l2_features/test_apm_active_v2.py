"""Unit tests for APM_ActiveV2 active_pressure (no DDB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.l2_features.bricks.active_pressure import (
    compute_daily_active_pressure,
    minute_raw_apm,
    smooth_active_pressure,
)
from factor_cutting.apm_active_v2 import ewm_smooth_daily, to_weekly_thu_hold
from factor_cutting.apm_active_v2_session_cut import (
    CLOSE_WINDOW_END,
    CLOSE_WINDOW_START,
    OPEN_WINDOW_END,
    OPEN_WINDOW_START,
    _window_imbalance,
    bartime_to_seconds,
    compute_daily_apm_session_cut,
)


def test_minute_raw_apm_range_and_nan():
    buy = pd.Series([100.0, 0.0, 50.0])
    sell = pd.Series([50.0, 0.0, 50.0])
    raw = minute_raw_apm(buy, sell)
    assert raw.iloc[0] == pytest.approx(1.0 / 3.0)
    assert np.isnan(raw.iloc[1])
    assert raw.iloc[2] == pytest.approx(0.0)


def test_daily_amount_weighted_pressure():
    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "active_buy_amt": 100,
            "active_sell_amt": 0,
            "amount": 100,
        },
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "active_buy_amt": 0,
            "active_sell_amt": 100,
            "amount": 300,
        },
    ]
    daily = compute_daily_active_pressure(pd.DataFrame(rows), min_minutes=1)
    # weights 100 vs 300 → (1*100 + (-1)*300) / 400 = -0.5
    assert daily["apm_raw"].iloc[0] == pytest.approx(-0.5)


def test_ewm_smooth():
    dates = pd.bdate_range("2024-01-02", periods=10)
    daily = pd.DataFrame(
        {
            "date": dates,
            "symbol": "test",
            "apm_raw": np.arange(0.1, 1.1, 0.1),
        }
    )
    out = ewm_smooth_daily(daily, span=5, min_periods=3)
    assert "apm_smooth" in out.columns
    assert out["apm_smooth"].notna().sum() >= 3
    out2 = smooth_active_pressure(daily)
    assert out2["apm_smooth"].notna().sum() >= 3


def test_weekly_thu_hold_places_on_thursday():
    idx = pd.bdate_range("2024-06-03", periods=10)  # Mon..
    fac = pd.DataFrame({"s": np.arange(len(idx), dtype=float)}, index=idx)
    weekly = to_weekly_thu_hold(fac, agg="mean")
    assert weekly.notna().any().any()
    # first week Mon-Thu mean on Thursday
    thu = idx[idx.weekday == 3][0]
    mon_thu = fac.loc[idx[(idx >= idx[0]) & (idx <= thu)], "s"].mean()
    assert weekly.loc[thu, "s"] == pytest.approx(mon_thu)


def test_session_cut_legacy_still_works():
    bt = pd.Series(
        [
            pd.Timestamp("1970-01-01 09:30:00"),
            pd.Timestamp("1970-01-01 14:30:00"),
        ]
    )
    sec = bartime_to_seconds(bt)
    assert int(sec.iloc[0]) == OPEN_WINDOW_START
    assert int(sec.iloc[1]) == CLOSE_WINDOW_START

    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "bartime": pd.Timestamp("1970-01-01 09:35:00"),
            "active_buy_amt": 100,
            "active_sell_amt": 50,
            "amount": 200,
        },
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "bartime": pd.Timestamp("1970-01-01 14:45:00"),
            "active_buy_amt": 80,
            "active_sell_amt": 70,
            "amount": 200,
        },
    ]
    imb = _window_imbalance(pd.DataFrame(rows), OPEN_WINDOW_START, OPEN_WINDOW_END)
    assert imb == pytest.approx(0.25)
    daily = compute_daily_apm_session_cut(pd.DataFrame(rows), min_minutes=1)
    assert "mo_imb" in daily.columns
    assert np.isfinite(daily["apm_raw"].iloc[0])


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        compute_daily_active_pressure(pd.DataFrame({"date": [], "symbol": []}))


def test_close_window_excludes_1500():
    rows = [
        {
            "bartime": pd.Timestamp("1970-01-01 14:45:00"),
            "active_buy_amt": 100,
            "active_sell_amt": 0,
            "amount": 100,
        },
        {
            "bartime": pd.Timestamp("1970-01-01 15:00:00"),
            "active_buy_amt": 1000,
            "active_sell_amt": 0,
            "amount": 1000,
        },
    ]
    imb = _window_imbalance(pd.DataFrame(rows), CLOSE_WINDOW_START, CLOSE_WINDOW_END)
    assert imb == pytest.approx(1.0)


def test_session_weight_assignment():
    from factor_cutting.apm_active_v2 import assign_session_weight

    bartime = pd.to_datetime(
        [
            "1970-01-01 09:30:00",
            "1970-01-01 10:00:00",
            "1970-01-01 11:00:00",
            "1970-01-01 14:30:00",
            "1970-01-01 14:59:00",
            "1970-01-01 15:00:00",
        ]
    )
    w = assign_session_weight(bartime)
    assert w[0] == pytest.approx(0.6)
    assert w[1] == pytest.approx(1.0)  # 10:00 mid
    assert w[2] == pytest.approx(1.0)
    assert w[3] == pytest.approx(1.5)
    assert w[4] == pytest.approx(1.5)
    assert w[5] == pytest.approx(1.0)  # 15:00 excluded from close window


def test_compute_daily_apm_session_weighted():
    from factor_cutting.apm_active_v2 import compute_daily_apm_session

    # same raw_apm=1/3 both minutes; unequal session weights must still give 1/3
    minutes = pd.DataFrame(
        {
            "date": ["2024-06-03", "2024-06-03"],
            "symbol": ["A", "A"],
            "active_buy_amt": [100.0, 200.0],
            "active_sell_amt": [50.0, 100.0],
            "amount": [200.0, 400.0],
            "bartime": pd.to_datetime(
                ["1970-01-01 09:30:00", "1970-01-01 14:30:00"]
            ),
        }
    )
    res = compute_daily_apm_session(minutes, min_minutes=1)
    assert len(res) == 1
    assert res["apm_raw"].iloc[0] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_compute_daily_apm_smart_filters_small_buys():
    from factor_cutting.apm_active_v2 import compute_daily_apm_smart

    rows = []
    # 10 prior days: avg_buy_size ≈ 50 → roll_mean after lag≈50
    for i, d in enumerate(pd.bdate_range("2024-05-01", periods=10)):
        rows.append(
            {
                "date": d,
                "symbol": "A",
                "active_buy_amt": 50.0,
                "active_sell_amt": 50.0,
                "amount": 100.0,
                "active_buy_count": 1.0,  # size=50
            }
        )
    # target day: one small (size=40) one large (size=120 > 50*1.2)
    target = pd.Timestamp("2024-05-15")
    rows.append(
        {
            "date": target,
            "symbol": "A",
            "active_buy_amt": 40.0,
            "active_sell_amt": 0.0,
            "amount": 100.0,
            "active_buy_count": 1.0,
        }
    )
    rows.append(
        {
            "date": target,
            "symbol": "A",
            "active_buy_amt": 120.0,
            "active_sell_amt": 0.0,
            "amount": 200.0,
            "active_buy_count": 1.0,
        }
    )
    minutes = pd.DataFrame(rows)
    daily = compute_daily_apm_smart(
        minutes, size_mult=1.2, lookback_days=5, roll_min_periods=3, min_minutes=1
    )
    hit = daily[daily["date"] == target]
    assert len(hit) == 1
    # only large buy kept → raw_apm = 1.0
    assert hit["apm_raw"].iloc[0] == pytest.approx(1.0)


def test_delta_apm_wide():
    from factor_cutting.apm_active_v2 import delta_apm_wide, smooth_delta_wide

    idx = pd.bdate_range("2024-01-02", periods=8)
    raw = pd.DataFrame({"s": np.arange(8, dtype=float)}, index=idx)
    d = delta_apm_wide(raw, lag=3)
    assert np.isnan(d.iloc[2, 0])
    assert d.iloc[3, 0] == pytest.approx(3.0)
    sm = smooth_delta_wide(d, span=5, min_periods=2)
    assert sm.notna().sum().sum() >= 2


def test_dynamic_size_threshold_is_lagged_quantile():
    from factor_cutting.apm_active_v2 import compute_dynamic_size_threshold

    dates = pd.bdate_range("2024-01-02", periods=10)
    daily_avg = pd.DataFrame(
        {
            "symbol": "A",
            "date": dates,
            "avg_buy_size": np.linspace(100, 190, 10),
        }
    )
    thr = compute_dynamic_size_threshold(
        daily_avg, lookback=5, quantile=0.8, roll_min_periods=5
    )
    assert pd.isna(thr.iloc[0])
    assert pd.isna(thr.iloc[4])
    # day index 5 uses days 0..4; 80% quantile of linspace(100,190,10)[:5]
    hist = daily_avg["avg_buy_size"].iloc[0:5]
    expected = float(hist.quantile(0.8))
    assert thr.iloc[5] == pytest.approx(expected)
    assert thr.iloc[5] != pytest.approx(daily_avg["avg_buy_size"].iloc[5])


def test_compute_smart_apm_v2_buy_sell_split():
    from factor_cutting.apm_active_v2 import compute_daily_smart_apm_v2

    rows = []
    # history days: size ~100 so threshold ~100
    for d in pd.bdate_range("2024-05-01", periods=8):
        rows.append(
            {
                "date": d,
                "symbol": "X",
                "active_buy_amt": 50.0,
                "active_sell_amt": 50.0,
                "amount": 100.0,
                "active_buy_count": 1.0,
            }
        )
    target = pd.Timestamp("2024-05-13")
    # big buy minute + big sell minute (size 200 >= threshold)
    rows.append(
        {
            "date": target,
            "symbol": "X",
            "active_buy_amt": 600.0,
            "active_sell_amt": 0.0,
            "amount": 600.0,
            "active_buy_count": 3.0,  # size=200
        }
    )
    rows.append(
        {
            "date": target,
            "symbol": "X",
            "active_buy_amt": 0.0,
            "active_sell_amt": 300.0,
            "amount": 300.0,
            "active_buy_count": 1.5,  # size=0? buy_amt/count=0 — use count so size from buy
            # for sell-only minute avg_buy_size may be 0; force via avg_buy_size column
        }
    )
    minutes = pd.DataFrame(rows)
    # override last row sizes so both pass filter
    minutes.loc[minutes.index[-1], "avg_buy_size"] = 200.0
    minutes.loc[minutes.index[-2], "avg_buy_size"] = 200.0
    # earlier rows need avg_buy_size for history
    for i in range(len(minutes) - 2):
        minutes.loc[minutes.index[i], "avg_buy_size"] = 100.0

    daily = compute_daily_smart_apm_v2(
        minutes, lookback=5, quantile=0.8, roll_min_periods=3, min_minutes=1
    )
    hit = daily[daily["date"] == target]
    assert len(hit) == 1
    # buy_int = 600/900, sell_int = 300/900, raw = 300/900
    assert hit["buy_intensity"].iloc[0] == pytest.approx(600.0 / 900.0)
    assert hit["sell_intensity"].iloc[0] == pytest.approx(300.0 / 900.0)
    assert hit["apm_raw"].iloc[0] == pytest.approx(300.0 / 900.0)


def test_asc_cs_gate():
    from factor_cutting.apm_active_v2 import apply_asc_cs_gate

    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-06-03")] * 2,
            "symbol": ["A", "B"],
            "apm_raw": [0.5, 0.4],
        }
    )
    asc = pd.DataFrame(
        {"A": [0.1], "B": [0.9]},
        index=[pd.Timestamp("2024-06-03")],
    )
    out = apply_asc_cs_gate(daily, asc, min_rank=0.5)
    # A has lower ASC → rank 0.5 with average method for 2 names? 
    # ranks: A=0.5, B=1.0 with pct=True average for 2 items → A gets 0.5, B gets 1.0
    # min_rank=0.5 → A: asc_rank < 0.5 is False if ==0.5; use strict < so A kept
    # For clearer gate: use 3 stocks
    daily3 = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-06-03")] * 3,
            "symbol": ["A", "B", "C"],
            "apm_raw": [1.0, 1.0, 1.0],
        }
    )
    asc3 = pd.DataFrame(
        {"A": [0.1], "B": [0.5], "C": [0.9]},
        index=[pd.Timestamp("2024-06-03")],
    )
    out3 = apply_asc_cs_gate(daily3, asc3, min_rank=0.5)
    # pct ranks ~ 1/3, 2/3, 1.0 → A gated
    a = out3[out3["symbol"] == "A"]["apm_raw"].iloc[0]
    c = out3[out3["symbol"] == "C"]["apm_raw"].iloc[0]
    assert np.isnan(a)
    assert np.isfinite(c)
