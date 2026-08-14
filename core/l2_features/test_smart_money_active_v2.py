"""Unit tests for SmartMoneyActiveV2 (no DDB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_cutting.smart_money_active_v2 import (
    _concentration_one_day,
    apply_minute_qc,
    compute_daily_smart_active,
    cs_zscore_min_n,
    ewm_smooth_daily,
    mad_winsorize_cs,
)


def test_concentration_top20_equals_sum_ratio():
    # 10 bars: amounts 1..10; top 20% = top 2 → 10+9=19 / 55
    am = np.arange(1, 11, dtype=float)
    c = _concentration_one_day(am, top_pct=0.20)
    assert abs(c - 19.0 / 55.0) < 1e-9


def test_concentration_empty_or_zero():
    assert np.isnan(_concentration_one_day(np.array([])))
    assert np.isnan(_concentration_one_day(np.zeros(5)))


def test_daily_smart_long_gt_short_when_buy_concentrated():
    rows = []
    d = pd.Timestamp("2024-06-03")
    # buy: 2 huge bars; sell: mild gradient (less concentrated)
    sell_amts = [10 + i for i in range(10)]  # 10..19
    for i in range(10):
        buy = 100.0 if i < 2 else 1.0
        rows.append(
            {
                "date": d,
                "symbol": "600000",
                "bartime": 9 * 3600 + 30 * 60 + i * 60,
                "active_buy_amt": buy,
                "active_sell_amt": float(sell_amts[i]),
            }
        )
    daily = compute_daily_smart_active(pd.DataFrame(rows), min_minutes=5)
    assert len(daily) == 1
    assert daily["smart_long"].iloc[0] > daily["smart_short"].iloc[0]


def test_ewm_produces_raw_diff():
    dates = pd.bdate_range("2024-01-02", periods=15)
    rows = []
    for d in dates:
        rows.append(
            {
                "date": d,
                "symbol": "600000",
                "smart_long": 0.6,
                "smart_short": 0.4,
            }
        )
    out = ewm_smooth_daily(pd.DataFrame(rows), span=10, min_periods=5)
    assert "smart_raw" in out.columns
    # after min_periods, raw ≈ 0.2
    assert out["smart_raw"].iloc[-1] == pytest.approx(0.2, abs=1e-6)
    assert out["smart_raw"].isna().sum() >= 4  # early NaN from min_periods


def test_apply_minute_qc_adjfactor_and_bad_ret():
    rows = [
        {
            "date": pd.Timestamp("2024-06-03"),
            "symbol": "600000",
            "bartime": 1,
            "close": 10.0,
            "amount": 100.0,
            "active_buy_amt": 40.0,
            "active_sell_amt": 30.0,
            "active_buy_count": 2,
            "active_sell_count": 3,
            "adjfactor": 2.0,
        },
        {
            "date": pd.Timestamp("2024-06-03"),
            "symbol": "600000",
            "bartime": 2,
            "close": 15.0,  # +50% after adj → drop
            "amount": 100.0,
            "active_buy_amt": 40.0,
            "active_sell_amt": 30.0,
            "active_buy_count": 2,
            "active_sell_count": 3,
            "adjfactor": 2.0,
        },
    ]
    out = apply_minute_qc(pd.DataFrame(rows), max_abs_ret=0.20)
    assert len(out) == 1
    assert out["close"].iloc[0] == pytest.approx(20.0)
    assert out["active_buy_amt"].iloc[0] == pytest.approx(80.0)
    assert "avg_buy_size" in out.columns


def test_mad_and_zscore():
    # Date × Symbol; day0 has a clear outlier
    idx = pd.date_range("2024-01-02", periods=1)
    fac = pd.DataFrame(
        [[0.0, 0.1, 0.0, 0.05, 10.0]],
        index=idx,
        columns=list("abcde"),
    )
    w = mad_winsorize_cs(fac, n_sig=5)
    assert float(w.iloc[0].max()) < 10.0
    z = cs_zscore_min_n(fac, min_n=3)
    assert abs(float(z.iloc[0].mean())) < 1e-8


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        compute_daily_smart_active(pd.DataFrame({"date": [], "symbol": []}))
