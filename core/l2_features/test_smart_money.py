"""Unit tests for SmartMoney10d Option B knife (no DDB)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_cutting.smart_money import _q_one_window, compute_daily_smart_money_q


def test_q_high_score_high_price_raises_q():
    # Two minutes: smart minute is expensive → Q > 1
    score = np.array([10.0, 1.0])
    volume = np.array([50.0, 50.0])  # equal vol → top 20% needs first bar only? 20% of 100=20 → first 50 exceeds
    amount = np.array([50.0 * 110.0, 50.0 * 100.0])  # prices 110 and 100
    close = np.array([110.0, 100.0])
    q = _q_one_window(score, volume, amount, close, top_cumvol_pct=0.20)
    assert q > 1.0


def test_q_high_score_low_price_lowers_q():
    score = np.array([10.0, 1.0])
    volume = np.array([50.0, 50.0])
    amount = np.array([50.0 * 90.0, 50.0 * 100.0])
    close = np.array([90.0, 100.0])
    q = _q_one_window(score, volume, amount, close, top_cumvol_pct=0.20)
    assert q < 1.0


def test_amount_fallback_close_x_volume():
    score = np.array([5.0, 1.0])
    volume = np.array([40.0, 60.0])
    amount = np.array([0.0, 0.0])
    close = np.array([10.0, 20.0])
    q = _q_one_window(score, volume, amount, close, top_cumvol_pct=0.20)
    assert np.isfinite(q)


def test_rolling_lookback_option_b():
    """10 days of minutes; Q only from day 10 onward."""
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=12)
    for d in dates:
        for i, (sc, vol, px) in enumerate(
            [(2.0, 100.0, 10.0), (1.0, 100.0, 10.0), (0.5, 100.0, 10.0)]
        ):
            rows.append(
                {
                    "date": d,
                    "symbol": "600000",
                    "bartime": 9 * 3600 + 31 * 60 + i * 60,
                    "close": px,
                    "volume": vol,
                    "amount": px * vol,
                    "smart_score": sc,
                }
            )
    # bump last day smart minute price so Q moves
    for r in rows:
        if r["date"] == dates[-1] and r["smart_score"] == 2.0:
            r["close"] = 12.0
            r["amount"] = 12.0 * r["volume"]

    df = pd.DataFrame(rows)
    out = compute_daily_smart_money_q(df, lookback_days=10, min_minutes=10)
    assert out["date"].min() == dates[9]
    assert len(out) == 3  # days 10,11,12 (0-index 9,10,11)
    assert out["Q"].notna().all()


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        compute_daily_smart_money_q(pd.DataFrame({"date": [], "symbol": []}))
