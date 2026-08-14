# ============================================================
# core/l2_features/test_ideal_amplitude_active_v2.py
# 单元测试
# ============================================================
"""Unit tests for IdealAmplitude_ActiveV2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_cutting.ideal_amplitude_active_v2 import (
    _active_net_volatility,
    _realized_amplitude,
    compute_daily_amplitude,
    ewm_smooth_daily,
)


def test_realized_amplitude():
    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "bartime": 1,
            "high": 12.0,
            "low": 10.0,
            "open": 10.0,
        },
    ]
    amp = _realized_amplitude(pd.DataFrame(rows))
    assert amp == pytest.approx(0.2, abs=1e-9)


def test_net_volatility():
    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "bartime": 1,
            "active_buy_amt": 100,
            "active_sell_amt": 50,
            "amount": 200,
        },
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "bartime": 2,
            "active_buy_amt": 60,
            "active_sell_amt": 90,
            "amount": 200,
        },
    ]
    vol = _active_net_volatility(pd.DataFrame(rows))
    # ratios: 0.25, -0.15 → sample std of [0.25, -0.15]
    expected = float(pd.Series([0.25, -0.15]).std())
    assert vol == pytest.approx(expected, abs=1e-12)
    assert vol > 0


def test_net_volatility_constant_is_nan_or_zero():
    """全天净额比例为常数 → 波动为 0 → amp_raw 应为 NaN。"""
    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "high": 11,
            "low": 10,
            "open": 10,
            "active_buy_amt": 100,
            "active_sell_amt": 50,
            "amount": 200,
        },
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "high": 11,
            "low": 10,
            "open": 10,
            "active_buy_amt": 150,
            "active_sell_amt": 75,
            "amount": 300,
        },
    ]
    daily = compute_daily_amplitude(pd.DataFrame(rows), min_minutes=1)
    assert daily.shape[0] == 1
    # ratios both 0.25 → std=0 → amp_raw NaN
    assert daily.loc[0, "active_net_vol"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(daily.loc[0, "amp_raw"])


def test_daily_amplitude():
    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "high": 11,
            "low": 10,
            "open": 10,
            "active_buy_amt": 100,
            "active_sell_amt": 80,
            "amount": 200,
        },
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "high": 11,
            "low": 10,
            "open": 10,
            "active_buy_amt": 120,
            "active_sell_amt": 70,
            "amount": 250,
        },
    ]
    daily = compute_daily_amplitude(pd.DataFrame(rows), min_minutes=1)
    assert daily.shape[0] == 1
    assert "realized_amp" in daily.columns
    assert "active_net_vol" in daily.columns
    assert "amp_raw" in daily.columns
    assert daily.loc[0, "realized_amp"] == pytest.approx(0.1, abs=1e-9)
    assert np.isfinite(daily.loc[0, "amp_raw"])


def test_ewm_smooth():
    dates = pd.bdate_range("2024-01-02", periods=10)
    daily = pd.DataFrame(
        {
            "date": dates,
            "symbol": "test",
            "amp_raw": np.linspace(0.5, 2.0, 10),
        }
    )
    out = ewm_smooth_daily(daily, span=5, min_periods=3)
    assert "amp_smooth" in out.columns
    assert out["amp_smooth"].notna().sum() >= 3
