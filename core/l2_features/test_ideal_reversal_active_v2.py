"""Unit tests for IdealReversal_ActiveV2 v3 (Thu hold + rolling gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.l2_features.bricks.active_size import (
    ACTIVE_SIZE_COL,
    compute_daily_active_size_concentration,
    concentration_one_day,
)
from factor_cutting.ideal_reversal_active_v2 import (
    build_reversal_factor,
    build_reversal_factor_v2,
    to_weekly_hold,
    to_weekly_thu_hold,
)


def test_concentration_top20_by_avg_size():
    amts = np.arange(1, 11, dtype=float)
    sizes = np.arange(1, 11, dtype=float)
    ratio = concentration_one_day(amts, sizes, top_pct=0.2)
    assert ratio == pytest.approx((10 + 9) / 55.0, abs=1e-9)


def test_daily_active_size_column_name():
    rows = [
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "active_buy_amt": 100,
            "avg_buy_size": 50,
        },
        {
            "date": "2024-06-03",
            "symbol": "000001",
            "active_buy_amt": 200,
            "avg_buy_size": 100,
        },
    ]
    out = compute_daily_active_size_concentration(pd.DataFrame(rows), min_minutes=1)
    assert ACTIVE_SIZE_COL in out.columns


def test_asc_gate_filters_low_asc():
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = pd.DataFrame(
        {"A": [10 + i for i in range(10)], "B": [20 - i for i in range(10)]},
        index=dates,
    )
    asc = pd.DataFrame({"A": 0.8, "B": 0.5}, index=dates)
    factor = build_reversal_factor(close, asc, window=5, min_cs_n=1, mode="asc_gate")
    assert factor.loc[dates[9], "A"] < 0
    assert pd.isna(factor.loc[dates[9], "B"])


def test_build_reversal_factor_v2_none_gate():
    dates = pd.bdate_range("2024-01-02", periods=25)
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 80, 25),
            "B": np.linspace(100, 120, 25),
        },
        index=dates,
    )
    asc = pd.DataFrame(0.5, index=dates, columns=["A", "B"])
    fac = build_reversal_factor_v2(
        close, asc, windows=[3, 5], gate_type="none", min_cs_n=1
    )
    assert fac.notna().any().any()
    # A fell → positive reversal contribution vs B
    assert fac.iloc[-1]["A"] > fac.iloc[-1]["B"]


def test_build_reversal_factor_v2_rolling_rank():
    dates = pd.bdate_range("2024-01-02", periods=30)
    rng = np.random.default_rng(42)
    close = pd.DataFrame(
        rng.normal(size=(30, 3)).cumsum(axis=0) + 100,
        index=dates,
        columns=list("ABC"),
    )
    asc = pd.DataFrame(
        {
            "A": np.linspace(0.7, 0.3, 30),
            "B": np.linspace(0.4, 0.6, 30),
            "C": np.linspace(0.5, 0.5, 30),
        },
        index=dates,
    )
    factor = build_reversal_factor_v2(
        close,
        asc,
        windows=[5],
        gate_type="rolling_rank",
        gate_threshold=0.5,
        gate_roll=10,
        min_cs_n=1,
    )
    assert factor.shape[0] > 0
    assert not factor.iloc[-5:].isna().all().all()


def test_to_weekly_thu_hold_ffills_after_signal():
    # 2024-06-03 Mon ... include Thu 2024-06-06
    dates = pd.bdate_range("2024-06-03", periods=10)
    fac = pd.DataFrame({"X": np.arange(1, 11, dtype=float)}, index=dates)
    weekly = to_weekly_thu_hold(fac, agg="mean")
    thu = pd.Timestamp("2024-06-06")
    assert thu in weekly.index
    # Mon-Thu values 1..4 mean = 2.5
    assert weekly.loc[thu, "X"] == pytest.approx(2.5)
    fri = pd.Timestamp("2024-06-07")
    if fri in weekly.index:
        assert weekly.loc[fri, "X"] == pytest.approx(2.5)


def test_weekly_friday_holds_between_fridays():
    idx = pd.bdate_range("2024-01-02", periods=10)
    fac = pd.DataFrame({"A": np.arange(10, dtype=float)}, index=idx)
    w = to_weekly_hold(fac, method="friday")
    for t in range(1, len(idx)):
        if idx[t].weekday() != 4:
            assert w.iloc[t, 0] == w.iloc[t - 1, 0]
