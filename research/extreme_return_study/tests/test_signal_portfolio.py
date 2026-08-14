"""Unit tests for signal ranking and portfolio construction (no DB)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[3]
for p in (str(_SRC.parent), str(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.signal import extreme_signal_panel, select_extreme_masks  # noqa: E402
from src.portfolio import (  # noqa: E402
    equal_weight_from_mask,
    overlapping_weights,
    portfolio_turnover,
)
from src.backtest import apply_cost, daily_pnl  # noqa: E402


def _toy_returns() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    # 5 stocks; day0 returns sorted so bottom2 / top2 are known
    data = {
        "A": [0.01, -0.05, 0.02, 0.00, -0.01],
        "B": [-0.08, 0.03, -0.01, 0.04, 0.02],
        "C": [0.05, -0.02, 0.06, -0.03, 0.01],
        "D": [-0.03, 0.07, -0.04, 0.01, -0.06],
        "E": [0.02, 0.01, 0.00, -0.05, 0.03],
    }
    return pd.DataFrame(data, index=idx)


def test_select_extreme_masks_bottom_top():
    ret = _toy_returns()
    loser, winner = select_extreme_masks(ret, n=2)
    # Day 0: returns A=0.01,B=-0.08,C=0.05,D=-0.03,E=0.02
    # bottom2: B, D; top2: C, E (or A?) — C=0.05, E=0.02, A=0.01 → C,E
    d0 = ret.index[0]
    assert set(loser.loc[d0][loser.loc[d0]].index) == {"B", "D"}
    assert set(winner.loc[d0][winner.loc[d0]].index) == {"C", "E"}
    print("test_select_extreme_masks_bottom_top OK")


def test_equal_weight_sums_to_one():
    ret = _toy_returns()
    loser, _ = select_extreme_masks(ret, n=2)
    w = equal_weight_from_mask(loser)
    row_sum = w.sum(axis=1)
    assert np.allclose(row_sum.values, 1.0)
    assert np.allclose(w[loser].fillna(0).values[loser.values], 0.5)
    print("test_equal_weight_sums_to_one OK")


def test_overlapping_weights_hold1_entry2_o2o():
    """o2o convention: formation close t → buy open t+1 → first pnl day t+2."""
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    form = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0, 0.0], "B": 0.0}, index=idx)
    w = overlapping_weights(form, hold_days=1, entry_lag=2)
    assert abs(w.loc[idx[2], "A"] - 1.0) < 1e-9
    assert abs(w.loc[idx[1], "A"]) < 1e-9
    assert abs(w.loc[idx[3], "A"]) < 1e-9
    print("test_overlapping_weights_hold1_entry2_o2o OK")


def test_overlapping_weights_hold2():
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    form = pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    w = overlapping_weights(form, hold_days=2, entry_lag=1)
    # Day1: only A (from formation d0)
    assert abs(w.loc[idx[1], "A"] - 1.0) < 1e-9
    # Day2: A (cohort0) + B (cohort1) → equal blend then renormalize → 0.5/0.5
    assert abs(w.loc[idx[2], "A"] - 0.5) < 1e-9
    assert abs(w.loc[idx[2], "B"] - 0.5) < 1e-9
    print("test_overlapping_weights_hold2 OK")


def test_turnover_and_cost():
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    w = pd.DataFrame(
        {"A": [1.0, 0.0, 1.0], "B": [0.0, 1.0, 0.0]},
        index=idx,
    )
    to = portfolio_turnover(w)
    # Day0: 0.5 * |1| = 0.5; Day1: 0.5*(| -1|+|1|)=1.0; Day2: same 1.0
    assert abs(to.iloc[0] - 0.5) < 1e-9
    assert abs(to.iloc[1] - 1.0) < 1e-9
    gross = pd.Series([0.01, 0.02, -0.01], index=idx)
    net = apply_cost(gross, to, one_way_cost=0.001)
    assert abs(net.iloc[0] - (0.01 - 0.001 * 0.5)) < 1e-12
    print("test_turnover_and_cost OK")


def test_daily_pnl():
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=idx)
    r = pd.DataFrame({"A": [0.02, -0.01], "B": [0.00, 0.03]}, index=idx)
    pnl = daily_pnl(w, r)
    assert abs(pnl.iloc[0] - 0.01) < 1e-12
    assert abs(pnl.iloc[1] - 0.01) < 1e-12
    print("test_daily_pnl OK")


def test_extreme_signal_reversal_orientation():
    ret = _toy_returns()
    sig = extreme_signal_panel(ret)
    # Lowest return should have highest (least negative? wait: -pct so lowest ret → lowest pct → highest signal)
    d0 = ret.index[0]
    assert sig.loc[d0, "B"] == sig.loc[d0].max()  # B worst loser
    assert sig.loc[d0, "C"] == sig.loc[d0].min()  # C best winner
    print("test_extreme_signal_reversal_orientation OK")


if __name__ == "__main__":
    test_select_extreme_masks_bottom_top()
    test_equal_weight_sums_to_one()
    test_overlapping_weights_hold1_entry2_o2o()
    test_overlapping_weights_hold2()
    test_turnover_and_cost()
    test_daily_pnl()
    test_extreme_signal_reversal_orientation()
    print("all ok")
