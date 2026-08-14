"""Synthetic unit checks for W-cut / amplitude cutting (no DB)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_cutting.ideal_amplitude import compute_ideal_amplitude
from factor_cutting.ideal_reversal import compute_ideal_reversal
from factor_cutting.engine import rolling_rank_split_sum


def test_w_cut_known_split():
    idx = pd.date_range("2020-01-01", periods=25, freq="B")
    # knife: first 10 days high ATS after day 19 window ends — construct last 20:
    # days 5..24: knife ranks so top10 = days with ret=0.01, bot10 ret=-0.01
    ret = pd.DataFrame({"A": np.zeros(25)}, index=idx)
    knife = pd.DataFrame({"A": np.zeros(25)}, index=idx)
    # window ending at last day uses rows 5..24 (20 days)
    # assign knife 20..11 for first 10 of window, 10..1 for last 10
    # easier: set ret and knife on last 20 rows
    r = np.array([0.02] * 10 + [-0.01] * 10, dtype=float)
    k = np.array(list(range(20, 10, -1)) + list(range(1, 11)), dtype=float)
    # high knife = first 10 = ret 0.02; low knife = last 10 = ret -0.01
    # M = 10*0.02 - 10*(-0.01) = 0.2 + 0.1 = 0.3
    ret.iloc[5:25, 0] = r
    knife.iloc[5:25, 0] = k
    fac, hi, lo = rolling_rank_split_sum(ret, knife, window=20, high_count=10, low_count=10)
    assert abs(fac.iloc[-1, 0] - 0.3) < 1e-9, fac.iloc[-1, 0]
    assert abs(hi.iloc[-1, 0] - 0.2) < 1e-9
    assert abs(lo.iloc[-1, 0] - (-0.1)) < 1e-9
    print("test_w_cut_known_split OK", fac.iloc[-1, 0])


def test_ideal_reversal_proxy():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(
        {"S1": 100 * np.cumprod(1 + rng.normal(0, 0.01, 40))},
        index=idx,
    )
    amount = pd.DataFrame({"S1": rng.uniform(1e6, 2e6, 40)}, index=idx)
    volume = pd.DataFrame({"S1": rng.uniform(1e5, 2e5, 40)}, index=idx)
    ret = close / close.shift(1) - 1
    fac = compute_ideal_reversal(ret, amount, volume=volume, window=20)
    assert fac.notna().sum().iloc[0] >= 15
    print("test_ideal_reversal_proxy OK", float(fac.dropna().iloc[-1, 0]))


def test_ideal_amplitude():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    close = pd.DataFrame({"S1": np.linspace(10, 20, 40)}, index=idx)
    high = close * 1.02
    low = close * 0.98
    open_ = close.copy()
    fac = compute_ideal_amplitude(high, low, close, open_=open_, window=20, lambda_frac=0.25)
    assert fac.notna().sum().iloc[0] >= 10
    print("test_ideal_amplitude OK", float(fac.dropna().iloc[-1, 0]))


if __name__ == "__main__":
    test_w_cut_known_split()
    test_ideal_reversal_proxy()
    test_ideal_amplitude()
    print("all ok")
