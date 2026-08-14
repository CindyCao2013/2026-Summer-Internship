"""Enhancement optimizer stays long-only and respects the turnover cap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphanet.config import EnhanceConfig
from alphanet.enhance import enhance_backtest, optimize_weights
from alphanet.synthetic import make_synthetic_panel


def test_optimize_weights_long_only_and_sum_to_one():
    synth = make_synthetic_panel(n_days=10, n_stocks=30, seed=6)
    dt = synth.calendar[-1]
    factor = synth.ret_1d.loc[dt]
    bench = pd.Series(1.0 / 30, index=factor.index)
    w = optimize_weights(
        factor,
        bench,
        None,
        synth.industry.loc[dt],
        synth.log_mcap.loc[dt],
        EnhanceConfig(),
        active_cap=0.02,
    )
    assert float(w.min()) >= -1e-8
    assert abs(float(w.sum()) - 1.0) < 1e-5
    assert float(w.max()) <= 0.05 + 1e-8


def test_enhance_backtest_returns_summary():
    synth = make_synthetic_panel(n_days=40, n_stocks=30, seed=7)
    factor = synth.ret_1d.shift(1)
    members = pd.DataFrame(1.0, index=factor.index, columns=factor.columns)
    result = enhance_backtest(
        factor.iloc[5:],
        synth.ret_1d,
        members,
        industry=synth.industry,
        log_mcap=synth.log_mcap,
        cfg=EnhanceConfig(active_weight_caps=(0.01,)),
        active_cap=0.01,
    )
    assert "information_ratio" in result["summary"]
    assert len(result["excess"]) > 5
