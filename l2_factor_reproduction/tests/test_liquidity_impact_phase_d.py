"""Phase D hysteresis / asymmetric-tail tests. No DDB / ClickHouse / Raw L2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.liquidity_impact_execution import (
    PHASE_D2_TAILS,
    PHASE_D_BUFFER_WIDTHS,
    PHASE_D_FACTOR,
    PHASE_D_HOLD,
    contract_payload,
)
from l2_factor_reproduction.python.liquidity_impact_phase_d import (
    hysteresis_hl_weights,
    pick_d1,
)


def test_phase_d_contract_is_frozen() -> None:
    payload = contract_payload()
    assert PHASE_D_FACTOR == "impact_per_trade"
    assert PHASE_D_HOLD == 5
    assert PHASE_D_BUFFER_WIDTHS == (0.00, 0.05, 0.10, 0.20)
    assert PHASE_D2_TAILS == ("G10-G1", "G9:G10-G1", "G8:G10-G1")
    assert payload["phase_d_no_4x3_grid"] is True
    assert payload["phase_d_no_hold_grid"] is True
    assert payload["phase_e_optional"] is False
    assert payload["phase_e_status"] == "CANCELLED"
    assert payload["sprint_status"] == "FROZEN_COMPLETE"
    assert payload["inventory"]["impact_per_trade"]["monotonicity_gate"] == "FAIL"
    assert payload["inventory"]["impact_per_trade"]["grade"] == "TAIL_STRATEGY_GRADE"
    assert "Phase E cvxpy on this factor" in payload["stopped_search"]
    assert payload["net_sharpe_gate_not_relaxed"] == 1.5
    assert payload["phase_d1_ic_retention_min"] == 0.90


def test_phase_d_refuses_daily_or_other_factor() -> None:
    from l2_factor_reproduction.python.liquidity_impact_phase_d import run_phase_d

    with pytest.raises(KeyError, match="5D"):
        run_phase_d(hold=1, verify_hash=False)
    with pytest.raises(KeyError, match="impact_per_trade"):
        run_phase_d(factor="signed_amount_impact", verify_hash=False)


def test_hysteresis_keeps_name_inside_buffer() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    names = [f"S{i}" for i in range(20)]
    signal = pd.DataFrame(0.0, index=idx, columns=names)
    signal.loc[idx[0]] = np.arange(20)
    day1 = np.arange(20) + 3
    day1[0] = 2
    day1[2] = 0
    day1[3] = 1
    signal.loc[idx[1]] = day1
    signal.loc[idx[2]] = np.arange(20)
    _hl0, _l0, short0 = hysteresis_hl_weights(signal, buffer=0.0)
    _hl5, _l5, short5 = hysteresis_hl_weights(signal, buffer=0.05)
    assert bool(short0.loc[idx[0], "S0"])
    assert not bool(short0.loc[idx[1], "S0"])
    assert bool(short5.loc[idx[1], "S0"])


def test_hysteresis_is_dollar_neutral() -> None:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-02", periods=6)
    signal = pd.DataFrame(rng.normal(size=(6, 30)), index=idx, columns=[f"S{i}" for i in range(30)])
    hl, long_w, short_w = hysteresis_hl_weights(signal, buffer=0.10, long_entry=0.80)
    np.testing.assert_allclose(long_w.sum(axis=1).fillna(0.0).to_numpy(), 1.0, atol=1e-10)
    np.testing.assert_allclose(short_w.sum(axis=1).fillna(0.0).to_numpy(), 1.0, atol=1e-10)
    np.testing.assert_allclose(hl.sum(axis=1).to_numpy(), 0.0, atol=1e-10)


def test_pick_d1_does_not_prefer_lowest_turnover() -> None:
    frame = pd.DataFrame(
        [
            [0.00, "b00", 1.47, 0.54, 1.97, 0.30, 1.0, False],
            [0.05, "b05", 1.56, 0.46, 1.95, 0.29, 1.0, True],
            [0.10, "b10", 1.50, 0.39, 1.82, 0.28, 1.0, True],
            [0.20, "b20", 1.30, 0.30, 1.55, 0.25, 1.0, False],
        ],
        columns=[
            "buffer_width",
            "buffer_label",
            "net_hl_sharpe",
            "avg_hl_turnover_l1",
            "gross_hl_sharpe",
            "net_hl_annu",
            "ic_retention",
            "economic_pass_no_mono",
        ],
    )
    picked = pick_d1(frame)
    assert picked["buffer_width"] == 0.05


def test_pick_d1_cleared_gate_prefers_simpler() -> None:
    frame = pd.DataFrame(
        [
            [0.00, "b00", 1.51, 0.54, 1.97, 0.30, 1.0, True],
            [0.05, "b05", 1.53, 0.46, 1.95, 0.29, 1.0, True],
        ],
        columns=[
            "buffer_width",
            "buffer_label",
            "net_hl_sharpe",
            "avg_hl_turnover_l1",
            "gross_hl_sharpe",
            "net_hl_annu",
            "ic_retention",
            "economic_pass_no_mono",
        ],
    )
    picked = pick_d1(frame)
    assert picked["buffer_width"] == 0.0
