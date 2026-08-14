"""Phase B staggered-holding tests. No DDB / ClickHouse / Raw L2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.liquidity_impact_execution import (
    PHASE_B_CANDIDATES,
    PHASE_B_HOLDS,
    PHASE_B_SLOW_GRID,
    contract_payload,
)
from l2_factor_reproduction.python.liquidity_impact_phase_b import (
    hl_signed_weights,
    hold_label,
    l1_turnover,
    pick_r3,
    stagger_weights,
)


def test_phase_b_grid_is_frozen() -> None:
    payload = contract_payload()
    assert PHASE_B_HOLDS == (1, 5, 10)
    assert PHASE_B_SLOW_GRID == ("daily", "staggered_5d", "staggered_10d")
    assert hold_label(1) == "daily"
    assert hold_label(5) == "staggered_5d"
    assert hold_label(10) == "staggered_10d"
    assert PHASE_B_CANDIDATES == (
        ("impact_per_trade", 1),
        ("signed_amount_impact", 3),
        ("signed_sqrt_amount_impact", 1),
        ("depth_recovery_5m", 10),
    )
    assert payload["phase_b_r3_pick_after"] is True
    assert "DO NOT average past signals then re-rank" in payload["phase_b_sleeve_definition"]
    assert payload["phase_d_no_4x3_grid"] is True
    assert payload["phase_e_optional"] is False
    assert payload["phase_e_status"] == "CANCELLED"
    assert payload["sprint_status"] == "FROZEN_COMPLETE"
    assert payload["net_sharpe_gate_not_relaxed"] == 1.5
    assert payload["phase_d_factor"] == "impact_per_trade"
    assert payload["phase_d_hold"] == 5


def test_phase_b_refuses_unfrozen_candidate() -> None:
    from l2_factor_reproduction.python.liquidity_impact_phase_b import run_phase_b

    with pytest.raises(KeyError, match="frozen"):
        run_phase_b(
            candidates=[("impact_asymmetry", 1)],
            verify_hash=False,
        )


def test_stagger_is_sleeve_average_not_signal_rerank() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    signal = pd.DataFrame(
        {
            "A": [3.0, 1.0, 1.0],
            "B": [2.0, 3.0, 2.0],
            "C": [1.0, 2.0, 3.0],
        },
        index=idx,
    )
    daily = hl_signed_weights(signal, n_groups=3)
    book = stagger_weights(daily, 3)
    averaged_signal = signal.rolling(3, min_periods=3).mean()
    reranked = hl_signed_weights(averaged_signal, n_groups=3)
    last = idx[-1]
    assert not np.allclose(book.loc[last].to_numpy(), reranked.loc[last].to_numpy())
    expected = (daily.loc[idx[0]] + daily.loc[idx[1]] + daily.loc[idx[2]]) / 3.0
    np.testing.assert_allclose(book.loc[last].to_numpy(), expected.to_numpy())


def test_netted_turnover_cancels_overlapping_sleeve_trades() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    cols = ["A", "B", "C"]
    daily = pd.DataFrame(0.0, index=idx, columns=cols)
    daily.loc[idx[0], ["A", "B"]] = [1.0, -1.0]
    daily.loc[idx[1], ["A", "C"]] = [1.0, -1.0]
    daily.loc[idx[2], ["A", "B"]] = [1.0, -1.0]
    book = stagger_weights(daily, 2)
    to = l1_turnover(book.dropna(how="all"))
    # After warmup, P1=(A,C) and P2=(A,B) net to the same book as P0+P1.
    assert float(to.iloc[-1]) < 1e-12
    sleeve_to = l1_turnover(daily)
    assert float(sleeve_to.iloc[-1]) > 1.5


def test_h1_turnover_equals_disjoint_leg_sum() -> None:
    idx = pd.bdate_range("2020-01-02", periods=4)
    rng = np.random.default_rng(0)
    signal = pd.DataFrame(
        rng.normal(size=(4, 20)),
        index=idx,
        columns=[f"S{i}" for i in range(20)],
    )
    from l2_factor_reproduction.python.liquidity_impact_phase_b import (
        decile_membership_weights,
    )

    long_w = decile_membership_weights(signal, 10, n_groups=10)
    short_w = decile_membership_weights(signal, 1, n_groups=10)
    signed = long_w.fillna(0.0) - short_w.fillna(0.0)
    to_signed = l1_turnover(signed)
    to_legs = l1_turnover(long_w) + l1_turnover(short_w)
    np.testing.assert_allclose(to_signed.to_numpy(), to_legs.to_numpy(), atol=1e-12)


def test_pick_r3_pareto_keeps_one() -> None:
    frame = pd.DataFrame(
        [
            ["impact_per_trade", 5, 2.20, 0.28, 0.72, -0.18, 1.0],
            ["signed_amount_impact", 5, 1.42, 0.17, 0.81, -0.25, 1.0],
            ["signed_sqrt_amount_impact", 5, 1.55, 0.19, 0.93, -0.22, 1.0],
        ],
        columns=[
            "factor",
            "hold_days",
            "net_hl_sharpe",
            "net_hl_annu",
            "avg_hl_turnover_l1",
            "net_hl_mdd",
            "yearly_sign_consistency",
        ],
    )
    picked = pick_r3(frame)
    assert picked["keep"] == ["impact_per_trade"]
    assert set(picked["drop"]) == {
        "signed_amount_impact",
        "signed_sqrt_amount_impact",
    }
    assert picked["reason"] == "pareto_dominance"


def test_pick_r3_caps_at_two() -> None:
    frame = pd.DataFrame(
        [
            ["impact_per_trade", 5, 1.60, 0.22, 0.80, -0.20, 1.0],
            ["signed_amount_impact", 5, 1.55, 0.21, 0.70, -0.19, 1.0],
            ["signed_sqrt_amount_impact", 5, 1.50, 0.20, 0.75, -0.21, 1.0],
        ],
        columns=[
            "factor",
            "hold_days",
            "net_hl_sharpe",
            "net_hl_annu",
            "avg_hl_turnover_l1",
            "net_hl_mdd",
            "yearly_sign_consistency",
        ],
    )
    picked = pick_r3(frame)
    assert len(picked["keep"]) <= 2
    assert "impact_per_trade" in picked["keep"]


def test_research_keep_collapses_near_alias() -> None:
    from l2_factor_reproduction.python.liquidity_impact_phase_b import (
        research_r3_keep,
    )

    selection = {
        "keep": ["impact_per_trade", "signed_sqrt_amount_impact"],
        "drop": ["signed_amount_impact"],
        "reason": "pareto_survivors",
        "best_rows": [
            {
                "factor": "impact_per_trade",
                "net_hl_sharpe": 1.47,
                "avg_hl_turnover_l1": 0.54,
            },
            {
                "factor": "signed_sqrt_amount_impact",
                "net_hl_sharpe": 0.98,
                "avg_hl_turnover_l1": 0.52,
            },
        ],
    }
    out = research_r3_keep(selection)
    assert out["research_keep"] == ["impact_per_trade"]
    assert "alias_collapse" in str(out["research_reason"])
