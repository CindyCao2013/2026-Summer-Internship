"""Sprint 16 execution-layer tests. No DDB / ClickHouse / Raw L2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.liquidity_impact_execution import (
    BUFFER_WIDTHS,
    CONTRACT_VERSION,
    FACTORS,
    FORBIDDEN,
    HORIZONS,
    R3_NEAR_ALIASES,
    REBALANCE_DAYS,
    SMOOTHING_WINDOWS,
    TIER1,
    TIER2,
    HorizonRow,
    classify_half_life,
    contract_payload,
    icir_newey_west,
    icir_nonoverlapping,
    interpolate_half_life,
    parity_ok,
    recommended_next_phase,
    summarize_decay,
)


def test_contract_forbids_formula_edits() -> None:
    payload = contract_payload()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["do_not_touch_primitive_formulas"] is True
    assert payload["phase_a_only_this_run"] is False
    assert payload["phase_c_no_production_to_gate"] is True
    assert payload["phase_c_smooth_layer"] == "daily_signal_exposure_not_primitive"
    assert payload["allowed"] == ["horizon", "smoothing", "rebalance", "buffer"]
    assert payload["feature_gate"]["abs_rank_ic_min"] == 0.02
    assert payload["phase_b_holds"] == [1, 5, 10]
    assert payload["sprint_status"] == "FROZEN_COMPLETE"
    assert payload["phase_e_status"] == "CANCELLED"
    assert payload["impact_per_trade_not_claimed_monotonic_strategy_grade"] is True
    assert set(payload["inventory"]) == set(FACTORS)
    assert payload["inventory"]["impact_per_trade"]["monotonicity_gate"] == "FAIL"
    assert "impact_asymmetry" in payload["phase_c_medium_parked"]
    assert "depth_recovery_5m" in payload["phase_c_slow_factors"]
    for item in (
        "5m recovery definition",
        "signed amount definition",
        "factor direction",
    ):
        assert item in FORBIDDEN
    assert tuple(payload["factors"]) == FACTORS
    assert set(TIER1).isdisjoint(TIER2)
    assert set(R3_NEAR_ALIASES) <= set(TIER1)


def test_frozen_grids_are_small() -> None:
    assert HORIZONS == (1, 2, 3, 5, 10, 20)
    assert SMOOTHING_WINDOWS == (3, 5, 10)
    assert REBALANCE_DAYS == (1, 2, 3, 5, 10)
    assert BUFFER_WIDTHS == (0.00, 0.05, 0.10, 0.20)


def test_half_life_fast_decay() -> None:
    # 3.4%, 1.4%, 0.4%, 0.1% → crosses 1.7% between h=1 and h=2
    half = interpolate_half_life((1, 2, 3, 5), (0.034, 0.014, 0.004, 0.001))
    assert 1.0 < half < 2.0
    assert classify_half_life(half) == "FAST"
    assert recommended_next_phase("FAST") == "buffer_hysteresis_not_lower_refresh"


def test_half_life_slow_decay() -> None:
    half = interpolate_half_life((1, 2, 3, 5, 10), (0.034, 0.030, 0.026, 0.021, 0.014))
    assert 5.0 < half < 10.0
    assert classify_half_life(half) == "SLOW"


def test_half_life_unresolved_if_never_halves() -> None:
    half = interpolate_half_life((1, 2, 3, 5, 10, 20), (0.03, 0.029, 0.028, 0.027, 0.026, 0.025))
    assert half == float("inf")
    assert classify_half_life(half) == "UNRESOLVED"


def test_summarize_decay_medium() -> None:
    rows = [
        HorizonRow("x", "full", 1, -0.04, -5.0, 0.35, 100, 1000.0, 1.0),
        HorizonRow("x", "full", 2, -0.032, -4.0, 0.36, 100, 1000.0, 0.8),
        HorizonRow("x", "full", 3, -0.024, -3.0, 0.38, 100, 1000.0, 0.6),
        HorizonRow("x", "full", 5, -0.016, -2.0, 0.42, 100, 1000.0, 0.4),
        HorizonRow("x", "full", 10, -0.008, -1.0, 0.45, 100, 1000.0, 0.2),
        HorizonRow("x", "full", 20, -0.002, -0.3, 0.48, 100, 1000.0, 0.05),
    ]
    out = summarize_decay(rows)
    assert out["decay_class"] == "MEDIUM"
    assert out["recommended_next_phase"] == "smoothing_and_3d_5d_rebalance"
    assert out["sign_flip_before_h20"] is False


def test_parity_tolerance() -> None:
    assert parity_ok(-0.034198, -0.034198351)
    assert not parity_ok(-0.034198, -0.040000)


def test_interpolate_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        interpolate_half_life((1,), (0.03,))
    with pytest.raises(ValueError):
        interpolate_half_life((1, 2), (0.03,))


def test_trailing_mean_has_no_lookahead() -> None:
    from l2_factor_reproduction.python.liquidity_impact_phase_c import (
        trailing_mean_wide,
    )

    idx = pd.bdate_range("2020-01-02", periods=10)
    wide = pd.DataFrame({"A": [0.0] * 9 + [1.0]}, index=idx)
    ma3 = trailing_mean_wide(wide, 3)
    assert float(ma3.iloc[8, 0]) == 0.0
    assert abs(float(ma3.iloc[9, 0]) - 1.0 / 3.0) < 1e-12
    assert pd.isna(ma3.iloc[1, 0])


def test_phase_c_refuses_medium_factors() -> None:
    from l2_factor_reproduction.python.liquidity_impact_phase_c import run_phase_c

    with pytest.raises(KeyError, match="SLOW-only"):
        run_phase_c(factors=["impact_asymmetry"], verify_hash=False)


def test_promote_phase_c_keeps_at_most_two() -> None:
    from l2_factor_reproduction.python.liquidity_impact_phase_c import (
        promote_phase_c,
    )

    frame = pd.DataFrame(
        [
            ["f", "RAW", 1.0, 1.00, 0.95, 2.40],
            ["f", "MA3", 2.1, 0.90, 0.90, 1.80],
            ["f", "MA5", 2.4, 0.84, 0.88, 1.30],
            ["f", "MA10", 2.0, 0.66, 0.80, 0.90],
        ],
        columns=[
            "factor",
            "version",
            "net_hl_sharpe",
            "ic_retention",
            "decile_mono_spearman",
            "avg_hl_turnover_l1",
        ],
    )
    promo = promote_phase_c(frame)
    kept = promo.loc[promo["promote"]]
    assert set(kept["version"]) == {"MA5", "MA10"}
    assert kept.loc[kept["version"] == "MA5", "decision"].iloc[0] == "A_net_sharpe"
    assert (
        kept.loc[kept["version"] == "MA10", "decision"].iloc[0]
        == "B_turnover_frontier"
    )


def test_newey_west_icir_shrinks_persistent_series() -> None:
    rng = np.random.default_rng(7)
    n = 400
    eps = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + eps[i]
    x = x + 0.05
    series = pd.Series(x)
    naive = series.mean() / series.std() * (250 ** 0.5)
    nw = icir_newey_west(series, lags=10)
    assert abs(nw) < abs(naive)


def test_icir_nonoverlapping_stride() -> None:
    series = pd.Series(np.linspace(0.02, 0.01, 20))
    icir, n_dates = icir_nonoverlapping(series, stride=5)
    assert n_dates == 4
    assert np.isfinite(icir)
