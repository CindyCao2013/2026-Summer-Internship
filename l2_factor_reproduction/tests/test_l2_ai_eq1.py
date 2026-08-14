"""EQ-1 gate tests. Thresholds are frozen; do not retune to names."""

from __future__ import annotations

import numpy as np
import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    PRIMARY_EXECUTION_CONTRACT,
)
from l2_factor_reproduction.l2_ai_stock_selection.qualification import (
    AUX_ABS_IC,
    AUX_MIN_EVIDENCE,
    CORE_ABS_HL_SHARPE,
    CORE_ABS_IC,
    CORE_ABS_MONO,
    POSITIVE_CONTROL_PARENTS,
    aux_evidence_count,
    build_tc2_parent_pool,
    classify_factor,
    core_metric_ok,
    core_stability_ok,
    daily_hl_and_deciles,
    decay_ok,
    gates_dict,
    monotonicity_from_deciles,
    nonlinear_ok,
    one_period_dominated,
)


def _row(**kwargs):
    base = {
        "horizon": 5,
        "rank_ic_mean": 0.0,
        "hl_sharpe": 0.0,
        "monotonicity": 0.0,
        "coverage": 0.90,
        "sign_consistency": 0.60,
        "n_ic_days": 200,
        "mutual_information": 0.0,
        "one_period_dominated": False,
        "legacy": {"rank_ic_mean": 0.0, "hl_sharpe": 0.0},
    }
    base.update(kwargs)
    return base


def test_gates_frozen_and_v2v_primary():
    g = gates_dict()
    assert g["frozen_before_names"] is True
    assert g["core"]["abs_rank_ic"] == 0.02
    assert g["core"]["abs_hl_sharpe"] == 3.0
    assert g["core"]["abs_monotonicity"] == 0.70
    assert g["auxiliary"]["min_evidence_pieces"] == 2
    assert g["auxiliary"]["no_one_metric_auto_pass"] is True
    assert PRIMARY_EXECUTION_CONTRACT == "EXEC_V2V_TPLUS1_V1"
    assert CORE_ABS_IC == 0.02
    assert CORE_ABS_HL_SHARPE == 3.0
    assert CORE_ABS_MONO == 0.70
    assert AUX_ABS_IC == 0.01
    assert AUX_MIN_EVIDENCE == 2


def test_core_requires_all_three_and_stability():
    strong = _row(rank_ic_mean=-0.03, hl_sharpe=-3.5, monotonicity=-0.80)
    assert core_metric_ok(strong)
    ok, reasons = core_stability_ok(strong)
    assert ok and not reasons
    weak_mono = _row(rank_ic_mean=0.03, hl_sharpe=3.5, monotonicity=0.50)
    assert not core_metric_ok(weak_mono)
    dominated = _row(
        rank_ic_mean=0.03, hl_sharpe=3.5, monotonicity=0.80, one_period_dominated=True
    )
    ok, reasons = core_stability_ok(dominated)
    assert not ok
    assert "ONE_PERIOD_DOMINATED" in reasons


def test_aux_needs_two_pieces():
    one = _row(rank_ic_mean=0.012, hl_sharpe=0.4, monotonicity=0.20)
    n, bits = aux_evidence_count(one)
    assert n == 1
    two = _row(rank_ic_mean=0.012, hl_sharpe=2.2, monotonicity=0.20)
    n, bits = aux_evidence_count(two)
    assert n == 2
    assert "IC" in bits and "SHARPE" in bits


def test_nonlinear_not_from_tree_gain():
    weak_lin = _row(rank_ic_mean=0.004, hl_sharpe=0.3, monotonicity=0.15, mutual_information=0.02)
    assert nonlinear_ok(weak_lin)
    strong_lin = _row(rank_ic_mean=0.025, mutual_information=0.05)
    assert not nonlinear_ok(strong_lin)
    no_mi = _row(rank_ic_mean=0.004, mutual_information=0.001)
    assert not nonlinear_ok(no_mi)


def test_decay_not_drop_and_no_tiny_ratio():
    legacy = {"rank_ic_mean": 0.025, "hl_sharpe": 2.5}
    exec_dead = {"rank_ic_mean": 0.004, "hl_sharpe": 0.3}
    assert decay_ok(legacy, exec_dead)
    tiny = {"rank_ic_mean": 0.001, "hl_sharpe": 0.1}
    assert not decay_ok(tiny, exec_dead)


def test_classify_priority_and_secondary_flags():
    core_row = _row(horizon=5, rank_ic_mean=-0.04, hl_sharpe=-3.2, monotonicity=-0.75)
    out = classify_factor("f_core", "trade_flow", [core_row])
    assert out["classification_primary"] == "CORE_ALPHA"
    assert out["best_horizon"] == 5
    assert "CORE_AT_5D" in out["secondary_flags"]

    aux_row = _row(rank_ic_mean=0.012, hl_sharpe=2.1, monotonicity=0.25)
    out = classify_factor("f_aux", "order_book", [aux_row])
    assert out["classification_primary"] == "AUXILIARY_ALPHA"

    nl_row = _row(rank_ic_mean=0.003, hl_sharpe=0.2, monotonicity=0.1, mutual_information=0.03)
    out = classify_factor("f_nl", "price_formation", [nl_row])
    assert out["classification_primary"] == "NONLINEAR_REVIEW"

    decay_row = _row(
        rank_ic_mean=0.003,
        hl_sharpe=0.2,
        monotonicity=0.1,
        mutual_information=0.0,
        legacy={"rank_ic_mean": 0.03, "hl_sharpe": 2.4},
    )
    out = classify_factor("f_decay", "order_book", [decay_row])
    assert out["classification_primary"] == "DECAY_TIMING_SENSITIVE"

    drop_row = _row(rank_ic_mean=0.002, hl_sharpe=0.1, monotonicity=0.05, mutual_information=0.0)
    out = classify_factor("f_drop", "ddb_reference_snapshot", [drop_row])
    assert out["classification_primary"] == "DROP"


def test_aux_plus_decay_flag_is_valid():
    row = _row(
        rank_ic_mean=0.012,
        hl_sharpe=2.2,
        monotonicity=0.22,
        legacy={"rank_ic_mean": 0.03, "hl_sharpe": 3.0},
    )
    out = classify_factor("f_aux_decay", "order_book", [row])
    assert out["classification_primary"] == "AUXILIARY_ALPHA"
    assert "DECAY_SENSITIVE" in out["secondary_flags"]


def test_one_period_dominated_rule():
    ics = {
        "h2023h1": 0.04,
        "h2023h2": 0.001,
        "h2024h1": 0.002,
        "h2024h2": -0.001,
        "y2023": 0.02,
        "y2024": 0.001,
    }
    assert one_period_dominated(ics)
    even = {
        "h2023h1": 0.015,
        "h2023h2": 0.012,
        "h2024h1": 0.011,
        "h2024h2": 0.014,
        "y2023": 0.013,
        "y2024": 0.012,
    }
    assert not one_period_dominated(even)


def test_tc2_pool_excludes_core_without_decay_or_control():
    table = pd.DataFrame(
        [
            {
                "factor_name": "a",
                "family": "x",
                "classification_primary": "CORE_ALPHA",
                "secondary_flags": "",
                "best_horizon": 5,
                "best_abs_rank_ic": 0.04,
            },
            {
                "factor_name": "b",
                "family": "x",
                "classification_primary": "NONLINEAR_REVIEW",
                "secondary_flags": "NONLINEAR_CANDIDATE",
                "best_horizon": 1,
                "best_abs_rank_ic": 0.004,
            },
            {
                "factor_name": "c",
                "family": "x",
                "classification_primary": "DROP",
                "secondary_flags": "DECAY_SENSITIVE",
                "best_horizon": 1,
                "best_abs_rank_ic": 0.003,
            },
            {
                "factor_name": "obi_l5_mean",
                "family": "order_book",
                "classification_primary": "CORE_ALPHA",
                "secondary_flags": "",
                "best_horizon": 5,
                "best_abs_rank_ic": 0.03,
            },
        ]
    )
    pool = build_tc2_parent_pool(table)
    names = set(pool["factor_name"])
    assert "a" not in names
    assert "b" in names
    assert "c" in names
    assert "obi_l5_mean" in names
    assert POSITIVE_CONTROL_PARENTS[0] == "obi_l5_mean"


def test_daily_hl_and_monotonicity_linear():
    dates = pd.bdate_range("2024-01-02", periods=40)
    cols = ["S{:02d}".format(i) for i in range(40)]
    rng = np.random.default_rng(1)
    f = pd.DataFrame(rng.normal(size=(40, 40)), index=dates, columns=cols)
    y = f * 0.2 + rng.normal(scale=0.05, size=f.shape)
    y = pd.DataFrame(y, index=dates, columns=cols)
    hl, dec = daily_hl_and_deciles(f, y)
    assert hl.notna().sum() > 20
    mono = monotonicity_from_deciles(dec)
    assert mono > 0.5
