"""Unit tests for L2 AI Stock Selection v1 skeleton (synthetic data only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.l2_ai_stock_selection.contracts import (
    COST_SCENARIOS_BPS,
    EXECUTION_CONVENTION,
    FORBIDDEN_ML_COLUMNS,
    allowed_rebalance_pairs,
    assert_no_forbidden_feature_columns,
    classify_time_scale,
    cost_from_l1,
    data_contract_dict,
)
from l2_factor_reproduction.l2_ai_stock_selection.horizon import (
    approx_half_life,
    consensus_selection,
    ic_horizon_row,
)
from l2_factor_reproduction.l2_ai_stock_selection.inventory import (
    family_summary,
    load_factor_inventory,
)
from l2_factor_reproduction.l2_ai_stock_selection.leakage import (
    assert_train_does_not_use_oos,
    score_dates_are_available,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import (
    binned_conditional_return,
    nonlinear_should_review,
    rank_ic,
)
from l2_factor_reproduction.l2_ai_stock_selection.ratio_catalog import (
    build_ratio_candidate_registry,
)
from l2_factor_reproduction.l2_ai_stock_selection.residual_alpha import (
    candidate_incremental_metrics,
    cross_section_ols_residual,
    residualize_panel,
)


def _toy_panel(n_dates=12, n_sym=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    cols = ["S{:02d}.SZ".format(i) for i in range(n_sym)]
    noise = rng.normal(0, 1.0, size=(n_dates, n_sym))
    factor = pd.DataFrame(noise, index=dates, columns=cols)
    # linear + a bit of noise
    y = factor * 0.4 + rng.normal(0, 1.0, size=factor.shape)
    y = pd.DataFrame(y, index=dates, columns=cols)
    size = pd.DataFrame(rng.normal(0, 1.0, size=factor.shape), index=dates, columns=cols)
    return factor, y, size


def test_execution_is_c2c_shift1():
    assert EXECUTION_CONVENTION["return_method"] == "c2c"
    assert int(EXECUTION_CONVENTION["signal_shift"]) == 1
    assert "open-to-open" in str(EXECUTION_CONVENTION["not_used"])


def test_forbidden_feature_columns():
    with pytest.raises(ValueError):
        assert_no_forbidden_feature_columns(("net_buy_ratio", "y_5d" if False else "label"))
    assert_no_forbidden_feature_columns(("net_buy_ratio", "obi_l5_mean"))
    assert "forward_return" in FORBIDDEN_ML_COLUMNS


def test_rebalance_grid_not_brute_force():
    pairs = allowed_rebalance_pairs()
    assert (1, 20) not in pairs
    assert (1, 1) in pairs
    assert (5, 1) in pairs
    assert (5, 5) in pairs
    assert (20, 5) in pairs
    assert (20, 1) not in pairs


def test_cost_scenarios_inherit_base_7p5():
    assert COST_SCENARIOS_BPS["BASE"] == 7.5
    base = cost_from_l1(2.0, "BASE")
    high = cost_from_l1(2.0, "HIGH")
    assert high > base
    # L1=2, 7.5bps, 250d → 2 * 7.5/1e4 * 250 = 0.375
    assert abs(base - 0.375) < 1e-12


def test_time_scale_mapping():
    assert classify_time_scale(1) == "fast"
    assert classify_time_scale(5) == "mid"
    assert classify_time_scale(20) == "slow"


def test_ols_residual_orthogonal_in_sample():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = 2.0 * x + rng.normal(scale=0.1, size=80)
    resid = cross_section_ols_residual(y, x.reshape(-1, 1))
    # residual should be nearly uncorrelated with x
    ok = np.isfinite(resid)
    corr = np.corrcoef(resid[ok], x[ok])[0, 1]
    assert abs(corr) < 1e-8


def test_residualize_panel_ignores_oos_dates():
    factor, y, size = _toy_panel()
    train = list(factor.index[:8])
    resid = residualize_panel(y, {"size": size}, train_dates=train)
    assert resid.loc[train].notna().any().any()
    oos = factor.index[8:]
    assert resid.loc[oos].isna().all().all()


def test_incremental_metrics_train_only():
    factor, y, size = _toy_panel()
    train = list(factor.index[:8])
    resid = residualize_panel(y, {"size": size}, train_dates=train)
    metrics = candidate_incremental_metrics(factor, y, resid, train_dates=train)
    assert metrics["n_train_dates"] == 8
    assert np.isfinite(metrics["raw_rank_ic"])
    assert np.isfinite(metrics["residual_rank_ic"])


def test_nonlinear_review_rule():
    assert nonlinear_should_review(0.001, 0.05) is True
    assert nonlinear_should_review(0.05, 0.05) is False
    assert nonlinear_should_review(0.001, 0.001) is False


def test_binned_curve_monotonic_linear():
    factor, y, _ = _toy_panel(n_dates=20, n_sym=50)
    bins = binned_conditional_return(factor, y, n_bins=5)
    assert len(bins) == 5
    # noisy but directionally increasing
    assert bins["mean_y"].iloc[-1] > bins["mean_y"].iloc[0]


def test_consensus_not_sharpe_weighted():
    ev = pd.DataFrame(
        {
            "factor": ["a", "b", "c"],
            "F_REGRESSION": [1, 1, 0],
            "L1": [1, 0, 0],
            "LIGHTGBM": [0, 1, 1],
            "PERMUTATION": [0, 0, 0],
            "STABILITY": [1, 0, 0],
            "INCREMENTAL_ALPHA": [0, 0, 0],
        }
    )
    out = consensus_selection(ev, min_methods=2)
    assert list(out["selection_count"]) == [3, 2, 1]
    assert list(out["selected"]) == [True, True, False]
    assert "sharpe" not in out.columns.str.lower()


def test_half_life_and_peak():
    ics = {1: 0.08, 3: 0.05, 5: 0.03, 10: 0.02, 20: 0.01}
    # 0.5 * 0.08 = 0.04; first h>=1 with |ic|<=0.04 is 5 → half_life=4
    assert approx_half_life(ics, 1) == 4.0
    factor, y, _ = _toy_panel()
    labels = {1: y, 3: y * 0.5, 5: y * 0.2, 10: y * 0.05, 20: y * 0.01}
    row = ic_horizon_row(factor, labels)
    assert row["peak_horizon"] == 1


def test_leakage_train_label_end():
    dates = pd.bdate_range("2024-01-02", periods=10)
    label_end = pd.Series(dates + pd.tseries.offsets.BDay(5), index=dates)
    with pytest.raises(ValueError):
        assert_train_does_not_use_oos(dates[:6], dates[5], label_end)
    assert_train_does_not_use_oos(dates[:3], dates[8], label_end)


def test_retroactive_score_flag():
    dates = pd.bdate_range("2025-03-01", periods=10)
    audit = score_dates_are_available(dates, pd.Timestamp("2025-03-10"))
    assert (audit["reason"] == "RETROACTIVE_SCORE").any()
    assert (audit["reason"] == "OK").any()


def test_inventory_reads_existing_registry():
    inv = load_factor_inventory()
    assert len(inv) >= 100
    assert "net_buy_ratio" in set(inv["factor_name"])
    fam = family_summary(inv)
    assert "trade_flow" in set(fam["factor_family"])
    assert fam["n_formulas"].sum() == len(inv)


def test_ratio_catalog_does_not_explode():
    inv = load_factor_inventory()
    ratios = build_ratio_candidate_registry(inv)
    n_proposed = int((ratios["status"] == "PROPOSED").sum())
    assert n_proposed <= 8
    assert (ratios["status"] == "ALREADY_IN_POOL").sum() >= 20
    assert "ofi_over_depth" in set(ratios["candidate_name"])


def test_data_contract_jsonable():
    payload = data_contract_dict()
    json.dumps(payload)
    assert payload["execution"]["return_method"] == "c2c"
    assert payload["horizons"] == [1, 3, 5, 10, 20]
    models = payload["model_layer"]["core_benchmarks"]
    names = [m["name"] for m in models]
    assert names[:4] == [
        "equal_weight_selected_factor",
        "ridge_elasticnet",
        "lightgbm",
        "xgboost",
    ]
    assert models[2]["priority"] == "P0"
    assert models[3]["priority"] == "P0"
    rf = [m for m in models if m["name"] == "random_forest"][0]
    assert rf["priority"] == "optional_diagnostic"


def test_nonlinear_keep_override_not_dropped():
    from l2_factor_reproduction.l2_ai_stock_selection.fs_jury import apply_jury_rules

    ev = pd.DataFrame(
        [
            {
                "factor": "nonlinear_alpha",
                "F_REGRESSION": 0,
                "RANK_IC": 0,
                "MUTUAL_INFO": 1,
                "L1": 0,
                "ELASTICNET": 0,
                "LIGHTGBM": 1,
                "XGBOOST": 1,
                "PERMUTATION": 1,
                "STABILITY": 0,
                "INCREMENTAL_ALPHA": 1,
            }
        ]
    )
    out = apply_jury_rules(ev, min_methods=2)
    assert bool(out.loc[0, "nonlinear_review"]) is True
    assert bool(out.loc[0, "nonlinear_keep_override"]) is False
    assert out.loc[0, "jury_state"] == "REVIEW"
    assert bool(out.loc[0, "selected"]) is False


def test_tree_gain_only_is_not_kept():
    from l2_factor_reproduction.l2_ai_stock_selection.fs_jury import apply_jury_rules

    ev = pd.DataFrame(
        [
            {
                "factor": "redundant_substitute",
                "F_REGRESSION": 0,
                "RANK_IC": 0,
                "MUTUAL_INFO": 0,
                "L1": 0,
                "ELASTICNET": 0,
                "LIGHTGBM": 1,
                "XGBOOST": 1,
                "PERMUTATION": 0,
                "STABILITY": 0,
                "INCREMENTAL_ALPHA": 0,
            }
        ]
    )
    out = apply_jury_rules(ev, min_methods=2)
    assert bool(out.loc[0, "tree_gain_without_confirmation"]) is True
    assert out.loc[0, "jury_state"] == "DROP"
    assert bool(out.loc[0, "selected"]) is False


def test_lightgbm_is_available_and_fits_tiny():
    from l2_factor_reproduction.l2_ai_stock_selection.learners import (
        fit_lightgbm,
        lgbm_available,
        predict_with_runtime,
    )
    from l2_factor_reproduction.l2_ai_stock_selection.model_contract import (
        COMPARISON_METRICS,
        CORE_BENCHMARKS,
    )

    assert lgbm_available() is True
    assert CORE_BENCHMARKS[2]["name"] == "lightgbm"
    assert "training_runtime_sec" in COMPARISON_METRICS
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(80, 4))
    y_tr = X_tr[:, 0] * 0.5 + rng.normal(scale=0.1, size=80)
    X_va = rng.normal(size=(20, 4))
    y_va = X_va[:, 0] * 0.5 + rng.normal(scale=0.1, size=20)
    model, meta = fit_lightgbm(X_tr, y_tr, X_va, y_va)
    pred, pred_sec = predict_with_runtime(model, X_va)
    assert pred.shape == (20,)
    assert meta["training_runtime_sec"] > 0
    assert pred_sec >= 0
    assert meta["name"] == "LightGBM"
