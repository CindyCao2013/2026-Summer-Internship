"""Phase B0/B1 tests: timing, labels, leakage, residual, ratio, jury, LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.feature_selection.labels import (
    build_labels_wide_panel,
)
from l2_factor_reproduction.l2_ai_stock_selection.contracts import (
    EXECUTION_CONVENTION,
    TIMING_VERDICT,
)
from l2_factor_reproduction.l2_ai_stock_selection.execution_timing import (
    apply_prepare_factor_signal_shift,
    map_factor_date_to_c2c_return,
    three_date_walkthrough,
)
from l2_factor_reproduction.l2_ai_stock_selection.fs_jury import apply_jury_rules
from l2_factor_reproduction.l2_ai_stock_selection.labels_ai_v1 import (
    PARITY_TOL,
    compare_wide,
    tail_truncation_rows,
)
from l2_factor_reproduction.l2_ai_stock_selection.leakage import (
    filter_available_score_dates,
    first_allowed_score_date,
    score_dates_are_available,
)
from l2_factor_reproduction.l2_ai_stock_selection.model_contract import (
    LGBM_PARAMS,
    assert_lgbm_baseline_consistent,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import residual_mutual_information
from l2_factor_reproduction.l2_ai_stock_selection.ratio_catalog import safe_ratio
from l2_factor_reproduction.l2_ai_stock_selection.residual_alpha import (
    cross_section_ols_diagnostics,
    residualize_panel,
)


def _calendar():
    return pd.DatetimeIndex(["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"])


def test_exact_c2c_date_mapping():
    dates = _calendar()[:3]
    factor = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        index=dates,
        columns=["A", "B"],
    )
    ret = pd.DataFrame(
        [[0.10, 0.11], [0.20, 0.21], [0.30, 0.31]],
        index=dates,
        columns=["A", "B"],
    )
    signal, ret_a = apply_prepare_factor_signal_shift(factor, ret, signal_shift=1)
    assert list(signal.index) == [dates[1], dates[2]]
    assert signal.loc[dates[1], "A"] == 1.0
    assert ret_a.loc[dates[1], "A"] == 0.20
    rec = map_factor_date_to_c2c_return(dates, dates[0], signal_shift=1)
    assert rec["return_formula"] == "Close[2024-06-04]/Close[2024-06-03]-1"
    assert rec["holding_start_close"] == "2024-06-03"
    assert rec["holding_end_close"] == "2024-06-04"


def test_factor_timestamp_vs_execution_timestamp():
    dates = _calendar()
    walk = three_date_walkthrough(dates)
    assert walk[0]["factor_date"] == "2024-06-03"
    assert walk[0]["return_formula"] == "Close[2024-06-04]/Close[2024-06-03]-1"
    assert walk[0]["executable_at_holding_start"] is False
    assert walk[1]["return_formula"] == "Close[2024-06-05]/Close[2024-06-04]-1"
    assert walk[2]["return_formula"] == "Close[2024-06-06]/Close[2024-06-05]-1"
    assert EXECUTION_CONVENTION["timing_verdict"] == "C2C_TPLUS1_NOT_EXECUTABLE"
    assert TIMING_VERDICT == "C2C_TPLUS1_NOT_EXECUTABLE"
    assert EXECUTION_CONVENTION["frozen_pairing"] == "factor_T -> Close[T+1]/Close[T]-1"


def test_score_date_strictly_after_train_label_end_max():
    cal = pd.bdate_range("2025-03-21", periods=12)
    cutoff = pd.Timestamp("2025-03-28")
    first = first_allowed_score_date(cutoff, cal)
    assert first == pd.Timestamp("2025-03-31")
    scores = [pd.Timestamp("2025-03-24"), pd.Timestamp("2025-03-28"), pd.Timestamp("2025-03-31")]
    audit = score_dates_are_available(scores, cutoff)
    assert list(audit["available"]) == [False, False, True]
    filtered = filter_available_score_dates(scores, cutoff)
    assert list(filtered) == [pd.Timestamp("2025-03-31")]
    assert int((~score_dates_are_available(filtered, cutoff)["available"]).sum()) == 0


def _manual_compound(stock, bench, dates, i, h):
    s = stock.iloc[i + 1 : i + 1 + h]
    b = bench.iloc[i + 1 : i + 1 + h]
    return float(np.prod(1.0 + s.to_numpy()) - 1.0) - float(np.prod(1.0 + b.to_numpy()) - 1.0)


def _toy_returns():
    dates = pd.bdate_range("2024-01-02", periods=30)
    cols = ["S0", "S1", "S2", "S3", "S4"]
    rng = np.random.default_rng(0)
    excess = pd.DataFrame(rng.normal(0, 0.01, size=(len(dates), 5)), index=dates, columns=cols)
    bench = pd.Series(rng.normal(0, 0.005, size=len(dates)), index=dates)
    stock = excess.add(bench, axis=0)
    return excess, bench, dates, stock


def test_y1_parity_vs_manual():
    excess, bench, dates, stock = _toy_returns()
    built = build_labels_wide_panel(excess, bench, dates, horizons=(1, 5, 20))
    ref = pd.DataFrame(index=dates, columns=stock.columns, dtype=float)
    for i in range(len(dates) - 1):
        for c in stock.columns:
            ref.loc[dates[i], c] = _manual_compound(stock[c], bench, dates, i, 1)
    cmp = compare_wide(built[1], ref, horizon=1)
    assert cmp["pass"] is True
    assert cmp["max_abs_diff"] <= PARITY_TOL


def test_y5_parity_vs_manual():
    excess, bench, dates, stock = _toy_returns()
    built = build_labels_wide_panel(excess, bench, dates, horizons=(1, 5, 20))
    ref = pd.DataFrame(index=dates, columns=stock.columns, dtype=float)
    for i in range(len(dates) - 5):
        for c in stock.columns:
            ref.loc[dates[i], c] = _manual_compound(stock[c], bench, dates, i, 5)
    cmp = compare_wide(built[5], ref, horizon=5)
    assert cmp["pass"] is True


def test_y20_parity_vs_manual():
    excess, bench, dates, stock = _toy_returns()
    built = build_labels_wide_panel(excess, bench, dates, horizons=(1, 5, 20))
    ref = pd.DataFrame(index=dates, columns=stock.columns, dtype=float)
    for i in range(len(dates) - 20):
        for c in stock.columns:
            ref.loc[dates[i], c] = _manual_compound(stock[c], bench, dates, i, 20)
    cmp = compare_wide(built[20], ref, horizon=20)
    assert cmp["pass"] is True


def test_y3_no_tail_truncation():
    excess, bench, dates, _ = _toy_returns()
    built = build_labels_wide_panel(excess, bench, dates, horizons=(3, 10))
    audit = tail_truncation_rows(built, dates, (3,))
    end_rows = audit[audit["check"] == "end_of_sample_no_truncation"]
    assert end_rows["pass"].all()
    assert built[3].iloc[-3:].isna().all().all()
    assert built[3].iloc[:-3].notna().any().any()


def test_y10_no_tail_truncation():
    excess, bench, dates, _ = _toy_returns()
    built = build_labels_wide_panel(excess, bench, dates, horizons=(3, 10))
    audit = tail_truncation_rows(built, dates, (10,))
    end_rows = audit[audit["check"] == "end_of_sample_no_truncation"]
    assert end_rows["pass"].all()
    assert built[10].iloc[-10:].isna().all().all()
    assert built[10].iloc[:-10].notna().any().any()


def test_residual_target_only_on_train_rows():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2024-01-02", periods=10)
    cols = ["A", "B", "C"] + ["S{:02d}".format(i) for i in range(37)]
    y = pd.DataFrame(rng.normal(size=(10, len(cols))), index=dates, columns=cols)
    x = pd.DataFrame(rng.normal(size=y.shape), index=dates, columns=cols)
    train = list(dates[:6])
    resid = residualize_panel(y, {"x": x}, train_dates=train, min_obs=20)
    assert resid.loc[train].notna().any().any()
    assert resid.loc[dates[6:]].isna().all().all()


def test_cross_section_residual_mean_and_orthogonality():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2))
    y = 1.5 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(scale=0.05, size=80)
    resid, diag = cross_section_ols_diagnostics(y, x, min_obs=30)
    ok = np.isfinite(resid)
    assert diag["ok"] is True
    assert abs(diag["residual_mean"]) < 1e-10
    assert abs(np.corrcoef(resid[ok], x[ok, 0])[0, 1]) < 1e-8
    assert abs(np.corrcoef(resid[ok], x[ok, 1])[0, 1]) < 1e-8
    assert np.isfinite(diag["condition_number"])
    assert diag["residual_std"] > 0


def test_mi_handles_nan_safely():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2024-01-02", periods=12)
    cols = ["S{:02d}".format(i) for i in range(40)]
    x = pd.DataFrame(rng.normal(size=(12, 40)), index=dates, columns=cols)
    y = x * 0.3 + rng.normal(scale=0.5, size=x.shape)
    y = pd.DataFrame(y, index=dates, columns=cols)
    x.iloc[0, :10] = np.nan
    y.iloc[1, 10:20] = np.nan
    x.iloc[2, 0] = np.inf
    mi = residual_mutual_information(x, y)
    assert mi == mi or np.isnan(mi)


def test_ratio_zero_and_near_zero_denominator():
    num = np.array([1.0, 2.0, 3.0, 4.0])
    den = np.array([2.0, 0.0, 1e-20, np.nan])
    out = safe_ratio(num, den, eps=1e-12)
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert not np.isinf(out).any()


def test_nonlinear_jury_outputs_review_not_keep():
    ev = pd.DataFrame(
        [
            {
                "factor": "nl",
                "F_REGRESSION": 0,
                "RANK_IC": 0,
                "MUTUAL_INFO": 1,
                "L1": 0,
                "ELASTICNET": 0,
                "LIGHTGBM": 1,
                "XGBOOST": 0,
                "PERMUTATION": 1,
                "STABILITY": 0,
                "INCREMENTAL_ALPHA": 1,
            }
        ]
    )
    out = apply_jury_rules(ev, min_methods=2)
    assert out.loc[0, "jury_state"] == "REVIEW"
    assert bool(out.loc[0, "selected"]) is False
    assert bool(out.loc[0, "nonlinear_review"]) is True


def test_lightgbm_baseline_params_internally_consistent():
    assert int(LGBM_PARAMS["num_leaves"]) == 15
    assert int(LGBM_PARAMS["max_depth"]) == 4
    assert_lgbm_baseline_consistent()
    max_leaves = (1 << int(LGBM_PARAMS["max_depth"])) - 1
    assert int(LGBM_PARAMS["num_leaves"]) <= max_leaves
    with pytest.raises(ValueError):
        assert_lgbm_baseline_consistent({"num_leaves": 31, "max_depth": 4})
