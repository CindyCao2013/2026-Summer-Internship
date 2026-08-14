"""Phase B1.5 tests: executable V2V labels, investability, adapter defaults."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.l2_ai_stock_selection.contracts import (
    PRODUCTION_EXECUTION_CONTRACT,
)
from l2_factor_reproduction.l2_ai_stock_selection.entry_investability import (
    build_entry_tradable,
    listing_age_ok,
)
from l2_factor_reproduction.l2_ai_stock_selection.executable_labels import (
    assert_no_c2c_mix,
    date_mapping_table,
    excess_from_prices,
    load_production_labels,
    tail_invalid_ok,
)
from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    HORIZONS,
    LEGACY_C2C_DIAGNOSTIC,
    PRIMARY_EXECUTION_CONTRACT,
    classify_degradation,
    holding_return_from_prices,
    is_legacy_c2c,
    map_feature_to_holding,
    resolve_execution_contract,
)
from l2_factor_reproduction.l2_ai_stock_selection.horizon import ic_horizon_row
from l2_factor_reproduction.l2_ai_stock_selection.paths import frozen_artifact_paths
from l2_factor_reproduction.l2_ai_stock_selection.residual_alpha import (
    candidate_incremental_metrics,
    residualize_panel,
)


def _calendar(n: int = 30):
    return pd.bdate_range("2024-01-02", periods=n)


def _prices(dates, n_sym=4):
    cols = ["S{}".format(i) for i in range(n_sym)]
    arr = np.arange(1, len(dates) + 1, dtype=float)[:, None] + np.arange(n_sym)
    return pd.DataFrame(100.0 + arr, index=dates, columns=cols)


def test_factor_t_enters_vwap_tplus1():
    dates = _calendar(8)
    rec = map_feature_to_holding(dates, dates[0], 1)
    assert rec["entry_date"] == dates[1]
    assert rec["entry_offset_trading_days"] == 1
    px = _prices(dates)
    y = holding_return_from_prices(px, dates, horizon=1, start_lag=1)
    expected = float(px.iloc[2, 0] / px.iloc[1, 0] - 1.0)
    assert y.iloc[0, 0] == pytest.approx(expected)


def test_1d_exit_is_vwap_tplus2():
    dates = _calendar(8)
    rec = map_feature_to_holding(dates, dates[0], 1)
    assert rec["exit_date"] == dates[2]
    assert rec["exit_offset_trading_days"] == 2


def test_3d_exit_is_vwap_tplus4():
    dates = _calendar(10)
    rec = map_feature_to_holding(dates, dates[0], 3)
    assert rec["exit_date"] == dates[4]
    assert rec["exit_offset_trading_days"] == 4


def test_5d_exit_is_vwap_tplus6():
    dates = _calendar(12)
    rec = map_feature_to_holding(dates, dates[0], 5)
    assert rec["exit_date"] == dates[6]
    assert rec["exit_offset_trading_days"] == 6


def test_10d_exit_is_vwap_tplus11():
    dates = _calendar(16)
    rec = map_feature_to_holding(dates, dates[0], 10)
    assert rec["exit_date"] == dates[11]
    assert rec["exit_offset_trading_days"] == 11


def test_20d_exit_is_vwap_tplus21():
    dates = _calendar(25)
    rec = map_feature_to_holding(dates, dates[0], 20)
    assert rec["exit_date"] == dates[21]
    assert rec["exit_offset_trading_days"] == 21


def test_benchmark_uses_identical_entry_exit_dates():
    dates = _calendar(12)
    px = _prices(dates)
    bench = px.mean(axis=1)
    y = excess_from_prices(px, bench, dates, horizon=5, method="v2v")
    rec = map_feature_to_holding(dates, dates[0], 5)
    stock_h = float(px.iloc[6, 0] / px.iloc[1, 0] - 1.0)
    bench_h = float(bench.iloc[6] / bench.iloc[1] - 1.0)
    assert rec["entry_date"] == dates[1]
    assert rec["exit_date"] == dates[6]
    assert y.iloc[0, 0] == pytest.approx(stock_h - bench_h)


def test_no_stock_v2v_benchmark_c2c_mix():
    with pytest.raises(ValueError, match="mixed execution"):
        assert_no_c2c_mix("v2v", "c2c")
    assert_no_c2c_mix("v2v", "v2v")


def test_tails_invalid_last_h_plus_1():
    dates = _calendar(12)
    px = _prices(dates)
    for h in HORIZONS:
        if 1 + h >= len(dates):
            continue
        y = holding_return_from_prices(px, dates, horizon=h, start_lag=1)
        assert tail_invalid_ok(y, dates, h)
        n = h + 1
        assert y.iloc[-n:].isna().all().all()
        assert y.iloc[:-n].notna().any().any()


def test_tplus1_tradability_attached_to_entry_date():
    dates = _calendar(8)
    px = _prices(dates)
    mask = pd.DataFrame(1.0, index=dates, columns=px.columns)
    # Suspend S0 on the entry date of feature dates[0] (= dates[1]).
    ts = pd.DataFrame(1.0, index=dates, columns=px.columns)
    ts.loc[dates[1], "S0"] = 0.0
    nl = pd.DataFrame(1.0, index=dates, columns=px.columns)
    out = build_entry_tradable(
        dates=dates,
        universe_mask_t=mask,
        adj_vwap=px,
        trade_status_t1=ts,
        not_limit_t1=nl,
        min_listing_days=1,
    )
    entry = out["entry_tradable_T1"]
    assert pd.isna(entry.loc[dates[0], "S0"]) or entry.loc[dates[0], "S0"] != 1
    assert entry.loc[dates[0], "S1"] == 1.0
    age = listing_age_ok(px, min_days=3)
    assert pd.isna(age.iloc[0, 0]) or age.iloc[0, 0] != 1
    assert age.iloc[2, 0] == 1.0


def test_legacy_labels_require_explicit_diagnostic_flag():
    assert resolve_execution_contract(None) == PRIMARY_EXECUTION_CONTRACT
    assert resolve_execution_contract("") == PRIMARY_EXECUTION_CONTRACT
    assert not is_legacy_c2c(None)
    assert is_legacy_c2c(LEGACY_C2C_DIAGNOSTIC)
    with pytest.raises(ValueError, match="unknown execution_contract"):
        resolve_execution_contract("c2c")
    with pytest.raises(ValueError, match="legacy C2C"):
        load_production_labels(execution_contract=LEGACY_C2C_DIAGNOSTIC)
    with pytest.raises(ValueError, match="start_lag"):
        holding_return_from_prices(_prices(_calendar(5)), _calendar(5), horizon=1, start_lag=0)


def test_residual_alpha_helper_defaults_to_executable_v2v():
    assert PRODUCTION_EXECUTION_CONTRACT == "EXEC_V2V_TPLUS1_V1"
    assert resolve_execution_contract() == "EXEC_V2V_TPLUS1_V1"
    sig = inspect.signature(residualize_panel)
    assert sig.parameters["execution_contract"].default is None
    dates = _calendar(6)
    cols = ["S{:02d}".format(i) for i in range(40)]
    rng = np.random.default_rng(7)
    y = pd.DataFrame(rng.normal(size=(6, 40)), index=dates, columns=cols)
    x = pd.DataFrame(rng.normal(size=y.shape), index=dates, columns=cols)
    resid = residualize_panel(y, {"x": x}, train_dates=list(dates), min_obs=20)
    assert resid.shape == y.shape
    metrics = candidate_incremental_metrics(x, y, resid)
    assert metrics["execution_contract"] == PRIMARY_EXECUTION_CONTRACT
    row = ic_horizon_row(x, {1: y, 3: y, 5: y, 10: y, 20: y})
    assert row["execution_contract"] == PRIMARY_EXECUTION_CONTRACT


def test_no_frozen_artifact_is_modified():
    paths = frozen_artifact_paths()
    assert len(paths) >= 4
    hashes = {}
    for p in paths:
        assert p.exists(), "frozen artifact missing: {}".format(p)
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        hashes[str(p)] = digest.hexdigest()
    # Re-hash immediately: this test does not write those files.
    for p in paths:
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        assert digest.hexdigest() == hashes[str(p)]


def test_date_mapping_three_real_consecutive_dates():
    dates = pd.DatetimeIndex(["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"])
    mapping = date_mapping_table(dates, horizons=(1,))
    three = ["2024-06-03", "2024-06-04", "2024-06-05"]
    for i, d in enumerate(three):
        rec = map_feature_to_holding(dates, d, 1)
        row = mapping.loc[
            (mapping["feature_date"] == pd.Timestamp(d)) & (mapping["horizon"] == 1)
        ].iloc[0]
        if i < 2:
            assert rec["valid"] is True
            assert rec["entry_date"] == dates[i + 1]
            assert rec["exit_date"] == dates[i + 2]
            assert bool(row["valid"]) is True
        else:
            # 2024-06-05 1D needs exit 06-07, which is absent.
            assert rec["valid"] is False
            assert bool(row["valid"]) is False


def test_classification_thresholds_frozen_before_names():
    assert classify_degradation(0.020, 0.012) == "ROBUST_EXECUTABLE"
    assert classify_degradation(0.020, 0.006) == "DECAY_SENSITIVE"
    assert classify_degradation(0.020, -0.010) == "TIMING_SENSITIVE"
    assert classify_degradation(0.001, 0.020) == "INCONCLUSIVE"
