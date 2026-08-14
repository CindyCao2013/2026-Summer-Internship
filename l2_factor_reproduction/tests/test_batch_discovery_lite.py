"""Batch Discovery Lite unit tests (no DDB / ClickHouse / ML panel)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.discovery_lite.contracts import (
    BDL_CONTRACT,
    DATE_STRIDE,
    LITE_END,
    LITE_START,
    RANK_IC_THRESHOLD,
    ICIR_THRESHOLD,
    lite_trading_dates,
)
from l2_factor_reproduction.discovery_lite.gates import (
    assign_deciles,
    coverage_metrics,
    decile_lite_one_day,
    discovery_priority_score,
    gate0_status,
    gate1_status,
    rank_ic_lite_for_wide,
    select_cluster_representatives,
)
from l2_factor_reproduction.discovery_lite.novelty import novelty_bucket, novelty_vs_existing
from l2_factor_reproduction.discovery_lite.redundancy import (
    candidate_correlation,
    cluster_candidates,
)


def _calendar(n: int = 20) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=n)


# ---------------------------------------------------------------------------
# A. Lite calendar
# ---------------------------------------------------------------------------


def test_lite_calendar_every_fifth_date_deterministic() -> None:
    cal = _calendar(40)
    a = lite_trading_dates(cal, start=cal[0], end=cal[-1], stride=5)
    b = lite_trading_dates(cal, start=cal[0], end=cal[-1], stride=5)
    assert a.equals(b)
    assert list(a) == list(cal[::5])
    assert DATE_STRIDE == 5


def test_lite_window_matches_fast_discovery() -> None:
    assert LITE_START == pd.Timestamp("2023-01-01")
    assert LITE_END == pd.Timestamp("2024-12-31")
    assert BDL_CONTRACT["rank_ic_threshold"] == RANK_IC_THRESHOLD
    assert BDL_CONTRACT["icir_threshold"] == ICIR_THRESHOLD


def test_lite_calendar_not_performance_sorted() -> None:
    cal = pd.DatetimeIndex(
        ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]
    )
    shuffled = cal[[4, 1, 0, 3, 2]]
    sampled = lite_trading_dates(shuffled, start=cal[0], end=cal[-1], stride=2)
    assert list(sampled) == list(cal[::2])


# ---------------------------------------------------------------------------
# B. Coverage
# ---------------------------------------------------------------------------


def test_coverage_toy_missing_matrix() -> None:
    dates = _calendar(4)
    symbols = ["A", "B", "C", "D"]
    data = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [1.0, np.nan, np.nan, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    wide = pd.DataFrame(data, index=dates, columns=symbols)
    mask = pd.DataFrame(1.0, index=dates, columns=symbols)
    metrics = coverage_metrics(wide, mask, dates)
    assert metrics["n_rows"] == 9
    assert metrics["row_coverage"] == pytest.approx(9 / 16)
    assert metrics["date_coverage"] == pytest.approx(0.75)
    assert metrics["symbol_coverage"] == pytest.approx(1.0)
    assert metrics["missing_ratio"] == pytest.approx(7 / 16)
    assert metrics["nonfinite_ratio"] == 0.0


def test_gate0_does_not_treat_na_as_zero() -> None:
    dates = _calendar(3)
    wide = pd.DataFrame(
        [[np.nan, np.nan], [np.nan, 0.0], [0.0, 0.0]],
        index=dates,
        columns=["A", "B"],
    )
    mask = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    metrics = coverage_metrics(wide, mask, dates)
    assert metrics["zero_ratio"] == pytest.approx(1.0)  # among finite values
    assert metrics["missing_ratio"] > 0
    status = gate0_status(
        metrics,
        formula_valid=True,
        primitive_available=True,
        pit_status="PASS",
        sparse_event=True,
    )
    assert status == "SPARSE_EVENT_REVIEW"


# ---------------------------------------------------------------------------
# C. RankIC
# ---------------------------------------------------------------------------


def _monotonic_panel(n_dates: int = 12, n_names: int = 30, sign: int = 1):
    dates = _calendar(n_dates)
    symbols = [f"S{i:03d}" for i in range(n_names)]
    # Cross-section identity ranks: factor_i == i, return_i == sign * i
    factor = pd.DataFrame(
        np.tile(np.arange(n_names, dtype=float), (n_dates, 1)),
        index=dates,
        columns=symbols,
    )
    ret = sign * factor.copy()
    mask = pd.DataFrame(1.0, index=dates, columns=symbols)
    return factor, ret, mask, dates


def test_rank_ic_lite_perfect_monotonic() -> None:
    factor, ret, mask, dates = _monotonic_panel(sign=1)
    lite = dates[::5]
    ic, metrics = rank_ic_lite_for_wide(
        factor, mask, ret, lite, start=dates[0], end=dates[-1]
    )
    # After T+1 shift, first lite date may drop; remaining days are still perfect.
    assert metrics["n_ic_dates"] >= 1
    assert metrics["mean_rank_ic_lite"] == pytest.approx(1.0, abs=1e-9)


def test_rank_ic_lite_perfect_negative() -> None:
    factor, ret, mask, dates = _monotonic_panel(sign=-1)
    lite = dates[::5]
    _ic, metrics = rank_ic_lite_for_wide(
        factor, mask, ret, lite, start=dates[0], end=dates[-1]
    )
    assert metrics["mean_rank_ic_lite"] == pytest.approx(-1.0, abs=1e-9)
    # Direction stays raw: no auto-flip.
    assert metrics["mean_rank_ic_lite"] < 0


def test_gate1_permissive_or_rule() -> None:
    weak_ic = gate1_status(
        {"mean_rank_ic_lite": 0.009, "icir_lite": 0.2, "n_ic_dates": 30}
    )
    weak_icir = gate1_status(
        {"mean_rank_ic_lite": 0.001, "icir_lite": 1.6, "n_ic_dates": 30}
    )
    fail = gate1_status(
        {"mean_rank_ic_lite": 0.001, "icir_lite": 0.2, "n_ic_dates": 30}
    )
    few = gate1_status(
        {"mean_rank_ic_lite": 0.05, "icir_lite": 5.0, "n_ic_dates": 5}
    )
    assert weak_ic == "PASS"
    assert weak_icir == "PASS"
    assert fail == "REJECT_LOW_SIGNAL"
    assert few == "REJECT_LOW_SIGNAL"


# ---------------------------------------------------------------------------
# D. Redundancy
# ---------------------------------------------------------------------------


def test_near_identical_factors_cluster() -> None:
    rng = np.random.default_rng(0)
    dates = _calendar(8)
    symbols = [f"S{i:03d}" for i in range(40)]
    base = pd.DataFrame(rng.normal(size=(8, 40)), index=dates, columns=symbols)
    noise = pd.DataFrame(rng.normal(scale=1e-6, size=(8, 40)), index=dates, columns=symbols)
    other = pd.DataFrame(rng.normal(size=(8, 40)), index=dates, columns=symbols)
    panel = pd.concat(
        [
            base.stack().rename("f_a"),
            (base + noise).stack().rename("f_b"),
            other.stack().rename("f_c"),
        ],
        axis=1,
    ).reset_index()
    panel.columns = ["TradeDate", "Symbol", "f_a", "f_b", "f_c"]
    corr = candidate_correlation(panel, ["f_a", "f_b", "f_c"], min_names=10)
    clusters = cluster_candidates(corr, threshold=0.80)
    by_name = clusters.set_index("factor")
    assert (
        by_name.loc["f_a", "redundancy_cluster_080"]
        == by_name.loc["f_b", "redundancy_cluster_080"]
    )
    assert (
        by_name.loc["f_c", "redundancy_cluster_080"]
        != by_name.loc["f_a", "redundancy_cluster_080"]
    )


def test_representative_selection_deterministic() -> None:
    registry = pd.DataFrame(
        [
            {
                "name": "a",
                "formula": "x",
                "primitive_dependencies": "p1",
                "pit_status": "PASS",
            },
            {
                "name": "b",
                "formula": "x+y+z",
                "primitive_dependencies": "p1,p2,p3",
                "pit_status": "PASS",
            },
        ]
    )
    cluster = pd.DataFrame(
        [
            {
                "factor": "a",
                "redundancy_cluster_080": "R1",
                "row_coverage": 0.9,
                "icir_lite": 1.0,
                "max_abs_corr_to_existing": 0.2,
                "pit_status": "PASS",
            },
            {
                "factor": "b",
                "redundancy_cluster_080": "R1",
                "row_coverage": 0.9,
                "icir_lite": 2.0,
                "max_abs_corr_to_existing": 0.2,
                "pit_status": "PASS",
            },
        ]
    )
    once = select_cluster_representatives(cluster, registry)
    twice = select_cluster_representatives(cluster, registry)
    assert once["is_representative"].tolist() == twice["is_representative"].tolist()
    # Simpler formula wins over higher ICIR when coverage/PIT tie.
    kept = set(once.loc[once["is_representative"], "factor"])
    assert "a" in kept


# ---------------------------------------------------------------------------
# E. Novelty
# ---------------------------------------------------------------------------


def test_identical_candidate_is_near_alias() -> None:
    dates = _calendar(6)
    symbols = [f"S{i:03d}" for i in range(30)]
    values = pd.DataFrame(
        np.arange(30, dtype=float)[None, :].repeat(6, axis=0),
        index=dates,
        columns=symbols,
    )
    panel = pd.concat(
        [values.stack().rename("cand"), values.stack().rename("frozen")],
        axis=1,
    ).reset_index()
    panel.columns = ["TradeDate", "Symbol", "cand", "frozen"]
    out = novelty_vs_existing(
        panel,
        ["cand"],
        ["frozen"],
        {"frozen": "trade_flow"},
        min_names=10,
    )
    assert out.iloc[0]["novelty_bucket"] == "NEAR_ALIAS"
    assert out.iloc[0]["max_abs_corr_to_existing"] == pytest.approx(1.0, abs=1e-9)
    assert novelty_bucket(0.4) == "HIGH_NOVELTY"
    assert novelty_bucket(0.6) == "MEDIUM_NOVELTY"
    assert novelty_bucket(0.8) == "LOW_NOVELTY"


# ---------------------------------------------------------------------------
# F. Decile
# ---------------------------------------------------------------------------


def test_monotonic_factor_high_decile_mono() -> None:
    n = 100
    signal = pd.Series(np.arange(n, dtype=float))
    ret = pd.Series(np.arange(n, dtype=float))
    day = decile_lite_one_day(signal, ret, n_groups=10, min_names=30)
    assert day is not None
    assert day["mono"] == pytest.approx(1.0, abs=1e-9)
    assert day["spread"] > 0
    bins = assign_deciles(signal, n_groups=10)
    assert set(bins.dropna().unique()) == set(float(i) for i in range(1, 11))


# ---------------------------------------------------------------------------
# G. Determinism
# ---------------------------------------------------------------------------


def test_priority_score_frozen_and_deterministic() -> None:
    row = {
        "rank_ic_lite": 0.02,
        "icir_lite": 2.0,
        "decile_mono_lite": 0.8,
        "coverage": 0.9,
        "max_abs_corr_to_existing": 0.3,
    }
    a = discovery_priority_score(row)
    b = discovery_priority_score(row)
    assert a == b
    assert 0 <= a <= 1


def test_contract_thresholds_not_placeholders() -> None:
    assert BDL_CONTRACT["coverage_threshold"] == 0.50
    assert BDL_CONTRACT["date_stride"] == 5
    assert BDL_CONTRACT["redundancy_corr_threshold"] == 0.80
    assert BDL_CONTRACT["near_alias_threshold"] == 0.90
    assert BDL_CONTRACT["decile_mono_threshold"] == 0.50
    assert BDL_CONTRACT["universe_definition"] == "000852.SH"
    assert len(BDL_CONTRACT["dry_run_existing_factors"]) >= 20
    assert len(BDL_CONTRACT["dry_run_existing_factors"]) <= 40
