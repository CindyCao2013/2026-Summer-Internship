"""FS-3 regression checks: labels, purge, jaccard, schema freeze, no-model scope."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.feature_selection.labels import (
    audit_label_boundaries,
    build_labels_wide_panel,
    compound_excess_path,
    next_trading_dates,
    recover_stock_returns,
)
from l2_factor_reproduction.feature_selection.selection_analysis import selection_frequency
from l2_factor_reproduction.feature_selection.selectors import (
    CANONICAL_SELECTORS,
    feature_schema_hash,
)
from l2_factor_reproduction.feature_selection.walkforward import (
    EXPECTED_FEATURE_SCHEMA_HASH,
    TRAINING_WINDOW_MONTHS,
    build_walkforward_windows,
    jaccard,
    month_end_oos_anchors,
    training_date_bounds,
)

PROJ = Path(__file__).resolve().parents[2]
FS1_INV = (
    PROJ
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs1_feature_panel"
    / "feature_inventory.csv"
)


def _toy_market(n_dates: int = 80, n_sym: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    cols = [f"S{i}.SZ" for i in range(n_sym)]
    stock = pd.DataFrame(rng.normal(0, 0.01, size=(n_dates, n_sym)), index=dates, columns=cols)
    bench = pd.Series(rng.normal(0, 0.005, size=n_dates), index=dates, name="benchmark_ret")
    excess = stock.sub(bench, axis=0)
    return excess, bench, pd.DatetimeIndex(dates)


def test_a_label_horizons_exact_trading_days():
    excess, bench, dates = _toy_market()
    lab = build_labels_wide_panel(excess, bench, dates)
    mid = dates[40]
    for h in (1, 5, 20):
        end = lab["_meta_end"][h].loc[mid]
        assert pd.notna(end)
        pos = dates.get_loc(mid)
        end_pos = dates.get_loc(pd.Timestamp(end))
        assert end_pos - pos == h
        w = next_trading_dates(dates, mid, h)
        assert w is not None and len(w) == h


def test_b_no_truncated_labels_at_end():
    excess, bench, dates = _toy_market(n_dates=40)
    lab = build_labels_wide_panel(excess, bench, dates)
    boundary = audit_label_boundaries(dates, lab["_meta_end"], lab["_meta_valid"])
    assert boundary["pass"].all()


def test_c_label_parity_y1_and_compound():
    excess, bench, dates = _toy_market()
    lab = build_labels_wide_panel(excess, bench, dates)
    stock = recover_stock_returns(excess, bench)
    dt = dates[10]
    nxt = dates[11]
    sym = excess.columns[0]
    assert abs(float(lab[1].loc[dt, sym]) - float(excess.loc[nxt, sym])) < 1e-12
    w = dates[11:16]
    ref = compound_excess_path(stock.loc[w, sym], bench.reindex(w))
    assert abs(float(lab[5].loc[dt, sym]) - ref) < 1e-12


def test_d_purge_no_overlap():
    excess, bench, dates = _toy_market(n_dates=400)
    lab = build_labels_wide_panel(excess, bench, dates)
    ends = {h: lab["_meta_end"][h] for h in (1, 5, 20)}
    windows = build_walkforward_windows(dates, ends)
    for w in windows:
        if w.status != "OK":
            continue
        assert pd.notna(w.train_label_end_max)
        assert w.train_label_end_max < w.oos_anchor


def test_e_deterministic_month_end_anchors():
    dates = pd.bdate_range("2020-01-01", periods=120)
    a1 = month_end_oos_anchors(pd.DatetimeIndex(dates))
    a2 = month_end_oos_anchors(pd.DatetimeIndex(dates))
    assert list(a1) == list(a2)
    # each anchor is last business day of its month in the calendar
    for a in a1:
        m = a.to_period("M")
        month_dates = [d for d in dates if pd.Timestamp(d).to_period("M") == m]
        assert a == max(month_dates)


def test_f_training_window_months():
    excess, bench, dates = _toy_market(n_dates=800)
    lab = build_labels_wide_panel(excess, bench, dates)
    oos = month_end_oos_anchors(dates)[-1]
    start, end, status = training_date_bounds(
        dates,
        oos,
        training_window_months=TRAINING_WINDOW_MONTHS,
        horizon=5,
        label_end_by_feature_date=lab["_meta_end"][5],
    )
    assert status == "OK"
    assert start is not None and end is not None
    # approximate: start >= end - 24 months
    assert start >= end - pd.DateOffset(months=TRAINING_WINDOW_MONTHS) - pd.Timedelta(days=5)


def test_g_selector_contract_names_frozen():
    assert list(CANONICAL_SELECTORS) == [
        "F_REGRESSION_KBEST",
        "MI_REGRESSION_KBEST",
        "F_REGRESSION_FPR",
        "F_REGRESSION_FDR",
        "L1_REGRESSION",
        "TREE_IMPORTANCE_REGRESSION",
    ]


def test_h_feature_schema_hash_frozen():
    if not FS1_INV.exists():
        pytest.skip("FS-1 inventory missing")
    inv = pd.read_csv(FS1_INV)
    elig = inv.loc[inv["eligible_for_fs"] == True]  # noqa: E712
    names = elig["factor"].tolist()
    fam = dict(zip(elig["factor"], elig["family"].astype(str)))
    assert len(names) == 127
    assert feature_schema_hash(names, families=fam) == EXPECTED_FEATURE_SCHEMA_HASH


def test_i_frequency_denominator_uses_local_eligibility():
    rows = []
    for i, oos in enumerate(["2021-01-29", "2021-02-26", "2021-03-31"]):
        rows.append(
            {
                "feature": "f1",
                "family": "fam",
                "selector_name": "F_REGRESSION_KBEST",
                "horizon": 1,
                "oos_anchor": oos,
                "selected": i == 0,
                "locally_eligible": i < 2,  # third window ineligible
                "effective_n": 1000 if i < 2 else 10,
                "coverage_ratio": 0.9,
            }
        )
    freq = selection_frequency(pd.DataFrame(rows))
    r = freq.iloc[0]
    assert r["n_eligible_windows"] == 2
    assert r["n_selected_windows"] == 1
    assert abs(r["selection_frequency"] - 0.5) < 1e-12
    assert abs(r["availability_frequency"] - 2 / 3) < 1e-12


def test_j_jaccard_toy():
    jac, inter, union, empty = jaccard(["a", "b"], ["b", "c"])
    assert inter == 1 and union == 3
    assert abs(jac - 1 / 3) < 1e-12
    jac2, _, _, empty2 = jaccard([], [])
    assert empty2 and np.isnan(jac2)


def test_k_no_learner_imports_in_fs3_modules():
    files = [
        PROJ / "l2_factor_reproduction" / "feature_selection" / "labels.py",
        PROJ / "l2_factor_reproduction" / "feature_selection" / "walkforward.py",
        PROJ / "l2_factor_reproduction" / "feature_selection" / "selection_analysis.py",
        PROJ / "l2_factor_reproduction" / "feature_selection" / "fs3_runner.py",
        PROJ
        / "l2_factor_reproduction"
        / "scripts"
        / "run_l2_walkforward_feature_selection.py",
    ]
    banned_modules = {"xgboost", "lightgbm", "catboost"}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned_modules, f"{path.name} imports {alias.name}"
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                assert mod not in banned_modules, f"{path.name} imports from {node.module}"
                for alias in node.names:
                    assert alias.name != "LogisticRegression"
                    assert alias.name != "XGBClassifier"
                    assert alias.name != "XGBRegressor"



def test_l_fast_discovery_immutability_marker():
    # FS-3 must not rewrite discovery outputs; marker file remains absent of FS-3 writes
    fd = PROJ / "research" / "results" / "l2_reproduction" / "fast_discovery"
    # just assert we did not create an fs3 marker inside FD
    assert not (fd / "FS3_MUTATION.txt").exists()
