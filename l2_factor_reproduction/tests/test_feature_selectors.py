"""Pytest coverage for FS-2 selector engine (synthetic only)."""

from __future__ import annotations

import numpy as np
import pytest

from l2_factor_reproduction.feature_selection.selector_diagnostics import (
    BH_HANDCHECK_ALPHA,
    BH_HANDCHECK_P,
    expected_bh_handcheck,
    fpr_vs_fdr_pvalue_fixture,
    make_fixture_constant,
    make_fixture_linear,
    metadata_frame,
)
from l2_factor_reproduction.feature_selection.selectors import (
    RESULT_COLUMNS,
    benjamini_hochberg_reject,
    build_selector,
    run_selector,
    validate_params,
)


def test_bh_handcrafted() -> None:
    reject = benjamini_hochberg_reject(BH_HANDCHECK_P, BH_HANDCHECK_ALPHA)
    got = [i + 1 for i, r in enumerate(reject) if r]
    assert got == expected_bh_handcheck()


def test_bh_edge_cases() -> None:
    assert not benjamini_hochberg_reject(np.array([0.2, 0.3, 0.4]), 0.05).any()
    assert benjamini_hochberg_reject(np.array([1e-6, 2e-6, 3e-6]), 0.05).all()
    out = benjamini_hochberg_reject(np.array([0.001, np.nan, 0.2]), 0.05)
    assert out.tolist() == [True, False, False]


def test_fpr_vs_fdr_not_alias() -> None:
    _p, _a, fpr, fdr = fpr_vs_fdr_pvalue_fixture()
    assert int(fpr.sum()) > int(fdr.sum())
    assert not np.array_equal(fpr, fdr)


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        validate_params("F_REGRESSION_KBEST", {"k": 0})
    with pytest.raises(ValueError):
        validate_params("F_REGRESSION_FDR", {"alpha": 1.0})
    with pytest.raises(ValueError):
        validate_params("MI_REGRESSION_KBEST", {"k": 2, "n_neighbors": 3})


def test_constant_feature() -> None:
    fix = make_fixture_constant()
    res = run_selector(
        "F_REGRESSION_KBEST",
        fix.X,
        fix.y,
        feature_names=fix.feature_names,
        feature_metadata=metadata_frame(fix),
        params={"k": 1},
    )
    row = res.table.set_index("feature").loc["x_constant"]
    assert bool(row["is_constant"]) is True
    assert bool(row["selected"]) is False
    assert row["status"] == "CONSTANT"
    assert list(res.table.columns) == list(RESULT_COLUMNS)


def test_build_selector_api() -> None:
    fix = make_fixture_linear()
    sel = build_selector("F_REGRESSION_KBEST", {"k": 2})
    sel.fit(fix.X, fix.y, feature_names=fix.feature_names)
    assert sel.get_support().sum() == 2
    assert len(sel.get_result()) == len(fix.feature_names)
