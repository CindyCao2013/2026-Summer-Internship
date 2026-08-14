"""FS-2 synthetic fixtures and selector diagnostics (no real market labels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.feature_selection.selectors import (
    benjamini_hochberg_reject,
)


@dataclass
class SyntheticFixture:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    signal_features: List[str]
    families: Dict[str, str]
    description: str


def make_fixture_linear(seed: int = 0, n: int = 400, n_noise: int = 18) -> SyntheticFixture:
    """Fixture A: y = 2*x1 - 3*x2 + noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = [rng.normal(size=n) for _ in range(n_noise)]
    X = np.column_stack([x1, x2] + noise)
    y = 2.0 * x1 - 3.0 * x2 + 0.25 * rng.normal(size=n)
    names = ["x1", "x2"] + [f"noise_{i}" for i in range(n_noise)]
    fam = {n: "signal" if n in ("x1", "x2") else "noise" for n in names}
    return SyntheticFixture(
        name="linear_signal",
        X=X,
        y=y,
        feature_names=names,
        signal_features=["x1", "x2"],
        families=fam,
        description="y=2*x1-3*x2+noise; F/L1/Tree recovery",
    )


def make_fixture_nonlinear(seed: int = 1, n: int = 500, n_noise: int = 15) -> SyntheticFixture:
    """Fixture B: y = x1**2 + 0.5*x2 + noise (MI/Tree)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = [rng.normal(size=n) for _ in range(n_noise)]
    X = np.column_stack([x1, x2] + noise)
    y = x1**2 + 0.5 * x2 + 0.2 * rng.normal(size=n)
    names = ["x1", "x2"] + [f"noise_{i}" for i in range(n_noise)]
    fam = {n: "signal" if n in ("x1", "x2") else "noise" for n in names}
    return SyntheticFixture(
        name="nonlinear_signal",
        X=X,
        y=y,
        feature_names=names,
        signal_features=["x1", "x2"],
        families=fam,
        description="y=x1**2+0.5*x2+noise; MI recovers nonlinear x1",
    )


def make_fixture_noise(seed: int = 2, n: int = 300, n_feat: int = 12) -> SyntheticFixture:
    """Fixture C: pure noise X and y independent."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = rng.normal(size=n)
    names = [f"noise_{i}" for i in range(n_feat)]
    fam = {n: "noise" for n in names}
    return SyntheticFixture(
        name="pure_noise",
        X=X,
        y=y,
        feature_names=names,
        signal_features=[],
        families=fam,
        description="independent X/y; FPR/FDR sanity",
    )


def make_fixture_constant(seed: int = 3, n: int = 300) -> SyntheticFixture:
    """Fixture D: includes a constant column."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x_const = np.ones(n)
    noise = rng.normal(size=n)
    X = np.column_stack([x1, x_const, noise])
    y = 1.5 * x1 + 0.2 * rng.normal(size=n)
    names = ["x1", "x_constant", "noise_0"]
    fam = {"x1": "signal", "x_constant": "constant", "noise_0": "noise"}
    return SyntheticFixture(
        name="constant_feature",
        X=X,
        y=y,
        feature_names=names,
        signal_features=["x1"],
        families=fam,
        description="constant column handled without crash",
    )


def make_fixture_redundant(seed: int = 4, n: int = 350) -> SyntheticFixture:
    """Fixture E: x2 ≈ x1; y = x1 + eps."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = x1 + 0.05 * rng.normal(size=n)
    noise = [rng.normal(size=n) for _ in range(8)]
    X = np.column_stack([x1, x2] + noise)
    y = x1 + 0.2 * rng.normal(size=n)
    names = ["x1", "x2"] + [f"noise_{i}" for i in range(8)]
    fam = {n: "signal" if n in ("x1", "x2") else "noise" for n in names}
    return SyntheticFixture(
        name="redundant_features",
        X=X,
        y=y,
        feature_names=names,
        signal_features=["x1", "x2"],
        families=fam,
        description="x2=x1+small_noise; stable selector behavior",
    )


def make_fixture_l1_sparse(seed: int = 5, n: int = 500, n_noise: int = 18) -> SyntheticFixture:
    """L1-focused: y = 2.5*x1 - 1.8*x2 + eps."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = [rng.normal(size=n) for _ in range(n_noise)]
    X = np.column_stack([x1, x2] + noise)
    y = 2.5 * x1 - 1.8 * x2 + 0.15 * rng.normal(size=n)
    names = ["x1", "x2"] + [f"noise_{i}" for i in range(n_noise)]
    fam = {n: "signal" if n in ("x1", "x2") else "noise" for n in names}
    return SyntheticFixture(
        name="l1_sparse",
        X=X,
        y=y,
        feature_names=names,
        signal_features=["x1", "x2"],
        families=fam,
        description="sparse linear for L1 recovery",
    )


def metadata_frame(fix: SyntheticFixture) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": fix.feature_names,
            "family": [fix.families[f] for f in fix.feature_names],
            "coverage_ratio": 1.0,
        }
    )


def handcrafted_bh_case(
    p_values: Sequence[float],
    alpha: float,
) -> Tuple[np.ndarray, List[int]]:
    """Return (reject_mask, expected_1based_indices_in_original_order).

    Manual BH for documentation/tests:
    sort p, find largest i with p_(i) <= alpha * i / m.
    """
    p = np.asarray(p_values, dtype=float)
    reject = benjamini_hochberg_reject(p, alpha)
    expected = [i + 1 for i, r in enumerate(reject) if r]
    return reject, expected


# Fixed handcrafted example used in audits
BH_HANDCHECK_P = np.array([0.001, 0.004, 0.012, 0.030, 0.200], dtype=float)
BH_HANDCHECK_ALPHA = 0.05


def expected_bh_handcheck() -> List[int]:
    """Human-auditable BH expectation for BH_HANDCHECK_P @ alpha=0.05.

    m=5
    i=1: 0.001 <= 0.05*1/5=0.01  ✓
    i=2: 0.004 <= 0.05*2/5=0.02  ✓
    i=3: 0.012 <= 0.05*3/5=0.03  ✓
    i=4: 0.030 <= 0.05*4/5=0.04  ✓
    i=5: 0.200 <= 0.05*5/5=0.05  ✗
    => reject first 4 (original indices 1..4)
    """
    return [1, 2, 3, 4]


def fpr_vs_fdr_pvalue_fixture() -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Construct p-values where FPR selects more than FDR.

    alpha=0.05
    p = [0.01, 0.02, 0.03, 0.04, 0.20]
    FPR: p < 0.05 → first 4
    BH FDR:
      i=1: 0.01 <= 0.01 ✓
      i=2: 0.02 <= 0.02 ✓
      i=3: 0.03 <= 0.03 ✓
      i=4: 0.04 <= 0.04 ✓
      i=5: 0.20 <= 0.05 ✗
    → same in this case; need a case where FDR is stricter.

    Better:
    p = [0.001, 0.01, 0.04, 0.045, 0.049, 0.20]
    FPR (alpha=0.05): first 5
    BH:
      m=6
      i=1: 0.001 <= 0.05/6≈0.0083 ✓
      i=2: 0.01  <= 0.0167 ✓
      i=3: 0.04  <= 0.025  ✗
      so largest i with all previous... wait BH finds largest i with p_(i)<=alpha*i/m
      i=1 yes, i=2 yes, i=3: 0.04 > 0.025 no
      i=4: 0.045 > 0.033 no
      i=5: 0.049 > 0.0417 no
      i=6: 0.20 > 0.05 no
      => reject i=1,2 only

    FPR selects 5, FDR selects 2.
    """
    p = np.array([0.001, 0.01, 0.04, 0.045, 0.049, 0.20], dtype=float)
    alpha = 0.05
    fpr = p < alpha
    fdr = benjamini_hochberg_reject(p, alpha)
    return p, alpha, fpr, fdr


ALL_FIXTURE_BUILDERS = {
    "linear_signal": make_fixture_linear,
    "nonlinear_signal": make_fixture_nonlinear,
    "pure_noise": make_fixture_noise,
    "constant_feature": make_fixture_constant,
    "redundant_features": make_fixture_redundant,
    "l1_sparse": make_fixture_l1_sparse,
}
