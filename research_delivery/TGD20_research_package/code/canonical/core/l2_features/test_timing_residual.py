"""Unit tests for timing residual engine (TGD Stage 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.l2_features.timing_residual import (
    attach_session_controls_from_minute,
    cs_ols_residual,
    merge_centers_with_controls,
    normalize_timing_feature_columns,
    prepare_tgd_from_residuals,
    residualize_timing_centers,
    segment_cum_return,
)


def test_cs_ols_residual_recovers_known_model():
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eps = rng.normal(scale=0.1, size=n)
    y = 1.5 + 2.0 * x1 - 0.5 * x2 + eps
    X = np.column_stack([x1, x2])
    resid = cs_ols_residual(y, X, min_obs=30)
    assert np.isfinite(resid).sum() == n
    # residuals should be close to true eps (same span)
    assert abs(resid.mean()) < 0.05
    assert np.corrcoef(resid, eps)[0, 1] > 0.9


def test_cs_ols_insufficient_names_all_nan():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1.0], [2.0], [3.0]])
    resid = cs_ols_residual(y, X, min_obs=30)
    assert np.isnan(resid).all()


def test_cs_ols_nan_rows_excluded():
    y = np.array([1.0, np.nan, 3.0, 4.0] + [float(i) for i in range(5, 40)])
    x = np.arange(len(y), dtype=float).reshape(-1, 1)
    x[1, 0] = np.nan
    resid = cs_ols_residual(y, x, min_obs=20)
    assert np.isnan(resid[1])
    assert np.isfinite(resid[0])
    assert np.isfinite(resid[2])


def _synthetic_daily(n_names: int = 60, n_days: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        ru = rng.normal(0.001, 0.001, n_names)
        rd = rng.normal(-0.001, 0.001, n_names)
        r1 = rng.normal(0, 0.01, n_names)
        r2 = rng.normal(0, 0.01, n_names)
        ovn = rng.normal(0, 0.005, n_names)
        # Gu/Gd linearly depend on controls + noise
        Gu = 50 + 1000 * ru + 20 * r1 + 10 * r2 + 5 * ovn + rng.normal(0, 1, n_names)
        Gd = 80 + 800 * (-rd) + 15 * r1 + 12 * r2 + 4 * ovn + rng.normal(0, 1, n_names)
        for i in range(n_names):
            rows.append(
                {
                    "date": date,
                    "symbol": f"{i:04d}.SZ",
                    "Gu": Gu[i],
                    "Gd": Gd[i],
                    "avg_up_return": ru[i],
                    "avg_down_return": rd[i],
                    "R1": r1[i],
                    "R2": r2[i],
                    "overnight_return": ovn[i],
                }
            )
    return pd.DataFrame(rows)


def test_residualize_timing_centers_outputs_epsilons():
    daily = _synthetic_daily()
    out = residualize_timing_centers(daily, min_obs=30)
    assert {"epsilon_u", "epsilon_d", "n_cs"}.issubset(out.columns)
    assert out.attrs.get("tgd_stage") == "timing_residual"
    # most residuals finite
    assert out["epsilon_u"].notna().mean() > 0.9
    assert out["epsilon_d"].notna().mean() > 0.9
    # within day, residuals roughly zero-mean
    for _, g in out.groupby("date"):
        assert abs(g["epsilon_u"].mean()) < 0.2
        assert abs(g["epsilon_d"].mean()) < 0.2


def test_residualize_accepts_mean_up_alias():
    daily = _synthetic_daily(n_names=40, n_days=1)
    daily = daily.rename(columns={"avg_up_return": "mean_up", "avg_down_return": "mean_down"})
    out = residualize_timing_centers(daily, min_obs=30)
    assert out["epsilon_u"].notna().any()


def test_residualize_too_few_names():
    daily = _synthetic_daily(n_names=10, n_days=1)
    out = residualize_timing_centers(daily, min_obs=30)
    assert out["epsilon_u"].isna().all()
    assert out["epsilon_d"].isna().all()


def test_segment_cum_return():
    r = np.array([np.nan, 0.01, 0.02, -0.01, 0.03])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert segment_cum_return(r, t, -1, 2) == pytest.approx(0.03)
    assert segment_cum_return(r, t, 2, 4) == pytest.approx(0.02)


def test_attach_session_controls_from_minute():
    schedule = [
        ("09:31:00", 100.0),
        ("09:45:00", 101.0),
        ("10:00:00", 102.0),  # still in R1 (t<=29): 09:31=0 ... 10:00=29
        ("10:15:00", 101.0),  # R2
        ("10:30:00", 100.5),
    ]
    rows = []
    for clock, c in schedule:
        rows.append(
            {
                "date": "2024-06-03",
                "symbol": "AAA",
                "bartime": f"2024-06-03 {clock}",
                "close": c,
            }
        )
    minute = pd.DataFrame(rows)
    ctrl = attach_session_controls_from_minute(minute)
    assert len(ctrl) == 1
    assert np.isfinite(ctrl.loc[0, "R1"])
    assert np.isfinite(ctrl.loc[0, "R2"])


def test_merge_and_prepare_stub():
    centers = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "symbol": ["X"],
            "Gu": [10.0],
            "Gd": [20.0],
            "mean_up": [0.01],
            "mean_down": [-0.01],
        }
    )
    controls = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "symbol": ["X"],
            "R1": [0.001],
            "R2": [-0.002],
            "overnight_return": [0.0],
        }
    )
    merged = merge_centers_with_controls(centers, controls)
    assert "avg_up_return" in normalize_timing_feature_columns(merged).columns
    # expand to enough names for residualize not needed here
    stub = prepare_tgd_from_residuals(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "symbol": ["X"],
                "epsilon_u": [0.1],
                "epsilon_d": [-0.2],
            }
        )
    )
    assert stub.attrs.get("tgd_stage") == "residuals_ready"
