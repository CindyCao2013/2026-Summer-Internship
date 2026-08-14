"""AlphaNet vs explicit daily-factor comparison (synthetic panels only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphanet.compare import (
    build_classic_style_factors,
    correlation_table,
    daily_cs_spearman,
    load_alphanet_factor,
    make_synthetic_alphanet,
    overlap_verdict,
    residualize_panel,
    run_comparison,
    select_pool_representatives,
    to_wide,
)
from alphanet.config import EvalConfig
from alphanet.data import panel_from_synthetic
from alphanet.neutralize import style_panels
from alphanet.synthetic import make_synthetic_panel


def test_to_wide_long_and_wide_roundtrip():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = ["000001.SZ", "000002.SZ"]
    wide = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], index=idx, columns=cols)
    long = wide.stack().rename("value").reset_index()
    long.columns = ["date", "symbol", "value"]
    got = to_wide(long)
    pd.testing.assert_frame_equal(got, wide, check_freq=False)
    pd.testing.assert_frame_equal(to_wide(wide), wide, check_freq=False)


def test_cs_spearman_identical_and_independent():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=25)
    cols = ["s{:02d}".format(i) for i in range(40)]
    a = pd.DataFrame(rng.normal(size=(25, 40)), index=idx, columns=cols)
    same = daily_cs_spearman(a, a, min_obs=10)
    assert float(same.mean()) > 0.99
    b = pd.DataFrame(rng.normal(size=(25, 40)), index=idx, columns=cols)
    indep = daily_cs_spearman(a, b, min_obs=10)
    assert abs(float(indep.mean())) < 0.15


def test_classic_styles_match_style_panels_not_random():
    synth = make_synthetic_panel(n_days=40, n_stocks=20, seed=4)
    panel = panel_from_synthetic(synth)
    styles = build_classic_style_factors(panel, window=10)
    expected = style_panels(panel.ret_1d, panel.features["turn"], 10)
    pd.testing.assert_frame_equal(styles["momentum_10d"], expected["momentum"])
    pd.testing.assert_frame_equal(styles["volatility_10d"], expected["volatility"])
    pd.testing.assert_frame_equal(styles["turnover_10d"], expected["turnover"])
    pd.testing.assert_frame_equal(styles["size"], panel.log_mcap)


def test_residualize_removes_linear_size():
    synth = make_synthetic_panel(n_days=50, n_stocks=36, seed=5)
    panel = panel_from_synthetic(synth)
    y = 2.0 * panel.log_mcap + 0.01 * panel.ret_1d
    resid = residualize_panel(y, {"size": panel.log_mcap}, min_obs=20)
    corr = daily_cs_spearman(resid, panel.log_mcap, min_obs=20)
    assert abs(float(corr.mean())) < 0.05
    assert float(resid.stack().std()) < 0.5 * float(y.stack().std())


def test_overlap_verdict_thresholds():
    assert overlap_verdict(0.12)[0] == "new_information"
    assert overlap_verdict(0.45)[0] == "partial_overlap"
    assert overlap_verdict(0.72)[0] == "likely_remix"


def test_select_representatives_skips_alias_one_per_family():
    summary = pd.DataFrame(
        {
            "factor": ["a1", "a2", "b1", "b2"],
            "family": ["trade_flow", "trade_flow", "order_size", "order_size"],
            "icir_raw": [1.0, 5.0, 3.0, 2.0],
            "near_alias_observed": [False, True, False, False],
        }
    )
    picked = select_pool_representatives(summary, max_per_family=1, max_total=10)
    names = set(picked["factor"])
    assert "a2" not in names
    assert "a1" in names
    assert "b1" in names
    assert "b2" not in names


def test_run_comparison_writes_report(tmp_path: Path):
    synth = make_synthetic_panel(n_days=70, n_stocks=40, seed=8)
    panel = panel_from_synthetic(synth)
    styles = build_classic_style_factors(panel, window=10)
    alpha = make_synthetic_alphanet(styles, seed=8)
    extra = pd.DataFrame(
        np.random.default_rng(3).normal(size=alpha.shape),
        index=alpha.index,
        columns=alpha.columns,
    )
    result = run_comparison(
        alpha,
        styles,
        {"pool_dummy": extra},
        variant="smoke",
        ret_1d=panel.ret_1d,
        mask=panel.tradable,
        eval_cfg=EvalConfig(n_groups=10, rebalance_every=3, min_cs_obs=8, fee_one_way=0.00075),
        output_dir=tmp_path,
        pool_status=pd.DataFrame(
            [{"factor": "pool_dummy", "family": "toy", "path": "", "status": "ok", "shape": ""}]
        ),
        data_note="unit test",
        min_obs=8,
    )
    assert result["report"].exists()
    text = result["report"].read_text(encoding="utf-8")
    assert "AlphaNet-smoke" in text
    assert "resid_classic_styles" in text
    mom_abs = float(result["corr_summary"].loc["momentum_10d", "mean_abs_spearman"])
    dummy_abs = float(result["corr_summary"].loc["pool_dummy", "mean_abs_spearman"])
    assert mom_abs > dummy_abs
    assert mom_abs > 0.40
    assert (tmp_path / "alphanet_smoke_mean_corr_heatmap.png").exists()


def test_load_alphanet_missing(tmp_path: Path, monkeypatch):
    from alphanet import compare as compare_mod

    monkeypatch.setattr(compare_mod, "FACTORS", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_alphanet_factor("v1")


def test_correlation_table_orders_by_abs():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=20)
    cols = ["s{:02d}".format(i) for i in range(30)]
    alpha = pd.DataFrame(rng.normal(size=(20, 30)), index=idx, columns=cols)
    strong = alpha * 0.9 + rng.normal(scale=0.1, size=alpha.shape)
    weak = pd.DataFrame(rng.normal(size=(20, 30)), index=idx, columns=cols)
    _, summary = correlation_table(alpha, {"strong": strong, "weak": weak}, min_obs=10)
    assert list(summary.index[:2]) == ["strong", "weak"]
