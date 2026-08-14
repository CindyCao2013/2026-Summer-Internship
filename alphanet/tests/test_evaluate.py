"""Dataset, leakage, neutralization, 10-layer protocol (synthetic only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphanet.config import EvalConfig, TrainConfig
from alphanet.data import cs_zscore, forward_return, panel_from_synthetic
from alphanet.dataset import AlphaNetDataset, in_sample_window, split_train_val_dates
from alphanet.evaluate import assign_deciles, decile_backtest, ic_test
from alphanet.neutralize import neutralize_panel
from alphanet.rolling import assert_fold_no_leak, build_folds, retrain_asofs
from alphanet.synthetic import make_synthetic_panel, stack_images
from alphanet.universe import next_session_tradable
from alphanet.variants import get_config, smoke_config


def test_stack_images_uses_only_past_inclusive():
    synth = make_synthetic_panel(n_days=40, n_stocks=12, seed=1)
    date = synth.calendar[20]
    images, symbols = stack_images(synth.features, date, lookback=10)
    assert images.shape[1:] == (9, 10)
    # last column of close equals close on `date`
    loc = synth.calendar.get_loc(date)
    expected = synth.features["close"].iloc[loc].reindex(symbols).to_numpy()
    np.testing.assert_allclose(images[:, 2, -1], expected, rtol=1e-5)
    # first column equals close 9 days earlier, not a future date
    expected0 = synth.features["close"].iloc[loc - 9].reindex(symbols).to_numpy()
    np.testing.assert_allclose(images[:, 2, 0], expected0, rtol=1e-5)


def test_forward_return_paper_vs_executable():
    synth = make_synthetic_panel(n_days=30, n_stocks=8, seed=2)
    paper = forward_return(synth.ret_1d, horizon=3, execution="paper_c2c")
    exe = forward_return(synth.ret_1d, horizon=3, execution="executable_tplus1")
    # paper at T uses r[T+1..T+3]; executable uses r[T+2..T+4]
    t = synth.calendar[10]
    one = 1.0 + synth.ret_1d
    manual_paper = one.loc[synth.calendar[11]] * one.loc[synth.calendar[12]] * one.loc[synth.calendar[13]] - 1
    np.testing.assert_allclose(paper.loc[t], manual_paper, rtol=1e-6)
    assert not np.allclose(paper.loc[t].fillna(0), exe.loc[t].fillna(0))


def test_cs_zscore_uses_same_date_only():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = ["a", "b", "c", "d"]
    panel = pd.DataFrame(rng.normal(size=(5, 4)), index=idx, columns=cols)
    z = cs_zscore(panel, min_obs=3)
    row = z.iloc[2]
    assert abs(float(row.mean())) < 1e-10
    assert abs(float(row.std(ddof=0)) - 1.0) < 1e-10


def test_train_val_split_is_time_ordered():
    cal = pd.bdate_range("2018-01-01", periods=40)
    tr, va = split_train_val_dates(cal, every=2, start=cal[0], end=cal[-1])
    assert len(tr) > 0 and len(va) > 0
    assert tr[-1] < va[0]


def test_dataset_labels_finite_and_images_shape():
    pytest.importorskip("torch")
    synth = make_synthetic_panel(n_days=50, n_stocks=20, seed=3)
    panel = panel_from_synthetic(synth)
    cfg = TrainConfig(horizon=3, sample_every=2, execution="paper_c2c")
    dates = panel.calendar[12:-5:2]
    ds = AlphaNetDataset(panel, dates, cfg, lookback=10, require_label=True)
    assert len(ds) > 10
    x, y = ds[0]
    assert tuple(x.shape) == (9, 10)
    assert np.isfinite(float(y))


def test_next_session_mask_shifts_forward():
    idx = pd.bdate_range("2020-01-01", periods=4)
    m = pd.DataFrame([[1, 1], [1, np.nan], [1, 1], [np.nan, 1]], index=idx, columns=["a", "b"])
    nxt = next_session_tradable(m)
    assert np.isnan(nxt.iloc[0, 1])
    assert nxt.iloc[1, 1] == 1


def test_neutralize_kills_size_exposure():
    synth = make_synthetic_panel(n_days=25, n_stocks=40, seed=4)
    # construct a factor that is mostly log_mcap
    signal = synth.log_mcap + 0.01 * synth.ret_1d
    neut = neutralize_panel(
        signal,
        industry=synth.industry,
        log_mcap=synth.log_mcap,
        ret_1d=synth.ret_1d,
        turn=synth.features["turn"],
        horizon=5,
        min_obs=15,
    )
    # residual should have near-zero cross-sectional corr with size on most days
    corrs = []
    for dt in signal.index[8:]:
        a = neut.loc[dt]
        b = synth.log_mcap.loc[dt]
        ok = a.notna() & b.notna()
        if int(ok.sum()) < 15:
            continue
        corrs.append(a[ok].corr(b[ok]))
    assert np.nanmean(np.abs(corrs)) < 0.15


def test_deciles_g1_is_top():
    idx = pd.bdate_range("2020-01-01", periods=3)
    cols = ["s{}".format(i) for i in range(20)]
    data = np.tile(np.arange(20, dtype=float), (3, 1))
    signal = pd.DataFrame(data, index=idx, columns=cols)
    groups = assign_deciles(signal, n=10, g1_is_top=True)
    # highest factor value is 19 → G1
    assert int(groups.iloc[0, 19]) == 1
    assert int(groups.iloc[0, 0]) == 10


def test_decile_backtest_hl_is_g1_minus_g10():
    synth = make_synthetic_panel(n_days=60, n_stocks=40, seed=5)
    signal = synth.ret_1d.shift(1)
    cfg = EvalConfig(n_groups=10, fee_one_way=0.00075, rebalance_every=5, g1_is_top=True)
    result = decile_backtest(signal, synth.ret_1d, eval_cfg=cfg, mask=synth.tradable)
    pnl = result["group_pnl"]
    np.testing.assert_allclose(pnl["H-L"], pnl[1] - pnl[10])
    assert "annu_ret" in result["table"].columns
    ic = ic_test(signal, synth.ret_1d, horizon=5, mask=synth.tradable)
    assert "rank_ic_mean" in ic["summary"]


def test_folds_do_not_train_after_oos(monkeypatch):
    cal = pd.bdate_range("2018-01-01", periods=200)
    cfg = smoke_config()
    from dataclasses import replace

    cfg = replace(cfg, start=str(cal[0].date()), end=str(cal[-1].date()))
    folds = build_folds(cal, cfg)
    assert folds, "expected at least one fold after warmup"
    for fold in folds:
        assert_fold_no_leak(fold)
        assert fold.train_end <= fold.oos_start
        assert fold.train_start < fold.train_end
        assert fold.val_start >= fold.train_end


def test_variant_names():
    assert get_config("v1_adam").train.optimizer == "adam"
    assert get_config("v1_executable").train.execution == "executable_tplus1"
    assert get_config("v1").eval.fee_one_way == pytest.approx(0.00075)
    assert get_config("v1_repo_fee").eval.fee_one_way == pytest.approx(0.00075)
    with pytest.raises(KeyError):
        get_config("not_a_variant")
