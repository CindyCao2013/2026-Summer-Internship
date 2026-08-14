"""Operator reference tests (numpy, no GPU / DDB)."""

from __future__ import annotations

import numpy as np
import pytest

from alphanet.config import ModelConfig, n_pairs, n_windows, total_flat_dim, v1_config
from alphanet.features import assert_names_match_dim
from alphanet.operators import (
    decay_weights,
    frame_windows,
    ts_corr,
    ts_cov,
    ts_decaylinear,
    ts_max,
    ts_mean,
    ts_min,
    ts_return,
    ts_std,
    ts_sum,
    ts_zscore,
)


def test_frame_windows_nonoverlap():
    x = np.arange(12, dtype=np.float64).reshape(1, 1, 12)
    w = frame_windows(x, d=4, stride=4)
    assert w.shape == (1, 1, 3, 4)
    np.testing.assert_array_equal(w[0, 0, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(w[0, 0, 2], [8, 9, 10, 11])


def test_unary_ops_known_values():
    x = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype=np.float64)
    np.testing.assert_allclose(ts_mean(x, 3, 3)[0, 0], [2.0, 5.0])
    np.testing.assert_allclose(ts_sum(x, 3, 3)[0, 0], [6.0, 15.0])
    np.testing.assert_allclose(ts_max(x, 3, 3)[0, 0], [3.0, 6.0])
    np.testing.assert_allclose(ts_min(x, 3, 3)[0, 0], [1.0, 4.0])
    np.testing.assert_allclose(ts_std(x, 3, 3)[0, 0], [np.std([1, 2, 3]), np.std([4, 5, 6])])


def test_zscore_is_last_in_window():
    x = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    z = ts_zscore(x, 3, 3)
    expected = (3.0 - 2.0) / np.std([1.0, 2.0, 3.0])
    np.testing.assert_allclose(z[0, 0, 0], expected, rtol=1e-5)


def test_return_last_over_first():
    x = np.array([[[10.0, 11.0, 12.0]]], dtype=np.float64)
    r = ts_return(x, 3, 3)
    np.testing.assert_allclose(r[0, 0, 0], 12.0 / 10.0 - 1.0)


def test_decaylinear_newest_gets_weight_d():
    x = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)
    w = decay_weights(3)
    expected = (1 * 1 + 2 * 2 + 3 * 3) / w.sum()
    np.testing.assert_allclose(ts_decaylinear(x, 3, 3)[0, 0, 0], expected)


def test_self_corr_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 20))
    corr = ts_corr(x, d=10, stride=10, pair_mode="full")
    # diagonal pairs 0-0, 1-1, 2-2 at flat index 0, 4, 8
    for f in range(3):
        np.testing.assert_allclose(corr[:, f * 3 + f], 1.0, atol=1e-5)


def test_corr_symmetric_full_pairs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 4, 20))
    corr = ts_corr(x, 10, 10, pair_mode="full")[0]
    for i in range(4):
        for j in range(4):
            np.testing.assert_allclose(corr[i * 4 + j], corr[j * 4 + i], atol=1e-5)


def test_cov_matches_numpy():
    x = np.array([[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]], dtype=np.float64)
    cov = ts_cov(x, 4, 4, pair_mode="full")
    expected = np.mean((np.array([1, 2, 3, 4]) - 2.5) * (np.array([4, 3, 2, 1]) - 2.5))
    np.testing.assert_allclose(cov[0, 1, 0], expected)


def test_v1_flat_dim_and_names():
    cfg = ModelConfig()
    assert n_windows(30, 10, 10) == 3
    assert n_pairs(9, "full") == 81
    assert total_flat_dim(cfg) == 1404
    n_names, dim = assert_names_match_dim(cfg)
    assert n_names == dim == 1404


def test_unique_pairs_count():
    assert n_pairs(9, "unique") == 36
    cfg = ModelConfig(pair_mode="unique")
    assert_names_match_dim(cfg)


def test_v1_config_fee_is_7p5_bps():
    cfg = v1_config()
    assert cfg.eval.fee_one_way == pytest.approx(0.00075)
    assert cfg.train.horizon == 10
    assert cfg.model.hidden_size == 30
    assert cfg.model.dropout == 0.5
    assert cfg.train.optimizer == "rmsprop"
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.train.n_seeds == 10
    assert cfg.eval.n_groups == 10
    assert cfg.eval.g1_is_top is True
