"""Torch layer / model shape tests. Skipped if torch is missing."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alphanet.config import ModelConfig, total_flat_dim
from alphanet.layers import build_extract_layer
from alphanet.model import build_model, count_parameters
from alphanet.operators import apply_extract_op
from alphanet.variants import smoke_config


def test_torch_mean_matches_numpy():
    rng = np.random.default_rng(0)
    x_np = rng.normal(size=(4, 9, 30)).astype(np.float32)
    ref = apply_extract_op(x_np, "ts_mean", 10, 10)
    layer = build_extract_layer("ts_mean", 9, 10, 10, "full")
    out = layer(torch.from_numpy(x_np)).detach().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_torch_corr_matches_numpy():
    rng = np.random.default_rng(2)
    x_np = rng.normal(size=(3, 6, 20)).astype(np.float32)
    ref = apply_extract_op(x_np, "ts_corr", 10, 10, pair_mode="full")
    layer = build_extract_layer("ts_corr", 6, 10, 10, "full")
    out = layer(torch.from_numpy(x_np)).detach().numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_v1_forward_shape_and_flat_dim():
    cfg = ModelConfig()
    model = build_model(cfg)
    x = torch.randn(8, 9, 30)
    y = model(x)
    assert y.shape == (8, 1)
    feat = model.feature_tensor(x)
    assert feat.shape == (8, total_flat_dim(cfg))
    assert count_parameters(model) > 0


def test_smoke_model_trains_one_step():
    cfg = smoke_config()
    model = build_model(cfg.model)
    x = torch.randn(16, cfg.model.n_features, cfg.model.lookback)
    y = torch.randn(16, 1)
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad.abs().sum() for p in model.parameters() if p.grad is not None]
    assert any(g > 0 for g in grads)


def test_trunc_normal_not_exploding():
    model = build_model(ModelConfig())
    w = model.hidden.weight.detach()
    assert float(w.abs().max()) < 0.5
    assert float(w.std()) < 0.2
