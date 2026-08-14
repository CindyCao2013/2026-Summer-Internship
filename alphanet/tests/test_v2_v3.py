"""AlphaNet-v2 / v3: ratio features, LSTM/GRU, 4:1 split."""

from __future__ import annotations

import numpy as np
import pytest

from alphanet.config import (
    FEATURE_NAMES_V2,
    N_FEATURES_V2,
    V3_EXTRACT_OPS,
    n_windows,
    v2_config,
    v3_config,
)
from alphanet.dataset import split_train_val_dates
from alphanet.ratios import RATIO_NAMES, add_ratio_features, safe_div
from alphanet.synthetic import make_synthetic_panel, stack_images
from alphanet.variants import get_config, smoke_v2_config, smoke_v3_config


def test_v2_config_matches_guide():
    cfg = v2_config("ALLA")
    assert cfg.model.architecture == "v2"
    assert cfg.model.n_features == 15
    assert len(cfg.model.feature_names) == 15
    assert cfg.model.feature_names == FEATURE_NAMES_V2
    assert cfg.train.optimizer == "adam"
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.train.batch_size == 2000
    assert cfg.train.train_frac == pytest.approx(0.8)
    assert cfg.train.horizon == 10
    assert cfg.end == "2020-07-31"
    assert cfg.eval.fee_one_way == pytest.approx(0.00075)
    assert cfg.enhance.two_way_turnover_cap == pytest.approx(0.60)
    assert v2_config("CSI800").train.batch_size == 800
    assert v2_config("CSI500").train.batch_size == 500


def test_v3_config_matches_guide():
    cfg = v3_config()
    assert cfg.model.architecture == "v3"
    assert cfg.model.extract_ops == V3_EXTRACT_OPS
    assert cfg.model.extract2_d == 5
    assert cfg.model.extract2_stride == 5
    assert n_windows(30, 10, 10) == 3
    assert n_windows(30, 5, 5) == 6
    assert cfg.train.optimizer == "adam"
    assert cfg.train.batch_size == 500
    assert cfg.train.train_frac == pytest.approx(0.8)
    assert cfg.universe == "CSI500"


def test_ratio_features_formulas():
    synth = make_synthetic_panel(n_days=8, n_stocks=6, seed=9)
    ratios = add_ratio_features(synth.features)
    for name in RATIO_NAMES:
        assert name in ratios
    np.testing.assert_allclose(
        ratios["low_high"].to_numpy(),
        safe_div(synth.features["low"], synth.features["high"]).to_numpy(),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        ratios["vwap_close"].to_numpy(),
        safe_div(synth.features["vwap"], synth.features["close"]).to_numpy(),
        equal_nan=True,
    )


def test_stack_images_15x_lookback():
    synth = make_synthetic_panel(n_days=20, n_stocks=10, seed=1)
    images, symbols = stack_images(
        synth.features,
        synth.calendar[12],
        lookback=10,
        feature_names=FEATURE_NAMES_V2,
    )
    assert images.shape[1:] == (N_FEATURES_V2, 10)
    assert len(symbols) == images.shape[0]


def test_train_val_split_four_to_one():
    cal = pd_bdate()
    tr, va = split_train_val_dates(cal, every=1, start=cal[0], end=cal[-1], train_frac=0.8)
    assert len(tr) + len(va) == len(cal)
    assert abs(len(tr) / len(cal) - 0.8) < 0.05
    assert tr[-1] < va[0]


def pd_bdate():
    import pandas as pd

    return pd.bdate_range("2018-01-01", periods=50)


def test_v2_lstm_forward_timestep_3():
    torch = pytest.importorskip("torch")
    from alphanet.model import AlphaNetV2, build_model

    cfg = v2_config().model
    model = build_model(cfg)
    assert isinstance(model, AlphaNetV2)
    x = torch.randn(4, 15, 30)
    y = model(x)
    assert y.shape == (4, 1)
    maps = model.extract(x)
    seq_len = next(iter(maps.values())).shape[-1]
    assert seq_len == 3
    hidden = model.feature_tensor(x)
    assert hidden.shape == (4, cfg.rnn_hidden)
    y.sum().backward()


def test_v3_two_gru_timesteps():
    torch = pytest.importorskip("torch")
    from alphanet.model import AlphaNetV3, build_model

    cfg = v3_config().model
    model = build_model(cfg)
    assert isinstance(model, AlphaNetV3)
    x = torch.randn(3, 15, 30)
    y = model(x)
    assert y.shape == (3, 1)
    w10 = next(iter(model.extract10(x).values())).shape[-1]
    w5 = next(iter(model.extract5(x).values())).shape[-1]
    assert w10 == 3
    assert w5 == 6
    assert len(model.extract10.op_names) == 6
    assert "ts_mean" not in model.extract10.op_names
    hidden = model.feature_tensor(x)
    assert hidden.shape == (3, 60)
    y.sum().backward()


def test_smoke_v2_and_v3_one_step():
    torch = pytest.importorskip("torch")
    from alphanet.model import build_model

    for cfg in (smoke_v2_config(), smoke_v3_config()):
        model = build_model(cfg.model)
        x = torch.randn(8, cfg.model.n_features, cfg.model.lookback)
        pred = model(x)
        assert pred.shape == (8, 1)
        loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
        loss.backward()
        assert torch.isfinite(loss)


def test_get_config_v2_v3_aliases():
    assert get_config("v2").variant == "v2"
    assert get_config("v2_csi800").universe == "CSI800"
    assert get_config("v3").model.architecture == "v3"
    assert get_config("smoke_v2").variant == "smoke_v2"
    assert get_config("smoke_v3").model.extract2_d == 2
