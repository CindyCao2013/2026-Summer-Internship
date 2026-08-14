"""Named variants: v1 paper clone, v2/v3 structure+feature upgrades, smoke configs."""

from __future__ import annotations

from dataclasses import replace

from alphanet.config import (
    EXTRACT_OPS,
    FEATURE_NAMES_V2,
    N_FEATURES_V2,
    PAIR_MODE_UNIQUE,
    V3_EXTRACT_OPS,
    AlphaNetConfig,
    ModelConfig,
    TrainConfig,
    v1_config,
    v2_config,
    v3_config,
    variant_config,
)


VARIANT_HELP = {
    "v1": "Paper AlphaNet-V1 (RMSprop, 9x9 pairs, paper c2c labels).",
    "v1_adam": "V1 + Adam optimizer.",
    "v1_wd": "V1 + Adam + weight decay 1e-4.",
    "v1_unique_pairs": "V1 but C(F,2) pairs instead of F×F.",
    "v1_horizon5": "V1 with m=5 labels and 5-day rebalance.",
    "v1_executable": "V1 with T+1 executable labels.",
    "v1_attn": "V1 + channel attention + Adam.",
    "v1_nested": "V1 + nested unary extract + Adam.",
    "v1_repo_fee": "Alias of v1; fee already matches repo BASE 万分之 7.5 (7.5 bps).",
    "v2": "AlphaNet-V2: 15 features, LSTM, 4:1 split, Adam, all-A batch 2000.",
    "v2_csi800": "V2 on CSI800 constituents, batch 800.",
    "v2_csi500": "V2 on CSI500 constituents, batch 500.",
    "v3": "AlphaNet-V3: dual extract d=10/5, two GRUs, CSI500, batch 500.",
    "v3_alla": "V3 on all-A universe.",
    "smoke": "CPU smoke for V1 (short window, 1 seed).",
    "smoke_v2": "CPU smoke for V2.",
    "smoke_v3": "CPU smoke for V3.",
}


def smoke_config() -> AlphaNetConfig:
    """Fast CPU config: short window, one seed, tiny IS."""
    base = v1_config()
    model = replace(
        base.model,
        lookback=12,
        extract_d=4,
        extract_stride=4,
        pool_d=3,
        pool_stride=3,
        hidden_size=8,
        dropout=0.0,
        pair_mode="unique",
    )
    train = replace(
        base.train,
        optimizer="adam",
        lr=1e-3,
        batch_size=32,
        max_epochs=5,
        patience=3,
        n_seeds=1,
        in_sample_days=40,
        sample_every=2,
        retrain_months=1,
        horizon=3,
        label_cs_zscore=True,
        label_min_obs=8,
        execution="paper_c2c",
    )
    eval_cfg = replace(base.eval, rebalance_every=3, min_cs_obs=8)
    return replace(base, variant="smoke", model=model, train=train, eval=eval_cfg)


def smoke_v2_config() -> AlphaNetConfig:
    base = v2_config("ALLA")
    model = replace(
        base.model,
        lookback=12,
        extract_d=4,
        extract_stride=4,
        hidden_size=8,
        rnn_hidden=8,
        dropout=0.0,
        pair_mode=PAIR_MODE_UNIQUE,
        extract_ops=EXTRACT_OPS,
        n_features=N_FEATURES_V2,
        feature_names=FEATURE_NAMES_V2,
    )
    train = replace(
        base.train,
        batch_size=32,
        max_epochs=4,
        patience=3,
        n_seeds=1,
        in_sample_days=40,
        sample_every=2,
        retrain_months=1,
        horizon=3,
        train_frac=0.8,
        label_min_obs=8,
        lr=1e-3,
    )
    eval_cfg = replace(base.eval, rebalance_every=3, min_cs_obs=8)
    return replace(base, variant="smoke_v2", model=model, train=train, eval=eval_cfg)


def smoke_v3_config() -> AlphaNetConfig:
    base = v3_config("CSI500")
    model = replace(
        base.model,
        lookback=12,
        extract_d=4,
        extract_stride=4,
        extract2_d=2,
        extract2_stride=2,
        extract_ops=V3_EXTRACT_OPS,
        extract2_ops=V3_EXTRACT_OPS,
        rnn_hidden=8,
        dropout=0.0,
        pair_mode=PAIR_MODE_UNIQUE,
        n_features=N_FEATURES_V2,
        feature_names=FEATURE_NAMES_V2,
    )
    train = replace(
        base.train,
        batch_size=32,
        max_epochs=4,
        patience=3,
        n_seeds=1,
        in_sample_days=40,
        sample_every=2,
        retrain_months=1,
        horizon=3,
        train_frac=0.8,
        label_min_obs=8,
        lr=1e-3,
    )
    eval_cfg = replace(base.eval, rebalance_every=3, min_cs_obs=8)
    return replace(base, variant="smoke_v3", model=model, train=train, eval=eval_cfg)


def list_variants() -> dict:
    return dict(VARIANT_HELP)


def get_config(name: str) -> AlphaNetConfig:
    key = str(name).strip().lower()
    if key in ("smoke", "ci"):
        return smoke_config()
    if key == "smoke_v2":
        return smoke_v2_config()
    if key == "smoke_v3":
        return smoke_v3_config()
    return variant_config(name)
