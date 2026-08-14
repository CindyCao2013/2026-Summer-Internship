"""Human-readable names for flattened AlphaNet-V1 features (SHAP / logs)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from alphanet.config import (
    BINARY_OPS,
    FEATURE_NAMES,
    POOL_OPS,
    ModelConfig,
    n_windows,
)
from alphanet.operators import pair_indices


def pair_name(i: int, j: int, feature_names: Sequence[str]) -> str:
    return "{}, {}".format(feature_names[i], feature_names[j])


def extract_feature_names(cfg: ModelConfig, feature_names: Sequence[str] = None) -> List[str]:
    names_in = tuple(feature_names) if feature_names is not None else tuple(cfg.feature_names or FEATURE_NAMES)
    names: List[str] = []
    windows = n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride)
    ii, jj = pair_indices(cfg.n_features, cfg.pair_mode)
    for op in cfg.extract_ops:
        if op in BINARY_OPS:
            channels = [
                "{}({})".format(op, pair_name(int(i), int(j), names_in))
                for i, j in zip(ii, jj)
            ]
        else:
            channels = ["{}({})".format(op, names_in[k]) for k in range(cfg.n_features)]
        for c in channels:
            for w in range(windows):
                names.append("{}[w{}]".format(c, w))
    return names


def pool_feature_names(cfg: ModelConfig, feature_names: Sequence[str] = None) -> List[str]:
    names_in = tuple(feature_names) if feature_names is not None else tuple(cfg.feature_names or FEATURE_NAMES)
    names: List[str] = []
    extract_windows = n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride)
    pool_windows = n_windows(extract_windows, cfg.pool_d, cfg.pool_stride)
    ii, jj = pair_indices(cfg.n_features, cfg.pair_mode)
    for op in cfg.extract_ops:
        if op in BINARY_OPS:
            channels = [
                "{}({})".format(op, pair_name(int(i), int(j), names_in))
                for i, j in zip(ii, jj)
            ]
        else:
            channels = ["{}({})".format(op, names_in[k]) for k in range(cfg.n_features)]
        for pool_op in POOL_OPS:
            for c in channels:
                for w in range(pool_windows):
                    names.append("{}({}, {})[w{}]".format(pool_op, c, cfg.pool_d, w))
    return names


def all_feature_names(cfg: ModelConfig, feature_names: Sequence[str] = None) -> List[str]:
    if str(cfg.architecture).lower() == "v2":
        return ["lstm_h{}".format(i) for i in range(int(cfg.rnn_hidden))]
    if str(cfg.architecture).lower() == "v3":
        h = int(cfg.rnn_hidden)
        return ["gru_d10_h{}".format(i) for i in range(h)] + ["gru_d5_h{}".format(i) for i in range(h)]
    return extract_feature_names(cfg, feature_names) + pool_feature_names(cfg, feature_names)


def assert_names_match_dim(cfg: ModelConfig) -> Tuple[int, int]:
    from alphanet.config import total_flat_dim

    names = all_feature_names(cfg)
    dim = total_flat_dim(cfg)
    if len(names) != dim:
        raise AssertionError("name count {} != flat dim {}".format(len(names), dim))
    return len(names), dim
