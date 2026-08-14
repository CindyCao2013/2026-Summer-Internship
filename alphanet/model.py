"""AlphaNet-V1 / V2 / V3 networks."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from alphanet.config import (
    BINARY_OPS,
    POOL_OPS,
    UNARY_OPS,
    ModelConfig,
    n_pairs,
    n_windows,
    total_flat_dim,
)
from alphanet.layers import ChannelAttention, build_extract_layer, build_pool_layer


def _trunc_normal_(module: nn.Module, std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ExtractBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, ops: Optional[Sequence[str]] = None, d: Optional[int] = None, stride: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.op_names = tuple(ops if ops is not None else cfg.extract_ops)
        self.d = int(cfg.extract_d if d is None else d)
        self.stride = int(cfg.extract_stride if stride is None else stride)
        self.ops = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        pairs = n_pairs(cfg.n_features, cfg.pair_mode)
        for op in self.op_names:
            self.ops[op] = build_extract_layer(
                op, cfg.n_features, self.d, self.stride, cfg.pair_mode
            )
            channels = pairs if op in BINARY_OPS else cfg.n_features
            self.bns[op] = nn.BatchNorm1d(channels)
        self._windows = n_windows(cfg.lookback, self.d, self.stride)
        self._pairs = pairs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        for op in self.op_names:
            feat = self.ops[op](x)
            out[op] = self.bns[op](feat)
        return out

    def channel_count(self, op: str) -> int:
        return self._pairs if op in BINARY_OPS else self.cfg.n_features

    def total_channels(self) -> int:
        return sum(self.channel_count(op) for op in self.op_names)


def maps_to_sequence(extracted: Dict[str, torch.Tensor], op_names: Sequence[str]) -> torch.Tensor:
    """``(B, C_i, W)`` maps → ``(B, W, sum C_i)`` for LSTM/GRU."""
    parts = [extracted[op] for op in op_names]
    cat = torch.cat(parts, dim=1)
    return cat.transpose(1, 2)


class PoolBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, extract: ExtractBlock):
        super().__init__()
        self.cfg = cfg
        self.pools = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        for op in extract.op_names:
            channels = extract.channel_count(op)
            for pool_op in POOL_OPS:
                key = "{}_{}".format(op, pool_op)
                self.pools[key] = build_pool_layer(pool_op, cfg.pool_d, cfg.pool_stride)
                self.bns[key] = nn.BatchNorm1d(channels)

    def forward(self, extracted: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for op, feat in extracted.items():
            for pool_op in POOL_OPS:
                key = "{}_{}".format(op, pool_op)
                pooled = self.pools[key](feat)
                out[key] = self.bns[key](pooled)
        return out


class NestedExtract(nn.Module):
    """Second extract pass on concatenated unary maps. Optimization only."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        n_unary = len(UNARY_OPS) * cfg.n_features
        inner = ModelConfig(
            n_features=n_unary,
            lookback=n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride),
            extract_d=min(3, n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride)),
            extract_stride=min(3, n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride)),
            pool_d=1,
            pool_stride=1,
            pair_mode="unique",
        )
        if inner.lookback < 2:
            inner = ModelConfig(
                n_features=n_unary,
                lookback=max(inner.lookback, 3),
                extract_d=1,
                extract_stride=1,
                pair_mode="unique",
            )
        self.inner_cfg = inner
        self.mean = build_extract_layer(
            "ts_mean", n_unary, inner.extract_d, inner.extract_stride, inner.pair_mode
        )
        self.bn = nn.BatchNorm1d(n_unary)

    def forward(self, extracted: Dict[str, torch.Tensor]) -> torch.Tensor:
        unary = torch.cat([extracted[op] for op in UNARY_OPS], dim=1)
        return self.bn(self.mean(unary))


class AlphaNetV1(nn.Module):
    """Paper V1: extract + pool + flatten + Dense(30, ReLU) + Dropout + Dense(1)."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.extract = ExtractBlock(self.cfg)
        self.pool = PoolBlock(self.cfg, self.extract)
        self.attention: Optional[nn.Module] = None
        extra = 0
        if self.cfg.use_attention:
            total_c = self.extract.total_channels()
            self.attention = ChannelAttention(total_c)
        self.nested: Optional[NestedExtract] = None
        if self.cfg.nested_extract:
            self.nested = NestedExtract(self.cfg)
            extra = self.nested.inner_cfg.n_features * n_windows(
                self.nested.inner_cfg.lookback,
                self.nested.inner_cfg.extract_d,
                self.nested.inner_cfg.extract_stride,
            )
        self.flat_dim = total_flat_dim(self.cfg) + extra
        self.hidden = nn.Linear(self.flat_dim, self.cfg.hidden_size)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.out = nn.Linear(self.cfg.hidden_size, 1)
        self.apply(lambda m: _trunc_normal_(m, self.cfg.trunc_normal_std))

    def _flatten_maps(self, maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for feat in maps.values():
            parts.append(feat.reshape(feat.shape[0], -1))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        extracted = self.extract(x)
        if self.attention is not None:
            concat = torch.cat([extracted[op] for op in self.extract.op_names], dim=1)
            gated = self.attention(concat)
            offset = 0
            new_extracted = {}
            for op in self.extract.op_names:
                c = self.extract.channel_count(op)
                new_extracted[op] = gated[:, offset : offset + c]
                offset += c
            extracted = new_extracted
        pooled = self.pool(extracted)
        flat_extract = self._flatten_maps(extracted)
        flat_pool = self._flatten_maps(pooled)
        pieces = [flat_extract, flat_pool]
        if self.nested is not None:
            nested = self.nested(extracted)
            pieces.append(nested.reshape(nested.shape[0], -1))
        features = torch.cat(pieces, dim=-1)
        if features.shape[-1] != self.hidden.in_features:
            # nested / attention variants may differ; project if needed
            if not hasattr(self, "_adapt"):
                raise RuntimeError(
                    "flat dim {} != Linear in_features {}".format(
                        features.shape[-1], self.hidden.in_features
                    )
                )
        hidden = self.dropout(torch.relu(self.hidden(features)))
        pred = self.out(hidden)
        if return_features:
            return pred, features
        return pred

    def feature_tensor(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x, return_features=True)
        return features


class AlphaNetV2(nn.Module):
    """15-d input, 10 extract ops, LSTM(time_step=3, hidden=30) + BN + Dense(1)."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig(architecture="v2")
        self.extract = ExtractBlock(self.cfg)
        self.lstm = nn.LSTM(
            input_size=self.extract.total_channels(),
            hidden_size=int(self.cfg.rnn_hidden),
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(int(self.cfg.rnn_hidden))
        self.out = nn.Linear(int(self.cfg.rnn_hidden), 1)
        self.apply(lambda m: _trunc_normal_(m, self.cfg.trunc_normal_std))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        maps = self.extract(x)
        seq = maps_to_sequence(maps, self.extract.op_names)
        packed, _ = self.lstm(seq)
        hidden = self.bn(packed[:, -1, :])
        pred = self.out(hidden)
        if return_features:
            return pred, hidden
        return pred

    def feature_tensor(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x, return_features=True)
        return features


class AlphaNetV3(nn.Module):
    """Dual extract (d=10 and d=5, 6 ops each) + two GRUs, concat → Dense(1)."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig(architecture="v3")
        ops = tuple(self.cfg.extract2_ops)
        self.extract10 = ExtractBlock(
            self.cfg, ops=ops, d=self.cfg.extract_d, stride=self.cfg.extract_stride
        )
        self.extract5 = ExtractBlock(
            self.cfg, ops=ops, d=self.cfg.extract2_d, stride=self.cfg.extract2_stride
        )
        h = int(self.cfg.rnn_hidden)
        self.gru10 = nn.GRU(self.extract10.total_channels(), h, batch_first=True)
        self.gru5 = nn.GRU(self.extract5.total_channels(), h, batch_first=True)
        self.bn10 = nn.BatchNorm1d(h)
        self.bn5 = nn.BatchNorm1d(h)
        self.out = nn.Linear(2 * h, 1)
        self.apply(lambda m: _trunc_normal_(m, self.cfg.trunc_normal_std))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        s10 = maps_to_sequence(self.extract10(x), self.extract10.op_names)
        s5 = maps_to_sequence(self.extract5(x), self.extract5.op_names)
        h10 = self.bn10(self.gru10(s10)[0][:, -1, :])
        h5 = self.bn5(self.gru5(s5)[0][:, -1, :])
        hidden = torch.cat([h10, h5], dim=-1)
        pred = self.out(hidden)
        if return_features:
            return pred, hidden
        return pred

    def feature_tensor(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.forward(x, return_features=True)
        return features


def build_model(cfg: Optional[ModelConfig] = None) -> nn.Module:
    cfg = cfg or ModelConfig()
    arch = str(cfg.architecture).lower()
    if arch == "v2":
        return AlphaNetV2(cfg)
    if arch == "v3":
        return AlphaNetV3(cfg)
    return AlphaNetV1(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
