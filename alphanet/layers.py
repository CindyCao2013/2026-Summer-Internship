"""PyTorch custom layers mirroring ``operators.py``.

All extract layers map ``(B, F, T) -> (B, C, W)``. Pool layers map
``(B, C, W) -> (B, C, W')``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphanet.operators import EPS


def pair_index_tensors(
    n_features: int,
    pair_mode: str,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pair_mode == "full":
        ii = torch.arange(n_features, device=device).repeat_interleave(n_features)
        jj = torch.arange(n_features, device=device).repeat(n_features)
        return ii, jj
    if pair_mode == "unique":
        ii, jj = torch.triu_indices(n_features, n_features, offset=1, device=device)
        return ii, jj
    raise ValueError("unknown pair_mode {!r}".format(pair_mode))


def unfold_time(x: torch.Tensor, d: int, stride: int) -> torch.Tensor:
    """``(B, C, T) -> (B, C, W, d)``."""
    return x.unfold(dimension=-1, size=d, step=stride)


class PairCorrCov(nn.Module):
    def __init__(self, n_features: int, d: int, stride: int, pair_mode: str, kind: str):
        super().__init__()
        if kind not in ("corr", "cov"):
            raise ValueError(kind)
        self.d = int(d)
        self.stride = int(stride)
        self.pair_mode = pair_mode
        self.kind = kind
        self.n_features = int(n_features)
        ii, jj = pair_index_tensors(n_features, pair_mode)
        self.register_buffer("ii", ii, persistent=False)
        self.register_buffer("jj", jj, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = unfold_time(x, self.d, self.stride)
        left = windows[:, self.ii]
        right = windows[:, self.jj]
        left_c = left - left.mean(dim=-1, keepdim=True)
        right_c = right - right.mean(dim=-1, keepdim=True)
        cov = (left_c * right_c).mean(dim=-1)
        if self.kind == "cov":
            return cov
        std_l = left.std(dim=-1, unbiased=False)
        std_r = right.std(dim=-1, unbiased=False)
        denom = std_l * std_r
        corr = torch.where(denom.abs() < EPS, torch.zeros_like(cov), cov / denom.clamp_min(EPS))
        return torch.nan_to_num(corr, nan=0.0)


class UnaryWindowOp(nn.Module):
    def __init__(self, op: str, d: int, stride: int):
        super().__init__()
        if op not in (
            "ts_stddev",
            "ts_zscore",
            "ts_return",
            "ts_decaylinear",
            "ts_mean",
            "ts_max",
            "ts_min",
            "ts_sum",
        ):
            raise KeyError(op)
        self.op = op
        self.d = int(d)
        self.stride = int(stride)
        if op == "ts_decaylinear":
            weights = torch.arange(1, self.d + 1, dtype=torch.float32)
            self.register_buffer("weights", weights, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = unfold_time(x, self.d, self.stride)
        op = self.op
        if op == "ts_mean":
            return windows.mean(dim=-1)
        if op == "ts_sum":
            return windows.sum(dim=-1)
        if op == "ts_max":
            return windows.max(dim=-1).values
        if op == "ts_min":
            return windows.min(dim=-1).values
        if op == "ts_stddev":
            return windows.std(dim=-1, unbiased=False)
        if op == "ts_zscore":
            mean = windows.mean(dim=-1)
            std = windows.std(dim=-1, unbiased=False)
            last = windows[..., -1]
            z = (last - mean) / std.clamp_min(EPS)
            return torch.nan_to_num(z, nan=0.0)
        if op == "ts_return":
            first = windows[..., 0]
            last = windows[..., -1]
            out = torch.where(first.abs() < EPS, torch.zeros_like(last), last / first - 1.0)
            return torch.nan_to_num(out, nan=0.0)
        # ts_decaylinear
        w = self.weights.to(dtype=windows.dtype, device=windows.device)
        return (windows * w).sum(dim=-1) / w.sum()


class ChannelAttention(nn.Module):
    """Squeeze-excite over operator channels. Optimization variant only."""

    def __init__(self, channels: int, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W)
        squeeze = x.mean(dim=-1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * gate.unsqueeze(-1)


def build_extract_layer(op: str, n_features: int, d: int, stride: int, pair_mode: str) -> nn.Module:
    if op == "ts_corr":
        return PairCorrCov(n_features, d, stride, pair_mode, "corr")
    if op == "ts_cov":
        return PairCorrCov(n_features, d, stride, pair_mode, "cov")
    return UnaryWindowOp(op, d, stride)


def build_pool_layer(op: str, d: int, stride: int) -> nn.Module:
    return UnaryWindowOp(op, d, stride)
