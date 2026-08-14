"""Frozen AlphaNet-V1/V2/V3 hyperparameters and the modified evaluation protocol.

Papers:
  HTSC *AlphaNet: 因子挖掘神经网络* (2020-06-14) — v1
  HTSC *再探 AlphaNet：结构和特征优化* (2020-08-24) — v2 / v3
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple

from alphanet.paths import RESULT_ROOT
from alphanet.ratios import RATIO_NAMES

FEATURE_NAMES: Tuple[str, ...] = (
    "return1",
    "open",
    "close",
    "high",
    "low",
    "vwap",
    "volume",
    "turn",
    "free_turn",
)
FEATURE_NAMES_V2: Tuple[str, ...] = FEATURE_NAMES + RATIO_NAMES
N_FEATURES = len(FEATURE_NAMES)
N_FEATURES_V2 = len(FEATURE_NAMES_V2)

BINARY_OPS: Tuple[str, ...] = ("ts_corr", "ts_cov")
UNARY_OPS: Tuple[str, ...] = (
    "ts_stddev",
    "ts_zscore",
    "ts_return",
    "ts_decaylinear",
    "ts_mean",
    "ts_max",
    "ts_min",
    "ts_sum",
)
EXTRACT_OPS: Tuple[str, ...] = BINARY_OPS + UNARY_OPS
# v3 keeps the six operators that survive ablation in the 2020-08 paper.
V3_EXTRACT_OPS: Tuple[str, ...] = (
    "ts_corr",
    "ts_cov",
    "ts_stddev",
    "ts_zscore",
    "ts_return",
    "ts_decaylinear",
)
POOL_OPS: Tuple[str, ...] = ("ts_mean", "ts_max", "ts_min")

PAIR_MODE_FULL = "full"
PAIR_MODE_UNIQUE = "unique"

PAPER_START = "2011-01-31"
PAPER_END = "2020-05-29"
PAPER_END_V23 = "2020-07-31"
PREHEAT_CALENDAR_DAYS = 2500

CSI500 = "000905.SH"
CSI800 = "000906.SH"
ANNUALIZATION_DAYS = 250
MAX_WORKERS = 10

# One-way cost: 万分之 7.5 = 7.5 bps = 0.075%. Same as repo BASE / L2 groupTest.
# Do not use 7.5‰ (per mille): that is 0.75% and 10x too large.
FEE_ONE_WAY = 0.00075
REPO_BASE_FEE_ONE_WAY = 0.00075

INDUSTRY_PREFERRED = "SW_L1"
INDUSTRY_FALLBACK = "CITICS_L1"

V2_BATCH = {"ALLA": 2000, "CSI800": 800, "CSI500": 500}


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "v1"
    n_features: int = N_FEATURES
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    lookback: int = 30
    extract_d: int = 10
    extract_stride: int = 10
    extract_ops: Tuple[str, ...] = EXTRACT_OPS
    extract2_d: int = 5
    extract2_stride: int = 5
    extract2_ops: Tuple[str, ...] = V3_EXTRACT_OPS
    pool_d: int = 3
    pool_stride: int = 3
    hidden_size: int = 30
    dropout: float = 0.5
    rnn_hidden: int = 30
    pair_mode: str = PAIR_MODE_FULL
    trunc_normal_std: float = 0.05
    use_attention: bool = False
    nested_extract: bool = False
    extra_feature_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "rmsprop"
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1000
    max_epochs: int = 200
    patience: int = 10
    n_seeds: int = 10
    seed0: int = 42
    in_sample_days: int = 1500
    sample_every: int = 3
    retrain_months: int = 6
    horizon: int = 10
    train_frac: float = 0.5
    label_cs_zscore: bool = True
    label_min_obs: int = 30
    execution: str = "paper_c2c"


@dataclass(frozen=True)
class EvalConfig:
    n_groups: int = 10
    fee_one_way: float = FEE_ONE_WAY
    excess_vs: str = "universe_ew"
    neutralize: bool = True
    rebalance_every: int = 10
    g1_is_top: bool = True
    flip_hl_for_display: bool = False
    min_cs_obs: int = 30


@dataclass(frozen=True)
class EnhanceConfig:
    benchmark: str = CSI500
    max_weight: float = 0.05
    two_way_turnover_cap: float = 0.30
    active_weight_caps: Tuple[float, ...] = (0.005, 0.01, 0.015, 0.02)
    industry_neutral: bool = True
    size_neutral: bool = True
    long_only: bool = True
    fee_one_way: float = FEE_ONE_WAY


@dataclass(frozen=True)
class AlphaNetConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    start: str = PAPER_START
    end: str = PAPER_END
    variant: str = "v1"
    universe: str = "ALLA"
    result_root: str = str(RESULT_ROOT)


def n_windows(length: int, d: int, stride: int) -> int:
    if length < d or d <= 0 or stride <= 0:
        return 0
    return (length - d) // stride + 1


def n_pairs(n_features: int, pair_mode: str) -> int:
    if pair_mode == PAIR_MODE_UNIQUE:
        return n_features * (n_features - 1) // 2
    if pair_mode == PAIR_MODE_FULL:
        return n_features * n_features
    raise ValueError("unknown pair_mode {!r}".format(pair_mode))


def extract_channel_count(cfg: ModelConfig, ops: Tuple[str, ...] = None) -> int:
    names = ops if ops is not None else cfg.extract_ops
    pairs = n_pairs(cfg.n_features, cfg.pair_mode)
    total = 0
    for op in names:
        total += pairs if op in BINARY_OPS else cfg.n_features
    return total


def extract_flat_dim(cfg: ModelConfig) -> int:
    windows = n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride)
    return extract_channel_count(cfg, cfg.extract_ops) * windows


def pool_flat_dim(cfg: ModelConfig) -> int:
    windows = n_windows(cfg.lookback, cfg.extract_d, cfg.extract_stride)
    pool_windows = n_windows(windows, cfg.pool_d, cfg.pool_stride)
    return len(POOL_OPS) * extract_channel_count(cfg, cfg.extract_ops) * pool_windows


def total_flat_dim(cfg: ModelConfig) -> int:
    return extract_flat_dim(cfg) + pool_flat_dim(cfg)


def v1_config() -> AlphaNetConfig:
    return AlphaNetConfig(variant="v1")


def v2_config(universe: str = "ALLA") -> AlphaNetConfig:
    key = str(universe).upper()
    if key not in V2_BATCH:
        raise KeyError("unknown v2 universe {!r}".format(universe))
    bench = CSI800 if key == "CSI800" else CSI500
    variant = "v2" if key == "ALLA" else "v2_{}".format(key.lower())
    return AlphaNetConfig(
        variant=variant,
        universe=key,
        start=PAPER_START,
        end=PAPER_END_V23,
        model=ModelConfig(
            architecture="v2",
            n_features=N_FEATURES_V2,
            feature_names=FEATURE_NAMES_V2,
            extract_ops=EXTRACT_OPS,
            pair_mode=PAIR_MODE_UNIQUE,
            dropout=0.0,
            rnn_hidden=30,
        ),
        train=TrainConfig(
            optimizer="adam",
            lr=1e-4,
            batch_size=int(V2_BATCH[key]),
            train_frac=0.8,
            horizon=10,
        ),
        enhance=EnhanceConfig(
            benchmark=bench,
            two_way_turnover_cap=0.60,
        ),
    )


def v3_config(universe: str = "CSI500") -> AlphaNetConfig:
    key = str(universe).upper()
    bench = CSI800 if key == "CSI800" else CSI500
    variant = "v3" if key == "CSI500" else "v3_{}".format(key.lower())
    return AlphaNetConfig(
        variant=variant,
        universe=key,
        start=PAPER_START,
        end=PAPER_END_V23,
        model=ModelConfig(
            architecture="v3",
            n_features=N_FEATURES_V2,
            feature_names=FEATURE_NAMES_V2,
            extract_ops=V3_EXTRACT_OPS,
            extract2_ops=V3_EXTRACT_OPS,
            extract2_d=5,
            extract2_stride=5,
            pair_mode=PAIR_MODE_UNIQUE,
            dropout=0.0,
            rnn_hidden=30,
        ),
        train=TrainConfig(
            optimizer="adam",
            lr=1e-4,
            batch_size=500,
            train_frac=0.8,
            horizon=10,
        ),
        enhance=EnhanceConfig(
            benchmark=bench,
            two_way_turnover_cap=0.60,
        ),
    )


def variant_config(name: str) -> AlphaNetConfig:
    """Named optimization variants from guide §11. ``v1`` is the paper clone."""
    base = v1_config()
    key = str(name).strip().lower()
    if key in ("v1", "paper", "alphanet_v1"):
        return base
    if key == "v1_adam":
        return replace(
            base,
            variant="v1_adam",
            train=replace(base.train, optimizer="adam"),
        )
    if key == "v1_wd":
        return replace(
            base,
            variant="v1_wd",
            train=replace(base.train, optimizer="adam", weight_decay=1e-4),
        )
    if key == "v1_unique_pairs":
        return replace(
            base,
            variant="v1_unique_pairs",
            model=replace(base.model, pair_mode=PAIR_MODE_UNIQUE),
        )
    if key == "v1_horizon5":
        return replace(
            base,
            variant="v1_horizon5",
            train=replace(base.train, horizon=5),
            eval=replace(base.eval, rebalance_every=5),
        )
    if key == "v1_executable":
        return replace(
            base,
            variant="v1_executable",
            train=replace(base.train, execution="executable_tplus1"),
        )
    if key == "v1_attn":
        return replace(
            base,
            variant="v1_attn",
            model=replace(base.model, use_attention=True),
            train=replace(base.train, optimizer="adam"),
        )
    if key == "v1_nested":
        return replace(
            base,
            variant="v1_nested",
            model=replace(base.model, nested_extract=True),
            train=replace(base.train, optimizer="adam"),
        )
    if key == "v1_repo_fee":
        return replace(
            base,
            variant="v1_repo_fee",
            eval=replace(base.eval, fee_one_way=REPO_BASE_FEE_ONE_WAY),
            enhance=replace(base.enhance, fee_one_way=REPO_BASE_FEE_ONE_WAY),
        )
    if key in ("v2", "alphanet_v2"):
        return v2_config("ALLA")
    if key == "v2_csi800":
        return v2_config("CSI800")
    if key == "v2_csi500":
        return v2_config("CSI500")
    if key in ("v3", "alphanet_v3", "v3_csi500"):
        return v3_config("CSI500")
    if key == "v3_alla":
        return v3_config("ALLA")
    raise KeyError("unknown variant {!r}".format(name))
