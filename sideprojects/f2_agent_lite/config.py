"""F² Agent Lite config — Scheme A rotation + Scheme B cross-attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_ROOT / "results"


DEFAULT_SYMBOLS = [
    "600519.SH",
    "300750.SZ",
    "600036.SH",
    "601318.SH",
    "000858.SZ",
    "002594.SZ",
    "600900.SH",
    "000333.SZ",
    "601012.SH",
    "300059.SZ",
]


@dataclass
class Config:
    symbols: List[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    # Dynamic universe presets for sweep / CSI experiments
    universe_preset: str = "FIXED10"  # FIXED10 | CSI300 | CSI500 | CSI1000 | CUSTOM
    universe_index_code: Optional[str] = None  # override, e.g. "000300.SH"
    custom_symbols: List[str] = field(default_factory=list)
    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-06-30"
    lookback_window: int = 30
    preheat_calendar_days: int = 90
    val_ratio: float = 0.15

    # "A" = per-name multimodal agents + rotation
    # "B" = cross-sectional temporal + stock attention + rotation
    scheme: str = "B"

    # Model
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    news_max_titles: int = 3
    fetch_news_titles: bool = False
    tfidf_max_features: int = 2048
    n_classes: int = 3
    # Best after score smoothing grid: G4 (thr=0.015, rebal=7)
    label_threshold: float = 0.015
    pred_horizon: int = 5
    use_industry_embed: bool = True
    industry_neutral_rank: bool = True
    # Causal score smoothing (per-symbol rolling mean, then shift(1))
    score_smooth_window: int = 3

    # ---- Alpha feature switches (input dims expand; model Linear adapts) ----
    # Ablation 1: north + fundamentals only (advanced/market/vol_scaling OFF)
    use_north_money: bool = True       # 北向持仓占比 / 变化
    use_fundamentals: bool = True      # ROE / EP_TTM / 营收增速
    use_advanced_alpha: bool = False   # 动量 / Amihud / 残差波动
    use_market_risk: bool = False      # 大盘波动 + 涨跌停占比
    # ---- 分钟因子开关（ClickHouse KLIN 1MIN）----
    use_minute_factors: bool = True
    minute_factor_lookback: int = 10
    # Distributed 表跨节点可读；LOCAL_* 仅本机有分片时可用
    minute_use_local_tables: bool = False
    # Extra calendar preheat when advanced/fundamental features need longer history
    alpha_preheat_calendar_days: int = 180

    # ---- ClickHouse 连接（默认复用 COMMON_CONST.DATA_DB_HFDATA）----
    clickhouse_host: Optional[str] = None
    clickhouse_port: int = 8123
    clickhouse_user: Optional[str] = None
    clickhouse_password: Optional[str] = None
    clickhouse_database: str = "cmds"
    clickhouse_secure: bool = False

    # Volatility scaling in rotation backtester (confidence / realized vol)
    use_vol_scaling: bool = False
    vol_scaling_window: int = 20
    vol_scaling_floor: float = 0.01  # avoid div-by-zero; annualized-ish daily vol floor

    # Training
    epochs: int = 20
    cs_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 42
    num_workers: int = 0
    # Anti-collapse: upweight LONG + mix CE with pairwise ranking
    # Best so far: ret top/bottom ranking (not label-pairwise alone)
    long_class_boost: float = 3.0
    ce_loss_weight: float = 0.3
    rank_loss_weight: float = 1.0
    ranking_margin: float = 0.1
    use_label_pairwise: bool = False
    use_ret_topbottom: bool = True

    run_single_name_backtest: bool = False
    initial_cash: float = 1_000_000.0
    cost_rate: float = 0.001
    allow_short: bool = True
    long_only: bool = False

    run_rotation_backtest: bool = True
    rotation_top_frac: float = 0.2
    rotation_bottom_frac: float = 0.2
    rotation_top_k: Optional[int] = None
    rotation_bottom_k: Optional[int] = None
    rotation_long_gross: float = 0.5
    rotation_short_gross: float = 0.5
    # Match holding period to prediction horizon (1 = daily rebalance)
    rebalance_every: Optional[int] = 7  # G4 best with score smoothing

    use_data_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: RESULTS_DIR / "cache")
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "symbols": self.symbols,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "lookback_window": self.lookback_window,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
            "label_threshold": self.label_threshold,
            "pred_horizon": self.pred_horizon,
            "use_industry_embed": self.use_industry_embed,
            "industry_neutral_rank": self.industry_neutral_rank,
            "score_smooth_window": self.score_smooth_window,
            "use_north_money": self.use_north_money,
            "use_fundamentals": self.use_fundamentals,
            "use_advanced_alpha": self.use_advanced_alpha,
            "use_market_risk": self.use_market_risk,
            "use_minute_factors": self.use_minute_factors,
            "minute_factor_lookback": self.minute_factor_lookback,
            "use_vol_scaling": self.use_vol_scaling,
            "vol_scaling_window": self.vol_scaling_window,
            "epochs": self.epochs,
            "cs_epochs": self.cs_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "long_class_boost": self.long_class_boost,
            "ce_loss_weight": self.ce_loss_weight,
            "rank_loss_weight": self.rank_loss_weight,
            "ranking_margin": self.ranking_margin,
            "cost_rate": self.cost_rate,
            "run_rotation_backtest": self.run_rotation_backtest,
            "rotation_top_frac": self.rotation_top_frac,
            "rotation_bottom_frac": self.rotation_bottom_frac,
            "rebalance_every": self.rebalance_every if self.rebalance_every is not None else self.pred_horizon,
            "backend": "pytorch_cross_sectional_B" if self.scheme.upper() == "B" else "pytorch_rotation_A",
        }
