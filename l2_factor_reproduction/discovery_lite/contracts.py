"""Frozen Batch Discovery Lite v1 contract.

Thresholds are engineering defaults. They must not be tuned from dry-run
results. Lite metrics are named ``*_lite`` and never overwrite Full Discovery
metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.python.fast_discovery import DISCOVERY_END, DISCOVERY_START

CONTRACT_VERSION = "BDL_V1"

LITE_START = DISCOVERY_START  # 2023-01-01 — same Fast Discovery window
LITE_END = DISCOVERY_END  # 2024-12-31
DATE_STRIDE = 5  # every 5th valid trading date; not optimized

COVERAGE_THRESHOLD = 0.50
CONSTANT_DATE_THRESHOLD = 0.80
NONFINITE_RATIO_THRESHOLD = 0.01
MIN_IC_DATES = 20

RANK_IC_THRESHOLD = 0.008
ICIR_THRESHOLD = 1.5

REDUNDANCY_CORR_THRESHOLD = 0.80
NEAR_ALIAS_THRESHOLD = 0.90
NOVELTY_HIGH = 0.50
NOVELTY_MEDIUM = 0.75

DECILE_MONO_THRESHOLD = 0.50
SPREAD_SIGN_CONSISTENCY_THRESHOLD = 0.55
N_DECILES = 10
MIN_CROSS_SECTION = 100
MAX_CLUSTER_REPRESENTATIVES = 2

SIGNAL_SHIFT = 1  # canonical T+1; reused from prepare_factor_signal
ICIR_ANNUALIZATION = 250  # same daily RankIC annualization as backtest.py

PRIORITY_WEIGHTS: Dict[str, float] = {
    "signal": 0.35,
    "stability": 0.20,
    "shape": 0.15,
    "coverage": 0.10,
    "novelty": 0.20,
}

REGISTRY_REQUIRED = (
    "name",
    "family",
    "formula",
    "mechanism",
    "lookback_days",
    "signed",
    "positive_value_meaning",
    "primitive_dependencies",
    "registry_status",
)
REGISTRY_OPTIONAL = (
    "category",
    "expected_redundancy",
    "normalization",
    "notes",
    "replacement_candidate",
    "near_alias_exception",
    "sparse_event",
)

# Dry-run universe frozen before execution (mechanism diversity, not performance).
DRY_RUN_EXISTING_FACTORS: Tuple[str, ...] = (
    # Trade Flow
    "net_buy_ratio",
    "net_buy_count_ratio",
    "buy_dominance",
    "avg_buy_trade_size",
    "flow_concentration",
    "flow_zscore_20d",
    # Order Size
    "small_order_ratio_1w",
    "large_order_ratio_20w",
    "order_size_entropy",
    "large_small_spread",
    "small_order_pressure",
    "large_order_direction",
    "large_order_shock_20d",
    # Order Book
    "obi_l1_mean",
    "obi_l5_mean",
    "weighted_obi_mean",
    "relative_spread_mean",
    "microprice_deviation_mean",
    "bid_depth_concentration",
    "obi_shock_20d",
    # Price Formation
    "close_auction_return",
    "overnight_gap",
    "realized_volatility",
    "path_efficiency",
    "closing_amount_share",
    "volume_abs_return_corr",
    "intraday_amihud",
    # Liquidity Impact
    "spread_per_depth",
    "depth_per_amount",
    "signed_amount_impact",
    "effective_spread_proxy",
    "large_trade_impact",
    "spread_recovery_5m",
    "amount_to_depth",
)

# Novelty reference: frozen pool factors NOT in the dry-run candidate list.
DRY_RUN_NOVELTY_REFERENCE: Tuple[str, ...] = (
    "trade_size_asymmetry",
    "flow_acceleration",
    "mid_order_ratio_4w_20w",
    "small_order_ratio_4w",
    "order_size_concentration",
    "obi_l10_mean",
    "relative_spread_volatility",
    "total_depth_level",
    "morning_return",
    "jump_share",
    "closing_30m_return",
    "signed_sqrt_amount_impact",
    "permanent_impact_1m",
    "depth_turnover",
)

OUTPUT_ROOT = Path(RESULT_ROOT) / "discovery_lite"

BDL_CONTRACT: Dict[str, Any] = {
    "contract_version": CONTRACT_VERSION,
    "lite_start": str(LITE_START.date()),
    "lite_end": str(LITE_END.date()),
    "date_stride": DATE_STRIDE,
    "coverage_threshold": COVERAGE_THRESHOLD,
    "constant_date_threshold": CONSTANT_DATE_THRESHOLD,
    "nonfinite_ratio_threshold": NONFINITE_RATIO_THRESHOLD,
    "min_ic_dates": MIN_IC_DATES,
    "rank_ic_threshold": RANK_IC_THRESHOLD,
    "icir_threshold": ICIR_THRESHOLD,
    "redundancy_corr_threshold": REDUNDANCY_CORR_THRESHOLD,
    "near_alias_threshold": NEAR_ALIAS_THRESHOLD,
    "novelty_high": NOVELTY_HIGH,
    "novelty_medium": NOVELTY_MEDIUM,
    "decile_mono_threshold": DECILE_MONO_THRESHOLD,
    "spread_sign_consistency_threshold": SPREAD_SIGN_CONSISTENCY_THRESHOLD,
    "n_deciles": N_DECILES,
    "min_cross_section": MIN_CROSS_SECTION,
    "max_cluster_representatives": MAX_CLUSTER_REPRESENTATIVES,
    "target_definition": (
        "Fast Discovery canonical: excess c2c vs UNIVERSE, "
        "prepare_factor_signal(signal_shift=1), tradability mask "
        "(not_limit * not_st * trade_status). Lite metrics use every "
        f"{DATE_STRIDE}th trading date only."
    ),
    "universe_definition": UNIVERSE,
    "signal_shift": SIGNAL_SHIFT,
    "icir_annualization": ICIR_ANNUALIZATION,
    "priority_weights": dict(PRIORITY_WEIGHTS),
    "dry_run_existing_factors": list(DRY_RUN_EXISTING_FACTORS),
    "dry_run_novelty_reference": list(DRY_RUN_NOVELTY_REFERENCE),
}


def lite_trading_dates(
    trading_dates: Sequence[pd.Timestamp],
    *,
    start: pd.Timestamp = LITE_START,
    end: pd.Timestamp = LITE_END,
    stride: int = DATE_STRIDE,
) -> pd.DatetimeIndex:
    """Deterministic every-Nth trading-date sample. Not performance-based."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    cal = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).normalize()
    cal = cal[(cal >= pd.Timestamp(start).normalize()) & (cal <= pd.Timestamp(end).normalize())]
    cal = cal.sort_values().unique()
    return pd.DatetimeIndex(cal[::stride], name="TradeDate")


def contract_to_json(path: Path, extra: Mapping[str, Any] | None = None) -> None:
    payload = dict(BDL_CONTRACT)
    if extra:
        payload.update(dict(extra))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
