"""Canonical ML dataset / execution / cost contracts for L2 AI Stock Selection v1.

These contracts inherit the frozen L2 reproduction conventions. They do not
silently change return labels, neutralization, or transaction-cost assumptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

from l2_factor_reproduction.l2_ai_stock_selection.model_contract import (
    model_contract_dict,
)

from l2_factor_reproduction.config.settings import UNIVERSE
from l2_factor_reproduction.python.candidate_pool_registry import (
    BASELINE_POLICY,
    POOL_ROOT,
)
from l2_factor_reproduction.python.evaluation_protocol_v2 import (
    FEE_BPS_PER_TRADED_NOTIONAL,
)
from l2_factor_reproduction.python.fast_discovery import FULL_END, FULL_START

AI_CONTRACT_VERSION = "l2_ai_stock_selection_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "research" / "results" / "l2_ai_stock_selection_v1"

# Inherit frozen L2 sample, do not invent a new window.
SAMPLE_START = FULL_START
SAMPLE_END = FULL_END
BENCHMARK_ID = UNIVERSE  # 000852.SH — excess-return benchmark, not a membership filter

# FS-3 already materializes 1/5/20. This system adds 3/10 without mutating FS-3 files.
CANONICAL_HORIZONS: Tuple[int, ...] = (1, 3, 5, 10, 20)
FS3_EXISTING_HORIZONS: Tuple[int, ...] = (1, 5, 20)
NEW_HORIZONS: Tuple[int, ...] = (3, 10)

LABEL_HORIZON_MAP: Dict[int, str] = {
    1: "forward_return_1d",
    3: "forward_return_3d",
    5: "forward_return_5d",
    10: "forward_return_10d",
    20: "forward_return_20d",
}

# Frozen-stack mapping (what the code actually does). Not an economic claim
# that Close[T] is a feasible execution timestamp.
EXECUTION_CONVENTION: Dict[str, object] = {
    "signal_availability": (
        "L2 daily factors are same-session aggregates; they are known only after "
        "the session close of TradeDate T. There is no uniform documented "
        "pre-close cutoff (14:30 / 14:55). Several families include 14:56–15:00 "
        "and/or the 15:00 close auction."
    ),
    "signal_shift": int(BASELINE_POLICY["signal_shift"]),  # 1
    "return_method": "c2c",
    "return_source": "WIND.ASHAREEODPRICES S_DQ_CLOSE/S_DQ_PRECLOSE-1",
    "excess": "stock_c2c - CSI1000_c2c (000852.SH)",
    "label_mapping": (
        "Traced indexes: after signal.shift(1), factor[T] is aligned with "
        "ret[T+1] = Close[T+1]/Close[T]-1. groupTest does not shift again. "
        "This is NOT executable at Close[T] if the factor is known only after Close[T]."
    ),
    "frozen_pairing": "factor_T -> Close[T+1]/Close[T]-1",
    "economically_executable": False,
    "timing_verdict": "C2C_TPLUS1_NOT_EXECUTABLE",
    "required_correction": (
        "Do not silently retain T+1 c2c as the AI-v1 production contract. "
        "Correct to (a) T+1 open/VWAP holding into a future return, or "
        "(b) shift the c2c window one extra session: factor T -> Close[T+1] to Close[T+2]."
    ),
    "frozen_historical_backtests": "UNCHANGED",
    "multi_day_method": (
        "stock_cum = prod(1+r_stock)-1; bench_cum = prod(1+r_bench)-1; "
        "excess = stock_cum - bench_cum"
    ),
    "not_used": "open-to-open / vwap-to-vwap are available in Factor_Dev_Lib but "
    "are NOT the L2 reproduction default.",
    "no_truncation": True,
}

TIMING_VERDICT = "C2C_TPLUS1_NOT_EXECUTABLE"
PRODUCTION_EXECUTION_CONTRACT = "EXEC_V2V_TPLUS1_V1"
PRODUCTION_LABEL_STATUS = "EXEC_V2V_TPLUS1_V1"
LEGACY_ARTIFACT_STATUS = "LEGACY_RESEARCH_BENCHMARK"

# Frozen Protocol v2.0 economics. Sensitivity scenarios are diagnostics only.
BASE_COST_BPS_PER_L1 = float(FEE_BPS_PER_TRADED_NOTIONAL)  # 7.5 bps per L1
COST_SCENARIOS_BPS: Dict[str, float] = {
    "LOW": 3.0,
    "BASE": BASE_COST_BPS_PER_L1,
    "HIGH": 15.0,
    "STRESS": 25.0,
}

PREPROCESS_CONTRACT_ID = "HUATAI_STYLE_IND_CAP_Z_V1"
REDUNDANCY_CORR_THRESHOLD = 0.80  # existing BDL / candidate-pool convention
UNIVERSE_MASK = "not_limit * not_st * trade_status (fast_context universe_mask)"

STYLE_CONTROLS: Tuple[str, ...] = (
    "industry",
    "size",
    "momentum_20d",
    "residual_volatility",
    "liquidity",
    "turnover_20d",
    "beta",
    "short_term_reversal",
)

# Economically meaningful horizon x rebalance pairs only.
REBALANCE_GRID: Dict[int, Tuple[int, ...]] = {
    1: (1,),
    3: (1, 3),
    5: (1, 3, 5),
    10: (3, 5, 10),
    20: (5, 10, 20),
}

FORBIDDEN_ML_COLUMNS: Tuple[str, ...] = (
    "label",
    "target",
    "y",
    "y1",
    "y5",
    "y20",
    "fwd_ret",
    "forward_return",
    "excess_return",
    "ret_fwd_1d",
    "ret_fwd_5d",
    "ret_fwd_20d",
)

IDENTIFIER_COLUMNS: Tuple[str, ...] = ("TradeDate", "Symbol")
UNIVERSE_METADATA_COLUMNS: Tuple[str, ...] = (
    "industry",
    "market_cap",
    "in_benchmark",
    "not_limit",
    "not_st",
    "trade_status",
    "tradable",
)

CANDIDATE_POOL_CSV = POOL_ROOT / "candidate_registry.csv"
FS1_INVENTORY_CSV = (
    Path(POOL_ROOT).parents[0]
    / "feature_selection"
    / "fs1_feature_panel_full"
    / "feature_inventory.csv"
)
FS3_LABEL_ROOT = (
    Path(POOL_ROOT).parents[0]
    / "feature_selection"
    / "fs3_walkforward_selection"
    / "labels"
)


def data_contract_dict() -> Dict[str, object]:
    return {
        "contract_version": AI_CONTRACT_VERSION,
        "grain": "TradeDate x Symbol",
        "identifiers": list(IDENTIFIER_COLUMNS),
        "universe_metadata": list(UNIVERSE_METADATA_COLUMNS),
        "benchmark_id": BENCHMARK_ID,
        "benchmark_role": "excess-return index; not an automatic membership filter",
        "sample_start": str(SAMPLE_START.date()),
        "sample_end": str(SAMPLE_END.date()),
        "execution": EXECUTION_CONVENTION,
        "timing_verdict": TIMING_VERDICT,
        "production_execution_contract": PRODUCTION_EXECUTION_CONTRACT,
        "production_label_status": PRODUCTION_LABEL_STATUS,
        "legacy_artifact_status": LEGACY_ARTIFACT_STATUS,
        "horizons": list(CANONICAL_HORIZONS),
        "label_columns": dict(LABEL_HORIZON_MAP),
        "preprocess_contract": PREPROCESS_CONTRACT_ID,
        "redundancy_corr_threshold": REDUNDANCY_CORR_THRESHOLD,
        "cost_base_bps_per_l1": BASE_COST_BPS_PER_L1,
        "cost_scenarios_bps": dict(COST_SCENARIOS_BPS),
        "rebalance_grid": {str(k): list(v) for k, v in REBALANCE_GRID.items()},
        "style_controls": list(STYLE_CONTROLS),
        "forbidden_feature_columns": list(FORBIDDEN_ML_COLUMNS),
        "model_layer": model_contract_dict(),
        "reuse": {
            "candidate_pool": str(CANDIDATE_POOL_CSV),
            "fs1_inventory": str(FS1_INVENTORY_CSV),
            "fs3_labels": str(FS3_LABEL_ROOT),
            "fast_context": "research/results/l2_reproduction/fast_context/full",
        },
    }


def classify_time_scale(lookback_days: int) -> str:
    """Map existing lookback metadata onto fast / mid / slow."""
    lb = int(lookback_days) if lookback_days == lookback_days else 1
    if lb <= 3:
        return "fast"
    if lb <= 10:
        return "mid"
    return "slow"


def allowed_rebalance_pairs() -> Tuple[Tuple[int, int], ...]:
    pairs = []
    for horizon, freqs in REBALANCE_GRID.items():
        for freq in freqs:
            pairs.append((horizon, freq))
    return tuple(pairs)


def cost_from_l1(avg_l1_turnover: float, scenario: str = "BASE") -> float:
    """Annualized implied cost = L1 * bps/1e4 * 250. Matches Factor_Dev_Lib."""
    if scenario not in COST_SCENARIOS_BPS:
        raise KeyError(f"unknown cost scenario {scenario!r}")
    bps = COST_SCENARIOS_BPS[scenario]
    return float(avg_l1_turnover) * float(bps) / 1e4 * 250.0


def assert_no_forbidden_feature_columns(
    columns: Union[Mapping[str, object], Tuple[str, ...]],
) -> None:
    names = tuple(columns) if not isinstance(columns, dict) else tuple(columns.keys())
    hit = [c for c in names if c in FORBIDDEN_ML_COLUMNS]
    if hit:
        raise ValueError(f"forbidden label/target columns in feature set: {hit}")
