"""Frozen contracts for Temporal / State Cutting Operators v1.

Sidecar feature-engineering module for L2 AI Stock Selection v1.
Does not replace candidate_pool_v1, tree models, or AlphaNet.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from l2_factor_reproduction.l2_ai_stock_selection.contracts import RESULT_ROOT
from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    PRIMARY_EXECUTION_CONTRACT,
)
from l2_factor_reproduction.python.candidate_pool_registry import POOL_ROOT

CUT_MODULE_ID = "TEMPORAL_STATE_CUTTING_OPERATORS_V1"
CUT_CONTRACT_VERSION = "cut_operators_v1"
CUT_RESULT_ROOT = RESULT_ROOT / "cut_operators"
CANDIDATE_POOL_CSV = POOL_ROOT / "candidate_registry.csv"

# Sidecar only. Never write generated candidates into candidate_pool_v1.
CUT_REGISTRY_NAME = "cut_candidate_registry.csv"
MAX_WORKERS = 10
MAX_DESCENDANTS_PER_PARENT = 10
PREFERRED_DESCENDANTS_PER_PARENT = (3, 6)
EVENT_Q_DEFAULT = 0.20
RATIO_EPSILON = 1e-12
MIN_COVERAGE_OBS = 5
MIN_SLOPE_OBS = 3
MIN_PERSISTENCE_PAIRS = 3
NEAR_DUPLICATE_CORR = 0.98
UNCONTROLLED_CARTESIAN_CAP = 50

PRODUCTION_EXECUTION_CONTRACT = PRIMARY_EXECUTION_CONTRACT
LEGACY_C2C_FLAG = "LEGACY_C2C_DIAGNOSTIC"

# Continuous-auction mkeys, matching liquidity_impact / cancel_lifecycle:
# AM 09:30-11:29 = 570-689; PM 13:00-14:59 = 780-899; 240 bars.
AM_MKEY_START = 9 * 60 + 30  # 570
AM_MKEY_END = 11 * 60 + 29  # 689
PM_MKEY_START = 13 * 60  # 780
PM_MKEY_END = 14 * 60 + 59  # 899
AUCTION_MKEY = 15 * 60  # 900
EXPECTED_CONTINUOUS_MINUTES = 240
LUNCH_MKEY_START = 11 * 60 + 30  # 690
LUNCH_MKEY_END = 13 * 60 - 1  # 779

# Half-open wall-clock [start, end). Last continuous bar of CLOSE is 14:59.
TIME_SEGMENTS: Tuple[Dict[str, object], ...] = (
    {
        "segment_name": "OPEN",
        "start_time": "09:30:00",
        "end_time": "10:00:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": AM_MKEY_START,
        "mkey_end": 10 * 60 - 1,  # 599 = 09:59
        "n_expected_minutes": 30,
        "contains_close_auction": False,
        "contains_1456_1500": False,
        "source_compatibility": "all_continuous_sources",
        "availability_timestamp": "after_1000_T",
        "uses_last_5min": False,
        "note": "Opening 30m continuous auction. Lunch not involved.",
    },
    {
        "segment_name": "MORNING",
        "start_time": "10:00:00",
        "end_time": "11:30:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": 10 * 60,  # 600
        "mkey_end": AM_MKEY_END,  # 689 = 11:29
        "n_expected_minutes": 90,
        "contains_close_auction": False,
        "contains_1456_1500": False,
        "source_compatibility": "all_continuous_sources",
        "availability_timestamp": "after_1130_T",
        "uses_last_5min": False,
        "note": "Remainder of morning continuous session after OPEN.",
    },
    {
        "segment_name": "AFTERNOON",
        "start_time": "13:00:00",
        "end_time": "14:30:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": PM_MKEY_START,  # 780
        "mkey_end": 14 * 60 + 29,  # 869 = 14:29
        "n_expected_minutes": 90,
        "contains_close_auction": False,
        "contains_1456_1500": False,
        "source_compatibility": "all_continuous_sources",
        "availability_timestamp": "after_1430_T",
        "uses_last_5min": False,
        "note": "Lunch 11:30-13:00 is excluded by construction.",
    },
    {
        "segment_name": "CLOSE",
        "start_time": "14:30:00",
        "end_time": "15:00:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": 14 * 60 + 30,  # 870
        "mkey_end": PM_MKEY_END,  # 899 = 14:59
        "n_expected_minutes": 30,
        "contains_close_auction": False,
        "contains_1456_1500": True,
        "source_compatibility": "continuous_sources; DDB omits observed 14:57-14:59",
        "availability_timestamp": "after_continuous_close_T",
        "uses_last_5min": True,
        "note": (
            "Continuous close 14:30-14:59. Does NOT include the 15:00 auction. "
            "Includes 14:56-14:59 where the source actually stores those bars."
        ),
    },
    {
        "segment_name": "COMMON_CLOSE",
        "start_time": "14:30:00",
        "end_time": "14:57:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": 14 * 60 + 30,  # 870
        "mkey_end": 14 * 60 + 56,  # 896 = 14:56
        "n_expected_minutes": 27,
        "contains_close_auction": False,
        "contains_1456_1500": True,
        "source_compatibility": "intersection_ddb_and_ch; not a tuned parameter",
        "availability_timestamp": "after_continuous_close_T",
        "uses_last_5min": True,
        "note": (
            "Robustness window ending at the latest reliably common minute "
            "(14:56). DDB omits observed 14:57-14:59. Do not optimize this cut."
        ),
        "optional_v1": True,
        "robustness_only": True,
    },
    {
        "segment_name": "FULL",
        "start_time": "09:30:00",
        "end_time": "15:00:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": AM_MKEY_START,
        "mkey_end": PM_MKEY_END,
        "n_expected_minutes": EXPECTED_CONTINUOUS_MINUTES,
        "contains_close_auction": False,
        "contains_1456_1500": True,
        "source_compatibility": "all_continuous_sources",
        "availability_timestamp": "after_continuous_close_T",
        "uses_last_5min": True,
        "note": "Union of AM+PM continuous bars. Lunch excluded. Auction excluded.",
    },
    {
        "segment_name": "EARLY_CLOSE",
        "start_time": "14:30:00",
        "end_time": "14:50:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": 14 * 60 + 30,  # 870
        "mkey_end": 14 * 60 + 49,  # 889
        "n_expected_minutes": 20,
        "contains_close_auction": False,
        "contains_1456_1500": False,
        "source_compatibility": "reliable_on_ch_and_ddb",
        "availability_timestamp": "after_1450_T",
        "uses_last_5min": False,
        "note": "Enabled only when source timestamps are reliable in 14:30-14:49.",
        "optional_v1": True,
    },
    {
        "segment_name": "LATE_CLOSE",
        "start_time": "14:50:00",
        "end_time": "15:00:00",
        "start_inclusive": True,
        "end_inclusive": False,
        "mkey_start": 14 * 60 + 50,  # 890
        "mkey_end": PM_MKEY_END,  # 899
        "n_expected_minutes": 10,
        "contains_close_auction": False,
        "contains_1456_1500": True,
        "source_compatibility": "PARTIAL_DDB; READY_CH",
        "availability_timestamp": "after_continuous_close_T",
        "uses_last_5min": True,
        "note": (
            "DDB Stock_one_minute structurally omits 14:57-14:59 and may "
            "impute price state from 14:56. Do not treat LATE_CLOSE as "
            "reliable on DDB without an explicit source flag."
        ),
        "optional_v1": True,
    },
    {
        "segment_name": "CLOSE_AUCTION",
        "start_time": "15:00:00",
        "end_time": "15:00:00",
        "start_inclusive": True,
        "end_inclusive": True,
        "mkey_start": AUCTION_MKEY,
        "mkey_end": AUCTION_MKEY,
        "n_expected_minutes": 1,
        "contains_close_auction": True,
        "contains_1456_1500": True,
        "source_compatibility": "ddb_15:00; ch_order_book_hour15; ch_tick_15:00:00; NOT lid/cancel",
        "availability_timestamp": "after_close_auction_T",
        "uses_last_5min": True,
        "note": "Never silently folded into CLOSE. Source must opt in.",
        "optional_v1": True,
    },
)

STATE_DEFINITIONS: Tuple[Dict[str, object], ...] = (
    {
        "state_name": "high_vol",
        "family": "VOLATILITY",
        "condition_primitive": "abs_minute_return",
        "rule": "abs(minute_return) > stock_day_median(abs(minute_return))",
        "threshold_scope": "within_stock_day",
        "complement": "low_vol",
    },
    {
        "state_name": "low_vol",
        "family": "VOLATILITY",
        "condition_primitive": "abs_minute_return",
        "rule": "abs(minute_return) <= stock_day_median(abs(minute_return))",
        "threshold_scope": "within_stock_day",
        "complement": "high_vol",
    },
    {
        "state_name": "high_spread",
        "family": "SPREAD",
        "condition_primitive": "relative_spread",
        "rule": "spread_t > stock_day_median(spread)",
        "threshold_scope": "within_stock_day",
        "complement": "low_spread",
    },
    {
        "state_name": "low_spread",
        "family": "SPREAD",
        "condition_primitive": "relative_spread",
        "rule": "spread_t <= stock_day_median(spread)",
        "threshold_scope": "within_stock_day",
        "complement": "high_spread",
    },
    {
        "state_name": "high_depth",
        "family": "DEPTH",
        "condition_primitive": "total_depth_l5",
        "rule": "depth_t > stock_day_median(depth)",
        "threshold_scope": "within_stock_day",
        "complement": "low_depth",
    },
    {
        "state_name": "low_depth",
        "family": "DEPTH",
        "condition_primitive": "total_depth_l5",
        "rule": "depth_t <= stock_day_median(depth)",
        "threshold_scope": "within_stock_day",
        "complement": "high_depth",
    },
    {
        "state_name": "price_up",
        "family": "PRICE_DIRECTION",
        "condition_primitive": "minute_return",
        "rule": "minute_return_t > 0",
        "threshold_scope": "within_stock_day",
        "complement": "price_down",
    },
    {
        "state_name": "price_down",
        "family": "PRICE_DIRECTION",
        "condition_primitive": "minute_return",
        "rule": "minute_return_t < 0",
        "threshold_scope": "within_stock_day",
        "complement": "price_up",
    },
    {
        "state_name": "high_trade_intensity",
        "family": "TRADE_INTENSITY",
        "condition_primitive": "amount",
        "rule": "amount_t > stock_day_median(amount)",
        "threshold_scope": "within_stock_day",
        "complement": "low_trade_intensity",
    },
    {
        "state_name": "low_trade_intensity",
        "family": "TRADE_INTENSITY",
        "condition_primitive": "amount",
        "rule": "amount_t <= stock_day_median(amount)",
        "threshold_scope": "within_stock_day",
        "complement": "high_trade_intensity",
    },
    {
        "state_name": "large_order_dominated",
        "family": "LARGE_ORDER",
        "condition_primitive": "large_order_amount",
        "rule": "large_order_share_t > stock_day_median(large_order_share); share = top20%_notional / minute_notional",
        "threshold_scope": "within_stock_day",
        "complement": "ordinary_order_state",
    },
    {
        "state_name": "ordinary_order_state",
        "family": "LARGE_ORDER",
        "condition_primitive": "large_order_amount",
        "rule": "large_order_share_t <= stock_day_median(large_order_share)",
        "threshold_scope": "within_stock_day",
        "complement": "large_order_dominated",
    },
)

EVENT_DEFINITIONS: Tuple[Dict[str, object], ...] = (
    {
        "event_name": "TOP_Q",
        "rule": "X in top 20% of finite observations within the stock-day",
        "q": EVENT_Q_DEFAULT,
        "threshold_scope": "within_stock_day",
        "grid_search": False,
    },
    {
        "event_name": "BOTTOM_Q",
        "rule": "X in bottom 20% of finite observations within the stock-day",
        "q": EVENT_Q_DEFAULT,
        "threshold_scope": "within_stock_day",
        "grid_search": False,
    },
    {
        "event_name": "EXTREME",
        "rule": "abs(within-day zscore(X)) > 2.0",
        "z_threshold": 2.0,
        "threshold_scope": "within_stock_day",
        "grid_search": False,
    },
    {
        "event_name": "SHOCK",
        "rule": (
            "X_t > expanding_mean(X_{<t} same stock-day) "
            "+ 2 * expanding_std(X_{<t}); causal prior-intraday baseline"
        ),
        "z_threshold": 2.0,
        "threshold_scope": "causal_intraday",
        "grid_search": False,
    },
    {
        "event_name": "LARGE_TRADE_EVENT",
        "rule": (
            "minute large_order_share in the top 20% of the same stock-day; "
            "large prints are trades above the stock-day 80th percentile of trade size"
        ),
        "q": EVENT_Q_DEFAULT,
        "threshold_scope": "within_stock_day",
        "grid_search": False,
    },
    {
        "event_name": "LIQUIDITY_SHOCK",
        "rule": (
            "spread in top 20% OR depth in bottom 20% OR "
            "abs(impact) in top 20%, within stock-day"
        ),
        "q": EVENT_Q_DEFAULT,
        "threshold_scope": "within_stock_day",
        "grid_search": False,
    },
)

AGGREGATORS: Tuple[str, ...] = (
    "mean",
    "sum",
    "std",
    "median",
    "last",
    "max",
    "min",
    "persistence",
    "slope",
    "event_share",
    "temporal_center",
    "tc_plus",
    "tc_minus",
    "temporal_gap",
    "temporal_dispersion",
)

# Only operators that make economic sense for the primitive class.
PRIMITIVE_CLASS_AGGREGATORS: Dict[str, Tuple[str, ...]] = {
    "flow_amount": (
        "sum",
        "mean",
        "last",
        "persistence",
        "slope",
        "event_share",
        "temporal_center",
        "tc_plus",
        "tc_minus",
        "temporal_gap",
        "temporal_dispersion",
    ),
    "imbalance_ratio": (
        "mean",
        "median",
        "last",
        "persistence",
        "slope",
        "std",
        "temporal_center",
        "tc_plus",
        "tc_minus",
        "temporal_gap",
        "temporal_dispersion",
    ),
    "spread_level": ("mean", "median", "last", "max", "std", "temporal_center", "temporal_dispersion"),
    "depth_level": ("mean", "median", "last", "min", "std", "temporal_center", "temporal_dispersion"),
    "impact": (
        "mean",
        "sum",
        "median",
        "last",
        "std",
        "temporal_center",
        "tc_plus",
        "tc_minus",
        "temporal_gap",
        "temporal_dispersion",
    ),
    "return_path": (
        "sum",
        "mean",
        "last",
        "persistence",
        "slope",
        "std",
        "temporal_center",
        "tc_plus",
        "tc_minus",
        "temporal_gap",
        "temporal_dispersion",
    ),
    "intensity": (
        "sum",
        "mean",
        "median",
        "max",
        "event_share",
        "temporal_center",
        "tc_plus",
        "tc_minus",
        "temporal_gap",
        "temporal_dispersion",
    ),
    "cancel_flow": ("sum", "mean", "last", "persistence", "slope"),
}

CONTRAST_OPERATORS: Tuple[str, ...] = (
    "DIFF",
    "RATIO",
    "NORMALIZED_DIFF",
    "SHARE",
    "ACCELERATION",
    "REVERSAL",
    "PERSISTENCE_CONTRAST",
)

MODES: Tuple[str, ...] = ("GENERAL_RESEARCH", "NONLINEAR_RESCUE")

RESCUE_CLASSES: Tuple[str, ...] = (
    "RESCUED_CORE",
    "RESCUED_AUXILIARY",
    "NONLINEAR_ONLY",
    "FAILED_RESCUE",
    "REDUNDANT_RESCUE",
)

# TC-2A report label. Same class as RESCUED_AUXILIARY.
RESCUED_AUX_LABEL = "RESCUED_AUX"

TIMING_LOCALIZATION_STATUSES: Tuple[str, ...] = (
    "TIMING_LOCALIZED_EXECUTABLE",
    "TIMING_LOCALIZED_TOO_FAST",
    "NO_CLEAR_TIMING_STRUCTURE",
)

PARENT_TYPE_LABELS: Tuple[str, ...] = (
    "LEVEL_PARENT",
    "PATH_PARENT",
    "DERIVED_TRANSFORM_PARENT",
)

RESCUE_CORE_GATES: Dict[str, float] = {
    "abs_rank_ic": 0.02,
    "hl_sharpe": 3.0,
    "monotonicity": 0.70,
}

REGISTRY_COLUMNS: Tuple[str, ...] = (
    "candidate_name",
    "base_primitive",
    "base_family",
    "cut_type",
    "cut_definition",
    "condition_primitive",
    "aggregation",
    "contrast_operator",
    "cut_start_time",
    "cut_end_time",
    "availability_timestamp",
    "contains_close_auction",
    "contains_1456_1500",
    "latest_source_timestamp",
    "factor_available_after",
    "uses_close_auction",
    "uses_last_5min",
    "execution_contract_compatible",
    "production_execution_compatible",
    "economic_interpretation",
    "parent_factor_if_rescue",
    "generation_reason",
    "status",
)

PARENT_CHILD_METRIC_COLUMNS: Tuple[str, ...] = (
    "parent_rank_ic",
    "child_rank_ic",
    "delta_abs_rank_ic",
    "parent_hl_sharpe",
    "child_hl_sharpe",
    "delta_hl_sharpe",
    "parent_monotonicity",
    "child_monotonicity",
    "delta_monotonicity",
    "parent_mi",
    "child_mi",
    "parent_residual_ic",
    "child_residual_ic",
    "correlation_parent_child",
    "correlation_to_existing_core",
)

# Economically justified X | Z pairs. Not an X × Z product.
CONDITIONAL_WHITELIST: Tuple[Dict[str, str], ...] = (
    {
        "target": "net_active_flow",
        "condition": "high_spread",
        "reason": "OFI/active flow is more informative when the quote is wide.",
    },
    {
        "target": "obi_5",
        "condition": "low_depth",
        "reason": "Displayed imbalance matters more in a thin book.",
    },
    {
        "target": "obi_5",
        "condition": "high_spread",
        "reason": "OBI predictive content is state-dependent on tightness.",
    },
    {
        "target": "signed_amount_impact",
        "condition": "large_order_dominated",
        "reason": "Impact during large-trade minutes is a different mechanism.",
    },
    {
        "target": "cancel_imbalance",
        "condition": "high_vol",
        "reason": "Cancel pressure around volatility shocks, not calm minutes.",
    },
    {
        "target": "net_active_flow",
        "condition": "price_down",
        "reason": "Active flow into falling prices vs chasing rising prices.",
    },
    {
        "target": "relative_spread",
        "condition": "high_vol",
        "reason": "Spread widening under volatility vs ordinary tightness.",
    },
    {
        "target": "minute_return",
        "condition": "high_trade_intensity",
        "reason": "Price path during active minutes vs idle minutes.",
    },
    {
        "target": "large_order_amount",
        "condition": "price_down",
        "reason": "Large prints into falling prices are a distinct flow state.",
    },
)

P0_PRIMITIVE_WHITELIST: Tuple[Dict[str, object], ...] = (
    {
        "base_primitive": "net_active_flow",
        "base_family": "trade_flow",
        "primitive_class": "flow_amount",
        "sequence_grain": "minute",
        "preferred_source": "ddb_stock_one_minute",
        "fallback_source": "ch_tick",
        "daily_cache_cuttable": False,
        "tc1": True,
    },
    {
        "base_primitive": "obi_5",
        "base_family": "order_book",
        "primitive_class": "imbalance_ratio",
        "sequence_grain": "minute_last_snapshot",
        "preferred_source": "ch_ssl2",
        "fallback_source": None,
        "daily_cache_cuttable": False,
        "tc1": True,
    },
    {
        "base_primitive": "large_order_amount",
        "base_family": "order_size",
        "primitive_class": "intensity",
        "sequence_grain": "minute_from_tick",
        "preferred_source": "ch_tick",
        "fallback_source": None,
        "daily_cache_cuttable": False,
        "tc1": True,
    },
    {
        "base_primitive": "minute_return",
        "base_family": "price_formation",
        "primitive_class": "return_path",
        "sequence_grain": "minute",
        "preferred_source": "ddb_stock_one_minute",
        "fallback_source": None,
        "daily_cache_cuttable": False,
        "tc1": True,
    },
    {
        "base_primitive": "relative_spread",
        "base_family": "liquidity",
        "primitive_class": "spread_level",
        "sequence_grain": "minute_last_snapshot",
        "preferred_source": "ch_ssl2",
        "fallback_source": None,
        "daily_cache_cuttable": False,
        "tc1": True,
    },
    {
        "base_primitive": "cancel_imbalance",
        "base_family": "cancel_lifecycle",
        "primitive_class": "cancel_flow",
        "sequence_grain": "minute",
        "preferred_source": "ddb_stock_one_minute_cancel_proxy",
        "fallback_source": "ch_tick_cancel",
        "daily_cache_cuttable": False,
        "tc1": True,
    },
)

# Compact TC-1 recipe book. Explicit rows, not a Cartesian product.
TC1_RECIPES: Tuple[Dict[str, object], ...] = (
    # trade_flow
    {
        "base_primitive": "net_active_flow",
        "cut_type": "time",
        "cut_name": "close",
        "aggregation": "sum",
        "reason": "Late-session aggressive flow vs the hidden daily net.",
    },
    {
        "base_primitive": "net_active_flow",
        "cut_type": "state",
        "cut_name": "high_spread",
        "aggregation": "sum",
        "condition_primitive": "relative_spread",
        "reason": "Active flow when the quote is wide.",
    },
    {
        "base_primitive": "net_active_flow",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "DIFF",
        "aggregation": "sum",
        "reason": "Path: morning buy vs close sell (or the reverse).",
    },
    {
        "base_primitive": "net_active_flow",
        "cut_type": "contrast",
        "cut_name": "close_share_full",
        "contrast_operator": "SHARE",
        "aggregation": "sum",
        "reason": "How much of daily signed flow arrives in the close.",
    },
    {
        "base_primitive": "net_active_flow",
        "cut_type": "event",
        "cut_name": "top_q",
        "aggregation": "sum",
        "reason": "Flow concentrated in the strongest OFI minutes.",
    },
    # order_book
    {
        "base_primitive": "obi_5",
        "cut_type": "time",
        "cut_name": "close",
        "aggregation": "mean",
        "reason": "Closing book imbalance, not the all-day mean.",
    },
    {
        "base_primitive": "obi_5",
        "cut_type": "state",
        "cut_name": "high_spread",
        "aggregation": "mean",
        "condition_primitive": "relative_spread",
        "reason": "OBI conditional on a wide spread.",
    },
    {
        "base_primitive": "obi_5",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "DIFF",
        "aggregation": "mean",
        "reason": "Book refill / reversal from open to close.",
    },
    {
        "base_primitive": "obi_5",
        "cut_type": "contrast",
        "cut_name": "highvol_minus_lowvol",
        "contrast_operator": "DIFF",
        "aggregation": "mean",
        "reason": "State-conditional OBI: high-vol vs low-vol minutes.",
    },
    {
        "base_primitive": "obi_5",
        "cut_type": "state",
        "cut_name": "low_depth",
        "aggregation": "mean",
        "condition_primitive": "total_depth_l5",
        "reason": "OBI when displayed depth is thin.",
    },
    # order_size
    {
        "base_primitive": "large_order_amount",
        "cut_type": "time",
        "cut_name": "close",
        "aggregation": "sum",
        "reason": "Large-order activity concentrated into the close.",
    },
    {
        "base_primitive": "large_order_amount",
        "cut_type": "state",
        "cut_name": "price_down",
        "aggregation": "sum",
        "condition_primitive": "minute_return",
        "reason": "Large prints into falling prices.",
    },
    {
        "base_primitive": "large_order_amount",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "NORMALIZED_DIFF",
        "aggregation": "sum",
        "reason": "Late vs early large-order share, scale-free (NORMALIZED_DIFF).",
    },
    {
        "base_primitive": "large_order_amount",
        "cut_type": "event",
        "cut_name": "large_trade",
        "aggregation": "event_share",
        "reason": "Share of minutes that are large-trade dominated.",
    },
    {
        "base_primitive": "large_order_amount",
        "cut_type": "contrast",
        "cut_name": "close_share_full",
        "contrast_operator": "SHARE",
        "aggregation": "sum",
        "reason": "Close share of daily large-order notional (SHARE of relative-large prints).",
    },
    # price_formation
    {
        "base_primitive": "minute_return",
        "cut_type": "time",
        "cut_name": "close",
        "aggregation": "sum",
        "reason": "Late-session price path, distinct from daily OC return.",
    },
    {
        "base_primitive": "minute_return",
        "cut_type": "state",
        "cut_name": "high_trade_intensity",
        "aggregation": "sum",
        "condition_primitive": "amount",
        "reason": "Returns during active minutes only.",
    },
    {
        "base_primitive": "minute_return",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "DIFF",
        "aggregation": "sum",
        "reason": "Open vs close return acceleration / reversal.",
    },
    {
        "base_primitive": "minute_return",
        "cut_type": "contrast",
        "cut_name": "reversal_close_vs_open",
        "contrast_operator": "REVERSAL",
        "aggregation": "sum",
        "reason": "Sign-aware early vs late reversal.",
    },
    {
        "base_primitive": "minute_return",
        "cut_type": "time",
        "cut_name": "open",
        "aggregation": "persistence",
        "reason": "Opening-path persistence, hidden by daily mean return.",
    },
    # liquidity
    {
        "base_primitive": "relative_spread",
        "cut_type": "time",
        "cut_name": "close",
        "aggregation": "mean",
        "reason": "Closing tightness vs the all-day mean spread.",
    },
    {
        "base_primitive": "relative_spread",
        "cut_type": "state",
        "cut_name": "high_vol",
        "aggregation": "mean",
        "condition_primitive": "abs_minute_return",
        "reason": "Spread in high-volatility minutes.",
    },
    {
        "base_primitive": "relative_spread",
        "cut_type": "contrast",
        "cut_name": "highvol_over_full",
        "contrast_operator": "RATIO",
        "aggregation": "mean",
        "reason": "High-vol spread relative to the full-day spread.",
    },
    {
        "base_primitive": "relative_spread",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "DIFF",
        "aggregation": "mean",
        "reason": "Intraday tightening or widening path.",
    },
    {
        "base_primitive": "relative_spread",
        "cut_type": "event",
        "cut_name": "liquidity_shock",
        "aggregation": "mean",
        "reason": "Spread during liquidity-shock minutes.",
    },
    # cancel
    {
        "base_primitive": "cancel_imbalance",
        "cut_type": "time",
        "cut_name": "close",
        "aggregation": "sum",
        "reason": "Late-session cancel pressure.",
    },
    {
        "base_primitive": "cancel_imbalance",
        "cut_type": "state",
        "cut_name": "high_vol",
        "aggregation": "sum",
        "condition_primitive": "abs_minute_return",
        "reason": "Cancels concentrated in volatile minutes.",
    },
    {
        "base_primitive": "cancel_imbalance",
        "cut_type": "contrast",
        "cut_name": "close_minus_open",
        "contrast_operator": "DIFF",
        "aggregation": "sum",
        "reason": "Cancel-pressure path from open to close.",
    },
    {
        "base_primitive": "cancel_imbalance",
        "cut_type": "contrast",
        "cut_name": "close_share_full",
        "contrast_operator": "SHARE",
        "aggregation": "sum",
        "reason": "Close share of signed cancel pressure.",
    },
    {
        "base_primitive": "cancel_imbalance",
        "cut_type": "event",
        "cut_name": "top_q",
        "aggregation": "sum",
        "reason": "Cancel imbalance in the most extreme cancel minutes.",
    },
    # extra economically justified variants to reach ~36, still << Cartesian
    {
        "base_primitive": "net_active_flow",
        "cut_type": "time",
        "cut_name": "afternoon",
        "aggregation": "sum",
        "reason": "Afternoon flow after lunch, distinct from morning.",
    },
    {
        "base_primitive": "obi_5",
        "cut_type": "time",
        "cut_name": "open",
        "aggregation": "mean",
        "reason": "Opening book, already a known economic segment.",
    },
    {
        "base_primitive": "minute_return",
        "cut_type": "time",
        "cut_name": "afternoon",
        "aggregation": "sum",
        "reason": "Afternoon price discovery after the lunch gap.",
    },
    {
        "base_primitive": "relative_spread",
        "cut_type": "time",
        "cut_name": "open",
        "aggregation": "mean",
        "reason": "Opening tightness; often widest quotes of the day.",
    },
    {
        "base_primitive": "cancel_imbalance",
        "cut_type": "time",
        "cut_name": "open",
        "aggregation": "sum",
        "reason": "Opening cancel/replace activity.",
    },
    {
        "base_primitive": "large_order_amount",
        "cut_type": "time",
        "cut_name": "open",
        "aggregation": "sum",
        "reason": "Opening large-print intensity.",
    },
)

SOURCE_TIMESTAMP_SEMANTICS: Tuple[Dict[str, object], ...] = (
    {
        "source_id": "ch_tick",
        "families": ("trade_flow", "order_size"),
        "session_window": "09:30:00 <= ExchTime < 15:00:01",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "bar_semantics": "tick; 15:00:00 is in-sample",
        "cached_grain": "daily_scalar",
        "sequence_requery": True,
    },
    {
        "source_id": "ch_ssl2",
        "families": ("order_book", "ddb_reference_snapshot", "liquidity_spread_depth"),
        "session_window": "continuous is_close_auction=0; hour=15 stored separately",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "bar_semantics": "minute-last snapshot; minute_index 0-239 continuous, 240 auction",
        "cached_grain": "daily_scalar_plus_open_close_30m_and_auction_fields",
        "sequence_requery": True,
    },
    {
        "source_id": "ddb_stock_one_minute",
        "families": ("price_formation", "trade_flow_proxy", "cancel_proxy", "amihud_rv"),
        "session_window": "[09:30,11:30)+[13:00,15:00); 15:00 auction separate",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "includes_1500_tick": True,
        "bar_semantics": (
            "DDB omits observed 14:57-14:59; may carry 14:56 price through "
            "those labels; amount/volume never filled; 09:25 and 15:00 outside continuous"
        ),
        "cached_grain": "daily_scalar; raw_minute_panel_written=false",
        "sequence_requery": True,
    },
    {
        "source_id": "ch_lid_joined_minute",
        "families": ("liquidity_impact",),
        "session_window": "09:30-11:29 + 13:00-14:59",
        "includes_1456_1500": True,
        "includes_close_auction": False,
        "includes_1500_tick": False,
        "bar_semantics": "240 continuous minutes; hour-15 excluded",
        "cached_grain": "daily_scalar; raw_minute_panel_written=false",
        "sequence_requery": True,
    },
    {
        "source_id": "ch_cancel_tick",
        "families": ("cancel_lifecycle",),
        "session_window": "mkey 570-689 and 780-899",
        "includes_1456_1500": True,
        "includes_close_auction": False,
        "includes_1500_tick": False,
        "bar_semantics": "14:59 included (899); 15:00 (900) excluded",
        "cached_grain": "daily_scalar",
        "sequence_requery": True,
    },
)

FAMILY_CUT_FEASIBILITY: Tuple[Dict[str, object], ...] = (
    {
        "family": "trade_flow",
        "daily_primitive": "trade_flow_daily",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "REQUIRES_SEQUENCE_REQUERY",
        "cheap_tc1_source": "ddb_stock_one_minute Active_buy/sell amount",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "2024-06 DDB minutes (preferred) or CH tick",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "already_has_open_close_segments": False,
        "cannot_cut_examples": "flow_zscore_20d (daily scalar transform)",
    },
    {
        "family": "order_book",
        "daily_primitive": "order_book_daily",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "PARTIAL_FROM_DAILY_SEGMENTS",
        "cheap_tc1_source": "opening_30m_* / closing_30m_* already in daily cache; full state cuts need CH SSL2",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "2024-06 CH SSL2 minute-last",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "already_has_open_close_segments": True,
        "cannot_cut_examples": "daily mean/std/p10/p90 already scalars",
    },
    {
        "family": "order_size",
        "daily_primitive": "order_size_distribution_daily",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "REQUIRES_SEQUENCE_REQUERY",
        "cheap_tc1_source": "none in cache; CH tick minute GROUP BY for 2024-06",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "2024-06 CH tick size buckets",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "already_has_open_close_segments": False,
        "cannot_cut_examples": "all frozen ratios are daily totals",
    },
    {
        "family": "price_formation",
        "daily_primitive": "price_formation_daily",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "PARTIAL_FROM_DAILY_SEGMENTS",
        "cheap_tc1_source": "morning/afternoon/closing_30m returns already daily; new cuts need DDB minutes",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "2024-06 DDB Stock_one_minute",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "already_has_open_close_segments": True,
        "cannot_cut_examples": "realized_kurtosis / jump_share as daily scalars cannot be recut without minutes",
    },
    {
        "family": "liquidity_impact",
        "daily_primitive": "liquidity_impact_daily",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "REQUIRES_SEQUENCE_REQUERY",
        "cheap_tc1_source": "none; lid.joined_minute_sql 2024-06 if impact cuts are required",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "2024-06 CH LOCAL tick+SSL2 join",
        "includes_1456_1500": True,
        "includes_close_auction": False,
        "already_has_open_close_segments": False,
        "cannot_cut_examples": "large_trade_impact / impact_convexity daily; FS-ineligible coverage",
    },
    {
        "family": "cancel_lifecycle",
        "daily_primitive": "cancel_lifecycle_daily",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "REQUIRES_SEQUENCE_REQUERY",
        "cheap_tc1_source": "DDB Bid/Ask cancel volume proxies on Stock_one_minute",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "2024-06 DDB cancel proxies (cheap) or CH cancel rebuild (expensive)",
        "includes_1456_1500": True,
        "includes_close_auction": False,
        "already_has_open_close_segments": False,
        "cannot_cut_examples": "cancel_pressure_shock_20d (daily rolling scalar)",
    },
    {
        "family": "ddb_reference_snapshot",
        "daily_primitive": "ddb_reference_snapshot",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": True,
        "cut_support": "NOT_P0",
        "cheap_tc1_source": "none; 3s SSL2 dynamics, not minute-cut P0",
        "requires_ch_ddb_scan_for_cuts": True,
        "scan_scope_tc1": "deferred",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "already_has_open_close_segments": False,
        "cannot_cut_examples": "all five formulas are daily means of 3s series",
    },
    {
        "family": "trade_flow_mcap_bridge",
        "daily_primitive": "trade_flow_daily + FloatMktCap",
        "has_intraday_sequence_in_cache": False,
        "has_minute_or_tick_input": False,
        "cut_support": "CANNOT_CUT",
        "cheap_tc1_source": "none",
        "requires_ch_ddb_scan_for_cuts": False,
        "scan_scope_tc1": "n/a",
        "includes_1456_1500": True,
        "includes_close_auction": True,
        "already_has_open_close_segments": False,
        "cannot_cut_examples": "net_buy_amount_mcap is a daily scalar / mcap ratio",
    },
)

ALPHANET_OPERATOR_VOCAB: Tuple[str, ...] = (
    "ts_corr",
    "ts_cov",
    "ts_stddev",
    "ts_zscore",
    "ts_return",
    "ts_decaylinear",
    "ts_min",
    "ts_max",
    "ts_sum",
    "ts_mean",
)


def time_segment(name: str) -> Dict[str, object]:
    key = str(name).strip().upper()
    for row in TIME_SEGMENTS:
        if str(row["segment_name"]) == key:
            return dict(row)
    raise KeyError("unknown time segment {!r}".format(name))


def primitive_spec(name: str) -> Dict[str, object]:
    for row in P0_PRIMITIVE_WHITELIST:
        if str(row["base_primitive"]) == name:
            return dict(row)
    raise KeyError("unknown P0 primitive {!r}".format(name))


def allowed_aggregators(primitive_class: str) -> Tuple[str, ...]:
    if primitive_class not in PRIMITIVE_CLASS_AGGREGATORS:
        raise KeyError("unknown primitive_class {!r}".format(primitive_class))
    return PRIMITIVE_CLASS_AGGREGATORS[primitive_class]


def close_t_execution_forbidden(record: Mapping[str, object]) -> bool:
    """True when the feature must not claim Close[T] execution."""
    if bool(record.get("uses_close_auction")):
        return True
    if bool(record.get("uses_last_5min")):
        return True
    if bool(record.get("contains_close_auction")):
        return True
    ts = str(record.get("availability_timestamp", "")).lower()
    if "auction" in ts or "continuous_close" in ts or "after_15" in ts:
        return True
    return True  # L2 same-session aggregates are never Close[T] executable


def operator_contract_dict() -> Dict[str, object]:
    return {
        "module_id": CUT_MODULE_ID,
        "contract_version": CUT_CONTRACT_VERSION,
        "pipeline_position": (
            "NONLINEAR_REVIEW -> temporal/state cutting -> rescued factors "
            "-> executable retest -> residual-alpha -> LightGBM/XGBoost"
        ),
        "does_not_replace": ["candidate_pool_v1", "tree_models", "alphanet"],
        "sidecar_registry_only": True,
        "candidate_pool_frozen": True,
        "max_workers": MAX_WORKERS,
        "max_descendants_per_parent": MAX_DESCENDANTS_PER_PARENT,
        "preferred_descendants_per_parent": list(PREFERRED_DESCENDANTS_PER_PARENT),
        "event_q": EVENT_Q_DEFAULT,
        "no_q_grid_search": True,
        "no_uncontrolled_cartesian": True,
        "legacy_c2c_default": False,
        "legacy_c2c_requires_explicit_flag": LEGACY_C2C_FLAG,
        "production_execution_contract": PRODUCTION_EXECUTION_CONTRACT,
        "time_segments": [dict(r) for r in TIME_SEGMENTS],
        "state_definitions": [dict(r) for r in STATE_DEFINITIONS],
        "event_definitions": [dict(r) for r in EVENT_DEFINITIONS],
        "aggregators": list(AGGREGATORS),
        "primitive_class_aggregators": {
            k: list(v) for k, v in PRIMITIVE_CLASS_AGGREGATORS.items()
        },
        "contrast_operators": list(CONTRAST_OPERATORS),
        "ratio_epsilon": RATIO_EPSILON,
        "conditional_whitelist": [dict(r) for r in CONDITIONAL_WHITELIST],
        "p0_primitive_whitelist": [dict(r) for r in P0_PRIMITIVE_WHITELIST],
        "tc1_recipes": [dict(r) for r in TC1_RECIPES],
        "tc1_recipe_count": len(TC1_RECIPES),
        "modes": list(MODES),
        "rescue_classes": list(RESCUE_CLASSES),
        "rescue_aux_label": RESCUED_AUX_LABEL,
        "timing_localization_statuses": list(TIMING_LOCALIZATION_STATUSES),
        "parent_type_labels": list(PARENT_TYPE_LABELS),
        "rescue_core_gates": dict(RESCUE_CORE_GATES),
        "registry_columns": list(REGISTRY_COLUMNS),
        "parent_child_metric_columns": list(PARENT_CHILD_METRIC_COLUMNS),
        "source_timestamp_semantics": [dict(r) for r in SOURCE_TIMESTAMP_SEMANTICS],
        "family_cut_feasibility": [dict(r) for r in FAMILY_CUT_FEASIBILITY],
        "alphanet_vocab_reserved_not_implemented": list(ALPHANET_OPERATOR_VOCAB),
        "operator_pipeline": ["INPUT_PRIMITIVE", "CUT_MASK", "AGGREGATOR", "CONTRAST"],
        "state_threshold_policy": (
            "V1 uses within-stock-day medians or frozen economic thresholds. "
            "No full-sample future quantiles. No return-optimized cut search."
        ),
        "information_availability": {
            "close_t_execution": "FORBIDDEN for all generated cut features",
            "ai_v1_eval": PRODUCTION_EXECUTION_CONTRACT,
            "required_fields": [
                "cut_start_time",
                "cut_end_time",
                "availability_timestamp",
                "contains_close_auction",
                "contains_1456_1500",
                "execution_contract_compatible",
                "latest_source_timestamp",
                "factor_available_after",
                "uses_close_auction",
                "uses_last_5min",
                "production_execution_compatible",
            ],
        },
        "outputs": {
            "code": "l2_factor_reproduction/l2_ai_stock_selection/cut_operators/",
            "results": str(CUT_RESULT_ROOT),
            "forbidden_write": str(CANDIDATE_POOL_CSV),
        },
    }
