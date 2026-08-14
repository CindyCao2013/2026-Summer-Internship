"""TC-2A frozen parent contract and explicit recipes.

Frozen before descendant generation. Negative control is net_buy_ratio.
Do not expand to the remaining TC-2 parents from this module.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

TC2A_WINDOW_START = "2023-01-01"
TC2A_WINDOW_END = "2024-12-31"
TC2A_EXECUTION_CONTRACT = "EXEC_V2V_TPLUS1_V1"
TC2A_MAX_DESCENDANTS_PER_PARENT = 6
TC2A_TARGET_CANDIDATE_RANGE = (40, 60)

# Frozen BEFORE any descendant is generated or inspected.
TC2A_NEGATIVE_CONTROL = "net_buy_ratio"
TC2A_NEGATIVE_CONTROL_REASON = (
    "Pre-registered EQ-1 DROP. Broad daily net-buy ratio with |RankIC|~0.006. "
    "Preferred over buy_dominance as the weaker all-day flow aggregate."
)
TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION = True
TC2A_POSITIVE_CONTROL = "signed_amount_impact"

PARENT_TYPES = (
    "LEVEL_PARENT",
    "PATH_PARENT",
    "DERIVED_TRANSFORM_PARENT",
)

RESEARCH_QUESTIONS = (
    "NONLINEAR_STRUCTURAL_RESCUE",
    "TIMING_LOCALIZATION",
    "POSITIVE_CONTROL",
    "NEGATIVE_CONTROL",
)

TC2A_RESCUE_STATUSES = (
    "RESCUED_CORE",
    "RESCUED_AUX",
    "NONLINEAR_ONLY",
    "FAILED_RESCUE",
    "REDUNDANT_RESCUE",
    "POSITIVE_CONTROL_CONFIRM",
    "NEGATIVE_CONTROL_NOT_PROMOTED",
)

TC2A_TIMING_STATUSES = (
    "TIMING_LOCALIZED_EXECUTABLE",
    "TIMING_LOCALIZED_TOO_FAST",
    "NO_CLEAR_TIMING_STRUCTURE",
)

# COMMON_CLOSE robustness: latest minute reliably present on both DDB and CH.
# DDB Stock_one_minute omits observed 14:57-14:59. Not a tuned parameter.
COMMON_CLOSE_START = "14:30:00"
COMMON_CLOSE_END = "14:57:00"  # half-open; last bar 14:56
COMMON_CLOSE_MKEY_START = 14 * 60 + 30  # 870
COMMON_CLOSE_MKEY_END = 14 * 60 + 56  # 896
COMMON_CLOSE_N_EXPECTED = 27

MATERIAL_IMPROVEMENT = {
    "delta_abs_ic": 0.005,
    "delta_hl_sharpe": 0.50,
    "delta_monotonicity": 0.10,
    "parent_child_redundant_corr": 0.95,
    "core_redundant_corr": 0.80,
}

OVERFLEXIBLE = {
    "neg_control_core_descendants": 2,
    "neg_control_aux_descendants": 3,
}

# ---------------------------------------------------------------------------
# Parent contract. 12 pre-registered names. Types assigned from construction,
# not from post-hoc performance.
# ---------------------------------------------------------------------------
TC2A_PARENTS: Tuple[Dict[str, object], ...] = (
    {
        "parent_factor": "vwap_close_deviation",
        "parent_type": "DERIVED_TRANSFORM_PARENT",
        "research_question": "NONLINEAR_STRUCTURAL_RESCUE",
        "family": "price_formation",
        "underlying_primitive": "close_and_vwap",
        "parent_transform": "(continuous_close - daily_vwap) / daily_vwap",
        "cut_level": "reapply_ratio_on_cut_window",
        "preferred_source": "ddb_stock_one_minute",
        "best_horizon": 5,
        "note": "Do not cut(vwap_close_deviation). Recompute VWAP and last close inside the window.",
    },
    {
        "parent_factor": "closing_30m_return",
        "parent_type": "PATH_PARENT",
        "research_question": "NONLINEAR_STRUCTURAL_RESCUE",
        "family": "price_formation",
        "underlying_primitive": "minute_return",
        "parent_transform": "sum(minute_return | CLOSE)",
        "cut_level": "do_not_recut_parent_CLOSE_as_rescue",
        "preferred_source": "ddb_stock_one_minute",
        "best_horizon": 3,
        "note": "Parent already is the CLOSE path. Descendants localize or contrast other windows.",
    },
    {
        "parent_factor": "afternoon_return",
        "parent_type": "PATH_PARENT",
        "research_question": "NONLINEAR_STRUCTURAL_RESCUE",
        "family": "price_formation",
        "underlying_primitive": "minute_return",
        "parent_transform": "sum(minute_return | AFTERNOON)",
        "cut_level": "do_not_recut_parent_AFTERNOON_as_rescue",
        "preferred_source": "ddb_stock_one_minute",
        "best_horizon": 3,
        "note": "Parent already is the AFTERNOON path.",
    },
    {
        "parent_factor": "close_location_value",
        "parent_type": "DERIVED_TRANSFORM_PARENT",
        "research_question": "NONLINEAR_STRUCTURAL_RESCUE",
        "family": "price_formation",
        "underlying_primitive": "ohlc",
        "parent_transform": "(2*continuous_close-high-low)/(high-low)",
        "cut_level": "reapply_clv_on_cut_window",
        "preferred_source": "ddb_stock_one_minute",
        "best_horizon": 1,
        "note": "Recompute CLV from segment last/high/low. Do not cut(CLV).",
    },
    {
        "parent_factor": "return_per_amount",
        "parent_type": "DERIVED_TRANSFORM_PARENT",
        "research_question": "NONLINEAR_STRUCTURAL_RESCUE",
        "family": "price_formation",
        "underlying_primitive": "return_and_amount",
        "parent_transform": "open_to_close_return / daily_amount",
        "cut_level": "reapply_ratio_on_cut_window",
        "preferred_source": "ddb_stock_one_minute",
        "best_horizon": 10,
        "note": "Cut return and amount, then the same ratio. Do not cut(return_per_amount).",
    },
    {
        "parent_factor": "impact_asymmetry",
        "parent_type": "DERIVED_TRANSFORM_PARENT",
        "research_question": "NONLINEAR_STRUCTURAL_RESCUE",
        "family": "liquidity_impact",
        "underlying_primitive": "fwd1_mid_and_signed_flow",
        "parent_transform": "buy_price_impact - sell_price_impact",
        "cut_level": "reapply_buy_minus_sell_impact_on_cut_window",
        "preferred_source": "ddb_active_flow_plus_close_fwd1_proxy",
        "best_horizon": 1,
        "note": "Minute proxy: DDB active buy/sell x next-minute close return. Same operator on the cut.",
    },
    {
        "parent_factor": "obi_l5_mean",
        "parent_type": "LEVEL_PARENT",
        "research_question": "TIMING_LOCALIZATION",
        "family": "order_book",
        "underlying_primitive": "obi_5",
        "parent_transform": "mean(obi_5 | FULL)",
        "cut_level": "subset_then_same_mean",
        "preferred_source": "ch_ssl2",
        "best_horizon": 3,
        "note": "Timing: where does the daily OBI mean live.",
    },
    {
        "parent_factor": "closing_obi_l5",
        "parent_type": "PATH_PARENT",
        "research_question": "TIMING_LOCALIZATION",
        "family": "order_book",
        "underlying_primitive": "obi_5",
        "parent_transform": "mean(obi_5 | CLOSE)",
        "cut_level": "do_not_recut_parent_CLOSE_as_rescue",
        "preferred_source": "ch_ssl2",
        "best_horizon": 3,
        "note": "Parent already is CLOSE OBI. Localize vs OPEN / gap / dispersion.",
    },
    {
        "parent_factor": "microprice_deviation_mean",
        "parent_type": "LEVEL_PARENT",
        "research_question": "TIMING_LOCALIZATION",
        "family": "order_book",
        "underlying_primitive": "microprice_deviation",
        "parent_transform": "mean(microprice_deviation | FULL)",
        "cut_level": "subset_then_same_mean",
        "preferred_source": "ch_ssl2",
        "best_horizon": 20,
        "note": "Signed L1 microprice vs mid. TemporalCenter uses |deviation| weights (not +/- split of spread-like mass).",
    },
    {
        "parent_factor": "large_order_pressure",
        "parent_type": "LEVEL_PARENT",
        "research_question": "TIMING_LOCALIZATION",
        "family": "order_size",
        "underlying_primitive": "large_order_signed_share",
        "parent_transform": "(large_buy-large_sell)/total_amt",
        "cut_level": "subset_then_same_ratio",
        "preferred_source": "ch_tick",
        "best_horizon": 1,
        "note": "Relative large prints (stock-day q80). Zero large-order denom stays missing. Report P(no activity).",
    },
    {
        "parent_factor": "signed_amount_impact",
        "parent_type": "LEVEL_PARENT",
        "research_question": "POSITIVE_CONTROL",
        "family": "liquidity_impact",
        "underlying_primitive": "minute_return_x_signed_amount",
        "parent_transform": "sum(r*signed_amt)/sum(|signed_amt|)",
        "cut_level": "subset_then_same_weighted_impact",
        "preferred_source": "ddb_close_return_x_signed_amount_proxy",
        "best_horizon": 20,
        "note": "EQ-1 CORE positive control. Cuts confirm the operator, not a rescue claim.",
    },
    {
        "parent_factor": "net_buy_ratio",
        "parent_type": "LEVEL_PARENT",
        "research_question": "NEGATIVE_CONTROL",
        "family": "trade_flow",
        "underlying_primitive": "net_active_flow",
        "parent_transform": "(active_buy-active_sell)/total_amt",
        "cut_level": "reapply_ratio_on_cut_window",
        "preferred_source": "ddb_stock_one_minute",
        "best_horizon": 10,
        "note": "FROZEN negative control. Same budget. Do not promote.",
    },
)


def _r(**kwargs) -> Dict[str, object]:
    row = dict(kwargs)
    if "candidate_name" not in row:
        raise ValueError("TC-2A recipe missing candidate_name")
    if "reason" not in row:
        raise ValueError("TC-2A recipe missing economic reason")
    return row


TC2A_RECIPES: Tuple[Dict[str, object], ...] = (
    # ----- vwap_close_deviation (DERIVED) 5 -----
    _r(
        parent_factor="vwap_close_deviation",
        candidate_name="vwap_close_deviation__time_close_reapply",
        base_primitive="vwap_close_deviation",
        cut_type="derived",
        cut_name="close",
        derived_op="vwap_close_deviation",
        aggregation="reapply",
        reason="Close vs its own window VWAP: late prints that disagree with late traded price.",
    ),
    _r(
        parent_factor="vwap_close_deviation",
        candidate_name="vwap_close_deviation__time_open_reapply",
        base_primitive="vwap_close_deviation",
        cut_type="derived",
        cut_name="open",
        derived_op="vwap_close_deviation",
        aggregation="reapply",
        reason="Open discovery vs opening VWAP; tests whether the daily close-VWAP gap is a morning artifact.",
    ),
    _r(
        parent_factor="vwap_close_deviation",
        candidate_name="vwap_close_deviation__contrast_close_minus_open",
        base_primitive="vwap_close_deviation",
        cut_type="derived_contrast",
        cut_name="close_minus_open",
        derived_op="vwap_close_deviation",
        contrast_operator="DIFF",
        aggregation="reapply",
        reason="Path of close-vs-VWAP from open window to close window.",
    ),
    _r(
        parent_factor="vwap_close_deviation",
        candidate_name="vwap_close_deviation__time_afternoon_reapply",
        base_primitive="vwap_close_deviation",
        cut_type="derived",
        cut_name="afternoon",
        derived_op="vwap_close_deviation",
        aggregation="reapply",
        reason="Afternoon VWAP disagreement, excluding the last 30m close scramble.",
    ),
    _r(
        parent_factor="vwap_close_deviation",
        candidate_name="vwap_close_deviation__state_high_vol_reapply",
        base_primitive="vwap_close_deviation",
        cut_type="derived",
        cut_name="high_vol",
        derived_op="vwap_close_deviation",
        aggregation="reapply",
        condition_primitive="abs_minute_return",
        reason="VWAP-close gap using only high-volatility minutes, where path vs consensus should be informative.",
    ),
    # ----- closing_30m_return (PATH) 5 -----
    _r(
        parent_factor="closing_30m_return",
        candidate_name="closing_30m_return__time_open",
        base_primitive="minute_return",
        cut_type="time",
        cut_name="open",
        aggregation="sum",
        reason="If closing-30m alpha is actually opening reversal, OPEN return should dominate V2V.",
    ),
    _r(
        parent_factor="closing_30m_return",
        candidate_name="closing_30m_return__contrast_close_minus_open",
        base_primitive="minute_return",
        cut_type="contrast",
        cut_name="close_minus_open",
        contrast_operator="DIFF",
        aggregation="sum",
        reason="Close-vs-open return path; parent hides the morning leg.",
    ),
    _r(
        parent_factor="closing_30m_return",
        candidate_name="closing_30m_return__contrast_afternoon_minus_morning",
        base_primitive="minute_return",
        cut_type="contrast",
        cut_name="afternoon_minus_morning",
        contrast_operator="DIFF",
        aggregation="sum",
        reason="Session-half contrast: afternoon continuation vs morning.",
    ),
    _r(
        parent_factor="closing_30m_return",
        candidate_name="closing_30m_return__temporal_gap",
        base_primitive="minute_return",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="TemporalGap of signed returns: when selling vs buying mass arrives.",
    ),
    _r(
        parent_factor="closing_30m_return",
        candidate_name="closing_30m_return__state_high_vol",
        base_primitive="minute_return",
        cut_type="state",
        cut_name="high_vol",
        aggregation="sum",
        condition_primitive="abs_minute_return",
        reason="Return-location in high-volatility minutes, not the calm path.",
    ),
    # ----- afternoon_return (PATH) 4 -----
    _r(
        parent_factor="afternoon_return",
        candidate_name="afternoon_return__time_morning",
        base_primitive="minute_return",
        cut_type="time",
        cut_name="morning",
        aggregation="sum",
        reason="Morning counterpart: does the afternoon parent just proxy a morning reversal?",
    ),
    _r(
        parent_factor="afternoon_return",
        candidate_name="afternoon_return__contrast_afternoon_minus_morning",
        base_primitive="minute_return",
        cut_type="contrast",
        cut_name="afternoon_minus_morning",
        contrast_operator="DIFF",
        aggregation="sum",
        reason="Explicit afternoon-minus-morning contrast the parent aggregation hides.",
    ),
    _r(
        parent_factor="afternoon_return",
        candidate_name="afternoon_return__temporal_gap",
        base_primitive="minute_return",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="Where in the day the signed return mass sits.",
    ),
    _r(
        parent_factor="afternoon_return",
        candidate_name="afternoon_return__state_high_vol",
        base_primitive="minute_return",
        cut_type="state",
        cut_name="high_vol",
        aggregation="sum",
        condition_primitive="abs_minute_return",
        reason="Afternoon-style path restricted to high-vol minutes.",
    ),
    # ----- close_location_value (DERIVED) 4 -----
    _r(
        parent_factor="close_location_value",
        candidate_name="close_location_value__time_close_reapply",
        base_primitive="close_location_value",
        cut_type="derived",
        cut_name="close",
        derived_op="close_location_value",
        aggregation="reapply",
        reason="CLV inside the close range: last print vs close-window high/low.",
    ),
    _r(
        parent_factor="close_location_value",
        candidate_name="close_location_value__time_open_reapply",
        base_primitive="close_location_value",
        cut_type="derived",
        cut_name="open",
        derived_op="close_location_value",
        aggregation="reapply",
        reason="Opening CLV; daily CLV may just be an open-drive leftover.",
    ),
    _r(
        parent_factor="close_location_value",
        candidate_name="close_location_value__contrast_close_minus_open",
        base_primitive="close_location_value",
        cut_type="derived_contrast",
        cut_name="close_minus_open",
        derived_op="close_location_value",
        contrast_operator="DIFF",
        aggregation="reapply",
        reason="Change in location-in-range from open window to close window.",
    ),
    _r(
        parent_factor="close_location_value",
        candidate_name="close_location_value__time_afternoon_reapply",
        base_primitive="close_location_value",
        cut_type="derived",
        cut_name="afternoon",
        derived_op="close_location_value",
        aggregation="reapply",
        reason="Afternoon location-in-range, excluding the last 30m.",
    ),
    # ----- return_per_amount (DERIVED) 5 -----
    _r(
        parent_factor="return_per_amount",
        candidate_name="return_per_amount__time_close_reapply",
        base_primitive="return_per_amount",
        cut_type="derived",
        cut_name="close",
        derived_op="return_per_amount",
        aggregation="reapply",
        reason="Close-window return per close amount: impact when liquidity is typically thinner.",
    ),
    _r(
        parent_factor="return_per_amount",
        candidate_name="return_per_amount__time_open_reapply",
        base_primitive="return_per_amount",
        cut_type="derived",
        cut_name="open",
        derived_op="return_per_amount",
        aggregation="reapply",
        reason="Open-window return per open amount.",
    ),
    _r(
        parent_factor="return_per_amount",
        candidate_name="return_per_amount__contrast_close_minus_open",
        base_primitive="return_per_amount",
        cut_type="derived_contrast",
        cut_name="close_minus_open",
        derived_op="return_per_amount",
        contrast_operator="DIFF",
        aggregation="reapply",
        reason="Close vs open impact-per-currency; daily ratio averages them.",
    ),
    _r(
        parent_factor="return_per_amount",
        candidate_name="return_per_amount__time_afternoon_reapply",
        base_primitive="return_per_amount",
        cut_type="derived",
        cut_name="afternoon",
        derived_op="return_per_amount",
        aggregation="reapply",
        reason="Afternoon return per afternoon amount.",
    ),
    _r(
        parent_factor="return_per_amount",
        candidate_name="return_per_amount__state_high_vol_reapply",
        base_primitive="return_per_amount",
        cut_type="derived",
        cut_name="high_vol",
        derived_op="return_per_amount",
        aggregation="reapply",
        condition_primitive="abs_minute_return",
        reason="Impact-per-currency only in high-volatility minutes.",
    ),
    # ----- impact_asymmetry (DERIVED) 4 -----
    _r(
        parent_factor="impact_asymmetry",
        candidate_name="impact_asymmetry__time_close_reapply",
        base_primitive="impact_asymmetry",
        cut_type="derived",
        cut_name="close",
        derived_op="impact_asymmetry",
        aggregation="reapply",
        reason="Buy vs sell 1m impact in the close, where inventory absorption differs.",
    ),
    _r(
        parent_factor="impact_asymmetry",
        candidate_name="impact_asymmetry__time_open_reapply",
        base_primitive="impact_asymmetry",
        cut_type="derived",
        cut_name="open",
        derived_op="impact_asymmetry",
        aggregation="reapply",
        reason="Open-window buy-sell impact gap.",
    ),
    _r(
        parent_factor="impact_asymmetry",
        candidate_name="impact_asymmetry__contrast_close_minus_open",
        base_primitive="impact_asymmetry",
        cut_type="derived_contrast",
        cut_name="close_minus_open",
        derived_op="impact_asymmetry",
        contrast_operator="DIFF",
        aggregation="reapply",
        reason="Path of impact asymmetry from open to close.",
    ),
    _r(
        parent_factor="impact_asymmetry",
        candidate_name="impact_asymmetry__state_large_order_dominated",
        base_primitive="impact_asymmetry",
        cut_type="derived",
        cut_name="large_order_dominated",
        derived_op="impact_asymmetry",
        aggregation="reapply",
        condition_primitive="large_order_amount",
        reason="Buy-sell impact gap only in large-order-dominated minutes.",
    ),
    # ----- obi_l5_mean (LEVEL, timing) 5 -----
    _r(
        parent_factor="obi_l5_mean",
        candidate_name="obi_l5_mean__time_close",
        base_primitive="obi_5",
        cut_type="time",
        cut_name="close",
        aggregation="mean",
        reason="Closing book imbalance vs the all-day mean.",
    ),
    _r(
        parent_factor="obi_l5_mean",
        candidate_name="obi_l5_mean__time_open",
        base_primitive="obi_5",
        cut_type="time",
        cut_name="open",
        aggregation="mean",
        reason="Opening OBI; tests whether daily mean alpha is an open leftover that dies overnight.",
    ),
    _r(
        parent_factor="obi_l5_mean",
        candidate_name="obi_l5_mean__contrast_close_minus_open",
        base_primitive="obi_5",
        cut_type="contrast",
        cut_name="close_minus_open",
        contrast_operator="DIFF",
        aggregation="mean",
        reason="OBI path: close book minus open book.",
    ),
    _r(
        parent_factor="obi_l5_mean",
        candidate_name="obi_l5_mean__temporal_gap",
        base_primitive="obi_5",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="TemporalGap of signed OBI: when ask-pressure vs bid-pressure mass arrives.",
    ),
    _r(
        parent_factor="obi_l5_mean",
        candidate_name="obi_l5_mean__state_high_spread",
        base_primitive="obi_5",
        cut_type="state",
        cut_name="high_spread",
        aggregation="mean",
        condition_primitive="relative_spread",
        reason="OBI conditional on a wide spread, where displayed imbalance is costlier to fade.",
    ),
    # ----- closing_obi_l5 (PATH, timing) 5 -----
    _r(
        parent_factor="closing_obi_l5",
        candidate_name="closing_obi_l5__time_open",
        base_primitive="obi_5",
        cut_type="time",
        cut_name="open",
        aggregation="mean",
        reason="Open OBI counterpart to the closing-OBI parent.",
    ),
    _r(
        parent_factor="closing_obi_l5",
        candidate_name="closing_obi_l5__contrast_close_minus_open",
        base_primitive="obi_5",
        cut_type="contrast",
        cut_name="close_minus_open",
        contrast_operator="DIFF",
        aggregation="mean",
        reason="Does closing OBI information live in the close-open change?",
    ),
    _r(
        parent_factor="closing_obi_l5",
        candidate_name="closing_obi_l5__contrast_afternoon_minus_morning",
        base_primitive="obi_5",
        cut_type="contrast",
        cut_name="afternoon_minus_morning",
        contrast_operator="DIFF",
        aggregation="mean",
        reason="Session-half OBI contrast.",
    ),
    _r(
        parent_factor="closing_obi_l5",
        candidate_name="closing_obi_l5__temporal_gap",
        base_primitive="obi_5",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="Continuous location of signed OBI mass.",
    ),
    _r(
        parent_factor="closing_obi_l5",
        candidate_name="closing_obi_l5__temporal_dispersion",
        base_primitive="obi_5",
        cut_type="temporal",
        cut_name="temporal_dispersion",
        aggregation="temporal_dispersion",
        signed=True,
        reason="How concentrated in time the |OBI| mass is; persistence of a late book.",
    ),
    # ----- microprice_deviation_mean (LEVEL, timing) 4 -----
    _r(
        parent_factor="microprice_deviation_mean",
        candidate_name="microprice_deviation_mean__time_close",
        base_primitive="microprice_deviation",
        cut_type="time",
        cut_name="close",
        aggregation="mean",
        reason="Closing microprice-vs-mid disagreement.",
    ),
    _r(
        parent_factor="microprice_deviation_mean",
        candidate_name="microprice_deviation_mean__time_open",
        base_primitive="microprice_deviation",
        cut_type="time",
        cut_name="open",
        aggregation="mean",
        reason="Opening microprice deviation; likely too fast for V2V if it is a quote flicker.",
    ),
    _r(
        parent_factor="microprice_deviation_mean",
        candidate_name="microprice_deviation_mean__contrast_close_minus_open",
        base_primitive="microprice_deviation",
        cut_type="contrast",
        cut_name="close_minus_open",
        contrast_operator="DIFF",
        aggregation="mean",
        reason="Path of microprice pressure from open to close.",
    ),
    _r(
        parent_factor="microprice_deviation_mean",
        candidate_name="microprice_deviation_mean__temporal_center_abs",
        base_primitive="microprice_deviation",
        cut_type="temporal",
        cut_name="temporal_center",
        aggregation="temporal_center",
        signed=False,
        weight="abs",
        reason="Nonnegative |microprice_deviation| weighting; no +/- split. When the disagreement mass sits.",
    ),
    # ----- large_order_pressure (LEVEL, timing) 5 -----
    _r(
        parent_factor="large_order_pressure",
        candidate_name="large_order_pressure__time_close",
        base_primitive="large_order_pressure",
        cut_type="time",
        cut_name="close",
        aggregation="mean",
        reason="Late-session relative large-print pressure.",
    ),
    _r(
        parent_factor="large_order_pressure",
        candidate_name="large_order_pressure__time_open",
        base_primitive="large_order_pressure",
        cut_type="time",
        cut_name="open",
        aggregation="mean",
        reason="Opening large-print pressure; likely too fast if it is an open print.",
    ),
    _r(
        parent_factor="large_order_pressure",
        candidate_name="large_order_pressure__contrast_close_minus_open",
        base_primitive="large_order_pressure",
        cut_type="contrast",
        cut_name="close_minus_open",
        contrast_operator="NORMALIZED_DIFF",
        aggregation="mean",
        reason="Normalized close-open large-order pressure path, bounded in [-1,1].",
    ),
    _r(
        parent_factor="large_order_pressure",
        candidate_name="large_order_pressure__temporal_gap",
        base_primitive="large_order_pressure",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="When large buy vs large sell mass arrives.",
    ),
    _r(
        parent_factor="large_order_pressure",
        candidate_name="large_order_pressure__state_price_down",
        base_primitive="large_order_pressure",
        cut_type="state",
        cut_name="price_down",
        aggregation="mean",
        condition_primitive="minute_return",
        reason="Large-order pressure into falling prices, not chasing upticks.",
    ),
    # ----- signed_amount_impact (LEVEL, positive control) 5 -----
    _r(
        parent_factor="signed_amount_impact",
        candidate_name="signed_amount_impact__time_close",
        base_primitive="signed_amount_impact",
        cut_type="derived",
        cut_name="close",
        derived_op="signed_amount_impact",
        aggregation="reapply",
        reason="Positive control: late-session signed flow x contemporaneous return.",
    ),
    _r(
        parent_factor="signed_amount_impact",
        candidate_name="signed_amount_impact__time_open",
        base_primitive="signed_amount_impact",
        cut_type="derived",
        cut_name="open",
        derived_op="signed_amount_impact",
        aggregation="reapply",
        reason="Opening signed impact; if this dies on V2V the CORE parent is not an open leftover.",
    ),
    _r(
        parent_factor="signed_amount_impact",
        candidate_name="signed_amount_impact__contrast_close_minus_open",
        base_primitive="signed_amount_impact",
        cut_type="derived_contrast",
        cut_name="close_minus_open",
        derived_op="signed_amount_impact",
        contrast_operator="DIFF",
        aggregation="reapply",
        reason="Close vs open signed impact path.",
    ),
    _r(
        parent_factor="signed_amount_impact",
        candidate_name="signed_amount_impact__state_large_order_dominated",
        base_primitive="signed_amount_impact",
        cut_type="derived",
        cut_name="large_order_dominated",
        derived_op="signed_amount_impact",
        aggregation="reapply",
        condition_primitive="large_order_amount",
        reason="Impact during large-order minutes, the hypothesized state for this parent.",
    ),
    _r(
        parent_factor="signed_amount_impact",
        candidate_name="signed_amount_impact__temporal_gap",
        base_primitive="net_active_flow",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="TemporalGap of signed flow mass that the impact parent weights by return.",
    ),
    # ----- net_buy_ratio (LEVEL, NEGATIVE CONTROL) 5 -----
    _r(
        parent_factor="net_buy_ratio",
        candidate_name="net_buy_ratio__time_close_reapply",
        base_primitive="net_buy_ratio",
        cut_type="derived",
        cut_name="close",
        derived_op="net_buy_ratio",
        aggregation="reapply",
        reason="NEGATIVE CONTROL: close-window net-buy ratio. Must not be promoted.",
    ),
    _r(
        parent_factor="net_buy_ratio",
        candidate_name="net_buy_ratio__time_open_reapply",
        base_primitive="net_buy_ratio",
        cut_type="derived",
        cut_name="open",
        derived_op="net_buy_ratio",
        aggregation="reapply",
        reason="NEGATIVE CONTROL: open-window net-buy ratio.",
    ),
    _r(
        parent_factor="net_buy_ratio",
        candidate_name="net_buy_ratio__contrast_close_minus_open",
        base_primitive="net_buy_ratio",
        cut_type="derived_contrast",
        cut_name="close_minus_open",
        derived_op="net_buy_ratio",
        contrast_operator="DIFF",
        aggregation="reapply",
        reason="NEGATIVE CONTROL: close-minus-open net-buy path.",
    ),
    _r(
        parent_factor="net_buy_ratio",
        candidate_name="net_buy_ratio__state_high_spread",
        base_primitive="net_active_flow",
        cut_type="state",
        cut_name="high_spread",
        aggregation="sum",
        condition_primitive="relative_spread",
        reason="NEGATIVE CONTROL: active flow when the quote is wide.",
    ),
    _r(
        parent_factor="net_buy_ratio",
        candidate_name="net_buy_ratio__temporal_gap",
        base_primitive="net_active_flow",
        cut_type="temporal",
        cut_name="temporal_gap",
        aggregation="temporal_gap",
        signed=True,
        reason="NEGATIVE CONTROL: TemporalGap of signed active flow.",
    ),
)


def parent_by_name() -> Dict[str, Dict[str, object]]:
    return {str(r["parent_factor"]): dict(r) for r in TC2A_PARENTS}


def recipes_for_parent(parent: str) -> List[Dict[str, object]]:
    return [dict(r) for r in TC2A_RECIPES if str(r["parent_factor"]) == parent]


def assert_tc2a_budget() -> None:
    from collections import Counter

    n = len(TC2A_RECIPES)
    lo, hi = TC2A_TARGET_CANDIDATE_RANGE
    if not (lo <= n <= hi):
        raise AssertionError(
            "TC-2A recipe count {} outside target {}-{}".format(n, lo, hi)
        )
    counts = Counter(str(r["parent_factor"]) for r in TC2A_RECIPES)
    expected = {str(p["parent_factor"]) for p in TC2A_PARENTS}
    if set(counts) != expected:
        raise AssertionError("recipe parents {} != contract {}".format(set(counts), expected))
    for parent, k in counts.items():
        if k > TC2A_MAX_DESCENDANTS_PER_PARENT:
            raise AssertionError(
                "{} has {} descendants > max {}".format(
                    parent, k, TC2A_MAX_DESCENDANTS_PER_PARENT
                )
            )
    names = [str(r["candidate_name"]) for r in TC2A_RECIPES]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate TC-2A candidate names")
    if str(TC2A_PARENTS[-1]["parent_factor"]) != TC2A_NEGATIVE_CONTROL:
        if TC2A_NEGATIVE_CONTROL not in expected:
            raise AssertionError("negative control missing from parent contract")
    nc = recipes_for_parent(TC2A_NEGATIVE_CONTROL)
    if not nc:
        raise AssertionError("negative control has no recipes")
    if not TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION:
        raise AssertionError("negative control must be frozen before inspection")


assert_tc2a_budget()
assert TC2A_NEGATIVE_CONTROL == "net_buy_ratio"
assert TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION is True
