"""Unit tests for Temporal / State Cutting Operators v1 (synthetic data only)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.aggregators import (
    apply_aggregator,
    assert_aggregator_allowed,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CANDIDATE_POOL_CSV,
    CUT_RESULT_ROOT,
    EVENT_Q_DEFAULT,
    MAX_DESCENDANTS_PER_PARENT,
    MAX_WORKERS,
    REGISTRY_COLUMNS,
    TC1_RECIPES,
    close_t_execution_forbidden,
    operator_contract_dict,
    time_segment,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contrast_ops import (
    apply_contrast,
    contrast_normalized_diff,
    contrast_ratio,
    ratio_denominator_diagnostics,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.event_cuts import (
    event_mask,
    shock_mask,
    top_q_mask,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.generator import (
    CartesianSearchError,
    assert_max_workers,
    assert_not_cartesian,
    generate_from_recipes,
    generate_tc1_candidates,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.nonlinear_rescue import (
    classify_rescue,
    nonlinear_review_parents,
    rescue_from_jury,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    assert_candidate_pool_unchanged,
    candidate_name,
    parse_candidate_name,
    snapshot_candidate_pool,
    write_registry,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.state_cuts import (
    grouped_state_mask,
    state_mask,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.time_cuts import (
    AM_MKEY_END,
    AUCTION_MKEY,
    PM_MKEY_START,
    assert_no_future_day,
    consecutive_mkeys,
    continuous_mkey_grid,
    is_lunch,
    mkey_from_hm,
    segment_availability,
    time_mask,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (
    CUT_OPERATORS,
    frozen_artifact_paths,
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _one_day_panel(trade_date="2024-06-03", symbol="600000.SH"):
    keys = list(continuous_mkey_grid()) + [AUCTION_MKEY]
    n = len(keys)
    rng = np.random.default_rng(0)
    hour = np.array(keys) // 60
    minute = np.array(keys) % 60
    ts = pd.to_datetime(trade_date) + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(
        minute, unit="m"
    )
    x = np.arange(n, dtype=float)
    ret = rng.normal(0, 0.001, size=n)
    spread = 0.001 + 0.0001 * rng.random(n)
    depth = 1e6 + 1e4 * rng.normal(size=n)
    amount = 5e5 + 1e4 * rng.random(n)
    large = np.where(rng.random(n) > 0.8, 3e5, 5e4)
    return pd.DataFrame(
        {
            "TradeDate": pd.Timestamp(trade_date),
            "Symbol": symbol,
            "mkey": keys,
            "timestamp": ts,
            "x": x,
            "minute_return": ret,
            "relative_spread": spread,
            "total_depth_l5": depth,
            "amount": amount,
            "large_order_amount": large,
            "impact": ret * 0.5,
        }
    )


# ---------------------------------------------------------------------------
# 1. time segment boundaries
# ---------------------------------------------------------------------------
def test_time_segment_boundaries():
    keys = continuous_mkey_grid()
    open_m = time_mask(keys, "OPEN")
    morning = time_mask(keys, "MORNING")
    afternoon = time_mask(keys, "AFTERNOON")
    close = time_mask(keys, "CLOSE")
    full = time_mask(keys, "FULL")
    assert int(open_m.sum()) == 30
    assert int(morning.sum()) == 90
    assert int(afternoon.sum()) == 90
    assert int(close.sum()) == 30
    assert int(full.sum()) == 240
    assert mkey_from_hm(9, 30) in keys[open_m]
    assert mkey_from_hm(9, 59) in keys[open_m]
    assert mkey_from_hm(10, 0) not in keys[open_m]
    assert mkey_from_hm(10, 0) in keys[morning]
    assert mkey_from_hm(11, 29) in keys[morning]
    assert mkey_from_hm(11, 30) not in keys
    assert mkey_from_hm(13, 0) in keys[afternoon]
    assert mkey_from_hm(14, 29) in keys[afternoon]
    assert mkey_from_hm(14, 30) in keys[close]
    assert mkey_from_hm(14, 59) in keys[close]
    spec = time_segment("OPEN")
    assert spec["start_inclusive"] is True
    assert spec["end_inclusive"] is False


# ---------------------------------------------------------------------------
# 2. lunch break handling
# ---------------------------------------------------------------------------
def test_lunch_break_excluded_from_all_continuous_segments():
    lunch = list(range(11 * 60 + 30, 13 * 60))
    assert all(is_lunch(k) for k in lunch)
    for name in ("OPEN", "MORNING", "AFTERNOON", "CLOSE", "FULL"):
        mask = time_mask(lunch, name)
        assert not mask.any(), name
    assert consecutive_mkeys(AM_MKEY_END, PM_MKEY_START) is False
    assert consecutive_mkeys(AM_MKEY_END, AM_MKEY_END + 1) is False


# ---------------------------------------------------------------------------
# 3. close auction inclusion / exclusion
# ---------------------------------------------------------------------------
def test_close_auction_not_in_close_or_full():
    keys = np.array(list(continuous_mkey_grid()) + [AUCTION_MKEY], dtype=np.int32)
    close = time_mask(keys, "CLOSE")
    full = time_mask(keys, "FULL")
    auction = time_mask(keys, "CLOSE_AUCTION")
    assert keys[close].max() == 14 * 60 + 59
    assert AUCTION_MKEY not in keys[close]
    assert AUCTION_MKEY not in keys[full]
    assert list(keys[auction]) == [AUCTION_MKEY]
    late = time_mask(keys, "LATE_CLOSE")
    assert mkey_from_hm(14, 56) in keys[late]
    assert AUCTION_MKEY not in keys[late]
    avail = segment_availability("CLOSE")
    assert avail["contains_close_auction"] is False
    assert avail["contains_1456_1500"] is True
    assert avail["close_t_execution"] is False
    ddb_late = segment_availability("LATE_CLOSE", source_id="ddb_stock_one_minute")
    assert ddb_late["late_close_reliable"] is False


# ---------------------------------------------------------------------------
# 4. state mask construction
# ---------------------------------------------------------------------------
def test_state_mask_within_day_median():
    panel = pd.DataFrame(
        {
            "minute_return": [0.01, -0.01, 0.02, -0.02],
            "relative_spread": [1.0, 2.0, 3.0, 4.0],
            "total_depth_l5": [10, 20, 30, 40],
            "amount": [1, 2, 3, 4],
            "large_order_amount": [1e4, 1e4, 3e5, 3e5],
        }
    )
    hi_sp = state_mask("high_spread", panel)
    lo_sp = state_mask("low_spread", panel)
    assert hi_sp.tolist() == [False, False, True, True]
    assert lo_sp.tolist() == [True, True, False, False]
    assert not np.any(hi_sp & lo_sp)
    up = state_mask("price_up", panel)
    assert up.tolist() == [True, False, True, False]
    large = state_mask("large_order_dominated", panel)
    assert large.tolist() == [False, False, True, True]


# ---------------------------------------------------------------------------
# 5. no future-day data in a stock-day cut
# ---------------------------------------------------------------------------
def test_no_future_day_in_stock_day_cut():
    with pytest.raises(ValueError):
        assert_no_future_day(["2024-06-03", "2024-06-04"], "2024-06-03")
    assert_no_future_day(["2024-06-03", "2024-06-03"], "2024-06-03")
    day1 = _one_day_panel("2024-06-03")
    day2 = _one_day_panel("2024-06-04")
    day1["relative_spread"] = 1.0
    day2["relative_spread"] = 100.0
    both = pd.concat([day1, day2], ignore_index=True)
    mask = grouped_state_mask(both, "high_spread")
    # Day-1 values are constant -> nothing strictly above median.
    m1 = mask[: len(day1)]
    m2 = mask[len(day1) :]
    assert not m1.any()
    assert not m2.any()
    day1b = day1.copy()
    day1b.loc[day1b.index[:5], "relative_spread"] = 3.0
    day2b = day2.copy()
    both = pd.concat([day1b, day2b], ignore_index=True)
    mask = grouped_state_mask(both, "high_spread")
    # High-spread minutes on day 1 must not use day-2's huge level as threshold.
    assert mask[: len(day1b)].sum() >= 1
    assert mask[: len(day1b)].sum() < len(day1b)


# ---------------------------------------------------------------------------
# 6. event quantiles use only permitted data
# ---------------------------------------------------------------------------
def test_event_quantiles_stock_day_only_and_fixed_q():
    x = np.arange(10, dtype=float)
    top = top_q_mask(x, q=EVENT_Q_DEFAULT)
    assert top.sum() == 2  # 20% of 10
    assert set(x[top]) == {8.0, 9.0}
    with pytest.raises(ValueError):
        top_q_mask(x, q=0.10)
    # SHOCK is causal: first observations cannot fire.
    shock = shock_mask(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 50.0, 50.0]), min_history=5)
    assert not shock[:5].any()
    assert shock[-1]


# ---------------------------------------------------------------------------
# 7. mean / sum / std operators
# ---------------------------------------------------------------------------
def test_mean_sum_std_operators():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    mask = np.array([True, True, True, False, False, False])
    assert apply_aggregator("sum", x, mask) == 6.0
    assert apply_aggregator("mean", x, mask, min_obs=3) == 2.0
    assert apply_aggregator("std", x, mask, min_obs=3) == pytest.approx(
        float(np.std([1.0, 2.0, 3.0], ddof=0))
    )


# ---------------------------------------------------------------------------
# 8-10. contrast operators, normalized diff, zero denominator
# ---------------------------------------------------------------------------
def test_contrast_operators_and_denominators():
    assert apply_contrast("DIFF", 5, 2) == 3
    assert apply_contrast("ACCELERATION", 8, 3) == 5
    assert apply_contrast("REVERSAL", 2, -4) == 4.0
    nd = contrast_normalized_diff(1.0, -1.0)
    assert nd == pytest.approx(1.0, abs=1e-8)
    assert np.isnan(contrast_ratio(1.0, 0.0))
    diag = ratio_denominator_diagnostics(1.0, 0.0)
    assert diag["zero_denominator"] is True
    assert diag["ratio_defined"] is False
    assert np.isnan(contrast_normalized_diff(0.0, 0.0)) or abs(
        contrast_normalized_diff(0.0, 0.0)
    ) < 1e-9


# ---------------------------------------------------------------------------
# 11. deterministic candidate naming
# ---------------------------------------------------------------------------
def test_deterministic_candidate_naming():
    a = candidate_name("ofi", "time", "close", aggregation="mean")
    b = candidate_name("ofi", "time", "close", aggregation="mean")
    assert a == b == "ofi__time_close__mean"
    assert candidate_name("obi", "state", "high_spread", aggregation="mean") == (
        "obi__state_high_spread__mean"
    )
    assert candidate_name("ofi", "contrast", "close_minus_open") == (
        "ofi__contrast_close_minus_open"
    )
    parsed = parse_candidate_name("ofi__time_close__mean")
    assert parsed["base_primitive"] == "ofi"
    assert parsed["cut_type"] == "time"
    assert parsed["cut_name"] == "close"
    assert parsed["aggregation"] == "mean"


# ---------------------------------------------------------------------------
# 12. candidate budget enforcement
# ---------------------------------------------------------------------------
def test_candidate_budget_enforcement():
    cuts = [
        ("open", "sum"),
        ("open", "mean"),
        ("open", "last"),
        ("morning", "sum"),
        ("morning", "mean"),
        ("afternoon", "sum"),
        ("afternoon", "mean"),
        ("close", "sum"),
        ("close", "mean"),
        ("close", "last"),
        ("full", "sum"),
        ("full", "mean"),
    ]
    recipes = [
        {
            "base_primitive": "net_active_flow",
            "cut_type": "time",
            "cut_name": cut,
            "aggregation": agg,
            "reason": "budget probe {}".format(i),
        }
        for i, (cut, agg) in enumerate(cuts)
    ]
    frame, budget = generate_from_recipes(
        recipes,
        parent="net_buy_ratio",
        max_descendants=MAX_DESCENDANTS_PER_PARENT,
    )
    assert len(recipes) == 12
    assert budget["accepted_count"] == MAX_DESCENDANTS_PER_PARENT
    assert budget["rejected_count"] >= 2
    assert "budget_exceeded" in budget["rejection_reason"]
    assert len(frame) == MAX_DESCENDANTS_PER_PARENT


# ---------------------------------------------------------------------------
# 13. availability timestamp propagation
# ---------------------------------------------------------------------------
def test_availability_timestamp_propagation():
    recs = [
        {
            "base_primitive": "net_active_flow",
            "base_family": "trade_flow",
            "cut_type": "time",
            "cut_name": "close",
            "aggregation": "sum",
            "reason": "close flow path",
        }
    ]
    frame, _ = generate_from_recipes(recs)
    row = frame.iloc[0]
    assert row["availability_timestamp"] == "after_continuous_close_T"
    assert bool(row["contains_1456_1500"]) is True
    assert bool(row["contains_close_auction"]) is False
    assert bool(row["uses_last_5min"]) is True
    assert bool(row["execution_contract_compatible"]) is True
    assert close_t_execution_forbidden(row) is True


# ---------------------------------------------------------------------------
# 14. parent-child linkage
# ---------------------------------------------------------------------------
def test_parent_child_linkage():
    recs = [
        {
            "base_primitive": "obi_5",
            "base_family": "order_book",
            "cut_type": "time",
            "cut_name": "close",
            "aggregation": "mean",
            "reason": "closing book",
        }
    ]
    frame, budget = generate_from_recipes(
        recs, mode="NONLINEAR_RESCUE", parent="obi_l5_mean"
    )
    assert frame.iloc[0]["parent_factor_if_rescue"] == "obi_l5_mean"
    assert budget["parent"] == "obi_l5_mean"
    jury = pd.DataFrame(
        [
            {
                "factor": "obi_l5_mean",
                "jury_state": "REVIEW",
                "nonlinear_review": True,
            },
            {"factor": "net_buy_ratio", "jury_state": "KEEP", "nonlinear_review": False},
        ]
    )
    assert nonlinear_review_parents(jury) == ["obi_l5_mean"]
    reg, bud = rescue_from_jury(jury)
    assert len(reg) <= MAX_DESCENDANTS_PER_PARENT
    assert set(reg["parent_factor_if_rescue"]) == {"obi_l5_mean"}


# ---------------------------------------------------------------------------
# 15. no candidate_pool_v1 mutation
# ---------------------------------------------------------------------------
def test_no_candidate_pool_v1_mutation(tmp_path):
    before = snapshot_candidate_pool()
    assert CANDIDATE_POOL_CSV.exists()
    frame, _ = generate_tc1_candidates()
    sidecar = tmp_path / "cut_candidate_registry.csv"
    write_registry(frame, sidecar)
    with pytest.raises(RuntimeError):
        write_registry(frame, CANDIDATE_POOL_CSV)
    assert_candidate_pool_unchanged(before)
    assert sidecar.exists()
    assert _sha(CANDIDATE_POOL_CSV) == before["sha256"]
    frozen = frozen_artifact_paths()
    assert any("candidate_pool_v1" in str(p) for p in frozen)


# ---------------------------------------------------------------------------
# 16. duplicate candidate detection
# ---------------------------------------------------------------------------
def test_duplicate_candidate_detection():
    recs = [
        {
            "base_primitive": "net_active_flow",
            "cut_type": "time",
            "cut_name": "close",
            "aggregation": "sum",
            "reason": "late flow A",
        },
        {
            "base_primitive": "net_active_flow",
            "cut_type": "time",
            "cut_name": "close",
            "aggregation": "sum",
            "reason": "late flow B",
        },
    ]
    frame, budget = generate_from_recipes(recs)
    assert len(frame) == 1
    assert "duplicate_candidate_name" in budget["rejection_reason"]


# ---------------------------------------------------------------------------
# 17. minimum coverage handling
# ---------------------------------------------------------------------------
def test_minimum_coverage_handling():
    x = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    mask = np.array([True, True, True, False, False])
    assert np.isnan(apply_aggregator("mean", x, mask, min_obs=5))
    assert apply_aggregator("mean", x, mask, min_obs=2) == 1.5


# ---------------------------------------------------------------------------
# 18. no uncontrolled Cartesian product
# ---------------------------------------------------------------------------
def test_no_uncontrolled_cartesian_product():
    with pytest.raises(CartesianSearchError):
        assert_not_cartesian(20, 8, 12, 10, 8)
    assert_not_cartesian(2, 2, 2, 1, 1)
    with pytest.raises(ValueError):
        assert_max_workers(MAX_WORKERS + 1)
    frame, budget = generate_tc1_candidates()
    assert 30 <= len(frame) <= 50
    assert frame["candidate_name"].is_unique
    counts = frame.groupby("base_primitive").size()
    assert counts.max() <= MAX_DESCENDANTS_PER_PARENT
    assert set(frame.columns) == set(REGISTRY_COLUMNS)
    # Recipes are explicit; product of all operator vocab would be far larger.
    n_recipes = len(TC1_RECIPES)
    assert n_recipes == len(frame) or n_recipes >= len(frame)
    contract = operator_contract_dict()
    assert contract["no_uncontrolled_cartesian"] is True
    assert contract["sidecar_registry_only"] is True
    assert CUT_OPERATORS.name == "cut_operators"
    assert CUT_RESULT_ROOT.name == "cut_operators"


def test_aggregator_not_mechanically_applied():
    with pytest.raises(ValueError):
        assert_aggregator_allowed("sum", "spread_level")
    assert_aggregator_allowed("mean", "spread_level")


def test_persistence_does_not_cross_lunch():
    from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.aggregators import (
        agg_persistence,
    )

    keys = np.array([687, 688, 689, PM_MKEY_START], dtype=np.int32)
    x = np.array([1.0, 1.0, 1.0, -1.0])
    mask = np.array([True, True, True, True])
    val = agg_persistence(x, mask, keys, min_pairs=1)
    # Lunch break must not create a 689 -> 780 pair (that pair would be 1 vs -1).
    assert val == pytest.approx(1.0)


def test_rescue_class_not_binary():
    assert (
        classify_rescue(abs_rank_ic=0.03, hl_sharpe=3.2, monotonicity=0.75)
        == "RESCUED_CORE"
    )
    assert (
        classify_rescue(
            abs_rank_ic=0.03,
            hl_sharpe=3.2,
            monotonicity=0.75,
            corr_core=0.90,
        )
        == "REDUNDANT_RESCUE"
    )
    assert (
        classify_rescue(
            abs_rank_ic=0.005,
            hl_sharpe=0.4,
            monotonicity=0.2,
            mi=0.05,
        )
        == "NONLINEAR_ONLY"
    )


def test_event_mask_column_dispatch():
    panel = _one_day_panel()
    top = event_mask("TOP_Q", panel, "x")
    assert top.sum() > 0
    liq = event_mask("LIQUIDITY_SHOCK", panel, "x")
    assert liq.dtype == bool
