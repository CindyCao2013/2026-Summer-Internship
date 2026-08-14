"""Pure unit tests for mid-trade-amount normalization invariants."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.mid_trade_amount_normalization import (  # noqa: E402
    A0_FACTOR_ID,
    A1_FACTOR_ID,
    A2_FACTOR_ID,
    A3_FACTOR_ID,
    FROZEN_CONFIG_HASH_FIELD,
    FROZEN_DIRECTION_POLICY,
    amount_share_from_aggregates,
    assert_a0_parity,
    assert_unique_symbol_trade_date,
    build_candidate_grid,
    build_lagged_trade_size_scales,
    candidate_grid_name,
    compare_a0_parity,
    compute_a0_share,
    compute_a1_share,
    compute_a2_share,
    compute_a3_share,
    compute_dynamic_factor_shares,
    evaluate_frozen_direction,
    format_fee_bps_label,
    freeze_a1_distribution_candidate,
    freeze_config,
    implied_annual_fee,
    merge_symbol_trade_date_one_to_one,
    relative_trade_size_adv_bps,
    validate_frozen_config,
)


def _daily_scale_fixture() -> tuple:
    dates = pd.bdate_range("2024-01-02", periods=22)
    rows = []
    for symbol, total_amount, median_amount in (
        ("LOW_ADV.SH", 20_000_000.0, 50_000.0),
        ("HIGH_ADV.SZ", 2_000_000_000.0, 500_000.0),
    ):
        for trade_date in dates:
            rows.append(
                {
                    "Symbol": symbol,
                    "TradeDate": trade_date,
                    "total_amount": total_amount,
                    "daily_median_trade_amount": median_amount,
                }
            )
    return pd.DataFrame(rows), dates


def _row(
    frame: pd.DataFrame,
    symbol: str,
    trade_date: pd.Timestamp,
) -> pd.Series:
    return frame.loc[
        frame["Symbol"].eq(symbol)
        & frame["TradeDate"].eq(pd.Timestamp(trade_date))
    ].iloc[0]


def test_toy_cross_stock_normalization_changes_relative_trade_size() -> None:
    amount = [100_000.0]
    low_adv = 20_000_000.0
    high_adv = 2_000_000_000.0

    assert compute_a0_share(amount) == pytest.approx(1.0)
    assert compute_a0_share(amount) == pytest.approx(
        compute_a0_share(amount)
    )
    assert relative_trade_size_adv_bps(amount[0], low_adv) == pytest.approx(
        50.0
    )
    assert relative_trade_size_adv_bps(
        amount[0], high_adv
    ) == pytest.approx(0.5)

    # A reasonable relative interval can classify the same RMB trade
    # differently; A0 necessarily classifies both the same way.
    assert compute_a1_share(amount, low_adv, 0.25, 1.0) == pytest.approx(0.0)
    assert compute_a1_share(amount, high_adv, 0.25, 1.0) == pytest.approx(
        1.0
    )


def test_scales_use_exact_prior_20_market_days_and_keep_evidence() -> None:
    primitive, dates = _daily_scale_fixture()
    scales = build_lagged_trade_size_scales(primitive, dates)

    first_valid = _row(scales, "LOW_ADV.SH", dates[20])
    assert first_valid["ADV20_lag1"] == pytest.approx(20_000_000.0)
    assert first_valid["ADV20_median_lag1"] == pytest.approx(20_000_000.0)
    assert first_valid["ATS20_lag1"] == pytest.approx(50_000.0)
    assert first_valid["ADV20_history_count"] == 20
    assert first_valid["ATS20_history_count"] == 20
    assert first_valid["history_count"] == 20
    assert first_valid["source_max_date"] == dates[19]
    assert first_valid["ADV20_source_max_date"] == dates[19]
    assert first_valid["ATS20_source_max_date"] == dates[19]

    before = scales.loc[
        scales["Symbol"].eq("LOW_ADV.SH")
        & scales["TradeDate"].lt(dates[20]),
        ["ADV20_lag1", "ADV20_median_lag1", "ATS20_lag1"],
    ]
    assert before.isna().all().all()


def test_current_day_mutation_cannot_change_same_day_scale() -> None:
    primitive, dates = _daily_scale_fixture()
    baseline = build_lagged_trade_size_scales(primitive, dates)

    mutated = primitive.copy()
    current = (
        mutated["Symbol"].eq("LOW_ADV.SH")
        & mutated["TradeDate"].eq(dates[20])
    )
    mutated.loc[current, "total_amount"] = 2_000_000_000.0
    mutated.loc[current, "daily_median_trade_amount"] = 5_000_000.0
    changed = build_lagged_trade_size_scales(mutated, dates)

    baseline_t = _row(baseline, "LOW_ADV.SH", dates[20])
    changed_t = _row(changed, "LOW_ADV.SH", dates[20])
    for column in (
        "ADV20_lag1",
        "ADV20_median_lag1",
        "ATS20_lag1",
        "source_max_date",
    ):
        assert changed_t[column] == baseline_t[column]

    # The mutation becomes eligible exactly one market day later.
    changed_next = _row(changed, "LOW_ADV.SH", dates[21])
    expected_adv = (19 * 20_000_000.0 + 2_000_000_000.0) / 20
    assert changed_next["ADV20_lag1"] == pytest.approx(expected_adv)
    assert changed_next["source_max_date"] == dates[20]


def test_missing_one_calendar_day_invalidates_scale_instead_of_looking_back() -> None:
    primitive, dates = _daily_scale_fixture()
    missing = primitive.loc[
        ~(
            primitive["Symbol"].eq("LOW_ADV.SH")
            & primitive["TradeDate"].eq(dates[7])
        )
    ]
    scales = build_lagged_trade_size_scales(missing, dates)

    low = _row(scales, "LOW_ADV.SH", dates[20])
    assert low["history_count"] == 19
    assert low["ADV20_history_count"] == 19
    assert low["ATS20_history_count"] == 19
    assert pd.isna(low["ADV20_lag1"])
    assert pd.isna(low["ADV20_median_lag1"])
    assert pd.isna(low["ATS20_lag1"])

    high = _row(scales, "HIGH_ADV.SZ", dates[20])
    assert high["history_count"] == 20
    assert pd.notna(high["ADV20_lag1"])
    assert pd.notna(high["ATS20_lag1"])


def test_a0_a1_a2_a3_helpers_share_amount_denominator_and_boundaries() -> None:
    amounts = [40_000.0, 100_000.0, 200_000.0, 300_000.0]
    total = sum(amounts)
    expected_middle = 300_000.0 / total

    # Lower bounds are exclusive; upper bounds are inclusive.
    assert compute_a0_share(amounts) == pytest.approx(expected_middle)
    assert compute_a2_share(amounts, 100_000.0, 0.4, 2.0) == pytest.approx(
        expected_middle
    )
    assert compute_a3_share(amounts, 40_000.0, 200_000.0) == pytest.approx(
        expected_middle
    )
    assert compute_a3_share(
        [100_000.0, 100_000.0],
        100_000.0,
        100_000.0,
    ) == pytest.approx(0.0)
    assert np.isnan(compute_a3_share(amounts, np.nan, np.nan))
    assert amount_share_from_aggregates(300_000.0, total) == pytest.approx(
        expected_middle
    )

    all_variants = compute_dynamic_factor_shares(
        amounts,
        adv20_lag1=20_000_000.0,
        ats20_lag1=100_000.0,
        daily_q20=40_000.0,
        daily_q80=200_000.0,
        a1_lower_bps=20.0,
        a1_upper_bps=100.0,
        a2_lower_multiple=0.4,
        a2_upper_multiple=2.0,
    )
    assert set(all_variants) == {
        A0_FACTOR_ID,
        A1_FACTOR_ID,
        A2_FACTOR_ID,
        A3_FACTOR_ID,
    }
    assert all_variants[A0_FACTOR_ID] == pytest.approx(expected_middle)
    assert all_variants[A2_FACTOR_ID] == pytest.approx(expected_middle)
    assert all_variants[A3_FACTOR_ID] == pytest.approx(expected_middle)


def test_symbol_trade_date_duplicate_is_a_hard_join_error() -> None:
    left = pd.DataFrame(
        {
            "Symbol": ["600000.SH"],
            "TradeDate": ["2024-01-02"],
            "left_value": [1.0],
        }
    )
    right = pd.DataFrame(
        {
            "Symbol": ["600000.SH", "600000.SH"],
            "TradeDate": ["2024-01-02", "2024-01-02 09:30:00"],
            "right_value": [2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match=r"Symbol\+TradeDate"):
        assert_unique_symbol_trade_date(right)
    with pytest.raises(ValueError, match=r"Symbol\+TradeDate"):
        merge_symbol_trade_date_one_to_one(left, right)


def test_candidate_grid_names_are_stable_and_invalid_pairs_fail() -> None:
    a1 = build_candidate_grid("A1")
    a2 = build_candidate_grid("A2")
    assert len(a1) == 9
    assert len(a2) == 9
    assert len({candidate.name for candidate in a1}) == 9
    assert len({candidate.name for candidate in a2}) == 9
    assert a1[0].name == "a1_adv20_l0p5_h5_bps"
    assert a2[0].name == "a2_ats20_l0p25_h1p5_x"
    assert candidate_grid_name("A1", 1.0, 10.0) == (
        "a1_adv20_l1_h10_bps"
    )
    assert all(candidate.lower < candidate.upper for candidate in a1 + a2)

    with pytest.raises(ValueError, match="lower < upper"):
        build_candidate_grid(
            "A1",
            lower_values=[10.0],
            upper_values=[5.0],
        )


def _freeze_distribution_fixture() -> pd.DataFrame:
    candidates = [
        # The first two are tied on overall distance and quintile spread.
        (candidate_grid_name("A1", 0.5, 5.0), 0.5, 5.0, [0.25] * 5),
        (candidate_grid_name("A1", 1.0, 5.0), 1.0, 5.0, [0.50] * 5),
        # Closest overall, but ineligible because Q1 coverage exceeds 80%.
        (
            candidate_grid_name("A1", 2.0, 5.0),
            2.0,
            5.0,
            [0.90, 0.25, 0.25, 0.25, 0.25],
        ),
    ]
    rows = []
    for candidate, lower, upper, coverages in candidates:
        for quintile, coverage in enumerate(coverages, start=1):
            rows.append(
                {
                    "candidate": candidate,
                    "market_cap_quintile": quintile,
                    "selected_amount": coverage * 100.0,
                    "total_amount": 100.0,
                    "lower_bps": lower,
                    "upper_bps": upper,
                }
            )
    return pd.DataFrame(rows)


def test_a1_distribution_only_freeze_is_deterministic_with_tie_breaks() -> None:
    distribution = _freeze_distribution_fixture()
    selected_a = freeze_a1_distribution_candidate(
        distribution.sample(frac=1.0, random_state=7),
        a0_overall_coverage=0.375,
    )
    selected_b = freeze_a1_distribution_candidate(
        distribution.sample(frac=1.0, random_state=19),
        a0_overall_coverage=0.375,
    )

    expected = candidate_grid_name("A1", 0.5, 5.0)
    assert selected_a == selected_b
    assert selected_a["candidate"] == expected
    assert selected_a["selection_basis"] == (
        "distribution_only_no_returns"
    )
    assert selected_a["eligible_candidate_count"] == 2
    assert selected_a["tie_break_order"][:4] == [
        "overall_coverage_gap_vs_a0",
        "quintile_coverage_gap",
        "lower_bps",
        "upper_bps",
    ]

    contaminated = distribution.assign(sharpe=99.0)
    with pytest.raises(ValueError, match="distribution-only"):
        freeze_a1_distribution_candidate(
            contaminated,
            a0_overall_coverage=0.375,
        )


def test_frozen_config_hash_is_canonical_and_tamper_evident() -> None:
    selected = freeze_a1_distribution_candidate(
        _freeze_distribution_fixture(),
        a0_overall_coverage=0.375,
    )
    config_a = {
        "a1": selected,
        "a2": {
            "lower_multiple": 0.5,
            "upper_multiple": 2.0,
        },
        "effective_direction": {
            "A0": -1,
            "A1": -1,
            "A2": -1,
            "A3": -1,
        },
    }
    config_b = {
        "effective_direction": {
            "A3": -1,
            "A2": -1,
            "A1": -1,
            "A0": -1,
        },
        "a2": {
            "upper_multiple": 2.0,
            "lower_multiple": 0.5,
        },
        "a1": selected,
    }
    frozen_a = freeze_config(config_a)
    frozen_b = freeze_config(config_b)
    assert frozen_a[FROZEN_CONFIG_HASH_FIELD] == frozen_b[
        FROZEN_CONFIG_HASH_FIELD
    ]
    assert validate_frozen_config(frozen_a) == frozen_a[
        FROZEN_CONFIG_HASH_FIELD
    ]

    tampered = copy.deepcopy(frozen_a)
    tampered["a1"]["upper_bps"] = 10.0
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_frozen_config(tampered)
    with pytest.raises(ValueError, match="expected Stage-B"):
        validate_frozen_config(frozen_a, expected_sha256="0" * 64)


def test_a0_parity_helper_checks_keys_nan_pattern_and_values() -> None:
    authoritative = pd.DataFrame(
        {
            "Symbol": ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"],
            "TradeDate": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"]
            ),
            "value": [0.1, 0.2, 0.3, np.nan],
        }
    )
    rebuilt = authoritative.sample(frac=1.0, random_state=3).copy()
    rebuilt.loc[rebuilt["Symbol"].eq("000002.SZ"), "value"] += 1e-12
    result = assert_a0_parity(rebuilt, authoritative)
    assert result["passed"]
    assert result["spearman"] == pytest.approx(1.0)
    assert result["max_abs_error"] <= 1e-10

    wrong_nan = rebuilt.copy()
    wrong_nan.loc[wrong_nan["Symbol"].eq("600001.SH"), "value"] = 0.4
    failed = compare_a0_parity(wrong_nan, authoritative)
    assert not failed["passed"]
    assert not failed["nan_pattern_match"]


def test_frozen_direction_never_auto_flips_when_window_changes() -> None:
    positive_raw_window = evaluate_frozen_direction(
        [0.01, 0.02, 0.03],
        effective_direction=-1,
    )
    negative_raw_window = evaluate_frozen_direction(
        [-0.01, -0.02, -0.03],
        effective_direction=-1,
    )

    for result in (positive_raw_window, negative_raw_window):
        assert result["effective_direction"] == -1
        assert result["direction_policy"] == FROZEN_DIRECTION_POLICY
        assert not result["direction_was_inferred"]
    assert not positive_raw_window["effective_hl_mean_positive"]
    assert negative_raw_window["effective_hl_mean_positive"]


def test_fee_bps_math_and_label_cannot_be_read_as_percent() -> None:
    turnover = 0.8
    expected = turnover * 7.5 / 10_000.0 * 250
    assert math.isclose(
        implied_annual_fee(turnover, fee_bps=7.5),
        expected,
    )
    label = format_fee_bps_label(7.5)
    assert label == "fee=7.5 bps"
    assert "7.5%" not in label
    assert "bps" in label
