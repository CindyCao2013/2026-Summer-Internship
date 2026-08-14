from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.order_book_factors import (
    ORDER_BOOK_FACTOR_NAMES,
    REQUIRED_PRIMITIVE_COLUMNS,
    build_order_book_feature_chunk,
    build_order_book_feature_frame,
    feature_to_narrow,
    registry_frame,
)


def _fixture(days: int = 30) -> pd.DataFrame:
    rows = []
    for symbol_offset, symbol in enumerate(("600000.SH", "000001.SZ")):
        for day in range(days):
            row = {
                column: 0.0 for column in REQUIRED_PRIMITIVE_COLUMNS
            }
            row.update(
                {
                    "symbol": symbol,
                    "TradeDate": pd.Timestamp("2024-01-02")
                    + pd.offsets.BDay(day),
                    "coverage_ratio": 1.0,
                    "obi_1_mean": -0.2 + day * 0.01,
                    "obi_5_mean": -0.1 + day * 0.02 + symbol_offset * 0.01,
                    "obi_10_mean": -0.05 + day * 0.01,
                    "weighted_obi_mean": -0.08 + day * 0.015,
                    "obi_1_std": 0.2,
                    "obi_5_std": 0.15,
                    "weighted_obi_std": 0.12,
                    "near_far_imbalance_mean": 0.03,
                    "bid_depth_hhi_mean": 0.2,
                    "ask_depth_hhi_mean": 0.25,
                    "depth_concentration_asymmetry_mean": -0.05,
                    "bid_depth_slope_mean": 80.0 + day,
                    "ask_depth_slope_mean": 90.0 + day,
                    "depth_slope_asymmetry_mean": -10.0,
                    "relative_spread_mean": 0.001 + day * 0.00001,
                    "relative_spread_std": 0.0002,
                    "microprice_deviation_mean": -0.0001 + day * 0.00001,
                    "microprice_deviation_std": 0.0003,
                    "book_vwap_gap_mean": 0.01,
                    "log_total_depth_mean": 12.0 + day * 0.02,
                    "log_total_depth_std": 0.3,
                    "opening_30m_obi_5": -0.2 + day * 0.01,
                    "closing_30m_obi_5": 0.1 + day * 0.01,
                    "opening_30m_relative_spread": 0.0015,
                    "closing_30m_relative_spread": 0.0010,
                    "opening_30m_log_depth": 11.5,
                    "closing_30m_log_depth": 12.5,
                    "obi_5_intraday_slope": 0.0005,
                    "obi_5_sign_persistence": 0.8,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_registry_freezes_32_nonduplicate_formulas() -> None:
    registry = registry_frame()
    assert len(registry) == 32
    assert tuple(registry["name"]) == ORDER_BOOK_FACTOR_NAMES
    assert not registry["name"].duplicated().any()
    assert registry.loc[
        registry["name"] == "obi_l5_mean", "alias_features"
    ].notna().all()


def test_static_formula_values_and_coverage_mask() -> None:
    primitive = _fixture(2)
    primitive.loc[
        (primitive["symbol"] == "600000.SH")
        & (primitive["TradeDate"] == primitive["TradeDate"].min()),
        "coverage_ratio",
    ] = 0.5
    features = build_order_book_feature_frame(primitive)
    valid = features.loc[
        (features["symbol"] == "600000.SH")
        & (features["TradeDate"] == features["TradeDate"].max())
    ].iloc[0]
    assert valid["obi_l5_mean"] == pytest.approx(-0.08)
    assert valid["depth_concentration_asymmetry"] == pytest.approx(-0.05)
    assert valid["depth_slope_asymmetry"] == pytest.approx(-10.0)
    assert valid["opening_closing_obi_change"] == pytest.approx(0.3)
    assert valid["opening_closing_spread_change"] == pytest.approx(-0.0005)
    assert valid["opening_closing_depth_change"] == pytest.approx(1.0)

    masked = features.loc[
        (features["symbol"] == "600000.SH")
        & (features["TradeDate"] == features["TradeDate"].min())
    ].iloc[0]
    assert masked[list(ORDER_BOOK_FACTOR_NAMES)].isna().all()


def test_inclusive_rolling_20d_uses_no_future_rows() -> None:
    primitive = _fixture(30)
    baseline = build_order_book_feature_frame(primitive)
    target_date = sorted(primitive["TradeDate"].unique())[19]
    target = baseline.loc[
        (baseline["symbol"] == "600000.SH")
        & (baseline["TradeDate"] == target_date),
        "obi_shock_20d",
    ].iloc[0]
    values = np.array([-0.1 + day * 0.02 for day in range(20)])
    expected = (values[-1] - values.mean()) / values.std(ddof=0)
    assert target == pytest.approx(expected)

    changed = primitive.copy()
    changed.loc[changed["TradeDate"] > target_date, "obi_5_mean"] = 0.99
    revised = build_order_book_feature_frame(changed)
    revised_target = revised.loc[
        (revised["symbol"] == "600000.SH")
        & (revised["TradeDate"] == target_date),
        "obi_shock_20d",
    ].iloc[0]
    assert revised_target == pytest.approx(target)


def test_streaming_chunk_rolling_matches_full_frame() -> None:
    primitive = _fixture(30).sort_values(["TradeDate", "symbol"])
    split_date = sorted(primitive["TradeDate"].unique())[14]
    first = primitive.loc[primitive["TradeDate"] <= split_date]
    second = primitive.loc[primitive["TradeDate"] > split_date]
    first_features, history = build_order_book_feature_chunk(first)
    second_features, _ = build_order_book_feature_chunk(second, history)
    streamed = pd.concat(
        [first_features, second_features], ignore_index=True
    ).sort_values(["symbol", "TradeDate"]).reset_index(drop=True)
    full = build_order_book_feature_frame(primitive).sort_values(
        ["symbol", "TradeDate"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(streamed, full)


def test_zero_rolling_std_and_narrow_schema_are_safe() -> None:
    primitive = _fixture(25)
    primitive["relative_spread_mean"] = 0.001
    features = build_order_book_feature_frame(primitive)
    assert features["spread_shock_20d"].isna().all()
    narrow = feature_to_narrow(features, "obi_l5_mean")
    assert list(narrow.columns) == [
        "symbol",
        "tradetime",
        "factorname",
        "value",
    ]
    assert set(narrow["factorname"]) == {"obi_l5_mean"}
    assert np.isfinite(narrow["value"]).all()
