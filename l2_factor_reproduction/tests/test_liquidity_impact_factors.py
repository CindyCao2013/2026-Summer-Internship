from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.liquidity_impact_factors import (
    LIQUIDITY_IMPACT_FACTOR_NAMES,
    LIQUIDITY_IMPACT_FACTOR_SPECS,
    build_liquidity_impact_feature_frame,
    feature_to_narrow,
    registry_frame,
)


def _primitive_fixture() -> pd.DataFrame:
    dates = pd.date_range("2024-06-03", periods=4, freq="B")
    symbols = ["600000.SH", "000001.SZ"]
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, trade_date in enumerate(dates):
            scale = 1 + symbol_index + date_index
            rows.append(
                {
                    "symbol": symbol,
                    "TradeDate": trade_date,
                    "coverage_ratio": 0.9875,
                    "expected_continuous_minutes": 240,
                    "spread_per_depth": 1e-5 * scale,
                    "depth_per_amount": 0.02 * scale,
                    "amount_to_depth": 50.0 * scale,
                    "depth_turnover": 0.5 * scale,
                    "liquidity_cost_state": 1e-7 * scale,
                    "signed_amount_impact": 1e-6 * scale,
                    "signed_sqrt_amount_impact": 2e-6 * scale,
                    "impact_per_trade": 3e-6 * scale,
                    "buy_price_impact": 4e-6 * scale,
                    "sell_price_impact": -2e-6 * scale,
                    "effective_spread_proxy": 5e-4 * scale,
                    "permanent_impact_1m": 1e-4 * scale,
                    "permanent_impact_5m": 2e-4 * scale,
                    "small_trade_impact": 6e-6 * scale,
                    "mid_trade_impact": 7e-6 * scale,
                    "large_trade_impact": 8e-6 * scale,
                    "super_large_trade_impact": 9e-6 * scale,
                    "spread_recovery_5m": 0.3,
                    "depth_recovery_5m": 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_registry_freezes_exactly_24_unique_level_formulas() -> None:
    assert len(LIQUIDITY_IMPACT_FACTOR_NAMES) == 24
    assert len(set(LIQUIDITY_IMPACT_FACTOR_NAMES)) == 24
    assert set(LIQUIDITY_IMPACT_FACTOR_NAMES) == set(
        LIQUIDITY_IMPACT_FACTOR_SPECS
    )
    assert all(
        spec.lookback_days == 1
        for spec in LIQUIDITY_IMPACT_FACTOR_SPECS.values()
    )


def test_registry_excludes_existing_family_levels() -> None:
    excluded = {
        "relative_spread_mean",
        "total_depth_level",
        "intraday_amihud",
        "range_per_amount",
        "return_per_amount",
        "net_buy_ratio",
        "obi_5",
        "obi_level",
        "realized_volatility",
    }
    assert excluded.isdisjoint(LIQUIDITY_IMPACT_FACTOR_NAMES)
    assert not any(
        name.startswith("rank_") for name in LIQUIDITY_IMPACT_FACTOR_NAMES
    )
    assert not any(
        name.endswith("_zscore") for name in LIQUIDITY_IMPACT_FACTOR_NAMES
    )


def test_proxy_named_factors_disclose_proxy_note() -> None:
    proxy_named = [
        name
        for name in LIQUIDITY_IMPACT_FACTOR_NAMES
        if "proxy" in name
    ]
    assert set(proxy_named) == {
        "effective_spread_proxy",
        "realized_spread_proxy_5m",
    }
    for name in proxy_named:
        assert LIQUIDITY_IMPACT_FACTOR_SPECS[name].proxy_note


def test_derived_formulas_match_frozen_definitions() -> None:
    primitive = _primitive_fixture()
    features = build_liquidity_impact_feature_frame(primitive)
    row = features.loc[
        features["symbol"].eq("600000.SH")
        & features["TradeDate"].eq(pd.Timestamp("2024-06-03"))
    ].iloc[0]
    expected = primitive.loc[0]
    assert row["impact_asymmetry"] == pytest.approx(
        expected["buy_price_impact"] - expected["sell_price_impact"]
    )
    assert row["realized_spread_proxy_5m"] == pytest.approx(
        expected["effective_spread_proxy"]
        - expected["permanent_impact_5m"]
    )
    assert row["adverse_selection_ratio"] == pytest.approx(
        expected["permanent_impact_5m"]
        / abs(expected["effective_spread_proxy"])
    )
    assert row["impact_convexity"] == pytest.approx(
        expected["large_trade_impact"] - expected["small_trade_impact"]
    )
    assert row["impact_decay_5m"] == pytest.approx(
        expected["permanent_impact_1m"] - expected["permanent_impact_5m"]
    )


def test_adverse_selection_ratio_safe_denominator() -> None:
    primitive = _primitive_fixture()
    primitive.loc[0, "effective_spread_proxy"] = 1e-13
    primitive.loc[1, "effective_spread_proxy"] = 1e-12 * 1.5
    primitive.loc[1, "permanent_impact_5m"] = 1.0
    features = build_liquidity_impact_feature_frame(primitive)
    row = features.loc[
        features["symbol"].eq("600000.SH")
        & features["TradeDate"].eq(pd.Timestamp("2024-06-03"))
    ].iloc[0]
    assert np.isnan(row["adverse_selection_ratio"])
    extreme = features.loc[
        features["symbol"].eq("600000.SH")
        & features["TradeDate"].eq(pd.Timestamp("2024-06-04"))
    ].iloc[0]
    assert np.isnan(extreme["adverse_selection_ratio"])
    others = features["adverse_selection_ratio"].dropna()
    assert np.isfinite(others).all()
    assert (others.abs() <= 1e6).all()


def test_coverage_gate_nulls_low_coverage_symbol_days() -> None:
    primitive = _primitive_fixture()
    primitive.loc[0, "coverage_ratio"] = 0.79
    features = build_liquidity_impact_feature_frame(primitive)
    gated = features.loc[
        features["symbol"].eq("600000.SH")
        & features["TradeDate"].eq(pd.Timestamp("2024-06-03"))
    ].iloc[0]
    assert gated[list(LIQUIDITY_IMPACT_FACTOR_NAMES)].isna().all()
    valid = features.loc[
        ~(
            features["symbol"].eq("600000.SH")
            & features["TradeDate"].eq(pd.Timestamp("2024-06-03"))
        )
    ]
    assert valid[list(LIQUIDITY_IMPACT_FACTOR_NAMES)].notna().all().all()


def test_expected_minutes_mismatch_gates_values() -> None:
    primitive = _primitive_fixture()
    primitive.loc[0, "expected_continuous_minutes"] = 239
    features = build_liquidity_impact_feature_frame(primitive)
    row = features.loc[
        features["symbol"].eq("600000.SH")
        & features["TradeDate"].eq(pd.Timestamp("2024-06-03"))
    ].iloc[0]
    assert row[list(LIQUIDITY_IMPACT_FACTOR_NAMES)].isna().all()


def test_narrow_output_has_frozen_schema_and_no_nonfinite_values() -> None:
    features = build_liquidity_impact_feature_frame(_primitive_fixture())
    narrow = feature_to_narrow(features, "signed_amount_impact")
    assert list(narrow.columns) == [
        "symbol",
        "tradetime",
        "factorname",
        "value",
    ]
    assert narrow["factorname"].eq("signed_amount_impact").all()
    assert np.isfinite(narrow["value"]).all()
    assert narrow["tradetime"].dt.strftime("%H:%M").eq("09:30").all()


def test_feature_frame_rejects_duplicate_symbol_day() -> None:
    primitive = _primitive_fixture()
    duplicated = pd.concat(
        [primitive, primitive.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="unique"):
        build_liquidity_impact_feature_frame(duplicated)


def test_registry_frame_is_stable_and_rejects_unknown_names() -> None:
    registry = registry_frame()
    assert registry["name"].tolist() == list(LIQUIDITY_IMPACT_FACTOR_NAMES)
    assert registry["name"].is_unique
    assert set(registry["category"]) == {
        "spread_depth_liquidity",
        "signed_price_impact",
        "temporary_permanent_impact",
        "size_conditioned_impact",
        "resilience_recovery",
    }
    with pytest.raises(KeyError, match="unknown"):
        registry_frame(["not_a_factor"])
    with pytest.raises(KeyError, match="unknown"):
        feature_to_narrow(
            build_liquidity_impact_feature_frame(_primitive_fixture()),
            "not_a_factor",
        )
