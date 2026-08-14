from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.price_formation_factors import (
    PRICE_FORMATION_FACTOR_NAMES,
    PRICE_FORMATION_FACTOR_SPECS,
    build_price_formation_feature_chunk,
    build_price_formation_feature_frame,
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
                    "daily_amount": 1_000_000.0 * scale,
                    "overnight_gap": 0.001 * scale,
                    "open_to_30m_return": 0.002 * scale,
                    "morning_return": 0.003 * scale,
                    "afternoon_return": -0.001 * scale,
                    "closing_30m_return": 0.0005 * scale,
                    "lunch_gap_return": -0.0002 * scale,
                    "close_auction_return": 0.0001 * scale,
                    "vwap_close_deviation": 0.0015 * scale,
                    "close_location_value": -0.5 + 0.1 * scale,
                    "path_efficiency": min(0.1 * scale, 1.0),
                    "intraday_return_sign_persistence": 0.4,
                    "minute_return_autocorr1": -0.1,
                    "variance_ratio_5m": 0.8,
                    "realized_variance": 0.0004 * scale,
                    "downside_semivariance_share": 0.55,
                    "realized_skewness": -0.2,
                    "realized_kurtosis": 4.0,
                    "jump_share": 0.1,
                    "max_abs_minute_return": 0.01,
                    "tail_return_share": 0.3,
                    "max_drawdown_intraday": 0.02,
                    "max_drawup": 0.03,
                    "opening_30m_amount_share": 0.2,
                    "closing_30m_amount_share": 0.15,
                    "morning_amount_share": 0.55,
                    "afternoon_amount_share": 0.45,
                    "volume_concentration_hhi": 0.01,
                    "amount_time_center": 0.48,
                    "volume_return_corr": 0.1,
                    "volume_abs_return_corr": 0.3,
                    "intraday_amihud": 1e-9,
                    "return_per_amount": 1e-10 * scale,
                    "range_per_amount": 2e-10 * scale,
                }
            )
    return pd.DataFrame(rows)


def test_registry_freezes_exactly_32_unique_level_formulas() -> None:
    assert len(PRICE_FORMATION_FACTOR_NAMES) == 32
    assert len(set(PRICE_FORMATION_FACTOR_NAMES)) == 32
    assert set(PRICE_FORMATION_FACTOR_NAMES) == set(
        PRICE_FORMATION_FACTOR_SPECS
    )
    assert all(
        spec.lookback_days == 1
        for spec in PRICE_FORMATION_FACTOR_SPECS.values()
    )


def test_registry_excludes_optional_inconsistent_active_flow() -> None:
    excluded = {
        "signed_flow_price_impact",
        "buy_sell_impact_asymmetry",
        "flow_price_efficiency",
    }
    assert excluded.isdisjoint(PRICE_FORMATION_FACTOR_NAMES)


def test_realized_variance_is_not_a_candidate_alias() -> None:
    assert "realized_volatility" in PRICE_FORMATION_FACTOR_NAMES
    assert "realized_variance" not in PRICE_FORMATION_FACTOR_NAMES
    assert not any(name.startswith("rank_") for name in PRICE_FORMATION_FACTOR_NAMES)
    assert not any(name.endswith("_zscore") for name in PRICE_FORMATION_FACTOR_NAMES)


def test_feature_formulas_match_frozen_definitions() -> None:
    primitive = _primitive_fixture()
    features = build_price_formation_feature_frame(primitive)
    key = primitive.loc[0, ["symbol", "TradeDate"]]
    actual = features.loc[
        features["symbol"].eq(key["symbol"])
        & features["TradeDate"].eq(key["TradeDate"])
    ].iloc[0]
    expected = primitive.loc[0]
    assert actual["realized_volatility"] == pytest.approx(
        np.sqrt(expected["realized_variance"])
    )
    assert actual["morning_afternoon_amount_imbalance"] == pytest.approx(
        expected["morning_amount_share"]
        - expected["afternoon_amount_share"]
    )
    assert actual["intraday_max_drawdown"] == pytest.approx(
        expected["max_drawdown_intraday"]
    )
    assert actual["intraday_max_drawup"] == pytest.approx(
        expected["max_drawup"]
    )


def test_coverage_and_zero_amount_gate_factor_values() -> None:
    primitive = _primitive_fixture()
    low_coverage_key = primitive.loc[0, ["symbol", "TradeDate"]]
    zero_amount_key = primitive.loc[1, ["symbol", "TradeDate"]]
    valid_key = primitive.loc[2, ["symbol", "TradeDate"]]
    primitive.loc[0, "coverage_ratio"] = 0.79
    primitive.loc[1, "daily_amount"] = 0
    features = build_price_formation_feature_frame(primitive)

    def row_for(key: pd.Series) -> pd.Series:
        return features.loc[
            features["symbol"].eq(key["symbol"])
            & features["TradeDate"].eq(key["TradeDate"])
        ].iloc[0]

    factor_columns = list(PRICE_FORMATION_FACTOR_NAMES)
    assert row_for(low_coverage_key)[factor_columns].isna().all()
    assert row_for(zero_amount_key)[factor_columns].isna().all()
    assert row_for(valid_key)[factor_columns].notna().all()


def test_streaming_feature_calculation_matches_full_frame() -> None:
    primitive = _primitive_fixture().sort_values(
        ["TradeDate", "symbol"], kind="stable"
    )
    full = build_price_formation_feature_frame(primitive).sort_values(
        ["symbol", "TradeDate"]
    ).reset_index(drop=True)
    midpoint = primitive["TradeDate"].sort_values().unique()[2]
    first = primitive.loc[primitive["TradeDate"] < midpoint]
    second = primitive.loc[primitive["TradeDate"] >= midpoint]
    first_features, history = build_price_formation_feature_chunk(first)
    second_features, next_history = build_price_formation_feature_chunk(
        second, history
    )
    streamed = pd.concat(
        [first_features, second_features], ignore_index=True
    ).sort_values(["symbol", "TradeDate"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(streamed, full)
    assert history.empty
    assert next_history.empty


def test_narrow_output_has_frozen_schema_and_no_nonfinite_values() -> None:
    features = build_price_formation_feature_frame(_primitive_fixture())
    narrow = feature_to_narrow(features, "path_efficiency")
    assert list(narrow.columns) == [
        "symbol",
        "tradetime",
        "factorname",
        "value",
    ]
    assert narrow["factorname"].eq("path_efficiency").all()
    assert np.isfinite(narrow["value"]).all()
    assert narrow["tradetime"].dt.strftime("%H:%M").eq("09:30").all()


def test_registry_frame_is_stable_and_rejects_unknown_names() -> None:
    registry = registry_frame()
    assert registry["name"].tolist() == list(PRICE_FORMATION_FACTOR_NAMES)
    assert registry["name"].is_unique
    assert set(registry["category"]) == {
        "intraday_path",
        "realized_distribution",
        "volume_timing",
        "price_impact_efficiency",
    }
    with pytest.raises(KeyError, match="unknown"):
        registry_frame(["not_a_factor"])


def test_feature_layer_rejects_duplicate_symbol_day() -> None:
    primitive = _primitive_fixture()
    duplicated = pd.concat([primitive, primitive.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        build_price_formation_feature_frame(duplicated)


def test_backtest_records_raw_and_effective_direction_metadata() -> None:
    from l2_factor_reproduction.python.backtest import (
        backtest_factor,
        compute_rank_ic,
    )

    dates = pd.date_range("2024-06-03", periods=6, freq="B")
    symbols = [f"S{i:03d}" for i in range(40)]
    rng = np.random.default_rng(7)
    values = rng.normal(size=(len(dates), len(symbols)))
    rows = []
    for date_index, trade_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "tradetime": trade_date,
                    "factorname": "synthetic",
                    "value": values[date_index, symbol_index],
                }
            )
    narrow = pd.DataFrame(rows)
    mask = pd.DataFrame(1, index=dates, columns=symbols)
    ret = pd.DataFrame(
        values * 0.01 + rng.normal(scale=0.02, size=values.shape),
        index=dates,
        columns=symbols,
    )
    _, _, rank_ic_effective, summary = backtest_factor(
        narrow,
        mask=mask,
        ret_matrix=ret,
        signal_shift=1,
    )
    assert summary["group_pnl_saved_direction"] == "effective"
    assert summary["factor_direction"] in (-1, 1)
    assert np.isfinite(summary["rank_ic_mean_raw"])
    assert 0.0 <= summary["positive_ic_fraction_raw"] <= 1.0
    shifted = narrow.pivot_table(
        index="tradetime", columns="symbol", values="value"
    ).sort_index().shift(1).dropna(how="all")
    raw_ic = compute_rank_ic(
        shifted, ret.reindex(index=shifted.index, columns=shifted.columns)
    )
    expected_effective = raw_ic * summary["factor_direction"]
    left = rank_ic_effective.sort_index()
    right = expected_effective.sort_index()
    assert left.index.equals(pd.DatetimeIndex(right.index))
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy())
    assert summary["rank_ic_mean_raw"] == pytest.approx(float(raw_ic.mean()))
