"""Formula, boundary, and no-lookahead tests for Order Size Family v1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.order_size_factors import (  # noqa: E402
    ORDER_SIZE_FACTOR_NAMES,
    build_order_size_feature_frame,
    feature_to_narrow,
)


def _fixture() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=23)
    for symbol, offset in [("600000.SH", 0), ("000001.SZ", 2)]:
        for i, date in enumerate(dates):
            delta = i + offset
            c20 = 600.0 - delta
            buy_c20 = 330.0 - delta / 2.0
            sell_c20 = 270.0 - delta / 2.0
            rows.append(
                {
                    "symbol": symbol,
                    "TradeDate": date,
                    "total_amt": 1000.0,
                    "trade_cnt": 100,
                    "active_buy_amt": 520.0,
                    "active_sell_amt": 480.0,
                    "cum_amt_10000": 100.0,
                    "cum_cnt_10000": 20,
                    "buy_cum_amt_10000": 60.0,
                    "sell_cum_amt_10000": 40.0,
                    "cum_amt_40000": 250.0,
                    "cum_cnt_40000": 40,
                    "buy_cum_amt_40000": 140.0,
                    "sell_cum_amt_40000": 110.0,
                    "cum_amt_50000": 300.0,
                    "cum_cnt_50000": 50,
                    "buy_cum_amt_50000": 170.0,
                    "sell_cum_amt_50000": 130.0,
                    "cum_amt_200000": c20,
                    "cum_cnt_200000": 80,
                    "buy_cum_amt_200000": buy_c20,
                    "sell_cum_amt_200000": sell_c20,
                    "cum_amt_1000000": 850.0,
                    "cum_cnt_1000000": 95,
                    "buy_cum_amt_1000000": 450.0,
                    "sell_cum_amt_1000000": 400.0,
                }
            )
    # Non-equity prefix must be removed before factor export.
    extra = dict(rows[0])
    extra["symbol"] = "510300.SH"
    rows.append(extra)
    frame = pd.DataFrame(rows)
    for column in [
        "trade_cnt",
        "cum_cnt_10000",
        "cum_cnt_40000",
        "cum_cnt_50000",
        "cum_cnt_200000",
        "cum_cnt_1000000",
    ]:
        frame[column] = frame[column].astype("uint64")
    return frame


def test_order_size_formula_values_and_ranges() -> None:
    features = build_order_size_feature_frame(_fixture())
    assert set(ORDER_SIZE_FACTOR_NAMES).issubset(features.columns)
    assert set(features["symbol"]) == {"600000.SH", "000001.SZ"}
    first = features.loc[features["symbol"] == "600000.SH"].iloc[0]

    shares = np.array([0.10, 0.20, 0.30, 0.25, 0.15])
    expected_entropy = float(
        -(shares * np.log(shares)).sum() / np.log(5)
    )
    assert np.isclose(first["small_order_ratio_1w"], 0.10)
    assert np.isclose(first["small_order_ratio_4w"], 0.25)
    assert np.isclose(first["mid_order_ratio_4w_20w"], 0.35)
    assert np.isclose(first["mid_order_ratio_5w_20w"], 0.30)
    assert np.isclose(first["large_order_ratio_20w"], 0.40)
    assert np.isclose(first["super_large_order_ratio_100w"], 0.15)
    assert np.isclose(first["large_small_spread"], 0.30)
    assert np.isclose(first["order_size_entropy"], expected_entropy)
    assert np.isclose(first["order_size_concentration"], 0.225)
    assert np.isclose(first["order_size_tail_share"], 0.25)
    assert np.isclose(first["small_order_pressure"], 0.02)
    assert np.isclose(first["mid_order_pressure"], 0.03)
    assert np.isclose(first["large_order_pressure"], -0.02)
    assert np.isclose(first["super_large_order_pressure"], -0.01)
    assert np.isclose(first["buy_large_order_ratio"], 190.0 / 520.0)
    assert np.isclose(first["sell_large_order_ratio"], 210.0 / 480.0)
    assert np.isclose(first["small_order_direction"], 0.20)
    assert np.isclose(first["large_order_direction"], -0.05)


def test_order_size_shocks_use_strictly_prior_window() -> None:
    raw = _fixture()
    features = build_order_size_feature_frame(raw)
    one = features.loc[
        features["symbol"] == "600000.SH"
    ].reset_index(drop=True)
    for name in [
        "large_order_shock_20d",
        "order_size_entropy_shock_20d",
    ]:
        assert one.loc[:19, name].isna().all()
        assert pd.notna(one.loc[20, name])

    future_date = raw.loc[
        raw["symbol"] == "600000.SH", "TradeDate"
    ].max()
    mutated = raw.copy()
    mask = (
        (mutated["symbol"] == "600000.SH")
        & (mutated["TradeDate"] == future_date)
    )
    mutated.loc[mask, "cum_amt_200000"] -= 100
    mutated.loc[mask, "buy_cum_amt_200000"] -= 50
    mutated.loc[mask, "sell_cum_amt_200000"] -= 50
    changed = build_order_size_feature_frame(mutated)
    changed_one = changed.loc[
        changed["symbol"] == "600000.SH"
    ].reset_index(drop=True)
    assert np.isclose(
        one.loc[20, "large_order_shock_20d"],
        changed_one.loc[20, "large_order_shock_20d"],
    )


def test_order_size_narrow_schema() -> None:
    features = build_order_size_feature_frame(_fixture())
    narrow = feature_to_narrow(
        features, "mid_order_ratio_4w_20w"
    )
    assert list(narrow.columns) == [
        "symbol",
        "tradetime",
        "factorname",
        "value",
    ]
    assert narrow["factorname"].eq(
        "mid_order_ratio_4w_20w"
    ).all()

