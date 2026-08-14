"""Formula and no-lookahead tests for the Trade Flow Family."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.trade_flow_factors import (  # noqa: E402
    TRADE_FLOW_FACTOR_NAMES,
    build_trade_flow_feature_frame,
    feature_to_narrow,
)


def _fixture() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=23)
    for symbol, offset in [("600000.SH", 0), ("000001.SZ", 20)]:
        for i, date in enumerate(dates):
            buy_amt = 100.0 + offset + i
            sell_amt = 80.0 + offset
            rows.append(
                {
                    "symbol": symbol,
                    "TradeDate": date,
                    "active_buy_amt": buy_amt,
                    "active_sell_amt": sell_amt,
                    "total_amt": buy_amt + sell_amt,
                    "active_buy_cnt": 10 + i % 3,
                    "active_sell_cnt": 12,
                    "trade_cnt": 22 + i % 3,
                }
            )
    # Prefix filter must remove this ETF-like symbol.
    rows.append(
        {
            "symbol": "510300.SH",
            "TradeDate": dates[0],
            "active_buy_amt": 100.0,
            "active_sell_amt": 100.0,
            "total_amt": 200.0,
            "active_buy_cnt": 10,
            "active_sell_cnt": 10,
            "trade_cnt": 20,
        }
    )
    out = pd.DataFrame(rows)
    # Reproduce ClickHouse countIf parquet dtypes.
    out["active_buy_cnt"] = out["active_buy_cnt"].astype("uint64")
    out["active_sell_cnt"] = out["active_sell_cnt"].astype("uint64")
    return out


def test_trade_flow_formula_identities() -> None:
    features = build_trade_flow_feature_frame(_fixture())
    assert set(TRADE_FLOW_FACTOR_NAMES).issubset(features.columns)
    assert set(features["symbol"]) == {"600000.SH", "000001.SZ"}

    first = features.loc[features["symbol"] == "600000.SH"].iloc[0]
    expected_nbr = (100.0 - 80.0) / 180.0
    expected_count = (10.0 - 12.0) / 22.0
    assert np.isclose(first["net_buy_ratio"], expected_nbr)
    assert np.isclose(first["buy_dominance"], (1.0 + expected_nbr) / 2.0)
    assert np.isclose(first["net_buy_count_ratio"], expected_count)
    assert np.isclose(first["avg_buy_trade_size"], 10.0)
    assert np.isclose(first["avg_sell_trade_size"], 80.0 / 12.0)
    assert np.isclose(first["trade_size_asymmetry"], 1.5)
    assert np.isclose(
        first["flow_concentration"],
        0.5 * (abs(expected_nbr) + abs(expected_count)),
    )
    assert features["net_buy_count_ratio"].between(-1.0, 1.0).all()


def test_rolling_features_use_only_past_observations() -> None:
    raw = _fixture()
    features = build_trade_flow_feature_frame(raw)
    one = features.loc[features["symbol"] == "600000.SH"].reset_index(drop=True)

    # Twenty strictly prior observations are required; index 20 is the first.
    assert one.loc[:19, "flow_zscore_20d"].isna().all()
    assert pd.notna(one.loc[20, "flow_zscore_20d"])
    assert one.loc[0, "flow_acceleration"] != one.loc[0, "flow_acceleration"]
    assert np.isclose(
        one.loc[1, "flow_acceleration"],
        one.loc[1, "net_buy_ratio"] - one.loc[0, "net_buy_ratio"],
    )

    # Mutating a future row cannot change a historical z-score.
    mutated = raw.copy()
    mask = (
        (mutated["symbol"] == "600000.SH")
        & (mutated["TradeDate"] == mutated["TradeDate"].max())
    )
    mutated.loc[mask, "active_buy_amt"] *= 100
    mutated.loc[mask, "total_amt"] = (
        mutated.loc[mask, "active_buy_amt"]
        + mutated.loc[mask, "active_sell_amt"]
    )
    changed = build_trade_flow_feature_frame(mutated)
    changed_one = changed.loc[
        changed["symbol"] == "600000.SH"
    ].reset_index(drop=True)
    assert np.isclose(
        one.loc[20, "flow_zscore_20d"],
        changed_one.loc[20, "flow_zscore_20d"],
    )


def test_narrow_schema() -> None:
    features = build_trade_flow_feature_frame(_fixture())
    narrow = feature_to_narrow(features, "net_buy_ratio")
    assert list(narrow.columns) == [
        "symbol",
        "tradetime",
        "factorname",
        "value",
    ]
    assert narrow["factorname"].eq("net_buy_ratio").all()

