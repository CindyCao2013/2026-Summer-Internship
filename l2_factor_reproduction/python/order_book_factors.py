"""Frozen Order Book Family v1 daily factor formulas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python.ch_order_book import COVERAGE_THRESHOLD


@dataclass(frozen=True)
class OrderBookFactorSpec:
    name: str
    formula: str
    category: str
    mechanism: str
    lookback_days: int
    signed: bool
    expected_redundancy: Optional[str] = None
    alias_features: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _spec(
    name: str,
    formula: str,
    category: str,
    mechanism: str,
    *,
    lookback: int = 1,
    signed: bool = True,
    redundancy: Optional[str] = None,
    aliases: Optional[str] = None,
) -> OrderBookFactorSpec:
    return OrderBookFactorSpec(
        name=name,
        formula=formula,
        category=category,
        mechanism=mechanism,
        lookback_days=lookback,
        signed=signed,
        expected_redundancy=redundancy,
        alias_features=aliases,
    )


ORDER_BOOK_FACTOR_SPECS: Dict[str, OrderBookFactorSpec] = {
    "obi_l1_mean": _spec(
        "obi_l1_mean",
        "obi_1_mean",
        "level_imbalance",
        "top_book_supply_demand",
        aliases="log(bid_depth_1/ask_depth_1) is a monotonic alias",
    ),
    "obi_l5_mean": _spec(
        "obi_l5_mean",
        "obi_5_mean",
        "level_imbalance",
        "near_book_supply_demand",
        redundancy="expected correlation with OBI depth variants",
        aliases="log(bid_depth_5/ask_depth_5) is a monotonic alias",
    ),
    "obi_l10_mean": _spec(
        "obi_l10_mean",
        "obi_10_mean",
        "level_imbalance",
        "full_book_supply_demand",
        redundancy="expected correlation with OBI depth variants",
        aliases="log(bid_depth_10/ask_depth_10) is a monotonic alias",
    ),
    "weighted_obi_mean": _spec(
        "weighted_obi_mean",
        "weighted_obi_mean with w_k=1/k",
        "level_imbalance",
        "distance_weighted_supply_demand",
        redundancy="expected correlation with OBI depth variants",
    ),
    "obi_l1_volatility": _spec(
        "obi_l1_volatility",
        "obi_1_std",
        "level_imbalance",
        "top_book_instability",
        signed=False,
    ),
    "obi_l5_volatility": _spec(
        "obi_l5_volatility",
        "obi_5_std",
        "level_imbalance",
        "near_book_instability",
        signed=False,
        redundancy="volatility companion of OBI level",
    ),
    "weighted_obi_volatility": _spec(
        "weighted_obi_volatility",
        "weighted_obi_std",
        "level_imbalance",
        "weighted_book_instability",
        signed=False,
        redundancy="volatility companion of weighted OBI level",
    ),
    "near_far_imbalance": _spec(
        "near_far_imbalance",
        "near_far_imbalance_mean",
        "book_shape",
        "near_vs_far_depth_shape",
    ),
    "bid_depth_concentration": _spec(
        "bid_depth_concentration",
        "bid_depth_hhi_mean",
        "book_shape",
        "bid_depth_concentration",
        signed=False,
    ),
    "ask_depth_concentration": _spec(
        "ask_depth_concentration",
        "ask_depth_hhi_mean",
        "book_shape",
        "ask_depth_concentration",
        signed=False,
    ),
    "depth_concentration_asymmetry": _spec(
        "depth_concentration_asymmetry",
        "bid_depth_hhi_mean-ask_depth_hhi_mean",
        "book_shape",
        "depth_concentration_asymmetry",
        redundancy="deterministic linear combination of bid/ask HHI",
    ),
    "bid_depth_slope": _spec(
        "bid_depth_slope",
        "bid_depth_slope_mean",
        "book_shape",
        "normalized_bid_depth_gradient",
        signed=False,
    ),
    "ask_depth_slope": _spec(
        "ask_depth_slope",
        "ask_depth_slope_mean",
        "book_shape",
        "normalized_ask_depth_gradient",
        signed=False,
    ),
    "depth_slope_asymmetry": _spec(
        "depth_slope_asymmetry",
        "bid_depth_slope_mean-ask_depth_slope_mean",
        "book_shape",
        "depth_gradient_asymmetry",
        redundancy="deterministic linear combination of bid/ask slope",
    ),
    "relative_spread_mean": _spec(
        "relative_spread_mean",
        "relative_spread_mean",
        "spread_price_pressure",
        "quoted_liquidity_cost",
        signed=False,
    ),
    "relative_spread_volatility": _spec(
        "relative_spread_volatility",
        "relative_spread_std",
        "spread_price_pressure",
        "spread_instability",
        signed=False,
        redundancy="volatility companion of relative spread level",
    ),
    "microprice_deviation_mean": _spec(
        "microprice_deviation_mean",
        "microprice_deviation_mean",
        "spread_price_pressure",
        "top_book_price_pressure",
    ),
    "microprice_deviation_volatility": _spec(
        "microprice_deviation_volatility",
        "microprice_deviation_std",
        "spread_price_pressure",
        "microprice_instability",
        signed=False,
        redundancy="volatility companion of microprice pressure",
    ),
    "book_vwap_gap": _spec(
        "book_vwap_gap",
        "book_vwap_gap_mean from self-computed ten-level VWAP",
        "spread_price_pressure",
        "full_book_liquidity_gap",
        signed=False,
    ),
    "total_depth_level": _spec(
        "total_depth_level",
        "log_total_depth_mean",
        "spread_price_pressure",
        "displayed_liquidity_level",
        signed=False,
    ),
    "total_depth_volatility": _spec(
        "total_depth_volatility",
        "log_total_depth_std",
        "spread_price_pressure",
        "displayed_liquidity_instability",
        signed=False,
        redundancy="volatility companion of total depth level",
    ),
    "opening_obi_l5": _spec(
        "opening_obi_l5",
        "opening_30m_obi_5",
        "intraday_timing",
        "opening_supply_demand",
    ),
    "closing_obi_l5": _spec(
        "closing_obi_l5",
        "closing_30m_obi_5",
        "intraday_timing",
        "closing_supply_demand",
    ),
    "opening_closing_obi_change": _spec(
        "opening_closing_obi_change",
        "closing_30m_obi_5-opening_30m_obi_5",
        "intraday_timing",
        "supply_demand_rotation",
        redundancy="deterministic combination of opening and closing OBI",
    ),
    "opening_closing_spread_change": _spec(
        "opening_closing_spread_change",
        "closing_30m_relative_spread-opening_30m_relative_spread",
        "intraday_timing",
        "liquidity_cost_rotation",
    ),
    "opening_closing_depth_change": _spec(
        "opening_closing_depth_change",
        "closing_30m_log_depth-opening_30m_log_depth",
        "intraday_timing",
        "displayed_liquidity_rotation",
    ),
    "obi_intraday_slope": _spec(
        "obi_intraday_slope",
        "obi_5_intraday_slope",
        "intraday_timing",
        "supply_demand_trend",
    ),
    "obi_sign_persistence": _spec(
        "obi_sign_persistence",
        "obi_5_sign_persistence",
        "intraday_timing",
        "supply_demand_sign_persistence",
        signed=False,
    ),
    "obi_shock_20d": _spec(
        "obi_shock_20d",
        "inclusive rolling_zscore_20d(obi_5_mean)",
        "dynamic_shock",
        "supply_demand_state_shock",
        lookback=20,
    ),
    "spread_shock_20d": _spec(
        "spread_shock_20d",
        "inclusive rolling_zscore_20d(relative_spread_mean)",
        "dynamic_shock",
        "liquidity_cost_shock",
        lookback=20,
    ),
    "depth_shock_20d": _spec(
        "depth_shock_20d",
        "inclusive rolling_zscore_20d(log_total_depth_mean)",
        "dynamic_shock",
        "displayed_liquidity_shock",
        lookback=20,
    ),
    "microprice_shock_20d": _spec(
        "microprice_shock_20d",
        "inclusive rolling_zscore_20d(microprice_deviation_mean)",
        "dynamic_shock",
        "price_pressure_shock",
        lookback=20,
    ),
}

ORDER_BOOK_FACTOR_NAMES = tuple(ORDER_BOOK_FACTOR_SPECS)
STATIC_FACTOR_NAMES = ORDER_BOOK_FACTOR_NAMES[:28]
SHOCK_SOURCES = {
    "obi_shock_20d": "obi_5_mean",
    "spread_shock_20d": "relative_spread_mean",
    "depth_shock_20d": "log_total_depth_mean",
    "microprice_shock_20d": "microprice_deviation_mean",
}
SHOCK_HISTORY_LENGTH = 19

STATIC_SOURCE_COLUMNS = {
    "obi_l1_mean": "obi_1_mean",
    "obi_l5_mean": "obi_5_mean",
    "obi_l10_mean": "obi_10_mean",
    "weighted_obi_mean": "weighted_obi_mean",
    "obi_l1_volatility": "obi_1_std",
    "obi_l5_volatility": "obi_5_std",
    "weighted_obi_volatility": "weighted_obi_std",
    "near_far_imbalance": "near_far_imbalance_mean",
    "bid_depth_concentration": "bid_depth_hhi_mean",
    "ask_depth_concentration": "ask_depth_hhi_mean",
    "depth_concentration_asymmetry": (
        "depth_concentration_asymmetry_mean"
    ),
    "bid_depth_slope": "bid_depth_slope_mean",
    "ask_depth_slope": "ask_depth_slope_mean",
    "depth_slope_asymmetry": "depth_slope_asymmetry_mean",
    "relative_spread_mean": "relative_spread_mean",
    "relative_spread_volatility": "relative_spread_std",
    "microprice_deviation_mean": "microprice_deviation_mean",
    "microprice_deviation_volatility": "microprice_deviation_std",
    "book_vwap_gap": "book_vwap_gap_mean",
    "total_depth_level": "log_total_depth_mean",
    "total_depth_volatility": "log_total_depth_std",
    "opening_obi_l5": "opening_30m_obi_5",
    "closing_obi_l5": "closing_30m_obi_5",
    "obi_intraday_slope": "obi_5_intraday_slope",
    "obi_sign_persistence": "obi_5_sign_persistence",
}

REQUIRED_PRIMITIVE_COLUMNS = tuple(
    dict.fromkeys(
        [
            "symbol",
            "TradeDate",
            "coverage_ratio",
            *STATIC_SOURCE_COLUMNS.values(),
            "opening_30m_obi_5",
            "closing_30m_obi_5",
            "opening_30m_relative_spread",
            "closing_30m_relative_spread",
            "opening_30m_log_depth",
            "closing_30m_log_depth",
            *SHOCK_SOURCES.values(),
        ]
    )
)


def _safe_zscore(
    values: pd.Series,
    symbols: pd.Series,
    window: int = 20,
) -> pd.Series:
    grouped = values.groupby(symbols, sort=False)
    mean = grouped.transform(
        lambda series: series.rolling(
            window, min_periods=window
        ).mean()
    )
    std = grouped.transform(
        lambda series: series.rolling(
            window, min_periods=window
        ).std(ddof=0)
    )
    result = (values - mean) / std.replace(0, np.nan)
    return result.where(np.isfinite(result))


def prepare_order_book_feature_primitive(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(
        set(REQUIRED_PRIMITIVE_COLUMNS).difference(primitive.columns)
    )
    if missing:
        raise ValueError(f"order-book primitive missing columns: {missing}")
    frame = primitive.loc[:, REQUIRED_PRIMITIVE_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
    for column in REQUIRED_PRIMITIVE_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "TradeDate"], kind="stable")
    if frame.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError("order-book primitive must be unique at symbol-day")
    return frame.reset_index(drop=True)


def _static_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for factor, source in STATIC_SOURCE_COLUMNS.items():
        features[factor] = frame[source]
    features["opening_closing_obi_change"] = (
        frame["closing_30m_obi_5"] - frame["opening_30m_obi_5"]
    )
    features["opening_closing_spread_change"] = (
        frame["closing_30m_relative_spread"]
        - frame["opening_30m_relative_spread"]
    )
    features["opening_closing_depth_change"] = (
        frame["closing_30m_log_depth"] - frame["opening_30m_log_depth"]
    )
    return features.loc[:, STATIC_FACTOR_NAMES]


def _validate_feature_ranges(features: pd.DataFrame) -> None:
    signed_bounded = (
        "obi_l1_mean",
        "obi_l5_mean",
        "obi_l10_mean",
        "weighted_obi_mean",
        "near_far_imbalance",
    )
    zero_one = (
        "bid_depth_concentration",
        "ask_depth_concentration",
        "obi_sign_persistence",
    )
    nonnegative = (
        "obi_l1_volatility",
        "obi_l5_volatility",
        "weighted_obi_volatility",
        "relative_spread_mean",
        "relative_spread_volatility",
        "microprice_deviation_volatility",
        "book_vwap_gap",
        "total_depth_volatility",
        "bid_depth_slope",
        "ask_depth_slope",
    )
    for factor in signed_bounded:
        if (features[factor].dropna().abs() > 1.0 + 1e-10).any():
            raise ValueError(f"{factor} outside [-1, 1]")
    for factor in zero_one:
        values = features[factor].dropna()
        if not values.between(-1e-10, 1.0 + 1e-10).all():
            raise ValueError(f"{factor} outside [0, 1]")
    for factor in nonnegative:
        if (features[factor].dropna() < -1e-10).any():
            raise ValueError(f"{factor} contains negative values")


def build_order_book_feature_frame(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    """Build all 32 factors in-memory; intended for tests and short windows."""
    frame = prepare_order_book_feature_primitive(primitive)
    eligible = frame["coverage_ratio"] >= COVERAGE_THRESHOLD
    features = _static_features(frame)
    for factor, source in SHOCK_SOURCES.items():
        values = frame[source].where(eligible)
        features[factor] = _safe_zscore(values, frame["symbol"])
    features = features.where(eligible).replace([np.inf, -np.inf], np.nan)
    _validate_feature_ranges(features)
    return pd.concat(
        [
            frame[["symbol", "TradeDate"]].reset_index(drop=True),
            features.loc[:, ORDER_BOOK_FACTOR_NAMES].reset_index(drop=True),
        ],
        axis=1,
    )


def build_order_book_feature_chunk(
    primitive: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one chronological chunk and return rolling state for the next."""
    frame = prepare_order_book_feature_primitive(primitive)
    eligible = frame["coverage_ratio"] >= COVERAGE_THRESHOLD
    current_bases = frame[
        ["symbol", "TradeDate", *SHOCK_SOURCES.values()]
    ].copy()
    current_bases.loc[
        ~eligible, list(SHOCK_SOURCES.values())
    ] = np.nan
    current_bases["_is_current"] = True
    current_bases["_row_order"] = np.arange(len(current_bases))
    if history is None or history.empty:
        combined = current_bases.copy()
    else:
        previous = history.copy()
        previous["_is_current"] = False
        previous["_row_order"] = -1
        combined = pd.concat([previous, current_bases], ignore_index=True)
    combined = combined.sort_values(
        ["symbol", "TradeDate", "_is_current"],
        kind="stable",
    ).reset_index(drop=True)

    features = _static_features(frame)
    for factor, source in SHOCK_SOURCES.items():
        combined[factor] = _safe_zscore(
            combined[source], combined["symbol"]
        )
        current_result = combined.loc[
            combined["_is_current"], ["_row_order", factor]
        ].sort_values("_row_order")
        features[factor] = current_result[factor].to_numpy()
    features = features.where(eligible).replace([np.inf, -np.inf], np.nan)
    _validate_feature_ranges(features)

    next_history = (
        combined.groupby("symbol", sort=False, group_keys=False)
        .tail(SHOCK_HISTORY_LENGTH)[
            ["symbol", "TradeDate", *SHOCK_SOURCES.values()]
        ]
        .reset_index(drop=True)
    )
    output = pd.concat(
        [
            frame[["symbol", "TradeDate"]].reset_index(drop=True),
            features.loc[:, ORDER_BOOK_FACTOR_NAMES].reset_index(drop=True),
        ],
        axis=1,
    )
    return output, next_history


def feature_to_narrow(
    feature_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    if factor_name not in ORDER_BOOK_FACTOR_SPECS:
        raise KeyError(f"unknown Order Book factor: {factor_name}")
    out = feature_frame[
        ["symbol", "TradeDate", factor_name]
    ].rename(columns={factor_name: "value"})
    out["tradetime"] = pd.to_datetime(out.pop("TradeDate")) + pd.Timedelta(
        hours=9, minutes=30
    )
    out["factorname"] = factor_name
    return (
        out[["symbol", "tradetime", "factorname", "value"]]
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )


def registry_frame(
    names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected: List[str] = (
        list(ORDER_BOOK_FACTOR_NAMES) if names is None else list(names)
    )
    unknown = sorted(set(selected).difference(ORDER_BOOK_FACTOR_SPECS))
    if unknown:
        raise KeyError(f"unknown Order Book factors: {unknown}")
    return pd.DataFrame(
        [ORDER_BOOK_FACTOR_SPECS[name].to_dict() for name in selected]
    )
