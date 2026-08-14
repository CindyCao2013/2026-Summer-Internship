"""Frozen Price Formation Family v1 daily factor formulas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python.price_formation_daily import (
    COVERAGE_THRESHOLD,
)


@dataclass(frozen=True)
class PriceFormationFactorSpec:
    name: str
    formula: str
    category: str
    mechanism: str
    lookback_days: int
    signed: bool
    expected_redundancy: Optional[str] = None
    alias_exclusions: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _spec(
    name: str,
    formula: str,
    category: str,
    mechanism: str,
    *,
    signed: bool = True,
    redundancy: Optional[str] = None,
    aliases: Optional[str] = None,
) -> PriceFormationFactorSpec:
    return PriceFormationFactorSpec(
        name=name,
        formula=formula,
        category=category,
        mechanism=mechanism,
        lookback_days=1,
        signed=signed,
        expected_redundancy=redundancy,
        alias_exclusions=aliases,
    )


PRICE_FORMATION_FACTOR_SPECS: Dict[str, PriceFormationFactorSpec] = {
    "overnight_gap": _spec(
        "overnight_gap",
        "log(open_price / previous_available_continuous_close)",
        "intraday_path",
        "overnight_price_discovery",
    ),
    "open_to_30m_return": _spec(
        "open_to_30m_return",
        "log(close_minute_29 / open_minute_0)",
        "intraday_path",
        "opening_price_discovery",
    ),
    "morning_return": _spec(
        "morning_return",
        "log(morning_close / open_price)",
        "intraday_path",
        "morning_price_discovery",
    ),
    "afternoon_return": _spec(
        "afternoon_return",
        "log(continuous_close / afternoon_open)",
        "intraday_path",
        "afternoon_price_discovery",
    ),
    "closing_30m_return": _spec(
        "closing_30m_return",
        "log(continuous_close / opening_price_of_last_30_minutes)",
        "intraday_path",
        "late_session_price_discovery",
    ),
    "lunch_gap_return": _spec(
        "lunch_gap_return",
        "log(afternoon_open / morning_close)",
        "intraday_path",
        "lunch_information_gap",
    ),
    "close_auction_return": _spec(
        "close_auction_return",
        "log(close_auction_price / continuous_close)",
        "intraday_path",
        "closing_auction_price_discovery",
    ),
    "vwap_close_deviation": _spec(
        "vwap_close_deviation",
        "(continuous_close - daily_vwap) / daily_vwap",
        "intraday_path",
        "close_vs_traded_price_consensus",
        redundancy="may correlate with late-session return",
        aliases="rank and z-score clones excluded",
    ),
    "close_location_value": _spec(
        "close_location_value",
        "(2*continuous_close-high_price-low_price)/(high_price-low_price)",
        "intraday_path",
        "close_location_in_daily_range",
        aliases="synonymous close-position formulas excluded",
    ),
    "path_efficiency": _spec(
        "path_efficiency",
        "abs(continuous_close-open_price)/sum(abs(minute_price_change))",
        "intraday_path",
        "directional_path_efficiency",
        signed=False,
    ),
    "intraday_return_sign_persistence": _spec(
        "intraday_return_sign_persistence",
        "mean(sign(r_t)==sign(r_t-1))",
        "intraday_path",
        "minute_direction_persistence",
        signed=False,
    ),
    "minute_return_autocorr1": _spec(
        "minute_return_autocorr1",
        "corr(r_t,r_t-1)",
        "intraday_path",
        "short_horizon_return_dependence",
    ),
    "variance_ratio_5m": _spec(
        "variance_ratio_5m",
        "mean(r_5m^2)/(5*mean(r_1m^2))",
        "intraday_path",
        "five_minute_price_efficiency",
        signed=False,
    ),
    "realized_volatility": _spec(
        "realized_volatility",
        "sqrt(sum(r_t^2))",
        "realized_distribution",
        "intraday_realized_volatility",
        signed=False,
        aliases="realized_variance retained only as primitive, not candidate",
    ),
    "downside_semivariance_share": _spec(
        "downside_semivariance_share",
        "sum(r_t^2 if r_t<0)/sum(r_t^2)",
        "realized_distribution",
        "downside_variance_composition",
        signed=False,
    ),
    "realized_skewness": _spec(
        "realized_skewness",
        "sqrt(N)*sum(r_t^3)/(sum(r_t^2)^(3/2))",
        "realized_distribution",
        "intraday_return_asymmetry",
    ),
    "realized_kurtosis": _spec(
        "realized_kurtosis",
        "N*sum(r_t^4)/(sum(r_t^2)^2)",
        "realized_distribution",
        "intraday_tail_thickness",
        signed=False,
    ),
    "jump_share": _spec(
        "jump_share",
        "max(realized_variance-bipower_variation,0)/realized_variance",
        "realized_distribution",
        "discontinuous_price_variation",
        signed=False,
    ),
    "max_abs_minute_return": _spec(
        "max_abs_minute_return",
        "max(abs(r_t))",
        "realized_distribution",
        "largest_intraday_price_move",
        signed=False,
    ),
    "tail_return_share": _spec(
        "tail_return_share",
        "sum(r_t^2 where abs(r_t)>=daily_q95)/sum(r_t^2)",
        "realized_distribution",
        "tail_variance_concentration",
        signed=False,
    ),
    "intraday_max_drawdown": _spec(
        "intraday_max_drawdown",
        "max(1-price_t/running_max_price_t)",
        "realized_distribution",
        "worst_intraday_peak_to_trough",
        signed=False,
    ),
    "intraday_max_drawup": _spec(
        "intraday_max_drawup",
        "max(price_t/running_min_price_t-1)",
        "realized_distribution",
        "largest_intraday_trough_to_peak",
        signed=False,
    ),
    "opening_amount_share": _spec(
        "opening_amount_share",
        "sum(amount_t,minute_index<30)/daily_amount",
        "volume_timing",
        "opening_trading_intensity",
        signed=False,
    ),
    "closing_amount_share": _spec(
        "closing_amount_share",
        "sum(amount_t,minute_index>=210)/daily_amount",
        "volume_timing",
        "closing_trading_intensity",
        signed=False,
    ),
    "morning_afternoon_amount_imbalance": _spec(
        "morning_afternoon_amount_imbalance",
        "morning_amount_share-afternoon_amount_share",
        "volume_timing",
        "session_trading_intensity_rotation",
        redundancy="deterministic combination of two primitive shares",
    ),
    "volume_concentration_hhi": _spec(
        "volume_concentration_hhi",
        "sum((amount_t/daily_amount)^2)",
        "volume_timing",
        "intraday_amount_concentration",
        signed=False,
    ),
    "amount_time_center": _spec(
        "amount_time_center",
        "sum(minute_index*amount_t)/(239*sum(amount_t))",
        "volume_timing",
        "normalized_trading_time_center",
        signed=False,
    ),
    "volume_return_corr": _spec(
        "volume_return_corr",
        "corr(volume_t,r_t)",
        "volume_timing",
        "signed_return_volume_alignment",
    ),
    "volume_abs_return_corr": _spec(
        "volume_abs_return_corr",
        "corr(volume_t,abs(r_t))",
        "volume_timing",
        "volatility_volume_alignment",
    ),
    "intraday_amihud": _spec(
        "intraday_amihud",
        "mean(abs(r_t)/amount_t for amount_t>0)",
        "price_impact_efficiency",
        "minute_price_impact_per_currency",
        signed=False,
    ),
    "return_per_amount": _spec(
        "return_per_amount",
        "open_to_close_return/daily_amount",
        "price_impact_efficiency",
        "signed_daily_return_per_currency",
        aliases="monotonic rescalings excluded",
    ),
    "range_per_amount": _spec(
        "range_per_amount",
        "((high_price-low_price)/open_price)/daily_amount",
        "price_impact_efficiency",
        "daily_range_per_currency",
        signed=False,
    ),
}

PRICE_FORMATION_FACTOR_NAMES = tuple(PRICE_FORMATION_FACTOR_SPECS)

DIRECT_SOURCES = {
    "overnight_gap": "overnight_gap",
    "open_to_30m_return": "open_to_30m_return",
    "morning_return": "morning_return",
    "afternoon_return": "afternoon_return",
    "closing_30m_return": "closing_30m_return",
    "lunch_gap_return": "lunch_gap_return",
    "close_auction_return": "close_auction_return",
    "vwap_close_deviation": "vwap_close_deviation",
    "close_location_value": "close_location_value",
    "path_efficiency": "path_efficiency",
    "intraday_return_sign_persistence": "intraday_return_sign_persistence",
    "minute_return_autocorr1": "minute_return_autocorr1",
    "variance_ratio_5m": "variance_ratio_5m",
    "downside_semivariance_share": "downside_semivariance_share",
    "realized_skewness": "realized_skewness",
    "realized_kurtosis": "realized_kurtosis",
    "jump_share": "jump_share",
    "max_abs_minute_return": "max_abs_minute_return",
    "tail_return_share": "tail_return_share",
    "intraday_max_drawdown": "max_drawdown_intraday",
    "intraday_max_drawup": "max_drawup",
    "opening_amount_share": "opening_30m_amount_share",
    "closing_amount_share": "closing_30m_amount_share",
    "volume_concentration_hhi": "volume_concentration_hhi",
    "amount_time_center": "amount_time_center",
    "volume_return_corr": "volume_return_corr",
    "volume_abs_return_corr": "volume_abs_return_corr",
    "intraday_amihud": "intraday_amihud",
    "return_per_amount": "return_per_amount",
    "range_per_amount": "range_per_amount",
}

REQUIRED_PRIMITIVE_COLUMNS = tuple(
    dict.fromkeys(
        [
            "symbol",
            "TradeDate",
            "coverage_ratio",
            "daily_amount",
            "realized_variance",
            "morning_amount_share",
            "afternoon_amount_share",
            *DIRECT_SOURCES.values(),
        ]
    )
)


def prepare_price_formation_feature_primitive(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(
        set(REQUIRED_PRIMITIVE_COLUMNS).difference(primitive.columns)
    )
    if missing:
        raise ValueError(
            f"price-formation primitive missing columns: {missing}"
        )
    frame = primitive.loc[:, REQUIRED_PRIMITIVE_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
    for column in REQUIRED_PRIMITIVE_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "TradeDate"], kind="stable")
    if frame.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError(
            "price-formation primitive must be unique at symbol-day"
        )
    return frame.reset_index(drop=True)


def _validate_feature_ranges(features: pd.DataFrame) -> None:
    bounded_signed = ("close_location_value",)
    bounded_zero_one = (
        "path_efficiency",
        "intraday_return_sign_persistence",
        "downside_semivariance_share",
        "jump_share",
        "tail_return_share",
        "opening_amount_share",
        "closing_amount_share",
        "volume_concentration_hhi",
        "amount_time_center",
    )
    nonnegative = (
        "variance_ratio_5m",
        "realized_volatility",
        "realized_kurtosis",
        "max_abs_minute_return",
        "intraday_max_drawdown",
        "intraday_max_drawup",
        "intraday_amihud",
        "range_per_amount",
    )
    for name in bounded_signed:
        if (features[name].dropna().abs() > 1.0 + 1e-10).any():
            raise ValueError(f"{name} outside [-1, 1]")
    for name in bounded_zero_one:
        values = features[name].dropna()
        if not values.between(-1e-10, 1.0 + 1e-10).all():
            raise ValueError(f"{name} outside [0, 1]")
    for name in nonnegative:
        if (features[name].dropna() < -1e-12).any():
            raise ValueError(f"{name} contains negative values")
    values = features.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Price Formation features contain infinite values")


def build_price_formation_feature_frame(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    frame = prepare_price_formation_feature_primitive(primitive)
    eligible = (
        frame["coverage_ratio"].ge(COVERAGE_THRESHOLD)
        & frame["daily_amount"].gt(0)
    )
    features = pd.DataFrame(index=frame.index)
    for factor, source in DIRECT_SOURCES.items():
        features[factor] = frame[source]
    features["realized_volatility"] = np.sqrt(
        frame["realized_variance"].clip(lower=0)
    )
    features["morning_afternoon_amount_imbalance"] = (
        frame["morning_amount_share"] - frame["afternoon_amount_share"]
    )
    features = features.loc[:, PRICE_FORMATION_FACTOR_NAMES]
    features = features.where(eligible).replace([np.inf, -np.inf], np.nan)
    _validate_feature_ranges(features)
    return pd.concat(
        [
            frame[["symbol", "TradeDate"]].reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )


def build_price_formation_feature_chunk(
    primitive: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a partition; level-only v1 has no rolling state."""
    if history is not None and not history.empty:
        expected = {"symbol", "TradeDate"}
        if not expected.issubset(history.columns):
            raise ValueError("unexpected Price Formation streaming history")
    features = build_price_formation_feature_frame(primitive)
    return features, pd.DataFrame(columns=["symbol", "TradeDate"])


def feature_to_narrow(
    feature_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    if factor_name not in PRICE_FORMATION_FACTOR_SPECS:
        raise KeyError(f"unknown Price Formation factor: {factor_name}")
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
        list(PRICE_FORMATION_FACTOR_NAMES) if names is None else list(names)
    )
    unknown = sorted(
        set(selected).difference(PRICE_FORMATION_FACTOR_SPECS)
    )
    if unknown:
        raise KeyError(f"unknown Price Formation factors: {unknown}")
    return pd.DataFrame(
        [PRICE_FORMATION_FACTOR_SPECS[name].to_dict() for name in selected]
    )
