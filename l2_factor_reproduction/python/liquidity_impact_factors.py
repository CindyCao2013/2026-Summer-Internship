"""Frozen Liquidity / Price Impact candidate formulas (Sprint 7, 24 formulas).

Every candidate is a deterministic function of the frozen
``l2_primitive_liquidity_impact_daily`` fields. No parameter grids, no
window search, no direction search. Effective direction is display-only and
never drives production decisions.

Duplication guard (verified against frozen registries): none of these
candidates re-register relative_spread_mean, total_depth_level,
intraday_amihud, range_per_amount, return_per_amount, net_buy_ratio,
order-size ratios, OBI level, or realized volatility. All candidates are
joint trade x quote / trade x depth / signed flow x forward price /
impact x recovery mechanisms.

Proxy disclosure: effective_spread_proxy and realized_spread_proxy_5m are
minute-approximation proxies (minute signed direction + minute-last
midquote), not exact prevailing-quote per-trade effective spreads. The
size-conditioned impacts bucket trades by amount at tick level, but the
forward response uses the minute-level forward mid return; this is a
documented minute approximation, so *_trade_impact fields are interpreted
as marginal-impact estimates, not exact per-trade measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.python.liquidity_impact_daily import (
    COVERAGE_THRESHOLD,
    EXPECTED_CONTINUOUS_MINUTES,
)


@dataclass(frozen=True)
class LiquidityImpactFactorSpec:
    name: str
    formula: str
    category: str
    mechanism: str
    lookback_days: int
    signed: bool
    expected_redundancy: Optional[str] = None
    proxy_note: Optional[str] = None

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
    proxy: Optional[str] = None,
) -> LiquidityImpactFactorSpec:
    return LiquidityImpactFactorSpec(
        name=name,
        formula=formula,
        category=category,
        mechanism=mechanism,
        lookback_days=1,
        signed=signed,
        expected_redundancy=redundancy,
        proxy_note=proxy,
    )


A = "spread_depth_liquidity"
B = "signed_price_impact"
C = "temporary_permanent_impact"
D = "size_conditioned_impact"
E = "resilience_recovery"

LIQUIDITY_IMPACT_FACTOR_SPECS: Dict[str, LiquidityImpactFactorSpec] = {
    spec.name: spec
    for spec in (
        _spec(
            "spread_per_depth",
            "mean(relative_spread / log1p(total_depth_5))",
            A,
            "trade_x_quote_x_depth_spread_cost_per_unit_depth",
            redundancy="joint transform; distinct from relative_spread_mean",
        ),
        _spec(
            "depth_per_amount",
            "mean(total_depth_5 / (trade_amount + 1 CNY)) on trade minutes",
            A,
            "trade_x_depth_available_depth_per_traded_amount",
        ),
        _spec(
            "amount_to_depth",
            "sum(trade_amount) / mean(total_depth_5)",
            A,
            "trade_x_depth_daily_amount_over_mean_depth",
            redundancy="joint ratio; distinct from total_depth_level",
        ),
        _spec(
            "depth_turnover",
            "sum(trade_volume) / mean(total_depth_5)",
            A,
            "trade_x_depth_volume_over_mean_depth",
        ),
        _spec(
            "liquidity_cost_state",
            "mean(relative_spread * trade_amount / total_depth_5)",
            A,
            "trade_x_quote_x_depth_joint_cost_state",
        ),
        _spec(
            "signed_amount_impact",
            "sum(minute_return*signed_amount)/sum(abs(signed_amount))",
            B,
            "signed_flow_x_contemporaneous_mid_return",
        ),
        _spec(
            "signed_sqrt_amount_impact",
            "sum(minute_return*sign(s)*sqrt(abs(s)))/sum(sqrt(abs(s)))",
            B,
            "sqrt_weighted_signed_flow_x_mid_return",
            redundancy="expected near signed_amount_impact (sqrt weighting)",
        ),
        _spec(
            "impact_per_trade",
            "sum(minute_return*signed_count)/sum(abs(signed_count))",
            B,
            "signed_trade_count_x_mid_return",
        ),
        _spec(
            "buy_price_impact",
            "sum(fwd_mid_ret_1m*active_buy_amount)/sum(active_buy_amount)",
            B,
            "active_buy_flow_x_forward_mid_return_1m",
            proxy=(
                "amount-weighted mean forward mid return after buy-active"
                " minutes; minute approximation"
            ),
        ),
        _spec(
            "sell_price_impact",
            "sum(fwd_mid_ret_1m*active_sell_amount)/sum(active_sell_amount)",
            B,
            "active_sell_flow_x_forward_mid_return_1m",
            proxy="symmetric minute approximation of buy_price_impact",
        ),
        _spec(
            "impact_asymmetry",
            "buy_price_impact - sell_price_impact",
            B,
            "normalized_buy_impact_minus_sell_impact",
        ),
        _spec(
            "effective_spread_proxy",
            "mean(2*sign(signed_amount)*(trade_vwap-midquote)/midquote)",
            C,
            "minute_signed_vwap_deviation_from_midquote",
            proxy=(
                "minute approximation; NOT per-trade prevailing-quote"
                " effective spread"
            ),
        ),
        _spec(
            "permanent_impact_1m",
            "mean(2*sign(signed_amount)*fwd_mid_ret_1m)",
            C,
            "minute_signed_permanent_mid_move_1m",
        ),
        _spec(
            "permanent_impact_5m",
            "mean(2*sign(signed_amount)*fwd_mid_ret_5m)",
            C,
            "minute_signed_permanent_mid_move_5m_fixed_window",
        ),
        _spec(
            "realized_spread_proxy_5m",
            "effective_spread_proxy - permanent_impact_5m",
            C,
            "effective_minus_permanent_spread_proxy",
            proxy="inherits the effective_spread_proxy minute approximation",
        ),
        _spec(
            "adverse_selection_ratio",
            "permanent_impact_5m / abs(effective_spread_proxy); "
            "NaN when |denominator| <= 1e-12 or |ratio| > 1e6 "
            "(numerical-safety bound)",
            C,
            "permanent_share_of_effective_spread_safe_ratio",
        ),
        _spec(
            "small_trade_impact",
            "sum(fwd1*signed_small_amount)/sum(abs(signed_small_amount))",
            D,
            "small_trade_signed_flow_x_forward_mid_return_1m",
            redundancy="frozen boundary <= 1e4 CNY (no re-optimization)",
            proxy="tick-level bucket with minute forward response",
        ),
        _spec(
            "mid_trade_impact",
            "sum(fwd1*signed_mid_amount)/sum(abs(signed_mid_amount))",
            D,
            "mid_trade_signed_flow_x_forward_mid_return_1m",
            redundancy="frozen boundary (4e4, 2e5] CNY",
            proxy="tick-level bucket with minute forward response",
        ),
        _spec(
            "large_trade_impact",
            "sum(fwd1*signed_large_amount)/sum(abs(signed_large_amount))",
            D,
            "large_trade_signed_flow_x_forward_mid_return_1m",
            redundancy="frozen boundary > 2e5 CNY",
            proxy="tick-level bucket with minute forward response",
        ),
        _spec(
            "super_large_trade_impact",
            "sum(fwd1*signed_super_large_amount)"
            "/sum(abs(signed_super_large_amount))",
            D,
            "super_large_trade_signed_flow_x_forward_mid_return_1m",
            redundancy="frozen boundary > 1e6 CNY; sparse by construction",
            proxy="tick-level bucket with minute forward response",
        ),
        _spec(
            "impact_convexity",
            "large_trade_impact - small_trade_impact",
            D,
            "large_minus_small_trade_impact",
        ),
        _spec(
            "spread_recovery_5m",
            "mean((spread_t-spread_t+5)/spread_t) over high-impact minutes",
            E,
            "post_high_impact_spread_recovery_ratio",
        ),
        _spec(
            "depth_recovery_5m",
            "mean((depth_t+5-depth_t)/depth_t) over high-impact minutes",
            E,
            "post_high_impact_depth_rebuild_ratio",
        ),
        _spec(
            "impact_decay_5m",
            "permanent_impact_1m - permanent_impact_5m",
            E,
            "temporary_component_of_impact_decay",
        ),
    )
}

LIQUIDITY_IMPACT_FACTOR_NAMES: Tuple[str, ...] = tuple(
    LIQUIDITY_IMPACT_FACTOR_SPECS.keys()
)

_DIRECT: Dict[str, str] = {
    name: name
    for name in (
        "spread_per_depth",
        "depth_per_amount",
        "amount_to_depth",
        "depth_turnover",
        "liquidity_cost_state",
        "signed_amount_impact",
        "signed_sqrt_amount_impact",
        "impact_per_trade",
        "buy_price_impact",
        "sell_price_impact",
        "effective_spread_proxy",
        "permanent_impact_1m",
        "permanent_impact_5m",
        "small_trade_impact",
        "mid_trade_impact",
        "large_trade_impact",
        "super_large_trade_impact",
        "spread_recovery_5m",
        "depth_recovery_5m",
    )
}

FACTOR_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    **{name: (lambda f: lambda frame: frame[f].astype(float))(field)
       for name, field in _DIRECT.items()},
    "impact_asymmetry": (
        lambda frame: frame["buy_price_impact"] - frame["sell_price_impact"]
    ),
    "realized_spread_proxy_5m": (
        lambda frame: frame["effective_spread_proxy"]
        - frame["permanent_impact_5m"]
    ),
    "adverse_selection_ratio": (
        lambda frame: (
            frame["permanent_impact_5m"]
            / frame["effective_spread_proxy"]
            .abs()
            .where(frame["effective_spread_proxy"].abs() > 1e-12)
        ).mask(lambda ratio: ratio.abs() > 1e6)
    ),
    "impact_convexity": (
        lambda frame: frame["large_trade_impact"]
        - frame["small_trade_impact"]
    ),
    "impact_decay_5m": (
        lambda frame: frame["permanent_impact_1m"]
        - frame["permanent_impact_5m"]
    ),
}

REQUIRED_PRIMITIVE_COLUMNS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            "symbol",
            "TradeDate",
            "coverage_ratio",
            "expected_continuous_minutes",
            *_DIRECT.values(),
        ]
    )
)


def prepare_liquidity_impact_feature_primitive(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(
        set(REQUIRED_PRIMITIVE_COLUMNS).difference(primitive.columns)
    )
    if missing:
        raise ValueError(
            f"liquidity-impact primitive missing columns: {missing}"
        )
    frame = primitive.loc[:, REQUIRED_PRIMITIVE_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"]).dt.normalize()
    for column in REQUIRED_PRIMITIVE_COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "TradeDate"], kind="stable")
    if frame.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError(
            "liquidity-impact primitive must be unique at symbol-day"
        )
    return frame.reset_index(drop=True)


def _validate_feature_ranges(features: pd.DataFrame) -> None:
    for column in LIQUIDITY_IMPACT_FACTOR_NAMES:
        if np.isinf(features[column].to_numpy(dtype=float)).any():
            raise ValueError(f"inf values in factor {column}")
    bounded = ("adverse_selection_ratio",)
    for column in bounded:
        values = features[column].dropna()
        if len(values) and (values.abs() > 1e6).any():
            raise ValueError(f"{column} extreme values beyond safe bound")


def build_liquidity_impact_feature_frame(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    """Wide symbol-day feature frame from the frozen daily primitive."""
    frame = prepare_liquidity_impact_feature_primitive(primitive)
    coverage_ok = (
        frame["expected_continuous_minutes"] == EXPECTED_CONTINUOUS_MINUTES
    ) & (frame["coverage_ratio"] >= COVERAGE_THRESHOLD)
    features = frame[["symbol", "TradeDate"]].copy()
    for name, function in FACTOR_FUNCTIONS.items():
        features[name] = function(frame).where(coverage_ok)
    _validate_feature_ranges(features)
    return features


def feature_to_narrow(
    feature_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    if factor_name not in LIQUIDITY_IMPACT_FACTOR_SPECS:
        raise KeyError(f"unknown Liquidity/Impact factor: {factor_name}")
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
        list(LIQUIDITY_IMPACT_FACTOR_NAMES) if names is None else list(names)
    )
    unknown = sorted(
        set(selected).difference(LIQUIDITY_IMPACT_FACTOR_SPECS)
    )
    if unknown:
        raise KeyError(f"unknown Liquidity/Impact factors: {unknown}")
    return pd.DataFrame(
        [
            LIQUIDITY_IMPACT_FACTOR_SPECS[name].to_dict()
            for name in selected
        ]
    )


__all__ = [
    "COVERAGE_THRESHOLD",
    "EXPECTED_CONTINUOUS_MINUTES",
    "FACTOR_FUNCTIONS",
    "LIQUIDITY_IMPACT_FACTOR_NAMES",
    "LIQUIDITY_IMPACT_FACTOR_SPECS",
    "REQUIRED_PRIMITIVE_COLUMNS",
    "LiquidityImpactFactorSpec",
    "build_liquidity_impact_feature_frame",
    "feature_to_narrow",
    "prepare_liquidity_impact_feature_primitive",
    "registry_frame",
]
