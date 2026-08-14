"""Order-size distribution primitive -> frozen candidate factor layer.

The formulas in this module are discovery features, not optimized production
signals. Thresholds and rolling windows are explicit in names/metadata so the
library cannot silently change an existing factor's meaning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


STOCK_PREFIXES = ("60", "68", "000", "001", "002", "003", "300", "301", "302")
BOUNDARIES = (10_000, 40_000, 50_000, 200_000, 1_000_000)


@dataclass(frozen=True)
class OrderSizeFactorSpec:
    name: str
    formula: str
    mechanism: str
    lookback_days: int
    signed: bool
    expected_redundancy: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


ORDER_SIZE_FACTOR_SPECS: Dict[str, OrderSizeFactorSpec] = {
    "small_order_ratio_1w": OrderSizeFactorSpec(
        "small_order_ratio_1w",
        "amount(0,1w]/total_amt",
        "size_distribution_small",
        1,
        False,
    ),
    "small_order_ratio_4w": OrderSizeFactorSpec(
        "small_order_ratio_4w",
        "amount(0,4w]/total_amt",
        "size_distribution_small",
        1,
        False,
        "legacy small-order threshold; contains small_order_ratio_1w",
    ),
    "mid_order_ratio_4w_20w": OrderSizeFactorSpec(
        "mid_order_ratio_4w_20w",
        "amount(4w,20w]/total_amt",
        "size_distribution_mid",
        1,
        False,
        "long-sample threshold-explicit form of frozen mid_order_ratio",
    ),
    "mid_order_ratio_5w_20w": OrderSizeFactorSpec(
        "mid_order_ratio_5w_20w",
        "amount(5w,20w]/total_amt",
        "size_distribution_mid",
        1,
        False,
        "near-neighbor of mid_order_ratio_4w_20w; retained as canonical bucket",
    ),
    "large_order_ratio_20w": OrderSizeFactorSpec(
        "large_order_ratio_20w",
        "amount(20w,+inf)/total_amt",
        "size_distribution_large",
        1,
        False,
    ),
    "super_large_order_ratio_100w": OrderSizeFactorSpec(
        "super_large_order_ratio_100w",
        "amount(100w,+inf)/total_amt",
        "size_distribution_tail",
        1,
        False,
    ),
    "large_small_spread": OrderSizeFactorSpec(
        "large_small_spread",
        "large_order_ratio_20w-small_order_ratio_1w",
        "size_distribution_spread",
        1,
        True,
        "deterministic linear combination of large and small shares",
    ),
    "order_size_entropy": OrderSizeFactorSpec(
        "order_size_entropy",
        "-sum(p_bucket*log(p_bucket))/log(5)",
        "size_distribution_shape",
        1,
        False,
    ),
    "order_size_concentration": OrderSizeFactorSpec(
        "order_size_concentration",
        "sum(p_bucket^2)",
        "size_distribution_shape",
        1,
        False,
        "nonlinear inverse-shape companion of order_size_entropy",
    ),
    "order_size_tail_share": OrderSizeFactorSpec(
        "order_size_tail_share",
        "p_(0,1w]+p_(100w,+inf)",
        "size_distribution_tail",
        1,
        False,
    ),
    "small_order_pressure": OrderSizeFactorSpec(
        "small_order_pressure",
        "(small_active_buy_amt-small_active_sell_amt)/total_amt",
        "size_conditioned_direction",
        1,
        True,
    ),
    "mid_order_pressure": OrderSizeFactorSpec(
        "mid_order_pressure",
        "(mid_active_buy_amt-mid_active_sell_amt)/total_amt",
        "size_conditioned_direction",
        1,
        True,
    ),
    "large_order_pressure": OrderSizeFactorSpec(
        "large_order_pressure",
        "(large_active_buy_amt-large_active_sell_amt)/total_amt",
        "size_conditioned_direction",
        1,
        True,
    ),
    "super_large_order_pressure": OrderSizeFactorSpec(
        "super_large_order_pressure",
        "(super_active_buy_amt-super_active_sell_amt)/total_amt",
        "size_conditioned_direction",
        1,
        True,
    ),
    "buy_large_order_ratio": OrderSizeFactorSpec(
        "buy_large_order_ratio",
        "large_active_buy_amt/active_buy_amt",
        "buy_side_size_composition",
        1,
        False,
    ),
    "sell_large_order_ratio": OrderSizeFactorSpec(
        "sell_large_order_ratio",
        "large_active_sell_amt/active_sell_amt",
        "sell_side_size_composition",
        1,
        False,
    ),
    "small_order_direction": OrderSizeFactorSpec(
        "small_order_direction",
        "(small_buy-small_sell)/(small_buy+small_sell)",
        "within_bucket_direction",
        1,
        True,
    ),
    "large_order_direction": OrderSizeFactorSpec(
        "large_order_direction",
        "(large_buy-large_sell)/(large_buy+large_sell)",
        "within_bucket_direction",
        1,
        True,
    ),
    "large_order_shock_20d": OrderSizeFactorSpec(
        "large_order_shock_20d",
        "zscore(large_order_ratio_20w_t vs prior 20 observations)",
        "size_distribution_shock",
        21,
        True,
    ),
    "order_size_entropy_shock_20d": OrderSizeFactorSpec(
        "order_size_entropy_shock_20d",
        "zscore(order_size_entropy_t vs prior 20 observations)",
        "size_distribution_shock",
        21,
        True,
    ),
}

ORDER_SIZE_FACTOR_NAMES = tuple(ORDER_SIZE_FACTOR_SPECS)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.where(np.isfinite(result))


def _required_columns() -> List[str]:
    columns = [
        "symbol",
        "TradeDate",
        "total_amt",
        "trade_cnt",
        "active_buy_amt",
        "active_sell_amt",
    ]
    for boundary in BOUNDARIES:
        columns.extend(
            [
                f"cum_amt_{boundary}",
                f"cum_cnt_{boundary}",
                f"buy_cum_amt_{boundary}",
                f"sell_cum_amt_{boundary}",
            ]
        )
    return columns


def prepare_order_size_primitive(primitive: pd.DataFrame) -> pd.DataFrame:
    required = _required_columns()
    missing = sorted(set(required).difference(primitive.columns))
    if missing:
        raise ValueError(f"order-size primitive missing columns: {missing}")
    df = primitive.loc[:, required].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
    bare = df["symbol"].str.split(".").str[0]
    df = df.loc[bare.str.startswith(STOCK_PREFIXES)].copy()
    for column in required[2:]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).astype("float64")
    df = df.sort_values(["symbol", "TradeDate"], kind="stable")
    if df.duplicated(["symbol", "TradeDate"]).any():
        raise ValueError("order-size primitive must be unique at symbol-day")

    cumulative = df[
        [f"cum_amt_{boundary}" for boundary in BOUNDARIES]
    ].to_numpy(dtype=float)
    buy_cumulative = df[
        [f"buy_cum_amt_{boundary}" for boundary in BOUNDARIES]
    ].to_numpy(dtype=float)
    sell_cumulative = df[
        [f"sell_cum_amt_{boundary}" for boundary in BOUNDARIES]
    ].to_numpy(dtype=float)
    tolerance = np.maximum(1e-4, np.abs(cumulative) * 1e-10)
    total = df["total_amt"].to_numpy(dtype=float)
    total_tolerance = np.maximum(1e-4, np.abs(total) * 1e-10)
    if (
        np.diff(cumulative, axis=1) < -tolerance[:, 1:]
    ).any():
        raise ValueError("cumulative amount buckets are not monotonic")
    if (
        np.diff(buy_cumulative, axis=1) < -tolerance[:, 1:]
    ).any() or (
        np.diff(sell_cumulative, axis=1) < -tolerance[:, 1:]
    ).any():
        raise ValueError("side cumulative amount buckets are not monotonic")
    if (
        cumulative[:, -1]
        > total + total_tolerance
    ).any():
        raise ValueError("cumulative bucket amount exceeds total amount")
    if (
        (buy_cumulative + sell_cumulative)
        > cumulative + tolerance
    ).any():
        raise ValueError("classified side amount exceeds bucket amount")
    if (
        buy_cumulative[:, -1]
        > df["active_buy_amt"].to_numpy(dtype=float) + total_tolerance
    ).any():
        raise ValueError("buy cumulative amount exceeds active buy total")
    if (
        sell_cumulative[:, -1]
        > df["active_sell_amt"].to_numpy(dtype=float) + total_tolerance
    ).any():
        raise ValueError("sell cumulative amount exceeds active sell total")
    return df


def _prior_zscore(
    values: pd.Series,
    symbols: pd.Series,
    window: int = 20,
) -> pd.Series:
    lagged = values.groupby(symbols, sort=False).shift(1)
    history = lagged.groupby(symbols, sort=False)
    mean = history.transform(
        lambda series: series.rolling(
            window, min_periods=window
        ).mean()
    )
    std = history.transform(
        lambda series: series.rolling(
            window, min_periods=window
        ).std(ddof=0)
    )
    return _safe_divide(values - mean, std)


def build_order_size_feature_frame(
    primitive: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 20 frozen Sprint-4 Order Size Family candidates."""
    df = prepare_order_size_primitive(primitive)
    total = df["total_amt"]

    c1 = df["cum_amt_10000"]
    c4 = df["cum_amt_40000"]
    c5 = df["cum_amt_50000"]
    c20 = df["cum_amt_200000"]
    c100 = df["cum_amt_1000000"]
    bucket_amounts = pd.DataFrame(
        {
            "b1": c1,
            "b2": c5 - c1,
            "b3": c20 - c5,
            "b4": c100 - c20,
            "b5": total - c100,
        },
        index=df.index,
    ).clip(lower=0.0)
    bucket_shares = bucket_amounts.div(
        total.replace(0, np.nan), axis=0
    )

    buy = df["active_buy_amt"]
    sell = df["active_sell_amt"]
    small_buy = df["buy_cum_amt_10000"]
    small_sell = df["sell_cum_amt_10000"]
    mid_buy = df["buy_cum_amt_200000"] - df["buy_cum_amt_40000"]
    mid_sell = (
        df["sell_cum_amt_200000"] - df["sell_cum_amt_40000"]
    )
    large_buy = (buy - df["buy_cum_amt_200000"]).clip(lower=0.0)
    large_sell = (sell - df["sell_cum_amt_200000"]).clip(lower=0.0)
    super_buy = (buy - df["buy_cum_amt_1000000"]).clip(lower=0.0)
    super_sell = (sell - df["sell_cum_amt_1000000"]).clip(lower=0.0)

    features = pd.DataFrame(index=df.index)
    features["small_order_ratio_1w"] = _safe_divide(c1, total)
    features["small_order_ratio_4w"] = _safe_divide(c4, total)
    features["mid_order_ratio_4w_20w"] = _safe_divide(
        c20 - c4, total
    )
    features["mid_order_ratio_5w_20w"] = _safe_divide(
        c20 - c5, total
    )
    features["large_order_ratio_20w"] = _safe_divide(
        total - c20, total
    )
    features["super_large_order_ratio_100w"] = _safe_divide(
        total - c100, total
    )
    features["large_small_spread"] = (
        features["large_order_ratio_20w"]
        - features["small_order_ratio_1w"]
    )

    entropy_terms = bucket_shares.where(bucket_shares > 0)
    features["order_size_entropy"] = (
        -(entropy_terms * np.log(entropy_terms)).sum(
            axis=1, min_count=1
        )
        / np.log(bucket_shares.shape[1])
    )
    features["order_size_concentration"] = (
        bucket_shares.pow(2).sum(axis=1, min_count=1)
    )
    features["order_size_tail_share"] = (
        bucket_shares["b1"] + bucket_shares["b5"]
    )

    features["small_order_pressure"] = _safe_divide(
        small_buy - small_sell, total
    )
    features["mid_order_pressure"] = _safe_divide(
        mid_buy - mid_sell, total
    )
    features["large_order_pressure"] = _safe_divide(
        large_buy - large_sell, total
    )
    features["super_large_order_pressure"] = _safe_divide(
        super_buy - super_sell, total
    )
    features["buy_large_order_ratio"] = _safe_divide(
        large_buy, buy
    )
    features["sell_large_order_ratio"] = _safe_divide(
        large_sell, sell
    )
    features["small_order_direction"] = _safe_divide(
        small_buy - small_sell, small_buy + small_sell
    )
    features["large_order_direction"] = _safe_divide(
        large_buy - large_sell, large_buy + large_sell
    )

    symbols = df["symbol"]
    features["large_order_shock_20d"] = _prior_zscore(
        features["large_order_ratio_20w"],
        symbols,
    )
    features["order_size_entropy_shock_20d"] = _prior_zscore(
        features["order_size_entropy"],
        symbols,
    )
    features = features.replace([np.inf, -np.inf], np.nan)

    bounded_zero_one = [
        "small_order_ratio_1w",
        "small_order_ratio_4w",
        "mid_order_ratio_4w_20w",
        "mid_order_ratio_5w_20w",
        "large_order_ratio_20w",
        "super_large_order_ratio_100w",
        "order_size_entropy",
        "order_size_concentration",
        "order_size_tail_share",
        "buy_large_order_ratio",
        "sell_large_order_ratio",
    ]
    for name in bounded_zero_one:
        invalid = (features[name] < -1e-10) | (
            features[name] > 1.0 + 1e-10
        )
        if invalid.any():
            raise ValueError(f"{name} outside [0, 1]")
        features[name] = features[name].clip(0.0, 1.0)
    bounded_signed = [
        "small_order_pressure",
        "mid_order_pressure",
        "large_order_pressure",
        "super_large_order_pressure",
        "small_order_direction",
        "large_order_direction",
    ]
    for name in bounded_signed:
        invalid = features[name].abs() > 1.0 + 1e-10
        if invalid.any():
            raise ValueError(f"{name} outside [-1, 1]")
        features[name] = features[name].clip(-1.0, 1.0)

    return pd.concat(
        [
            df[["symbol", "TradeDate"]].reset_index(drop=True),
            features[list(ORDER_SIZE_FACTOR_NAMES)].reset_index(drop=True),
        ],
        axis=1,
    )


def feature_to_narrow(
    feature_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    if factor_name not in ORDER_SIZE_FACTOR_SPECS:
        raise KeyError(
            f"unknown order-size factor {factor_name!r}; "
            f"valid={list(ORDER_SIZE_FACTOR_NAMES)}"
        )
    out = feature_frame[
        ["symbol", "TradeDate", factor_name]
    ].rename(columns={factor_name: "value"})
    out["tradetime"] = (
        pd.to_datetime(out.pop("TradeDate"))
        + pd.Timedelta(hours=9, minutes=30)
    )
    out["factorname"] = factor_name
    return out[
        ["symbol", "tradetime", "factorname", "value"]
    ].dropna(subset=["value"]).reset_index(drop=True)


def registry_frame(
    names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected: List[str] = (
        list(ORDER_SIZE_FACTOR_NAMES) if names is None else list(names)
    )
    unknown = sorted(set(selected).difference(ORDER_SIZE_FACTOR_SPECS))
    if unknown:
        raise KeyError(f"unknown order-size factors: {unknown}")
    return pd.DataFrame(
        [ORDER_SIZE_FACTOR_SPECS[name].to_dict() for name in selected]
    )
