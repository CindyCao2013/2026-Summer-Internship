"""Trade-flow daily primitive -> candidate factor feature layer.

This module is deliberately limited to formula construction. It does not query
raw Tick data, neutralize, optimize parameters, or combine factors. The
backtest layer applies the standard one-day signal shift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


STOCK_PREFIXES = ("60", "68", "000", "001", "002", "003", "300", "301", "302")
REQUIRED_COLUMNS = (
    "symbol",
    "TradeDate",
    "active_buy_amt",
    "active_sell_amt",
    "total_amt",
    "active_buy_cnt",
    "active_sell_cnt",
)


@dataclass(frozen=True)
class TradeFlowFactorSpec:
    """Frozen metadata for one candidate formula."""

    name: str
    formula: str
    mechanism: str
    lookback_days: int
    signed: bool
    expected_redundancy: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


TRADE_FLOW_FACTOR_SPECS: Dict[str, TradeFlowFactorSpec] = {
    "net_buy_ratio": TradeFlowFactorSpec(
        name="net_buy_ratio",
        formula="(active_buy_amt-active_sell_amt)/total_amt",
        mechanism="trade_direction_amount",
        lookback_days=1,
        signed=True,
    ),
    "net_buy_count_ratio": TradeFlowFactorSpec(
        name="net_buy_count_ratio",
        formula="(active_buy_cnt-active_sell_cnt)/(active_buy_cnt+active_sell_cnt)",
        mechanism="trade_direction_count",
        lookback_days=1,
        signed=True,
    ),
    "buy_dominance": TradeFlowFactorSpec(
        name="buy_dominance",
        formula="active_buy_amt/(active_buy_amt+active_sell_amt)",
        mechanism="trade_direction_amount",
        lookback_days=1,
        signed=False,
        expected_redundancy=(
            "near-alias of net_buy_ratio; exact affine alias when "
            "total_amt=active_buy_amt+active_sell_amt"
        ),
    ),
    "avg_buy_trade_size": TradeFlowFactorSpec(
        name="avg_buy_trade_size",
        formula="active_buy_amt/active_buy_cnt",
        mechanism="aggressor_trade_size",
        lookback_days=1,
        signed=False,
    ),
    "avg_sell_trade_size": TradeFlowFactorSpec(
        name="avg_sell_trade_size",
        formula="active_sell_amt/active_sell_cnt",
        mechanism="aggressor_trade_size",
        lookback_days=1,
        signed=False,
    ),
    "trade_size_asymmetry": TradeFlowFactorSpec(
        name="trade_size_asymmetry",
        formula="avg_buy_trade_size/avg_sell_trade_size",
        mechanism="aggressor_trade_size_asymmetry",
        lookback_days=1,
        signed=False,
        expected_redundancy="deterministic transform of the two average-size legs",
    ),
    "flow_concentration": TradeFlowFactorSpec(
        name="flow_concentration",
        formula=(
            "0.5*(abs(net_buy_ratio)+abs(net_buy_count_ratio))"
        ),
        mechanism="trade_direction_magnitude",
        lookback_days=1,
        signed=False,
        expected_redundancy="unsigned transform of amount/count imbalance",
    ),
    "flow_zscore_20d": TradeFlowFactorSpec(
        name="flow_zscore_20d",
        formula=(
            "(net_buy_ratio_t-mean(net_buy_ratio[t-20:t-1]))/"
            "std(net_buy_ratio[t-20:t-1])"
        ),
        mechanism="trade_flow_surprise",
        lookback_days=21,
        signed=True,
    ),
    "flow_acceleration": TradeFlowFactorSpec(
        name="flow_acceleration",
        formula="net_buy_ratio_t-net_buy_ratio_t-1",
        mechanism="trade_flow_dynamics",
        lookback_days=2,
        signed=True,
    ),
}

TRADE_FLOW_FACTOR_NAMES = tuple(TRADE_FLOW_FACTOR_SPECS)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorized division with zero and non-finite results mapped to NaN."""
    denominator = denominator.replace(0, np.nan)
    out = numerator / denominator
    return out.where(np.isfinite(out))


def _validate_columns(flow: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(flow.columns))
    if missing:
        raise ValueError(f"trade-flow primitive missing columns: {missing}")


def prepare_trade_flow_primitive(flow: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize, and filter the primitive to A-share-like symbols.

    ST, suspension and price-limit filters remain in the unified backtest mask;
    this first filter only removes funds, bonds and other non-equity symbols.
    """
    _validate_columns(flow)
    df = flow.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
    bare = df["symbol"].str.split(".").str[0]
    df = df.loc[bare.str.startswith(STOCK_PREFIXES)].copy()

    for column in REQUIRED_COLUMNS[2:]:
        # ClickHouse countIf/count columns arrive as UInt64. Cast before any
        # subtraction; otherwise buy_cnt < sell_cnt silently underflows to a
        # huge positive integer and corrupts count imbalance.
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).astype("float64")

    df = df.sort_values(["symbol", "TradeDate"], kind="stable")
    duplicates = df.duplicated(["symbol", "TradeDate"], keep=False)
    if duplicates.any():
        examples = (
            df.loc[duplicates, ["symbol", "TradeDate"]]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        raise ValueError(
            "trade-flow primitive must be unique at symbol-day; "
            f"examples={examples}"
        )
    return df


def build_trade_flow_feature_frame(flow: pd.DataFrame) -> pd.DataFrame:
    """Build all frozen Sprint-3 Trade Flow Family candidates.

    ``flow_zscore_20d`` uses the *preceding* 20 observations as its baseline.
    The current day's flow enters only the numerator. ``flow_acceleration`` is
    the one-day first difference. Both remain known only after the current
    close and are shifted by one trading day in ``backtest_factor``.
    """
    df = prepare_trade_flow_primitive(flow)

    buy_amt = df["active_buy_amt"]
    sell_amt = df["active_sell_amt"]
    total_amt = df["total_amt"]
    buy_cnt = df["active_buy_cnt"]
    sell_cnt = df["active_sell_cnt"]

    directed_amt = buy_amt - sell_amt
    directed_cnt = buy_cnt - sell_cnt
    classified_amt = buy_amt + sell_amt
    classified_cnt = buy_cnt + sell_cnt

    features = pd.DataFrame(index=df.index)
    features["net_buy_ratio"] = _safe_divide(directed_amt, total_amt)
    features["net_buy_count_ratio"] = _safe_divide(
        directed_cnt, classified_cnt
    )
    invalid_count_ratio = features["net_buy_count_ratio"].abs() > 1.0 + 1e-12
    if invalid_count_ratio.any():
        sample = features.loc[
            invalid_count_ratio, "net_buy_count_ratio"
        ].head(5).tolist()
        raise ValueError(
            "count imbalance must be within [-1, 1]; "
            f"sample invalid values={sample}"
        )
    features["buy_dominance"] = _safe_divide(buy_amt, classified_amt)
    features["avg_buy_trade_size"] = _safe_divide(buy_amt, buy_cnt)
    features["avg_sell_trade_size"] = _safe_divide(sell_amt, sell_cnt)
    features["trade_size_asymmetry"] = _safe_divide(
        features["avg_buy_trade_size"],
        features["avg_sell_trade_size"],
    )
    features["flow_concentration"] = 0.5 * (
        features["net_buy_ratio"].abs()
        + features["net_buy_count_ratio"].abs()
    )

    symbol = df["symbol"]
    flow_level = features["net_buy_ratio"]
    lagged_flow = flow_level.groupby(symbol, sort=False).shift(1)
    history = lagged_flow.groupby(symbol, sort=False)
    history_mean = history.transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    history_std = history.transform(
        lambda x: x.rolling(20, min_periods=20).std(ddof=0)
    )
    features["flow_zscore_20d"] = _safe_divide(
        flow_level - history_mean, history_std
    )
    features["flow_acceleration"] = flow_level - lagged_flow

    features = features.replace([np.inf, -np.inf], np.nan)
    return pd.concat(
        [
            df[["symbol", "TradeDate"]].reset_index(drop=True),
            features[list(TRADE_FLOW_FACTOR_NAMES)].reset_index(drop=True),
        ],
        axis=1,
    )


def feature_to_narrow(
    feature_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """Convert one feature column to the standard factor narrow schema."""
    if factor_name not in TRADE_FLOW_FACTOR_SPECS:
        raise KeyError(
            f"unknown trade-flow factor {factor_name!r}; "
            f"valid={list(TRADE_FLOW_FACTOR_NAMES)}"
        )
    out = feature_frame.loc[
        :, ["symbol", "TradeDate", factor_name]
    ].rename(columns={factor_name: "value"})
    out["tradetime"] = (
        pd.to_datetime(out.pop("TradeDate"))
        + pd.Timedelta(hours=9, minutes=30)
    )
    out["factorname"] = factor_name
    out = out[["symbol", "tradetime", "factorname", "value"]]
    return out.dropna(subset=["value"]).reset_index(drop=True)


def registry_frame(
    names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return the frozen formula registry as a serializable table."""
    selected: List[str] = (
        list(TRADE_FLOW_FACTOR_NAMES) if names is None else list(names)
    )
    unknown = sorted(set(selected).difference(TRADE_FLOW_FACTOR_SPECS))
    if unknown:
        raise KeyError(f"unknown trade-flow factors: {unknown}")
    return pd.DataFrame(
        [TRADE_FLOW_FACTOR_SPECS[name].to_dict() for name in selected]
    )
