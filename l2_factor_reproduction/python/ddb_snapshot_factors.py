"""DolphinDB reference snapshot family: frozen registry + narrow conversion.

Sprint 6A.  Five official formulas (docs.dolphindb.cn §3.1) replicated on
company ClickHouse SSL2 data; see
research/results/l2_reproduction/primitives/ddb_reference_snapshot/
formula_mapping.md for the verbatim code and line-by-line semantics.

Family taxonomy: the existing Order Book family describes static book state;
this family adds LOB dynamics (level10_diff_buy), an order-book/price-
formation bridge (level10_infer_price_trend), a quote-change/trade-price
composite (tra_price_weighted_net_buy_quote_volume_ratio), a time-varying
book slope (time_weighted_order_slope) and a strict-replication redundancy
control against weighted_obi_mean (wavg_soir).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class DDBSnapshotFactorSpec:
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
) -> DDBSnapshotFactorSpec:
    return DDBSnapshotFactorSpec(
        name=name,
        formula=formula,
        category=category,
        mechanism=mechanism,
        lookback_days=lookback,
        signed=signed,
        expected_redundancy=redundancy,
        alias_features=aliases,
    )


DDB_SNAPSHOT_FACTOR_SPECS: Dict[str, DDBSnapshotFactorSpec] = {
    "time_weighted_order_slope": _spec(
        "time_weighted_order_slope",
        "(log(ask1eff)-log(bid1eff)) / nullFill(mavg(ffill("
        "log(askQty1)-log(bidQty1)),20,1),0); 3s series -> minute-last "
        "-> daily mean",
        "lob_slope_dynamic",
        "time-varying book slope from level-1 log price gap scaled by "
        "smoothed log quantity asymmetry; unstable near logQty parity "
        "(small-denominator diagnostics frozen)",
        redundancy=(
            "partial overlap with relative_spread / depth slope family; "
            "distinct denominator dynamics"
        ),
    ),
    "wavg_soir": _spec(
        "wavg_soir",
        "rowWavg(level (bq-aq)/(bq+aq), 10..1) ffill nullFill0, "
        "standardized vs prev-only 19-snapshot mean/pop-std; 3s -> "
        "minute-last -> daily mean",
        "order_imbalance_dynamic",
        "per-level imbalance weighted then normalized against own "
        "trailing window (current value excluded from its baseline)",
        redundancy=(
            "expected high correlation with weighted_obi_mean; kept as "
            "strict-replication redundancy control, not counted as new "
            "independent alpha"
        ),
        aliases="weighted_obi_mean",
    ),
    "tra_price_weighted_net_buy_quote_volume_ratio": _spec(
        "tra_price_weighted_net_buy_quote_volume_ratio",
        "level-1 bid/ask change decomposition x inter-snapshot avg trade "
        "price; msum20 ratio; 3s -> minute-last -> daily mean",
        "lob_dynamics",
        "net buy quote change weighted by realized average trade price "
        "between snapshots; no existing pool counterpart",
    ),
    "level10_diff_buy": _spec(
        "level10_diff_buy",
        "rowAlign('bid') price-level alignment of adjacent ten-level bid "
        "arrays, per-price qty diff x price, msum20; 3s -> minute-last "
        "-> daily mean",
        "lob_dynamics",
        "true dynamic order-book flow: amount added/removed per price "
        "level after controlling for grid moves (NOT depth.diff)",
    ),
    "level10_infer_price_trend": _spec(
        "level10_infer_price_trend",
        "amount-weighted ten-level implied price, ffill, "
        "linearTimeTrend(60) slope, mavg20; 3s -> minute-last -> "
        "daily mean",
        "price_formation_bridge",
        "intraday trend of the unfilled-order-book implied price; "
        "bridges static book state and price formation (uses book "
        "prices, not traded prices)",
    ),
}

DDB_SNAPSHOT_FACTOR_NAMES: tuple = tuple(DDB_SNAPSHOT_FACTOR_SPECS)


def registry_frame(
    names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    selected: List[str] = (
        list(DDB_SNAPSHOT_FACTOR_NAMES) if names is None else list(names)
    )
    unknown = sorted(set(selected).difference(DDB_SNAPSHOT_FACTOR_SPECS))
    if unknown:
        raise KeyError(f"unknown DDB snapshot factors: {unknown}")
    return pd.DataFrame(
        [DDB_SNAPSHOT_FACTOR_SPECS[name].to_dict() for name in selected]
    )


def primitive_to_narrow(
    daily_frame: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """Symbol-day primitive row -> (symbol, tradetime, factorname, value)."""
    if factor_name not in DDB_SNAPSHOT_FACTOR_SPECS:
        raise KeyError(f"unknown DDB snapshot factor: {factor_name}")
    column = f"{factor_name}_mean"
    out = daily_frame[["symbol", "TradeDate", column]].rename(
        columns={column: "value"}
    )
    out["tradetime"] = pd.to_datetime(out.pop("TradeDate")) + pd.Timedelta(
        hours=9, minutes=30
    )
    out["factorname"] = factor_name
    return (
        out[["symbol", "tradetime", "factorname", "value"]]
        .dropna(subset=["value"])
        .reset_index(drop=True)
    )
