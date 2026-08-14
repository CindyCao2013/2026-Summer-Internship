"""L2 minute factor panel helpers (long schema for evaluation)."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

PANEL_COLUMNS = (
    "date",
    "bartime",
    "symbol",
    "factor",
    "value",
    "source",
    "aggregation",
)


def minute_wide_to_long(
    wide: pd.DataFrame,
    *,
    factor_columns: Sequence[str],
    source: str = "SSL2",
    aggregation_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Convert minute_time/symbol wide features to evaluation long panel."""
    if wide.empty:
        return pd.DataFrame(columns=list(PANEL_COLUMNS))
    frame = wide.copy()
    frame["minute_time"] = pd.to_datetime(frame["minute_time"])
    long = frame.melt(
        id_vars=["minute_time", "symbol"],
        value_vars=[c for c in factor_columns if c in frame.columns],
        var_name="factor",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    # Naive calendar date — avoids tz-aware parquet / DDB upload friction.
    mt = long["minute_time"]
    if getattr(mt.dt, "tz", None) is not None:
        mt = mt.dt.tz_localize(None)
    long["date"] = mt.dt.normalize()
    long["bartime"] = mt.dt.strftime("%H:%M")
    long["source"] = source
    agg_map = aggregation_map or {}
    long["aggregation"] = long["factor"].map(
        lambda f: agg_map.get(f, f.rsplit("_", 1)[-1] if "_" in f else "unknown")
    )
    long["symbol"] = long["symbol"].astype(str)
    return long[list(PANEL_COLUMNS)].reset_index(drop=True)


def filter_bartimes(
    panel: pd.DataFrame,
    bartimes: Iterable[str],
) -> pd.DataFrame:
    keep = set(bartimes)
    return panel[panel["bartime"].isin(keep)].copy()


def to_evaluation_signal(
    panel: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """Narrow signal expected by intraday DDB evaluation helpers."""
    sub = panel[panel["factor"] == factor_name].copy()
    if sub.empty:
        return pd.DataFrame(columns=["tradetime", "symbol", "factorname", "value"])
    # Rebuild tradetime from date + bartime (timezone-naive for DDB upload).
    dates = pd.to_datetime(sub["date"], utc=False)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    tradetime = pd.to_datetime(
        dates.dt.strftime("%Y-%m-%d") + " " + sub["bartime"].astype(str)
    )
    out = pd.DataFrame(
        {
            "tradetime": tradetime,
            "symbol": sub["symbol"].astype(str).values,
            "factorname": factor_name,
            "value": pd.to_numeric(sub["value"], errors="coerce").values,
        }
    )
    return out.dropna(subset=["value"]).reset_index(drop=True)
