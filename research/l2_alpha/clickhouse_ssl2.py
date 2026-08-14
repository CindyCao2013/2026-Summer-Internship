"""ClickHouse SSL2 → minute feature extractor (server-side array ops)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd

from research.l2_alpha.l2_factor_registry import (
    L2_DIAGNOSTIC_COLUMNS,
    L2_PHASE2_FACTORS,
)
from research.l2_alpha.schema import (
    DATABASE,
    DEFAULT_WOI_LAMBDA,
    FACTOR_NAMES,
    N_DEPTH_LEVELS,
    NARROW_COLUMNS,
    SNAPSHOT_TABLES,
)

# Phase-2 discovery columns (+ diagnostics) produced by minute_agg SQL.
PHASE2_WIDE_COLUMNS = tuple(L2_PHASE2_FACTORS.keys()) + L2_DIAGNOSTIC_COLUMNS


def _ch_literal_dt64(ts: str) -> str:
    """Format 'YYYY-MM-DD' or full timestamp for DateTime64 Asia/Shanghai."""
    text = str(ts).strip()
    if len(text) == 10:
        text = f"{text} 00:00:00"
    return f"toDateTime64('{text}', 6, 'Asia/Shanghai')"


def _symbol_filter_sql(symbols: Optional[Sequence[str]]) -> str:
    if not symbols:
        return ""
    bare = sorted({str(s).split(".")[0] for s in symbols})
    in_list = ", ".join(f"'{c}'" for c in bare)
    return f"AND Symbol IN ({in_list})"


def _bartime_filter_sql(bartimes: Optional[Sequence[str]]) -> str:
    """Push evaluation slots into CH so we do not scan all ms snapshots."""
    if not bartimes:
        return ""
    slots = []
    for bt in bartimes:
        hh, mm = str(bt).strip().split(":")[:2]
        slots.append(f"({int(hh)}, {int(mm)})")
    return (
        "AND (toHour(ExchTime), toMinute(ExchTime)) IN ("
        + ", ".join(slots)
        + ")"
    )


def snapshot_feature_select_sql(
    *,
    table: str,
    exchange_suffix: str,
    start: str,
    end: str,
    lam: float = DEFAULT_WOI_LAMBDA,
    symbols: Optional[Sequence[str]] = None,
    has_withdraw: bool = True,
    bartimes: Optional[Sequence[str]] = None,
) -> str:
    """Emit one row per snapshot with base LOB metrics (CH array ops)."""
    symbol_filter = _symbol_filter_sql(symbols)
    bartime_filter = _bartime_filter_sql(bartimes)

    if has_withdraw:
        withdraw_exprs = """
    ifNull(toFloat64(BidWithdrawVolume), 0) AS b_wd,
    ifNull(toFloat64(AskWithdrawVolume), 0) AS a_wd,"""
    else:
        withdraw_exprs = """
    CAST(0 AS Float64) AS b_wd,
    CAST(0 AS Float64) AS a_wd,"""

    n = N_DEPTH_LEVELS
    return f"""
WITH
    arrayResize(arrayMap(x -> toFloat64(x), BidVolumes), {n}, NULL) AS bid_vol,
    arrayResize(arrayMap(x -> toFloat64(x), AskVolumes), {n}, NULL) AS ask_vol,
    arrayResize(arrayMap(x -> toFloat64(x), BidPrices), {n}, NULL) AS bid_px,
    arrayResize(arrayMap(x -> toFloat64(x), AskPrices), {n}, NULL) AS ask_px,
    arrayMap(i -> exp(-{lam} * (i - 1)), range(1, {n} + 1)) AS wts,
    arraySum(arrayMap((v, w) -> ifNull(v, 0) * w, bid_vol, wts)) AS w_bid,
    arraySum(arrayMap((v, w) -> ifNull(v, 0) * w, ask_vol, wts)) AS w_ask,
    arraySum(arrayMap(v -> ifNull(v, 0), bid_vol)) AS sum_bid,
    arraySum(arrayMap(v -> ifNull(v, 0), ask_vol)) AS sum_ask,
    bid_vol[1] AS b0,
    ask_vol[1] AS a0,
    bid_px[1] AS bp0,
    ask_px[1] AS ap0,
    if(isFinite(bp0) AND isFinite(ap0), (bp0 + ap0) / 2, NULL) AS mid
SELECT
    ExchTime AS exch_time,
    concat(Symbol, '{exchange_suffix}') AS symbol,
    if(isFinite(b0) AND isFinite(a0) AND (b0 + a0) != 0,
       (b0 - a0) / (b0 + a0), NULL) AS top_oi,
    if((sum_bid + sum_ask) != 0,
       (sum_bid - sum_ask) / (sum_bid + sum_ask), NULL) AS depth_oi,
    if((w_bid + w_ask) != 0,
       (w_bid - w_ask) / (w_bid + w_ask), NULL) AS weighted_oi,
    if(isFinite(bp0) AND isFinite(ap0) AND isFinite(b0) AND isFinite(a0)
          AND (b0 + a0) != 0 AND mid > 0,
       ((bp0 * a0 + ap0 * b0) / (b0 + a0) - mid) / mid, NULL) AS micro_bias,
    if(isFinite(mid) AND mid > 0 AND isFinite(ap0) AND isFinite(bp0),
       (ap0 - bp0) / mid, NULL) AS rel_spread,
{withdraw_exprs}
    (b_wd - a_wd) AS cancel_signed,
    (b_wd + a_wd) AS cancel_total,
    if((b_wd + a_wd) != 0, (b_wd - a_wd) / (b_wd + a_wd), NULL) AS cancel_imb,
    if(isFinite(mid) AND mid > 0
          AND isFinite(toFloat64(BidVWAP)) AND isFinite(toFloat64(AskVWAP)),
       (toFloat64(AskVWAP) - mid) / mid - (mid - toFloat64(BidVWAP)) / mid,
       NULL) AS liq_skew,
    if((sum_bid + sum_ask) != 0,
       arrayMax(arrayMap(v -> ifNull(v, 0), arrayConcat(bid_vol, ask_vol)))
         / (sum_bid + sum_ask),
       NULL) AS wall_ratio
FROM {DATABASE}.`{table}`
WHERE ExchTime >= {_ch_literal_dt64(start)}
  AND ExchTime < {_ch_literal_dt64(end)}
  AND length(BidPrices) >= 1
  AND length(AskPrices) >= 1
  AND length(BidVolumes) >= 1
  AND length(AskVolumes) >= 1
  {symbol_filter}
  {bartime_filter}
"""


def minute_last_feature_sql(
    *,
    table: str,
    exchange_suffix: str,
    start: str,
    end: str,
    lam: float = DEFAULT_WOI_LAMBDA,
    symbols: Optional[Sequence[str]] = None,
    factor_names: Sequence[str] = FACTOR_NAMES,
    has_withdraw: bool = True,
) -> str:
    """Legacy Phase-1 aggregator: last snapshot in each minute (argMax)."""
    inner = snapshot_feature_select_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        start=start,
        end=end,
        lam=lam,
        symbols=symbols,
        has_withdraw=has_withdraw,
    )
    # Map legacy long names onto base metrics.
    rename = {
        "l2_top_book_imbalance": "top_oi",
        "l2_depth_imbalance": "depth_oi",
        "l2_weighted_oi": "weighted_oi",
        "l2_microprice_bias": "micro_bias",
        "l2_relative_spread": "rel_spread",
        "l2_cancel_pressure": "cancel_imb",
        "l2_liquidity_skew": "liq_skew",
        "l2_liquidity_wall": "wall_ratio",
    }
    factor_select = ",\n    ".join(
        f"argMax({rename[name]}, exch_time) AS {name}"
        for name in factor_names
        if name in rename
    )
    return f"""
SELECT
    toStartOfMinute(exch_time) AS minute_time,
    symbol,
    {factor_select}
FROM (
{inner}
)
GROUP BY minute_time, symbol
"""


def minute_agg_feature_sql(
    *,
    table: str,
    exchange_suffix: str,
    start: str,
    end: str,
    lam: float = DEFAULT_WOI_LAMBDA,
    symbols: Optional[Sequence[str]] = None,
    has_withdraw: bool = True,
    bartimes: Optional[Sequence[str]] = None,
) -> str:
    """Phase-2 minute aggregation (mean/max/std/sum_ratio) inside ClickHouse."""
    inner = snapshot_feature_select_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        start=start,
        end=end,
        lam=lam,
        symbols=symbols,
        has_withdraw=has_withdraw,
        bartimes=bartimes,
    )
    # cancel_pressure_sum = minute flow ratio from summed withdraw volumes
    # (not sum of per-snapshot ratios). NULL when table has no withdraw
    # (SZSE → cancel_total always 0).
    cancel_expr = (
        "if(sum(cancel_total) > 0, "
        "sum(cancel_signed) / sum(cancel_total), NULL)"
        if has_withdraw
        else "CAST(NULL AS Nullable(Float64))"
    )
    return f"""
SELECT
    toStartOfMinute(exch_time) AS minute_time,
    symbol,
    avg(weighted_oi) AS l2_weighted_oi_mean,
    max(weighted_oi) AS l2_weighted_oi_max,
    stddevPop(weighted_oi) AS l2_weighted_oi_std,
    avg(micro_bias) AS l2_microprice_bias_mean,
    avg(depth_oi) AS l2_depth_imbalance_mean,
    {cancel_expr} AS l2_cancel_pressure_sum
FROM (
{inner}
)
GROUP BY minute_time, symbol
"""


def connect_hf_client():
    import clickhouse_connect

    from COMMON_CONST import DATA_DB_HFDATA

    return clickhouse_connect.get_client(**DATA_DB_HFDATA)


def extract_minute_features(
    start: str,
    end: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    lam: float = DEFAULT_WOI_LAMBDA,
    tables: Iterable[tuple] = SNAPSHOT_TABLES,
    client=None,
) -> pd.DataFrame:
    """Phase-1 long narrow frame (last-snapshot aggregation)."""
    own = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, has_withdraw in tables:
            sql = minute_last_feature_sql(
                table=table,
                exchange_suffix=suffix,
                start=start,
                end=end,
                lam=lam,
                symbols=symbols,
                has_withdraw=has_withdraw,
            )
            result = client.query(sql)
            wide = pd.DataFrame(
                result.result_rows, columns=list(result.column_names)
            )
            if wide.empty:
                continue
            wide["minute_time"] = pd.to_datetime(wide["minute_time"])
            long = wide.melt(
                id_vars=["minute_time", "symbol"],
                value_vars=[c for c in FACTOR_NAMES if c in wide.columns],
                var_name="factorname",
                value_name="value",
            )
            frames.append(long)
    finally:
        if own:
            client.close()

    if not frames:
        return pd.DataFrame(columns=list(NARROW_COLUMNS))

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"minute_time": "tradetime"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["symbol"] = out["symbol"].astype(str)
    out["factorname"] = out["factorname"].astype(str)
    return out[list(NARROW_COLUMNS)].sort_values(
        ["tradetime", "factorname", "symbol"]
    ).reset_index(drop=True)


def extract_minute_agg_wide(
    start: str,
    end: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    lam: float = DEFAULT_WOI_LAMBDA,
    tables: Iterable[tuple] = SNAPSHOT_TABLES,
    client=None,
    bartimes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Phase-2 wide minute panel with mean/max/std/sum_ratio columns."""
    own = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, has_withdraw in tables:
            sql = minute_agg_feature_sql(
                table=table,
                exchange_suffix=suffix,
                start=start,
                end=end,
                lam=lam,
                symbols=symbols,
                has_withdraw=has_withdraw,
                bartimes=bartimes,
            )
            result = client.query(sql)
            wide = pd.DataFrame(
                result.result_rows, columns=list(result.column_names)
            )
            if wide.empty:
                continue
            wide["minute_time"] = pd.to_datetime(wide["minute_time"])
            frames.append(wide)
    finally:
        if own:
            client.close()

    if not frames:
        cols = ["minute_time", "symbol", *PHASE2_WIDE_COLUMNS]
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    for col in PHASE2_WIDE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["minute_time", "symbol"]).reset_index(drop=True)
