"""Snapshot → 1-minute LOB primitives (ClickHouse server-side)."""

from __future__ import annotations

from typing import Optional, Sequence

from research.l2_alpha.clickhouse_ssl2 import snapshot_feature_select_sql
from research.l2_alpha.schema import DEFAULT_WOI_LAMBDA


def minute_primitives_sql(
    *,
    table: str,
    exchange_suffix: str,
    start: str,
    end: str,
    lam: float = DEFAULT_WOI_LAMBDA,
    symbols: Optional[Sequence[str]] = None,
    has_withdraw: bool = True,
) -> str:
    """Full-session minute bars (avg within minute) for rolling derived layer.

    Do not push PREHEAT bartime filters here — rolling needs continuous minutes.
    """
    inner = snapshot_feature_select_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        start=start,
        end=end,
        lam=lam,
        symbols=symbols,
        has_withdraw=has_withdraw,
        bartimes=None,
    )
    # cancel_imb: use flow-ratio of summed withdraws within the minute when
    # withdraw stats exist; else NULL (SZSE).
    if has_withdraw:
        cancel_expr = (
            "if(sum(cancel_total) > 0, "
            "sum(cancel_signed) / sum(cancel_total), NULL)"
        )
    else:
        cancel_expr = "CAST(NULL AS Nullable(Float64))"

    return f"""
SELECT
    toStartOfMinute(exch_time) AS minute_time,
    symbol,
    avg(depth_oi) AS depth_oi,
    avg(weighted_oi) AS weighted_oi,
    avg(micro_bias) AS micro_bias,
    avg(rel_spread) AS rel_spread,
    {cancel_expr} AS cancel_imb
FROM (
{inner}
)
GROUP BY minute_time, symbol
"""
