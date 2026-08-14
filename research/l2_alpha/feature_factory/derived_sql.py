"""Rolling derived features over minute primitives (ClickHouse windows)."""

from __future__ import annotations

from typing import List, Optional, Sequence

from research.l2_alpha.feature_factory.primitives_sql import minute_primitives_sql
from research.l2_alpha.feature_factory.registry import (
    L2_BASE_TRANSFORMS,
    SSE_ONLY_BASES,
    factor_name,
)
from research.l2_alpha.schema import DEFAULT_WOI_LAMBDA


def _window_frame(window: int) -> str:
    # Inclusive current row: W minutes → W-1 preceding.
    preceding = max(window - 1, 0)
    return f"ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW"


def _derived_select_exprs(*, has_withdraw: bool) -> List[str]:
    exprs: List[str] = []
    for base, specs in L2_BASE_TRANSFORMS.items():
        if base in SSE_ONLY_BASES and not has_withdraw:
            for transform, window in specs:
                name = factor_name(base, transform, window)
                exprs.append(f"CAST(NULL AS Nullable(Float64)) AS {name}")
            continue
        for transform, window in specs:
            name = factor_name(base, transform, window)
            frame = _window_frame(window)
            order = (
                f"PARTITION BY symbol ORDER BY minute_time "
                f"{frame}"
            )
            if transform == "mean" and window == 1:
                exprs.append(f"{base} AS {name}")
            elif transform == "mean":
                exprs.append(f"avg({base}) OVER ({order}) AS {name}")
            elif transform == "std":
                exprs.append(f"stddevPop({base}) OVER ({order}) AS {name}")
            elif transform == "delta":
                exprs.append(
                    f"({base} - lagInFrame({base}, {window}) "
                    f"OVER (PARTITION BY symbol ORDER BY minute_time "
                    f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) "
                    f"AS {name}"
                )
            elif transform == "slope":
                # Simple endpoint slope over window (proxy for trend).
                exprs.append(
                    f"(({base} - lagInFrame({base}, {window - 1}) "
                    f"OVER (PARTITION BY symbol ORDER BY minute_time "
                    f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)) "
                    f"/ {window - 1}) AS {name}"
                )
            elif transform == "persistence":
                exprs.append(
                    f"avg(if({base} > 0, 1., 0.)) OVER ({order}) AS {name}"
                )
            elif transform == "zscore":
                exprs.append(
                    f"(({base} - avg({base}) OVER ({order})) "
                    f"/ nullIf(stddevPop({base}) OVER ({order}), 0)) AS {name}"
                )
            else:
                raise ValueError(f"Unknown transform {transform}")
    return exprs


def _bartime_filter_minutes(bartimes: Optional[Sequence[str]]) -> str:
    if not bartimes:
        return ""
    slots = []
    for bt in bartimes:
        hh, mm = str(bt).strip().split(":")[:2]
        slots.append(f"({int(hh)}, {int(mm)})")
    return (
        "WHERE (toHour(minute_time), toMinute(minute_time)) IN ("
        + ", ".join(slots)
        + ")"
    )


def derived_feature_sql(
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
    """Minute primitives → rolling derived → optional PREHEAT bartime filter.

    Rolling windows must run on the full-day minute series; bartime filter is
    applied only in the outer query.
    """
    primitives = minute_primitives_sql(
        table=table,
        exchange_suffix=exchange_suffix,
        start=start,
        end=end,
        lam=lam,
        symbols=symbols,
        has_withdraw=has_withdraw,
    )
    derived = ",\n    ".join(_derived_select_exprs(has_withdraw=has_withdraw))
    bartime_where = _bartime_filter_minutes(bartimes)
    return f"""
SELECT *
FROM (
    SELECT
        minute_time,
        symbol,
        {derived}
    FROM (
{primitives}
    )
)
{bartime_where}
"""
