"""Extract L2 Feature Factory wide panels from ClickHouse."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client
from research.l2_alpha.feature_factory.derived_sql import derived_feature_sql
from research.l2_alpha.feature_factory.registry import L2_FF_DERIVED_COLUMNS
from research.l2_alpha.schema import DEFAULT_WOI_LAMBDA, SNAPSHOT_TABLES


def extract_derived_wide(
    start: str,
    end: str,
    *,
    symbols: Optional[Sequence[str]] = None,
    bartimes: Optional[Sequence[str]] = None,
    lam: float = DEFAULT_WOI_LAMBDA,
    tables: Iterable[tuple] = SNAPSHOT_TABLES,
    client=None,
) -> pd.DataFrame:
    """Wide minute panel with CH-side derived columns (no CS ranks yet)."""
    own = client is None
    client = client or connect_hf_client()
    frames: List[pd.DataFrame] = []
    try:
        for table, suffix, has_withdraw in tables:
            sql = derived_feature_sql(
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
        cols = ["minute_time", "symbol", *L2_FF_DERIVED_COLUMNS]
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    return out
