"""Monthly parquet cache for active_size brick.

Canonical cache:

  research/cache/bricks/active_size/daily_YYYYMM.parquet

Legacy IdealReversal path (read fallback):

  research/cache/ideal_reversal_active_v2/daily_brick/daily_YYYYMM.parquet
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.l2_features.bricks.active_size.concentration import (
    ACTIVE_SIZE_COL,
    compute_daily_active_size_concentration,
    normalize_active_size_columns,
)
from factor_cutting.smart_money_active_v2 import apply_minute_qc
from core.l2_features.smart_money_active_v2_builder import (
    _filter_a_share,
    _month_starts,
    _ym_tag,
    load_minute_active_raw,
)
from factor_data_loaders import connect_ddb

CACHE_ROOT = Path("research/cache/bricks/active_size")
LEGACY_IDEAL_REV_DIR = Path("research/cache/ideal_reversal_active_v2/daily_brick")


def build_active_size_daily_from_minutes(
    minutes: pd.DataFrame,
    min_minutes: int = 30,
) -> pd.DataFrame:
    qc = apply_minute_qc(minutes)
    daily = compute_daily_active_size_concentration(qc, min_minutes=min_minutes)
    if "volume" in qc.columns:
        vol = qc.groupby(["symbol", "date"], sort=False)["volume"].sum().reset_index()
        daily = daily.merge(vol, on=["symbol", "date"], how="left")
        halted = daily["volume"].fillna(0) <= 0
        daily.loc[halted, ACTIVE_SIZE_COL] = np.nan
    return daily


def _load_month_parquet(path: Path) -> pd.DataFrame:
    return normalize_active_size_columns(pd.read_parquet(path))


def ensure_active_size_daily_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Build / load monthly active_size concentration bricks."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, _c1 in _month_starts(start, end):
            tag = _ym_tag(c0)
            path = CACHE_ROOT / f"daily_{tag}.parquet"
            legacy = LEGACY_IDEAL_REV_DIR / f"daily_{tag}.parquet"

            if use_cache and path.exists() and not refresh_cache:
                parts.append(_load_month_parquet(path))
                continue

            # promote legacy IdealReversal brick (inst_ratio → rename)
            if use_cache and legacy.exists() and not refresh_cache:
                daily = _load_month_parquet(legacy)
                daily.to_parquet(path, index=False)
                parts.append(daily)
                continue

            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            print(f"  active_size brick {tag} (from minute cache/DDB) ...", flush=True)
            minutes = load_minute_active_raw(
                m0, m1, session=s, use_cache=use_cache, refresh_cache=refresh_cache
            )
            if minutes.empty:
                continue
            minutes = _filter_a_share(minutes)
            daily = build_active_size_daily_from_minutes(minutes)
            daily.to_parquet(path, index=False)
            print(f"  wrote {path.name} rows={len(daily):,}", flush=True)
            parts.append(daily)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(columns=["date", "symbol", ACTIVE_SIZE_COL])
    out = pd.concat(parts, ignore_index=True)
    out = normalize_active_size_columns(out)
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
