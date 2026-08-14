"""Monthly parquet cache for active_pressure brick (+ session / smart).

Canonical caches:

  research/cache/bricks/active_pressure/daily_YYYYMM.parquet
  research/cache/bricks/active_pressure_session/daily_YYYYMM.parquet
  research/cache/bricks/active_pressure_smart/daily_YYYYMM.parquet
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.l2_features.bricks.active_pressure.pressure import (
    PRESSURE_COL,
    compute_daily_active_pressure,
)
from core.l2_features.bricks.active_pressure.pressure_enhanced import (
    SMARTV2_1_QUANTILE,
    compute_daily_apm_session,
    compute_daily_apm_smart,
    compute_daily_smart_apm_v2,
)
from factor_cutting.smart_money_active_v2 import apply_minute_qc
from core.l2_features.smart_money_active_v2_builder import (
    _filter_a_share,
    _month_starts,
    _ym_tag,
    load_minute_active_raw,
)
from factor_data_loaders import connect_ddb

CACHE_ROOT = Path("research/cache/bricks/active_pressure")
CACHE_ROOT_SESSION = Path("research/cache/bricks/active_pressure_session")
CACHE_ROOT_SMART = Path("research/cache/bricks/active_pressure_smart")
CACHE_ROOT_SMARTV2 = Path("research/cache/bricks/active_pressure_smartv2")
CACHE_ROOT_SMARTV2_1 = Path("research/cache/bricks/active_pressure_smartv2_1")

# Extra calendar days of minutes for smart rolling lookback across month edges
SMART_PREHEAT_CALENDAR_DAYS = 35


def _mask_halt(daily: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in qc.columns or daily.empty:
        return daily
    vol = qc.groupby(["symbol", "date"], sort=False)["volume"].sum().reset_index()
    out = daily.merge(vol, on=["symbol", "date"], how="left")
    halted = out["volume"].fillna(0) <= 0
    out.loc[halted, PRESSURE_COL] = np.nan
    return out


def build_active_pressure_daily_from_minutes(
    minutes: pd.DataFrame,
    min_minutes: int = 30,
) -> pd.DataFrame:
    qc = apply_minute_qc(minutes)
    daily = compute_daily_active_pressure(qc, min_minutes=min_minutes)
    return _mask_halt(daily, qc)


def build_session_pressure_daily_from_minutes(
    minutes: pd.DataFrame,
    min_minutes: int = 30,
) -> pd.DataFrame:
    qc = apply_minute_qc(minutes)
    daily = compute_daily_apm_session(qc, min_minutes=min_minutes)
    return _mask_halt(daily, qc)


def build_smart_pressure_daily_from_minutes(
    minutes: pd.DataFrame,
) -> pd.DataFrame:
    qc = apply_minute_qc(minutes)
    daily = compute_daily_apm_smart(qc)
    return _mask_halt(daily, qc)


def build_smartv2_pressure_daily_from_minutes(
    minutes: pd.DataFrame,
) -> pd.DataFrame:
    qc = apply_minute_qc(minutes)
    daily = compute_daily_smart_apm_v2(qc)
    return _mask_halt(daily, qc)


def build_smartv2_1_pressure_daily_from_minutes(
    minutes: pd.DataFrame,
) -> pd.DataFrame:
    qc = apply_minute_qc(minutes)
    daily = compute_daily_smart_apm_v2(qc, quantile=SMARTV2_1_QUANTILE)
    return _mask_halt(daily, qc)


def _ensure_variant_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    cache_root: Path,
    label: str,
    build_fn: Callable[[pd.DataFrame], pd.DataFrame],
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    minute_preheat_days: int = 0,
) -> pd.DataFrame:
    cache_root.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, _c1 in _month_starts(start, end):
            tag = _ym_tag(c0)
            path = cache_root / f"daily_{tag}.parquet"
            if use_cache and path.exists() and not refresh_cache:
                parts.append(pd.read_parquet(path))
                continue

            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            load_start = m0 - dt.timedelta(days=minute_preheat_days)
            print(f"  {label} brick {tag} (from minute cache/DDB) ...", flush=True)
            minutes = load_minute_active_raw(
                load_start,
                m1,
                session=s,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
            )
            if minutes.empty:
                continue
            minutes = _filter_a_share(minutes)
            daily = build_fn(minutes)
            # Persist only in-month rows
            daily = daily[
                (daily["date"] >= pd.Timestamp(m0))
                & (daily["date"] <= pd.Timestamp(m1))
            ]
            daily.to_parquet(path, index=False)
            print(f"  wrote {path.name} rows={len(daily):,}", flush=True)
            parts.append(daily)
    finally:
        if own:
            s.close()

    if not parts:
        return pd.DataFrame(columns=["date", "symbol", PRESSURE_COL])
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def ensure_active_pressure_daily_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Build / load monthly baseline active_pressure bricks."""
    return _ensure_variant_bricks(
        start,
        end,
        cache_root=CACHE_ROOT,
        label="active_pressure",
        build_fn=build_active_pressure_daily_from_minutes,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        minute_preheat_days=0,
    )


def ensure_active_pressure_session_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Build / load session-weighted active_pressure bricks."""
    return _ensure_variant_bricks(
        start,
        end,
        cache_root=CACHE_ROOT_SESSION,
        label="active_pressure_session",
        build_fn=build_session_pressure_daily_from_minutes,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        minute_preheat_days=0,
    )


def ensure_active_pressure_smart_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Build / load smart-filtered active_pressure bricks (needs lookback minutes)."""
    return _ensure_variant_bricks(
        start,
        end,
        cache_root=CACHE_ROOT_SMART,
        label="active_pressure_smart",
        build_fn=build_smart_pressure_daily_from_minutes,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        minute_preheat_days=SMART_PREHEAT_CALENDAR_DAYS,
    )


def ensure_active_pressure_smartv2_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Build / load SmartV2 bricks (quantile filter + buy/sell split)."""
    return _ensure_variant_bricks(
        start,
        end,
        cache_root=CACHE_ROOT_SMARTV2,
        label="active_pressure_smartv2",
        build_fn=build_smartv2_pressure_daily_from_minutes,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        minute_preheat_days=SMART_PREHEAT_CALENDAR_DAYS,
    )


def ensure_active_pressure_smartv2_1_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """SmartV2.1 bricks: size quantile=0.90 + buy/sell split."""
    return _ensure_variant_bricks(
        start,
        end,
        cache_root=CACHE_ROOT_SMARTV2_1,
        label="active_pressure_smartv2_1",
        build_fn=build_smartv2_1_pressure_daily_from_minutes,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        minute_preheat_days=SMART_PREHEAT_CALENDAR_DAYS,
    )
