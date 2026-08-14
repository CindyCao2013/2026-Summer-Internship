# ============================================================
# core/l2_features/ideal_amplitude_active_v2_builder.py
# 缓存构建与面板生成
# ============================================================
"""IdealAmplitude_ActiveV2 panel builder — amplitude / active net vol.

Cache layout:
  research/cache/ideal_amplitude_active_v2/
    daily_brick/daily_YYYYMM.parquet
    factor_panel/IdealAmplitude_ActiveV2_*.parquet

Reuses minute_raw cache from smart_money_active_v2 via load_minute_active_raw.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from factor_cutting.ideal_amplitude_active_v2 import (
    FORMULA_VERSION,
    apply_amplitude_gate,
    compute_daily_amplitude,
    ewm_smooth_daily,
)
from factor_cutting.smart_money_active_v2 import apply_minute_qc
from factor_data_loaders import connect_ddb
from core.l2_features.smart_money_active_v2_builder import (
    _month_starts,
    _ym_tag,
    load_minute_active_raw,
)

CACHE_ROOT = Path("research/cache/ideal_amplitude_active_v2")
DAILY_BRICK_DIR = CACHE_ROOT / "daily_brick"
FACTOR_PANEL_DIR = CACHE_ROOT / "factor_panel"


def build_daily_brick(minutes: pd.DataFrame) -> pd.DataFrame:
    """分钟 → QC(复权) → 日度 amp_raw + 组件."""
    qc = apply_minute_qc(minutes)
    return compute_daily_amplitude(qc)


def ensure_daily_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    DAILY_BRICK_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    own = session is None
    s = session or connect_ddb()
    try:
        for c0, _c1 in _month_starts(start, end):
            path = DAILY_BRICK_DIR / f"daily_{_ym_tag(c0)}.parquet"
            if use_cache and path.exists() and not refresh_cache:
                parts.append(pd.read_parquet(path))
                continue
            m0 = dt.datetime(c0.year, c0.month, 1)
            if c0.month == 12:
                m1 = dt.datetime(c0.year + 1, 1, 1) - dt.timedelta(days=1)
            else:
                m1 = dt.datetime(c0.year, c0.month + 1, 1) - dt.timedelta(days=1)
            print(f"  minute→ideal_amp daily {_ym_tag(c0)} ...", flush=True)
            minutes = load_minute_active_raw(
                m0, m1, session=s, use_cache=use_cache, refresh_cache=refresh_cache
            )
            if minutes.empty:
                continue
            daily = build_daily_brick(minutes)
            daily.to_parquet(path, index=False)
            print(f"  wrote {path.name} rows={len(daily):,}", flush=True)
            parts.append(daily)
    finally:
        if own:
            s.close()
    if not parts:
        return pd.DataFrame(
            columns=["date", "symbol", "realized_amp", "active_net_vol", "amp_raw"]
        )
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return (
        out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def mask_limit_days(
    daily: pd.DataFrame,
    not_limit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """涨跌停日 amp_raw 置 NaN。"""
    if not_limit is None or daily.empty:
        return daily
    out = daily.copy()
    nl = not_limit.copy()
    nl.index = pd.to_datetime(nl.index)
    long_nl = nl.stack().rename("not_limit").reset_index()
    long_nl.columns = ["date", "symbol", "not_limit"]
    long_nl["date"] = pd.to_datetime(long_nl["date"])
    long_nl["symbol"] = long_nl["symbol"].astype(str)
    merged = out.merge(long_nl, on=["date", "symbol"], how="left")
    bad = merged["not_limit"].isna()
    if "amp_raw" in merged.columns:
        merged.loc[bad, "amp_raw"] = np.nan
    return merged.drop(columns=["not_limit"], errors="ignore")


def build_ideal_amplitude_panel(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    span: int = 5,
    min_periods: int = 3,
    preheat_calendar_days: int = 40,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    use_gate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回 wide (amp_smooth) + long."""
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    if symbols is not None:
        tag = f"{tag}_n{len(symbols)}"
    if use_gate:
        tag = f"{tag}_gate"
    wide_path = FACTOR_PANEL_DIR / f"IdealAmplitude_ActiveV2_{tag}.parquet"
    long_path = FACTOR_PANEL_DIR / f"IdealAmplitude_ActiveV2_long_{tag}.parquet"
    if use_cache and wide_path.exists() and long_path.exists() and not refresh_cache:
        return pd.read_parquet(wide_path), pd.read_parquet(long_path)

    brick_start = start - dt.timedelta(days=preheat_calendar_days)
    daily = ensure_daily_bricks(
        brick_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    if symbols is not None:
        daily = daily[daily["symbol"].isin(symbols)]
    daily = mask_limit_days(daily, not_limit=not_limit)
    if use_gate:
        daily = apply_amplitude_gate(daily)
    smoothed = ewm_smooth_daily(daily, span=span, min_periods=min_periods)
    smoothed = smoothed[
        (smoothed["date"] >= pd.Timestamp(start))
        & (smoothed["date"] <= pd.Timestamp(end))
    ]
    wide = smoothed.pivot(index="date", columns="symbol", values="amp_smooth").sort_index()
    wide.to_parquet(wide_path)
    smoothed.to_parquet(long_path, index=False)

    meta = {
        "formula_version": FORMULA_VERSION,
        "span": span,
        "min_periods": min_periods,
        "gate": use_gate,
        "n_dates": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "identity": "IdealAmplitude_ActiveV2 (not old price amplitude)",
    }
    (FACTOR_PANEL_DIR / f"IdealAmplitude_ActiveV2_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return wide, smoothed


def coverage_report(wide: pd.DataFrame) -> dict:
    if wide.empty:
        return {"n_days": 0, "n_symbols": 0, "coverage_cell": 0.0}
    finite = np.isfinite(wide.to_numpy(dtype=float))
    return {
        "n_days": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "coverage_cell": float(finite.mean()),
    }


def distribution_report(wide: pd.DataFrame) -> dict:
    if wide.empty:
        return {}
    vals = wide.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p50": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }
