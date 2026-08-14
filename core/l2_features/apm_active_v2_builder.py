"""APM_ActiveV2 panel builder — active_pressure brick → factor variants.

Cache layout:

  research/cache/bricks/active_pressure/daily_YYYYMM.parquet
  research/cache/bricks/active_pressure_session/daily_YYYYMM.parquet
  research/cache/bricks/active_pressure_smart/daily_YYYYMM.parquet
  research/cache/apm_active_v2/factor_panel/APM_ActiveV2_*.parquet

Minute raw is reused from SmartMoneyActiveV2 cache via ``load_minute_active_raw``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.l2_features.bricks.active_pressure import (
    BRICK_VERSION,
    PRESSURE_COL,
    PRESSURE_EWM_COL,
    ensure_active_pressure_daily_bricks,
    ensure_active_pressure_session_bricks,
    ensure_active_pressure_smart_bricks,
    ensure_active_pressure_smartv2_bricks,
    ensure_active_pressure_smartv2_1_bricks,
    smooth_active_pressure,
)
from core.l2_features.bricks.active_pressure.pressure_enhanced import (
    DELTA_LAG,
    SMARTV2_1F_ASC_MIN_RANK,
    SMARTV2_1F_EWM_SPAN,
    SMARTV2_1F_MIN_PERIODS,
    SMARTV2_1_ASC_MIN_RANK,
    SMARTV2_1_EWM_SPAN,
    SMARTV2_1_MIN_PERIODS,
    SMARTV2_ASC_MIN_RANK,
    SMARTV2_EWM_SPAN,
    SMARTV2_MIN_PERIODS,
    apply_asc_cs_gate,
    long_to_smooth_wide,
    delta_apm_wide,
    smooth_delta_wide,
)
from core.l2_features.bricks.active_size import (
    ACTIVE_SIZE_COL,
    ensure_active_size_daily_bricks,
)
from factor_cutting.apm_active_v2 import (
    EWM_MIN_PERIODS,
    EWM_SPAN,
    FORMULA_VERSION,
    to_weekly_hold,
    to_weekly_thu_hold,
)

CACHE_ROOT = Path("research/cache/apm_active_v2")
FACTOR_PANEL_DIR = CACHE_ROOT / "factor_panel"


def mask_limit_days(
    daily: pd.DataFrame,
    not_limit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """涨跌停日将 apm_raw 置 NaN。not_limit: 1=ok, NaN=limit."""
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
    if PRESSURE_COL in merged.columns:
        merged.loc[bad, PRESSURE_COL] = np.nan
    return merged.drop(columns=["not_limit"], errors="ignore")


def _filter_symbols(daily: pd.DataFrame, symbols: Optional[list]) -> pd.DataFrame:
    if symbols is None or daily.empty:
        return daily
    sym_set = set(str(x) for x in symbols)
    return daily[daily["symbol"].isin(sym_set)]


def build_apm_active_v2_panel(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    span: int = EWM_SPAN,
    min_periods: int = EWM_MIN_PERIODS,
    preheat_calendar_days: int = 30,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide + long panels. wide values = apm_smooth (EWM pressure)."""
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_pressure_v1"
    if symbols is not None:
        tag = f"{tag}_n{len(symbols)}"
    wide_path = FACTOR_PANEL_DIR / f"APM_ActiveV2_{tag}.parquet"
    long_path = FACTOR_PANEL_DIR / f"APM_ActiveV2_long_{tag}.parquet"
    if use_cache and wide_path.exists() and long_path.exists() and not refresh_cache:
        return pd.read_parquet(wide_path), pd.read_parquet(long_path)

    brick_start = start - dt.timedelta(days=preheat_calendar_days)
    daily = ensure_active_pressure_daily_bricks(
        brick_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    daily = _filter_symbols(daily, symbols)
    daily = mask_limit_days(daily, not_limit=not_limit)
    smoothed = smooth_active_pressure(daily, span=span, min_periods=min_periods)
    smoothed = smoothed[
        (smoothed["date"] >= pd.Timestamp(start))
        & (smoothed["date"] <= pd.Timestamp(end))
    ]
    wide = smoothed.pivot(
        index="date", columns="symbol", values=PRESSURE_EWM_COL
    ).sort_index()
    wide.to_parquet(wide_path)
    smoothed.to_parquet(long_path, index=False)

    meta = {
        "formula_version": FORMULA_VERSION,
        "brick_version": BRICK_VERSION,
        "ewm_span": span,
        "ewm_min_periods": min_periods,
        "n_dates": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "identity": "APM_ActiveV2 = Active Pressure Metric (not price APM; not session-cut)",
    }
    (FACTOR_PANEL_DIR / f"APM_ActiveV2_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return wide, smoothed


def build_apm_enhanced_variants(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 40,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    names: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """Session / Smart / Delta / SmartV2 enhanced APM panels (pre-CS)."""
    want = set(names) if names is not None else {
        "APM_ActiveV2_Session",
        "APM_ActiveV2_Smart",
        "APM_ActiveV2_Delta",
        "APM_ActiveV2_SmartV2",
        "APM_ActiveV2_SmartV2_1F",
        "APM_ActiveV2_SmartV2_1",
    }
    brick_start = start - dt.timedelta(days=preheat_calendar_days)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    out: Dict[str, pd.DataFrame] = {}

    if "APM_ActiveV2_Session" in want:
        sess_daily = ensure_active_pressure_session_bricks(
            brick_start,
            end,
            session=session,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        sess_daily = mask_limit_days(_filter_symbols(sess_daily, symbols), not_limit)
        out["APM_ActiveV2_Session"] = long_to_smooth_wide(sess_daily, start=t0, end=t1)

    if "APM_ActiveV2_Smart" in want:
        smart_daily = ensure_active_pressure_smart_bricks(
            brick_start,
            end,
            session=session,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        smart_daily = mask_limit_days(_filter_symbols(smart_daily, symbols), not_limit)
        out["APM_ActiveV2_Smart"] = long_to_smooth_wide(smart_daily, start=t0, end=t1)

    if "APM_ActiveV2_Delta" in want:
        base_daily = ensure_active_pressure_daily_bricks(
            brick_start,
            end,
            session=session,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        base_daily = mask_limit_days(_filter_symbols(base_daily, symbols), not_limit)
        raw_wide = (
            base_daily.pivot(index="date", columns="symbol", values=PRESSURE_COL).sort_index()
            if not base_daily.empty
            else pd.DataFrame()
        )
        delta_raw = delta_apm_wide(raw_wide, lag=DELTA_LAG)
        delta_smooth = smooth_delta_wide(
            delta_raw, span=EWM_SPAN, min_periods=EWM_MIN_PERIODS
        )
        if not delta_smooth.empty:
            out["APM_ActiveV2_Delta"] = delta_smooth.loc[
                (delta_smooth.index >= t0) & (delta_smooth.index <= t1)
            ]
        else:
            out["APM_ActiveV2_Delta"] = delta_smooth

    for fname, profile in (
        ("APM_ActiveV2_SmartV2", "v2"),
        ("APM_ActiveV2_SmartV2_1F", "v2_1f"),
        ("APM_ActiveV2_SmartV2_1", "v2_1"),
    ):
        if fname in want:
            out[fname] = build_smartv2_panel(
                start,
                end,
                session=session,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                preheat_calendar_days=preheat_calendar_days,
                not_limit=not_limit,
                symbols=symbols,
                profile=profile,
            )

    return out


def build_smartv2_panel(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 40,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    profile: str = "v2",
    asc_min_rank: Optional[float] = None,
    ewm_span: Optional[int] = None,
    ewm_min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """SmartV2 family panels.

    Profiles
    --------
    v2    : q80 brick + ASC>=50% + EWM span=2  (original V2)
    v2_1f : q80 brick + ASC off + EWM span=5   (fast ablate; reuse V2 brick)
    v2_1  : q90 brick + ASC off + EWM span=5   (full V2.1 hotfix)
    """
    brick_start = start - dt.timedelta(days=preheat_calendar_days)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)

    if profile == "v2":
        ensure_fn = ensure_active_pressure_smartv2_bricks
        span = SMARTV2_EWM_SPAN if ewm_span is None else ewm_span
        min_periods = SMARTV2_MIN_PERIODS if ewm_min_periods is None else ewm_min_periods
        gate = SMARTV2_ASC_MIN_RANK if asc_min_rank is None else asc_min_rank
    elif profile == "v2_1f":
        ensure_fn = ensure_active_pressure_smartv2_bricks
        span = SMARTV2_1F_EWM_SPAN if ewm_span is None else ewm_span
        min_periods = SMARTV2_1F_MIN_PERIODS if ewm_min_periods is None else ewm_min_periods
        gate = SMARTV2_1F_ASC_MIN_RANK if asc_min_rank is None else asc_min_rank
    elif profile == "v2_1":
        ensure_fn = ensure_active_pressure_smartv2_1_bricks
        span = SMARTV2_1_EWM_SPAN if ewm_span is None else ewm_span
        min_periods = SMARTV2_1_MIN_PERIODS if ewm_min_periods is None else ewm_min_periods
        gate = SMARTV2_1_ASC_MIN_RANK if asc_min_rank is None else asc_min_rank
    else:
        raise ValueError(f"Unknown SmartV2 profile: {profile}")

    daily = ensure_fn(
        brick_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    daily = mask_limit_days(_filter_symbols(daily, symbols), not_limit)

    if gate > 0:
        asc_daily = ensure_active_size_daily_bricks(
            brick_start,
            end,
            session=session,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        asc_daily = _filter_symbols(asc_daily, symbols)
        if not asc_daily.empty and ACTIVE_SIZE_COL in asc_daily.columns:
            asc_wide = asc_daily.pivot(
                index="date", columns="symbol", values=ACTIVE_SIZE_COL
            ).sort_index()
            daily = apply_asc_cs_gate(daily, asc_wide, min_rank=gate)

    return long_to_smooth_wide(
        daily,
        start=t0,
        end=t1,
        span=span,
        min_periods=min_periods,
    )


def build_apm_raw_variants(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 30,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    include_enhanced: bool = True,
    names: Optional[list] = None,
) -> Dict[str, pd.DataFrame]:
    """Return raw (pre-CS) panels; ``names`` limits which variants are built."""
    want = set(names) if names is not None else None
    baseline = {
        "APM_ActiveV2",
        "APM_ActiveV2_Raw",
        "APM_ActiveV2_Weekly",
        "APM_ActiveV2_Weekly_Thu",
    }
    enhanced = {
        "APM_ActiveV2_Session",
        "APM_ActiveV2_Smart",
        "APM_ActiveV2_Delta",
        "APM_ActiveV2_SmartV2",
        "APM_ActiveV2_SmartV2_1F",
        "APM_ActiveV2_SmartV2_1",
    }
    need_baseline = True if want is None else bool(want & baseline)
    need_enhanced = include_enhanced and (True if want is None else bool(want & enhanced))

    out: Dict[str, pd.DataFrame] = {}
    if need_baseline:
        wide_smooth, long = build_apm_active_v2_panel(
            start,
            end,
            session=session,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            preheat_calendar_days=preheat_calendar_days,
            not_limit=not_limit,
            symbols=symbols,
        )
        raw_wide = (
            long.pivot(index="date", columns="symbol", values=PRESSURE_COL).sort_index()
            if not long.empty and PRESSURE_COL in long.columns
            else wide_smooth.copy() * np.nan
        )
        cand = {
            "APM_ActiveV2": wide_smooth,
            "APM_ActiveV2_Raw": raw_wide,
            "APM_ActiveV2_Weekly": to_weekly_hold(wide_smooth, method="friday"),
            "APM_ActiveV2_Weekly_Thu": to_weekly_thu_hold(wide_smooth, agg="mean"),
        }
        for k, v in cand.items():
            if want is None or k in want:
                out[k] = v

    if need_enhanced:
        enh_preheat = max(preheat_calendar_days, 40)
        enh_names = None if want is None else sorted(want & enhanced)
        out.update(
            build_apm_enhanced_variants(
                start,
                end,
                session=session,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
                preheat_calendar_days=enh_preheat,
                not_limit=not_limit,
                symbols=symbols,
                names=enh_names,
            )
        )
    return out


def coverage_report(wide: pd.DataFrame) -> dict:
    if wide.empty:
        return {"n_days": 0, "n_symbols": 0, "coverage_cell": 0.0}
    finite = np.isfinite(wide.to_numpy(dtype=float))
    return {
        "n_days": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "coverage_cell": float(finite.mean()),
        "mean_daily_coverage": float(finite.mean(axis=1).mean()) if wide.shape[0] else 0.0,
    }


def distribution_report(wide: pd.DataFrame) -> dict:
    vals = wide.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0}
    s = pd.Series(vals)
    return {
        "n": int(len(vals)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p01": float(s.quantile(0.01)),
        "p50": float(s.quantile(0.50)),
        "p99": float(s.quantile(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }
