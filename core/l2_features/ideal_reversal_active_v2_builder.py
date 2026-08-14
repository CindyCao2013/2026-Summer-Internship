"""IdealReversal_ActiveV2 panel builder.

Uses shared brick ``core.l2_features.bricks.active_size`` for the daily
``active_size_concentration`` feature (not \"institutional participation\").

Cache:
  research/cache/bricks/active_size/daily_YYYYMM.parquet   # shared brick
  research/cache/ideal_reversal_active_v2/factor_panel/    # factor only
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.l2_features.bricks.active_size import (
    ACTIVE_SIZE_COL,
    ACTIVE_SIZE_EWM_COL,
    TOP_SIZE_PCT,
    ensure_active_size_daily_bricks,
    smooth_active_size_concentration,
)
from core.l2_features.bricks.active_size.concentration import (
    BRICK_VERSION,
    EWM_MIN_PERIODS,
    EWM_SPAN,
    normalize_active_size_columns,
)
from factor_cutting.ideal_reversal_active_v2 import (
    FORMULA_VERSION,
    REVERSAL_WINDOW,
    ReversalMode,
    WeeklyMethod,
    build_reversal_factor,
    build_reversal_factor_v2,
    to_weekly_hold,
    to_weekly_thu_hold,
)

CACHE_ROOT = Path("research/cache/ideal_reversal_active_v2")
FACTOR_PANEL_DIR = CACHE_ROOT / "factor_panel"


def mask_limit_days(
    daily: pd.DataFrame,
    not_limit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if not_limit is None or daily.empty:
        return daily
    out = normalize_active_size_columns(daily)
    nl = not_limit.copy()
    nl.index = pd.to_datetime(nl.index)
    long_nl = nl.stack().rename("not_limit").reset_index()
    long_nl.columns = ["date", "symbol", "not_limit"]
    long_nl["date"] = pd.to_datetime(long_nl["date"])
    long_nl["symbol"] = long_nl["symbol"].astype(str)
    merged = out.merge(long_nl, on=["date", "symbol"], how="left")
    bad = merged["not_limit"].isna()
    if ACTIVE_SIZE_COL in merged.columns:
        merged.loc[bad, ACTIVE_SIZE_COL] = np.nan
    return merged.drop(columns=["not_limit"], errors="ignore")


def ensure_daily_bricks(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Delegate to shared active_size brick (promotes legacy IdealRev cache)."""
    return ensure_active_size_daily_bricks(
        start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )


ensure_daily_inst_bricks = ensure_daily_bricks  # deprecated alias


def _align_close_asc(
    close: pd.DataFrame,
    asc_smooth: pd.DataFrame,
    symbols: Optional[list],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    asc_wide = (
        asc_smooth.pivot(index="date", columns="symbol", values=ACTIVE_SIZE_EWM_COL)
        .sort_index()
    )
    asc_wide.index = pd.to_datetime(asc_wide.index)
    asc_wide.columns = asc_wide.columns.astype(str)

    close_use = close.copy()
    close_use.index = pd.to_datetime(close_use.index)
    close_use.columns = close_use.columns.astype(str)
    if symbols is not None:
        cols = [c for c in close_use.columns if c in set(str(x) for x in symbols)]
        close_use = close_use[cols]

    common_cols = close_use.columns.intersection(asc_wide.columns)
    all_idx = close_use.index.union(asc_wide.index).sort_values()
    return (
        close_use.reindex(index=all_idx, columns=common_cols),
        asc_wide.reindex(index=all_idx, columns=common_cols),
    )


def _load_asc_long(
    start: dt.datetime,
    end: dt.datetime,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    preheat_calendar_days: int = 60,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
) -> pd.DataFrame:
    brick_start = start - dt.timedelta(days=preheat_calendar_days)
    daily = ensure_active_size_daily_bricks(
        brick_start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    daily = normalize_active_size_columns(daily)
    if symbols is not None:
        sym_set = set(str(x) for x in symbols)
        daily = daily[daily["symbol"].isin(sym_set)]
    daily = mask_limit_days(daily, not_limit=not_limit)
    return smooth_active_size_concentration(
        daily, span=EWM_SPAN, min_periods=EWM_MIN_PERIODS
    )


def build_ideal_reversal_raw_variants(
    start: dt.datetime,
    end: dt.datetime,
    close: pd.DataFrame,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    window: int = REVERSAL_WINDOW,
    preheat_calendar_days: int = 60,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    weekly_method: WeeklyMethod = "friday",
    icir_weights: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """Build daily / weekly / Thu / RollingGate raw panels.

    Keys include legacy Friday weekly variants plus:
      IdealReversal_ActiveV2_Weekly_Thu
      IdealReversal_ActiveV2_RollingGate
      IdealReversal_ActiveV2_Weekly_Thu_RollingGate
    """
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        f"_w{window}_asc_ewm{EWM_SPAN}_{weekly_method}_v3"
    )
    if symbols is not None:
        tag = f"{tag}_n{len(symbols)}"

    names = [
        "IdealReversal_ActiveV2",
        "IdealReversal_ActiveV2_Weekly",
        "IdealReversal_ActiveV2_PureRev",
        "IdealReversal_ActiveV2_Weekly_PureRev",
        "IdealReversal_ActiveV2_Weekly_Thu",
        "IdealReversal_ActiveV2_RollingGate",
        "IdealReversal_ActiveV2_Weekly_Thu_RollingGate",
    ]
    paths = {n: FACTOR_PANEL_DIR / f"{n}_{tag}.parquet" for n in names}
    if (
        use_cache
        and (not refresh_cache)
        and all(p.exists() for p in paths.values())
    ):
        return {n: pd.read_parquet(paths[n]) for n in names}

    asc_smooth = _load_asc_long(
        start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        preheat_calendar_days=preheat_calendar_days,
        not_limit=not_limit,
        symbols=symbols,
    )
    close_al, asc_al = _align_close_asc(close, asc_smooth, symbols)

    gated = build_reversal_factor(close_al, asc_al, window=window, mode="asc_gate")
    pure = build_reversal_factor(close_al, asc_al, window=window, mode="pure_rev")
    pure_mw = build_reversal_factor_v2(
        close_al,
        asc_al,
        windows=[3, 5, 10],
        gate_type="none",
        icir_weights=icir_weights,
    )
    rolling = build_reversal_factor_v2(
        close_al,
        asc_al,
        windows=[3, 5, 10],
        gate_type="rolling_rank",
        gate_threshold=0.6,
        gate_roll=20,
        icir_weights=icir_weights,
    )

    def _clip(w: pd.DataFrame) -> pd.DataFrame:
        return w.loc[
            (w.index >= pd.Timestamp(start)) & (w.index <= pd.Timestamp(end))
        ]

    gated, pure = _clip(gated), _clip(pure)
    pure_mw, rolling = _clip(pure_mw), _clip(rolling)

    out = {
        "IdealReversal_ActiveV2": gated,
        "IdealReversal_ActiveV2_Weekly": to_weekly_hold(gated, method=weekly_method),
        "IdealReversal_ActiveV2_PureRev": pure,
        "IdealReversal_ActiveV2_Weekly_PureRev": to_weekly_hold(
            pure, method=weekly_method
        ),
        # Mon–Thu mean signal, Thu placement + ffill (fresher than Friday-only)
        "IdealReversal_ActiveV2_Weekly_Thu": to_weekly_thu_hold(pure_mw, agg="mean"),
        "IdealReversal_ActiveV2_RollingGate": rolling,
        "IdealReversal_ActiveV2_Weekly_Thu_RollingGate": to_weekly_thu_hold(
            rolling, agg="mean"
        ),
    }

    for name, wide in out.items():
        wide.to_parquet(paths[name])

    meta = {
        "formula_version": FORMULA_VERSION,
        "brick_version": BRICK_VERSION,
        "weekly_method": weekly_method,
        "modes": [
            "asc_gate",
            "pure_rev",
            "weekly_thu_pure_mw",
            "rolling_rank_mw",
            "weekly_thu_rolling",
        ],
        "identity_note": (
            "ASC = active_size_concentration (observable), not institutional participation; "
            "Weekly_Thu uses Mon-Thu mean placed on Thursday; "
            "RollingGate uses CS ASC rank rolling mean gate + windows [3,5,10]"
        ),
        "n_dates": int(gated.shape[0]),
        "n_symbols": int(gated.shape[1]),
    }
    (FACTOR_PANEL_DIR / f"IdealReversal_variants_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return out


def build_ideal_reversal_active_v2_panel(
    start: dt.datetime,
    end: dt.datetime,
    close: pd.DataFrame,
    *,
    session=None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    window: int = REVERSAL_WINDOW,
    preheat_calendar_days: int = 60,
    not_limit: Optional[pd.DataFrame] = None,
    symbols: Optional[list] = None,
    mode: ReversalMode = "asc_gate",
    freq: str = "D",
    weekly_method: WeeklyMethod = "friday",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one wide factor + long ASC.

    freq: 'D' daily | 'W' weekly hold (friday / every_5d / week_mean via weekly_method)
    mode: 'asc_gate' | 'pure_rev'
    """
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        f"_w{window}_asc_ewm{EWM_SPAN}_{mode}_{freq}_{weekly_method}"
    )
    if symbols is not None:
        tag = f"{tag}_n{len(symbols)}"
    wide_path = FACTOR_PANEL_DIR / f"IdealReversal_ActiveV2_{tag}.parquet"
    long_path = FACTOR_PANEL_DIR / f"IdealReversal_ActiveV2_long_{tag}.parquet"
    if use_cache and wide_path.exists() and long_path.exists() and not refresh_cache:
        return pd.read_parquet(wide_path), pd.read_parquet(long_path)

    asc_smooth = _load_asc_long(
        start,
        end,
        session=session,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        preheat_calendar_days=preheat_calendar_days,
        not_limit=not_limit,
        symbols=symbols,
    )
    close_al, asc_al = _align_close_asc(close, asc_smooth, symbols)
    wide = build_reversal_factor(close_al, asc_al, window=window, mode=mode)
    wide = wide.loc[
        (wide.index >= pd.Timestamp(start)) & (wide.index <= pd.Timestamp(end))
    ]
    if str(freq).upper().startswith("W"):
        wide = to_weekly_hold(wide, method=weekly_method)

    long_out = asc_smooth[
        (asc_smooth["date"] >= pd.Timestamp(start))
        & (asc_smooth["date"] <= pd.Timestamp(end))
    ]

    wide.to_parquet(wide_path)
    long_out.to_parquet(long_path, index=False)
    meta = {
        "formula_version": FORMULA_VERSION,
        "brick_version": BRICK_VERSION,
        "feature_name": ACTIVE_SIZE_COL,
        "mode": mode,
        "freq": freq,
        "weekly_method": weekly_method if str(freq).upper().startswith("W") else None,
        "identity_note": (
            "active_size_concentration is an observable (avg-size concentration), "
            "NOT institutional participation"
        ),
        "top_size_pct": TOP_SIZE_PCT,
        "asc_ewm_span": EWM_SPAN,
        "reversal_window": window,
        "n_dates": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
    }
    (FACTOR_PANEL_DIR / f"IdealReversal_ActiveV2_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return wide, long_out


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
