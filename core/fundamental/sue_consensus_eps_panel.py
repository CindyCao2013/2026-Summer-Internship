"""SUE_ConsensusEPS panel builder — III-B1 Phase1.

Identity: (EPS_actual - EPS_consensus) / |EPS_consensus|
PIT: est_dt < known_dt; signal_date >= known_dt
known_dt: first date when EPS is public (min express/income with finite EPS).
  notice_dt retained for provenance; NP-only notice is not an EPS-known event.
Panel v0: impulse only (no hold/decay lock)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sue_data import load_sue_raw_bundle

CACHE_ROOT = Path("research/cache/sue_consensus_eps")
EVENTS_DIR = CACHE_ROOT / "events"
PANELS_DIR = CACHE_ROOT / "panels"
META_DIR = CACHE_ROOT / "meta"
SUE_P0_ROOT = Path("research/cache/sue_p0")

FORMULA_VERSION = "sue_consensus_eps_v1_impulse"
EPS_CONS_MIN_ABS = 1e-6

EVENT_COLS = [
    "symbol",
    "fiscal_period",
    "actual_eps",
    "consensus_eps",
    "est_dt",
    "notice_dt",
    "express_dt",
    "income_dt",
    "known_dt",
    "sue",
    "source_actual",
]


def _source_dates(
    notice: pd.DataFrame,
    express: pd.DataFrame,
    income: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (symbol, fiscal_period) with notice/express/income dates + EPS as-of known."""
    frames = []
    for df, src in ((notice, "notice"), (express, "express"), (income, "income")):
        if df is None or df.empty:
            continue
        t = df.copy()
        t["fiscal_period"] = t["report_period"].astype(str)
        t["source"] = src
        if "eps" not in t.columns:
            t["eps"] = np.nan
        frames.append(
            t[["symbol", "fiscal_period", "known_date", "eps", "source"]].rename(
                columns={"known_date": "disc_dt"}
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "fiscal_period",
                "notice_dt",
                "express_dt",
                "income_dt",
                "known_dt",
                "actual_eps",
                "source_actual",
            ]
        )

    all_ev = pd.concat(frames, ignore_index=True)
    all_ev["disc_dt"] = pd.to_datetime(all_ev["disc_dt"])
    all_ev["symbol"] = all_ev["symbol"].astype(str)

    # pivot first date per source
    date_wide = (
        all_ev.groupby(["symbol", "fiscal_period", "source"], as_index=False)["disc_dt"]
        .min()
        .pivot(index=["symbol", "fiscal_period"], columns="source", values="disc_dt")
        .reset_index()
    )
    for c in ("notice", "express", "income"):
        if c not in date_wide.columns:
            date_wide[c] = pd.NaT
    date_wide = date_wide.rename(
        columns={"notice": "notice_dt", "express": "express_dt", "income": "income_dt"}
    )
    # ConsensusEPS: known_dt = first date when EPS is public (express/income), not NP-only notice.
    # notice_dt kept for provenance; notice-only rows without EPS do not create SUE events.
    eps_dates = []
    for df, src in ((express, "express"), (income, "income")):
        if df is None or df.empty:
            continue
        t = df.dropna(subset=["eps"]).copy()
        if t.empty:
            continue
        t["fiscal_period"] = t["report_period"].astype(str)
        t = t.groupby(["symbol", "fiscal_period"], as_index=False)["known_date"].min()
        t["source"] = src
        eps_dates.append(t.rename(columns={"known_date": "eps_dt"}))
    if not eps_dates:
        date_wide["known_dt"] = pd.NaT
        date_wide["actual_eps"] = np.nan
        date_wide["source_actual"] = None
        return date_wide

    ed = pd.concat(eps_dates, ignore_index=True)
    # earliest EPS disclosure date
    known_eps = ed.groupby(["symbol", "fiscal_period"], as_index=False)["eps_dt"].min()
    date_wide = date_wide.merge(known_eps, on=["symbol", "fiscal_period"], how="left")
    date_wide["known_dt"] = date_wide["eps_dt"]
    date_wide = date_wide.drop(columns=["eps_dt"])

    # actual EPS on that known_dt (prefer income if both same day)
    rank = {"income": 3, "express": 2, "notice": 1}
    on_known = all_ev.merge(
        date_wide[["symbol", "fiscal_period", "known_dt"]],
        on=["symbol", "fiscal_period"],
        how="inner",
    )
    on_known = on_known[on_known["disc_dt"] == on_known["known_dt"]].copy()
    on_known["src_rank"] = on_known["source"].map(rank).fillna(0)
    on_known = on_known.dropna(subset=["eps"])
    if on_known.empty:
        date_wide["actual_eps"] = np.nan
        date_wide["source_actual"] = None
    else:
        best = (
            on_known.sort_values(["symbol", "fiscal_period", "src_rank"])
            .groupby(["symbol", "fiscal_period"], as_index=False)
            .last()
        )
        date_wide = date_wide.merge(
            best[["symbol", "fiscal_period", "eps", "source"]].rename(
                columns={"eps": "actual_eps", "source": "source_actual"}
            ),
            on=["symbol", "fiscal_period"],
            how="left",
        )
    return date_wide


def build_sue_consensus_eps_events(
    bundle: dict,
    *,
    eps_cons_min_abs: float = EPS_CONS_MIN_ABS,
) -> Tuple[pd.DataFrame, dict]:
    """Build provenance-rich SUE_ConsensusEPS event table + PIT audit."""
    disc = _source_dates(bundle.get("notice"), bundle.get("express"), bundle.get("income"))
    disc = disc.dropna(subset=["known_dt", "actual_eps"])
    if disc.empty:
        empty = pd.DataFrame(columns=EVENT_COLS)
        return empty, {"n_events": 0, "frac_est_before_known": np.nan, "hard_fail": True}

    cons = bundle.get("consensus")
    if cons is None or cons.empty:
        empty = pd.DataFrame(columns=EVENT_COLS)
        return empty, {"n_events": 0, "error": "no_consensus", "hard_fail": True}

    c = cons.dropna(subset=["est_dt", "eps_avg"]).copy()
    c["fiscal_period"] = c["report_period"].astype(str)
    c["symbol"] = c["symbol"].astype(str)
    c["est_dt"] = pd.to_datetime(c["est_dt"])
    c = c.sort_values("est_dt").drop_duplicates(
        ["symbol", "fiscal_period", "est_dt"], keep="last"
    )

    merged = disc.merge(
        c[["symbol", "fiscal_period", "est_dt", "eps_avg"]],
        on=["symbol", "fiscal_period"],
        how="inner",
    )
    n_before_filter = len(merged)
    valid = merged["est_dt"] < merged["known_dt"]
    n_rejected = int((~valid).sum()) if n_before_filter else 0
    merged = merged.loc[valid]
    if merged.empty:
        empty = pd.DataFrame(columns=EVENT_COLS)
        return empty, {
            "n_events": 0,
            "n_rejected_est_ge_known": n_rejected,
            "frac_est_before_known": 0.0,
            "hard_fail": True,
        }

    # last consensus before known_dt
    idx = merged.groupby(["symbol", "fiscal_period", "known_dt"])["est_dt"].idxmax()
    best = merged.loc[idx].copy()
    eps_c = best["eps_avg"].astype(float)
    ok = eps_c.notna() & (eps_c.abs() >= eps_cons_min_abs)
    best = best.loc[ok].copy()
    eps_c = eps_c.loc[ok]
    best["consensus_eps"] = eps_c
    best["actual_eps"] = best["actual_eps"].astype(float)
    best["sue"] = (best["actual_eps"] - best["consensus_eps"]) / best["consensus_eps"].abs()

    out = best.rename(columns={"known_dt": "known_dt"}).copy()
    # ensure columns
    for col in ("notice_dt", "express_dt", "income_dt"):
        if col not in out.columns:
            out[col] = pd.NaT
    out = out[
        [
            "symbol",
            "fiscal_period",
            "actual_eps",
            "consensus_eps",
            "est_dt",
            "notice_dt",
            "express_dt",
            "income_dt",
            "known_dt",
            "sue",
            "source_actual",
        ]
    ]
    # one event per symbol-period (keep earliest known if dup)
    out = (
        out.sort_values(["symbol", "fiscal_period", "known_dt"])
        .drop_duplicates(["symbol", "fiscal_period"], keep="first")
        .reset_index(drop=True)
    )

    # PIT hard checks
    pit_ok = (out["est_dt"] < out["known_dt"]).all()
    audit = {
        "formula_version": FORMULA_VERSION,
        "n_events": int(len(out)),
        "n_rejected_est_ge_known": n_rejected,
        "frac_est_before_known": 1.0 if pit_ok else float((out["est_dt"] < out["known_dt"]).mean()),
        "pit_hard_pass": bool(pit_ok),
        "hard_fail": not bool(pit_ok),
        "n_symbols": int(out["symbol"].nunique()),
        "known_dt_min": str(out["known_dt"].min().date()) if len(out) else None,
        "known_dt_max": str(out["known_dt"].max().date()) if len(out) else None,
        "source_actual_mix": out["source_actual"].value_counts(dropna=False).to_dict(),
        "sue_describe": {
            "mean": float(out["sue"].mean()),
            "std": float(out["sue"].std()),
            "p01": float(out["sue"].quantile(0.01)),
            "p50": float(out["sue"].quantile(0.50)),
            "p99": float(out["sue"].quantile(0.99)),
        },
    }
    return out, audit


def events_to_impulse_panel(
    events: pd.DataFrame,
    trade_index: pd.DatetimeIndex,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Map known_dt → first trading day >= known_dt; impulse elsewhere NaN."""
    if events is None or events.empty:
        cols = list(columns) if columns is not None else []
        return pd.DataFrame(np.nan, index=trade_index, columns=cols)

    ev = events.dropna(subset=["known_dt", "sue", "symbol"]).copy()
    ev["known_dt"] = pd.to_datetime(ev["known_dt"])
    ev["symbol"] = ev["symbol"].astype(str)
    if columns is not None:
        col_set = set(str(c) for c in columns)
        ev = ev[ev["symbol"].isin(col_set)]
    if ev.empty:
        cols = list(columns) if columns is not None else []
        return pd.DataFrame(np.nan, index=trade_index, columns=cols)

    idx = trade_index.sort_values()
    pos = idx.searchsorted(ev["known_dt"].to_numpy())
    valid = pos < len(idx)
    ev = ev.iloc[np.asarray(valid)].copy()
    pos = pos[valid]
    ev["_td"] = idx[pos]
    # signal_date >= known_dt by construction of searchsorted
    wide = ev.pivot_table(index="_td", columns="symbol", values="sue", aggfunc="last")
    cols = list(columns) if columns is not None else sorted(wide.columns.astype(str))
    return wide.reindex(index=trade_index, columns=cols)


def build_and_cache_events(
    start: dt.datetime,
    end: dt.datetime,
    *,
    history_start: Optional[dt.datetime] = None,
    sue_p0_root: Path = SUE_P0_ROOT,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Load sue_p0 bundle → events parquet + PIT meta."""
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    path = EVENTS_DIR / f"SUE_ConsensusEPS_events_{tag}.parquet"
    meta_path = META_DIR / f"pit_audit_{tag}.json"
    if path.exists() and meta_path.exists() and not refresh:
        ev = pd.read_parquet(path)
        audit = json.loads(meta_path.read_text(encoding="utf-8"))
        return ev, audit

    hist = history_start or (start - dt.timedelta(days=800))
    bundle = load_sue_raw_bundle(
        start,
        end,
        history_start=hist,
        cache_root=sue_p0_root,
        keep_cache=True,
    )
    events, audit = build_sue_consensus_eps_events(bundle)
    # filter known_dt into [start, end] for cache tag semantics
    if not events.empty:
        events = events[
            (events["known_dt"] >= pd.Timestamp(start))
            & (events["known_dt"] <= pd.Timestamp(end))
        ].reset_index(drop=True)
        audit["n_events_in_window"] = int(len(events))
    events.to_parquet(path, index=False)
    meta_path.write_text(json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8")
    return events, audit


def build_and_cache_impulse_panel(
    events: pd.DataFrame,
    trade_index: pd.DatetimeIndex,
    *,
    tag: str,
    symbols: Optional[Sequence[str]] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    path = PANELS_DIR / f"SUE_ConsensusEPS_impulse_{tag}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    wide = events_to_impulse_panel(events, trade_index, columns=symbols)
    wide.to_parquet(path)
    return wide


def coverage_report(wide: pd.DataFrame) -> dict:
    finite = np.isfinite(wide.to_numpy(dtype=float))
    n_days, n_syms = wide.shape
    daily_n = finite.sum(axis=1)
    return {
        "n_days": int(n_days),
        "n_symbols": int(n_syms),
        "coverage_cell": float(finite.mean()) if n_days and n_syms else 0.0,
        "mean_names_per_day": float(daily_n.mean()) if n_days else 0.0,
        "median_names_per_day": float(np.median(daily_n)) if n_days else 0.0,
        "pct_days_ge_1": float((daily_n >= 1).mean()) if n_days else 0.0,
        "pct_days_ge_50": float((daily_n >= 50).mean()) if n_days else 0.0,
        "n_event_cells": int(finite.sum()),
    }


def distribution_on_event_days(wide: pd.DataFrame) -> dict:
    """CS stats only on days with ≥2 finite SUE values."""
    rows = []
    for d, row in wide.iterrows():
        v = row.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        rows.append(
            {
                "date": d,
                "n": len(v),
                "cs_std": float(np.std(v, ddof=1)),
                "cs_mean": float(np.mean(v)),
                "frac_zero": float(np.mean(np.abs(v) < 1e-12)),
                "frac_abs_gt_10": float(np.mean(np.abs(v) > 10)),
            }
        )
    if not rows:
        return {"n_active_days": 0}
    df = pd.DataFrame(rows)
    return {
        "n_active_days": int(len(df)),
        "mean_cs_std": float(df["cs_std"].mean()),
        "median_cs_std": float(df["cs_std"].median()),
        "mean_n": float(df["n"].mean()),
        "mean_frac_zero": float(df["frac_zero"].mean()),
        "mean_frac_abs_gt_10": float(df["frac_abs_gt_10"].mean()),
    }
