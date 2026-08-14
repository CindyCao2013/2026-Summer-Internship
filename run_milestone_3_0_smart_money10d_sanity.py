#!/usr/bin/env python
"""III-A4.1 Phase 1.5 — SmartMoney10d sanity gate (before 252d / Phase2A).

Gates:
  1. Cross-sectional dispersion of Q (daily std)
  2. Rank uniqueness / tie ratio
  3. Raw RankIC(Q, r_{t+1}) — NO sign flip

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_sanity.py
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_sanity.py --max-symbols 800
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from core.l2_features.smart_money_panel_builder import (
    FACTOR_PANEL_DIR,
    FORMULA_VERSION,
    build_smart_money10d_panel,
    load_minute_feature,
)

REPO = Path(__file__).resolve().parent
SANITY_OUT = REPO / "research/reports/smart_money_v1/sanity"
SIGNAL_SHIFT = 1


def log(msg: str) -> None:
    print(msg, flush=True)


def month_bounds(year: int, month: int) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, month, 1)
    if month == 12:
        end = dt.datetime(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        end = dt.datetime(year, month + 1, 1) - dt.timedelta(days=1)
    return start, end


def daily_cs_dispersion(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional mean/std/n of raw Q (no zscore)."""
    rows = []
    for d, row in wide.iterrows():
        v = row.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            rows.append(
                {
                    "date": d,
                    "n": len(v),
                    "cs_mean": np.nan,
                    "cs_std": np.nan,
                    "cs_mad": np.nan,
                    "cs_iqr": np.nan,
                }
            )
            continue
        rows.append(
            {
                "date": d,
                "n": int(len(v)),
                "cs_mean": float(np.mean(v)),
                "cs_std": float(np.std(v, ddof=1)),
                "cs_mad": float(np.median(np.abs(v - np.median(v)))),
                "cs_iqr": float(np.quantile(v, 0.75) - np.quantile(v, 0.25)),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def daily_rank_uniqueness(wide: pd.DataFrame) -> pd.DataFrame:
    """Rank uniqueness and tie diagnostics (method=average ranks)."""
    rows = []
    for d, row in wide.iterrows():
        v = row.dropna()
        n = int(len(v))
        if n == 0:
            rows.append(
                {
                    "date": d,
                    "n": 0,
                    "n_unique_values": 0,
                    "n_unique_ranks": 0,
                    "tie_ratio": np.nan,
                    "value_collision_ratio": np.nan,
                }
            )
            continue
        ranks = v.rank(method="average")
        n_u_val = int(v.nunique())
        n_u_rank = int(ranks.nunique())
        # tie_ratio: 1 - unique_ranks/n  (0 = all distinct ranks after average ties)
        # value collision: duplicates on raw Q
        rows.append(
            {
                "date": d,
                "n": n,
                "n_unique_values": n_u_val,
                "n_unique_ranks": n_u_rank,
                "tie_ratio": float(1.0 - n_u_rank / n),
                "value_collision_ratio": float(1.0 - n_u_val / n),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def summarize_series(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--max-symbols", type=int, default=800)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    SANITY_OUT.mkdir(parents=True, exist_ok=True)
    start, end = month_bounds(args.year, args.month)

    log("=== III-A4.1 Phase 1.5 SmartMoney10d SANITY ===")
    log(f"formula={FORMULA_VERSION} — raw Q, no sign flip, no Registry")
    log(f"window={start.date()}..{end.date()} max_symbols={args.max_symbols}")

    feat = load_minute_feature(start - dt.timedelta(days=40), end, use_cache=True)
    all_syms = sorted(feat["symbol"].unique().tolist())
    syms = all_syms if args.max_symbols <= 0 else all_syms[: args.max_symbols]
    log(f"Building/loading panel n_symbols={len(syms)}")
    wide, _long = build_smart_money10d_panel(
        start,
        end,
        use_cache=True,
        refresh_cache=args.refresh_cache,
        preheat_calendar_days=40,
        symbols=syms,
    )
    # strip all-nan columns
    wide = wide.dropna(axis=1, how="all")
    log(f"panel shape={wide.shape}")

    # --- Gate 1: CS dispersion ---
    disp = daily_cs_dispersion(wide)
    disp.to_csv(SANITY_OUT / "daily_cs_dispersion.csv")
    disp_sum = {
        "cs_std": summarize_series(disp["cs_std"]),
        "cs_mad": summarize_series(disp["cs_mad"]),
        "cs_iqr": summarize_series(disp["cs_iqr"]),
        "cs_mean": summarize_series(disp["cs_mean"]),
        "n_names": summarize_series(disp["n"].astype(float)),
    }
    log(
        f"Gate1 CS std: mean={disp_sum['cs_std'].get('mean')} "
        f"median={disp_sum['cs_std'].get('median')} "
        f"min={disp_sum['cs_std'].get('min')} max={disp_sum['cs_std'].get('max')}"
    )

    # --- Gate 2: rank uniqueness ---
    uniq = daily_rank_uniqueness(wide)
    uniq.to_csv(SANITY_OUT / "daily_rank_uniqueness.csv")
    uniq_sum = {
        "n_unique_ranks": summarize_series(uniq["n_unique_ranks"].astype(float)),
        "n_unique_values": summarize_series(uniq["n_unique_values"].astype(float)),
        "tie_ratio": summarize_series(uniq["tie_ratio"]),
        "value_collision_ratio": summarize_series(uniq["value_collision_ratio"]),
        "n": summarize_series(uniq["n"].astype(float)),
    }
    mean_n = uniq_sum["n"].get("mean", np.nan)
    mean_ur = uniq_sum["n_unique_ranks"].get("mean", np.nan)
    log(
        f"Gate2 unique ranks: mean={mean_ur:.1f} / n≈{mean_n:.1f} "
        f"tie_ratio_mean={uniq_sum['tie_ratio'].get('mean')}"
    )

    # --- Gate 3: raw direction RankIC (no flip) ---
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end + dt.timedelta(days=5), method="c2c")
    ret = ret.reindex(index=wide.index.union(ret.index), columns=wide.columns)
    # extend index one day for shift alignment if needed
    ic = daily_rank_ic_series(wide, ret, signal_shift=SIGNAL_SHIFT)
    ic.to_csv(SANITY_OUT / "daily_raw_rank_ic.csv", header=["rank_ic"])
    ic_clean = ic.dropna()
    rank_ic_mean = float(ic_clean.mean()) if len(ic_clean) else float("nan")
    # short window → ICIR unreliable; still report
    icir = float(icir_from_daily(ic)) if len(ic_clean) >= 10 else float("nan")
    direction = (
        "negative_matches_paper"
        if rank_ic_mean < 0
        else ("positive_empirical_reverse" if rank_ic_mean > 0 else "flat_or_nan")
    )
    log(
        f"Gate3 raw RankIC(Q,r): mean={rank_ic_mean:.4f} n_days={len(ic_clean)} "
        f"ICIR~{icir} → {direction} (NO flip applied)"
    )

    # --- Pass / soft-fail ---
    # Dispersion: near-constant if mean cs_std << 1e-4; healthy if ~1e-3–1e-2
    mean_std = disp_sum["cs_std"].get("mean")
    mean_tie = uniq_sum["tie_ratio"].get("mean")
    mean_std = float(mean_std) if mean_std is not None and np.isfinite(mean_std) else 0.0
    mean_tie = float(mean_tie) if mean_tie is not None and np.isfinite(mean_tie) else 1.0
    unique_frac = (mean_ur / mean_n) if mean_n and mean_n > 0 else 0.0

    gates = {
        "dispersion_not_near_constant": mean_std >= 1e-4,
        "dispersion_in_useful_band": mean_std >= 5e-4,  # soft: 0.0005–0.01 preferred
        "rank_unique_frac_gt_0p5": unique_frac >= 0.5,
        "tie_ratio_lt_0p5": mean_tie <= 0.5,
        "raw_rank_ic_finite": bool(np.isfinite(rank_ic_mean)),
        "raw_rank_ic_not_zero_noise": abs(rank_ic_mean) >= 0.005,  # soft signal on short window
    }
    # Hard pass: not near-constant + ranks usable + IC finite
    hard_ok = (
        gates["dispersion_not_near_constant"]
        and gates["rank_unique_frac_gt_0p5"]
        and gates["tie_ratio_lt_0p5"]
        and gates["raw_rank_ic_finite"]
    )
    soft_ok = hard_ok and gates["dispersion_in_useful_band"]

    report = {
        "phase": "1.5_sanity",
        "formula_version": FORMULA_VERSION,
        "window": f"{start.date()}_{end.date()}",
        "n_days": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "signal": "raw_Q",
        "sign_flip": False,
        "gate1_cs_dispersion": disp_sum,
        "gate2_rank_uniqueness": uniq_sum,
        "gate3_raw_direction": {
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_std": float(ic_clean.std()) if len(ic_clean) else None,
            "n_ic_days": int(len(ic_clean)),
            "icir_short_window": icir,
            "paper_expected": "RankIC < 0",
            "empirical_direction": direction,
            "note": "Do not invert formula; record empirical_direction only.",
        },
        "gates": gates,
        "hard_pass": hard_ok,
        "soft_pass": soft_ok,
        "interpretation": (
            "Useful CS dispersion + rankable Q"
            if soft_ok
            else (
                "Borderline: not near-constant but weak band/IC — proceed Phase2A carefully"
                if hard_ok
                else "FAIL: near-constant or unrankable — do not run Phase2"
            )
        ),
        "next": (
            "Phase2A CSI1000 2023-2025 scout"
            if hard_ok
            else "Stop — diagnose Q construction before Phase2"
        ),
        "forbidden": ["formula_change", "sign_flip", "Registry", "252d_ALL"],
        "panel_cache": str(FACTOR_PANEL_DIR),
    }
    out = SANITY_OUT / "sanity_report.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"Wrote {out}")
    log(f"SANITY hard={'PASS' if hard_ok else 'FAIL'} soft={'PASS' if soft_ok else 'FAIL'}")
    log(f"→ {report['interpretation']}")
    if not hard_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
