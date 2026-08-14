#!/usr/bin/env python
"""III-A4.1 Phase 1 — SmartMoney10d cache smoke (no 252d pack / no Registry).

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_smoke.py
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_smoke.py --year 2024 --month 6
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

from core.l2_features.smart_money_panel_builder import (
    FORMULA_VERSION,
    MINUTE_FEATURE_DIR,
    MINUTE_RAW_DIR,
    build_smart_money10d_panel,
    coverage_report,
    distribution_report,
    load_minute_feature,
    load_minute_raw,
)

REPO = Path(__file__).resolve().parent
SMOKE_OUT = REPO / "research/reports/smart_money_v1/smoke"


def log(msg: str) -> None:
    print(msg, flush=True)


def month_bounds(year: int, month: int) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, month, 1)
    if month == 12:
        end = dt.datetime(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        end = dt.datetime(year, month + 1, 1) - dt.timedelta(days=1)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=400,
        help="L3 subsample for smoke speed (0 = all symbols in feature cache)",
    )
    args = parser.parse_args()

    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    start, end = month_bounds(args.year, args.month)
    # need prior month for 10d lookback on month start
    if args.month == 1:
        pre_start = dt.datetime(args.year - 1, 12, 1)
    else:
        pre_start = dt.datetime(args.year, args.month - 1, 1)

    log("=== III-A4.1 Phase 1 SmartMoney10d SMOKE ===")
    log(f"formula={FORMULA_VERSION} window=Option B (rolling 10d) beta=0.25")
    log(f"smoke month={start.date()}..{end.date()} preheat_from={pre_start.date()}")
    log("No APM / no Active_* / no Registry / no 252d pack")

    log("L1 minute_raw ...")
    raw = load_minute_raw(
        pre_start, end, use_cache=True, refresh_cache=args.refresh_cache
    )
    raw_cols = list(raw.columns)
    assert not any(c.lower().startswith("active_") for c in raw_cols), raw_cols
    log(f"  rows={len(raw):,} cols={raw_cols}")
    (SMOKE_OUT / "l1_raw_schema.json").write_text(
        json.dumps({"n_rows": len(raw), "columns": raw_cols}, indent=2) + "\n",
        encoding="utf-8",
    )

    log("L2 minute_feature (smart_score) ...")
    feat = load_minute_feature(
        pre_start, end, use_cache=True, refresh_cache=args.refresh_cache
    )
    feat_cols = list(feat.columns)
    assert "smart_score" in feat_cols and "ret_1m" in feat_cols
    assert not any(c.lower().startswith("active_") for c in feat_cols)
    log(f"  rows={len(feat):,} cols={feat_cols}")
    (SMOKE_OUT / "l2_feature_schema.json").write_text(
        json.dumps({"n_rows": len(feat), "columns": feat_cols}, indent=2) + "\n",
        encoding="utf-8",
    )

    log("L3 factor_panel SmartMoney10d ...")
    syms = None
    if args.max_symbols and args.max_symbols > 0:
        # stable subsample from feature universe
        all_syms = sorted(feat["symbol"].unique().tolist())
        syms = all_syms[: args.max_symbols]
        log(f"  L3 subsample n_symbols={len(syms)} (of {len(all_syms)})")
    wide, long = build_smart_money10d_panel(
        start,
        end,
        use_cache=True,
        refresh_cache=args.refresh_cache,
        preheat_calendar_days=40,
        symbols=syms,
    )
    cov = coverage_report(wide)
    dist = distribution_report(wide)
    log(f"  wide shape={wide.shape} coverage_cell={cov['coverage_cell']:.3f}")
    log(
        f"  Q describe: mean={dist.get('mean')} p50={dist.get('p50')} "
        f"frac_|Q-1|<0.01={dist.get('frac_abs_dev_lt_0p01')}"
    )

    # pass / soft-fail gates (smoke: L1/L2 full; L3 may be subsample)
    gates = {
        "l1_rows_gt_0": len(raw) > 0,
        "l2_rows_gt_0": len(feat) > 0,
        "no_active_cols": True,
        "panel_days_gt_0": cov["n_days"] > 0,
        "coverage_cell_gt_0p50": cov["coverage_cell"] > 0.50,
        "q_finite_gt_0": dist.get("n", 0) > 0,
        "q_near_1_mass": dist.get("frac_abs_dev_lt_0p05", 0) > 0.3,
    }
    ok = all(gates.values())
    report = {
        "formula_version": FORMULA_VERSION,
        "lookback": "option_B_rolling_10d",
        "beta": 0.25,
        "paper_window_check": "Option B confirmed (Kaiyuan 步骤1: past 10 trading days)",
        "smoke_window": f"{start.date()}_{end.date()}",
        "l3_max_symbols": args.max_symbols,
        "cache_dirs": {
            "minute_raw": str(MINUTE_RAW_DIR),
            "minute_feature": str(MINUTE_FEATURE_DIR),
        },
        "coverage": cov,
        "distribution": dist,
        "gates": gates,
        "smoke_pass": ok,
        "next": "252d pack only if smoke_pass" if ok else "fix gates before full run",
        "forbidden": ["APM", "Active_*", "Registry", "252d_before_smoke"],
    }
    out_path = SMOKE_OUT / "smoke_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"SMOKE {'PASS' if ok else 'FAIL'}: {gates}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
