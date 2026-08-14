#!/usr/bin/env python
"""APM_ActiveV2 smoke — one month Active Pressure Metric panel.

Usage:
  OMP_NUM_THREADS=1 python run_apm_active_v2_smoke.py
  OMP_NUM_THREADS=1 python run_apm_active_v2_smoke.py --year 2024 --month 6
  OMP_NUM_THREADS=1 python run_apm_active_v2_smoke.py --max-symbols 200
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

import Factor_Dev_Lib as FDL
from factor_cutting.apm_active_v2 import FORMULA_VERSION
from core.l2_features.apm_active_v2_builder import (
    build_apm_active_v2_panel,
    build_apm_raw_variants,
    coverage_report,
    distribution_report,
)
from core.l2_features.smart_money_active_v2_builder import load_minute_active_raw
from factor_formulas_apm_active_v2 import process_factor_cross_section

REPO = Path(__file__).resolve().parent
SMOKE_OUT = REPO / "research/reports/apm_active_v2/smoke"


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
    parser.add_argument("--max-symbols", type=int, default=400)
    parser.add_argument("--with-cs-post", action="store_true")
    args = parser.parse_args()

    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    start, end = month_bounds(args.year, args.month)
    if args.month == 1:
        pre_start = dt.datetime(args.year - 1, 12, 1)
    else:
        pre_start = dt.datetime(args.year, args.month - 1, 1)

    log("=== APM_ActiveV2 SMOKE (Active Pressure Metric) ===")
    log(f"formula={FORMULA_VERSION}")
    log("identity=amount-weighted (buy-sell)/(buy+sell) EWM5 (NOT price APM / NOT session-cut)")
    log(f"smoke month={start.date()}..{end.date()} preheat_from={pre_start.date()}")

    log("L1 minute_active_raw (reuse SmartMoney cache) ...")
    raw = load_minute_active_raw(
        pre_start, end, use_cache=True, refresh_cache=args.refresh_cache
    )
    raw_cols = list(raw.columns)
    assert "active_buy_amt" in raw_cols and "active_sell_amt" in raw_cols, raw_cols
    assert "amount" in raw_cols, raw_cols
    log(f"  rows={len(raw):,} cols={raw_cols}")
    (SMOKE_OUT / "l1_schema.json").write_text(
        json.dumps({"n_rows": len(raw), "columns": raw_cols}, indent=2) + "\n",
        encoding="utf-8",
    )

    syms = None
    if args.max_symbols and args.max_symbols > 0:
        all_syms = sorted(raw["symbol"].unique().tolist())
        syms = all_syms[: args.max_symbols]
        log(f"  subsample n_symbols={len(syms)} (of {len(all_syms)})")

    log("limit mask ...")
    not_limit = FDL.get_EOD_Not_Limit(pre_start, end)

    log("L3 factor panel + variants ...")
    wide, long = build_apm_active_v2_panel(
        start,
        end,
        use_cache=True,
        refresh_cache=args.refresh_cache,
        preheat_calendar_days=30,
        not_limit=not_limit,
        symbols=syms,
    )
    variants = build_apm_raw_variants(
        start,
        end,
        use_cache=True,
        refresh_cache=args.refresh_cache,
        preheat_calendar_days=30,
        not_limit=not_limit,
        symbols=syms,
    )
    cov = coverage_report(wide)
    dist = distribution_report(wide)
    log(f"  wide shape={wide.shape} coverage_cell={cov['coverage_cell']:.3f}")
    log(
        f"  apm_smooth describe: mean={dist.get('mean')} p50={dist.get('p50')} "
        f"std={dist.get('std')}"
    )
    log(f"  variants={list(variants.keys())}")

    report = {
        "formula_version": FORMULA_VERSION,
        "start": str(start.date()),
        "end": str(end.date()),
        "coverage": cov,
        "distribution": dist,
        "n_long_rows": int(len(long)),
        "variants": {k: list(v.shape) for k, v in variants.items()},
        "identity": "APM_ActiveV2 Active Pressure Metric",
        "brick": "active_pressure",
        "not_price_apm": True,
        "not_session_cut": True,
    }

    if args.with_cs_post and not wide.empty:
        log("CS post-process (MAD+zscore, no industry in smoke) ...")
        processed = process_factor_cross_section(
            wide, industry=None, neutralize=False
        )
        report["cs_post_coverage"] = coverage_report(processed)
        log(f"  processed coverage={report['cs_post_coverage']['coverage_cell']:.3f}")

    (SMOKE_OUT / "smoke_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    log(f"wrote {SMOKE_OUT / 'smoke_report.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
