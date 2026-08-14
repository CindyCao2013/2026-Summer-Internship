#!/usr/bin/env python
"""IdealReversal_ActiveV2 smoke — one month inst EWM + reversal panel.

Usage:
  OMP_NUM_THREADS=1 python run_ideal_reversal_active_v2_smoke.py
  OMP_NUM_THREADS=1 python run_ideal_reversal_active_v2_smoke.py --year 2024 --month 6
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
from core.l2_features.ideal_reversal_active_v2_builder import (
    FORMULA_VERSION,
    build_ideal_reversal_active_v2_panel,
    coverage_report,
    distribution_report,
)
from core.l2_features.smart_money_active_v2_builder import load_minute_active_raw
from factor_data_loaders import load_eod_wide_tables
from factor_formulas_ideal_reversal_active_v2 import process_factor_cross_section

REPO = Path(__file__).resolve().parent
SMOKE_OUT = REPO / "research/reports/ideal_reversal_active_v2/smoke"


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
    parser.add_argument("--max-symbols", type=int, default=200)
    parser.add_argument("--with-cs-post", action="store_true")
    args = parser.parse_args()

    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    start, end = month_bounds(args.year, args.month)
    pre_start = start - dt.timedelta(days=60)

    log("=== IdealReversal_ActiveV2 SMOKE ===")
    log(f"formula={FORMULA_VERSION}")
    log("identity=EWM5 active_size_concentration × (-ret_5d) on high-ASC names")
    log("NOTE: ASC is observable concentration — NOT institutional participation")
    log(f"smoke month={start.date()}..{end.date()} preheat_from={pre_start.date()}")

    log("L1 minute_active_raw ...")
    raw = load_minute_active_raw(
        pre_start, end, use_cache=True, refresh_cache=args.refresh_cache
    )
    assert "active_buy_amt" in raw.columns and "active_buy_count" in raw.columns
    log(f"  rows={len(raw):,}")

    syms = None
    if args.max_symbols and args.max_symbols > 0:
        all_syms = sorted(raw["symbol"].unique().tolist())
        syms = all_syms[: args.max_symbols]
        log(f"  subsample n_symbols={len(syms)} (of {len(all_syms)})")

    log("EOD close + limit mask ...")
    eod, session = load_eod_wide_tables(pre_start - dt.timedelta(days=20), end)
    close = eod.close
    not_limit = FDL.get_EOD_Not_Limit(pre_start, end)
    session.close()

    log("L3 factor panel ...")
    wide, long = build_ideal_reversal_active_v2_panel(
        start,
        end,
        close,
        use_cache=True,
        refresh_cache=args.refresh_cache,
        preheat_calendar_days=60,
        not_limit=not_limit,
        symbols=syms,
    )
    cov = coverage_report(wide)
    dist = distribution_report(wide)
    log(f"  wide shape={wide.shape} coverage_cell={cov['coverage_cell']:.3f}")
    log(
        f"  factor describe: mean={dist.get('mean')} p50={dist.get('p50')} "
        f"std={dist.get('std')}"
    )

    report = {
        "formula_version": FORMULA_VERSION,
        "start": str(start.date()),
        "end": str(end.date()),
        "coverage": cov,
        "distribution": dist,
        "n_long_rows": int(len(long)),
        "identity": "IdealReversal_ActiveV2",
        "universes_allowed": ["ALL", "CSI1000", "CSI500", "CSI300"],
        "not_kaiyuan_wcut": True,
    }

    if args.with_cs_post and not wide.empty:
        log("CS post-process (MAD+zscore, neutralize off in smoke) ...")
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
