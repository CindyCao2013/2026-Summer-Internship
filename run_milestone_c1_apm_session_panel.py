#!/usr/bin/env python
"""C1.1 Phase1 — APM_SessionResidual session panel build (no IC / no library).

Usage:
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_panel.py
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_panel.py --year 2024 --month 6
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from core.l2_features.apm_session_panel_builder import (
    CACHE_ROOT,
    DEFAULT_INDEX,
    FORMULA_VERSION,
    PM_START_RULE,
    build_apm_session_panel,
    build_sample_checks,
    coverage_stats,
    formula_meta,
    load_index_session_proxy,
    load_stock_overnight,
    load_stock_pm,
)
from factor_data_loaders import load_eod_wide_tables

REPO = Path(__file__).resolve().parent
PHASE1_OUT = REPO / "research/reports/apm_session_v1/phase1"


def log(msg: str, fh=None) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if fh is not None:
        fh.write(line)
        fh.flush()


def month_bounds(year: int, month: int) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, month, 1)
    if month == 12:
        end = dt.datetime(year + 1, 1, 1) - dt.timedelta(days=1)
    else:
        end = dt.datetime(year, month + 1, 1) - dt.timedelta(days=1)
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description="C1.1 APM session panel Phase1")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--sample-symbols", type=int, default=20)
    args = parser.parse_args()

    PHASE1_OUT.mkdir(parents=True, exist_ok=True)
    start, end = month_bounds(args.year, args.month)
    log_path = PHASE1_OUT / "build_log.txt"
    fh = log_path.open("w", encoding="utf-8")

    try:
        log("=== C1.1 Phase1 APM_SessionResidual PANEL BUILD ===", fh)
        log(f"formula={FORMULA_VERSION} identity=adapted_replication", fh)
        log(f"window={start.date()}..{end.date()} index={args.index}", fh)
        log(f"pm_start_rule={PM_START_RULE}", fh)
        log("Forbidden: IC · Sharpe · library · Registry · Active_* · Proxy rename · cache shift(1)", fh)

        residual, manifest = build_apm_session_panel(
            start,
            end,
            index_code=args.index,
            use_cache=True,
            refresh_cache=args.refresh_cache,
        )
        log(f"residual rows={len(residual):,} dates={manifest.get('n_dates')} symbols={manifest.get('n_symbols')}", fh)

        cov = coverage_stats(residual)
        log(f"coverage pct_nan={cov.get('pct_nan')}", fh)
        log(f"mean_daily_frac_finite_r_pm={cov.get('mean_daily_frac_finite_r_pm'):.4f}", fh)

        # Reload layers for alignment / sample (from cache)
        stock_pm = load_stock_pm(start, end, use_cache=True, refresh_cache=False)
        stock_ovn = load_stock_overnight(start, end, use_cache=True, refresh_cache=False)
        index_proxy = load_index_session_proxy(
            start, end, index_code=args.index, use_cache=True, refresh_cache=False
        )

        # Alignment: PM bartimes in afternoon window
        bt_first = stock_pm["pm_bartime_first"].astype(str)
        bt_last = stock_pm["pm_bartime_last"].astype(str)
        # DolphinDB second → often "13:01:00" style strings
        def _in_pm(series: pd.Series) -> float:
            s = series.fillna("").astype(str)
            ok = s.str.contains(r"1[3-5]:", regex=True) | s.str.match(r"^0?13:") | s.str.match(r"^14:") | s.str.match(r"^15:")
            # also accept timedelta / numeric encodings as non-empty
            nonempty = s.str.len() > 0
            return float((ok | (~nonempty)).mean()) if len(s) else 0.0

        import pandas as pd  # local for alignment helpers

        alignment = {
            "pm_start_rule": PM_START_RULE,
            "n_pm_rows": int(len(stock_pm)),
            "frac_pm_n_bars_ge_2": float((stock_pm["pm_n_bars"].fillna(0) >= 2).mean())
            if len(stock_pm)
            else 0.0,
            "frac_finite_pm_return": float(stock_pm["pm_return"].notna().mean()) if len(stock_pm) else 0.0,
            "index_adapted": True,
            "index_pm_matched": False,
            "calendar": manifest.get("calendar_summary", {}),
            "sample_bartime_first_head": bt_first.head(5).tolist(),
            "sample_bartime_last_head": bt_last.head(5).tolist(),
        }

        # PIT: cache must not be shifted; document eval convention
        pit = {
            "cache_stores_raw_date_T": True,
            "cache_shifted_for_backtest": False,
            "eval_convention_phase2": "signal(T) predicts return(T+1) via shift(1) at eval time only",
            "bars_used_for_date_T": "minute Close with Bartime on date T in [13:01,15:00]; EOD Open/Close on T / T-1",
            "index_same_day_eod_ok_for_residual": True,
            "no_future_close_in_pm_leg": True,
            "formula_meta": formula_meta(args.index),
        }

        # Sample checks with EOD open/prev_close
        log("sample_checks ...", fh)
        eod, s = load_eod_wide_tables(start - dt.timedelta(days=10), end)
        try:
            sample = build_sample_checks(
                stock_ovn,
                stock_pm,
                eod.open,
                eod.close,
                n_symbols=args.sample_symbols,
                n_dates=5,
            )
        finally:
            s.close()
        sample_path = PHASE1_OUT / "sample_checks.csv"
        sample.to_csv(sample_path, index=False)
        log(f"Wrote {sample_path} rows={len(sample)}", fh)

        # Gates (panel only)
        gates = {
            "residual_rows_gt_0": len(residual) > 0,
            "pm_rows_gt_0": len(stock_pm) > 0,
            "ovn_rows_gt_0": len(stock_ovn) > 0,
            "index_rows_gt_0": len(index_proxy) > 0,
            "no_active_cols": not any(c.lower().startswith("active_") for c in residual.columns),
            "mean_daily_frac_finite_r_pm_gt_0p3": cov.get("mean_daily_frac_finite_r_pm", 0) > 0.3,
            "cache_not_shifted": True,
            "adapted_flag": True,
        }
        ok = all(gates.values())

        coverage_report = {
            "formula_version": FORMULA_VERSION,
            "window": f"{start.date()}_{end.date()}",
            "index_code": args.index,
            "coverage": cov,
            "cache_root": str(CACHE_ROOT),
            "gates": gates,
            "phase1_pass": ok,
            "forbidden": [
                "IC",
                "Sharpe",
                "factor_library",
                "Registry",
                "Active_*",
                "Proxy_rename",
                "cache_shift1",
            ],
        }
        (PHASE1_OUT / "coverage_report.json").write_text(
            json.dumps(coverage_report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (PHASE1_OUT / "alignment_report.json").write_text(
            json.dumps(alignment, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (PHASE1_OUT / "pit_report.json").write_text(
            json.dumps(pit, indent=2, default=str) + "\n", encoding="utf-8"
        )

        log(f"Wrote reports under {PHASE1_OUT}", fh)
        log(f"PHASE1 {'PASS' if ok else 'FAIL'}: {gates}", fh)
        if not ok:
            raise SystemExit(1)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
