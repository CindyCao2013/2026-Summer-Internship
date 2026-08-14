#!/usr/bin/env python
"""C1.2 Phase2 — APM_SessionResidual sanity (paper object constructability).

NOT alpha evaluation: no RankIC / ICIR / Sharpe / decile / Pack / library.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_sanity.py
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_sanity.py --year 2024 --month 6
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

import numpy as np
import pandas as pd

import Factor_Dev_Lib
from core.l2_features.apm_session_panel_builder import (
    CACHE_ROOT,
    FORMULA_VERSION,
    build_apm_session_panel,
    formula_meta,
)
from core.l2_features.apm_session_signal import (
    DEFAULT_MIN_PERIODS,
    DEFAULT_WINDOW,
    build_apm_stat_panel,
    build_ret20_long,
    cs_residualize_vs_ret20,
    daily_coverage_table,
    distribution_table,
)

REPO = Path(__file__).resolve().parent
SANITY_OUT = REPO / "research/reports/apm_session_v1/sanity"
SIGNAL_CACHE = CACHE_ROOT / "signal"


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


def prior_month_start(year: int, month: int) -> dt.datetime:
    if month == 1:
        return dt.datetime(year - 1, 12, 1)
    return dt.datetime(year, month - 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="C1.2 APM session Phase2 sanity")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--sample-symbols", type=int, default=20)
    args = parser.parse_args()

    SANITY_OUT.mkdir(parents=True, exist_ok=True)
    SIGNAL_CACHE.mkdir(parents=True, exist_ok=True)
    start, end = month_bounds(args.year, args.month)
    pre_start = prior_month_start(args.year, args.month)

    fh = (SANITY_OUT / "build_log.txt").open("w", encoding="utf-8")
    try:
        log("=== C1.2 Phase2 APM_SessionResidual SANITY ===", fh)
        log("Goal: paper object constructability — NOT alpha evaluation", fh)
        log(f"formula={FORMULA_VERSION} window={args.window} min_periods={args.min_periods}", fh)
        log(f"eval={start.date()}..{end.date()} preheat_from={pre_start.date()}", fh)
        log(
            "Forbidden: RankIC · ICIR · Sharpe · decile · Pack · library · Registry · Active_*",
            fh,
        )

        # Ensure residual panel covers preheat + eval
        log("ensure residual panel (preheat+eval) ...", fh)
        residual_full, manifest = build_apm_session_panel(
            pre_start,
            end,
            use_cache=True,
            refresh_cache=args.refresh_cache,
        )
        log(
            f"  residual_full rows={len(residual_full):,} "
            f"dates={manifest.get('n_dates')} symbols={manifest.get('n_symbols')}",
            fh,
        )

        # Step 1 — residual distributions on eval window
        residual_eval = residual_full[
            (residual_full["date"] >= pd.Timestamp(start))
            & (residual_full["date"] <= pd.Timestamp(end))
        ].copy()
        resid_dist = distribution_table(
            residual_eval, ["alpha_on", "alpha_pm", "delta_alpha"]
        )
        resid_dist.to_csv(SANITY_OUT / "residual_distribution.csv", index=False)
        log(f"residual_distribution:\n{resid_dist.to_string(index=False)}", fh)

        # Step 2 — rolling APM_stat
        log("build APM_stat ...", fh)
        apm_full = build_apm_stat_panel(
            residual_full, window=args.window, min_periods=args.min_periods
        )
        apm_eval = apm_full[
            (apm_full["date"] >= pd.Timestamp(start))
            & (apm_full["date"] <= pd.Timestamp(end))
        ].copy()
        tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_w{args.window}"
        apm_path = SIGNAL_CACHE / f"apm_stat_{tag}.parquet"
        apm_eval.to_parquet(apm_path, index=False)
        log(f"  wrote {apm_path} rows={len(apm_eval):,}", fh)

        sig_dist = distribution_table(apm_eval, ["delta_alpha", "apm_stat"])
        sig_dist.to_csv(SANITY_OUT / "signal_distribution.csv", index=False)

        # Step 3 — Ret20 CS residual constructability
        log("Ret20 + CS residualize ...", fh)
        ret = Factor_Dev_Lib.get_Ret_Matrix(
            pre_start - dt.timedelta(days=40), end + dt.timedelta(days=5), method="c2c"
        )
        ret20_long = build_ret20_long(ret, window=20)
        apm_cs_full, align = cs_residualize_vs_ret20(apm_full, ret20_long, signal_col="apm_stat")
        apm_cs_eval = apm_cs_full[
            (apm_cs_full["date"] >= pd.Timestamp(start))
            & (apm_cs_full["date"] <= pd.Timestamp(end))
        ].copy()
        align_eval = align[
            (align["date"] >= pd.Timestamp(start)) & (align["date"] <= pd.Timestamp(end))
        ].copy()
        cs_path = SIGNAL_CACHE / f"apm_cs_{tag}.parquet"
        apm_cs_eval.to_parquet(cs_path, index=False)
        align_eval.to_csv(SANITY_OUT / "ret20_alignment.csv", index=False)
        log(f"  wrote {cs_path} rows={len(apm_cs_eval):,}", fh)
        log(
            f"  mean ret20 coverage={float(align_eval['coverage'].mean()):.4f} "
            f"mean n_valid={float(align_eval['n_valid'].mean()):.1f}",
            fh,
        )

        # Daily coverage
        cov = daily_coverage_table(apm_cs_eval, ["apm_stat", "apm_cs", "delta_alpha"])
        cov.to_csv(SANITY_OUT / "daily_coverage.csv", index=False)

        # Sample signal
        counts = apm_eval.groupby("symbol")["apm_stat"].apply(
            lambda s: int(np.isfinite(s.to_numpy(dtype=float)).sum())
        )
        syms = counts.sort_values(ascending=False).head(args.sample_symbols).index.tolist()
        dates = sorted(apm_eval["date"].unique())[:5]
        sample = apm_cs_eval[
            apm_cs_eval["symbol"].isin(syms) & apm_cs_eval["date"].isin(dates)
        ][
            ["date", "symbol", "apm_stat", "ret20", "apm_cs"]
        ].merge(
            residual_eval[["date", "symbol", "alpha_on", "alpha_pm", "delta_alpha"]],
            on=["date", "symbol"],
            how="left",
        )
        sample.to_csv(SANITY_OUT / "sample_signal.csv", index=False)

        # Step 4 — PIT alignment (document only; no IC)
        cal_dates = sorted(pd.to_datetime(apm_eval["date"]).unique())
        pit_sample = None
        if len(cal_dates) >= 2:
            t0 = pd.Timestamp(cal_dates[0])
            t1 = pd.Timestamp(cal_dates[1])
            pit_sample = {
                "signal_date": t0.strftime("%Y-%m-%d"),
                "return_date_for_eval": t1.strftime("%Y-%m-%d"),
                "note": "evaluation must join signal.shift(1) to ret at return_date, "
                "or signal(T) to return(T+1); cache stores unshifted signal(T)",
            }
        pit = {
            "cache_shift": False,
            "evaluation_shift": True,
            "future_return_used": "T+1",
            "eval_convention": "signal(T) predicts return(T+1)",
            "method": "signal.shift(1) joined to ret_t at evaluation only",
            "sample_check": pit_sample,
            "formula_meta": formula_meta(),
        }
        (SANITY_OUT / "pit_alignment_report.json").write_text(
            json.dumps(pit, indent=2, default=str) + "\n", encoding="utf-8"
        )

        # Gates (constructability only)
        mean_frac_stat = float(cov["frac_finite_apm_stat"].mean()) if len(cov) else 0.0
        mean_frac_cs = float(cov["frac_finite_apm_cs"].mean()) if len(cov) else 0.0
        delta_cs_std = float(
            resid_dist.loc[resid_dist["column"] == "delta_alpha", "mean_daily_cs_std"].iloc[0]
        ) if (resid_dist["column"] == "delta_alpha").any() else 0.0

        gates = {
            "residual_rows_gt_0": len(residual_eval) > 0,
            "apm_stat_finite": mean_frac_stat > 0.3,
            "cross_section_delta_std": delta_cs_std > 1e-6,
            "coverage_apm_stat_gt_0p3": mean_frac_stat > 0.3,
            "coverage_apm_cs_gt_0p3": mean_frac_cs > 0.3,
            "no_active_cols": not any(
                str(c).lower().startswith("active_") for c in apm_cs_eval.columns
            ),
            "cache_shift_false": True,
        }
        ok = all(gates.values())

        summary = {
            "milestone": "C1.2",
            "factor_id": "APM_SessionResidual",
            "identity_class": "adapted_replication",
            "formula_version": FORMULA_VERSION,
            "goal": "paper_object_constructability_not_alpha",
            "window": f"{start.date()}_{end.date()}",
            "preheat_from": str(pre_start.date()),
            "rolling_window": args.window,
            "min_periods": args.min_periods,
            "residual_distribution": resid_dist.to_dict(orient="records"),
            "signal_distribution": sig_dist.to_dict(orient="records"),
            "mean_frac_finite_apm_stat": mean_frac_stat,
            "mean_frac_finite_apm_cs": mean_frac_cs,
            "mean_daily_cs_std_delta_alpha": delta_cs_std,
            "gates": gates,
            "phase2_pass": ok,
            "forbidden": [
                "RankIC",
                "ICIR",
                "Sharpe",
                "decile",
                "turnover",
                "cost",
                "factor_library",
                "Pack_v1",
                "Registry",
            ],
            "next": "C1.3 Scout" if ok else "fix constructability before scout",
        }
        (SANITY_OUT / "sanity_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (CACHE_ROOT / "meta" / "phase2_manifest.json").write_text(
            json.dumps(
                {
                    "apm_stat_path": str(apm_path),
                    "apm_cs_path": str(cs_path),
                    "cache_shifted_for_backtest": False,
                    "formula_version": FORMULA_VERSION,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        log(f"Wrote reports under {SANITY_OUT}", fh)
        log(f"PHASE2 {'PASS' if ok else 'FAIL'}: {gates}", fh)
        if not ok:
            raise SystemExit(1)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
