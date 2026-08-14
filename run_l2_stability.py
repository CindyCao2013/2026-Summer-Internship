#!/usr/bin/env python
"""L2 v2 stability — time-slice stack enhancement for retained conditioning signals.

Answers: is L2 enhancer uplift stable across regimes (not a single-period artifact)?

Usage:
  OMP_NUM_THREADS=1 python run_l2_stability.py
"""

from __future__ import annotations

import datetime as dt
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import factor_config as cfg
import pandas as pd

from factor_attribution import OHLCV_FROZEN_REPS
from l2_conditioning_layer import L2_STATE_LAYER
from l2_stack_enhancement import stack_enhancement_row
from run_l2_validation import build_any_eod, load_context

OUT = cfg.RESEARCH_DIR

TIME_SLICES = [
    ("2020_2022", dt.datetime(2020, 1, 1), dt.datetime(2022, 12, 31)),
    ("2023_2024", dt.datetime(2023, 1, 1), dt.datetime(2024, 12, 31)),
    ("2025_ytd", dt.datetime(2025, 1, 1), dt.datetime(2025, 12, 31)),
]

RETAIN = [x["factor"] for x in L2_STATE_LAYER]


def slice_panel(panel: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    return panel.loc[(panel.index >= start) & (panel.index <= end)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading full context (cached L2)...", flush=True)
    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)

    baseline_full = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]
    rows = []

    for label, start, end in TIME_SLICES:
        ret_s = slice_panel(ctx["ret"], start, end)
        if len(ret_s) < 60:
            print(f"  SKIP {label}: too few days ({len(ret_s)})")
            continue
        baseline = [slice_panel(p, start, end) for p in baseline_full]
        for fname in RETAIN:
            if fname not in ctx["l2_panels"]:
                continue
            panel = slice_panel(ctx["l2_panels"][fname], start, end)
            row = stack_enhancement_row(baseline, panel, ret_s, fname)
            row["time_slice"] = label
            row["n_days"] = len(ret_s)
            rows.append(row)
        print(f"  done {label} n={len(ret_s)}", flush=True)

    df = pd.DataFrame(rows)
    out = OUT / "l2_v2_stability.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")

    if len(df):
        summary = df.groupby("factor_name").agg(
            n_slices=("time_slice", "count"),
            mean_ic_delta=("stack_ic_delta", "mean"),
            mean_sharpe_delta=("stack_sharpe_delta", "mean"),
            pass_rate=("stack_enhancement_pass", "mean"),
        )
        print("\n=== Stability summary ===")
        print(summary.to_string())
        summary.to_csv(OUT / "l2_v2_stability_summary.csv")


if __name__ == "__main__":
    main()
