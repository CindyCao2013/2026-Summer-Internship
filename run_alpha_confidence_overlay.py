#!/usr/bin/env python
"""Alpha stack + L2 confidence overlay — Stage 2 integration test.

Tests S' = z(D1-D5 stack) * (1 + λ * z(L2_state)) on full sample + time slices.

Usage:
  OMP_NUM_THREADS=1 python run_alpha_confidence_overlay.py
"""

from __future__ import annotations

import datetime as dt
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import factor_config as cfg
import pandas as pd

from alpha_confidence_overlay import run_overlay_grid
from factor_attribution import OHLCV_FROZEN_REPS
from l2_conditioning_layer import DEFAULT_OVERLAY_LAMBDA, L2_STATE_LAYER
from run_l2_validation import load_context

OUT = cfg.RESEARCH_DIR

TIME_SLICES = [
    ("full", None, None),
    ("2020_2022", dt.datetime(2020, 1, 1), dt.datetime(2022, 12, 31)),
    ("2023_2024", dt.datetime(2023, 1, 1), dt.datetime(2024, 12, 31)),
    ("2025_ytd", dt.datetime(2025, 1, 1), dt.datetime(2025, 12, 31)),
]


def slice_ctx(ctx, start, end):
    if start is None:
        return ctx["ret"], ctx["frozen_panels"], ctx["l2_panels"]
    ret = ctx["ret"].loc[(ctx["ret"].index >= start) & (ctx["ret"].index <= end)]
    frozen = {k: v.loc[start:end] for k, v in ctx["frozen_panels"].items()}
    l2 = {k: v.loc[start:end] for k, v in ctx["l2_panels"].items()}
    return ret, frozen, l2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading context...", flush=True)
    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    retain = [x["factor"] for x in L2_STATE_LAYER if x["factor"] in ctx["l2_panels"]]

    rows = []
    for label, start, end in TIME_SLICES:
        ret, frozen, l2 = slice_ctx(ctx, start, end)
        if len(ret) < 60:
            continue
        baseline = [frozen[r] for r in OHLCV_FROZEN_REPS if r in frozen]
        sub_l2 = {k: l2[k] for k in retain}
        grid = run_overlay_grid(baseline, sub_l2, ret, lambdas=[DEFAULT_OVERLAY_LAMBDA])
        grid["time_slice"] = label
        grid["n_days"] = len(ret)
        rows.append(grid)
        print(f"  {label} n={len(ret)}", flush=True)

    df = pd.concat(rows, ignore_index=True)
    out = OUT / "alpha_confidence_overlay.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    print(df[["time_slice", "overlay_label", "ic_delta", "sharpe_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
