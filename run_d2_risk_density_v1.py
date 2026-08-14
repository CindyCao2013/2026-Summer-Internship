#!/usr/bin/env python
"""D2 Volatility / Risk Alpha Density Mining v1.

Internal family density + cross-dimension orthogonal IC + stack λ-grid on D1+D4+D5 base.
Discovery: first 504 trading days; Confirmation: remaining OOS days.

Usage:
  OMP_NUM_THREADS=1 python run_d2_risk_density_v1.py
  OMP_NUM_THREADS=1 python run_d2_risk_density_v1.py --no-reports
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import compare_stacks
from alpha_dimension_density import (
    DISCOVERY_DAYS,
    DEFAULT_STACK_LAMBDAS,
    analyze_candidate_on_period,
    build_quad_base_without,
    build_tri_base_stack,
    cross_dimension_independence_table,
    evaluate_dim_lambda_grid,
    publish_dimension_density,
    split_discovery_confirmation,
    summarize_dimension_verdict,
)
from alpha_frozen_stack_v1 import FROZEN_OHLCV_REPS
from alpha_research_report import (
    build_factor_report,
    publish_factor_report,
    report_summary_row,
)
from factor_attribution import combine_equal_weight
from factor_formulas_risk_d2 import D2_REPRESENTATIVE, D2_RISK_DENSITY_CANDIDATES
from l2_stack_enhancement import stack_enhancement_row
from run_dimension_density_context import build_candidate, load_dimension_density_context, log

OUT = Path("research/reports/d2_risk_density_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="D2 Volatility Alpha Density v1")
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--no-reports", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== D2 Volatility / Risk Alpha Density v1 ===")
    ctx = load_dimension_density_context()
    ret_full = ctx["ret"]
    ret_disc, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    log(f"Full sample: {ret_full.index[0].date()} -> {ret_full.index[-1].date()} ({len(ret_full)}d)")
    log(f"Discovery: {len(ret_disc)}d | Confirmation: {len(ret_conf)}d")

    d2_rep = ctx["frozen_panels"][D2_REPRESENTATIVE]
    tri_base = build_tri_base_stack(ctx["frozen_panels"])

    density_rows = []
    report_summaries = []

    for name, source, hypothesis, family in D2_RISK_DENSITY_CANDIDATES:
        log(f"\n--- {name} ({source}, {family}) ---")
        try:
            wide = build_candidate(name, source, ctx)
            panel = wide.reindex(index=ret_full.index, columns=ret_full.columns)

            mono = None
            if not args.no_reports:
                report = build_factor_report(
                    name,
                    panel,
                    ctx["close"],
                    start_day=ctx["start"],
                    end_day=ctx["end"],
                    session=ctx["session"],
                    df_not_limit=ctx["df_not_limit"],
                    df_not_st=ctx["df_not_st"],
                    df_trade_status=ctx["df_trade_status"],
                    universes=ctx["universes"],
                    get_ret_matrix=ctx["get_ret_matrix"],
                )
                mono = report.aggregate.get("monotonicity_score")
                publish_factor_report(report, OUT)
                report_summaries.append(report_summary_row(report))

            for period_label, ret_w in [("discovery", ret_disc), ("confirmation", ret_conf)]:
                if len(ret_w) < 60:
                    continue
                enh = stack_enhancement_row(ctx["frozen_list"], panel, ret_w, name)
                row = analyze_candidate_on_period(
                    name,
                    panel,
                    ret_w,
                    d2_rep,
                    ctx["exposure_panels"],
                    ctx["frozen_panels"],
                    ctx["frozen_list"],
                    period=period_label,
                    hypothesis=hypothesis,
                    dim_label="d2",
                    anchor_key="ic_after_d2",
                    stack_enhancement_pass=bool(enh.get("stack_enhancement_pass")),
                    monotonicity=mono if period_label == "discovery" else None,
                )
                row["source"] = source
                row["family"] = family
                density_rows.append(row)
                if period_label == "discovery":
                    log(
                        f"  [{period_label}] IC={row['ic_raw']:.4f} | after D2={row['ic_after_d2']:.4f} | "
                        f"corr={row['corr_rep']:.3f} | {row['density_classification']}"
                    )
        except Exception as exc:
            log(f"  FAILED {name}: {exc}")
        gc.collect()

    cross_rows = []
    for period_label, ret_w in [("discovery", ret_disc), ("confirmation", ret_conf)]:
        if len(ret_w) < 60:
            continue
        tbl = cross_dimension_independence_table(
            d2_rep,
            ret_w,
            ctx["cross_dim_anchors"],
            period=period_label,
            factor_name=D2_REPRESENTATIVE,
        )
        cross_rows.append(tbl)
        log(f"\n=== Cross-dim independence ({period_label}) ===")
        log(
            tbl[
                ["orthogonalize_vs", "rank_ic_corr", "residual_ic_mean", "residual_ic_t", "independence_verdict"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    stack_rows = []
    for period_label, ret_w in [("discovery", ret_disc), ("confirmation", ret_conf)]:
        if len(ret_w) < 60:
            continue
        grid = evaluate_dim_lambda_grid(
            tri_base.reindex(index=ret_w.index, columns=ret_w.columns),
            d2_rep.reindex(index=ret_w.index, columns=ret_w.columns),
            ret_w,
            lambdas=DEFAULT_STACK_LAMBDAS,
            period=period_label,
            dim_label="D2",
        )
        stack_rows.append(grid)

    stack_df = pd.concat(stack_rows, ignore_index=True) if stack_rows else pd.DataFrame()
    density_df = pd.DataFrame(density_rows)
    cross_df = pd.concat(cross_rows, ignore_index=True) if cross_rows else pd.DataFrame()

    quad_no_d2 = build_quad_base_without(ctx["frozen_panels"], "D2")
    synergy: dict = {}
    if len(ret_disc) >= 60:
        quad_s = quad_no_d2.reindex(index=ret_disc.index, columns=ret_disc.columns)
        full_s = combine_equal_weight(
            [ctx["frozen_panels"][s["factor"]] for s in FROZEN_OHLCV_REPS]
        ).reindex(index=ret_disc.index, columns=ret_disc.columns)
        _, _, delta = compare_stacks(quad_s, full_s, ret_disc)
        synergy["discovery_quad_vs_full5"] = delta

    stack_disc = stack_df[stack_df["period"] == "discovery"] if not stack_df.empty else pd.DataFrame()
    stack_conf = stack_df[stack_df["period"] == "confirmation"] if not stack_df.empty else pd.DataFrame()

    verdict = summarize_dimension_verdict(
        "D2",
        D2_REPRESENTATIVE,
        density_df,
        cross_df,
        stack_disc,
        stack_conf,
        extra={"synergy_quad_vs_full5_discovery": synergy.get("discovery_quad_vs_full5", {})},
    )

    if not stack_df.empty:
        log("\n=== Stack λ-grid (discovery) ===")
        log(
            stack_disc[["lambda", "hl_sharpe", "rank_ic", "hl_sharpe_delta"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )
        log(f"\nProposed role: {verdict['proposed_role']}")

    publish_dimension_density(
        OUT,
        prefix="d2_risk",
        density_df=density_df,
        cross_df=cross_df,
        stack_df=stack_df,
        verdict=verdict,
    )
    if report_summaries:
        pd.DataFrame(report_summaries).to_csv(OUT / "candidate_reports_summary.csv", index=False)
    log(f"\nPublished -> {OUT}")


if __name__ == "__main__":
    main()
