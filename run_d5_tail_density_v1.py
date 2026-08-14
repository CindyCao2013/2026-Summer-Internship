#!/usr/bin/env python
"""D5 Tail Fragility Alpha Density Mining v1.

Tests whether tail-family candidates add independent alpha beyond D5 rep,
or are redundant volatility/lottery/behavioral projections.

Usage:
  OMP_NUM_THREADS=1 python run_d5_tail_density_v1.py
  OMP_NUM_THREADS=1 python run_d5_tail_density_v1.py --sample-days 504 --no-reports
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_density_mining import analyze_density_candidate, save_density_summary
from alpha_research_report import (
    build_factor_report,
    publish_factor_report,
    report_summary_row,
)
from factor_attribution import OHLCV_FROZEN_REPS
from factor_formulas import build_factor
from factor_formulas_cn_broker import build_cn_broker_factor
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_tail_d5 import D5_REPRESENTATIVE, D5_TAIL_DENSITY_CANDIDATES
from l2_stack_enhancement import stack_enhancement_row
from run_l2_validation import build_any_eod, load_context

OUT = Path("research/reports/d5_tail_density_v1")
SAMPLE_DAYS = 504


def log(msg: str) -> None:
    print(msg, flush=True)


def build_candidate(name: str, source: str, ctx: dict) -> pd.DataFrame:
    if name in ctx["frozen_panels"]:
        return ctx["frozen_panels"][name]
    if source == "cn_broker":
        return build_cn_broker_factor(name, ctx["pv_cache"])
    if source == "pv":
        return build_factor(name, ctx["pv_cache"])
    if source == "eod_engine":
        return build_eod_engine_factor(name, ctx["pv_cache"])
    raise ValueError(f"Unknown source: {source}")


def load_context_extended(sample_days: int) -> dict:
    start = cfg.START_DAY
    end = cfg.END_DAY

    ctx = load_context(sample_days=sample_days, engine="v2", build_cluster_reps=False)

    from factor_data_loaders import load_eod_enriched_tables
    import intraday_lib
    from factor_formulas import build_factor_cache
    from factor_formulas_liquidity_norm import build_liquidity_norm_cache

    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    norm_cache = build_liquidity_norm_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_float_mktcap=enriched.float_mktcap,
        df_total_mktcap=enriched.total_mktcap,
        df_turnover=enriched.turnover,
    )

    sample_start = max(0, len(ctx["ret"]) - sample_days - 25)
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret_s = ret_full.iloc[sample_start:]

    d5_rep = ctx["frozen_panels"].get(D5_REPRESENTATIVE)
    if d5_rep is None:
        d5_rep = build_any_eod(D5_REPRESENTATIVE, pv_cache, norm_cache).loc[start:end].iloc[sample_start:]
        ctx["frozen_panels"][D5_REPRESENTATIVE] = d5_rep

    frozen_list = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]

    def get_ret_matrix(s, e, idx):
        return Factor_Dev_Lib.get_Ret_Matrix(s, e, method="c2c", base_index=idx)

    ctx.update(
        {
            "session": session,
            "start": start,
            "end": end,
            "ret": ret_s,
            "close": ctx["close"],
            "pv_cache": pv_cache,
            "norm_cache": norm_cache,
            "d5_rep": d5_rep,
            "frozen_list": frozen_list,
            "universes": cfg.UNIVERSE_LIST,
            "get_ret_matrix": get_ret_matrix,
            "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
            "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
            "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
        }
    )
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(description="D5 Tail Alpha Density v1")
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DAYS)
    parser.add_argument("--no-reports", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== D5 Tail Fragility Alpha Density v1 ===")
    ctx = load_context_extended(args.sample_days)

    baseline_panels = list(ctx["frozen_list"])
    density_rows = []
    report_summaries = []

    for name, source, hypothesis, family in D5_TAIL_DENSITY_CANDIDATES:
        log(f"\n--- {name} ({source}, {family}) ---")
        try:
            wide = build_candidate(name, source, ctx)
            panel = wide.reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)

            enh = stack_enhancement_row(baseline_panels, panel, ctx["ret"], name)
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

            row = analyze_density_candidate(
                name,
                panel,
                ctx["ret"],
                ctx["d5_rep"],
                ctx["exposure_panels"],
                ctx["frozen_panels"],
                ctx["frozen_list"],
                hypothesis=hypothesis,
                sample_dates=args.sample_days,
                stack_enhancement_pass=bool(enh.get("stack_enhancement_pass")),
                monotonicity=mono,
                anchor_key="ic_after_d5",
                dim_label="d5",
            )
            row["source"] = source
            row["family"] = family
            row["hl_sharpe_all"] = report_summaries[-1].get("hl_sharpe") if report_summaries else np.nan
            density_rows.append(row)
            log(
                f"  IC raw={row['ic_raw']:.4f} | after D5={row['ic_after_d5']:.4f} | "
                f"corr D5={row['corr_rep']:.3f} | class={row['density_classification']}"
            )
        except Exception as exc:
            log(f"  FAILED {name}: {exc}")
        gc.collect()

    if density_rows:
        df = pd.DataFrame(density_rows).sort_values("ic_after_d5", ascending=False, key=abs)
        path = save_density_summary(df, OUT, filename="d5_tail_density_summary.csv")
        log(f"\nSaved density summary -> {path}")
        log("\n=== Classification ===")
        log(
            df[
                ["factor_name", "family", "ic_raw", "ic_after_d5", "corr_rep", "density_classification"]
            ].to_string(index=False)
        )

        independent = df[df["density_classification"] == "independent_alpha"]["factor_name"].tolist()
        partial = df[df["density_classification"] == "partial_independent"]["factor_name"].tolist()
        redundant = df[df["density_classification"].isin(["redundant", "amplification_artifact"])]["factor_name"].tolist()
        enhancers = df[df["density_classification"] == "enhancer"]["factor_name"].tolist()

        verdict = {
            "dimension": "D5",
            "representative": D5_REPRESENTATIVE,
            "research_status": "density_v1",
            "candidates_tested": len(df),
            "independent_alpha": independent,
            "partial_independent": partial,
            "redundant": redundant,
            "enhancers": enhancers,
            "recommendation": (
                "expand_d5_family" if independent else "review_tail_structure"
            ),
        }
        (OUT / "d5_tail_density_verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False) + "\n"
        )
        if report_summaries:
            pd.DataFrame(report_summaries).to_csv(OUT / "candidate_reports_summary.csv", index=False)


if __name__ == "__main__":
    main()
