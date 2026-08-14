#!/usr/bin/env python
"""D4 Behavioral Alpha Density Mining v1.

Tests whether new behavioral candidates add independent alpha beyond D4 rep,
or are redundant expressions of winner_sentiment_reversal.

Usage:
  OMP_NUM_THREADS=1 python run_d4_behavioral_density_v1.py
  OMP_NUM_THREADS=1 python run_d4_behavioral_density_v1.py --sample-days 504 --no-reports
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_density_mining import analyze_density_candidate, save_density_summary
from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from alpha_research_report import (
    build_factor_report,
    publish_factor_report,
    report_summary_row,
)
from factor_attribution import OHLCV_FROZEN_REPS
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_behavioral_d4 import (
    D4_BEHAVIORAL_DENSITY_CANDIDATES,
    D4_REPRESENTATIVE,
    build_behavioral_d4_factor,
)
from factor_formulas_cn_broker import build_cn_broker_factor
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_liquidity_norm import build_liquidity_norm_cache
from l2_stack_enhancement import stack_enhancement_row
from run_l2_validation import build_any_eod

OUT = Path("research/reports/d4_behavioral_density_v1")
SAMPLE_DAYS = 504


def log(msg: str) -> None:
    print(msg, flush=True)


def build_candidate(name: str, source: str, pv_cache, norm_cache) -> pd.DataFrame:
    if source == "eod_engine":
        return build_eod_engine_factor(name, pv_cache)
    if source == "cn_broker":
        return build_cn_broker_factor(name, pv_cache)
    if source == "behavioral_d4":
        return build_behavioral_d4_factor(name, pv_cache)
    raise ValueError(f"Unknown source: {source}")


def load_context(sample_days: int) -> dict:
    start = cfg.START_DAY
    end = cfg.END_DAY
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

    close = enriched.close.loc[start:end]
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    sample_start = max(0, len(ret) - sample_days - 25)
    ret_s = ret.iloc[sample_start:]

    frozen_panels = {}
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        try:
            frozen_panels[spec.representative] = build_any_eod(
                spec.representative, pv_cache, norm_cache
            ).loc[start:end].iloc[sample_start:]
        except Exception as exc:
            log(f"  frozen SKIP {spec.representative}: {exc}")

    d4_rep = frozen_panels.get(D4_REPRESENTATIVE)
    if d4_rep is None:
        d4_rep = build_eod_engine_factor(D4_REPRESENTATIVE, pv_cache).loc[start:end].iloc[sample_start:]
        frozen_panels[D4_REPRESENTATIVE] = d4_rep

    exposure_panels = {
        "size": np.log(enriched.float_mktcap.replace(0, np.nan)).iloc[sample_start:],
        "liquidity": frozen_panels.get(
            "low_vol_liquidity_quality_60d",
            pv_cache.get("amount_mean_20d").iloc[sample_start:],
        ),
        "volatility": frozen_panels.get(
            "volatility_60d", pv_cache.get("volatility_60d").iloc[sample_start:]
        ),
    }

    frozen_list = [frozen_panels[r] for r in OHLCV_FROZEN_REPS if r in frozen_panels]

    def get_ret_matrix(s, e, idx):
        return Factor_Dev_Lib.get_Ret_Matrix(s, e, method="c2c", base_index=idx)

    return {
        "start": start,
        "end": end,
        "session": session,
        "close": close.iloc[sample_start:],
        "ret": ret_s,
        "pv_cache": pv_cache,
        "norm_cache": norm_cache,
        "frozen_panels": frozen_panels,
        "frozen_list": frozen_list,
        "d4_rep": d4_rep,
        "exposure_panels": exposure_panels,
        "universes": cfg.UNIVERSE_LIST,
        "get_ret_matrix": get_ret_matrix,
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 Behavioral Alpha Density v1")
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DAYS)
    parser.add_argument("--no-reports", action="store_true", help="Skip per-factor alpha reports")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== D4 Behavioral Alpha Density v1 ===")
    ctx = load_context(args.sample_days)

    baseline_panels = list(ctx["frozen_list"])
    density_rows = []
    report_summaries = []

    for name, source, hypothesis in D4_BEHAVIORAL_DENSITY_CANDIDATES:
        log(f"\n--- {name} ({source}) ---")
        try:
            wide = build_candidate(name, source, ctx["pv_cache"], ctx["norm_cache"])
            panel = wide.reindex(index=ctx["close"].index, columns=ctx["close"].columns)

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
                ctx["d4_rep"],
                ctx["exposure_panels"],
                ctx["frozen_panels"],
                ctx["frozen_list"],
                hypothesis=hypothesis,
                sample_dates=args.sample_days,
                stack_enhancement_pass=bool(enh.get("stack_enhancement_pass")),
                monotonicity=mono,
                anchor_key="ic_after_d4",
                dim_label="d4",
            )
            row["source"] = source
            row["hl_sharpe_all"] = report_summaries[-1].get("hl_sharpe") if report_summaries else np.nan
            density_rows.append(row)
            log(
                f"  IC raw={row['ic_raw']:.4f} | after D4={row['ic_after_d4']:.4f} | "
                f"corr D4={row['corr_d4_rep']:.3f} | class={row['density_classification']}"
            )
        except Exception as exc:
            log(f"  FAILED {name}: {exc}")
        gc.collect()

    if density_rows:
        df = pd.DataFrame(density_rows).sort_values("ic_after_d4", ascending=False, key=abs)
        path = save_density_summary(df, OUT, filename="d4_behavioral_density_summary.csv")
        log(f"\nSaved density summary -> {path}")
        log("\n=== Classification ===")
        log(df[["factor_name", "ic_raw", "ic_after_d4", "corr_d4_rep", "density_classification"]].to_string(index=False))

        independent = df[df["density_classification"] == "independent_alpha"]["factor_name"].tolist()
        partial = df[df["density_classification"] == "partial_independent"]["factor_name"].tolist()
        redundant = df[
            df["density_classification"].isin(["redundant", "amplification_artifact"])
        ]["factor_name"].tolist()
        verdict = {
            "dimension": "D4",
            "representative": D4_REPRESENTATIVE,
            "candidates_tested": len(df),
            "independent_alpha": independent,
            "partial_independent": partial,
            "redundant": redundant,
            "recommendation": (
                "expand_d4_family"
                if independent
                else (
                    "monitor_partial_candidates"
                    if partial
                    else "d4_saturated_use_rep_only"
                )
            ),
        }
        import json

        (OUT / "d4_behavioral_density_verdict.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False) + "\n"
        )
        if report_summaries:
            pd.DataFrame(report_summaries).to_csv(OUT / "candidate_reports_summary.csv", index=False)


if __name__ == "__main__":
    main()
