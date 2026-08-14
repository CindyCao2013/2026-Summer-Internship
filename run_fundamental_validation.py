#!/usr/bin/env python
"""Fundamental Batch 1 validation — same pipeline as CN Broker (attribution / cluster / bundle).

Factors: ep_ttm_ind_neutral, bp_ind_neutral (+ raw ep/bp for comparison)

Usage:
  OMP_NUM_THREADS=1 python run_fundamental_validation.py --stage attribution
  OMP_NUM_THREADS=1 python run_fundamental_validation.py --stage cluster
  OMP_NUM_THREADS=1 python run_fundamental_validation.py --stage bundle
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

import Factor_Dev_Lib
from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from cn_broker_validation import build_validation_verdict, classify_new_clusters
from factor_attribution import (
    OHLCV_FROZEN_REPS,
    build_attribution_row,
    incremental_bundle_test,
    rank_ic_by_horizon,
)
from factor_data_loaders import load_derivative_wide_tables, load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_fundamental import (
    FUNDAMENTAL_BATCH1_LIST,
    FUNDAMENTAL_REGISTRY,
    build_fundamental_cache,
    build_fundamental_factor,
    filter_available_fundamental_factors,
)
from factor_formulas_liquidity_norm import build_liquidity_norm_cache, build_liquidity_norm_factor
from run_cn_broker_validation import extend_correlation

OUT = cfg.RESEARCH_DIR
LOCK_PATH = OUT / ".fundamental_validation.lock"
EXISTING_CORR_PATH = OUT / "alpha_factor_correlation.csv"
EXISTING_CLUSTERS_PATH = OUT / "alpha_latent_clusters.csv"
SAMPLE_DATES = 252


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(f"Lock active: {LOCK_PATH} — stop other validation first")
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


FUNDAMENTAL_META = {
    "ep_ttm_ind_neutral": {
        "family": "value",
        "hypothesis": "Industry-neutral earnings yield — pricing vs peers",
    },
    "bp_ind_neutral": {
        "family": "value",
        "hypothesis": "Industry-neutral book-to-price — value dimension",
    },
    "ep_ttm": {"family": "value", "hypothesis": "Raw EP (sanity-masked) — baseline"},
    "bp": {"family": "value", "hypothesis": "Raw BP (sanity-masked) — baseline"},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def build_any_eod(name: str, pv_cache, norm_cache) -> pd.DataFrame:
    if name in {"low_vol_liquidity_quality_60d"}:
        try:
            return build_liquidity_norm_factor(name, norm_cache)
        except Exception:
            pass
    try:
        return build_eod_engine_factor(name, pv_cache)
    except Exception:
        return build_factor(name, pv_cache)


def load_context(sample_days: int, build_cluster_reps: bool = False) -> dict:
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log(f"Loading EOD + derivative ({start.date()}->{end.date()})...")
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    der_tables, _ = load_derivative_wide_tables(preheat, end, session=session)

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
    fund_cache = build_fundamental_cache(der_tables)

    close = enriched.close.loc[start:end]
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    sample_start = max(0, len(ret) - sample_days - 25)
    ret_s = ret.iloc[sample_start:]
    close_s = close.iloc[sample_start:]

    frozen_panels = {}
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        try:
            frozen_panels[spec.representative] = build_any_eod(
                spec.representative, pv_cache, norm_cache
            ).loc[start:end].iloc[sample_start:]
            log(f"  frozen OK {spec.representative}")
        except Exception as exc:
            log(f"  frozen SKIP {spec.representative}: {exc}")

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

    names = filter_available_fundamental_factors(
        FUNDAMENTAL_BATCH1_LIST,
        has_pb=der_tables.pb is not None,
        has_pe=der_tables.pe_ttm is not None,
        has_ps=der_tables.ps_ttm is not None,
    )
    fund_panels = {}
    for name in names:
        try:
            wide = build_fundamental_factor(name, fund_cache)
            wide = wide.reindex(index=close.index, columns=close.columns)
            fund_panels[name] = wide.loc[start:end].iloc[sample_start:]
            log(f"  fundamental OK {name}")
        except Exception as exc:
            log(f"  fundamental SKIP {name}: {exc}")
        gc.collect()

    ohlcv_rep_panels = {}
    if build_cluster_reps and EXISTING_CLUSTERS_PATH.exists():
        reps = pd.read_csv(EXISTING_CLUSTERS_PATH)["representative"].unique().tolist()
        for name in reps:
            if name in fund_panels or name in frozen_panels:
                continue
            try:
                ohlcv_rep_panels[name] = build_any_eod(name, pv_cache, norm_cache).loc[start:end].iloc[sample_start:]
            except Exception:
                pass
            gc.collect()

    return {
        "ret": ret_s,
        "close": close_s,
        "fund_panels": fund_panels,
        "frozen_panels": frozen_panels,
        "exposure_panels": exposure_panels,
        "ohlcv_rep_panels": ohlcv_rep_panels,
    }


def run_attribution(ctx: dict) -> pd.DataFrame:
    rows = []
    decay_rows = []
    for i, (fname, panel) in enumerate(ctx["fund_panels"].items(), 1):
        meta = FUNDAMENTAL_META.get(fname, {})
        rows.append(
            build_attribution_row(
                fname,
                panel,
                ctx["ret"],
                ctx["exposure_panels"],
                ctx["frozen_panels"],
                cn_family=meta.get("family", "fundamental"),
                hypothesis=meta.get("hypothesis", ""),
            )
        )
        decay = rank_ic_by_horizon(panel, ctx["close"])
        decay["factor_name"] = fname
        decay_rows.append(decay)
        log(f"  attribution {i}/{len(ctx['fund_panels'])} {fname}")
        gc.collect()

    attr = pd.DataFrame(rows)
    if decay_rows:
        pd.concat(decay_rows, ignore_index=True).to_csv(OUT / "fundamental_ic_decay.csv", index=False)
    attr.to_csv(OUT / "fundamental_attribution.csv", index=False)
    return attr


def run_cluster(ctx: dict) -> pd.DataFrame:
    from alpha_information_space import cluster_summary, hierarchical_cluster, intrinsic_dimension, dedupe_ranking

    fund = ctx["fund_panels"]
    if EXISTING_CORR_PATH.exists():
        existing = pd.read_csv(EXISTING_CORR_PATH, index_col=0)
        combined_panels = {**ctx["ohlcv_rep_panels"], **fund}
        corr = extend_correlation(existing, combined_panels)
    else:
        from alpha_information_space import correlation_matrix

        corr = correlation_matrix({**ctx["ohlcv_rep_panels"], **fund})

    labels = hierarchical_cluster(corr, distance_threshold=0.35)
    rank = pd.DataFrame({"factor_name": list(corr.index), "production_score": 0.0})
    clusters = cluster_summary(corr, labels, rank)
    tags = classify_new_clusters(clusters, list(fund.keys()))
    combined = clusters.merge(tags, on="cluster_id", how="left")

    corr.to_csv(OUT / "fundamental_combined_correlation.csv")
    combined.to_csv(OUT / "fundamental_combined_clusters.csv", index=False)
    tags.to_csv(OUT / "fundamental_cluster_tags.csv", index=False)
    pd.DataFrame([intrinsic_dimension(corr)]).to_csv(OUT / "fundamental_variance_summary.csv", index=False)
    return tags


def run_bundle(ctx: dict, attr: pd.DataFrame) -> pd.DataFrame:
    baseline = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]
    rows = []
    for fname, panel in ctx["fund_panels"].items():
        row = incremental_bundle_test(baseline, panel, ctx["ret"])
        row["factor_name"] = fname
        if fname in attr["factor_name"].values:
            row["ic_ohlcv_stack_residual"] = attr.loc[
                attr["factor_name"] == fname, "ic_after_ohlcv_stack"
            ].values[0]
            row["strict_pass"] = bool(attr.loc[attr["factor_name"] == fname, "strict_pass"].values[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fundamental_incremental_bundle.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fundamental Batch 1 validation")
    parser.add_argument("--stage", choices=["all", "attribution", "cluster", "bundle"], default="all")
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DATES)
    args = parser.parse_args()

    global LOCK_PATH
    acquire_lock()
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        ctx = load_context(args.sample_days, build_cluster_reps=args.stage in ("all", "cluster"))

        attr = pd.DataFrame()
        if args.stage in ("all", "attribution"):
            log("=== Fundamental: Attribution + IC decay ===")
            attr = run_attribution(ctx)
        elif (OUT / "fundamental_attribution.csv").exists():
            attr = pd.read_csv(OUT / "fundamental_attribution.csv")

        tags = pd.DataFrame()
        if args.stage in ("all", "cluster"):
            log("=== Fundamental: Combined clustering ===")
            tags = run_cluster(ctx)

        incr = pd.DataFrame()
        if args.stage in ("all", "bundle"):
            log("=== Fundamental: Incremental bundle ===")
            if attr.empty:
                attr = pd.read_csv(OUT / "fundamental_attribution.csv")
            incr = run_bundle(ctx, attr)

        if args.stage == "all" and len(attr) and len(tags) and len(incr):
            verdict = build_validation_verdict(attr, tags, incr)
            verdict["track"] = "fundamental_batch1"
            verdict.to_csv(OUT / "fundamental_verdict.csv", index=False)
            log("\n=== Verdict ===")
            log(verdict.to_string(index=False))

        if len(attr):
            log("\n=== Attribution (strict triage) ===")
            cols = [
                "factor_name",
                "ic_raw",
                "ic_after_ohlcv_stack",
                "strict_pass",
                "conclusion",
                "conclusion_loose",
            ]
            log(attr[cols].to_string(index=False))

        log(f"\nDone -> {OUT}/fundamental_*.csv")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
