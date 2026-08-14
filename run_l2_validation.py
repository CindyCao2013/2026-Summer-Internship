#!/usr/bin/env python
"""L2 Microstructure validation — dual gate (dimension + stack enhancement).

Engines:
  v1 (CLOSED): cn_voi_20d, cn_oir_20d, cn_mpb_20d — level bricks
  v2 (active): 6 event-driven factors — see factor_formulas_l2_v2.py

Usage:
  OMP_NUM_THREADS=1 python run_l2_validation.py --engine v1 --stage all
  OMP_NUM_THREADS=1 python run_l2_validation.py --engine v2 --stage all
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
from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from factor_attribution import (
    OHLCV_FROZEN_REPS,
    build_attribution_row,
    incremental_bundle_test,
    rank_ic_by_horizon,
)
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2 import build_l2_factor, filter_l2_factors
from factor_formulas_l2_v2 import build_l2_v2_factor, filter_l2_v2_factors
from factor_formulas_liquidity_norm import build_liquidity_norm_cache, build_liquidity_norm_factor
from factor_taxonomy_cn import (
    CN_FACTOR_TAXONOMY,
    L2_MICROSTRUCTURE_V1_LIST,
    L2_MICROSTRUCTURE_V2_LIST,
)
from l2_data_loaders import build_l2_daily_cache
from l2_stack_enhancement import run_stack_enhancement_test
from l2_validation import (
    build_l2_validation_verdict,
    classify_l2_clusters,
    publish_l2_v1_archive,
)
from run_cn_broker_validation import extend_correlation

OUT = cfg.RESEARCH_DIR
LOCK_PATH = OUT / ".l2_validation.lock"
EXISTING_CORR_PATH = OUT / "alpha_factor_correlation.csv"
EXISTING_CLUSTERS_PATH = OUT / "alpha_latent_clusters.csv"
SAMPLE_DATES = 252


def output_prefix(engine: str) -> str:
    return "l2_v2" if engine == "v2" else "l2"


def track_name(engine: str) -> str:
    return f"l2_microstructure_{engine}"


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(f"Lock active: {LOCK_PATH} — stop other validation first")
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


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


def load_context(sample_days: int, engine: str, build_cluster_reps: bool = False) -> dict:
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log(f"[{engine}] Loading EOD + L2 aggregates ({start.date()}->{end.date()})...")
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    log("  L2 daily + imbalance-duration (cached after first run)...")
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

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

    if engine == "v2":
        names = filter_l2_v2_factors(L2_MICROSTRUCTURE_V2_LIST)
        build_fn = build_l2_v2_factor
    else:
        names = filter_l2_factors(L2_MICROSTRUCTURE_V1_LIST)
        build_fn = build_l2_factor

    l2_panels = {}
    for name in names:
        try:
            wide = build_fn(name, l2_cache)
            wide = wide.reindex(index=close.index, columns=close.columns)
            l2_panels[name] = wide.loc[start:end].iloc[sample_start:]
            log(f"  l2 OK {name}")
        except Exception as exc:
            log(f"  l2 SKIP {name}: {exc}")
        gc.collect()

    ohlcv_rep_panels = {}
    if build_cluster_reps and EXISTING_CLUSTERS_PATH.exists():
        reps = pd.read_csv(EXISTING_CLUSTERS_PATH)["representative"].unique().tolist()
        for name in reps:
            if name in l2_panels or name in frozen_panels:
                continue
            try:
                ohlcv_rep_panels[name] = build_any_eod(name, pv_cache, norm_cache).loc[start:end].iloc[sample_start:]
            except Exception:
                pass
            gc.collect()

    return {
        "engine": engine,
        "ret": ret_s,
        "close": close_s,
        "l2_panels": l2_panels,
        "frozen_panels": frozen_panels,
        "exposure_panels": exposure_panels,
        "ohlcv_rep_panels": ohlcv_rep_panels,
    }


def _path(prefix: str, name: str) -> Path:
    return OUT / f"{prefix}_{name}.csv"


def run_attribution(ctx: dict, prefix: str) -> pd.DataFrame:
    rows = []
    decay_rows = []
    for i, (fname, panel) in enumerate(ctx["l2_panels"].items(), 1):
        meta = CN_FACTOR_TAXONOMY.get(fname, {})
        rows.append(
            build_attribution_row(
                fname,
                panel,
                ctx["ret"],
                ctx["exposure_panels"],
                ctx["frozen_panels"],
                cn_family=meta.get("cn_family", "microstructure"),
                hypothesis=meta.get("hypothesis", ""),
            )
        )
        decay = rank_ic_by_horizon(panel, ctx["close"])
        decay["factor_name"] = fname
        decay_rows.append(decay)
        log(f"  attribution {i}/{len(ctx['l2_panels'])} {fname}")
        gc.collect()

    attr = pd.DataFrame(rows)
    if decay_rows:
        pd.concat(decay_rows, ignore_index=True).to_csv(_path(prefix, "ic_decay"), index=False)
    attr.to_csv(_path(prefix, "attribution"), index=False)
    return attr


def run_cluster(ctx: dict, prefix: str) -> pd.DataFrame:
    from alpha_information_space import cluster_summary, hierarchical_cluster, intrinsic_dimension

    l2 = ctx["l2_panels"]
    if EXISTING_CORR_PATH.exists():
        existing = pd.read_csv(EXISTING_CORR_PATH, index_col=0)
        combined_panels = {**ctx["ohlcv_rep_panels"], **l2}
        corr = extend_correlation(existing, combined_panels)
    else:
        from alpha_information_space import correlation_matrix

        corr = correlation_matrix({**ctx["ohlcv_rep_panels"], **l2})

    labels = hierarchical_cluster(corr, distance_threshold=0.35)
    rank = pd.DataFrame({"factor_name": list(corr.index), "production_score": 0.0})
    clusters = cluster_summary(corr, labels, rank)
    tags = classify_l2_clusters(clusters, list(l2.keys()))
    combined = clusters.merge(tags, on="cluster_id", how="left")

    corr.to_csv(_path(prefix, "combined_correlation"))
    combined.to_csv(_path(prefix, "combined_clusters"), index=False)
    tags.to_csv(_path(prefix, "cluster_tags"), index=False)
    pd.DataFrame([intrinsic_dimension(corr)]).to_csv(_path(prefix, "variance_summary"), index=False)
    return tags


def run_stack_enhancement(ctx: dict, prefix: str) -> pd.DataFrame:
    baseline = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]
    df = run_stack_enhancement_test(baseline, ctx["l2_panels"], ctx["ret"])
    df.to_csv(_path(prefix, "stack_enhancement"), index=False)
    return df


def run_bundle(ctx: dict, attr: pd.DataFrame, prefix: str) -> pd.DataFrame:
    baseline = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]
    rows = []
    for fname, panel in ctx["l2_panels"].items():
        row = incremental_bundle_test(baseline, panel, ctx["ret"])
        row["factor_name"] = fname
        if fname in attr["factor_name"].values:
            row["ic_ohlcv_stack_residual"] = attr.loc[
                attr["factor_name"] == fname, "ic_after_ohlcv_stack"
            ].values[0]
            row["strict_pass"] = bool(attr.loc[attr["factor_name"] == fname, "strict_pass"].values[0])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(_path(prefix, "incremental_bundle"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="L2 Microstructure validation (dual gate)")
    parser.add_argument("--engine", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--stage",
        choices=["all", "attribution", "cluster", "bundle", "stack_enhancement"],
        default="all",
    )
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DATES)
    args = parser.parse_args()

    prefix = output_prefix(args.engine)
    track = track_name(args.engine)

    acquire_lock()
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        ctx = load_context(
            args.sample_days,
            engine=args.engine,
            build_cluster_reps=args.stage in ("all", "cluster"),
        )

        attr = pd.DataFrame()
        if args.stage in ("all", "attribution"):
            log(f"=== L2 {args.engine}: Attribution + IC decay ===")
            attr = run_attribution(ctx, prefix)
        elif _path(prefix, "attribution").exists():
            attr = pd.read_csv(_path(prefix, "attribution"))

        tags = pd.DataFrame()
        if args.stage in ("all", "cluster"):
            log(f"=== L2 {args.engine}: Combined clustering ===")
            tags = run_cluster(ctx, prefix)
        elif _path(prefix, "cluster_tags").exists():
            tags = pd.read_csv(_path(prefix, "cluster_tags"))

        incr = pd.DataFrame()
        if args.stage in ("all", "bundle"):
            log(f"=== L2 {args.engine}: Incremental bundle ===")
            if attr.empty:
                attr = pd.read_csv(_path(prefix, "attribution"))
            incr = run_bundle(ctx, attr, prefix)
        elif _path(prefix, "incremental_bundle").exists():
            incr = pd.read_csv(_path(prefix, "incremental_bundle"))

        enh = pd.DataFrame()
        if args.stage in ("all", "stack_enhancement"):
            log(f"=== L2 {args.engine}: Stack enhancement (D1–D5 + candidate) ===")
            enh = run_stack_enhancement(ctx, prefix)
        elif _path(prefix, "stack_enhancement").exists():
            enh = pd.read_csv(_path(prefix, "stack_enhancement"))

        if args.stage == "all" and len(attr) and len(tags) and len(incr):
            if enh.empty:
                enh = run_stack_enhancement(ctx, prefix)
            verdict = build_l2_validation_verdict(
                attr, tags, incr, enhancement_df=enh, track=track
            )
            verdict.to_csv(_path(prefix, "verdict"), index=False)
            if args.engine == "v1":
                publish_l2_v1_archive(attr, tags, enh, OUT)
            log("\n=== Verdict (dual gate) ===")
            log(verdict.to_string(index=False))
            if verdict.iloc[0]["dimension_gate_pass"]:
                log("\n*** Dimension gate PASS: new L2 alpha dimension ***")
            elif verdict.iloc[0]["enhancement_gate_pass"]:
                log("\n*** Enhancement gate PASS: L2 improves frozen stack ***")
            else:
                log("\n*** Both gates FAIL — refine hypothesis ***")

        if args.stage == "stack_enhancement" and len(enh):
            log("\n=== Stack enhancement ===")
            cols = [
                "factor_name",
                "candidate_solo_ic",
                "stack_ic_delta",
                "stack_sharpe_delta",
                "stack_enhancement_pass",
            ]
            log(enh[cols].to_string(index=False))
            if attr.empty and _path(prefix, "attribution").exists():
                attr = pd.read_csv(_path(prefix, "attribution"))
            if tags.empty and _path(prefix, "cluster_tags").exists():
                tags = pd.read_csv(_path(prefix, "cluster_tags"))
            if incr.empty and _path(prefix, "incremental_bundle").exists():
                incr = pd.read_csv(_path(prefix, "incremental_bundle"))
            if len(attr) and len(tags) and len(incr):
                verdict = build_l2_validation_verdict(
                    attr, tags, incr, enhancement_df=enh, track=track
                )
                verdict.to_csv(_path(prefix, "verdict"), index=False)
            if args.engine == "v1" and len(attr):
                publish_l2_v1_archive(attr, tags, enh, OUT)
                log("\nPublished l2_v1_triage.csv + updated verdict")

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

        log(f"\nDone -> {OUT}/{prefix}_*.csv")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
