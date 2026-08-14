#!/usr/bin/env python
"""Run CN Broker v1 validation — single-process, CPU-safe.

Resource policy:
  - ONE heavy job at a time (process lock)
  - BLAS/OpenMP limited to 1 thread (avoid 2000%+ CPU)
  - Default lite mode: 252-day window, partial correlation vs existing OHLCV map

Usage:
  python run_cn_broker_validation.py              # full lite pipeline
  python run_cn_broker_validation.py --stage attribution
  python run_cn_broker_validation.py --stage cluster
  python run_cn_broker_validation.py --stage bundle
  python run_cn_broker_validation.py --with-universe   # slower: CSI300/500/1000 IC
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
import sys
from pathlib import Path

# Limit numpy/pandas BLAS threads BEFORE import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import pandas as pd

import Factor_Dev_Lib
from alpha_dimension_map import OHLCV_PRODUCTION_DIMENSIONS
from alpha_information_space import (
    cluster_summary,
    correlation_matrix,
    dedupe_ranking,
    hierarchical_cluster,
    intrinsic_dimension,
)
from cn_broker_validation import (
    build_validation_verdict,
    classify_new_clusters,
    run_full_cn_broker_validation,
)
from factor_attribution import (
    OHLCV_FROZEN_REPS,
    build_attribution_row,
    incremental_bundle_test,
    rank_ic_by_horizon,
    universe_ic_table,
)
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import FACTOR_REGISTRY, build_factor, build_factor_cache
from factor_formulas_cn_broker import CN_BROKER_REGISTRY, build_cn_broker_factor
from factor_formulas_eod_engine import EOD_ENGINE_REGISTRY, build_eod_engine_factor
from factor_formulas_liquidity_norm import (
    LIQUIDITY_NORM_REGISTRY,
    build_liquidity_norm_cache,
    build_liquidity_norm_factor,
)
from factor_runner import get_universe_mask
from factor_taxonomy_cn import CN_FACTOR_TAXONOMY, EOD_CN_BROKER_ALL_LIST

OUT = cfg.RESEARCH_DIR
CACHE_DIR = OUT / "cache"
LOCK_PATH = OUT / ".cn_broker_validation.lock"
RANKING_PATH = OUT / "robust_alpha_ranking.csv"
EXISTING_CORR_PATH = OUT / "alpha_factor_correlation.csv"
EXISTING_CLUSTERS_PATH = OUT / "alpha_latent_clusters.csv"

SAMPLE_DATES = 252  # lite default (~1 year)


def log(msg: str) -> None:
    print(msg, flush=True)


def acquire_lock() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        pid = LOCK_PATH.read_text().strip()
        raise RuntimeError(
            f"Another validation run may be active (lock {LOCK_PATH}, pid={pid}). "
            "Stop it first: pkill -f run_cn_broker_validation.py"
        )
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink(missing_ok=True)


def build_any_factor(name: str, pv_cache, norm_cache) -> pd.DataFrame:
    if name in CN_BROKER_REGISTRY:
        return build_cn_broker_factor(name, pv_cache)
    if name in LIQUIDITY_NORM_REGISTRY:
        return build_liquidity_norm_factor(name, norm_cache)
    if name in EOD_ENGINE_REGISTRY:
        return build_eod_engine_factor(name, pv_cache)
    if name in FACTOR_REGISTRY:
        return build_factor(name, pv_cache)
    raise KeyError(name)


def extend_correlation(existing: pd.DataFrame, new_panels: dict, sample_per_factor: int = 40_000) -> pd.DataFrame:
    """Append CN factors to existing OHLCV correlation without recomputing OHLCV×OHLCV."""
    from alpha_information_space import stack_panel

    all_names = list(existing.index) + [n for n in new_panels if n not in existing.index]
    corr = pd.DataFrame(1.0, index=all_names, columns=all_names)
    corr.loc[existing.index, existing.columns] = existing.values

    sampled_old = {}
    for name in existing.index:
        if name not in new_panels:
            continue
        s = stack_panel(new_panels[name]).dropna()
        if len(s):
            sampled_old[name] = s.sample(n=min(sample_per_factor, len(s)), random_state=42)

    sampled_new = {}
    for name, wide in new_panels.items():
        if name in existing.index:
            continue
        s = stack_panel(wide).dropna()
        if len(s) > sample_per_factor:
            s = s.sample(n=sample_per_factor, random_state=42)
        sampled_new[name] = s

    for new_name, ns in sampled_new.items():
        for old_name in existing.index:
            if old_name in sampled_old:
                os_ = sampled_old[old_name]
            elif old_name in new_panels:
                os_ = stack_panel(new_panels[old_name]).dropna()
                if len(os_) > sample_per_factor:
                    os_ = os_.sample(n=sample_per_factor, random_state=42)
            else:
                continue
            aligned = pd.concat([ns, os_], axis=1, join="inner").dropna()
            c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) > 5000 else float("nan")
            corr.loc[new_name, old_name] = c
            corr.loc[old_name, new_name] = c
        # new × new
        for new_name2, ns2 in sampled_new.items():
            if new_name >= new_name2:
                continue
            aligned = pd.concat([ns, ns2], axis=1, join="inner").dropna()
            c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) > 5000 else float("nan")
            corr.loc[new_name, new_name2] = c
            corr.loc[new_name2, new_name] = c

    return corr


def load_data_and_panels(sample_dates: int, build_cluster_reps: bool = False):
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log(f"Loading EOD ({start.date()}->{end.date()})...")
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
    sample_start = max(0, len(ret) - sample_dates - 25)
    ret_s = ret.iloc[sample_start:]
    close_s = close.iloc[sample_start:]

    import numpy as np

    log_float = np.log(norm_cache.norm_data.float_mktcap.replace(0, np.nan))
    frozen_panels = {}
    for spec in OHLCV_PRODUCTION_DIMENSIONS:
        try:
            frozen_panels[spec.representative] = build_any_factor(
                spec.representative, pv_cache, norm_cache
            ).loc[start:end].iloc[sample_start:]
            log(f"  frozen OK {spec.representative}")
        except Exception as exc:
            log(f"  frozen SKIP {spec.representative}: {exc}")

    exposure_panels = {
        "size": log_float.iloc[sample_start:],
        "liquidity": frozen_panels.get(
            "low_vol_liquidity_quality_60d", pv_cache.get("amount_mean_20d").iloc[sample_start:]
        ),
        "volatility": frozen_panels.get(
            "volatility_60d", pv_cache.get("volatility_60d").iloc[sample_start:]
        ),
    }

    cn_panels = {}
    for name in EOD_CN_BROKER_ALL_LIST:
        try:
            cn_panels[name] = build_cn_broker_factor(name, pv_cache).loc[start:end].iloc[sample_start:]
            log(f"  cn OK {name}")
        except Exception as exc:
            log(f"  cn SKIP {name}: {exc}")
        gc.collect()

    ohlcv_rep_panels = {}
    if build_cluster_reps and EXISTING_CLUSTERS_PATH.exists():
        reps = pd.read_csv(EXISTING_CLUSTERS_PATH)["representative"].unique().tolist()
        for name in reps:
            if name in cn_panels or name in frozen_panels:
                ohlcv_rep_panels[name] = cn_panels.get(name) or frozen_panels.get(name)
                continue
            try:
                ohlcv_rep_panels[name] = build_any_factor(name, pv_cache, norm_cache).loc[start:end].iloc[sample_start:]
            except Exception:
                pass
            gc.collect()

    return {
        "session": session,
        "ret": ret_s,
        "close": close_s,
        "cn_panels": cn_panels,
        "frozen_panels": frozen_panels,
        "exposure_panels": exposure_panels,
        "ohlcv_rep_panels": ohlcv_rep_panels,
    }


def run_attribution(ctx: dict, with_universe: bool) -> pd.DataFrame:
    rows = []
    decay_rows = []
    uni_rows = []
    uni_stab = []
    masks = {}
    if with_universe:
        for uni, idx in cfg.UNIVERSE_LIST.items():
            if idx is None:
                continue
            try:
                masks[uni] = get_universe_mask(ctx["session"], cfg.START_DAY, cfg.END_DAY, idx)
                masks[uni] = masks[uni].iloc[-len(ctx["ret"]):]
            except Exception as exc:
                log(f"  [WARN] mask {uni}: {exc}")

    for i, (fname, panel) in enumerate(ctx["cn_panels"].items(), 1):
        meta = CN_FACTOR_TAXONOMY.get(fname, {})
        rows.append(
            build_attribution_row(
                fname, panel, ctx["ret"], ctx["exposure_panels"], ctx["frozen_panels"],
                cn_family=meta.get("cn_family", ""),
                hypothesis=meta.get("hypothesis", ""),
            )
        )
        decay = rank_ic_by_horizon(panel, ctx["close"])
        decay["factor_name"] = fname
        decay_rows.append(decay)
        if masks:
            uni = universe_ic_table(panel, ctx["ret"], masks)
            for _, u in uni.iterrows():
                uni_rows.append({"factor_name": fname, **u.to_dict()})
            uni_stab.append(
                {
                    "factor_name": fname,
                    "universe_stability": uni.attrs.get("universe_stability"),
                    "sign_consistency": uni.attrs.get("sign_consistency"),
                }
            )
        log(f"  attribution {i}/{len(ctx['cn_panels'])} {fname}")
        gc.collect()

    attr = pd.DataFrame(rows)
    if decay_rows:
        decay_df = pd.concat(decay_rows, ignore_index=True)
        best_h = (
            decay_df.groupby("factor_name")
            .apply(lambda g: g.loc[g["abs_rank_ic"].idxmax(), "horizon_days"])
            .reset_index(name="best_horizon_days")
        )
        attr = attr.merge(best_h, on="factor_name", how="left")
        decay_df.to_csv(OUT / "cn_broker_ic_decay.csv", index=False)
    if uni_rows:
        pd.DataFrame(uni_rows).to_csv(OUT / "cn_broker_universe_stability.csv", index=False)
    if uni_stab:
        attr = attr.merge(pd.DataFrame(uni_stab), on="factor_name", how="left")
    attr.to_csv(OUT / "cn_broker_attribution.csv", index=False)
    return attr


def run_cluster(ctx: dict, ranking: pd.DataFrame) -> tuple:
    cn = ctx["cn_panels"]
    if EXISTING_CORR_PATH.exists():
        log("  extending existing OHLCV correlation matrix (no full rebuild)...")
        existing = pd.read_csv(EXISTING_CORR_PATH, index_col=0)
        combined_panels = {**ctx["ohlcv_rep_panels"], **cn}
        corr = extend_correlation(existing, combined_panels)
    else:
        log("  no saved correlation — building from cluster reps + CN...")
        combined_panels = {**ctx["ohlcv_rep_panels"], **cn}
        corr = correlation_matrix(combined_panels)

    intrinsic = intrinsic_dimension(corr)
    labels = hierarchical_cluster(corr, distance_threshold=0.35)
    rank = dedupe_ranking(ranking) if len(ranking) else pd.DataFrame(
        {"factor_name": list(corr.index), "production_score": 0.0}
    )
    clusters = cluster_summary(corr, labels, rank)
    tags = classify_new_clusters(clusters, list(cn.keys()))
    combined = clusters.merge(tags, on="cluster_id", how="left")

    corr.to_csv(OUT / "cn_broker_combined_correlation.csv")
    combined.to_csv(OUT / "cn_broker_combined_clusters.csv", index=False)
    tags.to_csv(OUT / "cn_broker_cluster_tags.csv", index=False)
    pd.DataFrame([intrinsic]).to_csv(OUT / "cn_broker_variance_summary.csv", index=False)
    return combined, tags, intrinsic


def run_bundle(ctx: dict, attr: pd.DataFrame) -> pd.DataFrame:
    baseline = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]
    rows = []
    for fname, panel in ctx["cn_panels"].items():
        row = incremental_bundle_test(baseline, panel, ctx["ret"])
        row["factor_name"] = fname
        if fname in attr["factor_name"].values:
            row["ic_ohlcv_stack_residual"] = attr.loc[
                attr["factor_name"] == fname, "ic_after_ohlcv_stack"
            ].values[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cn_broker_incremental_bundle.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="CN Broker v1 validation (CPU-safe)")
    parser.add_argument(
        "--stage",
        choices=["all", "attribution", "cluster", "bundle"],
        default="all",
    )
    parser.add_argument("--with-universe", action="store_true", help="Per-universe IC (slower)")
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DATES)
    args = parser.parse_args()

    if not RANKING_PATH.exists():
        raise FileNotFoundError(f"Missing {RANKING_PATH}")

    acquire_lock()
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        ranking = pd.read_csv(RANKING_PATH)
        need_reps = args.stage in ("all", "cluster")
        ctx = load_data_and_panels(args.sample_days, build_cluster_reps=need_reps)

        attr = pd.DataFrame()
        if args.stage in ("all", "attribution"):
            log("=== Stage: Attribution + IC decay ===")
            attr = run_attribution(ctx, with_universe=args.with_universe)
        elif (OUT / "cn_broker_attribution.csv").exists():
            attr = pd.read_csv(OUT / "cn_broker_attribution.csv")

        tags = pd.DataFrame()
        if args.stage in ("all", "cluster"):
            log("=== Stage: Combined clustering ===")
            _, tags, _ = run_cluster(ctx, ranking)

        incr = pd.DataFrame()
        if args.stage in ("all", "bundle"):
            log("=== Stage: Incremental bundle ===")
            if attr.empty and (OUT / "cn_broker_attribution.csv").exists():
                attr = pd.read_csv(OUT / "cn_broker_attribution.csv")
            incr = run_bundle(ctx, attr)

        if args.stage == "all" and len(attr) and len(tags) and len(incr):
            verdict = build_validation_verdict(attr, tags, incr)
            verdict.to_csv(OUT / "cn_broker_verdict.csv", index=False)
            log("\n=== Stage Gate ===")
            log(verdict.to_string(index=False))

        log(f"\nDone. Outputs -> {OUT}/cn_broker_*.csv")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
