#!/usr/bin/env python
"""Alpha Validation Harness v1 — unified entry for factor QC.

Usage:
  OMP_NUM_THREADS=1 python run_alpha_validation_harness.py --track fundamental_phase2
  OMP_NUM_THREADS=1 python run_alpha_validation_harness.py --track fundamental_phase2 --stage attribution
  OMP_NUM_THREADS=1 python run_alpha_validation_harness.py --track fundamental_batch1
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
from alpha_validation_harness import (
    HarnessConfig,
    build_harness_summary_verdict,
    run_attribution_stage,
    run_bundle_stage,
    run_cluster_stage,
    run_enhancement_stage,
    save_harness_outputs,
)
from factor_data_loaders import (
    load_derivative_wide_tables,
    load_eod_enriched_tables,
    load_financial_ttmhis_long,
)
from factor_formulas import build_factor, build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_fundamental import (
    FUNDAMENTAL_BATCH1_LIST,
    FUNDAMENTAL_PHASE2_BATCH_LIST,
    FUNDAMENTAL_QUALITY_D7_BATCH_LIST,
    build_fundamental_cache,
    build_fundamental_factor,
    filter_available_fundamental_factors,
)
from factor_formulas_value import (
    FUNDAMENTAL_VALUE_D6_BATCH_LIST,
    build_value_factor,
    filter_available_value_factors,
)
from factor_formulas_liquidity_norm import build_liquidity_norm_cache, build_liquidity_norm_factor
from run_cn_broker_validation import extend_correlation

OUT = cfg.RESEARCH_DIR
SAMPLE_DATES = 252

TRACK_SPECS = {
    "fundamental_phase2": {
        "factor_list": FUNDAMENTAL_PHASE2_BATCH_LIST,
        "candidate_prefix": "fundamental",
        "output_prefix": "fundamental_phase2",
        "needs_finance_ann": True,
        "meta": {
            "roe": {
                "family": "quality",
                "hypothesis": "ROE TTM (ann_date aligned) — profitability quality",
            },
            "roe_stability": {
                "family": "quality",
                "hypothesis": "ROE stability (8-report rolling std) — D7 Quality pillar",
            },
        },
    },
    "fundamental_quality_d7": {
        "factor_list": FUNDAMENTAL_QUALITY_D7_BATCH_LIST,
        "candidate_prefix": "fundamental",
        "output_prefix": "fundamental_quality_d7",
        "needs_finance_ann": True,
        "meta": {
            "quality_composite": {
                "family": "quality",
                "hypothesis": "D7 rep: equal-z(roe_stability, GP/A, CFO/NI)",
                "role": "dimension_representative",
            },
            "gross_profitability": {
                "family": "quality",
                "hypothesis": "GP/A brick (Novy-Marx gross profitability)",
                "role": "composite_brick",
            },
            "cfo_quality": {
                "family": "quality",
                "hypothesis": "CFO/NI brick (cash earnings quality)",
                "role": "composite_brick",
            },
        },
    },
    "fundamental_value_d6": {
        "factor_list": FUNDAMENTAL_VALUE_D6_BATCH_LIST,
        "candidate_prefix": "value",
        "output_prefix": "fundamental_value_d6",
        "needs_finance_ann": True,
        "builder": "value",
        "meta": {
            "value_composite": {
                "family": "value",
                "hypothesis": "D6 rep: equal-z(EP, BP, CFP) ind+size neutral",
                "role": "dimension_representative",
            },
            "value_ep": {
                "family": "value",
                "hypothesis": "EP = 1/PE TTM, industry + size neutral",
                "role": "composite_brick",
            },
            "value_bp": {
                "family": "value",
                "hypothesis": "BP = 1/PB, industry + size neutral",
                "role": "composite_brick",
            },
            "value_cfp": {
                "family": "value",
                "hypothesis": "CFP = CFO TTM / float mktcap, ann_date + ind + size neutral",
                "role": "composite_brick",
            },
        },
    },
    "fundamental_batch1": {
        "factor_list": FUNDAMENTAL_BATCH1_LIST,
        "candidate_prefix": "fundamental",
        "output_prefix": "fundamental_batch1",
        "needs_finance_ann": False,
        "meta": {
            "ep_ttm_ind_neutral": {
                "family": "value",
                "hypothesis": "Industry-neutral earnings yield",
            },
            "bp_ind_neutral": {
                "family": "value",
                "hypothesis": "Industry-neutral book-to-price",
            },
            "ep_ttm": {"family": "value", "hypothesis": "Raw EP (sanity-masked)"},
            "bp": {"family": "value", "hypothesis": "Raw BP (sanity-masked)"},
        },
    },
}

EXISTING_CORR_PATH = OUT / "alpha_factor_correlation.csv"
EXISTING_CLUSTERS_PATH = OUT / "alpha_latent_clusters.csv"
LOCK_PATH = OUT / ".alpha_validation_harness.lock"


def log(msg: str) -> None:
    print(msg, flush=True)


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(f"Lock active: {LOCK_PATH}")
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


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


def load_harness_context(track: str, sample_days: int, build_cluster_reps: bool = False) -> dict:
    spec = TRACK_SPECS[track]
    start = cfg.START_DAY
    end = cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log(f"[{track}] Loading EOD + derivative ({start.date()}->{end.date()})...")
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    der_tables, _ = load_derivative_wide_tables(preheat, end, session=session)

    finance_long = pd.DataFrame()
    if spec.get("needs_finance_ann"):
        log("Loading ASHARETTMHIS announcements (ann_date)...")
        finance_long, _ = load_financial_ttmhis_long(preheat, end, session=session)
        log(f"  finance_long rows: {len(finance_long):,}")

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
    fund_cache = build_fundamental_cache(
        der_tables,
        close=enriched.close,
        finance_long=finance_long if len(finance_long) else None,
    )

    close = enriched.close.loc[start:end]
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    sample_start = max(0, len(ret) - sample_days - 25)
    ret_s = ret.iloc[sample_start:]
    close_s = close.iloc[sample_start:]

    frozen_panels = {}
    for dim in OHLCV_PRODUCTION_DIMENSIONS:
        try:
            frozen_panels[dim.representative] = build_any_eod(
                dim.representative, pv_cache, norm_cache
            ).loc[start:end].iloc[sample_start:]
            log(f"  frozen OK {dim.representative}")
        except Exception as exc:
            log(f"  frozen SKIP {dim.representative}: {exc}")

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

    names = filter_available_value_factors(
        spec["factor_list"],
        has_pb=der_tables.pb is not None,
        has_pe=der_tables.pe_ttm is not None,
        has_finance_ann=len(finance_long) > 0,
    ) if spec.get("builder") == "value" else filter_available_fundamental_factors(
        spec["factor_list"],
        has_pb=der_tables.pb is not None,
        has_pe=der_tables.pe_ttm is not None,
        has_ps=der_tables.ps_ttm is not None,
        has_finance_ann=len(finance_long) > 0,
    )
    build_fn = build_value_factor if spec.get("builder") == "value" else build_fundamental_factor
    candidate_panels = {}
    for name in names:
        try:
            wide = build_fn(name, fund_cache)
            wide = wide.reindex(index=close.index, columns=close.columns)
            candidate_panels[name] = wide.loc[start:end].iloc[sample_start:]
            log(f"  candidate OK {name}")
        except Exception as exc:
            log(f"  candidate SKIP {name}: {exc}")
        gc.collect()

    ohlcv_rep_panels = {}
    if build_cluster_reps and EXISTING_CLUSTERS_PATH.exists():
        reps = pd.read_csv(EXISTING_CLUSTERS_PATH)["representative"].unique().tolist()
        for name in reps:
            if name in candidate_panels or name in frozen_panels:
                continue
            try:
                ohlcv_rep_panels[name] = build_any_eod(name, pv_cache, norm_cache).loc[start:end].iloc[sample_start:]
            except Exception:
                pass
            gc.collect()

    harness_config = HarnessConfig(
        track=track,
        candidate_prefix=spec["candidate_prefix"],
        output_prefix=spec["output_prefix"],
    )

    return {
        "ret": ret_s,
        "close": close_s,
        "candidate_panels": candidate_panels,
        "frozen_panels": frozen_panels,
        "exposure_panels": exposure_panels,
        "ohlcv_rep_panels": ohlcv_rep_panels,
        "harness_config": harness_config,
        "factor_meta": spec.get("meta", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Validation Harness v1")
    parser.add_argument(
        "--track",
        choices=list(TRACK_SPECS.keys()),
        default="fundamental_phase2",
    )
    parser.add_argument("--stage", choices=["all", "attribution", "cluster", "bundle"], default="all")
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DATES)
    args = parser.parse_args()

    acquire_lock()
    try:
        ctx = load_harness_context(
            args.track,
            args.sample_days,
            build_cluster_reps=args.stage in ("all", "cluster"),
        )
        config: HarnessConfig = ctx["harness_config"]
        panels = ctx["candidate_panels"]
        if not panels:
            log("No candidate panels built — exit.")
            sys.exit(1)

        attr = pd.DataFrame()
        ic_decay = pd.DataFrame()
        if args.stage in ("all", "attribution"):
            log("=== Harness: Attribution + IC decay ===")
            attr, ic_decay = run_attribution_stage(
                panels,
                ctx["ret"],
                ctx["close"],
                ctx["exposure_panels"],
                ctx["frozen_panels"],
                ctx["factor_meta"],
            )
        elif (OUT / f"{config.output_prefix}_attribution.csv").exists():
            attr = pd.read_csv(OUT / f"{config.output_prefix}_attribution.csv")

        tags = pd.DataFrame()
        corr = pd.DataFrame()
        combined = pd.DataFrame()
        if args.stage in ("all", "cluster"):
            log("=== Harness: Clustering ===")
            corr, combined, tags = run_cluster_stage(
                panels,
                ctx["ohlcv_rep_panels"],
                list(panels.keys()),
                config=config,
                extend_corr_fn=extend_correlation,
                existing_corr_path=EXISTING_CORR_PATH,
            )

        incr = pd.DataFrame()
        if args.stage in ("all", "bundle"):
            log("=== Harness: Incremental bundle ===")
            if attr.empty:
                attr = pd.read_csv(OUT / f"{config.output_prefix}_attribution.csv")
            incr = run_bundle_stage(panels, ctx["frozen_panels"], ctx["ret"], attr)

        enh = pd.DataFrame()
        if args.stage == "all" and config.enable_enhancement_gate:
            log("=== Harness: Stack enhancement ===")
            enh = run_enhancement_stage(panels, ctx["frozen_panels"], ctx["ret"])

        if args.stage == "all" and len(attr) and len(tags) and len(incr):
            summary, factor_verdicts = build_harness_summary_verdict(
                attr, tags, incr, enh if len(enh) else None, config=config
            )
            save_harness_outputs(
                OUT,
                config=config,
                attribution_df=attr,
                ic_decay_df=ic_decay,
                corr=corr,
                combined_clusters=combined,
                cluster_tags=tags,
                incremental_df=incr,
                enhancement_df=enh if len(enh) else None,
                summary_verdict=summary,
                factor_verdicts=factor_verdicts,
            )
            log("\n=== Harness Summary ===")
            log(summary.to_string(index=False))
            log("\n=== Per-Factor Verdicts ===")
            log(factor_verdicts.to_string(index=False))
        elif len(attr):
            attr.to_csv(OUT / f"{config.output_prefix}_attribution.csv", index=False)
            if len(ic_decay):
                ic_decay.to_csv(OUT / f"{config.output_prefix}_ic_decay.csv", index=False)

    finally:
        release_lock()


if __name__ == "__main__":
    main()
