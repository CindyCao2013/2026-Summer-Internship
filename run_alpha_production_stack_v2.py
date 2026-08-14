#!/usr/bin/env python
"""Build and evaluate Alpha Production Stack v2.

Usage:
  OMP_NUM_THREADS=1 python run_alpha_production_stack_v2.py
  OMP_NUM_THREADS=1 python run_alpha_production_stack_v2.py --sample-days 504
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

from alpha_production_stack_v2 import (
    DEFAULT_ENHANCER_LAMBDAS,
    evaluate_stack_v2,
    load_enhancer_lambdas_from_attribution,
    publish_production_stack_v2,
)
from factor_data_loaders import load_derivative_wide_tables, load_eod_enriched_tables, load_financial_ttmhis_long
from factor_formulas_fundamental import build_fundamental_cache, build_fundamental_factor
from run_l2_validation import build_any_eod, load_context

OUT = cfg.RESEARCH_DIR
LOCK_PATH = OUT / ".alpha_production_stack_v2.lock"


def log(msg: str) -> None:
    print(msg, flush=True)


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        raise RuntimeError(f"Lock active: {LOCK_PATH}")
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Production Stack v2")
    parser.add_argument("--sample-days", type=int, default=504)
    args = parser.parse_args()

    acquire_lock()
    try:
        log("=== Alpha Production Stack v2 ===")
        ctx = load_context(sample_days=args.sample_days, engine="v2", build_cluster_reps=False)
        frozen = ctx["frozen_panels"]
        ret = ctx["ret"]

        start = cfg.START_DAY
        end = cfg.END_DAY
        preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
        enriched, session = load_eod_enriched_tables(preheat, end)
        session.run(intraday_lib.ddb_functions)

        finance_long, _ = load_financial_ttmhis_long(preheat, end, session=session)
        der_tables, _ = load_derivative_wide_tables(preheat, end, session=session)
        fund_cache = build_fundamental_cache(
            der_tables, close=enriched.close, finance_long=finance_long
        )
        sample_start = max(0, len(ret) - args.sample_days - 25)
        quality = build_fundamental_factor("quality_composite", fund_cache)
        quality = quality.loc[start:end].iloc[sample_start:]
        gc.collect()

        cancel = ctx["l2_panels"].get("cn_cancel_shock")
        if cancel is None:
            raise RuntimeError("cn_cancel_shock panel missing from L2 context")

        enhancers = {
            "cn_cancel_shock": cancel,
            "quality_composite": quality.reindex(index=cancel.index, columns=cancel.columns),
        }

        lambdas = load_enhancer_lambdas_from_attribution()
        log(f"Enhancer lambdas: {lambdas}")

        metrics = evaluate_stack_v2(frozen, enhancers, ret, lambdas=lambdas)
        path = publish_production_stack_v2(
            metrics,
            lambdas,
            out_dir=OUT,
            sample_days=args.sample_days,
            n_days=len(ret),
        )

        log("\n=== Stack v2 Metrics ===")
        log(f"  IC base:     {metrics.ic_base:.4f}")
        log(f"  IC enhanced: {metrics.ic_enhanced:.4f}  (Δ {metrics.ic_delta:+.4f})")
        log(f"  H-L Sharpe base:     {metrics.sharpe_base:.3f}")
        log(f"  H-L Sharpe enhanced: {metrics.sharpe_enhanced:.3f}  (Δ {metrics.sharpe_delta:+.3f})")
        log(f"\nPublished -> {path}")

    finally:
        release_lock()


if __name__ == "__main__":
    main()
