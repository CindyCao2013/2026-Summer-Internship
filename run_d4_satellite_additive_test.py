#!/usr/bin/env python
"""D4 Satellite Additive Test — base D1-D5 + λ·satellite (max_return, gain_exhaustion).

Usage:
  OMP_NUM_THREADS=1 python run_d4_satellite_additive_test.py
  OMP_NUM_THREADS=1 python run_d4_satellite_additive_test.py --lambdas 0 0.05 0.1 0.2 0.3
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
import pandas as pd

from alpha_d4_expansion_stack import (
    D4_SATELLITE_FACTORS,
    DEFAULT_SATELLITE_LAMBDAS,
    build_base_stack,
    classify_satellite_uplift,
    evaluate_lambda_grid,
    publish_satellite_additive_test,
)
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_behavioral_d4 import build_behavioral_d4_factor
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_liquidity_norm import build_liquidity_norm_cache
from factor_runner import get_universe_mask
from run_l2_validation import load_context

OUT = Path("research/results/d4_density_stack_validation")


def log(msg: str) -> None:
    print(msg, flush=True)


def build_satellite_panels(ctx: dict, start: dt.datetime, end: dt.datetime) -> dict:
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, _ = load_eod_enriched_tables(preheat, end)
    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    panels = {}
    for name in D4_SATELLITE_FACTORS:
        if name == "d4_consecutive_gain_exhaustion_20d":
            wide = build_behavioral_d4_factor(name, pv_cache).loc[start:end]
        else:
            wide = build_eod_engine_factor(name, pv_cache).loc[start:end]
        panels[name] = wide.reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
    return panels


def load_session(start: dt.datetime, end: dt.datetime):
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    _, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    return session


def load_universe_masks(session, index: pd.Index) -> dict:
    masks = {}
    for uni, idx_code in cfg.UNIVERSE_LIST.items():
        if idx_code is None:
            continue
        m = get_universe_mask(session, cfg.START_DAY, cfg.END_DAY, idx_code)
        masks[uni] = m.reindex(index=index)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 satellite additive test")
    parser.add_argument("--sample-days", type=int, default=9999)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=DEFAULT_SATELLITE_LAMBDAS,
        help="Satellite blend weights (same λ per factor)",
    )
    args = parser.parse_args()

    log("=== D4 Satellite Additive Test ===")
    ctx = load_context(sample_days=args.sample_days, engine="v2", build_cluster_reps=False)
    ret = ctx["ret"]
    log(f"Evaluation window: {ret.index[0].date()} -> {ret.index[-1].date()} ({len(ret)} days)")

    baseline = build_base_stack(ctx["frozen_panels"])
    satellites = build_satellite_panels(ctx, cfg.START_DAY, cfg.END_DAY)
    session = load_session(cfg.START_DAY, cfg.END_DAY)
    universe_masks = load_universe_masks(session, ret.index)
    gc.collect()

    full_df, regime_df = evaluate_lambda_grid(
        baseline,
        satellites,
        ret,
        lambdas=args.lambdas,
        universe_masks=universe_masks,
        satellite_factors=D4_SATELLITE_FACTORS,
    )
    verdict = classify_satellite_uplift(
        full_df,
        promote_label="promote_d4_satellite_layer",
        reject_label="keep_d4_base_only",
    )

    summary = {
        "task": "d4_satellite_additive_test",
        "architecture": "base_d1_d5 + lam*z(max_daily_return) + lam*z(consecutive_gain_exhaustion)",
        "d4_base_rep": "winner_sentiment_reversal_5d",
        "satellite_factors": D4_SATELLITE_FACTORS,
        "lambdas_tested": args.lambdas,
        "eval_start": str(ret.index[0].date()),
        "eval_end": str(ret.index[-1].date()),
        "n_trading_days": len(ret),
        **verdict,
    }

    path = publish_satellite_additive_test(summary, full_df, regime_df, OUT, file_prefix="d4_satellite")

    log("\n=== Lambda Grid (full sample) ===")
    cols = ["lambda", "rank_ic", "icir", "hl_sharpe", "monotonicity_score", "rank_ic_delta", "hl_sharpe_delta"]
    log(full_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    log("\n=== Regime Sharpe by λ ===")
    pivot = regime_df.pivot(index="lambda", columns="period", values="hl_sharpe")
    log(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    log(f"\nRecommendation: {verdict['recommendation']}")
    log(f"Robust uplift zone (λ=0.1 & 0.2, IC↑ & Sharpe↑): {verdict['robust_uplift_zone']}")
    log(f"Published -> {path}")


if __name__ == "__main__":
    main()
