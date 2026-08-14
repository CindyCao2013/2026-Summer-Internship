#!/usr/bin/env python
"""D4 Behavioral Alpha Expansion Stack Validation.

Compare frozen D1–D5 base vs D4-expanded base (3-factor equal-weight composite).

Usage:
  OMP_NUM_THREADS=1 python run_d4_density_stack_validation.py
  OMP_NUM_THREADS=1 python run_d4_density_stack_validation.py --sample-days 9999
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
    D4_EXPANSION_FACTORS,
    build_base_stack,
    build_d4_composite,
    compare_stacks,
    evaluate_regime_comparison,
    evaluate_stack_signal,
    publish_d4_stack_validation,
)
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_behavioral_d4 import build_behavioral_d4_factor
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_liquidity_norm import build_liquidity_norm_cache
from factor_runner import get_universe_mask
from run_l2_validation import build_any_eod, load_context

OUT = Path("research/results/d4_density_stack_validation")


def log(msg: str) -> None:
    print(msg, flush=True)


def build_d4_expansion_panels(ctx: dict, start: dt.datetime, end: dt.datetime) -> dict:
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
    panels = {}
    for name in D4_EXPANSION_FACTORS:
        if name in ctx["frozen_panels"]:
            wide = ctx["frozen_panels"][name]
        elif name == "d4_consecutive_gain_exhaustion_20d":
            wide = build_behavioral_d4_factor(name, pv_cache).loc[start:end]
        elif name == "max_daily_return_20d":
            wide = build_eod_engine_factor(name, pv_cache).loc[start:end]
        else:
            wide = build_any_eod(name, pv_cache, norm_cache).loc[start:end]
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
    parser = argparse.ArgumentParser(description="D4 density stack validation")
    parser.add_argument(
        "--sample-days",
        type=int,
        default=9999,
        help="Trading days for evaluation (9999 = full 2020-2025 sample)",
    )
    args = parser.parse_args()

    log("=== D4 Behavioral Expansion Stack Validation ===")
    ctx = load_context(sample_days=args.sample_days, engine="v2", build_cluster_reps=False)
    ret = ctx["ret"]
    log(f"Evaluation window: {ret.index[0].date()} -> {ret.index[-1].date()} ({len(ret)} days)")

    d4_panels = build_d4_expansion_panels(ctx, cfg.START_DAY, cfg.END_DAY)
    d4_composite = build_d4_composite(d4_panels)
    baseline = build_base_stack(ctx["frozen_panels"], d4_composite=None)
    expanded = build_base_stack(ctx["frozen_panels"], d4_composite=d4_composite)

    session = load_session(cfg.START_DAY, cfg.END_DAY)
    universe_masks = load_universe_masks(session, ret.index)
    gc.collect()

    base_m, exp_m, delta = compare_stacks(baseline, expanded, ret, universe_masks, period="full")
    regime_df = evaluate_regime_comparison(baseline, expanded, ret, universe_masks=universe_masks)

    universe_rows = []
    for stack_name, signal in [("baseline_d1_d5", baseline), ("d4_expanded_base", expanded)]:
        m = evaluate_stack_signal(signal, ret, universe_masks)
        for uni, ic in m.get("universe_ic", {}).items():
            universe_rows.append(
                {"stack": stack_name, "universe": uni, "rank_ic": ic, "abs_rank_ic": abs(ic)}
            )
    universe_df = pd.DataFrame(universe_rows)

    summary = {
        "task": "d4_density_stack_validation",
        "sample_days_requested": args.sample_days,
        "eval_start": str(ret.index[0].date()),
        "eval_end": str(ret.index[-1].date()),
        "n_trading_days": len(ret),
        "d4_expansion_factors": D4_EXPANSION_FACTORS,
        "d4_composite": "equal_weight_cs_z(winner_sentiment, max_daily_return, consecutive_gain_exhaustion)",
        "baseline": {"name": "frozen_d1_d5", "d4_slot": "winner_sentiment_reversal_5d", **base_m},
        "expanded": {"name": "d4_expanded_d1_d5", "d4_slot": "d4_composite_3factor", **exp_m},
        "full_sample_delta": delta,
        "recommendation": (
            "promote_d4_composite"
            if delta.get("rank_ic_delta", 0) > 0.005 and delta.get("hl_sharpe_delta", 0) > 0.05
            else "keep_d4_rep_only"
        ),
    }

    path = publish_d4_stack_validation(summary, regime_df, universe_df, OUT)

    log("\n=== Full Sample ===")
    log(f"  Baseline IC:   {base_m['rank_ic']:.4f}  Sharpe: {base_m['hl_sharpe']:.3f}  mono: {base_m['monotonicity_score']:.2f}")
    log(f"  Expanded IC:   {exp_m['rank_ic']:.4f}  Sharpe: {exp_m['hl_sharpe']:.3f}  mono: {exp_m['monotonicity_score']:.2f}")
    log(f"  Delta IC:      {delta['rank_ic_delta']:+.4f}  Sharpe: {delta['hl_sharpe_delta']:+.3f}")
    log(f"\nRecommendation: {summary['recommendation']}")
    log(f"Published -> {path}")


if __name__ == "__main__":
    main()
