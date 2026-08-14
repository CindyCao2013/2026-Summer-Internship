#!/usr/bin/env python
"""L2 Trade Flow State Engine — final validation before freeze.

Checks (retained: cancel_shock, voi_shock, mpb_shock):
  A. Time-slice stack enhancement     → run_l2_stability.py (existing)
  B. Universe stability               → this script
  C. Market-structure exposure chain  → alpha_attribution_engine

Usage:
  OMP_NUM_THREADS=1 python run_l2_final_validation.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

from alpha_attribution_engine import build_exposure_attribution_row
from factor_attribution import OHLCV_FROZEN_REPS
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor, build_factor_cache
from factor_runner import get_universe_mask
from l2_conditioning_layer import L2_STATE_LAYER
from l2_stack_enhancement import stack_enhancement_row
from run_l2_validation import build_any_eod, load_context

OUT = cfg.RESEARCH_DIR
RETAIN = [x["factor"] for x in L2_STATE_LAYER]
UNIVERSES = {
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
    "ALL": None,
}


def apply_universe_mask(panel: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    if mask is None:
        return panel
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m == 1)


def build_market_exposures(enriched, pv_cache, norm_cache, index_slice) -> dict:
    start, end = cfg.START_DAY, cfg.END_DAY
    size = np.log(enriched.float_mktcap.replace(0, np.nan))
    vol = build_any_eod("volatility_60d", pv_cache, norm_cache)
    liq = build_factor("amount_20d_mean", pv_cache)
    mom = build_factor("momentum_20d", pv_cache)
    return {
        "size": size.loc[index_slice],
        "volatility": vol.loc[index_slice],
        "liquidity": liq.loc[index_slice],
        "momentum_reversal": mom.loc[index_slice],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading L2 v2 context...", flush=True)
    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    baseline_full = [ctx["frozen_panels"][r] for r in OHLCV_FROZEN_REPS if r in ctx["frozen_panels"]]

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
    from factor_formulas_liquidity_norm import build_liquidity_norm_cache

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
    idx = ctx["ret"].index
    market_exp = build_market_exposures(enriched, pv_cache, norm_cache, idx)

    # B. Universe stability
    print("\n=== B. Universe stack enhancement ===", flush=True)
    uni_rows = []
    for uni_name, idx_code in UNIVERSES.items():
        mask = None
        if idx_code is not None:
            mask = get_universe_mask(session, start, end, idx_code).reindex(index=idx)
        ret_u = apply_universe_mask(ctx["ret"], mask)
        baseline_u = [apply_universe_mask(p, mask) for p in baseline_full]
        for fname in RETAIN:
            if fname not in ctx["l2_panels"]:
                continue
            panel_u = apply_universe_mask(ctx["l2_panels"][fname], mask)
            row = stack_enhancement_row(baseline_u, panel_u, ret_u, fname)
            row["universe"] = uni_name
            uni_rows.append(row)
        print(f"  {uni_name}", flush=True)

    uni_df = pd.DataFrame(uni_rows)
    uni_path = OUT / "l2_v2_universe_stability.csv"
    uni_df.to_csv(uni_path, index=False)
    print(f"Saved -> {uni_path}")

    # C. Exposure attribution on L2 state signals
    print("\n=== C. L2 enhancer exposure attribution ===", flush=True)
    exp_rows = []
    for fname in RETAIN:
        if fname not in ctx["l2_panels"]:
            continue
        exp_rows.append(
            build_exposure_attribution_row(
                fname,
                ctx["l2_panels"][fname],
                ctx["ret"],
                market_exp,
                frozen_panels=ctx["frozen_panels"],
                role="l2_trade_flow_state",
            )
        )
    exp_df = pd.DataFrame(exp_rows)
    exp_path = OUT / "l2_v2_exposure_attribution.csv"
    exp_df.to_csv(exp_path, index=False)
    print(f"Saved -> {exp_path}")
    cols = ["factor_name", "ic_raw", "ic_after_size", "ic_after_volatility", "ic_true_residual", "dominant_exposure_drop"]
    print(exp_df[cols].to_string(index=False))

    # Verdict summary
    time_summary_path = OUT / "l2_v2_stability_summary.csv"
    time_ok = time_summary_path.exists()
    uni_pass = {}
    for fname in RETAIN:
        sub = uni_df[uni_df["factor_name"] == fname] if len(uni_df) else pd.DataFrame()
        if sub.empty:
            uni_pass[fname] = False
            continue
        uni_pass[fname] = bool((sub["stack_enhancement_pass"] == True).all())  # noqa: E712

    verdict = {
        "engine": "L2 Trade Flow State Engine",
        "status": "FROZEN_ENHANCER_LAYER",
        "retained_factors": RETAIN,
        "dropped_factors": ["cn_flow_persistence", "cn_imbalance_duration", "cn_liquidity_consumption"],
        "data_ceiling": "no LOB/SSL2 in DDB — see L2_DATA_LINEAGE.md",
        "time_stability_artifact": str(time_summary_path) if time_ok else "run run_l2_stability.py",
        "universe_all_pass": uni_pass,
        "primary_enhancer_target": "D4 behavioral (see alpha_enhancer_targets_v1.csv)",
        "next_research_track": "Fundamental Quality block (D7)",
        "stopped": ["L2 v3 order book", "VOI/MPB window search", "liquidity_consumption optimization"],
    }
    verdict_path = OUT / "l2_trade_flow_state_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"\nVerdict -> {verdict_path}")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
