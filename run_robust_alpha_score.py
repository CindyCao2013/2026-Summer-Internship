#!/usr/bin/env python
"""Run Robust Alpha Score Engine across the full factor library."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import factor_config as cfg
import pandas as pd

import Factor_Dev_Lib
from alpha_frozen_stack_v1 import publish_frozen_stack_v1
from robust_alpha_engine import (
    attach_monotonicity,
    build_robust_ranking,
    collect_factor_summaries,
)

OUT_DIR = cfg.RESEARCH_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_market_regime_proxy(start: dt.datetime, end: dt.datetime) -> tuple:
    """CSI300 index c2c return + 20d rolling vol as regime proxy."""
    idx_ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c", base_index="000300.SH")
    if idx_ret is None or idx_ret.empty:
        return None, None
    market_ret = idx_ret.mean(axis=1)
    market_vol = market_ret.rolling(20, min_periods=10).std()
    return market_ret, market_vol


def main():
    summaries = collect_factor_summaries()
    print(f"Loaded {len(summaries)} universe-level summaries")
    if "track" in summaries.columns:
        by_track = summaries.groupby("track")["factor_name"].nunique()
        print("Factors by track:")
        for track, n in by_track.sort_index().items():
            print(f"  {track}: {n}")

    market_ret, market_vol = load_market_regime_proxy(cfg.START_DAY, cfg.END_DAY)

    ranking = build_robust_ranking(
        summaries,
        market_ret=market_ret,
        market_vol=market_vol,
        min_universes=3,
    )
    ranking = attach_monotonicity(ranking, summaries)

    stack_path = publish_frozen_stack_v1(ranking, out_dir=OUT_DIR)
    print(f"Published frozen stack v1 -> {stack_path}")

    out_path = OUT_DIR / "robust_alpha_ranking.csv"
    ranking.to_csv(out_path, index=False)
    print(f"Saved robust ranking -> {out_path} ({len(ranking)} factors)")

    # Production tier: stable + meaningful IC in all universes
    prod = ranking[
        (ranking["universe_stability"] >= 0.5)
        & (ranking["sign_consistency"] >= 0.75)
        & (ranking["mean_abs_ic"] >= 0.02)
        & (ranking["all_universe_ic_hit"] >= 0.75)
    ].copy()
    prod_path = OUT_DIR / "robust_alpha_production_tier.csv"
    prod.to_csv(prod_path, index=False)
    print(f"Saved production tier -> {prod_path} ({len(prod)} factors)")

    cols = [
        "track",
        "factor_name",
        "production_score",
        "mean_abs_ic",
        "universe_stability",
        "sign_consistency",
        "ic_range",
        "mean_hl_sharpe",
        "mono_score_all",
        "regime_stability",
    ]
    cols = [c for c in cols if c in ranking.columns]
    print("\n=== Top 15 by production_score (IC × stability) ===")
    print(ranking[cols].head(15).to_string(index=False))

    if len(prod):
        print("\n=== Production tier (stable across universes) ===")
        print(prod[cols].head(10).to_string(index=False))
    else:
        print("\n(No factors pass production tier filters yet — run eod_engine_robust batch)")


if __name__ == "__main__":
    main()
