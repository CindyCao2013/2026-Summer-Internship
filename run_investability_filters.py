#!/usr/bin/env python
"""Investability filters for Alpha Library v1 — fill real numbers into frozen JSON.

Runs three layers on confirmation (951d OOS) for:
  - each Base3 source (D1, D4, D5)
  - Base3 equal-weight stack
  - Base3 + λ=0.2 cn_cancel_shock (active enhancer)

Usage:
  OMP_NUM_THREADS=1 python run_investability_filters.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
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
from alpha_d4_expansion_stack import build_satellite_stack
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    classify_market_regimes,
    evaluate_investability,
    lambda_stability,
    regime_net_sharpes,
    strip_internal,
    weight_perturbation_stability,
    yearly_net_sharpes,
)
from factor_attribution import combine_equal_weight
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_l2_v2 import build_l2_v2_factor
from l2_data_loaders import build_l2_daily_cache
from run_l2_validation import load_context

OUT = Path("research/results/investability_v1")
LIBRARY_PATH = Path("research/alpha_library_v1/alpha_library_v1.0-frozen.json")
POOL_PATH = Path("research/frozen_candidate_pool_v1.json")

BASE3 = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]
SOURCE_META = {
    "low_vol_liquidity_quality_60d": {
        "source_id": "D1_low_vol_liquidity_quality_60d",
        "dim": "D1",
        "role": "production_base",
        "failure_conditions": [
            "Liquidity dry-up / extreme Amihud regime where low-vol names become illiquid traps",
            "Policy-driven small-cap liquidity shocks that invert size-liquidity ranking",
        ],
    },
    "winner_sentiment_reversal_5d": {
        "source_id": "D4_winner_sentiment_reversal_5d",
        "dim": "D4",
        "role": "production_base",
        "failure_conditions": [
            "Shrinking-volume downtrends where winner reversal fails (momentum persistence)",
            "Strong one-way bullish retail chase regimes with limited mean reversion",
        ],
    },
    "upside_fragility_20d": {
        "source_id": "D5_upside_fragility_20d",
        "dim": "D5",
        "role": "production_base",
        "failure_conditions": [
            "Sustained one-sided bull markets where upside fragility IC decays",
            "Low-volatility grind-up regimes with few fragile-rally events",
        ],
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_masks(start, end):
    return {
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
    }


def load_cancel_panel(ctx, start, end) -> pd.DataFrame:
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
    wide = build_l2_v2_factor("cn_cancel_shock", l2_cache).loc[start:end]
    return wide.reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)


def load_amount_close(start, end):
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, _ = load_eod_enriched_tables(preheat, end)
    return enriched.amount.loc[start:end], enriched.close.loc[start:end]


def pack_source_investability(
    name: str,
    inv: dict,
    net_pnl: pd.Series,
    regime: pd.Series,
) -> dict:
    meta = SOURCE_META[name]
    return {
        "factor": name,
        "source_id": meta["source_id"],
        "dim": meta["dim"],
        "role": meta["role"],
        "failure_conditions": meta["failure_conditions"],
        "investability": {
            "layer1_net_performance": {
                "round_trip_cost": inv["round_trip_cost"],
                "gross_sharpe": inv["gross_sharpe_tradable"],
                "net_sharpe": inv["net_sharpe_tradable"],
                "net_annu_ret": inv["net_annu_ret_tradable"],
                "net_max_drawdown": inv["net_max_drawdown_tradable"],
                "net_calmar": inv["net_calmar_tradable"],
                "rank_ic": inv["rank_ic_tradable"],
                "icir": inv["icir_tradable"],
            },
            "layer2_tradability": {
                "coverage_mean": inv["coverage_mean"],
                "coverage_p10": inv["coverage_p10"],
                "rank_ic_raw": inv["rank_ic_raw"],
                "rank_ic_after_tradability_filter": inv["rank_ic_tradable"],
                "net_sharpe_no_filter": inv["net_sharpe_no_tradability_filter"],
                "net_sharpe_delta_from_filter": inv["net_sharpe_delta_tradability"],
                "annu_one_way_turnover": inv["annu_one_way_turnover"],
                "capacity_cny_approx": inv["capacity_cny_approx"],
            },
            "layer3_stability": {
                "yearly_net_sharpe": yearly_net_sharpes(net_pnl),
                "regime_net_sharpe": regime_net_sharpes(net_pnl, regime),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Investability filters v1")
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== Investability Filters v1 ===")

    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    ret_full = ctx["ret"]
    ret_disc, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    log(f"Confirmation: {ret_conf.index[0].date()} -> {ret_conf.index[-1].date()} ({len(ret_conf)}d)")

    log("Loading tradability masks...")
    masks = load_masks(cfg.START_DAY, cfg.END_DAY)
    amount, close = load_amount_close(cfg.START_DAY, cfg.END_DAY)
    amount = amount.reindex(index=ret_full.index, columns=ret_full.columns)
    close = close.reindex(index=ret_full.index, columns=ret_full.columns)

    # Equal-weight market proxy for bull/bear/sideways regimes
    mkt = ret_full.mean(axis=1)
    regime_full = classify_market_regimes(mkt)
    regime_conf = regime_full.reindex(ret_conf.index)

    trad_kw = dict(
        df_not_limit=masks["df_not_limit"],
        df_not_st=masks["df_not_st"],
        df_trade_status=masks["df_trade_status"],
        close=close,
        amount=amount,
        round_trip_cost=args.cost,
        apply_tradability=True,
    )

    # --- Per-source ---
    source_rows = []
    source_packs = []
    for name in BASE3:
        log(f"\n--- Source {name} ---")
        panel = ctx["frozen_panels"][name].reindex(index=ret_conf.index, columns=ret_conf.columns)
        inv = evaluate_investability(panel, ret_conf, **trad_kw)
        pack = pack_source_investability(name, inv, inv["_net_pnl"], regime_conf)
        source_packs.append(pack)
        row = {"factor": name, "period": "confirmation", **strip_internal(inv)}
        source_rows.append(row)
        log(
            f"  net Sharpe={inv['net_sharpe_tradable']:.3f} | gross={inv['gross_sharpe_tradable']:.3f} | "
            f"IC={inv['rank_ic_tradable']:.4f} | TO={inv['annu_one_way_turnover']:.1f} | "
            f"cov={inv['coverage_mean']:.3f}"
        )
        gc.collect()

    # --- Base3 stack ---
    log("\n--- Base3 equal-weight ---")
    base3 = combine_equal_weight([ctx["frozen_panels"][n] for n in BASE3])
    base3_c = base3.reindex(index=ret_conf.index, columns=ret_conf.columns)
    inv_b3 = evaluate_investability(base3_c, ret_conf, **trad_kw)
    log(
        f"  net Sharpe={inv_b3['net_sharpe_tradable']:.3f} | gross={inv_b3['gross_sharpe_tradable']:.3f} | "
        f"IC={inv_b3['rank_ic_tradable']:.4f} | TO={inv_b3['annu_one_way_turnover']:.1f}"
    )

    # Weight perturbation stability on Base3
    log("  weight ±30% stability...")
    panels_c = {
        n: ctx["frozen_panels"][n].reindex(index=ret_conf.index, columns=ret_conf.columns)
        for n in BASE3
    }
    stab = weight_perturbation_stability(panels_c, ret_conf, scale=0.30, tradability_kwargs=trad_kw)
    log(f"  stable={stab['stable']} | net Sharpe range [{stab['net_sharpe_min']:.3f}, {stab['net_sharpe_max']:.3f}]")

    # --- cancel enhancer ---
    log("\n--- Loading cn_cancel_shock ---")
    cancel = load_cancel_panel(ctx, cfg.START_DAY, cfg.END_DAY)
    cancel_c = cancel.reindex(index=ret_conf.index, columns=ret_conf.columns)

    # Additive stack (already shifted inside build_satellite_stack)
    stack_sig = build_satellite_stack(
        base3_c, {"cn_cancel_shock": cancel_c}, 0.2, satellite_factors=["cn_cancel_shock"]
    )
    trad_kw0 = dict(trad_kw)
    trad_kw0["signal_shift"] = 0
    inv_enh = evaluate_investability(stack_sig, ret_conf, **trad_kw0)
    log(
        f"  Base3+cancel@0.2 net Sharpe={inv_enh['net_sharpe_tradable']:.3f} | "
        f"gross={inv_enh['gross_sharpe_tradable']:.3f}"
    )

    lam_stab = lambda_stability(
        base3_c, cancel_c, ret_conf, center_lambda=0.2, scale=0.30, tradability_kwargs=trad_kw
    )
    log(f"  lambda ±30% stable={lam_stab['stable']} | {lam_stab['net_sharpe_by_lambda']}")

    # Standalone cancel investability (for enhancer card)
    inv_cancel = evaluate_investability(cancel_c, ret_conf, **trad_kw)

    # --- Assemble library update ---
    base3_invest = {
        "period": "confirmation_951d",
        "round_trip_cost": args.cost,
        "cost_definition": "A-share bilateral ~0.15% (stamp 0.1% sell + commission/fees ~0.05%)",
        "layer1_net_performance": {
            "gross_sharpe": inv_b3["gross_sharpe_tradable"],
            "net_sharpe": inv_b3["net_sharpe_tradable"],
            "net_annu_ret": inv_b3["net_annu_ret_tradable"],
            "net_max_drawdown": inv_b3["net_max_drawdown_tradable"],
            "net_calmar": inv_b3["net_calmar_tradable"],
            "rank_ic": inv_b3["rank_ic_tradable"],
            "icir": inv_b3["icir_tradable"],
        },
        "layer2_tradability": {
            "filters": ["not_limit", "not_st", "trade_status", "min_listing_days_60"],
            "coverage_mean": inv_b3["coverage_mean"],
            "coverage_p10": inv_b3["coverage_p10"],
            "rank_ic_raw": inv_b3["rank_ic_raw"],
            "rank_ic_after_tradability_filter": inv_b3["rank_ic_tradable"],
            "net_sharpe_no_filter": inv_b3["net_sharpe_no_tradability_filter"],
            "net_sharpe_delta_from_filter": inv_b3["net_sharpe_delta_tradability"],
            "annu_one_way_turnover": inv_b3["annu_one_way_turnover"],
            "capacity_cny_approx": inv_b3["capacity_cny_approx"],
            "capacity_note": "median(ADV_book × n_names × 5% participation); amount in Wind 千元 scaled to CNY",
        },
        "layer3_stability": {
            "weight_perturbation_pm30pct": stab,
            "yearly_net_sharpe": yearly_net_sharpes(inv_b3["_net_pnl"]),
            "regime_net_sharpe": regime_net_sharpes(inv_b3["_net_pnl"], regime_conf),
        },
    }

    cancel_invest = {
        "factor": "cn_cancel_shock",
        "role": "conditional_enhancer",
        "interaction_mode": "state_modifier",
        "interaction_note": (
            "Preferred use: modulate D4 weight under liquidity-withdrawal states "
            "(not pure linear_additive). Current production stack still uses additive λ=0.2 "
            "as interim; Combination Layer C5 will implement state_modifier."
        ),
        "production_status": "active",
        "standalone_investability_confirmation": {
            "layer1_net_performance": {
                "gross_sharpe": inv_cancel["gross_sharpe_tradable"],
                "net_sharpe": inv_cancel["net_sharpe_tradable"],
                "rank_ic": inv_cancel["rank_ic_tradable"],
                "icir": inv_cancel["icir_tradable"],
            },
            "layer2_tradability": {
                "coverage_mean": inv_cancel["coverage_mean"],
                "annu_one_way_turnover": inv_cancel["annu_one_way_turnover"],
                "capacity_cny_approx": inv_cancel["capacity_cny_approx"],
            },
        },
        "on_base3_additive_lambda_0.2": {
            "gross_sharpe": inv_enh["gross_sharpe_tradable"],
            "net_sharpe": inv_enh["net_sharpe_tradable"],
            "net_annu_ret": inv_enh["net_annu_ret_tradable"],
            "net_max_drawdown": inv_enh["net_max_drawdown_tradable"],
            "net_calmar": inv_enh["net_calmar_tradable"],
            "rank_ic": inv_enh["rank_ic_tradable"],
            "icir": inv_enh["icir_tradable"],
            "annu_one_way_turnover": inv_enh["annu_one_way_turnover"],
            "net_sharpe_delta_vs_base3": (
                float(inv_enh["net_sharpe_tradable"] - inv_b3["net_sharpe_tradable"])
                if pd.notna(inv_enh["net_sharpe_tradable"]) and pd.notna(inv_b3["net_sharpe_tradable"])
                else np.nan
            ),
            "lambda_stability_pm30pct": lam_stab,
            "yearly_net_sharpe": yearly_net_sharpes(inv_enh["_net_pnl"]),
            "regime_net_sharpe": regime_net_sharpes(inv_enh["_net_pnl"], regime_conf),
        },
    }

    inactive_enhancers = [
        {
            "factor": "amihud_shock_reversal_5d",
            "role": "conditional_enhancer",
            "interaction_mode": "conditional",
            "production_status": "inactive_on_base3",
            "note": "Taxonomy retained; Base3 confirmation Sharpe uplift negative",
        },
        {
            "factor": "value_composite",
            "role": "conditional_enhancer",
            "interaction_mode": "conditional",
            "production_status": "inactive_on_base3",
            "note": "IC-only uplift on Base3; Sharpe negative — taxonomy only",
        },
        {
            "factor": "quality_composite",
            "role": "conditional_enhancer",
            "interaction_mode": "conditional",
            "production_status": "inactive_on_base3",
            "note": "Marginal λ=0.05 only; not robust at 0.1/0.2",
        },
    ]

    library = {
        "version": "alpha_library_v1.0-frozen",
        "frozen_at": "2026-07-09",
        "status": "frozen",
        "input_pool": str(POOL_PATH),
        "freeze_metadata": {
            "frozen_date": "2026-07-09",
            "base_dimension_count": 3,
            "base_sources": [
                "D1_low_vol_liquidity_quality_60d",
                "D4_winner_sentiment_reversal_5d",
                "D5_upside_fragility_20d",
            ],
            "active_enhancers": ["cn_cancel_shock"],
            "inactive_enhancers": [e["factor"] for e in inactive_enhancers],
            "research_satellites_count": 6,
            "author": "team",
            "note": "OHLCV Alpha Library v1.0 with investability numbers. Freeze commit/tag should be made in git-enabled environment (v1.0-frozen).",
        },
        "base": {
            "name": "Base3",
            "sources": [
                "D1_low_vol_liquidity_quality_60d",
                "D4_winner_sentiment_reversal_5d",
                "D5_upside_fragility_20d",
            ],
            "blend": "equal_weight_cs_z(D1, D4, D5)",
            "source_cards": source_packs,
            "stack_investability_confirmation_951d": base3_invest,
            "pre_cost_reference": {
                "rank_ic": 0.0383,
                "icir": 5.685,
                "hl_sharpe": 3.761,
                "note": "Pre-cost metrics from run_base3_library_freeze_v1 (no tradability/cost)",
            },
        },
        "enhancers": {
            "active": [cancel_invest],
            "inactive": inactive_enhancers,
        },
        "research_satellites": [
            "max_daily_return_20d",
            "d4_consecutive_gain_exhaustion_20d",
            "lower_shadow_support_20d",
            "drawup_drawdown_ratio_20d",
            "range_contraction_20d",
            "range_expansion_20d",
        ],
        "dropped": [
            {
                "factor": "volatility_60d",
                "source_id": "D2_volatility_60d",
                "role": "drop",
                "reason": "Absorbed by D1",
            }
        ],
        "investability_gates": {
            "status": "computed",
            "period": "confirmation_951d",
            "round_trip_cost": args.cost,
            "summary": {
                "base3_net_sharpe": inv_b3["net_sharpe_tradable"],
                "base3_plus_cancel_net_sharpe": inv_enh["net_sharpe_tradable"],
                "base3_weight_stability": stab["stable"],
                "cancel_lambda_stability": lam_stab["stable"],
            },
        },
        "next_phase": "alpha_combination_layer_c1_c5",
        "parallel_offline": "research_satellites_regime_risk_only",
    }

    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY_PATH.write_text(json.dumps(library, indent=2, ensure_ascii=False, default=str) + "\n")

    pd.DataFrame(source_rows).to_csv(OUT / "source_investability_confirmation.csv", index=False)
    summary = {
        "base3": strip_internal(inv_b3),
        "base3_plus_cancel_0.2": strip_internal(inv_enh),
        "cancel_standalone": strip_internal(inv_cancel),
        "weight_stability": stab,
        "lambda_stability": lam_stab,
    }
    (OUT / "investability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    log(f"\nLibrary updated -> {LIBRARY_PATH}")
    log(f"Artifacts -> {OUT}")
    log(
        f"\nSUMMARY confirmation net Sharpe | Base3={inv_b3['net_sharpe_tradable']:.3f} | "
        f"Base3+cancel={inv_enh['net_sharpe_tradable']:.3f} | "
        f"Δ={inv_enh['net_sharpe_tradable'] - inv_b3['net_sharpe_tradable']:+.3f}"
    )


if __name__ == "__main__":
    main()
