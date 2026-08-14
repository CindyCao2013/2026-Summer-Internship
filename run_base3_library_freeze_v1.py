#!/usr/bin/env python
"""Base3 freeze validation — Base3 vs Base5 + enhancer λ-grid on Base3.

Architecture (Library v1.0-frozen):
  Base3 = eq_wt(D1, D4, D5)
  Base5 = eq_wt(D1..D5)   # legacy comparison only
  Enhancers on Base3: cn_cancel_shock, amihud_shock_reversal_5d, value_composite, quality_composite

Usage:
  OMP_NUM_THREADS=1 python run_base3_library_freeze_v1.py
  OMP_NUM_THREADS=1 python run_base3_library_freeze_v1.py --discovery-days 504
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import intraday_lib
import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import (
    DEFAULT_SATELLITE_LAMBDAS,
    build_satellite_stack,
    evaluate_stack_signal,
)
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_frozen_stack_v1 import FROZEN_OHLCV_REPS
from factor_attribution import align_signal, combine_equal_weight, hl_sharpe_from_composite
from factor_data_loaders import (
    load_derivative_wide_tables,
    load_eod_enriched_tables,
    load_financial_ttmhis_long,
)
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_fundamental import build_fundamental_cache, build_fundamental_factor
from factor_formulas_l2_v2 import build_l2_v2_factor
from factor_formulas_value import build_value_factor
from l2_data_loaders import build_l2_daily_cache
from run_l2_validation import load_context

OUT = Path("research/results/base3_library_v1")
LIBRARY_OUT = Path("research/alpha_library_v1")

BASE3_FACTORS = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]
BASE5_FACTORS = [spec["factor"] for spec in FROZEN_OHLCV_REPS]

ENHANCER_SPECS = [
    ("cn_cancel_shock", "l2"),
    ("amihud_shock_reversal_5d", "eod_engine"),
    ("value_composite", "value"),
    ("quality_composite", "quality"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def build_eq_stack(frozen_panels: Dict[str, pd.DataFrame], names: Sequence[str]) -> pd.DataFrame:
    parts = [frozen_panels[n] for n in names]
    return combine_equal_weight(parts)


def max_drawdown_from_hl(signal: pd.DataFrame, ret: pd.DataFrame, signal_shift: int = 1) -> float:
    """Max drawdown of cumulative H-L daily PnL (direction-adjusted)."""
    sig = align_signal(signal, signal_shift)
    aligned = ret.reindex_like(sig)
    daily_hl = []
    for dt_i in sig.index:
        s = sig.loc[dt_i]
        r = aligned.loc[dt_i]
        mask = s.notna() & r.notna()
        if mask.sum() < 50:
            continue
        ranks = s[mask].rank(pct=True)
        top = r[mask][ranks >= 0.9].mean()
        bot = r[mask][ranks <= 0.1].mean()
        daily_hl.append(top - bot)
    if len(daily_hl) < 50:
        return np.nan
    s = pd.Series(daily_hl)
    if s.mean() < 0:
        s = -s
    cum = s.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return float(dd.min())


def evaluate_base_window(signal: pd.DataFrame, ret: pd.DataFrame, period: str) -> dict:
    m = evaluate_stack_signal(signal, ret)
    sharpe, ann_ret, direction = hl_sharpe_from_composite(signal, ret)
    return {
        "period": period,
        "n_days": len(ret),
        "rank_ic": m["rank_ic"],
        "icir": m["icir"],
        "ic_positive_ratio": m["ic_positive_ratio"],
        "hl_sharpe": sharpe,
        "hl_annu_ret": ann_ret,
        "hl_max_drawdown": max_drawdown_from_hl(signal, ret),
        "monotonicity_score": m["monotonicity_score"],
        "direction": direction,
    }


def load_enhancer_panels(ctx: dict, start: dt.datetime, end: dt.datetime) -> Dict[str, pd.DataFrame]:
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
    panels: Dict[str, pd.DataFrame] = {}

    # L2 cancel shock
    try:
        l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
        wide = build_l2_v2_factor("cn_cancel_shock", l2_cache).loc[start:end]
        panels["cn_cancel_shock"] = wide.reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
        log("  enhancer OK cn_cancel_shock")
    except Exception as exc:
        log(f"  enhancer SKIP cn_cancel_shock: {exc}")

    # Amihud
    try:
        wide = build_eod_engine_factor("amihud_shock_reversal_5d", pv_cache).loc[start:end]
        panels["amihud_shock_reversal_5d"] = wide.reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
        log("  enhancer OK amihud_shock_reversal_5d")
    except Exception as exc:
        log(f"  enhancer SKIP amihud_shock_reversal_5d: {exc}")

    # Fundamentals
    try:
        finance_long, _ = load_financial_ttmhis_long(preheat, end, session=session)
        der_tables, _ = load_derivative_wide_tables(preheat, end, session=session)
        fund_cache = build_fundamental_cache(
            der_tables,
            close=enriched.close,
            finance_long=finance_long if finance_long is not None and len(finance_long) else None,
        )
        for name, builder in [
            ("value_composite", build_value_factor),
            ("quality_composite", build_fundamental_factor),
        ]:
            try:
                wide = builder(name, fund_cache).loc[start:end]
                panels[name] = wide.reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
                log(f"  enhancer OK {name}")
            except Exception as exc:
                log(f"  enhancer SKIP {name}: {exc}")
    except Exception as exc:
        log(f"  fundamental load SKIP: {exc}")

    return panels


def evaluate_enhancer_on_base3(
    base3: pd.DataFrame,
    enhancer_name: str,
    enhancer_panel: pd.DataFrame,
    ret: pd.DataFrame,
    lambdas: Sequence[float],
    period: str,
) -> pd.DataFrame:
    # Baseline must use the same stack construction as λ>0 (already align+z inside
    # build_satellite_stack). Evaluating raw base3 would double-shift and bias deltas.
    baseline_signal = build_satellite_stack(
        base3,
        {enhancer_name: enhancer_panel},
        0.0,
        satellite_factors=[enhancer_name],
    )
    baseline = evaluate_stack_signal(baseline_signal, ret, signal_shift=0)
    rows = []
    for lam in lambdas:
        signal = build_satellite_stack(
            base3,
            {enhancer_name: enhancer_panel},
            lam,
            satellite_factors=[enhancer_name],
        )
        metrics = evaluate_stack_signal(signal, ret, signal_shift=0)
        rows.append(
            {
                "period": period,
                "enhancer": enhancer_name,
                "lambda": lam,
                **metrics,
                "rank_ic_delta": metrics["rank_ic"] - baseline["rank_ic"],
                "icir_delta": metrics["icir"] - baseline["icir"],
                "hl_sharpe_delta": metrics["hl_sharpe"] - baseline["hl_sharpe"],
                "monotonicity_delta": metrics["monotonicity_score"] - baseline["monotonicity_score"],
            }
        )
    return pd.DataFrame(rows)


def classify_enhancer_uplift(grid: pd.DataFrame) -> dict:
    zone = grid[grid["lambda"].isin([0.1, 0.2])]
    if zone.empty:
        return {"retain": False, "recommendation": "insufficient_data"}
    sharpe_ok = (zone["hl_sharpe_delta"] > 0).all()
    ic_ok = (zone["rank_ic_delta"] > 0).all()
    if sharpe_ok:
        rec = "retain_enhancer"
    elif ic_ok:
        rec = "conditional_ic_only"
    else:
        rec = "review_or_drop"
    return {
        "retain": bool(sharpe_ok),
        "ic_uplift_at_01_02": bool(ic_ok),
        "sharpe_uplift_at_01_02": bool(sharpe_ok),
        "max_hl_sharpe_delta": float(grid["hl_sharpe_delta"].max()),
        "recommendation": rec,
    }


def build_library_document(
    base_compare: pd.DataFrame,
    enhancer_verdicts: dict,
    pool: dict,
) -> dict:
    conf = base_compare[base_compare["period"] == "confirmation"]
    b3 = conf[conf["stack"] == "Base3"].iloc[0].to_dict() if not conf.empty else {}
    b5 = conf[conf["stack"] == "Base5"].iloc[0].to_dict() if not conf.empty else {}

    return {
        "version": "alpha_library_v1.0-frozen",
        "frozen_at": "2026-07-09",
        "status": "frozen",
        "input_pool": "research/frozen_candidate_pool_v1.json",
        "base": {
            "name": "Base3",
            "sources": pool.get("ohlcv_base_sources", []),
            "blend": "equal_weight_cs_z(D1, D4, D5)",
            "confirmation_951d": {
                "rank_ic": b3.get("rank_ic"),
                "icir": b3.get("icir"),
                "hl_sharpe": b3.get("hl_sharpe"),
                "hl_annu_ret": b3.get("hl_annu_ret"),
                "hl_max_drawdown": b3.get("hl_max_drawdown"),
                "monotonicity_score": b3.get("monotonicity_score"),
            },
            "vs_legacy_base5_confirmation": {
                "base5_hl_sharpe": b5.get("hl_sharpe"),
                "sharpe_delta_base3_minus_base5": (
                    float(b3["hl_sharpe"]) - float(b5["hl_sharpe"])
                    if b3.get("hl_sharpe") is not None and b5.get("hl_sharpe") is not None
                    else None
                ),
                "ic_delta_base3_minus_base5": (
                    float(b3["rank_ic"]) - float(b5["rank_ic"])
                    if b3.get("rank_ic") is not None and b5.get("rank_ic") is not None
                    else None
                ),
            },
        },
        "enhancers": [
            {
                "factor": name,
                "role": "conditional_enhancer",
                "base3_revalidation": enhancer_verdicts.get(name, {}),
            }
            for name, _ in ENHANCER_SPECS
        ],
        "research_satellites": pool.get("research_satellites_global", []),
        "dropped": [
            {
                "factor": "volatility_60d",
                "source_id": "D2_volatility_60d",
                "role": "drop",
                "reason": "Absorbed by D1",
            }
        ],
        "investability_gates": {
            "status": "template_pending_cost_layer",
            "layers": [
                {
                    "layer": 1,
                    "name": "net_performance",
                    "checks": ["hl_sharpe_after_cost", "icir_oos", "max_drawdown"],
                    "note": "Fill after transaction-cost model; Base3 confirmation metrics are pre-cost",
                },
                {
                    "layer": 2,
                    "name": "tradability",
                    "checks": ["limit_up_down_coverage", "avg_turnover_decile", "universe_stability"],
                    "note": "Use Tier-A reports + universe IC tables",
                },
                {
                    "layer": 3,
                    "name": "parameter_stability",
                    "checks": ["lambda_grid_robustness", "regime_slice_sign_consistency"],
                    "note": "Enhancer λ∈{0.1,0.2} must retain Sharpe uplift on confirmation",
                },
            ],
        },
        "next_phase": "alpha_combination_layer_on_base3",
        "parallel_offline": "research_satellites_regime_risk_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Base3 Library v1 freeze validation")
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_SATELLITE_LAMBDAS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LIBRARY_OUT.mkdir(parents=True, exist_ok=True)

    log("=== Base3 Library v1 Freeze Validation ===")
    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    ret_full = ctx["ret"]
    ret_disc, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    log(f"Full: {ret_full.index[0].date()} -> {ret_full.index[-1].date()} ({len(ret_full)}d)")
    log(f"Discovery: {len(ret_disc)}d | Confirmation: {len(ret_conf)}d")

    base3 = build_eq_stack(ctx["frozen_panels"], BASE3_FACTORS)
    base5 = build_eq_stack(ctx["frozen_panels"], BASE5_FACTORS)

    compare_rows = []
    for period, ret_w in [("discovery", ret_disc), ("confirmation", ret_conf), ("full", ret_full)]:
        if len(ret_w) < 60:
            continue
        for name, stack in [("Base3", base3), ("Base5", base5)]:
            sig = stack.reindex(index=ret_w.index, columns=ret_w.columns)
            row = evaluate_base_window(sig, ret_w, period)
            row["stack"] = name
            compare_rows.append(row)
            log(
                f"  [{period}] {name}: IC={row['rank_ic']:.4f} ICIR={row['icir']:.2f} "
                f"Sharpe={row['hl_sharpe']:.3f} MDD={row['hl_max_drawdown']:.4f}"
            )

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(OUT / "base3_vs_base5_metrics.csv", index=False)

    # Deltas confirmation
    conf = compare_df[compare_df["period"] == "confirmation"]
    if len(conf) >= 2:
        b3 = conf[conf["stack"] == "Base3"].iloc[0]
        b5 = conf[conf["stack"] == "Base5"].iloc[0]
        delta = {
            "period": "confirmation",
            "rank_ic_delta": float(b3["rank_ic"] - b5["rank_ic"]),
            "icir_delta": float(b3["icir"] - b5["icir"]),
            "hl_sharpe_delta": float(b3["hl_sharpe"] - b5["hl_sharpe"]),
            "hl_max_drawdown_delta": float(b3["hl_max_drawdown"] - b5["hl_max_drawdown"]),
        }
        log(
            f"\nConfirmation Base3−Base5: ΔIC={delta['rank_ic_delta']:+.4f} "
            f"ΔSharpe={delta['hl_sharpe_delta']:+.3f} ΔMDD={delta['hl_max_drawdown_delta']:+.4f}"
        )
        (OUT / "base3_vs_base5_delta.json").write_text(json.dumps(delta, indent=2) + "\n")

    log("\n=== Loading enhancers ===")
    enhancers = load_enhancer_panels(ctx, cfg.START_DAY, cfg.END_DAY)
    gc.collect()

    enhancer_grids = []
    enhancer_verdicts = {}
    for name, _src in ENHANCER_SPECS:
        panel = enhancers.get(name)
        if panel is None:
            enhancer_verdicts[name] = {"retain": False, "recommendation": "missing_panel"}
            continue
        log(f"\n--- Enhancer λ-grid on Base3: {name} ---")
        for period, ret_w in [("discovery", ret_disc), ("confirmation", ret_conf)]:
            if len(ret_w) < 60:
                continue
            grid = evaluate_enhancer_on_base3(
                base3.reindex(index=ret_w.index, columns=ret_w.columns),
                name,
                panel.reindex(index=ret_w.index, columns=ret_w.columns),
                ret_w,
                args.lambdas,
                period,
            )
            enhancer_grids.append(grid)
            if period == "confirmation":
                verdict = classify_enhancer_uplift(grid)
                enhancer_verdicts[name] = verdict
                log(
                    grid[["lambda", "hl_sharpe", "rank_ic", "hl_sharpe_delta"]].to_string(
                        index=False, float_format=lambda x: f"{x:.4f}"
                    )
                )
                log(f"  -> {verdict['recommendation']}")

    if enhancer_grids:
        enh_df = pd.concat(enhancer_grids, ignore_index=True)
        enh_df.to_csv(OUT / "base3_enhancer_lambda_grid.csv", index=False)
    else:
        enh_df = pd.DataFrame()

    (OUT / "base3_enhancer_verdicts.json").write_text(
        json.dumps(enhancer_verdicts, indent=2, default=str) + "\n"
    )

    pool = json.loads(Path("research/frozen_candidate_pool_v1.json").read_text())
    library = build_library_document(compare_df, enhancer_verdicts, pool)
    library_path = LIBRARY_OUT / "alpha_library_v1.0-frozen.json"
    library_path.write_text(json.dumps(library, indent=2, ensure_ascii=False, default=str) + "\n")

    summary = {
        "task": "base3_library_v1_freeze",
        "tag": "v1.0-frozen",
        "base3_factors": BASE3_FACTORS,
        "enhancer_verdicts": enhancer_verdicts,
        "artifacts": {
            "metrics": str(OUT / "base3_vs_base5_metrics.csv"),
            "enhancer_grid": str(OUT / "base3_enhancer_lambda_grid.csv"),
            "library": str(library_path),
            "pool": "research/frozen_candidate_pool_v1.json",
        },
    }
    (OUT / "base3_library_freeze_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    log(f"\nLibrary frozen -> {library_path}")
    log(f"Results -> {OUT}")


if __name__ == "__main__":
    main()
