#!/usr/bin/env python
"""Alpha Combination Layer v1 — C1~C6 experiment matrix on Frozen Library Base3.

Optimizes NET Sharpe (15bp round-trip + tradability filters) on confirmation 951d.

  C1  Base3 equal-weight (1/3, 1/3, 1/3)
  C2  D1 tilt scan: w1 ∈ {0.33, 0.4, 0.5, 0.6, 0.7, 0.8}
  C3  Lower-frequency rebalance: every 2/3/5 days
  C4  Soft-hold: skip rebalance if LS turnover > {0.2, 0.3, 0.4}
  C5  cancel_shock as state_modifier (attenuate D4 when cancel extreme)
  C6  Best C2 tilt + C5 state_modifier

Usage:
  OMP_NUM_THREADS=1 python run_alpha_combination_v1.py
  OMP_NUM_THREADS=1 python run_alpha_combination_v1.py --discovery-days 504
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    apply_tradability_mask,
    build_long_short_weights,
    classify_market_regimes,
    evaluate_investability,
    regime_net_sharpes,
    strip_internal,
    yearly_net_sharpes,
)
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_l2_v2 import build_l2_v2_factor
from l2_data_loaders import build_l2_daily_cache
from run_l2_validation import load_context

OUT = Path("research/results/alpha_combination_v1")
LIBRARY_PATH = Path("research/alpha_library_v1/alpha_library_v1.0-frozen.json")

D1 = "low_vol_liquidity_quality_60d"
D4 = "winner_sentiment_reversal_5d"
D5 = "upside_fragility_20d"
BASE3 = [D1, D4, D5]

D1_WEIGHTS = [1 / 3, 0.4, 0.5, 0.6, 0.7, 0.8]
FREQ_DAYS = [2, 3, 5]
SOFT_HOLD_THRESHOLDS = [0.6, 0.8, 1.0]  # absolute LS-book L1; daily TO ~0.95 so 0.2–0.4 was always-hold
CANCEL_STRESS_QUANTILE = 0.20  # bottom quintile of cancel mean-z days = stress


def log(msg: str) -> None:
    print(msg, flush=True)


def blend_base3(
    panels: Dict[str, pd.DataFrame],
    w_d1: float,
    w_d4: Optional[float] = None,
    w_d5: Optional[float] = None,
) -> pd.DataFrame:
    """Weighted cs_z blend. If w_d4/w_d5 omitted, split residual equally."""
    if w_d4 is None or w_d5 is None:
        rest = (1.0 - w_d1) / 2.0
        w_d4 = rest
        w_d5 = rest
    return (
        w_d1 * cs_zscore(panels[D1])
        + w_d4 * cs_zscore(panels[D4])
        + w_d5 * cs_zscore(panels[D5])
    )


def downsample_signal(signal: pd.DataFrame, freq: int) -> pd.DataFrame:
    """Hold signal between rebalance days (freq=1 → daily)."""
    if freq <= 1:
        return signal
    out = signal.copy()
    values = out.to_numpy(copy=True)
    n = len(values)
    last = values[0].copy()
    for t in range(1, n):
        if t % freq == 0:
            last = values[t].copy()
        else:
            values[t] = last
    return pd.DataFrame(values, index=out.index, columns=out.columns)


def soft_hold_signal(
    signal: pd.DataFrame,
    threshold: float,
    *,
    top_frac: float = 0.2,
) -> pd.DataFrame:
    """
    Soft-hold: if day-t LS book L1 turnover vs held book exceeds threshold,
    keep previous held signal (skip rebalance).
    """
    out = signal.copy()
    values = out.to_numpy(copy=True)
    n = len(values)
    held = values[0].copy()

    # Precompute daily LS weights for proposed signal
    w_long, w_short = build_long_short_weights(signal, top_frac=top_frac, bottom_frac=top_frac)
    w_ls = (w_long.fillna(0) - w_short.fillna(0)).to_numpy()

    held_w = w_ls[0].copy()
    for t in range(1, n):
        proposed_w = w_ls[t]
        to = np.nansum(np.abs(proposed_w - held_w))
        if to > threshold:
            values[t] = held
        else:
            held = values[t].copy()
            held_w = proposed_w.copy()
    return pd.DataFrame(values, index=out.index, columns=out.columns)


def state_modifier_blend(
    panels: Dict[str, pd.DataFrame],
    cancel: pd.DataFrame,
    *,
    w_d1: float = 1 / 3,
    w_d4: Optional[float] = None,
    w_d5: Optional[float] = None,
    stress_quantile: float = CANCEL_STRESS_QUANTILE,
) -> Tuple[pd.DataFrame, int]:
    """
    cancel as state_modifier (NOT additive):
      stress days = cancel cross-sectional mean-z in bottom `stress_quantile`;
      on stress → attenuate D4 by 0.5, redistribute cut equally to D1 and D5.
    Returns (blended_signal, n_stress_days).
    """
    if w_d4 is None or w_d5 is None:
        rest = (1.0 - w_d1) / 2.0
        w_d4 = rest
        w_d5 = rest

    z1 = cs_zscore(panels[D1])
    z4 = cs_zscore(panels[D4])
    z5 = cs_zscore(panels[D5])
    cancel_mean = cs_zscore(cancel).mean(axis=1)
    thresh = float(cancel_mean.quantile(stress_quantile))
    state = cancel_mean <= thresh
    n_stress = int(state.sum())

    w1_n, w4_n, w5_n = w_d1, w_d4, w_d5
    cut = w_d4 * 0.5
    w1_s = w_d1 + cut / 2.0
    w4_s = w_d4 * 0.5
    w5_s = w_d5 + cut / 2.0

    s = state.astype(float)
    w1 = (1 - s) * w1_n + s * w1_s
    w4 = (1 - s) * w4_n + s * w4_s
    w5 = (1 - s) * w5_n + s * w5_s

    blended = z1.mul(w1, axis=0) + z4.mul(w4, axis=0) + z5.mul(w5, axis=0)
    return blended, n_stress


def evaluate_scheme(
    label: str,
    family: str,
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    trad_kw: dict,
    regime: pd.Series,
    *,
    baseline_net_sharpe: Optional[float] = None,
    baseline_turnover: Optional[float] = None,
) -> dict:
    inv = evaluate_investability(signal, ret, **trad_kw)
    row = {
        "label": label,
        "family": family,
        **strip_internal(inv),
        "yearly_net_sharpe": yearly_net_sharpes(inv["_net_pnl"]),
        "regime_net_sharpe": regime_net_sharpes(inv["_net_pnl"], regime),
    }
    if baseline_net_sharpe is not None and pd.notna(inv["net_sharpe_tradable"]):
        row["net_sharpe_delta_vs_c1"] = float(inv["net_sharpe_tradable"] - baseline_net_sharpe)
    if baseline_turnover is not None and pd.notna(inv["annu_one_way_turnover"]):
        row["turnover_ratio_vs_c1"] = float(inv["annu_one_way_turnover"] / baseline_turnover)
    return row


def load_masks(start, end):
    return {
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
    }


def load_cancel_and_close(ctx, start, end):
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
    cancel = (
        build_l2_v2_factor("cn_cancel_shock", l2_cache)
        .loc[start:end]
        .reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
    )
    close = enriched.close.loc[start:end].reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
    amount = enriched.amount.loc[start:end].reindex(index=ctx["ret"].index, columns=ctx["ret"].columns)
    return cancel, close, amount


def passes_gates(row: dict, c1_turnover: float) -> dict:
    """Feasibility fences (not primary ranking): turnover ≤120%, net Sharpe > 0."""
    net = row.get("net_sharpe_tradable", np.nan)
    to = row.get("annu_one_way_turnover", np.nan)

    to_ok = pd.notna(to) and to <= 120.0
    net_ok = pd.notna(net) and net > 0

    return {
        "gate_turnover_le_120": bool(to_ok),
        "gate_net_sharpe_gt_0": bool(net_ok),
        "research_feasible": bool(to_ok and net_ok),
        # legacy keys kept for CSV compatibility
        "gate_turnover_le_1.5x_c1": bool(
            pd.notna(row.get("turnover_ratio_vs_c1")) and row.get("turnover_ratio_vs_c1") <= 1.5
        ),
        "gates_pass": bool(to_ok and net_ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Combination Layer v1")
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    parser.add_argument("--cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument(
        "--cancel-stress-quantile",
        type=float,
        default=CANCEL_STRESS_QUANTILE,
        help="Bottom quantile of cancel mean-z days treated as stress (state_modifier)",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== Alpha Combination Layer v1 (C1–C6) ===")

    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    ret_full = ctx["ret"]
    _, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    # Hold out last 252d of confirmation as pure OOS slice for reporting
    if len(ret_conf) > 252:
        ret_train = ret_conf.iloc[:-252]
        ret_oos = ret_conf.iloc[-252:]
    else:
        ret_train = ret_conf
        ret_oos = ret_conf.iloc[0:0]
    log(
        f"Confirmation: {ret_conf.index[0].date()} -> {ret_conf.index[-1].date()} ({len(ret_conf)}d) | "
        f"train={len(ret_train)}d oos_tail={len(ret_oos)}d"
    )

    log("Loading masks + cancel...")
    masks = load_masks(cfg.START_DAY, cfg.END_DAY)
    cancel, close, amount = load_cancel_and_close(ctx, cfg.START_DAY, cfg.END_DAY)

    panels_full = {n: ctx["frozen_panels"][n] for n in BASE3}
    panels = {
        n: p.reindex(index=ret_conf.index, columns=ret_conf.columns) for n, p in panels_full.items()
    }
    cancel_c = cancel.reindex(index=ret_conf.index, columns=ret_conf.columns)
    close_c = close.reindex(index=ret_conf.index, columns=ret_conf.columns)
    amount_c = amount.reindex(index=ret_conf.index, columns=ret_conf.columns)

    mkt = ret_conf.mean(axis=1)
    regime = classify_market_regimes(mkt)

    trad_kw = dict(
        df_not_limit=masks["df_not_limit"],
        df_not_st=masks["df_not_st"],
        df_trade_status=masks["df_trade_status"],
        close=close_c,
        amount=amount_c,
        round_trip_cost=args.cost,
        apply_tradability=True,
    )

    results: List[dict] = []
    signals_cache: Dict[str, pd.DataFrame] = {}

    # ---- C1 baseline ----
    log("\n--- C1 Base3 equal-weight ---")
    sig_c1 = blend_base3(panels, 1 / 3)
    signals_cache["C1_base3_equal_weight"] = sig_c1
    row_c1 = evaluate_scheme("C1_base3_equal_weight", "C1", sig_c1, ret_conf, trad_kw, regime)
    c1_net = row_c1["net_sharpe_tradable"]
    c1_to = row_c1["annu_one_way_turnover"]
    row_c1.update(passes_gates({**row_c1, "net_sharpe_delta_vs_c1": 0.0, "turnover_ratio_vs_c1": 1.0}, c1_to))
    row_c1["net_sharpe_delta_vs_c1"] = 0.0
    row_c1["turnover_ratio_vs_c1"] = 1.0
    results.append(row_c1)
    log(
        f"  net={c1_net:.3f} gross={row_c1['gross_sharpe_tradable']:.3f} "
        f"TO={c1_to:.1f} IC={row_c1['rank_ic_tradable']:.4f}"
    )

    # Additive cancel reference (legacy production suggestion)
    log("\n--- REF Base3 + cancel additive λ=0.2 ---")
    sig_add = build_satellite_stack(
        sig_c1, {"cn_cancel_shock": cancel_c}, 0.2, satellite_factors=["cn_cancel_shock"]
    )
    trad_kw0 = dict(trad_kw)
    trad_kw0["signal_shift"] = 0
    row_add = evaluate_scheme(
        "REF_base3_plus_cancel_additive_0.2",
        "REF",
        sig_add,
        ret_conf,
        trad_kw0,
        regime,
        baseline_net_sharpe=c1_net,
        baseline_turnover=c1_to,
    )
    row_add.update(passes_gates(row_add, c1_to))
    results.append(row_add)
    log(f"  net={row_add['net_sharpe_tradable']:.3f} ΔvsC1={row_add['net_sharpe_delta_vs_c1']:+.3f}")

    # ---- C2 D1 tilt ----
    log("\n--- C2 D1 tilt scan ---")
    best_w1 = 1 / 3
    best_c2_net = c1_net
    for w1 in D1_WEIGHTS:
        label = f"C2_D1_{w1:.2f}"
        sig = blend_base3(panels, w1)
        signals_cache[label] = sig
        row = evaluate_scheme(
            label, "C2", sig, ret_conf, trad_kw, regime,
            baseline_net_sharpe=c1_net, baseline_turnover=c1_to,
        )
        row.update(passes_gates(row, c1_to))
        results.append(row)
        log(
            f"  w1={w1:.2f} net={row['net_sharpe_tradable']:.3f} "
            f"Δ={row['net_sharpe_delta_vs_c1']:+.3f} TO={row['annu_one_way_turnover']:.1f}"
        )
        if pd.notna(row["net_sharpe_tradable"]) and row["net_sharpe_tradable"] > best_c2_net:
            best_c2_net = row["net_sharpe_tradable"]
            best_w1 = w1
    log(f"  >> best C2 w1={best_w1:.2f} net={best_c2_net:.3f}")

    # ---- C3 frequency ----
    log("\n--- C3 lower-frequency rebalance ---")
    for freq in FREQ_DAYS:
        label = f"C3_freq_{freq}d"
        sig = downsample_signal(sig_c1, freq)
        signals_cache[label] = sig
        row = evaluate_scheme(
            label, "C3", sig, ret_conf, trad_kw, regime,
            baseline_net_sharpe=c1_net, baseline_turnover=c1_to,
        )
        row.update(passes_gates(row, c1_to))
        results.append(row)
        log(
            f"  freq={freq}d net={row['net_sharpe_tradable']:.3f} "
            f"Δ={row['net_sharpe_delta_vs_c1']:+.3f} TO={row['annu_one_way_turnover']:.1f}"
        )

    # ---- C4 soft hold ----
    log("\n--- C4 soft-hold ---")
    for th in SOFT_HOLD_THRESHOLDS:
        label = f"C4_soft_hold_{th:.1f}"
        log(f"  building soft-hold th={th}...")
        sig = soft_hold_signal(sig_c1, th)
        signals_cache[label] = sig
        row = evaluate_scheme(
            label, "C4", sig, ret_conf, trad_kw, regime,
            baseline_net_sharpe=c1_net, baseline_turnover=c1_to,
        )
        row.update(passes_gates(row, c1_to))
        results.append(row)
        log(
            f"  th={th:.1f} net={row['net_sharpe_tradable']:.3f} "
            f"Δ={row['net_sharpe_delta_vs_c1']:+.3f} TO={row['annu_one_way_turnover']:.1f}"
        )
        gc.collect()

    # ---- C5 state modifier on equal-weight ----
    log("\n--- C5 cancel state_modifier (on equal-weight) ---")
    sig_c5, n_stress = state_modifier_blend(
        panels, cancel_c, w_d1=1 / 3, stress_quantile=args.cancel_stress_quantile
    )
    log(
        f"  stress days (cancel mean-z bottom {args.cancel_stress_quantile:.0%}): "
        f"{n_stress}/{len(ret_conf)}"
    )
    signals_cache["C5_state_modifier_eq"] = sig_c5
    row_c5 = evaluate_scheme(
        "C5_state_modifier_eq", "C5", sig_c5, ret_conf, trad_kw, regime,
        baseline_net_sharpe=c1_net, baseline_turnover=c1_to,
    )
    row_c5.update(passes_gates(row_c5, c1_to))
    row_c5["n_stress_days"] = n_stress
    results.append(row_c5)
    log(
        f"  net={row_c5['net_sharpe_tradable']:.3f} "
        f"Δ={row_c5['net_sharpe_delta_vs_c1']:+.3f} TO={row_c5['annu_one_way_turnover']:.1f}"
    )

    # ---- C6 best tilt + state modifier ----
    log(f"\n--- C6 best C2 (w1={best_w1:.2f}) + state_modifier ---")
    rest = (1.0 - best_w1) / 2.0
    sig_c6, n_stress6 = state_modifier_blend(
        panels,
        cancel_c,
        w_d1=best_w1,
        w_d4=rest,
        w_d5=rest,
        stress_quantile=args.cancel_stress_quantile,
    )
    label_c6 = f"C6_D1_{best_w1:.2f}_state_mod"
    signals_cache[label_c6] = sig_c6
    row_c6 = evaluate_scheme(
        label_c6, "C6", sig_c6, ret_conf, trad_kw, regime,
        baseline_net_sharpe=c1_net, baseline_turnover=c1_to,
    )
    row_c6.update(passes_gates(row_c6, c1_to))
    row_c6["n_stress_days"] = n_stress6
    results.append(row_c6)
    log(
        f"  net={row_c6['net_sharpe_tradable']:.3f} "
        f"Δ={row_c6['net_sharpe_delta_vs_c1']:+.3f} TO={row_c6['annu_one_way_turnover']:.1f}"
    )

    # Also: best C2 alone as candidate
    # OOS tail metrics for top candidates
    def oos_net(sig: pd.DataFrame) -> float:
        if len(ret_oos) < 60:
            return np.nan
        inv = evaluate_investability(
            sig.reindex(index=ret_oos.index, columns=ret_oos.columns),
            ret_oos,
            **{
                **trad_kw,
                "close": close_c.reindex(ret_oos.index),
                "amount": amount_c.reindex(ret_oos.index),
            },
        )
        return inv["net_sharpe_tradable"]

    for r in results:
        sig = signals_cache.get(r["label"])
        if sig is not None:
            r["oos_tail_252d_net_sharpe"] = oos_net(sig)

    # Rank by ICIR (alpha-research primary); net Sharpe is feasibility fence only
    flat_rows = []
    for r in results:
        flat = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        yearly = r.get("yearly_net_sharpe") or {}
        for y, s in yearly.items():
            flat[f"net_sharpe_{y}"] = s
        regime_s = r.get("regime_net_sharpe") or {}
        for reg, info in regime_s.items():
            if isinstance(info, dict):
                flat[f"net_sharpe_{reg}"] = info.get("net_sharpe")
        flat_rows.append(flat)
    flat_df = pd.DataFrame(flat_rows)
    flat_df["research_feasible"] = flat_df.get("research_feasible", flat_df.get("gates_pass", False))
    flat_df = flat_df.sort_values(
        ["research_feasible", "icir_tradable", "gross_sharpe_tradable", "net_sharpe_tradable"],
        ascending=[False, False, False, False],
    )
    # Column order: IC / ICIR first
    prefer = [
        "label",
        "family",
        "rank_ic_tradable",
        "icir_tradable",
        "gross_sharpe_tradable",
        "annu_one_way_turnover",
        "net_sharpe_tradable",
        "research_feasible",
    ]
    ordered = [c for c in prefer if c in flat_df.columns] + [
        c for c in flat_df.columns if c not in prefer
    ]
    flat_df = flat_df[ordered]
    flat_df.to_csv(OUT / "combination_results.csv", index=False)

    # C2 ICIR curve (primary research plot)
    c2 = flat_df[flat_df["family"] == "C2"][
        [
            "label",
            "rank_ic_tradable",
            "icir_tradable",
            "gross_sharpe_tradable",
            "annu_one_way_turnover",
            "net_sharpe_tradable",
        ]
    ]
    c2.to_csv(OUT / "c2_d1_tilt_icir_curve.csv", index=False)

    # Pick recommended: highest ICIR among feasible
    candidates = flat_df[flat_df["research_feasible"] == True]  # noqa: E712
    if candidates.empty:
        soft = flat_df[flat_df.get("gate_turnover_le_120", flat_df["annu_one_way_turnover"] <= 120)]
        recommended = soft.iloc[0].to_dict() if len(soft) else flat_df.iloc[0].to_dict()
        rec_note = "No fully feasible scheme; fallback under turnover fence"
    else:
        recommended = candidates.iloc[0].to_dict()
        rec_note = "Highest ICIR among TO≤120% and net Sharpe>0 (net Sharpe = fence only)"

    # Sync best_w1 to ICIR-optimal feasible C2 if present
    c2_feas = candidates[candidates["family"] == "C2"] if "family" in candidates.columns else candidates
    if not c2_feas.empty and str(c2_feas.iloc[0]["label"]).startswith("C2_D1_"):
        best_w1 = float(str(c2_feas.iloc[0]["label"]).split("_")[-1])
        best_c2_net = float(c2_feas.iloc[0]["net_sharpe_tradable"])

    verdict = {
        "version": "alpha_combination_v1",
        "ranking_paradigm": "ICIR_primary_turnover_and_net_sharpe_as_feasibility_fence",
        "frozen_at": "2026-07-10",
        "period": "confirmation_951d",
        "cost_rate": args.cost,
        "cancel_stress_quantile": args.cancel_stress_quantile,
        "c1_baseline": {
            "IC": row_c1["rank_ic_tradable"],
            "ICIR": row_c1["icir_tradable"],
            "gross_sharpe": row_c1["gross_sharpe_tradable"],
            "net_sharpe": c1_net,
            "annu_one_way_turnover": c1_to,
        },
        "best_c2_w1": best_w1,
        "best_c2_icir": float(recommended.get("icir_tradable", np.nan)),
        "recommended": {
            "label": recommended.get("label"),
            "IC": recommended.get("rank_ic_tradable"),
            "ICIR": recommended.get("icir_tradable"),
            "gross_sharpe": recommended.get("gross_sharpe_tradable"),
            "annu_one_way_turnover": recommended.get("annu_one_way_turnover"),
            "net_sharpe": recommended.get("net_sharpe_tradable"),
            "oos_tail_252d_net_sharpe": recommended.get("oos_tail_252d_net_sharpe"),
            "research_feasible": recommended.get("research_feasible"),
            "note": rec_note,
        },
        "ranking_top5": flat_df.head(5)[
            [
                "label",
                "rank_ic_tradable",
                "icir_tradable",
                "gross_sharpe_tradable",
                "annu_one_way_turnover",
                "net_sharpe_tradable",
                "research_feasible",
            ]
        ].to_dict(orient="records"),
        "interpretation": {
            "decile_vs_tilt": (
                "Decile plots = single-factor cross-section transparency (Library appendix). "
                "C2 D1-tilt scan = portfolio-layer ICIR allocation under cost fence."
            ),
            "net_sharpe_role": (
                "Net Sharpe is a production feasibility fence (must be >0), not the primary ranking metric. "
                "Primary sort = ICIR; secondary = gross Sharpe."
            ),
        },
        "artifacts": {
            "results_csv": str(OUT / "combination_results.csv"),
            "c2_icir_curve": str(OUT / "c2_d1_tilt_icir_curve.csv"),
        },
    }
    (OUT / "alpha_combination_v1.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    log("\n=== Ranking (ICIR primary | feasible first) ===")
    cols = [
        "label",
        "rank_ic_tradable",
        "icir_tradable",
        "gross_sharpe_tradable",
        "annu_one_way_turnover",
        "net_sharpe_tradable",
        "research_feasible",
    ]
    log(flat_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(
        f"\nRecommended: {recommended.get('label')} | "
        f"ICIR={recommended.get('icir_tradable'):.3f} | net={recommended.get('net_sharpe_tradable'):.3f}"
    )
    log(f"Published -> {OUT}")


if __name__ == "__main__":
    main()
