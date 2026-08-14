#!/usr/bin/env python
"""Milestone 2.2.1 — IdealReversal OS gap-fill (Research Track Expansion).

Fill execution layer for IdealReversal Research Pack.
Does NOT retune formula. Does NOT auto-promote Registry status past testing.
Does NOT run portfolio / composite optimization.

Outputs:
  research/reports/ideal_reversal_v1/execution/
  research/reports/factors/IdealReversal/execution/
  Registry metric_note update (NetSharpe/TO from best execution)

Usage:
  OMP_NUM_THREADS=1 python run_milestone_2_2_1_ideal_reversal.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import DEFAULT_ROUND_TRIP_COST
from factor_attribution import cs_zscore
from factor_cutting.ideal_reversal import compute_ideal_reversal
from factor_cutting.trade_count import load_trade_count_daily
from factor_data_loaders import load_eod_enriched_tables
from run_milestone_1d7_pack_completion import run_d1_execution_grid

REPO = Path(__file__).resolve().parent
PACK = REPO / "research/reports/factors/IdealReversal"
EXEC_OUT = REPO / "research/reports/ideal_reversal_v1/execution"
CACHE = REPO / "research/cache/ideal_reversal_panels"
TOP_FRAC = 0.10
SIGNAL_SHIFT = 1


def log(msg: str) -> None:
    print(msg, flush=True)


def update_registry(best: dict) -> None:
    reg_path = REPO / "research/registry/factor_registry.yaml"
    data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    for row in data["factors"]:
        if row["factor_id"] != "IdealReversal":
            continue
        row["NetSharpe"] = float(best["net_sharpe"]) if pd.notna(best.get("net_sharpe")) else None
        row["Turnover"] = float(best["daily_turnover"]) if pd.notna(best.get("daily_turnover")) else None
        row["metric_note"] = (
            f"Cutting harvest RankICIR≈-8.6 · mono≈0.44 (below soft bar); "
            f"execution best {best.get('label')} NetSharpe={best.get('net_sharpe'):.2f} "
            f"TO={best.get('daily_turnover'):.3f} @15bp — stay testing"
        )
        row["notes"] = (
            "Paper replication / cutting track. Execution filled 2.2.1. "
            "Do not auto-admit; soft bar not passed."
        )
        row["validation_stage"] = "research_pack_complete"
        break
    reg_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # CSV mirror
    csv_path = REPO / "research/registry/factor_registry.csv"
    df = pd.read_csv(csv_path)
    m = df["factor_id"] == "IdealReversal"
    if m.any():
        df.loc[m, "NetSharpe"] = best.get("net_sharpe")
        df.loc[m, "Turnover"] = best.get("daily_turnover")
        df.loc[m, "metric_note"] = (
            f"exec best Net={best.get('net_sharpe'):.2f}; soft-bar fail; stay testing"
        )
        df.to_csv(csv_path, index=False)
    log("  Registry IdealReversal metrics updated (status remains testing)")


def update_report_content(best: dict) -> None:
    path = REPO / "factor_specs/IdealReversal_report_content.yaml"
    text = path.read_text(encoding="utf-8")
    # Replace execution narrative block if present
    new_exec = (
        "execution_narrative: |\n"
        f"  Execution grid filled (Milestone 2.2.1). Best: `{best.get('label')}` "
        f"Net Sharpe≈{best.get('net_sharpe'):.2f}, daily TO≈{best.get('daily_turnover'):.3f} "
        f"@ {DEFAULT_ROUND_TRIP_COST} RT. Soft bar still not passed — Registry stays `testing`.\n"
    )
    if "execution_narrative:" in text:
        import re

        text = re.sub(
            r"execution_narrative: \|.*?(?=\n[a-z_]+:|\Z)",
            new_exec.rstrip() + "\n",
            text,
            count=1,
            flags=re.S,
        )
    path.write_text(text, encoding="utf-8")
    log(f"  updated {path.name} execution_narrative")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-days",
        type=int,
        default=DISCOVERY_DAYS,
        help="Use confirmation window after discovery (default 504).",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=252,
        help="If set (>0), use only last N trading days for compute+exec (default 252; 0=confirmation).",
    )
    parser.add_argument("--full-sample", action="store_true", help="Skip discovery split")
    args = parser.parse_args()

    log("=== Milestone 2.2.1 IdealReversal OS gap-fill ===")
    log("Research Track Expansion — no portfolio / no formula retune")

    EXEC_OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    close_full = enriched.close.loc[start:end]
    amount_full = enriched.amount.loc[start:end]
    volume_full = enriched.volume.loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    # Restrict compute window (numba often unavailable → keep panels short)
    warm = 40
    idx = close_full.index
    if args.full_sample:
        close, amount, volume = close_full, amount_full, volume_full
        window_note = "full_sample"
        eval_index = close.index
    elif args.eval_days and args.eval_days > 0:
        n = min(args.eval_days, len(idx))
        eval_index = idx[-n:]
        slice_start = idx[max(0, len(idx) - n - warm)]
        close = close_full.loc[slice_start:end]
        amount = amount_full.loc[slice_start:end]
        volume = volume_full.loc[slice_start:end]
        window_note = f"last_{n}d"
    else:
        _, close_conf = split_discovery_confirmation(close_full, args.discovery_days)
        eval_index = close_conf.index
        conf_start = eval_index[0]
        pos = idx.get_loc(conf_start)
        slice_start = idx[max(0, pos - warm)]
        close = close_full.loc[slice_start:end]
        amount = amount_full.loc[slice_start:end]
        volume = volume_full.loc[slice_start:end]
        window_note = f"confirmation_after_{args.discovery_days}"

    ret_1d = close / close.shift(1) - 1.0
    cache_tag = f"{close.index[0].date()}_{close.index[-1].date()}"
    cache_path = CACHE / f"ideal_reversal_{cache_tag}.pkl"
    meta_path = CACHE / f"ideal_reversal_{cache_tag}_meta.json"

    if cache_path.exists():
        log(f"Load cached panel {cache_path}")
        fac = pd.read_pickle(cache_path)
        src = json.loads(meta_path.read_text()).get("knife_source", "cached")
    else:
        log("Load trade_count (ATS knife) ...")
        try:
            trade_count = load_trade_count_daily(close.index[0] - dt.timedelta(days=30), end)
            trade_count = trade_count.reindex_like(close)
            knife = "trade_count"
        except Exception as exc:
            log(f"  WARN trade_count unavailable ({exc}); using amount/volume proxy")
            trade_count = None
            knife = "amount_per_volume_proxy"

        log(f"Compute IdealReversal on {len(close)}d (slow without numba) ...")
        fac, _, _, src = compute_ideal_reversal(
            ret_1d,
            amount,
            trade_count=trade_count,
            volume=volume,
            return_legs=True,
        )
        fac.to_pickle(cache_path)
        meta_path.write_text(json.dumps({"knife_source": src, "requested": knife}) + "\n")
        log(f"  cached → {cache_path}")

    log(f"  knife_source={src}")
    fac = fac.reindex(index=eval_index)
    ret = ret_full.reindex(index=eval_index, columns=fac.columns)
    signal = cs_zscore(fac)
    ret = ret.reindex(index=signal.index, columns=signal.columns)

    log(f"  panel {signal.index[0].date()}→{signal.index[-1].date()} ({len(signal)}d) [{window_note}]")
    # Reuse D1-shaped execution grid helper (same E1–E4 stages)
    all_rows = run_d1_execution_grid(signal, ret, signal_mode="raw_cs_z")
    ranked = all_rows.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False)
    ranked.to_csv(EXEC_OUT / "all_experiments.csv", index=False)
    ranked.to_csv(EXEC_OUT / "execution_summary.csv", index=False)

    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    baseline = {
        "factor": "IdealReversal",
        "knife_source": src,
        "frozen_formula": False,
        "signal_mode": "raw_cs_z",
        "signal_shift": SIGNAL_SHIFT,
        "cost_round_trip": DEFAULT_ROUND_TRIP_COST,
        "best_label": best.get("label"),
        "best_net_sharpe": best.get("net_sharpe"),
        "best_gross_sharpe": best.get("gross_sharpe"),
        "best_daily_turnover": best.get("daily_turnover"),
        "soft_bar": "fail_stay_testing",
        "eval_window": window_note,
        "n_days": int(len(signal)),
        "note": "Milestone 2.2.1 execution gap-fill — formula not retuned; no Registry promote",
    }
    (EXEC_OUT / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2, default=str) + "\n", encoding="utf-8"
    )

    (PACK / "execution").mkdir(parents=True, exist_ok=True)
    (PACK / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXEC_OUT / "execution_summary.csv", PACK / "execution_summary.csv")
    shutil.copy2(EXEC_OUT / "execution_summary.csv", PACK / "execution" / "execution_summary.csv")
    shutil.copy2(EXEC_OUT / "baseline_metrics.json", PACK / "artifacts" / "execution_baseline_2_2_1.json")

    summary_path = PACK / "factor_summary.csv"
    if summary_path.exists() and len(ranked):
        df = pd.read_csv(summary_path)
        df = df[df["mode"] != "execution_best"]
        b = ranked.iloc[0]
        row = {
            "factor": "IdealReversal",
            "period": f"full_{len(signal)}d",
            "universe": "ALL",
            "mode": "execution_best",
            "rank_ic": b.get("rank_ic"),
            "annu_ic": (float(b["rank_ic"]) * np.sqrt(250) if pd.notna(b.get("rank_ic")) else np.nan),
            "icir": b.get("icir"),
            "hl_annu_ret": b.get("gross_annu_ret"),
            "hl_sharpe": b.get("gross_sharpe"),
            "hl_mdd": b.get("mdd_net"),
            "daily_turnover": b.get("daily_turnover"),
            "implied_annu_fee": b.get("implied_annu_fee"),
            "net_sharpe": b.get("net_sharpe"),
            "monotonicity": np.nan,
            "direction": b.get("direction", 1),
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(summary_path, index=False)
        shutil.copy2(summary_path, PACK / "artifacts" / "factor_summary.csv")

    update_registry(best)
    update_report_content(best)

    log(
        f"  IdealReversal best: {best.get('label')} "
        f"net={best.get('net_sharpe')} TO={best.get('daily_turnover')}"
    )
    log(f"  installed → {PACK / 'execution' / 'execution_summary.csv'}")
    log("=== 2.2.1 complete (status remains testing) ===")


if __name__ == "__main__":
    main()
