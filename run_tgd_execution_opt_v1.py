#!/usr/bin/env python
"""TGD Execution Optimization v1 — maximize net alpha without changing TGD20.

Factor layer FROZEN. This script only changes signal → portfolio → trade.

  E0  Freeze baseline (raw + size+industry from Stage-4 confirmation)
  E1  Rebalance frequency: daily / Friday / 5d / 10d / 20d
  E2  Turnover control: buffer ranking + min holding
  E3  Weight method: EW / rank / zscore
  E4  Cost-aware summary → max Net Sharpe

Usage:
  OMP_NUM_THREADS=1 python run_tgd_execution_opt_v1.py
  OMP_NUM_THREADS=1 python run_tgd_execution_opt_v1.py --signal size_industry
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import DEFAULT_ROUND_TRIP_COST
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from execution_layer import evaluate_execution
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from liquidity_normalization import panel_cross_sectional_residual

OUT = Path("research/reports/tgd_v1/execution")
NEUT_CSV = Path("research/reports/tgd_v1/neutralization/neut_summary.csv")
FACTOR = "TGD20"
SIGNAL_SHIFT = 1
TOP_FRAC = 0.10  # 10-group H-L book (decile), not investability default 0.20


def log(msg: str) -> None:
    print(msg, flush=True)


def freeze_baseline(out: Path) -> dict:
    """E0: persist Stage-4 confirmation metrics as immutable baseline."""
    rows = pd.read_csv(NEUT_CSV) if NEUT_CSV.exists() else pd.DataFrame()
    by_mode = {r["mode"]: r.to_dict() for _, r in rows.iterrows()} if len(rows) else {}
    raw = by_mode.get("raw", {})
    si = by_mode.get("size_industry", {})
    payload = {
        "factor": FACTOR,
        "frozen": True,
        "note": "Factor definition frozen. Execution experiments must not alter TGD20.",
        "signal_shift": SIGNAL_SHIFT,
        "portfolio": "10-group H-L (top/bottom 10%)",
        "rebalance_baseline": "daily",
        "cost": f"{DEFAULT_ROUND_TRIP_COST:.4f} round-trip",
        "confirmation_window": "discovery_days split (Stage-4)",
        "baseline_raw": {
            "rank_ic": raw.get("rank_ic"),
            "icir": raw.get("icir"),
            "hl_sharpe": raw.get("hl_sharpe"),
            "daily_turnover_hl": raw.get("daily_turnover_hl"),
            "net_sharpe_15bp": raw.get("net_sharpe_15bp"),
            "implied_annu_fee": raw.get("implied_annu_fee"),
            "source": "Stage-4 groupTest+investability (raw)",
        },
        "baseline_size_industry": {
            "rank_ic": si.get("rank_ic"),
            "icir": si.get("icir"),
            "hl_sharpe": si.get("hl_sharpe"),
            "daily_turnover_hl": si.get("daily_turnover_hl"),
            "net_sharpe_15bp": si.get("net_sharpe_15bp"),
            "implied_annu_fee": si.get("implied_annu_fee"),
            "source": "Stage-4 groupTest+investability (size+industry)",
        },
        "execution_book": {
            "top_frac": TOP_FRAC,
            "note": "Execution LS uses top/bottom 10% to match decile H-L research book",
        },
    }
    (out / "baseline_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return payload


def build_signal_panel(
    mode: str,
    tgd: pd.DataFrame,
    industry: pd.DataFrame,
    float_mkt: pd.DataFrame,
) -> pd.DataFrame:
    raw = cs_zscore(tgd)
    if mode == "raw":
        return raw
    if mode == "size_industry":
        return cs_zscore(neutralize_size_industry(tgd, industry.reindex_like(tgd), float_mkt))
    if mode == "size":
        log_size = np.log(float_mkt.replace(0, np.nan)).reindex_like(tgd)
        return cs_zscore(panel_cross_sectional_residual(tgd, [log_size]))
    raise ValueError(mode)


def run_grid(signal: pd.DataFrame, ret: pd.DataFrame, *, signal_mode: str) -> dict:
    """E1–E3 experiment grids. Objective = max net_sharpe."""
    results = {"rebalance": [], "buffer": [], "holding": [], "weight": [], "combo": []}

    # --- E1 rebalance frequency ---
    log("\n--- E1 rebalance frequency ---")
    e1_specs = [
        ("daily", dict(rebalance_freq=1, friday_only=False)),
        ("weekly_friday", dict(rebalance_freq=1, friday_only=True)),
        ("every_5d", dict(rebalance_freq=5, friday_only=False)),
        ("every_10d", dict(rebalance_freq=10, friday_only=False)),
        ("every_20d", dict(rebalance_freq=20, friday_only=False)),
    ]
    for label, kw in e1_specs:
        row = evaluate_execution(
            signal,
            ret,
            label=f"{signal_mode}|{label}",
            stage="E1",
            top_frac=TOP_FRAC,
            weight_method="ew",
            **kw,
        )
        results["rebalance"].append(row)
        log(
            f"  {label}: gross={row['gross_sharpe']:.2f} net={row['net_sharpe']:.2f} "
            f"TO={row['daily_turnover']:.3f} ICIR={row['icir']:.2f}"
        )

    best_e1 = max(results["rebalance"], key=lambda r: (r["net_sharpe"] if pd.notna(r["net_sharpe"]) else -1e9))
    best_freq = best_e1.get("rebalance_freq", 1)
    best_friday = best_e1.get("rebalance_freq") == "friday"
    # map back kwargs
    if best_e1["label"].endswith("weekly_friday"):
        base_rb = dict(rebalance_freq=1, friday_only=True)
    else:
        freq = int(best_e1["rebalance_freq"]) if best_e1["rebalance_freq"] != "friday" else 1
        base_rb = dict(rebalance_freq=freq, friday_only=False)
    log(f"  E1 best by Net Sharpe: {best_e1['label']} → {best_e1['net_sharpe']:.2f}")

    # --- E2.1 buffer (on daily + on best E1) ---
    log("\n--- E2.1 buffer ranking ---")
    buffer_specs = [
        ("buffer_10_20", 0.10, 0.20),
        ("buffer_10_30", 0.10, 0.30),
        ("buffer_5_15", 0.05, 0.15),
    ]
    for rb_name, rb_kw in [("daily", dict(rebalance_freq=1, friday_only=False)), ("best_e1", base_rb)]:
        for blabel, entry, exit_ in buffer_specs:
            row = evaluate_execution(
                signal,
                ret,
                label=f"{signal_mode}|{rb_name}|{blabel}",
                stage="E2_buffer",
                entry_frac=entry,
                exit_frac=exit_,
                weight_method="ew",
                **rb_kw,
            )
            results["buffer"].append(row)
            log(
                f"  {rb_name}|{blabel}: gross={row['gross_sharpe']:.2f} "
                f"net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}"
            )

    # --- E2.2 min holding ---
    log("\n--- E2.2 min holding period ---")
    for hold in [1, 5, 10]:
        for rb_name, rb_kw in [("daily", dict(rebalance_freq=1, friday_only=False)), ("best_e1", base_rb)]:
            row = evaluate_execution(
                signal,
                ret,
                label=f"{signal_mode}|{rb_name}|hold_{hold}d",
                stage="E2_hold",
                top_frac=TOP_FRAC,
                min_hold=hold,
                weight_method="ew",
                **rb_kw,
            )
            results["holding"].append(row)
            log(
                f"  {rb_name}|hold_{hold}d: gross={row['gross_sharpe']:.2f} "
                f"net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}"
            )

    # --- E3 weights (on best_e1 plain) ---
    log("\n--- E3 weight methods ---")
    for method in ["ew", "rank", "zscore"]:
        row = evaluate_execution(
            signal,
            ret,
            label=f"{signal_mode}|best_e1|{method}",
            stage="E3",
            top_frac=TOP_FRAC,
            weight_method=method,
            **base_rb,
        )
        results["weight"].append(row)
        log(
            f"  {method}: gross={row['gross_sharpe']:.2f} net={row['net_sharpe']:.2f} "
            f"TO={row['daily_turnover']:.3f}"
        )

    # --- E4 combo: best_e1 + best buffer + ew ---
    log("\n--- E4 cost-aware combos ---")
    best_buf = max(results["buffer"], key=lambda r: (r["net_sharpe"] if pd.notna(r["net_sharpe"]) else -1e9))
    # also try best_e1 + buffer_10_20 + hold_5
    combo_specs = [
        ("best_e1_plain", dict(top_frac=TOP_FRAC, **base_rb)),
        (
            "best_e1_buffer_10_20",
            dict(entry_frac=0.10, exit_frac=0.20, **base_rb),
        ),
        (
            "best_e1_buffer_10_20_hold5",
            dict(entry_frac=0.10, exit_frac=0.20, min_hold=5, **base_rb),
        ),
        (
            "daily_buffer_10_20",
            dict(entry_frac=0.10, exit_frac=0.20, rebalance_freq=1, friday_only=False),
        ),
    ]
    for clabel, kw in combo_specs:
        row = evaluate_execution(
            signal,
            ret,
            label=f"{signal_mode}|{clabel}",
            stage="E4_combo",
            weight_method="ew",
            **kw,
        )
        results["combo"].append(row)
        log(
            f"  {clabel}: gross={row['gross_sharpe']:.2f} net={row['net_sharpe']:.2f} "
            f"TO={row['daily_turnover']:.3f}"
        )

    results["meta"] = {
        "signal_mode": signal_mode,
        "best_e1_label": best_e1["label"],
        "best_e1_net": best_e1["net_sharpe"],
        "best_buffer_label": best_buf["label"],
        "best_buffer_net": best_buf["net_sharpe"],
        "base_rb": {k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in base_rb.items()},
    }
    return results


def write_summary(out: Path, baseline: dict, all_rows: pd.DataFrame, meta: dict) -> None:
    base_raw = baseline.get("baseline_raw", {})
    base_si = baseline.get("baseline_size_industry", {})
    # pick best overall by net sharpe
    ranked = all_rows.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False)
    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    daily = all_rows[all_rows["label"].str.endswith("|daily") | all_rows["label"].str.contains(r"\|daily$")]
    daily_row = daily.iloc[0].to_dict() if len(daily) else {}

    lines = [
        "# TGD Execution Optimization v1",
        "",
        "**Factor layer:** TGD20 frozen — no window / Gu/Gd / residual changes.",
        "**Objective:** maximize **Net Sharpe** (15bp RT), not gross.",
        "",
        "## E0 Baseline (Stage-4 confirmation, frozen)",
        "",
        "| Mode | RankIC | ICIR | H-L Sharpe | Daily TO | Net@15bp |",
        "|------|--------|------|------------|----------|----------|",
        f"| raw | {base_raw.get('rank_ic', float('nan')):.4f} | {base_raw.get('icir', float('nan')):.2f} | "
        f"{base_raw.get('hl_sharpe', float('nan')):.2f} | {base_raw.get('daily_turnover_hl', float('nan')):.2f} | "
        f"{base_raw.get('net_sharpe_15bp', float('nan')):.2f} |",
        f"| size+industry | {base_si.get('rank_ic', float('nan')):.4f} | {base_si.get('icir', float('nan')):.2f} | "
        f"{base_si.get('hl_sharpe', float('nan')):.2f} | {base_si.get('daily_turnover_hl', float('nan')):.2f} | "
        f"{base_si.get('net_sharpe_15bp', float('nan')):.2f} |",
        "",
        f"Execution experiments use **`{meta.get('signal_mode')}`** signal + top/bottom **10%** LS book.",
        "",
        "## Leaderboard (by Net Sharpe)",
        "",
        "| Rank | Label | Stage | Gross | Net | Daily TO | ICIR |",
        "|------|-------|-------|------:|----:|---------:|-----:|",
    ]
    for i, (_, r) in enumerate(ranked.head(12).iterrows(), 1):
        lines.append(
            f"| {i} | `{r['label']}` | {r['stage']} | {r['gross_sharpe']:.2f} | "
            f"{r['net_sharpe']:.2f} | {r['daily_turnover']:.3f} | {r['icir']:.2f} |"
        )

    lines += [
        "",
        "## Recommended investable config",
        "",
        f"- **Best Net Sharpe:** `{best.get('label')}` → net **{best.get('net_sharpe', float('nan')):.2f}** "
        f"(gross {best.get('gross_sharpe', float('nan')):.2f}, TO {best.get('daily_turnover', float('nan')):.3f})",
        f"- E1 best rebalance: `{meta.get('best_e1_label')}`",
        f"- E2 best buffer: `{meta.get('best_buffer_label')}`",
        "",
        "## Artifacts",
        "",
        "- `baseline_metrics.json`",
        "- `rebalance_frequency.csv`",
        "- `buffer_test.csv`",
        "- `holding_period.csv`",
        "- `weight_method.csv`",
        "- `combo_test.csv`",
        "- `all_experiments.csv`",
        "",
        "## Principle",
        "",
        "Do not retune TGD20. Execution only: fewer / smarter trades to harvest the same alpha.",
        "",
    ]
    (out / "execution_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signal",
        choices=["raw", "size_industry", "size"],
        default="size_industry",
        help="Neutralized signal for execution grid (default: size_industry per E0)",
    )
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== TGD Execution Optimization v1 ===")
    log("Factor TGD20 FROZEN — optimizing execution only")

    baseline = freeze_baseline(OUT)
    log(f"E0 baseline frozen → {OUT / 'baseline_metrics.json'}")

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, _ = load_eod_enriched_tables(preheat, end)
    industry = load_citics_industry_panel(start, end)
    tgd, _ = build_tgd20_wide_from_eod_l2(
        start, end, open_=enriched.open, close=enriched.close, use_cache=True
    )
    tgd = tgd.loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret_full = ret_full.reindex(index=tgd.index, columns=tgd.columns)
    _, ret = split_discovery_confirmation(ret_full, args.discovery_days)
    if ret.empty:
        ret = ret_full
    tgd_c = tgd.reindex(index=ret.index, columns=ret.columns)
    float_mkt = enriched.float_mktcap.reindex_like(tgd_c)
    signal = build_signal_panel(args.signal, tgd_c, industry.reindex_like(tgd_c), float_mkt)
    log(f"Confirmation: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d) | signal={args.signal}")

    results = run_grid(signal, ret, signal_mode=args.signal)

    reb = pd.DataFrame(results["rebalance"])
    buf = pd.DataFrame(results["buffer"])
    hold = pd.DataFrame(results["holding"])
    wgt = pd.DataFrame(results["weight"])
    combo = pd.DataFrame(results["combo"])
    reb.to_csv(OUT / "rebalance_frequency.csv", index=False)
    buf.to_csv(OUT / "buffer_test.csv", index=False)
    hold.to_csv(OUT / "holding_period.csv", index=False)
    wgt.to_csv(OUT / "weight_method.csv", index=False)
    combo.to_csv(OUT / "combo_test.csv", index=False)

    all_rows = pd.concat([reb, buf, hold, wgt, combo], ignore_index=True)
    all_rows.to_csv(OUT / "all_experiments.csv", index=False)
    write_summary(OUT, baseline, all_rows, results["meta"])

    best = all_rows.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False).iloc[0]
    log(f"\nBest Net Sharpe: {best['label']} = {best['net_sharpe']:.2f} (TO={best['daily_turnover']:.3f})")
    log(f"Wrote {OUT / 'execution_summary.md'}")


if __name__ == "__main__":
    main()
