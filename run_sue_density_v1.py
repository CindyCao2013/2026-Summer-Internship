#!/usr/bin/env python
"""P0 SUE density vs Base3 — earliest-known ann dates, hold vs decay, size+industry residual.

Hard requirements:
  1. Earliest known date among notice / express / income
  2. Event-hold AND daily-decay signal modes
  3. Industry + ln_mktcap neutralization before residual IC vs Base3
  4. Include unexpected_profit_notice_surprise_20d

Usage:
  OMP_NUM_THREADS=1 python run_sue_density_v1.py
  OMP_NUM_THREADS=1 python run_sue_density_v1.py --sample-days 504 --keep-cache
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
from alpha_d4_expansion_stack import daily_rank_ic_series, evaluate_stack_signal, icir_from_daily
from alpha_dimension_density import residual_ic_stats
from factor_attribution import align_signal, combine_equal_weight, cs_zscore, hl_sharpe_from_composite
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_sue import (
    SUE_FACTOR_LIST,
    build_sue_event_tables,
    build_sue_panels,
    neutralize_size_industry,
)
from industry_neutral import load_citics_industry_panel
from sue_data import load_sue_raw_bundle

OUT = Path("research/reports/sue_density_v1")
BASE3 = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]
STACK_LAMBDAS = [0.0, 0.1, 0.2, 0.3]


def log(msg: str) -> None:
    print(msg, flush=True)


def coverage_stats(panel: pd.DataFrame) -> dict:
    n = panel.notna().sum(axis=1)
    return {
        "mean_names": float(n.mean()) if len(n) else 0.0,
        "median_names": float(n.median()) if len(n) else 0.0,
        "pct_days_ge_50": float((n >= 50).mean()) if len(n) else 0.0,
        "pct_days_ge_200": float((n >= 200).mean()) if len(n) else 0.0,
    }


def turnover_proxy(panel: pd.DataFrame, n_groups: int = 10) -> float:
    """Approx one-way decile turnover (annualized from daily membership churn)."""
    sig = align_signal(panel, 1)
    turns = []
    prev = None
    for dt_ in sig.index:
        s = sig.loc[dt_].dropna()
        if len(s) < n_groups * 5:
            continue
        ranks = s.rank(pct=True)
        long = set(ranks[ranks >= 0.9].index)
        short = set(ranks[ranks <= 0.1].index)
        book = long | short
        if prev is not None and len(book | prev) > 0:
            churn = 1.0 - len(book & prev) / max(len(book | prev), 1)
            turns.append(churn)
        prev = book
    if not turns:
        return float("nan")
    return float(np.mean(turns) * 250)  # rough annualized one-way


def build_base3(pv_cache, start, end) -> dict:
    panels = {}
    for name in BASE3:
        panels[name] = build_eod_engine_factor(name, pv_cache).loc[start:end]
    return panels


def evaluate_factor(
    name: str,
    mode: str,
    panel_raw_cs: pd.DataFrame,
    panel_neut: pd.DataFrame,
    ret: pd.DataFrame,
    base3_list: list,
    base3_combo: pd.DataFrame,
) -> dict:
    panel = panel_neut.reindex(index=ret.index, columns=ret.columns)
    cov = coverage_stats(panel)
    ic_daily = daily_rank_ic_series(panel, ret)
    ic_mean = float(ic_daily.mean()) if len(ic_daily.dropna()) else np.nan
    icir = icir_from_daily(ic_daily)
    sharpe, ann, direction = hl_sharpe_from_composite(panel, ret)
    to = turnover_proxy(panel)

    # residual vs each Base3 leg and vs equal Base3
    resid = {"vs_base3_combo": residual_ic_stats(panel, ret, base3_combo)}
    for bname, bp in zip(BASE3, base3_list):
        resid[f"vs_{bname}"] = residual_ic_stats(panel, ret, bp)

    # stack λ grid on neutralized signal
    stack_rows = []
    b = cs_zscore(base3_combo)
    s = cs_zscore(panel)
    for lam in STACK_LAMBDAS:
        combo = (1.0 - lam) * b + lam * s if lam > 0 else b
        st = evaluate_stack_signal(combo, ret)
        stack_rows.append({"lambda": lam, **st})

    base_icir = stack_rows[0].get("icir", np.nan)
    best = max(stack_rows, key=lambda r: (r.get("icir") if pd.notna(r.get("icir")) else -1e9))
    uplift = (
        float(best["icir"] - base_icir)
        if pd.notna(best.get("icir")) and pd.notna(base_icir)
        else np.nan
    )

    resid_combo = resid["vs_base3_combo"]
    verdict = "drop"
    if (
        pd.notna(resid_combo.get("residual_ic_t"))
        and abs(resid_combo["residual_ic_t"]) >= 2.0
        and pd.notna(uplift)
        and uplift >= 0
        and (not pd.notna(to) or to <= 120)
    ):
        verdict = "enhancer_candidate"
    elif pd.notna(resid_combo.get("residual_ic_t")) and abs(resid_combo["residual_ic_t"]) >= 2.0:
        verdict = "independent_but_stack_weak"
    elif pd.notna(icir) and abs(icir) >= 0.5:
        verdict = "raw_signal_only"

    return {
        "factor": name,
        "mode": mode,
        "rank_ic": ic_mean,
        "icir": icir,
        "gross_hl_sharpe": sharpe,
        "hl_annu_ret": ann,
        "direction": direction,
        "turnover_ann_proxy": to,
        "coverage": cov,
        "residual": resid,
        "stack": stack_rows,
        "best_lambda": best.get("lambda"),
        "stack_icir_uplift": uplift,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-days", type=int, default=504)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log("=== P0 SUE Density vs Base3 ===")
    log("Loading EOD enriched + industry...")
    enriched, _ = load_eod_enriched_tables(preheat, end)
    industry = load_citics_industry_panel(start, end)
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    sample_start = max(0, len(ret_full) - args.sample_days)
    ret = ret_full.iloc[sample_start:]
    log(f"Ret sample: {ret.index[0].date()} -> {ret.index[-1].date()} ({len(ret)}d)")

    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    base3 = build_base3(pv_cache, start, end)
    base3_s = {k: v.reindex(index=ret.index, columns=ret.columns) for k, v in base3.items()}
    base3_list = [base3_s[k] for k in BASE3]
    base3_combo = combine_equal_weight([cs_zscore(p) for p in base3_list])

    cache_root = Path(args.cache_dir) if args.cache_dir else None
    # Keep cache during run if user asked; else temp cleared inside loader
    bundle = load_sue_raw_bundle(
        start,
        end,
        history_start=start - dt.timedelta(days=900),
        cache_root=cache_root,
        keep_cache=bool(args.keep_cache or cache_root),
    )
    event_tables = build_sue_event_tables(bundle)
    for k, ev in event_tables.items():
        log(f"  events {k}: {len(ev):,}")

    trade_index = enriched.close.loc[start:end].index
    columns = enriched.close.columns
    float_mkt = enriched.float_mktcap.reindex(index=trade_index, columns=columns)
    ind = industry.reindex(index=trade_index, columns=columns)

    rows = []
    for mode in ("hold", "decay"):
        log(f"--- mode={mode} ---")
        panels = build_sue_panels(
            event_tables, trade_index, columns, mode=mode, hold_days=20, half_life=5
        )
        for name in SUE_FACTOR_LIST:
            raw = panels[name].reindex(index=ret.index, columns=ret.columns)
            neut = neutralize_size_industry(raw, ind.reindex_like(raw), float_mkt.reindex_like(raw))
            # re-z after neutralization
            neut = cs_zscore(neut)
            m = evaluate_factor(name, mode, raw, neut, ret, base3_list, base3_combo)
            rows.append(m)
            rc = m["residual"]["vs_base3_combo"]
            log(
                f"  [{mode}] {name}: IC={m['rank_ic']:.4f} ICIR={m['icir']:.2f} "
                f"resid_t={rc.get('residual_ic_t', np.nan):.2f} "
                f"stack_uplift={m['stack_icir_uplift']:.3f} TO≈{m['turnover_ann_proxy']:.0f} "
                f"-> {m['verdict']}"
            )

    # flatten for CSV
    flat = []
    for r in rows:
        rc = r["residual"]["vs_base3_combo"]
        flat.append(
            {
                "factor": r["factor"],
                "mode": r["mode"],
                "rank_ic": r["rank_ic"],
                "icir": r["icir"],
                "gross_hl_sharpe": r["gross_hl_sharpe"],
                "turnover_ann_proxy": r["turnover_ann_proxy"],
                "mean_names": r["coverage"]["mean_names"],
                "residual_ic_mean": rc.get("residual_ic_mean"),
                "residual_icir": rc.get("residual_icir"),
                "residual_ic_t": rc.get("residual_ic_t"),
                "best_lambda": r["best_lambda"],
                "stack_icir_uplift": r["stack_icir_uplift"],
                "verdict": r["verdict"],
            }
        )
    summary = pd.DataFrame(flat).sort_values(["factor", "mode"])
    summary.to_csv(OUT / "sue_density_summary.csv", index=False)

    # pick best per factor across modes
    picks = []
    for fac, g in summary.groupby("factor"):
        # prefer significant residual t then uplift
        g2 = g.copy()
        g2["score"] = g2["residual_ic_t"].abs().fillna(0) + g2["stack_icir_uplift"].fillna(-1)
        best = g2.sort_values("score", ascending=False).iloc[0]
        picks.append(best.to_dict())

    verdict = {
        "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
        "sample_days": int(len(ret)),
        "hard_requirements": [
            "earliest_known_date_notice_express_income",
            "event_hold_and_daily_decay",
            "size_industry_neutral_before_residual_ic",
            "unexpected_profit_notice_surprise_20d_included",
        ],
        "base3": BASE3,
        "factors": SUE_FACTOR_LIST,
        "rows": rows,
        "best_by_factor": picks,
        "note": (
            "All IC/residual metrics use industry+ln_mktcap neutralized panels. "
            "Stack = (1-λ)·z(Base3 equal) + λ·z(SUE)."
        ),
    }
    (OUT / "sue_density_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    log(f"\nWrote {OUT / 'sue_density_summary.csv'}")
    log(f"Wrote {OUT / 'sue_density_verdict.json'}")
    log("\nBest by factor:")
    for p in picks:
        log(
            f"  {p['factor']} [{p['mode']}]: resid_t={p['residual_ic_t']:.2f} "
            f"uplift={p['stack_icir_uplift']:.3f} -> {p['verdict']}"
        )


if __name__ == "__main__":
    main()
