#!/usr/bin/env python
"""P1 event density vs Base3 — major-holder / insider (size+ind residual, hold+decay).

Usage:
  OMP_NUM_THREADS=1 python run_event_density_v1.py --sample-days 504
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
from factor_attribution import combine_equal_weight, cs_zscore, hl_sharpe_from_composite
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_events_p1 import (
    P1_FACTOR_LIST,
    build_p1_event_tables,
    build_p1_panels,
    load_p1_bundle,
)
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from run_sue_density_v1 import coverage_stats, turnover_proxy

OUT = Path("research/reports/event_density_v1")
BASE3 = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]
STACK_LAMBDAS = [0.0, 0.1, 0.2, 0.3]


def log(msg: str) -> None:
    print(msg, flush=True)


def evaluate(name, mode, panel, ret, base3_list, base3_combo) -> dict:
    panel = panel.reindex(index=ret.index, columns=ret.columns)
    ic_daily = daily_rank_ic_series(panel, ret)
    ic_mean = float(ic_daily.mean()) if len(ic_daily.dropna()) else np.nan
    icir = icir_from_daily(ic_daily)
    sharpe, ann, direction = hl_sharpe_from_composite(panel, ret)
    to = turnover_proxy(panel)
    resid = {"vs_base3_combo": residual_ic_stats(panel, ret, base3_combo)}
    for bname, bp in zip(BASE3, base3_list):
        resid[f"vs_{bname}"] = residual_ic_stats(panel, ret, bp)
    b, s = cs_zscore(base3_combo), cs_zscore(panel)
    stack_rows = []
    for lam in STACK_LAMBDAS:
        combo = (1.0 - lam) * b + lam * s if lam > 0 else b
        stack_rows.append({"lambda": lam, **evaluate_stack_signal(combo, ret)})
    base_icir = stack_rows[0].get("icir", np.nan)
    best = max(stack_rows, key=lambda r: (r.get("icir") if pd.notna(r.get("icir")) else -1e9))
    uplift = (
        float(best["icir"] - base_icir)
        if pd.notna(best.get("icir")) and pd.notna(base_icir)
        else np.nan
    )
    rc = resid["vs_base3_combo"]
    verdict = "drop"
    if (
        pd.notna(rc.get("residual_ic_t"))
        and abs(rc["residual_ic_t"]) >= 2.0
        and pd.notna(uplift)
        and uplift >= 0
        and (not pd.notna(to) or to <= 120)
    ):
        verdict = "enhancer_candidate"
    elif pd.notna(rc.get("residual_ic_t")) and abs(rc["residual_ic_t"]) >= 2.0:
        verdict = "independent_but_stack_weak"
    elif pd.notna(icir) and abs(icir) >= 0.5:
        verdict = "raw_signal_only"
    return {
        "factor": name,
        "mode": mode,
        "rank_ic": ic_mean,
        "icir": icir,
        "gross_hl_sharpe": sharpe,
        "turnover_ann_proxy": to,
        "coverage": coverage_stats(panel),
        "residual": resid,
        "stack": stack_rows,
        "best_lambda": best.get("lambda"),
        "stack_icir_uplift": uplift,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-days", type=int, default=504)
    parser.add_argument("--cache-dir", type=str, default="research/cache/p1_events")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    log("=== P1 Event Density vs Base3 ===")
    enriched, _ = load_eod_enriched_tables(preheat, end)
    industry = load_citics_industry_panel(start, end)
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret = ret_full.iloc[max(0, len(ret_full) - args.sample_days) :]
    log(f"Ret: {ret.index[0].date()} -> {ret.index[-1].date()} ({len(ret)}d)")

    pv = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    base3 = {
        n: build_eod_engine_factor(n, pv).loc[start:end].reindex(index=ret.index, columns=ret.columns)
        for n in BASE3
    }
    base3_list = [base3[n] for n in BASE3]
    base3_combo = combine_equal_weight(base3_list)

    bundle = load_p1_bundle(start, end, cache_dir=Path(args.cache_dir), keep_cache=True)
    events = build_p1_event_tables(bundle)
    for k, ev in events.items():
        log(f"  events {k}: {len(ev):,}")

    trade_index = enriched.close.loc[start:end].index
    cols = enriched.close.columns
    float_mkt = enriched.float_mktcap.reindex(index=trade_index, columns=cols)
    ind = industry.reindex(index=trade_index, columns=cols)

    rows = []
    for mode in ("hold", "decay"):
        log(f"--- mode={mode} ---")
        panels = build_p1_panels(events, trade_index, cols, mode=mode)
        for name in P1_FACTOR_LIST:
            raw = panels[name].reindex(index=ret.index, columns=ret.columns)
            neut = cs_zscore(
                neutralize_size_industry(raw, ind.reindex_like(raw), float_mkt.reindex_like(raw))
            )
            m = evaluate(name, mode, neut, ret, base3_list, base3_combo)
            rows.append(m)
            rc = m["residual"]["vs_base3_combo"]
            log(
                f"  [{mode}] {name}: IC={m['rank_ic']:.4f} ICIR={m['icir']:.2f} "
                f"resid_t={rc.get('residual_ic_t', np.nan):.2f} "
                f"uplift={m['stack_icir_uplift']:.3f} -> {m['verdict']}"
            )

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
                "residual_ic_t": rc.get("residual_ic_t"),
                "stack_icir_uplift": r["stack_icir_uplift"],
                "verdict": r["verdict"],
            }
        )
    summary = pd.DataFrame(flat)
    summary.to_csv(OUT / "event_density_summary.csv", index=False)
    verdict = {
        "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
        "note": "Equity incentive table incomplete in Wind — excluded. Size+industry neutralized.",
        "blocked_sources": ["ASHARESTOCKINCENTIVEIMPLEMENT (~444 rows only)"],
        "rows": rows,
    }
    (OUT / "event_density_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
