#!/usr/bin/env python
"""Operability checklist — long excess, turnover, size, MDD ± limit filter.

Targets:
  - amount+ATS residual_add (best dual from knife family)
  - ideal_amplitude
  - ats_trade_count single (baseline)

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_operability.py --preset ddb
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

import pandas as pd

from factor_cutting.cutting_analysis.event_knife import apply_not_limit, load_not_limit_mask
from factor_cutting.cutting_analysis.knife_family_analysis import synthesize_dual_knife
from factor_cutting.cutting_analysis.operability import (
    flatten_operability,
    operability_report,
    write_operability_markdown,
)
from factor_cutting.ideal_amplitude import compute_ideal_amplitude
from factor_cutting.knives import build_knife
from factor_cutting.trade_count import load_trade_count_daily
from factor_cutting.w_cut import w_cut
from run_factor_cutting_v2 import PRESETS, load_panels, log, parse_day

OUT = Path("research/reports/factor_cutting_v1/operability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="ddb")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--source", choices=["ddb", "oracle"], default="")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-limit", action="store_true")
    args = parser.parse_args()

    start, end, source = PRESETS[args.preset]
    if args.start:
        start = parse_day(args.start)
    if args.end:
        end = parse_day(args.end)
    if args.source:
        source = args.source

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== Operability | {start.date()} -> {end.date()} | {source} ===")

    bundle = load_panels(
        start, end, source, keep_cache=args.keep_cache or args.preset == "paper"
    )
    close = bundle["close"]
    open_ = bundle["open"]
    high = bundle["high"]
    low = bundle["low"]
    amount = bundle["amount"]
    ret = bundle["ret"].loc[start:end]
    float_mkt = bundle["float_mktcap"]
    close_s = close.loc[:end]
    amount_s = amount.reindex_like(close_s)
    if float_mkt is not None:
        float_mkt = float_mkt.reindex_like(close_s)

    trade_count = None
    try:
        tc_start = max(start - dt.timedelta(days=80), dt.datetime(2018, 11, 27))
        if end >= tc_start:
            log("Loading trade_count...")
            trade_count = load_trade_count_daily(tc_start, end).reindex_like(close_s)
    except Exception as exc:
        log(f"WARNING: trade_count ({exc})")

    ret_1d = close_s / close_s.shift(1) - 1.0
    obj = ret_1d.loc[start:end]
    amount_w = amount_s.loc[start:end]
    tc_w = trade_count.loc[start:end] if trade_count is not None else None
    float_w = float_mkt.loc[start:end] if float_mkt is not None else None

    factors: dict[str, pd.DataFrame] = {}

    # ATS single
    if tc_w is not None and tc_w.notna().any().any():
        ats_knife = build_knife("ats_trade_count", amount=amount_w, trade_count=tc_w)
        factors["ats_trade_count"] = w_cut(obj, ats_knife, window=20)
        amt_cut = w_cut(obj, build_knife("amount", amount=amount_w), window=20)
        factors["amount_plus_ats_residual_add"] = synthesize_dual_knife(
            amt_cut, factors["ats_trade_count"], mode="residual_add"
        )
    else:
        log("WARNING: no ATS — dual skipped")

    # ideal amplitude
    log("Computing ideal_amplitude...")
    amp = compute_ideal_amplitude(
        high.reindex_like(close_s),
        low.reindex_like(close_s),
        close_s,
        open_=open_.reindex_like(close_s),
        window=20,
    )
    factors["ideal_amplitude"] = amp.loc[start:end]

    not_limit = None
    if not args.skip_limit:
        log("Loading not-limit mask...")
        not_limit = load_not_limit_mask(start, end)
        not_limit = not_limit.reindex(index=obj.index, columns=obj.columns)

    rows = []
    notes = []
    for name, panel in factors.items():
        for mode in ("raw", "filter_signal"):
            if mode == "filter_signal":
                if not_limit is None:
                    continue
                use = apply_not_limit(panel, not_limit)
            else:
                use = panel
            log(f"Operability: {name} / {mode}...")
            rep = operability_report(
                use,
                ret,
                float_mktcap=float_w,
                label=name,
                mode=mode,
            )
            # persist group pnl / turnover
            sub = OUT / name / mode
            sub.mkdir(parents=True, exist_ok=True)
            rep["panels"]["group_pnl"].to_csv(sub / "group_pnl.csv")
            rep["panels"]["group_to"].to_csv(sub / "group_turnover.csv")
            rep["panels"]["long_excess"].to_csv(sub / "long_excess.csv", header=["ret"])
            rep["panels"]["hl_pnl"].to_csv(sub / "hl_signed.csv", header=["ret"])
            # drop panels before flatten
            flat = flatten_operability(rep)
            rows.append(flat)
            log(
                f"  long_excess_ann={flat['long_excess_ann']:.2%} "
                f"sharpe={flat['long_excess_sharpe']:.2f} "
                f"hl_ann={flat['hl_ann_ret']:.2%} "
                f"monthly_TO_hl≈{flat['monthly_to_hl_approx']:.2f} "
                f"size_pctile={flat['size_median_pctile']:.2f}"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "operability_summary.csv", index=False)

    # Quick verdicts
    for name in summary["label"].unique():
        sub = summary[summary["label"] == name]
        raw = sub[sub["mode"] == "raw"]
        if raw.empty:
            continue
        r = raw.iloc[0]
        note = (
            f"`{name}` raw: long_excess={r['long_excess_ann']:.1%}/Sharpe {r['long_excess_sharpe']:.2f}, "
            f"HL={r['hl_ann_ret']:.1%}/Sharpe {r['hl_sharpe']:.2f}, "
            f"monthly HL TO≈{r['monthly_to_hl_approx']:.2f}, "
            f"long size pctile={r['size_median_pctile']:.2f} "
            f"(bottom20={r['size_frac_bottom_20']:.0%})"
        )
        notes.append(note)
        filt = sub[sub["mode"] == "filter_signal"]
        if not filt.empty:
            f = filt.iloc[0]
            notes.append(
                f"`{name}` filter_signal: long_excess={f['long_excess_ann']:.1%} "
                f"(Δ={f['long_excess_ann']-r['long_excess_ann']:+.1%}), "
                f"HL Sharpe {f['hl_sharpe']:.2f}"
            )

    write_operability_markdown(OUT / "summary.md", summary, notes)
    (OUT / "verdict.json").write_text(
        json.dumps(
            {
                "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
                "source": source,
                "preset": args.preset,
                "rows": summary.to_dict(orient="records"),
                "notes": notes,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"\nWrote {OUT}/")
    for n in notes:
        log(f"  {n}")


if __name__ == "__main__":
    main()
