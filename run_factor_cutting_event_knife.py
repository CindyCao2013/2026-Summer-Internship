#!/usr/bin/env python
"""Event knife validation — limit-up/down filter vs raw RankIC.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_event_knife.py --preset ddb
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_event_knife.py --preset smoke
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

from factor_attribution import cs_zscore
from factor_cutting.cutting_analysis.event_knife import (
    compare_limit_filter,
    eval_factor,
    load_not_limit_mask,
    write_event_knife_report,
)
from factor_cutting.cutting_analysis.knife_family_analysis import synthesize_dual_knife
from factor_cutting.knives import build_knife
from factor_cutting.trade_count import load_trade_count_daily
from factor_cutting.w_cut import w_cut
from run_factor_cutting_v2 import PRESETS, load_panels, log, parse_day

OUT = Path("research/reports/factor_cutting_v1/event_knife")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="ddb")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--source", choices=["ddb", "oracle"], default="")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    start, end, source = PRESETS[args.preset]
    if args.start:
        start = parse_day(args.start)
    if args.end:
        end = parse_day(args.end)
    if args.source:
        source = args.source

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== Event Knife (limit filter) | {start.date()} -> {end.date()} | {source} ===")

    bundle = load_panels(
        start, end, source, keep_cache=args.keep_cache or args.preset == "paper"
    )
    close = bundle["close"]
    amount = bundle["amount"]
    volume = bundle["volume"]
    ret = bundle["ret"].loc[start:end]
    close_s = close.loc[:end]
    amount_s = amount.reindex_like(close_s)
    volume_s = volume.reindex_like(close_s)

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
    volume_w = volume_s.loc[start:end]
    tc_w = trade_count.loc[start:end] if trade_count is not None else None

    if tc_w is not None and tc_w.notna().any().any():
        knife_name = "ats_trade_count"
        knife = build_knife(knife_name, amount=amount_w, trade_count=tc_w)
    else:
        knife_name = "amount"
        knife = build_knife(knife_name, amount=amount_w)
    log(f"Primary knife={knife_name}")

    log("Loading not-limit mask (get_EOD_Not_Limit)...")
    not_limit = load_not_limit_mask(start, end)
    not_limit = not_limit.reindex(index=obj.index, columns=obj.columns)
    drop_frac = 1.0 - float(
        (obj.notna() & not_limit.notna()).sum().sum() / max(obj.notna().sum().sum(), 1)
    )
    log(f"  not_limit drop_frac≈{drop_frac:.2%} shape={not_limit.shape}")

    log("Comparing limit filter modes on ideal_reversal...")
    cmp_df, panels = compare_limit_filter(
        obj, knife, ret, not_limit, window=20, label=f"ideal_reversal_{knife_name}"
    )
    cmp_df.to_csv(OUT / "ideal_reversal_limit_compare.csv", index=False)
    for _, r in cmp_df.iterrows():
        log(
            f"  [{r['mode']}] RankIC={r['rank_ic']:.4f} ICIR={r['icir']:.2f} "
            f"Δ={r['rank_ic_delta']:+.4f} retn={r['ic_retention_vs_raw']:.3f}"
        )

    # Dual amount+ATS under raw vs filter modes
    dual_rows = []
    if knife_name == "ats_trade_count":
        log("Dual amount+ATS under raw / filter modes...")
        from factor_cutting.cutting_analysis.event_knife import apply_not_limit

        amt_knife = build_knife("amount", amount=amount_w)
        amt_cut = w_cut(obj, amt_knife, window=20)
        ats_cut = panels["raw"]
        dual_raw = synthesize_dual_knife(amt_cut, ats_cut, mode="residual_add")
        dual_rows.append(
            eval_factor(dual_raw, ret, label="amount+ATS:residual_add", mode="raw")
        )
        dual_rows.append(
            eval_factor(
                apply_not_limit(dual_raw, not_limit),
                ret,
                label="amount+ATS:residual_add",
                mode="filter_signal",
            )
        )
        amt_cut_f = w_cut(
            apply_not_limit(obj, not_limit),
            apply_not_limit(amt_knife, not_limit),
            window=20,
        )
        dual_cut = synthesize_dual_knife(
            amt_cut_f, panels["filter_cut"], mode="residual_add"
        )
        dual_rows.append(
            eval_factor(
                dual_cut, ret, label="amount+ATS:residual_add", mode="filter_cut"
            )
        )
        dual_df = pd.DataFrame(dual_rows)
        base_ic = dual_df.iloc[0]["rank_ic"]
        base_icir = dual_df.iloc[0]["icir"]
        dual_df["rank_ic_delta"] = dual_df["rank_ic"] - base_ic
        dual_df["icir_delta"] = dual_df["icir"] - base_icir
        dual_df["ic_retention_vs_raw"] = dual_df["rank_ic"] / base_ic
        dual_df.to_csv(OUT / "dual_amount_ats_limit_compare.csv", index=False)
        for _, r in dual_df.iterrows():
            log(
                f"  dual[{r['mode']}] RankIC={r['rank_ic']:.4f} ICIR={r['icir']:.2f} "
                f"Δ={r['rank_ic_delta']:+.4f}"
            )
    else:
        dual_df = pd.DataFrame()

    coverage = {
        "drop_frac": float(cmp_df.iloc[0].get("cov_drop_frac", drop_frac)),
        "mean_names_ref": float(cmp_df.iloc[0].get("cov_mean_names_ref", np.nan)),
        "mean_names_kept": float(cmp_df.iloc[0].get("cov_mean_names_kept", np.nan)),
    }
    period = f"{ret.index[0].date()} -> {ret.index[-1].date()}"

    # Verdict line
    best = cmp_df.loc[cmp_df["mode"] != "raw"].copy()
    best["_abs"] = best["rank_ic"].abs()
    best_row = best.sort_values("_abs", ascending=False).iloc[0]
    raw_row = cmp_df[cmp_df["mode"] == "raw"].iloc[0]
    improved = abs(best_row["rank_ic"]) > abs(raw_row["rank_ic"])
    verdict_line = (
        f"best_mode=`{best_row['mode']}` RankIC={best_row['rank_ic']:.4f} "
        f"(raw={raw_row['rank_ic']:.4f}, retention={best_row['ic_retention_vs_raw']:.3f}) "
        f"→ {'LIMIT FILTER HELPS' if improved else 'LIMIT FILTER NO UPLIFT / HURTS'}"
    )
    log(f"  {verdict_line}")

    write_event_knife_report(
        OUT / "summary.md",
        cmp_df,
        period=period,
        coverage=coverage,
        extra_notes=[
            f"Primary knife: `{knife_name}`",
            verdict_line,
            (
                "Dual amount+ATS also compared — see dual_amount_ats_limit_compare.csv"
                if not dual_df.empty
                else "Dual synth skipped (no ATS)"
            ),
        ],
    )
    if not dual_df.empty:
        with (OUT / "summary.md").open("a", encoding="utf-8") as f:
            f.write("\n## Dual amount+ATS (residual_add)\n\n")
            f.write("| Mode | RankIC | ICIR | ΔRankIC | Retention |\n")
            f.write("|------|--------|------|---------|----------|\n")
            for _, r in dual_df.iterrows():
                f.write(
                    f"| `{r['mode']}` | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
                    f"{r['rank_ic_delta']:+.4f} | {r['ic_retention_vs_raw']:.3f} |\n"
                )

    verdict = {
        "period": period,
        "source": source,
        "preset": args.preset,
        "primary_knife": knife_name,
        "coverage": coverage,
        "ideal_reversal_compare": cmp_df.to_dict(orient="records"),
        "dual_compare": dual_df.to_dict(orient="records") if not dual_df.empty else [],
        "verdict": verdict_line,
    }
    (OUT / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
