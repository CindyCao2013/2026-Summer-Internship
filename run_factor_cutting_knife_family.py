#!/usr/bin/env python
"""Knife-family incremental IC + dual-knife synthesis (Factor Cutting).

Focus period: 2018–2025 (ATS available). Answers:
  1. After controlling amount, does volume still have residual IC?
  2. After controlling amount, does ats_trade_count still add?
  3. Does dual-knife synth beat any single knife on |ICIR|?

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_knife_family.py --preset ddb
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_knife_family.py --preset smoke
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

from factor_cutting.cutting_analysis.knife_family_analysis import (
    dual_knife_ic_table,
    family_attribution_report,
    pairwise_residual_ic,
    write_family_markdown,
)
from factor_cutting.info_layer import search_knives
from factor_cutting.knives import build_knife
from factor_cutting.trade_count import load_trade_count_daily
from run_factor_cutting_v2 import PRESETS, load_panels, log, parse_day

OUT = Path("research/reports/factor_cutting_v1/knife_family")

# Core candidates for the multi-knife question
FOCUS_KNIVES = ["amount", "volume", "turnover_proxy", "ats_trade_count"]
FOCUS_PAIRS = [
    ("amount", "volume"),
    ("amount", "ats_trade_count"),
    ("volume", "ats_trade_count"),
    ("amount", "turnover_proxy"),
]


def _build_focus_knives(
    *,
    amount: pd.DataFrame,
    volume: pd.DataFrame,
    trade_count: pd.DataFrame | None,
    float_mktcap: pd.DataFrame | None,
    ret_1d: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in FOCUS_KNIVES:
        try:
            out[name] = build_knife(
                name,
                amount=amount,
                volume=volume,
                trade_count=trade_count,
                float_mktcap=float_mktcap,
                ret_1d=ret_1d,
            )
        except (ValueError, KeyError) as exc:
            log(f"  skip {name}: {exc}")
    return out


def _key_questions(pair_df: pd.DataFrame) -> list[str]:
    lines = []
    q1 = pair_df[(pair_df["knife"] == "volume") & (pair_df["vs_knife"] == "amount")]
    if not q1.empty:
        r = q1.iloc[0]
        sig = abs(r["residual_ic_t"]) >= 2.0
        lines.append(
            f"1. volume | amount: resid_IC={r['residual_ic']:.4f} "
            f"t={r['residual_ic_t']:.2f} |corr|={r['abs_corr']:.2f} "
            f"→ {'YES residual' if sig else 'NO / weak'} after controlling amount"
        )
    q2 = pair_df[
        (pair_df["knife"] == "ats_trade_count") & (pair_df["vs_knife"] == "amount")
    ]
    if not q2.empty:
        r = q2.iloc[0]
        sig = abs(r["residual_ic_t"]) >= 2.0
        lines.append(
            f"2. ATS | amount: resid_IC={r['residual_ic']:.4f} "
            f"t={r['residual_ic_t']:.2f} |corr|={r['abs_corr']:.2f} "
            f"→ {'YES incremental' if sig else 'NO / weak'} after controlling amount"
        )
    q2b = pair_df[
        (pair_df["knife"] == "ats_trade_count") & (pair_df["vs_knife"] == "volume")
    ]
    if not q2b.empty:
        r = q2b.iloc[0]
        lines.append(
            f"   ATS | volume: resid_IC={r['residual_ic']:.4f} "
            f"t={r['residual_ic_t']:.2f} |corr|={r['abs_corr']:.2f}"
        )
    return lines


def _synth_verdict(synth_df: pd.DataFrame) -> str:
    singles = synth_df[synth_df["kind"] == "single"].copy()
    duals = synth_df[synth_df["kind"] == "dual"].copy()
    if singles.empty or duals.empty:
        return "insufficient"
    best_single = singles.loc[singles["icir"].abs().idxmax()]
    best_dual = duals.loc[duals["icir"].abs().idxmax()]
    beat = abs(best_dual["icir"]) > abs(best_single["icir"])
    return (
        f"best_single=`{best_single['label']}` ICIR={best_single['icir']:.2f}; "
        f"best_dual=`{best_dual['label']}` ICIR={best_dual['icir']:.2f}; "
        f"{'DUAL WINS' if beat else 'single wins (no synth uplift)'}"
    )


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
    log(f"=== Knife Family Incremental IC | {start.date()} -> {end.date()} | {source} ===")

    bundle = load_panels(
        start, end, source, keep_cache=args.keep_cache or args.preset == "paper"
    )
    close = bundle["close"]
    amount = bundle["amount"]
    volume = bundle["volume"]
    ret_full = bundle["ret"]
    float_mkt = bundle["float_mktcap"]

    ret = ret_full.loc[start:end]
    close_s = close.loc[:end]
    amount_s = amount.reindex_like(close_s)
    volume_s = volume.reindex_like(close_s)
    if float_mkt is not None:
        float_mkt = float_mkt.reindex_like(close_s)

    trade_count = None
    try:
        tc_start = max(start - dt.timedelta(days=80), dt.datetime(2018, 11, 27))
        if end >= tc_start:
            log("Loading trade_count...")
            trade_count = load_trade_count_daily(tc_start, end).reindex_like(close_s)
            log(f"  trade_count mean names={trade_count.notna().sum(axis=1).mean():.0f}")
    except Exception as exc:
        log(f"WARNING: trade_count ({exc})")

    ret_1d = close_s / close_s.shift(1) - 1.0
    obj = ret_1d.loc[start:end]  # ret20 object = rolling return path inside w_cut

    # Object for W-cut is daily return panel (same as ideal_reversal pipeline)
    amount_w = amount_s.loc[start:end]
    volume_w = volume_s.loc[start:end]
    float_w = float_mkt.loc[start:end] if float_mkt is not None else None
    tc_w = trade_count.loc[start:end] if trade_count is not None else None

    log("Building focus knives...")
    knives = _build_focus_knives(
        amount=amount_w,
        volume=volume_w,
        trade_count=tc_w,
        float_mktcap=float_w,
        ret_1d=obj,
    )
    log(f"  knives={list(knives.keys())}")

    log("Family attribution + cut factors...")
    eval_df, corr_df, indep_df, cut_factors = family_attribution_report(
        obj, knives, ret, window=20
    )
    eval_df.to_csv(OUT / "knife_family_eval.csv", index=False)
    corr_df.to_csv(OUT / "knife_cut_corr.csv")
    indep_df.to_csv(OUT / "knife_independence.csv", index=False)
    write_family_markdown(OUT / "knife_family_attribution.md", eval_df, corr_df, indep_df)

    log("Pairwise residual IC matrix...")
    pair_df = pairwise_residual_ic(cut_factors, ret)
    pair_df.to_csv(OUT / "pairwise_residual_ic.csv", index=False)

    # Pivot residual IC / t for readability
    if not pair_df.empty:
        resid_mat = pair_df.pivot(index="knife", columns="vs_knife", values="residual_ic")
        t_mat = pair_df.pivot(index="knife", columns="vs_knife", values="residual_ic_t")
        resid_mat.to_csv(OUT / "residual_ic_matrix.csv")
        t_mat.to_csv(OUT / "residual_t_matrix.csv")

    log("Dual-knife synthesis...")
    synth_df = dual_knife_ic_table(cut_factors, ret, pairs=FOCUS_PAIRS)
    synth_df.to_csv(OUT / "dual_knife_synth.csv", index=False)

    log("search_knives API (focus candidates)...")
    result = search_knives(
        obj,
        ret,
        candidates=FOCUS_KNIVES,
        amount=amount_w,
        volume=volume_w,
        trade_count=tc_w,
        float_mktcap=float_w,
        ret_1d=obj,
    )
    (OUT / "search_knives.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    q_lines = _key_questions(pair_df)
    synth_line = _synth_verdict(synth_df)
    for line in q_lines:
        log(f"  {line}")
    log(f"  3. synth: {synth_line}")

    # Markdown summary
    lines = [
        "# Knife Family Incremental IC",
        "",
        f"**Period:** `{ret.index[0].date()} → {ret.index[-1].date()}` · source=`{source}`",
        f"**Object:** daily ret (W-cut window=20) · **Knives:** `{FOCUS_KNIVES}`",
        "",
        "## Key questions",
        "",
    ]
    lines.extend(f"- {x}" for x in q_lines)
    lines += ["", f"- 3. Dual-knife synth: {synth_line}", ""]

    lines += [
        "## Effectiveness ranking",
        "",
        "| Knife | Family | IC_spread | Separation | Effectiveness |",
        "|-------|--------|-----------|------------|---------------|",
    ]
    for _, r in eval_df.iterrows():
        lines.append(
            f"| `{r['knife']}` | {r['family']} | {r['ic_spread']:.4f} | "
            f"{r['separation']:.4f} | {r['effectiveness']:.4f} |"
        )

    lines += ["", "## Independence (vs closest peer)", ""]
    if indep_df is not None and not indep_df.empty:
        lines += [
            "| Knife | Peer | |corr| | resid_t | Independent? |",
            "|-------|------|-------|---------|--------------|",
        ]
        for _, r in indep_df.iterrows():
            lines.append(
                f"| `{r['knife']}` | `{r['closest_peer']}` | {r['abs_corr_to_peer']:.2f} | "
                f"{r['resid_t_vs_peer']:.2f} | {r['independent']} |"
            )

    lines += ["", "## Cut-factor correlation", "", "```", corr_df.round(3).to_string(), "```", ""]

    if not pair_df.empty:
        lines += [
            "## Residual IC matrix (row residualized vs column)",
            "",
            "```",
            resid_mat.round(4).to_string(),
            "```",
            "",
            "## Residual t-stat matrix",
            "",
            "```",
            t_mat.round(2).to_string(),
            "```",
            "",
        ]

    lines += ["## Dual-knife synthesis", ""]
    lines += [
        "| Label | Kind | RankIC | ICIR | n_days |",
        "|-------|------|--------|------|--------|",
    ]
    for _, r in synth_df.sort_values("icir", key=lambda s: s.abs(), ascending=False).iterrows():
        lines.append(
            f"| `{r['label']}` | {r['kind']} | {r['rank_ic']:.4f} | "
            f"{r['icir']:.2f} | {int(r['n_days']) if pd.notna(r['n_days']) else '-'} |"
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- `amount`/`volume` are same participation family; high corr → expect weak residual.",
        "- `ats_trade_count` is trader_structure; residual vs amount/volume is the multi-knife prize.",
        "- Dual synth: `equal_z` = blend; `residual_add` = base + orthogonal component.",
        "- Do **not** replace paper ATS with volume solely because raw effectiveness is higher.",
        "",
        f"search_knives best=`{result.get('best_knife')}` "
        f"independent=`{result.get('independent_knives')}`",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    verdict = {
        "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
        "source": source,
        "preset": args.preset,
        "focus_knives": FOCUS_KNIVES,
        "key_questions": q_lines,
        "synth_verdict": synth_line,
        "search_knives": result,
        "independence": indep_df.to_dict(orient="records") if indep_df is not None else [],
    }
    (OUT / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    log(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
