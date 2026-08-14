#!/usr/bin/env python
"""Factor Cutting Research Platform v2.1

Adds:
  1. Neutralization ladder (raw / size / industry / size+industry)
  2. Knife family attribution (participation vs trader_structure vs liquidity)
  3. Information Layer API: search_knives(...)

Does NOT open minute layer.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v21.py --preset smoke
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v21.py --preset ddb
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_v21.py --preset paper --keep-cache
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
    family_attribution_report,
    write_family_markdown,
)
from factor_cutting.cutting_analysis.knife_ic import ic_stats
from factor_cutting.cutting_analysis.neutralization import (
    neutralization_ladder,
    write_factor_decay_report,
)
from factor_cutting.info_layer import search_knives
from factor_cutting.knives import available_knives, build_knife
from factor_cutting.trade_count import load_trade_count_daily
from factor_cutting.w_cut import w_cut
from run_factor_cutting_v2 import (
    PRESETS,
    load_panels,
    load_universe_masks,
    log,
    parse_day,
    try_load_industry,
)

OUT = Path("research/reports/factor_cutting_v21")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--source", choices=["ddb", "oracle"], default="")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-universe", action="store_true")
    args = parser.parse_args()

    start, end, source = PRESETS[args.preset]
    if args.start:
        start = parse_day(args.start)
    if args.end:
        end = parse_day(args.end)
    if args.source:
        source = args.source

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== Factor Cutting v2.1 | {start.date()} -> {end.date()} | {source} ===")

    bundle = load_panels(
        start, end, source, keep_cache=args.keep_cache or args.preset == "paper"
    )
    close = bundle["close"]
    amount = bundle["amount"]
    volume = bundle["volume"]
    ret_full = bundle["ret"]
    float_mkt = bundle["float_mktcap"]
    session = bundle["session"]

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
    except Exception as exc:
        log(f"WARNING: trade_count ({exc})")

    industry = try_load_industry(start, end, close_s)
    if float_mkt is None and source == "ddb":
        log("WARNING: no float_mktcap")

    universe_masks = {}
    if not args.skip_universe:
        universe_masks = load_universe_masks(session, start, end, ret.index)

    ret_1d = close_s / close_s.shift(1) - 1.0
    obj = ret_1d.loc[start:end]

    # Primary paper knife cut
    if trade_count is not None and trade_count.notna().any().any():
        knife_name = "ats_trade_count"
        knife = build_knife(knife_name, amount=amount_s, trade_count=trade_count)
    else:
        knife_name = "ats_volume"
        knife = build_knife(knife_name, amount=amount_s, volume=volume_s)
    log(f"Primary knife={knife_name}")
    ideal_rev = w_cut(ret_1d, knife, window=20).loc[start:end]

    # --- Task 2: neutralization ladder ---
    log("Neutralization ladder...")
    ladder = neutralization_ladder(
        ideal_rev, ret, industry=industry, float_mktcap=float_mkt
    )
    ladder.to_csv(OUT / "factor_decay_ideal_reversal.csv", index=False)
    write_factor_decay_report(OUT / "factor_decay_report.md", "ideal_reversal", ladder)
    for _, r in ladder.iterrows():
        log(
            f"  [{r['mode']}] RankIC={r['rank_ic']:.4f} ICIR={r['icir']:.2f} "
            f"retn={r['ic_retention_vs_raw']}"
        )

    # Universe IC on size_industry if available else raw
    if universe_masks:
        from factor_cutting.cutting_analysis.neutralization import neutralize_panel
        from factor_cutting.cutting_analysis.stability import universe_stability_table
        from factor_attribution import cs_zscore

        mode = "size_industry" if "size_industry" in set(ladder["mode"]) else "raw"
        try:
            neut = neutralize_panel(
                ideal_rev, mode, industry=industry, float_mktcap=float_mkt
            )
        except ValueError:
            neut = ideal_rev
            mode = "raw"
        uni = universe_stability_table(cs_zscore(neut), ret, universe_masks)
        uni.to_csv(OUT / f"universe_{mode}.csv", index=False)
        log(f"Universe table under mode={mode}")

    # --- Task 3: knife family attribution ---
    log("Knife family attribution...")
    knives = available_knives(
        amount=amount_s.loc[start:end],
        volume=volume_s.loc[start:end],
        trade_count=trade_count.loc[start:end] if trade_count is not None else None,
        ret_1d=obj,
    )
    eval_df, corr_df, indep_df, _ = family_attribution_report(obj, knives, ret)
    eval_df.to_csv(OUT / "knife_family_eval.csv", index=False)
    corr_df.to_csv(OUT / "knife_cut_corr.csv")
    indep_df.to_csv(OUT / "knife_independence.csv", index=False)
    write_family_markdown(OUT / "knife_family_attribution.md", eval_df, corr_df, indep_df)
    for _, r in eval_df.iterrows():
        log(
            f"  {r['knife']} [{r['family']}]: eff={r['effectiveness']:.4f} "
            f"spread={r['ic_spread']:.4f}"
        )

    # --- Task 4: Information Layer API ---
    log("search_knives API...")
    result = search_knives(
        obj,
        ret,
        amount=amount_s.loc[start:end],
        volume=volume_s.loc[start:end],
        trade_count=trade_count.loc[start:end] if trade_count is not None else None,
        ret_1d=obj,
    )
    (OUT / "search_knives.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    log(
        f"  best={result.get('best_knife')} "
        f"eff={result.get('effectiveness')} "
        f"independent={result.get('independent_knives')}"
    )

    # Primary factor headline IC
    raw_ic = ic_stats(ideal_rev, ret)
    verdict = {
        "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
        "n_days": int(len(ret)),
        "source": source,
        "preset": args.preset,
        "primary_knife": knife_name,
        "ideal_reversal_raw": {
            "rank_ic": raw_ic["rank_ic"],
            "icir": raw_ic["icir"],
        },
        "neutralization": ladder.to_dict(orient="records"),
        "search_knives": result,
        "note": (
            "v2.1: neutralization ladder + knife families + search_knives. "
            "Do not replace ATS with volume solely because score is higher — "
            "check independence / family attribution."
        ),
    }
    (OUT / "verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Factor Cutting Research Platform v2.1",
                "",
                f"Period: `{verdict['period']}` · source=`{source}`",
                "",
                "## Outputs",
                "- `factor_decay_report.md` — raw vs size vs industry vs size+industry",
                "- `knife_family_attribution.md` — participation / trader_structure / liquidity",
                "- `search_knives.json` — Information Layer API result",
                "",
                f"Best knife (API): `{result.get('best_knife')}` · "
                f"independent: `{result.get('independent_knives')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
