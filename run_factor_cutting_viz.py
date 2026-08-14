#!/usr/bin/env python
"""Factor Cutting Visualization Layer — generate researcher-facing plot packs.

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_viz.py --preset smoke
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_viz.py --preset ddb
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

from factor_attribution import cs_zscore
from factor_cutting.cutting_analysis.neutralization import neutralization_ladder
from factor_cutting.cutting_analysis.stability import universe_stability_table
from factor_cutting.cutting_analysis.visualization import generate_cutting_viz_pack
from factor_cutting.ideal_amplitude import compute_ideal_amplitude
from factor_cutting.knives import build_knife
from factor_cutting.trade_count import load_trade_count_daily
from factor_cutting.w_cut import w_cut
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from run_factor_cutting_v2 import (
    PRESETS,
    load_panels,
    load_universe_masks,
    log,
    parse_day,
    try_load_industry,
)

OUT = Path("research/reports/factor_cutting_v1")
BASE3 = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--source", choices=["ddb", "oracle"], default="")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--factors", type=str, default="ideal_reversal,ideal_amplitude")
    args = parser.parse_args()

    start, end, source = PRESETS[args.preset]
    if args.start:
        start = parse_day(args.start)
    if args.end:
        end = parse_day(args.end)
    if args.source:
        source = args.source

    log(f"=== Cutting Viz Layer | {start.date()} -> {end.date()} | {source} ===")
    bundle = load_panels(
        start, end, source, keep_cache=args.keep_cache or args.preset == "paper"
    )
    close = bundle["close"]
    open_ = bundle["open"]
    high = bundle["high"]
    low = bundle["low"]
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
            trade_count = load_trade_count_daily(tc_start, end).reindex_like(close_s)
            log(f"trade_count mean names={trade_count.notna().sum(axis=1).mean():.0f}")
    except Exception as exc:
        log(f"WARNING: trade_count ({exc})")

    industry = try_load_industry(start, end, close_s)
    universe_masks = {}
    if not args.skip_universe:
        universe_masks = load_universe_masks(session, start, end, ret.index)

    # Base3 for residual plots
    log("Building Base3 anchors...")
    pv = build_factor_cache(
        df_close=bundle["close"],
        df_open=bundle["open"],
        df_high=bundle["high"],
        df_low=bundle["low"],
        df_volume=bundle["volume"],
        df_amount=bundle["amount"],
        df_turnover=None,
    )
    base3_list = [
        build_eod_engine_factor(n, pv).reindex(index=ret.index, columns=ret.columns) for n in BASE3
    ]

    ret_1d = close_s / close_s.shift(1) - 1.0
    # Original Ret20 as cumulative return (paper sign: negative IC)
    ret20 = close_s / close_s.shift(20) - 1.0

    wanted = {x.strip() for x in args.factors.split(",") if x.strip()}
    results = {}

    if "ideal_reversal" in wanted:
        log("--- ideal_reversal viz ---")
        if trade_count is not None and trade_count.notna().any().any():
            knife_name = "ats_trade_count"
            knife = build_knife(knife_name, amount=amount_s, trade_count=trade_count)
        else:
            knife_name = "ats_volume"
            knife = build_knife(knife_name, amount=amount_s, volume=volume_s)
        spread, high_leg, low_leg = w_cut(ret_1d, knife, window=20, return_legs=True)
        spread = spread.loc[start:end]
        high_leg = high_leg.loc[start:end]
        low_leg = low_leg.loc[start:end]
        knife_s = knife.loc[start:end]
        orig = ret20.loc[start:end]

        ladder = neutralization_ladder(
            spread, ret, industry=industry, float_mktcap=float_mkt
        )
        uni_df = None
        if universe_masks:
            uni_df = universe_stability_table(cs_zscore(spread), ret, universe_masks)

        pack = generate_cutting_viz_pack(
            factor_name="ideal_reversal",
            out_root=OUT / "ideal_reversal",
            original=orig,
            high=high_leg,
            low=low_leg,
            spread=spread,
            ret=ret,
            knife=knife_s,
            knife_name=knife_name,
            universe_df=uni_df,
            neut_ladder=ladder,
            base3_panels=base3_list,
            original_name="Ret20",
        )
        results["ideal_reversal"] = pack
        log(
            f"  sep={pack['separation']:.4f} purity={pack['purity']:.3f} "
            f"HL_sharpe={pack['hl_sharpe']:.2f}"
        )

    if "ideal_amplitude" in wanted:
        log("--- ideal_amplitude viz ---")
        amp, v_high, v_low = compute_ideal_amplitude(
            high.reindex_like(close_s),
            low.reindex_like(close_s),
            close_s,
            open_=open_.reindex_like(close_s),
            return_legs=True,
        )
        # traditional amplitude = 20d mean range
        from factor_cutting.ideal_amplitude import daily_amplitude

        trad = daily_amplitude(high.reindex_like(close_s), low.reindex_like(close_s))
        trad20 = trad.rolling(20, min_periods=10).mean().loc[start:end]

        ladder = neutralization_ladder(
            amp.loc[start:end], ret, industry=industry, float_mktcap=float_mkt
        )
        uni_df = None
        if universe_masks:
            uni_df = universe_stability_table(cs_zscore(amp.loc[start:end]), ret, universe_masks)

        # reuse viz pack with amplitude naming via a thin adapter:
        # build_ic_comparison_table hardcodes M_high/M_low — patch names in post for amplitude
        pack = generate_cutting_viz_pack(
            factor_name="ideal_amplitude",
            out_root=OUT / "ideal_amplitude",
            original=trad20,
            high=v_high.loc[start:end],
            low=v_low.loc[start:end],
            spread=amp.loc[start:end],
            ret=ret,
            knife=close_s.loc[start:end],
            knife_name="close_price_state",
            universe_df=uni_df,
            neut_ladder=ladder,
            base3_panels=base3_list,
            original_name="Amp20",
            high_name="V_high",
            low_name="V_low",
            spread_name="V (spread)",
        )
        results["ideal_amplitude"] = {
            k: (float(v) if isinstance(v, (float, int)) else str(v) if k != "ic_table" else v.to_dict())
            for k, v in pack.items()
            if k != "residual"
        }
        results["ideal_amplitude"]["residual"] = pack.get("residual", {})
        log(f"  amplitude HL_sharpe={pack['hl_sharpe']:.2f}")

    (OUT / "viz_verdict.json").write_text(
        json.dumps(
            {
                "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
                "preset": args.preset,
                "factors": {
                    k: {
                        "separation": v.get("separation"),
                        "purity": v.get("purity"),
                        "hl_sharpe": v.get("hl_sharpe"),
                        "residual_ic_t": (v.get("residual") or {}).get("residual_ic_t"),
                    }
                    for k, v in results.items()
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    log(f"\nWrote plot packs under {OUT}/{{factor}}/")


if __name__ == "__main__":
    main()
