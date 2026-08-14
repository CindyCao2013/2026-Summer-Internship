#!/usr/bin/env python
"""Proxy-knife backfill for ideal_reversal (pre-2018 without L2 trade_count).

1. On 2018+ overlap: match candidate OHLCV knives to ats_trade_count via IC-series corr
2. Pick best proxy (prefer corr >= 0.8)
3. Stitch ATS (when trade_count OK) + proxy (earlier) → full-history factor
4. Write honest report under research/reports/factor_cutting_v1/ideal_reversal_proxy/

Usage:
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_proxy_reversal.py --preset paper --keep-cache
  OMP_NUM_THREADS=1 PYTHONPATH=. python run_factor_cutting_proxy_reversal.py --preset ddb
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

from factor_cutting.cutting_analysis.knife_ic import ic_stats, monthly_ic_stats
from factor_cutting.cutting_analysis.neutralization import neutralization_ladder
from factor_cutting.cutting_analysis.proxy_knife import (
    CORR_ACCEPT,
    match_proxies_to_ats,
    stitch_ats_with_proxy,
    write_proxy_match_report,
)
from factor_cutting.cutting_analysis.stability import yearly_ic_table
from factor_cutting.trade_count import load_trade_count_daily
from run_factor_cutting_v2 import PRESETS, load_panels, log, parse_day, try_load_industry

OUT = Path("research/reports/factor_cutting_v1/ideal_reversal_proxy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="paper")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--source", choices=["ddb", "oracle"], default="")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--corr-accept", type=float, default=CORR_ACCEPT)
    parser.add_argument(
        "--force-knife",
        type=str,
        default="",
        help="Skip match; use this proxy knife name (e.g. ats_volume)",
    )
    args = parser.parse_args()

    start, end, source = PRESETS[args.preset]
    if args.start:
        start = parse_day(args.start)
    if args.end:
        end = parse_day(args.end)
    if args.source:
        source = args.source

    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== Proxy Reversal | {start.date()} -> {end.date()} | {source} ===")

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
    else:
        # Oracle path: try DDB mktcap overlap only
        try:
            from factor_data_loaders import load_eod_enriched_tables

            en, _ = load_eod_enriched_tables(max(start, dt.datetime(2018, 1, 2)), end)
            float_mkt = en.float_mktcap.reindex_like(close_s)
            log("  float_mktcap from DDB overlap for turnover_proxy")
        except Exception as exc:
            log(f"WARNING: float_mktcap unavailable ({exc})")

    trade_count = None
    try:
        tc_start = max(start - dt.timedelta(days=80), dt.datetime(2018, 11, 27))
        if end >= tc_start:
            trade_count = load_trade_count_daily(tc_start, end).reindex_like(close_s)
            log(f"trade_count mean names={trade_count.notna().sum(axis=1).mean():.0f}")
    except Exception as exc:
        log(f"WARNING: trade_count ({exc})")

    if trade_count is None or not trade_count.notna().any().any():
        raise RuntimeError("Need trade_count on 2018+ for ATS benchmark match")

    ret_1d = close_s / close_s.shift(1) - 1.0

    log("Matching proxies to ATS on overlap...")
    ranking, cut_panels, ats_factor = match_proxies_to_ats(
        ret_1d,
        ret_full.reindex_like(close_s),
        amount=amount_s,
        volume=volume_s,
        trade_count=trade_count,
        float_mktcap=float_mkt,
        match_end=pd.Timestamp(end),
    )
    ranking.to_csv(OUT / "proxy_match_ranking.csv", index=False)
    log("Match ranking:")
    for _, r in ranking.iterrows():
        log(
            f"  {r['knife']}: corr={r['ic_series_corr_vs_ats']:.3f} "
            f"IC={r['rank_ic']:.4f} ICIR={r['icir']:.2f}"
        )

    prox = ranking[ranking["role"] == "proxy"]
    if args.force_knife:
        best_knife = args.force_knife
        best_corr = float(
            prox.loc[prox["knife"] == best_knife, "ic_series_corr_vs_ats"].iloc[0]
        ) if (prox["knife"] == best_knife).any() else float("nan")
    else:
        best_row = prox.iloc[0]
        best_knife = str(best_row["knife"])
        best_corr = float(best_row["ic_series_corr_vs_ats"])

    accept = best_corr >= args.corr_accept if pd.notna(best_corr) else False
    log(f"Best proxy={best_knife} corr={best_corr:.3f} accept={accept}")

    proxy_factor = cut_panels[best_knife]
    stitched = stitch_ats_with_proxy(ats_factor, proxy_factor, trade_count)
    # evaluation window
    fac = stitched.loc[start:end]
    mode = "stitch_ats_plus_proxy" if accept or True else "proxy_only"
    # always stitch: ATS when available is strictly better; proxy fills holes
    st = ic_stats(fac, ret)
    mon = monthly_ic_stats(fac, ret)
    yearly = yearly_ic_table(fac, ret)
    yearly.to_csv(OUT / "yearly.csv", index=False)

    industry = try_load_industry(start, end, close_s)
    ladder = neutralization_ladder(fac, ret, industry=industry, float_mktcap=float_mkt)
    ladder.to_csv(OUT / "neutralization.csv", index=False)

    # Coverage diagnostics
    cover = trade_count.notna().sum(axis=1)
    n_ats_days = int((cover >= 200).sum())
    n_proxy_only = int((cover < 200).loc[start:end].sum()) if len(cover) else 0

    full_stats = {
        "knife": best_knife,
        "mode": mode,
        "corr_vs_ats": best_corr,
        "corr_accept": args.corr_accept,
        "passed_threshold": bool(accept),
        "rank_ic": st["rank_ic"],
        "icir": st["icir"],
        "monthly_rank_ic": mon["monthly_rank_ic"],
        "monthly_icir": mon["monthly_icir"],
        "n_days": st["n_days"],
        "n_ats_days": n_ats_days,
        "n_proxy_fill_days_in_sample": n_proxy_only,
        "label": f"ideal_reversal_proxy_{best_knife}",
        "honesty": (
            "NOT paper ATS on full history. "
            f"ATS used where trade_count coverage>=200; else `{best_knife}`."
        ),
    }

    write_proxy_match_report(
        OUT / "proxy_match_report.md",
        ranking,
        best_knife=best_knife,
        best_corr=best_corr,
        accept=args.corr_accept,
        full_stats=full_stats,
    )

    # Simple IC bar vs pure ATS on overlap for the report
    ats_ov = ic_stats(
        ats_factor.loc[ats_factor.index >= pd.Timestamp("2018-11-27")].loc[start:end],
        ret.loc[ret.index >= pd.Timestamp("2018-11-27")],
    )
    (OUT / "summary.md").write_text(
        "\n".join(
            [
                "# ideal_reversal — Proxy Knife Full History",
                "",
                f"**Label:** `{full_stats['label']}`",
                "",
                full_stats["honesty"],
                "",
                f"- Best proxy: `{best_knife}` · IC-series corr vs ATS = **{best_corr:.3f}** "
                f"({'PASS' if accept else 'BELOW'} threshold {args.corr_accept})",
                f"- Full sample RankIC **{st['rank_ic']:.4f}** · ICIR **{st['icir']:.2f}** · "
                f"monthly **{mon['monthly_rank_ic']:.4f}**",
                f"- Overlap ATS-only RankIC {ats_ov['rank_ic']:.4f} (reference)",
                f"- ATS days (coverage≥200): {n_ats_days} · proxy-fill days in sample: {n_proxy_only}",
                "",
                "## Yearly RankIC (stitched)",
                "",
                yearly.to_string(index=False),
                "",
                "## Neutralization",
                "",
                ladder.to_string(index=False),
                "",
                "See `proxy_match_report.md` for knife ranking.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (OUT / "verdict.json").write_text(
        json.dumps(
            {
                "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
                "preset": args.preset,
                "match_ranking": ranking.to_dict(orient="records"),
                "full_stats": full_stats,
                "neutralization": ladder.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )

    # Save stitched panel lightly as parquet for reuse
    try:
        fac.to_parquet(OUT / "factor_panel.parquet")
    except Exception:
        fac.stack().to_csv(OUT / "factor_panel_long.csv", header=["value"])

    log(f"\nWrote {OUT}/")
    log(
        f"FULL {full_stats['label']}: RankIC={st['rank_ic']:.4f} "
        f"monthly={mon['monthly_rank_ic']:.4f} corr={best_corr:.3f}"
    )


if __name__ == "__main__":
    main()
