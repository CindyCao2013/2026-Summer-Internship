#!/usr/bin/env python
"""Factor Cutting baseline v1 — daily ideal_reversal / ideal_amplitude (+ APM proxy).

Stage: Phase-1 daily baseline (no minute factor zoo).
Gate: RankIC / ICIR / H-L + residual vs Base3 + stack λ (same harness family as SUE).

Usage:
  OMP_NUM_THREADS=1 /opt/conda/anaconda3/bin/python run_factor_cutting_baseline_v1.py
  OMP_NUM_THREADS=1 /opt/conda/anaconda3/bin/python run_factor_cutting_baseline_v1.py \\
      --sample-days 504 --knife-proxy
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
from factor_cutting.engine import knife_quantile_mechanism
from factor_cutting.ideal_reversal import avg_trade_amount, compute_ideal_reversal
from factor_cutting.registry import CUTTING_FACTOR_LIST, compute_cutting_factor
from factor_cutting.trade_count import load_trade_count_daily
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor

OUT = Path("research/reports/factor_cutting_baseline_v1")
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
        "pct_days_ge_200": float((n >= 200).mean()) if len(n) else 0.0,
    }


def turnover_proxy(panel: pd.DataFrame, n_groups: int = 10) -> float:
    sig = align_signal(panel, 1)
    turns = []
    prev = None
    for dt_ in sig.index:
        s = sig.loc[dt_].dropna()
        if len(s) < n_groups * 5:
            continue
        ranks = s.rank(pct=True)
        book = set(ranks[ranks >= 0.9].index) | set(ranks[ranks <= 0.1].index)
        if prev is not None and len(book | prev) > 0:
            turns.append(1.0 - len(book & prev) / max(len(book | prev), 1))
        prev = book
    if not turns:
        return float("nan")
    return float(np.mean(turns) * 250)


def build_base3(pv_cache, start, end) -> dict:
    return {name: build_eod_engine_factor(name, pv_cache).loc[start:end] for name in BASE3}


def evaluate_one(
    name: str,
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    base3_list: list,
    base3_combo: pd.DataFrame,
    *,
    knife_source: str = "",
) -> dict:
    panel = cs_zscore(panel.reindex(index=ret.index, columns=ret.columns))
    cov = coverage_stats(panel)
    ic_daily = daily_rank_ic_series(panel, ret)
    ic_mean = float(ic_daily.mean()) if len(ic_daily.dropna()) else np.nan
    icir = icir_from_daily(ic_daily)
    sharpe, ann, direction = hl_sharpe_from_composite(panel, ret)
    to = turnover_proxy(panel)

    resid = {"vs_base3_combo": residual_ic_stats(panel, ret, base3_combo)}
    for bname, bp in zip(BASE3, base3_list):
        resid[f"vs_{bname}"] = residual_ic_stats(panel, ret, bp)

    # stack with sign so positive-IC direction aligns with Base3
    stack_rows = []
    b = cs_zscore(base3_combo)
    signed = panel * (1 if direction >= 0 else -1)
    s = cs_zscore(signed)
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
        "knife_source": knife_source,
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
    parser.add_argument(
        "--knife-proxy",
        action="store_true",
        help="Force amount/volume ATS proxy (skip L2 trade_count load)",
    )
    parser.add_argument("--skip-mechanism", action="store_true")
    parser.add_argument("--refresh-trade-count", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    log("=== Factor Cutting Baseline v1 ===")
    log("Loading EOD enriched...")
    enriched, _ = load_eod_enriched_tables(preheat, end)
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    sample_start = max(0, len(ret_full) - args.sample_days)
    ret = ret_full.iloc[sample_start:]
    log(f"Ret sample: {ret.index[0].date()} -> {ret.index[-1].date()} ({len(ret)}d)")

    # Restrict compute window to sample + factor preheat (window=20 + buffer)
    warm_days = 60
    compute_start = ret.index[0] - pd.Timedelta(days=warm_days * 2)
    close = enriched.close.loc[compute_start:]
    open_ = enriched.open.loc[compute_start:]
    high = enriched.high.loc[compute_start:]
    low = enriched.low.loc[compute_start:]
    amount = enriched.amount.loc[compute_start:]
    volume = enriched.volume.loc[compute_start:]

    trade_count = None
    knife_source = "amount_per_volume_proxy"
    if not args.knife_proxy:
        try:
            log("Loading daily trade_count (L2 Active_buy/sell_count sum)...")
            tc_start = pd.Timestamp(compute_start).to_pydatetime()
            trade_count = load_trade_count_daily(
                tc_start, end, refresh_cache=args.refresh_trade_count
            )
            trade_count = trade_count.reindex(index=close.index, columns=close.columns)
            knife_source = "trade_count"
            log(f"  trade_count coverage mean names={trade_count.notna().sum(axis=1).mean():.0f}")
        except Exception as exc:
            log(f"WARNING: trade_count load failed ({exc}); falling back to amount/volume proxy")
            trade_count = None
            knife_source = "amount_per_volume_proxy"

    # Base3 still needs longer history — build from full enriched, then slice
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

    # classic Ret20 for comparison
    ret20 = -(close / close.shift(20) - 1.0)
    panels = {
        "reversal_20d_baseline": ret20,
    }
    meta_knife = {"reversal_20d_baseline": "n/a"}

    log("--- compute cutting factors ---")
    for name in CUTTING_FACTOR_LIST:
        log(f"  building {name}...")
        if name == "ideal_reversal":
            ret_1d = close / close.shift(1) - 1.0
            fac, _, _, src = compute_ideal_reversal(
                ret_1d,
                amount,
                trade_count=trade_count,
                volume=volume,
                return_legs=True,
            )
            panels[name] = fac
            meta_knife[name] = src
        else:
            panels[name] = compute_cutting_factor(
                name,
                close=close,
                open_=open_,
                high=high,
                low=low,
                amount=amount,
                volume=volume,
                trade_count=trade_count,
            )
            meta_knife[name] = knife_source if name == "ideal_reversal" else "n/a"

    rows = []
    for name, panel in panels.items():
        m = evaluate_one(
            name,
            panel.loc[start:end],
            ret,
            base3_list,
            base3_combo,
            knife_source=meta_knife.get(name, ""),
        )
        rows.append(m)
        rc = m["residual"]["vs_base3_combo"]
        log(
            f"  {name}: RankIC={m['rank_ic']:.4f} ICIR={m['icir']:.2f} "
            f"HL={m['gross_hl_sharpe']:.2f} resid_t={rc.get('residual_ic_t', np.nan):.2f} "
            f"uplift={m['stack_icir_uplift']:.3f} knife={m['knife_source']} -> {m['verdict']}"
        )

    flat = []
    for r in rows:
        rc = r["residual"]["vs_base3_combo"]
        flat.append(
            {
                "factor": r["factor"],
                "knife_source": r["knife_source"],
                "rank_ic": r["rank_ic"],
                "icir": r["icir"],
                "gross_hl_sharpe": r["gross_hl_sharpe"],
                "direction": r["direction"],
                "turnover_ann_proxy": r["turnover_ann_proxy"],
                "mean_names": r["coverage"]["mean_names"],
                "residual_ic_mean": rc.get("residual_ic_mean"),
                "residual_ic_t": rc.get("residual_ic_t"),
                "best_lambda": r["best_lambda"],
                "stack_icir_uplift": r["stack_icir_uplift"],
                "verdict": r["verdict"],
            }
        )
    summary = pd.DataFrame(flat).sort_values("factor")
    summary.to_csv(OUT / "factor_cutting_summary.csv", index=False)

    # Stage-3 mechanism: ATS knife vs future return + W-cut legs
    if not args.skip_mechanism:
        log("--- mechanism: ATS knife quantiles vs fwd return ---")
        ats, src = avg_trade_amount(amount, trade_count=trade_count, volume=volume)
        mech = knife_quantile_mechanism(ret, ats.reindex_like(ret), n_quantiles=10)
        mech.to_csv(OUT / "ideal_reversal_knife_mechanism.csv", index=False)
        log(f"  wrote knife mechanism ({src})")

        # Paper claim: M_high = strong reversal, M_low = weak momentum
        ret_1d = close / close.shift(1) - 1.0
        _, m_high, m_low, _ = compute_ideal_reversal(
            ret_1d, amount, trade_count=trade_count, volume=volume, return_legs=True
        )
        leg_rows = []
        for leg_name, leg in (("M_high", m_high), ("M_low", m_low), ("M", panels["ideal_reversal"])):
            p = leg.reindex(index=ret.index, columns=ret.columns)
            ic_d = daily_rank_ic_series(p, ret)
            leg_rows.append(
                {
                    "leg": leg_name,
                    "rank_ic": float(ic_d.mean()) if len(ic_d.dropna()) else np.nan,
                    "icir": icir_from_daily(ic_d),
                }
            )
            log(f"  leg {leg_name}: RankIC={leg_rows[-1]['rank_ic']:.4f} ICIR={leg_rows[-1]['icir']:.2f}")
        pd.DataFrame(leg_rows).to_csv(OUT / "ideal_reversal_legs_ic.csv", index=False)

    verdict = {
        "period": f"{ret.index[0].date()} -> {ret.index[-1].date()}",
        "sample_days": int(len(ret)),
        "phase": "factor_cutting_baseline_v1",
        "base3": BASE3,
        "factors": list(panels.keys()),
        "knife_policy": knife_source,
        "note": (
            "Daily baseline only. APM paper + Smart Money stubbed. "
            "ideal_reversal knife prefers L2 daily trade_count; --knife-proxy uses amount/volume."
        ),
        "rows": rows,
        "schema": "research/factor_cutting/factor_definition.yaml",
    }
    (OUT / "factor_cutting_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    log(f"\nWrote {OUT / 'factor_cutting_summary.csv'}")
    log(f"Wrote {OUT / 'factor_cutting_verdict.json'}")


if __name__ == "__main__":
    main()
