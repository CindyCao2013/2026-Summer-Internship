#!/usr/bin/env python
"""Phase III A3 — ActiveTradeProxy (APM daily proxy) OS path.

Honest admission: NOT paper APM (needs minute/session).
Fills cutting-family slot with research_proxy label.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_active_trade_proxy.py --eval-days 252
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, decile_group_means, icir_from_daily
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from factor_attribution import cs_zscore
from factor_cutting.active_trade import compute_apm_overnight_day_proxy
from factor_data_loaders import load_eod_enriched_tables
from factor_report_generator_v2 import generate_pack
from run_milestone_1d7_pack_completion import run_d1_execution_grid

REPO = Path(__file__).resolve().parent
PACK = REPO / "research/reports/factors/ActiveTradeProxy"
EXEC_OUT = REPO / "research/reports/active_trade_proxy_v1/execution"
CACHE = REPO / "research/cache/active_trade_proxy_panels"
TOP_FRAC = 0.10
SIGNAL_SHIFT = 1
FID = "ActiveTradeProxy"


def log(msg: str) -> None:
    print(msg, flush=True)


def mono_score(sig_book: pd.DataFrame, ret: pd.DataFrame) -> float:
    dec = decile_group_means(sig_book, ret, n_groups=10, signal_shift=SIGNAL_SHIFT)
    vals = dec.dropna().values
    if len(vals) < 2:
        return float("nan")
    diffs = np.diff(vals)
    return float(np.mean(diffs > 0))


def build_pack_skeleton() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    for d in ("mechanism", "execution", "diagnostics", "artifacts", "charts", "figures"):
        (PACK / d).mkdir(exist_ok=True)


def register_factor(metrics: dict, best: dict) -> None:
    reg_path = REPO / "research/registry/factor_registry.yaml"
    data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    row = {
        "factor_id": FID,
        "display_name": "Active Trade Proxy (APM overnight−day)",
        "family": ["microstructure", "trading_behavior"],
        "source": "paper_proxy",
        "data_level": "EOD",
        "status": "testing",
        "formula_frozen": False,
        "production_ready": False,
        "validation_stage": "research_pack_complete",
        "benchmark": "Dual_Benchmark_v1",
        "pack_path": f"research/reports/factors/{FID}",
        "library_formula_id": "apm_overnight_day_proxy",
        "RankICIR": float(metrics["rank_icir"]) if metrics.get("rank_icir") is not None else None,
        "NetSharpe": float(best["net_sharpe"]) if best.get("net_sharpe") is not None else None,
        "Turnover": float(best["daily_turnover"]) if best.get("daily_turnover") is not None else None,
        "metric_note": (
            f"RESEARCH PROXY — not paper APM. RankICIR≈{float(metrics.get('rank_icir', float('nan'))):.2f}; "
            f"exec Net≈{float(best['net_sharpe']) if best.get('net_sharpe') is not None else float('nan'):.2f}; "
            f"mono≈{float(metrics.get('monotonicity', float('nan'))):.2f}. Stay testing."
        ),
        "correlation_cluster": None,
        "notes": (
            "Phase III A3. Paper ActiveTrade/APM needs minute/session. "
            "This is compute_apm_overnight_day_proxy only. Do not claim paper replication."
        ),
    }
    ids = {f["factor_id"] for f in data["factors"]}
    if FID in ids:
        data["factors"] = [row if f["factor_id"] == FID else f for f in data["factors"]]
    else:
        data["factors"].append(row)
    reg_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    csv_path = REPO / "research/registry/factor_registry.csv"
    df = pd.read_csv(csv_path)
    new = {c: None for c in df.columns}
    new.update(
        {
            "factor_id": FID,
            "display_name": "Active Trade Proxy (APM overnight-day)",
            "family": "microstructure|trading_behavior",
            "source": "paper_proxy",
            "data_level": "EOD",
            "status": "testing",
            "formula_frozen": False,
            "benchmark": "Dual_Benchmark_v1",
            "production_ready": False,
            "validation_stage": "research_pack_complete",
            "RankICIR": metrics.get("rank_icir"),
            "NetSharpe": best.get("net_sharpe"),
            "Turnover": best.get("daily_turnover"),
            "library_formula_id": "apm_overnight_day_proxy",
            "pack_path": f"research/reports/factors/{FID}",
            "notes": "Phase III A3 research_proxy; NOT paper APM",
        }
    )
    new = {c: new.get(c) for c in df.columns}
    if (df["factor_id"] == FID).any():
        for c, v in new.items():
            df.loc[df["factor_id"] == FID, c] = v
    else:
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    log("  Registry: ActiveTradeProxy added/updated as testing (research_proxy)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-days", type=int, default=252)
    parser.add_argument("--skip-generator", action="store_true")
    args = parser.parse_args()

    log("=== Phase III A3 ActiveTradeProxy (NOT paper APM) ===")
    log("Portfolio frozen. Honest daily proxy only.")

    build_pack_skeleton()
    EXEC_OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    close_full = enriched.close.loc[start:end]
    open_full = enriched.open.loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    warm = 40
    idx = close_full.index
    n = min(args.eval_days, len(idx))
    eval_index = idx[-n:]
    slice_start = idx[max(0, len(idx) - n - warm)]
    close = close_full.loc[slice_start:end]
    open_ = open_full.loc[slice_start:end]

    cache_tag = f"{close.index[0].date()}_{close.index[-1].date()}"
    cache_path = CACHE / f"active_trade_proxy_{cache_tag}.pkl"
    if cache_path.exists():
        log(f"Load cached {cache_path}")
        fac = pd.read_pickle(cache_path)
    else:
        log(f"Compute overnight−day proxy on {len(close)}d ...")
        fac = compute_apm_overnight_day_proxy(open_, close, window=20)
        fac.to_pickle(cache_path)

    fac = fac.reindex(index=eval_index)
    signal = cs_zscore(fac)
    ret = ret_full.reindex(index=signal.index, columns=signal.columns)

    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    direction = 1 if ic.mean() >= 0 else -1
    sig_book = signal * direction
    mono = mono_score(sig_book, ret)
    gross, to = daily_hl_pnl_and_turnover(
        sig_book, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    net = net_pnl_series(gross, to, DEFAULT_ROUND_TRIP_COST)
    perf_n = series_performance(net.dropna())
    metrics = {
        "window": f"last_{n}d",
        "rank_ic": float(ic.mean()),
        "rank_icir": float(icir_from_daily(ic)),
        "net_sharpe_plain": perf_n["sharpe"],
        "daily_turnover_plain": float(to.mean()),
        "monotonicity": mono,
        "direction_book": direction,
        "honesty": "research_proxy_not_paper_apm",
        "n_days": int(len(signal)),
    }
    (PACK / "artifacts" / "fresh_eval.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"  proxy: ICIR={metrics['rank_icir']:.2f} plainNet={metrics['net_sharpe_plain']:.2f} "
        f"mono={metrics['monotonicity']:.2f} dir={direction}"
    )

    # Protocol charts from this eval window (no cutting PNG harvest)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = PACK / "charts"
    figs = PACK / "figures"
    charts.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    ic_s = ic.dropna()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ic_s.index, ic_s.cumsum().values)
    ax.set_title("ActiveTradeProxy — cumulative RankIC (proxy)")
    ax.axhline(0, color="gray", lw=0.8)
    fig.tight_layout()
    for dest in (charts / "ic_curve.png", figs / "ic_curve.png", PACK / "ic_curve.png"):
        fig.savefig(dest, dpi=120)
    plt.close(fig)

    dec = decile_group_means(sig_book, ret, n_groups=10, signal_shift=SIGNAL_SHIFT)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dec.index.astype(str), dec.values)
    ax.set_title("Decile mean forward return (signed proxy)")
    fig.tight_layout()
    for dest in (charts / "decile_return.png", figs / "decile_return.png", PACK / "decile_return.png"):
        fig.savefig(dest, dpi=120)
    plt.close(fig)

    nav = (1 + net.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(nav.index, nav.values)
    ax.set_title("Cumulative net L/S (proxy, 15bp)")
    fig.tight_layout()
    for dest in (
        charts / "cumulative_long_short.png",
        figs / "cumulative_long_short.png",
        PACK / "cumulative_long_short.png",
    ):
        fig.savefig(dest, dpi=120)
    plt.close(fig)
    log("  wrote protocol charts (proxy window)")

    # Summary + mechanism
    summary = pd.DataFrame(
        [
            {
                "factor": FID,
                "period": f"last_{n}d",
                "universe": "ALL",
                "mode": "raw",
                "rank_ic": metrics["rank_ic"],
                "annu_ic": metrics["rank_ic"] * math.sqrt(250),
                "icir": metrics["rank_icir"],
                "hl_annu_ret": None,
                "hl_sharpe": None,
                "hl_mdd": None,
                "daily_turnover": metrics["daily_turnover_plain"],
                "implied_annu_fee": None,
                "net_sharpe": metrics["net_sharpe_plain"],
                "monotonicity": mono,
                "direction": direction,
            }
        ]
    )
    summary.to_csv(PACK / "factor_summary.csv", index=False)

    mech = pd.DataFrame(
        [
            {
                "signal": "overnight_minus_day_tstat",
                "category": "research_proxy",
                "rank_ic": metrics["rank_ic"],
                "icir": metrics["rank_icir"],
                "hl_sharpe": None,
                "net_sharpe": metrics["net_sharpe_plain"],
                "monotonicity": mono,
                "daily_turnover": metrics["daily_turnover_plain"],
            },
            {
                "signal": "paper_APM",
                "category": "stub_needs_minute",
                "rank_ic": None,
                "icir": None,
                "hl_sharpe": None,
                "net_sharpe": None,
                "monotonicity": None,
                "daily_turnover": None,
            },
        ]
    )
    mech.to_csv(PACK / "mechanism.csv", index=False)
    mech.to_csv(PACK / "mechanism_analysis.csv", index=False)
    shutil.copy2(PACK / "mechanism.csv", PACK / "mechanism" / "mechanism.csv")

    yearly = pd.DataFrame(
        [
            {
                "period": f"last_{n}d",
                "kind": "block",
                "n_days": n,
                "rank_ic": metrics["rank_ic"],
                "icir": metrics["rank_icir"],
                "pos_ic_frac": float((ic > 0).mean()),
            }
        ]
    )
    yearly.to_csv(PACK / "yearly_stability.csv", index=False)
    yearly.to_csv(PACK / "stability.csv", index=False)

    log("Execution grid ...")
    all_rows = run_d1_execution_grid(sig_book, ret, signal_mode="signed_proxy")
    ranked = all_rows.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False)
    ranked.to_csv(EXEC_OUT / "all_experiments.csv", index=False)
    ranked.to_csv(EXEC_OUT / "execution_summary.csv", index=False)
    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    baseline = {
        "factor": FID,
        "honesty": "research_proxy_not_paper_apm",
        "best_label": best.get("label"),
        "best_net_sharpe": best.get("net_sharpe"),
        "best_daily_turnover": best.get("daily_turnover"),
        "note": "Phase III A3 — proxy only",
    }
    (EXEC_OUT / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2, default=str) + "\n", encoding="utf-8"
    )
    shutil.copy2(EXEC_OUT / "execution_summary.csv", PACK / "execution_summary.csv")
    shutil.copy2(EXEC_OUT / "execution_summary.csv", PACK / "execution" / "execution_summary.csv")
    shutil.copy2(EXEC_OUT / "baseline_metrics.json", PACK / "artifacts" / "execution_baseline.json")

    if len(ranked):
        b = ranked.iloc[0]
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "factor": FID,
                            "period": f"last_{n}d",
                            "universe": "ALL",
                            "mode": "execution_best",
                            "rank_ic": b.get("rank_ic"),
                            "annu_ic": (
                                float(b["rank_ic"]) * math.sqrt(250)
                                if pd.notna(b.get("rank_ic"))
                                else np.nan
                            ),
                            "icir": b.get("icir"),
                            "hl_annu_ret": b.get("gross_annu_ret"),
                            "hl_sharpe": b.get("gross_sharpe"),
                            "hl_mdd": b.get("mdd_net"),
                            "daily_turnover": b.get("daily_turnover"),
                            "implied_annu_fee": b.get("implied_annu_fee"),
                            "net_sharpe": b.get("net_sharpe"),
                            "monotonicity": mono,
                            "direction": direction,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        summary.to_csv(PACK / "factor_summary.csv", index=False)

    register_factor(metrics, best)

    if not args.skip_generator:
        log("Report Generator v2 ...")
        result = generate_pack(FID)
        log(json.dumps(result.get("validation", {}), indent=2))

    md = REPO / "docs" / "milestone_3_0_active_trade_proxy.md"
    md.write_text(
        "\n".join(
            [
                "# Phase III A3 — ActiveTradeProxy",
                "",
                "**Honesty:** NOT paper APM / ActiveTrade. Daily overnight−day t-stat proxy only.",
                "",
                f"**RankICIR:** {metrics['rank_icir']:.2f}",
                f"**Plain Net Sharpe:** {metrics['net_sharpe_plain']:.2f}",
                f"**Mono:** {metrics['monotonicity']:.2f}",
                f"**Exec best:** {best.get('label')} Net≈{best.get('net_sharpe')}",
                "",
                "Registry status: `testing` (research_proxy).",
                "Pack: `research/reports/factors/ActiveTradeProxy/`",
                "",
                "Next: SmartMoney when minute ready, else III-B SUE. Portfolio remains frozen.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Wrote {md}")
    log("=== ActiveTradeProxy complete ===")


if __name__ == "__main__":
    main()
