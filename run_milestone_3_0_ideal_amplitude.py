#!/usr/bin/env python
"""Phase III A2 — IdealAmplitude full OS path (Alpha Library Expansion).

paper → harvest pack → eval/execution → Report Generator v2 → Registry

Constraints:
  - No portfolio / composite optimization
  - TGD/D1/Flow frozen
  - Admit as testing (mono soft-bar fail expected)

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_ideal_amplitude.py
  OMP_NUM_THREADS=1 python run_milestone_3_0_ideal_amplitude.py --eval-days 252
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
from factor_cutting.ideal_amplitude import compute_ideal_amplitude
from factor_data_loaders import load_eod_enriched_tables
from factor_report_generator_v2 import generate_pack
from run_milestone_1d7_pack_completion import run_d1_execution_grid

REPO = Path(__file__).resolve().parent
CUT_SRC = REPO / "research/reports/factor_cutting_v1/ideal_amplitude"
PACK = REPO / "research/reports/factors/IdealAmplitude"
EXEC_OUT = REPO / "research/reports/ideal_amplitude_v1/execution"
CACHE = REPO / "research/cache/ideal_amplitude_panels"
TOP_FRAC = 0.10
SIGNAL_SHIFT = 1


def log(msg: str) -> None:
    print(msg, flush=True)


def assemble_harvest_pack() -> dict:
    """Build Research Pack skeleton from cutting_v1 harvest (IdealReversal pattern)."""
    PACK.mkdir(parents=True, exist_ok=True)
    for d in ("mechanism", "execution", "diagnostics", "artifacts", "charts", "figures"):
        (PACK / d).mkdir(exist_ok=True)

    meta = json.loads((CUT_SRC / "summary.json").read_text(encoding="utf-8"))
    legs = pd.read_csv(CUT_SRC / "legs.csv")
    legs_map = {r["leg"]: r for _, r in legs.iterrows()}

    rank_ic = float(meta["rank_ic"])
    icir = float(meta["icir"])
    rows = [
        dict(
            factor="IdealAmplitude",
            period="cutting_v1_harvest",
            universe="ALL",
            mode="raw",
            rank_ic=rank_ic,
            annu_ic=rank_ic * math.sqrt(250),
            icir=icir,
            hl_annu_ret=float(meta["hl_annu_ret"]),
            hl_sharpe=float(meta["hl_sharpe"]),
            hl_mdd=None,
            daily_turnover=None,
            implied_annu_fee=None,
            net_sharpe=None,
            monotonicity=float(meta["monotonicity"]),
            direction=int(meta["direction"]),
        )
    ]
    neut_path = CUT_SRC / "robustness" / "neutralization.csv"
    if neut_path.exists():
        ndf = pd.read_csv(neut_path)
        for _, r in ndf.iterrows():
            mode = str(r["mode"]).lower()
            if mode not in ("raw", "size", "industry", "size_industry"):
                continue
            ric = float(r["rank_ic"])
            rows.append(
                dict(
                    factor="IdealAmplitude",
                    period="cutting_v1_harvest",
                    universe="ALL",
                    mode=mode,
                    rank_ic=ric,
                    annu_ic=ric * math.sqrt(250),
                    icir=float(r["icir"]),
                    hl_annu_ret=float(meta["hl_annu_ret"]),
                    hl_sharpe=float(meta["hl_sharpe"]),
                    hl_mdd=None,
                    daily_turnover=None,
                    implied_annu_fee=None,
                    net_sharpe=None,
                    monotonicity=float(meta["monotonicity"]),
                    direction=int(meta["direction"]),
                )
            )
    summary = pd.DataFrame(rows).drop_duplicates(subset=["mode"], keep="first")
    summary.to_csv(PACK / "factor_summary.csv", index=False)
    shutil.copy2(PACK / "factor_summary.csv", PACK / "artifacts" / "factor_summary.csv")

    hi, lo, sp = legs_map["high"], legs_map["low"], legs_map["spread"]
    mech = pd.DataFrame(
        [
            dict(
                signal="V_high",
                category="cutting_leg",
                rank_ic=hi["rank_ic"],
                icir=hi["icir"],
                hl_sharpe=None,
                net_sharpe=None,
                monotonicity=None,
                daily_turnover=None,
            ),
            dict(
                signal="V_low",
                category="cutting_leg",
                rank_ic=lo["rank_ic"],
                icir=lo["icir"],
                hl_sharpe=None,
                net_sharpe=None,
                monotonicity=None,
                daily_turnover=None,
            ),
            dict(
                signal="V_spread",
                category="cutting_output",
                rank_ic=sp["rank_ic"],
                icir=sp["icir"],
                hl_sharpe=meta["hl_sharpe"],
                net_sharpe=None,
                monotonicity=meta["monotonicity"],
                daily_turnover=None,
            ),
            dict(
                signal="Amp20_baseline",
                category="object",
                rank_ic=-0.0596,
                icir=-4.44,
                hl_sharpe=None,
                net_sharpe=None,
                monotonicity=None,
                daily_turnover=None,
            ),
        ]
    )
    mech.to_csv(PACK / "mechanism.csv", index=False)
    mech.to_csv(PACK / "mechanism_analysis.csv", index=False)
    shutil.copy2(PACK / "mechanism.csv", PACK / "mechanism" / "mechanism.csv")

    yearly = pd.DataFrame(
        [
            {
                "period": "full_sample",
                "kind": "block",
                "n_days": int(meta["n_days"]),
                "rank_ic": rank_ic,
                "icir": icir,
                "pos_ic_frac": float(meta["ic_pos_ratio"]),
            }
        ]
    )
    yearly.to_csv(PACK / "yearly_stability.csv", index=False)
    yearly.to_csv(PACK / "stability.csv", index=False)

    # Copy any PNGs if present
    chart_map = {
        CUT_SRC / "ic_analysis" / "rank_ic_timeseries.png": "ic_curve.png",
        CUT_SRC / "portfolio" / "decile_return.png": "decile_return.png",
        CUT_SRC / "portfolio" / "long_short_curve.png": "cumulative_long_short.png",
    }
    for s, name in chart_map.items():
        if s.exists():
            shutil.copy2(s, PACK / name)
            shutil.copy2(s, PACK / "figures" / name)
            shutil.copy2(s, PACK / "charts" / name)

    for p in (CUT_SRC / "mechanism").glob("*.png"):
        shutil.copy2(p, PACK / "figures" / p.name)

    for name in ("mechanism.md", "summary.md", "legs.csv"):
        s = CUT_SRC / name
        if s.exists():
            shutil.copy2(s, PACK / "mechanism" / name)

    return meta


def run_fresh_eval(eval_days: int) -> dict:
    """Compute IdealAmplitude + execution on last N days; attach to pack."""
    EXEC_OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    close_full = enriched.close.loc[start:end]
    high_full = enriched.high.loc[start:end]
    low_full = enriched.low.loc[start:end]
    open_full = enriched.open.loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    warm = 40
    idx = close_full.index
    n = min(eval_days, len(idx))
    eval_index = idx[-n:]
    slice_start = idx[max(0, len(idx) - n - warm)]
    close = close_full.loc[slice_start:end]
    high = high_full.loc[slice_start:end]
    low = low_full.loc[slice_start:end]
    open_ = open_full.loc[slice_start:end]

    cache_tag = f"{close.index[0].date()}_{close.index[-1].date()}"
    cache_path = CACHE / f"ideal_amplitude_{cache_tag}.pkl"
    if cache_path.exists():
        log(f"Load cached panel {cache_path}")
        fac = pd.read_pickle(cache_path)
    else:
        log(f"Compute IdealAmplitude on {len(close)}d ...")
        fac = compute_ideal_amplitude(high, low, close, open_=open_, return_legs=False)
        fac.to_pickle(cache_path)
        log(f"  cached → {cache_path}")

    fac = fac.reindex(index=eval_index)
    signal = cs_zscore(fac)
    ret = ret_full.reindex(index=signal.index, columns=signal.columns)

    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    gross, to = daily_hl_pnl_and_turnover(
        signal, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    # paper negative IC → direction for book
    direction = -1 if ic.mean() < 0 else 1
    sig_book = signal * direction
    gross_b, to_b = daily_hl_pnl_and_turnover(
        sig_book, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    net = net_pnl_series(gross_b, to_b, DEFAULT_ROUND_TRIP_COST)
    perf_n = series_performance(net.dropna())
    dec = decile_group_means(sig_book, ret, n_groups=10, signal_shift=SIGNAL_SHIFT)
    # monotonicity: fraction of adjacent decile steps with correct sign
    vals = dec.dropna().values
    if len(vals) >= 2:
        diffs = np.diff(vals)
        mono = float(np.mean(diffs > 0)) if direction > 0 else float(np.mean(diffs < 0))
    else:
        mono = np.nan

    fresh = {
        "window": f"last_{n}d",
        "rank_ic": float(ic.mean()),
        "rank_icir": float(icir_from_daily(ic)),
        "gross_sharpe": series_performance((gross_b if gross_b.mean() >= 0 else -gross_b).dropna())["sharpe"],
        "net_sharpe_plain": perf_n["sharpe"],
        "daily_turnover_plain": float(to_b.mean()),
        "monotonicity_proxy": mono,
        "direction_book": direction,
        "n_days": int(len(signal)),
    }
    (PACK / "artifacts" / "fresh_eval_lastNd.json").write_text(
        json.dumps(fresh, indent=2) + "\n", encoding="utf-8"
    )
    log(
        f"  fresh: ICIR={fresh['rank_icir']:.2f} plainNet={fresh['net_sharpe_plain']:.2f} "
        f"mono~{fresh['monotonicity_proxy']:.2f}"
    )

    log("Execution grid ...")
    # Use signed book for execution (positive alpha direction)
    all_rows = run_d1_execution_grid(sig_book, ret, signal_mode="signed_cs_z")
    ranked = all_rows.dropna(subset=["net_sharpe"]).sort_values("net_sharpe", ascending=False)
    ranked.to_csv(EXEC_OUT / "all_experiments.csv", index=False)
    ranked.to_csv(EXEC_OUT / "execution_summary.csv", index=False)
    best = ranked.iloc[0].to_dict() if len(ranked) else {}
    baseline = {
        "factor": "IdealAmplitude",
        "eval_window": f"last_{n}d",
        "cost_round_trip": DEFAULT_ROUND_TRIP_COST,
        "best_label": best.get("label"),
        "best_net_sharpe": best.get("net_sharpe"),
        "best_daily_turnover": best.get("daily_turnover"),
        "soft_bar": "fail_mono_stay_testing",
        "note": "Phase III A2 — formula not retuned; no portfolio",
    }
    (EXEC_OUT / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2, default=str) + "\n", encoding="utf-8"
    )
    shutil.copy2(EXEC_OUT / "execution_summary.csv", PACK / "execution_summary.csv")
    shutil.copy2(EXEC_OUT / "execution_summary.csv", PACK / "execution" / "execution_summary.csv")
    shutil.copy2(EXEC_OUT / "baseline_metrics.json", PACK / "artifacts" / "execution_baseline.json")

    # Append execution_best row to summary
    if len(ranked):
        df = pd.read_csv(PACK / "factor_summary.csv")
        df = df[df["mode"] != "execution_best"]
        b = ranked.iloc[0]
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "factor": "IdealAmplitude",
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
        df.to_csv(PACK / "factor_summary.csv", index=False)

    return {"fresh": fresh, "best": best, "meta_harvest_icir": None}


def register_factor(meta: dict, best: dict) -> None:
    reg_path = REPO / "research/registry/factor_registry.yaml"
    data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    ids = {f["factor_id"] for f in data["factors"]}
    row = {
        "factor_id": "IdealAmplitude",
        "display_name": "Ideal Amplitude",
        "family": ["microstructure", "volatility"],
        "source": "paper",
        "data_level": "EOD",
        "status": "testing",
        "formula_frozen": False,
        "production_ready": False,
        "validation_stage": "research_pack_complete",
        "benchmark": "Dual_Benchmark_v1",
        "pack_path": "research/reports/factors/IdealAmplitude",
        "library_formula_id": "IdealAmplitude",
        "RankICIR": float(meta["icir"]),
        "NetSharpe": float(best["net_sharpe"]) if best.get("net_sharpe") is not None else None,
        "Turnover": float(best["daily_turnover"]) if best.get("daily_turnover") is not None else None,
        "metric_note": (
            f"Cutting harvest RankICIR≈{meta['icir']:.2f} Sharpe≈{meta['hl_sharpe']:.2f} "
            f"mono≈{meta['monotonicity']:.2f} (soft-bar mono fail); "
            f"exec best Net≈{best.get('net_sharpe', float('nan')):.2f} — stay testing"
        ),
        "correlation_cluster": None,
        "notes": (
            "Phase III A2 paper replication. Cutting family with IdealReversal. "
            "Do not auto-admit; mono soft bar not passed."
        ),
    }
    if "IdealAmplitude" in ids:
        data["factors"] = [row if f["factor_id"] == "IdealAmplitude" else f for f in data["factors"]]
    else:
        data["factors"].append(row)
    reg_path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    csv_path = REPO / "research/registry/factor_registry.csv"
    df = pd.read_csv(csv_path)
    new = {
        "factor_id": "IdealAmplitude",
        "display_name": "Ideal Amplitude",
        "family": "microstructure|volatility",
        "source": "paper",
        "data_level": "EOD",
        "status": "testing",
        "formula_frozen": False,
        "benchmark": "Dual_Benchmark_v1",
        "production_ready": False,
        "validation_stage": "research_pack_complete",
        "RankICIR": meta["icir"],
        "NetSharpe": best.get("net_sharpe"),
        "Turnover": best.get("daily_turnover"),
        "library_formula_id": "IdealAmplitude",
        "pack_path": "research/reports/factors/IdealAmplitude",
        "notes": "Phase III A2; soft-bar mono fail; stay testing",
    }
    # align columns
    for c in df.columns:
        if c not in new:
            new[c] = None
    new = {c: new.get(c) for c in df.columns}
    if (df["factor_id"] == "IdealAmplitude").any():
        for c, v in new.items():
            df.loc[df["factor_id"] == "IdealAmplitude", c] = v
    else:
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    log("  Registry: IdealAmplitude added/updated as testing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-days", type=int, default=252)
    parser.add_argument("--skip-exec", action="store_true")
    parser.add_argument("--skip-generator", action="store_true")
    args = parser.parse_args()

    log("=== Phase III A2 IdealAmplitude — Alpha Library Expansion ===")
    log("Portfolio frozen. No composite optimization.")

    meta = assemble_harvest_pack()
    log(
        f"Harvest: RankICIR={meta['icir']:.2f} Sharpe={meta['hl_sharpe']:.2f} "
        f"mono={meta['monotonicity']:.2f}"
    )

    best = {}
    if not args.skip_exec:
        out = run_fresh_eval(args.eval_days)
        best = out["best"]
        log(
            f"Exec best: {best.get('label')} net={best.get('net_sharpe')} "
            f"TO={best.get('daily_turnover')}"
        )
    else:
        best = {"net_sharpe": None, "daily_turnover": None}

    register_factor(meta, best)

    if not args.skip_generator:
        log("Report Generator v2 ...")
        result = generate_pack("IdealAmplitude")
        log(json.dumps(result.get("validation", {}), indent=2))

    # milestone stub
    md = REPO / "docs" / "milestone_3_0_ideal_amplitude.md"
    md.write_text(
        "\n".join(
            [
                "# Phase III A2 — IdealAmplitude",
                "",
                "**Status:** admitted as `testing`",
                f"**Harvest RankICIR:** {meta['icir']:.2f}",
                f"**Harvest H-L Sharpe:** {meta['hl_sharpe']:.2f}",
                f"**Monotonicity:** {meta['monotonicity']:.2f} (soft-bar fail)",
                f"**Exec best:** {best.get('label')} Net≈{best.get('net_sharpe')}",
                "",
                "Pack: `research/reports/factors/IdealAmplitude/`",
                "Script: `run_milestone_3_0_ideal_amplitude.py`",
                "",
                "Next: ActiveTrade (Track A3). Portfolio remains frozen.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Wrote {md}")
    log("=== IdealAmplitude OS path complete ===")


if __name__ == "__main__":
    main()
