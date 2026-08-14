#!/usr/bin/env python3
"""Scaffold / refresh research_delivery factor cards from Pack artifacts.

Does NOT run backtests. Only packages existing experiment PNGs + summary metrics.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "research_delivery"
FACTORS = DELIVERY / "factors"
PACK_ROOT = ROOT / "research" / "reports" / "factors"

# factor_id -> plot sources relative to repo (or pack)
PLOT_MAP = {
    "TGD20": {
        "pack": PACK_ROOT / "TGD20",
        "plots": {
            "ic_curve.png": "ic_analysis/ic_curve.png",
            "decile_return.png": "quantile_analysis/decile_return.png",
            "cumulative_long_short.png": "quantile_analysis/cumulative_long_short.png",
            "stability_yearly.png": "stability/stability_yearly.png",
            "turnover.png": "execution/turnover.png",
        },
    },
    "D1_LiquidityQuality60d": {
        "pack": PACK_ROOT / "D1_LiquidityQuality60d",
        "plots": {
            "ic_curve.png": "ic_analysis/ic_curve.png",
            "decile_return.png": "quantile_analysis/decile_return.png",
            "cumulative_long_short.png": "quantile_analysis/cumulative_long_short.png",
            "stability_yearly.png": "stability/stability_yearly.png",
            "turnover.png": "execution/turnover.png",
        },
    },
    "FlowDensity20": {
        "pack": PACK_ROOT / "FlowDensity20",
        "plots": {
            "ic_curve.png": "ic_analysis/ic_curve.png",
            "decile_return.png": "quantile_analysis/decile_return.png",
            "cumulative_long_short.png": "quantile_analysis/cumulative_long_short.png",
            "stability_yearly.png": "stability/stability_yearly.png",
            "turnover.png": "execution/turnover.png",
        },
    },
    "APM_SessionResidual": {
        "pack": PACK_ROOT / "APM_SessionResidual",
        "plots": {
            "ic_curve.png": "ic_analysis/ic_curve.png",
            "decile_return.png": "quantile_analysis/decile_return.png",
            "stability_yearly.png": "stability/stability_yearly.png",
            "turnover.png": "execution/turnover_curve.png",
        },
    },
    "IdealReversal": {
        "pack": PACK_ROOT / "IdealReversal",
        "plots": {
            "ic_curve.png": "ic_analysis/ic_curve.png",
            "decile_return.png": "quantile_analysis/decile_return.png",
            "cumulative_long_short.png": "quantile_analysis/cumulative_long_short.png",
            "stability_yearly.png": "stability/stability_yearly.png",
            "turnover.png": "execution/turnover.png",
        },
    },
    "IdealAmplitude": {
        "pack": PACK_ROOT / "IdealAmplitude",
        "plots": {
            "ic_curve.png": "ic_analysis/ic_curve.png",
            "decile_return.png": "quantile_analysis/decile_return.png",
            "cumulative_long_short.png": "quantile_analysis/cumulative_long_short.png",
            "stability_yearly.png": "stability/stability_yearly.png",
            "turnover.png": "execution/turnover.png",
        },
    },
    "SmartMoney10d": {
        "pack": ROOT / "research" / "reports" / "smart_money_v1" / "phase2a",
        "plots": {
            "decile_return.png": "figures/decile_return.png",
            "stability_yearly.png": "figures/stability_yearly.png",
        },
    },
    "AmihudShockReversal5d": {
        "pack": ROOT
        / "research"
        / "reports"
        / "d1_liquidity_density_v1"
        / "confirmation_1455d"
        / "amihud_shock_reversal_5d",
        "plots": {
            "ic_curve.png": "figures/ic_timeseries.png",
            "decile_return.png": "figures/quantile_return.png",
            "cumulative_long_short.png": "figures/cumulative_long_short.png",
            "ic_decay.png": "figures/ic_decay.png",
        },
    },
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def sync_plots(fid: str) -> list[str]:
    cfg = PLOT_MAP[fid]
    pack: Path = cfg["pack"]
    out = FACTORS / fid / "plots"
    missing = []
    for name, rel in cfg["plots"].items():
        src = pack / rel
        if not src.exists():
            missing.append(str(src))
            continue
        link_or_copy(src, out / name)
    return missing


def load_summary(fid: str) -> dict:
    pack = PLOT_MAP[fid]["pack"]
    yml = pack / "summary.yaml"
    if yml.exists():
        return yaml.safe_load(yml.read_text()) or {}
    return {}


def write_metrics_csv(fid: str, summary: dict) -> None:
    vh = summary.get("validation_headline") or {}
    path = FACTORS / fid / "metrics.csv"
    rows = [
        ("factor_id", fid),
        ("status", summary.get("status", "")),
        ("family", summary.get("family", "")),
        ("RankIC_raw", vh.get("RankIC_raw", "")),
        ("RankIC_size_industry", vh.get("RankIC_size_industry", "")),
        ("ICIR_raw", vh.get("ICIR_raw", "")),
        ("ICIR_size_industry", vh.get("ICIR_size_industry", "")),
        ("monotonicity", vh.get("monotonicity", "")),
        ("execution_best_net_Sharpe", vh.get("execution_best_net_Sharpe", "")),
        ("execution_best_daily_turnover", vh.get("execution_best_daily_turnover", "")),
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def main() -> None:
    missing_all = []
    for fid in PLOT_MAP:
        (FACTORS / fid).mkdir(parents=True, exist_ok=True)
        missing = sync_plots(fid)
        if missing:
            missing_all.extend(missing)
        summary = load_summary(fid)
        if summary:
            write_metrics_csv(fid, summary)
        # formula pointer
        pack = PLOT_MAP[fid]["pack"]
        formula = pack / "formula.md"
        dst = FACTORS / fid / "formula.md"
        if formula.exists():
            link_or_copy(formula, dst)
        print(f"[ok] plots synced: {fid} (missing={len(missing)})")
    if missing_all:
        print("MISSING plot sources:")
        for m in missing_all:
            print(" ", m)


if __name__ == "__main__":
    main()
