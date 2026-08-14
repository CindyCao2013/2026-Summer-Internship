"""Sprint 16 Phase C — SLOW-group daily-signal smoothing.

MA is applied to already-materialized daily factor exposure, never to the
L2 primitive. Order:

    factor_narrow (frozen)
    -> wide Date x Symbol exposure
    -> trailing MA_k (min_periods=k, no future bars)
    -> T+1 backtest_factor (mask, excess c2c, effective-direction)

RAW is window=1. Production TO<1 is not a Phase C gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Factor_Dev_Lib import calAnnuRet, calMDD, calSharpe
from l2_factor_reproduction.config.settings import UNIVERSE
from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.candidate_pool import yearly_ic_table
from l2_factor_reproduction.python.evaluation_protocol_v2 import (
    FEE_RATE_L1,
    fee_adjusted_group_pnl,
)
from l2_factor_reproduction.python.fast_discovery import (
    FULL_END,
    FULL_START,
    compute_fast_metrics,
    load_fast_context,
)
from l2_factor_reproduction.python.liquidity_impact_execution import (
    CONTRACT_VERSION,
    H1_PARITY_ATOL,
    MEDIUM_PARKED,
    OUT_ROOT,
    PHASE_B_SLOW_GRID,
    PHASE_C_IC_RETENTION_MIN,
    PHASE_C_MONO_MIN,
    PHASE_C_PROMOTE_MAX,
    PHASE_C_WINDOWS,
    PLOT_DPI,
    SLOW_FACTORS,
    contract_payload,
    load_baseline_h1,
    load_factor_wide,
    parity_ok,
)

VERSION_LABELS = {1: "RAW", 3: "MA3", 5: "MA5", 10: "MA10"}


def version_label(window: int) -> str:
    if window not in VERSION_LABELS:
        raise KeyError(f"unsupported smoothing window {window}")
    return VERSION_LABELS[window]


def trailing_mean_wide(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Per-symbol trailing mean on a Date x Symbol panel. No centered window."""
    if window < 1:
        raise ValueError("window must be >= 1")
    panel = wide.copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    if window == 1:
        return panel
    return panel.rolling(window=window, min_periods=window).mean()


def wide_to_narrow(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    panel = wide.copy()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "tradetime"
    stacked = panel.stack()
    if isinstance(stacked, pd.DataFrame):
        stacked = stacked.iloc[:, 0]
    frame = stacked.rename("value").reset_index()
    frame.columns = ["tradetime", "symbol", "value"]
    frame = frame.dropna(subset=["value"])
    frame["tradetime"] = pd.to_datetime(frame["tradetime"]) + pd.Timedelta(
        hours=9, minutes=30
    )
    frame["factorname"] = name
    frame["symbol"] = frame["symbol"].astype(str)
    return frame[["symbol", "tradetime", "factorname", "value"]].reset_index(
        drop=True
    )


def yearly_sign_consistency(rank_ic: pd.Series) -> Tuple[float, int, int]:
    table = yearly_ic_table(rank_ic)
    valid = table.loc[table["count"] >= 30]
    if valid.empty or rank_ic.dropna().empty:
        return float("nan"), 0, 0
    full_sign = float(np.sign(rank_ic.dropna().mean()))
    same = int((np.sign(valid["mean"]) == full_sign).sum())
    n_years = int(len(valid))
    return float(same / n_years), same, n_years


def _leg_stats(group_pnl: pd.DataFrame) -> Dict[str, float]:
    cols = sorted(
        (c for c in group_pnl.columns if str(c) != "H-L"),
        key=lambda c: int(c),
    )
    g1 = group_pnl[cols[0]]
    g10 = group_pnl[cols[-1]]
    g1_annu = float(calAnnuRet(g1))
    g10_annu = float(calAnnuRet(g10))
    denom = g10_annu - g1_annu
    long_share = float(g10_annu / denom) if abs(denom) > 1e-12 else float("nan")
    return {
        "g1_annu": g1_annu,
        "g10_annu": g10_annu,
        "g10_excess_sharpe": float(calSharpe(g10)),
        "g1_sharpe": float(calSharpe(g1)),
        "long_share_of_hl": long_share,
        "short_share_of_hl": (
            float(1.0 - long_share) if np.isfinite(long_share) else float("nan")
        ),
    }


def evaluate_smoothing_version(
    name: str,
    raw_wide: pd.DataFrame,
    window: int,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.Series]:
    smoothed = trailing_mean_wide(raw_wide, window)
    narrow = wide_to_narrow(smoothed, name)
    group_pnl, group_to, rank_ic, summary = backtest_factor(
        narrow,
        start_day=FULL_START,
        end_day=FULL_END,
        mask=mask,
        ret_matrix=ret,
        signal_shift=1,
    )
    fast = compute_fast_metrics(group_pnl, group_to, summary)
    net_pnl = fee_adjusted_group_pnl(group_pnl, group_to, fee_rate_l1=FEE_RATE_L1)
    hl_net = net_pnl["H-L"].dropna()
    net_mdd, _ = calMDD(hl_net)
    legs = _leg_stats(group_pnl)
    sign_frac, same_years, n_years = yearly_sign_consistency(rank_ic)
    to_l1 = float(group_to["H-L"].mean())
    row: Dict[str, object] = {
        "factor": name,
        "version": version_label(window),
        "ma_window": int(window),
        "rank_ic_raw": float(summary["rank_ic_mean_raw"]),
        "rank_ic_effective": float(summary["rank_ic_mean"]),
        "icir_effective": float(summary["rank_icir"]),
        "gross_hl_annu": float(summary["hl_annu_ret_flipped"]),
        "gross_hl_sharpe": float(summary["hl_sharpe_flipped"]),
        "gross_hl_mdd": float(summary["hl_mdd_flipped"]),
        "avg_hl_turnover_l1": to_l1,
        "implied_annu_fee": float(to_l1 * FEE_RATE_L1 * 250),
        "net_hl_annu": float(calAnnuRet(hl_net)),
        "net_hl_sharpe": float(calSharpe(hl_net)),
        "net_hl_mdd": float(net_mdd),
        "decile_mono_spearman": float(fast["decile_mono_spearman"]),
        "adjacent_violations": int(fast["adjacent_violations"]),
        "yearly_sign_consistency": sign_frac,
        "same_sign_years": same_years,
        "n_years": n_years,
        "n_days": int(summary["n_days"]),
        "n_names_avg": float(summary["n_names_avg"]),
        "factor_direction": int(summary["factor_direction"]),
        **legs,
    }
    return row, group_pnl, group_to, rank_ic


def attach_phase_c_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ic_retention"] = np.nan
    out["turnover_reduction"] = np.nan
    for _name, block in out.groupby("factor", sort=False):
        raw = block.loc[block["version"] == "RAW"]
        if raw.empty:
            continue
        raw_ic = abs(float(raw["rank_ic_effective"].iloc[0]))
        raw_to = float(raw["avg_hl_turnover_l1"].iloc[0])
        idx = block.index
        if raw_ic > 0:
            out.loc[idx, "ic_retention"] = block["rank_ic_effective"].abs() / raw_ic
        if raw_to > 0:
            out.loc[idx, "turnover_reduction"] = 1.0 - (
                block["avg_hl_turnover_l1"] / raw_to
            )
    return out


def promote_phase_c(frame: pd.DataFrame) -> pd.DataFrame:
    """At most two versions per factor: best NetSharpe, and lowest-TO frontier."""
    rows: List[Dict[str, object]] = []
    for name, block in frame.groupby("factor", sort=False):
        keep: Dict[str, str] = {}
        best_sr = block.loc[block["net_hl_sharpe"].idxmax()]
        keep[str(best_sr["version"])] = "A_net_sharpe"
        eligible = block.loc[
            (block["ic_retention"] >= PHASE_C_IC_RETENTION_MIN)
            & (block["decile_mono_spearman"] > PHASE_C_MONO_MIN)
        ]
        if not eligible.empty:
            cheapest = eligible.loc[eligible["avg_hl_turnover_l1"].idxmin()]
            label = str(cheapest["version"])
            if label in keep:
                keep[label] = "A_and_B"
            else:
                keep[label] = "B_turnover_frontier"
        if len(keep) > PHASE_C_PROMOTE_MAX:
            keep = dict(list(keep.items())[:PHASE_C_PROMOTE_MAX])
        for _, row in block.iterrows():
            version = str(row["version"])
            rows.append(
                {
                    "factor": name,
                    "version": version,
                    "decision": keep.get(version, "drop"),
                    "promote": version in keep,
                    "reason": keep.get(version, "not_selected"),
                }
            )
    return pd.DataFrame(rows)


def plot_phase_c_frontier(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    markers = {"RAW": "o", "MA3": "s", "MA5": "D", "MA10": "^"}
    for name, block in metrics.groupby("factor", sort=False):
        ordered = block.sort_values("ma_window")
        ax.plot(
            ordered["avg_hl_turnover_l1"],
            ordered["net_hl_sharpe"],
            linewidth=1.0,
            alpha=0.45,
        )
        for _, row in ordered.iterrows():
            ax.scatter(
                row["avg_hl_turnover_l1"],
                row["net_hl_sharpe"],
                marker=markers.get(str(row["version"]), "o"),
                s=70,
                label=f"{name} {row['version']}",
            )
    ax.axhline(1.5, color="0.6", linestyle="--", linewidth=0.8)
    ax.set_xlabel("H-L L1 turnover (daily)")
    ax.set_ylabel("Net H-L Sharpe (7.5 bp / L1)")
    ax.set_title(f"Phase C SLOW smoothing frontier — {UNIVERSE}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_retention_turnover(metrics: pd.DataFrame, path: Path) -> None:
    versions = ["RAW", "MA3", "MA5", "MA10"]
    factors = [name for name in SLOW_FACTORS if name in set(metrics["factor"])]
    x = np.arange(len(factors))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, version in enumerate(versions):
        block = metrics.loc[metrics["version"] == version].set_index("factor")
        ic = [
            float(block.loc[f, "ic_retention"]) if f in block.index else np.nan
            for f in factors
        ]
        to_red = [
            float(block.loc[f, "turnover_reduction"]) if f in block.index else np.nan
            for f in factors
        ]
        axes[0].bar(x + (i - 1.5) * width, ic, width=width, label=version)
        axes[1].bar(x + (i - 1.5) * width, to_red, width=width, label=version)
    axes[0].axhline(0.60, color="0.4", linestyle="--", linewidth=0.8)
    axes[0].set_title("IC retention vs RAW")
    axes[1].set_title("Turnover reduction vs RAW")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(factors, rotation=20, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|IC_smooth| / |IC_raw|")
    axes[1].set_ylabel("1 - TO_smooth / TO_raw")
    fig.suptitle(f"Phase C SLOW smoothing — {UNIVERSE}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_phase_c_report(
    metrics: pd.DataFrame,
    promotions: pd.DataFrame,
    parity: pd.DataFrame,
    path: Path,
) -> None:
    merged = metrics.merge(promotions, on=["factor", "version"], how="left")
    keep = merged.loc[merged["promote"] == True]  # noqa: E712
    show_cols = [
        "factor",
        "version",
        "rank_ic_effective",
        "ic_retention",
        "avg_hl_turnover_l1",
        "turnover_reduction",
        "gross_hl_sharpe",
        "net_hl_sharpe",
        "decile_mono_spearman",
        "yearly_sign_consistency",
        "decision",
    ]
    lines = [
        "# Sprint 16 Phase C — SLOW smoothing (daily signal MA)",
        "",
        f"Contract: `{CONTRACT_VERSION}`",
        "Layer: frozen daily exposure -> trailing MA_k -> T+1. Primitive formulas unchanged.",
        f"Factors: {', '.join(SLOW_FACTORS)}",
        f"Parked MEDIUM (not in this run): {', '.join(MEDIUM_PARKED)}",
        "Phase C does **not** apply the production TO<1 gate.",
        "",
        "## RAW parity vs candidate pool",
        "",
        "```",
        parity.to_string(index=False),
        "```",
        "",
        "## Metrics",
        "",
        "```",
        merged[show_cols].to_string(index=False),
        "```",
        "",
        "## Promotions (max 2 per factor)",
        "",
        "A = highest Net Sharpe. B = lowest L1 turnover among "
        f"ICRetention>={PHASE_C_IC_RETENTION_MIN:.0%} and mono>{PHASE_C_MONO_MIN}.",
        "",
        "```",
        keep.to_string(index=False) if not keep.empty else "(none)",
        "```",
        "",
        "## Phase B grid for promoted versions only",
        "",
        ", ".join(PHASE_B_SLOW_GRID),
        "",
        "R3 names remain one cluster. After B/D, pick by Net Sharpe + turnover + IC stability.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase_c(
    *,
    output_root: Optional[Path] = None,
    factors: Optional[Sequence[str]] = None,
    windows: Sequence[int] = PHASE_C_WINDOWS,
    verify_hash: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(output_root) if output_root is not None else OUT_ROOT / "phase_c"
    out.mkdir(parents=True, exist_ok=True)
    names = list(factors) if factors is not None else list(SLOW_FACTORS)
    unknown = sorted(set(names).difference(SLOW_FACTORS))
    if unknown:
        raise KeyError(f"Phase C is SLOW-only; refused: {unknown}")

    (out / "contract.json").write_text(
        json.dumps(contract_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    mask, ret = load_fast_context("full", verify_hash=verify_hash)
    rows: List[Dict[str, object]] = []
    parity_rows: List[Dict[str, object]] = []
    for name in names:
        wide = load_factor_wide(name)
        for window in windows:
            row, group_pnl, group_to, rank_ic = evaluate_smoothing_version(
                name, wide, int(window), mask, ret
            )
            rows.append(row)
            if window == 1:
                baseline = load_baseline_h1(name)
                observed = float(row["rank_ic_raw"])
                parity_rows.append(
                    {
                        "factor": name,
                        "rank_ic_raw": observed,
                        "candidate_pool_rank_ic_raw": baseline,
                        "abs_delta": abs(observed - baseline),
                        "parity_pass": parity_ok(
                            observed, baseline, atol=H1_PARITY_ATOL
                        ),
                    }
                )
            dest = out / "versions" / name / str(row["version"])
            dest.mkdir(parents=True, exist_ok=True)
            group_pnl.to_csv(dest / "group_pnl.csv")
            group_to.to_csv(dest / "group_to.csv")
            rank_ic.to_csv(dest / "rank_ic.csv")
            (dest / "summary.json").write_text(
                json.dumps(row, indent=2, default=float), encoding="utf-8"
            )

    metrics = attach_phase_c_ratios(pd.DataFrame(rows))
    promotions = promote_phase_c(metrics)
    parity = pd.DataFrame(parity_rows)
    metrics.to_csv(out / "smoothing_metrics.csv", index=False)
    promotions.to_csv(out / "promotions.csv", index=False)
    parity.to_csv(out / "raw_parity.csv", index=False)
    plot_phase_c_frontier(metrics, out / "net_sharpe_turnover_frontier.png")
    plot_retention_turnover(metrics, out / "ic_retention_turnover.png")
    write_phase_c_report(metrics, promotions, parity, out / "report.md")
    return metrics, promotions, parity
