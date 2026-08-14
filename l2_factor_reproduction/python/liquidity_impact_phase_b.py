"""Sprint 16 Phase B — staggered holding on frozen Phase C versions.

Each day t, rank the *current* signal S_t into a new equal-weight decile
sleeve P_t and hold that sleeve for H trading days. The book is:

    w_t = (1/H) * sum_{j=0..H-1} w^{sleeve}_{t-j}

Turnover is L1 on this netted weight, not the sum / average of sleeve TO.

This is not signal smoothing: past signals are never averaged before rank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Factor_Dev_Lib import _rank_to_bins_npqcut, calAnnuRet, calMDD, calSharpe
from l2_factor_reproduction.config.settings import N_GROUPS, UNIVERSE
from l2_factor_reproduction.python.backtest import (
    compute_rank_ic,
    prepare_factor_signal,
)
from l2_factor_reproduction.python.evaluation_protocol_v2 import FEE_RATE_L1
from l2_factor_reproduction.python.fast_discovery import (
    FULL_END,
    FULL_START,
    load_fast_context,
)
from l2_factor_reproduction.python.liquidity_impact_execution import (
    CONTRACT_VERSION,
    FEATURE_GATE,
    H1_PARITY_ATOL,
    NET_GATE,
    OUT_ROOT,
    PHASE_B_CANDIDATES,
    PHASE_B_HOLDS,
    PHASE_D_BUFFER_WIDTHS,
    PLOT_DPI,
    POOL_FACTOR_ROOT,
    R3_PHASE_B,
    contract_payload,
    load_baseline_h1,
    load_factor_wide,
    parity_ok,
)
from l2_factor_reproduction.python.liquidity_impact_phase_c import (
    trailing_mean_wide,
    version_label,
    yearly_sign_consistency,
)

HOLD_LABELS = {1: "daily", 5: "staggered_5d", 10: "staggered_10d"}
PHASE_C_METRICS = OUT_ROOT / "phase_c" / "smoothing_metrics.csv"


def hold_label(hold: int) -> str:
    if hold not in HOLD_LABELS:
        raise KeyError(f"unsupported hold {hold}; frozen grid is {PHASE_B_HOLDS}")
    return HOLD_LABELS[hold]


def load_frozen_direction(name: str) -> int:
    if PHASE_C_METRICS.exists():
        frame = pd.read_csv(PHASE_C_METRICS)
        block = frame.loc[
            (frame["factor"] == name) & (frame["version"] == "RAW")
        ]
        if not block.empty:
            return int(block["factor_direction"].iloc[0])
    payload = json.loads(
        (POOL_FACTOR_ROOT / name / "summary.json").read_text(encoding="utf-8")
    )
    return int(payload["factor_direction"])


def decile_membership_from_rank(
    rank: pd.DataFrame, group: int
) -> pd.DataFrame:
    mask = rank.eq(float(group))
    count = mask.sum(axis=1).replace(0, np.nan)
    return mask.div(count, axis=0)


def decile_membership_weights(
    signal: pd.DataFrame, group: int, n_groups: int = N_GROUPS
) -> pd.DataFrame:
    rank = _rank_to_bins_npqcut(signal, n_groups)
    return decile_membership_from_rank(rank, group)


def hl_signed_weights(
    signal: pd.DataFrame, n_groups: int = N_GROUPS
) -> pd.DataFrame:
    rank = _rank_to_bins_npqcut(signal, n_groups)
    long_w = decile_membership_from_rank(rank, n_groups)
    short_w = decile_membership_from_rank(rank, 1)
    return long_w.fillna(0.0) - short_w.fillna(0.0)


def daily_sleeves(
    signal: pd.DataFrame, n_groups: int = N_GROUPS
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    rank = _rank_to_bins_npqcut(signal, n_groups)
    groups = {
        group: decile_membership_from_rank(rank, group)
        for group in range(1, n_groups + 1)
    }
    hl = groups[n_groups].fillna(0.0) - groups[1].fillna(0.0)
    return groups, hl


def stagger_weights(daily_w: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Average the last `hold` daily sleeve snapshots. No signal re-rank."""
    if hold < 1:
        raise ValueError("hold must be >= 1")
    filled = daily_w.fillna(0.0).sort_index()
    if hold == 1:
        return filled
    return filled.rolling(window=hold, min_periods=hold).mean()


def l1_turnover(weights: pd.DataFrame) -> pd.Series:
    filled = weights.fillna(0.0)
    turnover = filled.diff().abs().sum(axis=1)
    if len(filled):
        turnover.iloc[0] = filled.iloc[0].abs().sum()
    return turnover


def book_pnl(weights: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    aligned = ret.reindex(index=weights.index, columns=weights.columns)
    return weights.mul(aligned).sum(axis=1)


def economic_strategy_pass(row: Dict[str, object]) -> bool:
    return (
        float(row["net_hl_annu"]) > NET_GATE["net_annu_min"]
        and float(row["net_hl_sharpe"]) > NET_GATE["net_sharpe_min"]
        and float(row["avg_hl_turnover_l1"]) < NET_GATE["turnover_l1_max"]
        and float(row["ic_retention"]) >= NET_GATE["ic_retention_min"]
    )


def feature_pass(row: Dict[str, object]) -> bool:
    return (
        abs(float(row["rank_ic_effective"])) >= FEATURE_GATE["abs_rank_ic_min"]
        and abs(float(row["icir_effective"])) >= FEATURE_GATE["abs_icir_min"]
        and float(row["decile_mono_spearman"]) > FEATURE_GATE["monotonicity_min"]
        and float(row["yearly_sign_consistency"]) >= FEATURE_GATE["yearly_sign_min"]
    )


def assign_grade(row: Dict[str, object]) -> str:
    strategy = economic_strategy_pass(row) and (
        float(row["decile_mono_spearman"]) > NET_GATE["monotonicity_min"]
    )
    if strategy:
        return "strategy_grade"
    if feature_pass(row):
        return "feature_grade"
    return "neither"


def _best_hold_per_factor(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _name, block in frame.groupby("factor", sort=False):
        rows.append(block.loc[block["net_hl_sharpe"].idxmax()])
    return pd.DataFrame(rows).reset_index(drop=True)


def _dominates(a: pd.Series, b: pd.Series) -> bool:
    weakly = (
        float(a["net_hl_sharpe"]) >= float(b["net_hl_sharpe"])
        and float(a["net_hl_annu"]) >= float(b["net_hl_annu"])
        and float(a["avg_hl_turnover_l1"]) <= float(b["avg_hl_turnover_l1"])
        and float(a["net_hl_mdd"]) >= float(b["net_hl_mdd"])
        and float(a["yearly_sign_consistency"])
        >= float(b["yearly_sign_consistency"])
    )
    strictly = (
        float(a["net_hl_sharpe"]) > float(b["net_hl_sharpe"])
        or float(a["net_hl_annu"]) > float(b["net_hl_annu"])
        or float(a["avg_hl_turnover_l1"]) < float(b["avg_hl_turnover_l1"])
        or float(a["net_hl_mdd"]) > float(b["net_hl_mdd"])
        or float(a["yearly_sign_consistency"])
        > float(b["yearly_sign_consistency"])
    )
    return weakly and strictly


def pick_r3(metrics: pd.DataFrame) -> Dict[str, object]:
    r3 = metrics.loc[metrics["factor"].isin(R3_PHASE_B)].copy()
    if r3.empty:
        return {"keep": [], "drop": list(R3_PHASE_B), "reason": "no_r3_rows"}
    best = _best_hold_per_factor(r3)
    names = [str(x) for x in best["factor"]]
    if len(best) == 1:
        return {
            "keep": names,
            "drop": [n for n in R3_PHASE_B if n not in names],
            "reason": "single_r3_row",
            "best_rows": best.to_dict(orient="records"),
        }
    for _, a in best.iterrows():
        others = [b for _, b in best.iterrows() if a["factor"] != b["factor"]]
        if others and all(_dominates(a, b) for b in others):
            keep = [str(a["factor"])]
            return {
                "keep": keep,
                "drop": [n for n in names if n not in keep],
                "reason": "pareto_dominance",
                "best_rows": best.to_dict(orient="records"),
            }
    dominated = set()
    for _, a in best.iterrows():
        for _, b in best.iterrows():
            if a["factor"] != b["factor"] and _dominates(a, b):
                dominated.add(str(b["factor"]))
    survivors = [n for n in names if n not in dominated]
    if len(survivors) > 2:
        ordered = (
            best.set_index("factor")
            .loc[survivors]
            .sort_values("net_hl_sharpe", ascending=False)
        )
        keep = [str(x) for x in ordered.index[:2]]
        reason = "no_strict_dominance_keep_top2_net_sharpe"
    else:
        keep = survivors
        reason = "pareto_survivors" if dominated else "no_strict_dominance"
    return {
        "keep": keep,
        "drop": [n for n in names if n not in keep],
        "reason": reason,
        "best_rows": best.to_dict(orient="records"),
    }


def research_r3_keep(selection: Dict[str, object]) -> Dict[str, object]:
    """Collapse ρ≈0.92 aliases when one name is clearly better economically.

    Mechanical Pareto can keep two names because of a 0.02 TO gap. Phase D
    should not pay that tax.
    """
    keep = [str(x) for x in selection.get("keep") or []]
    rows = selection.get("best_rows") or []
    if len(keep) <= 1 or not rows:
        selection["research_keep"] = keep
        selection["research_reason"] = selection.get("reason")
        return selection
    block = pd.DataFrame(rows)
    block = block.loc[block["factor"].isin(keep)].sort_values(
        "net_hl_sharpe", ascending=False
    )
    top = block.iloc[0]
    second = block.iloc[1]
    sr_gap = float(top["net_hl_sharpe"]) - float(second["net_hl_sharpe"])
    to_gap = abs(
        float(top["avg_hl_turnover_l1"]) - float(second["avg_hl_turnover_l1"])
    )
    if sr_gap >= 0.30 and to_gap <= 0.10:
        selection["research_keep"] = [str(top["factor"])]
        selection["research_reason"] = (
            f"alias_collapse: NetSR gap {sr_gap:.2f}, TO gap {to_gap:.3f}; "
            "do not take a rho~0.92 duplicate into Phase D"
        )
    else:
        selection["research_keep"] = keep[:2]
        selection["research_reason"] = "keep_pareto_max2"
    return selection


def judge_depth_recovery(metrics: pd.DataFrame) -> Dict[str, object]:
    block = metrics.loc[metrics["factor"] == "depth_recovery_5m"].copy()
    if block.empty:
        return {"verdict": "missing", "phase_d": False}
    block = block.sort_values("hold_days")
    daily = block.loc[block["hold_days"] == 1]
    if daily.empty:
        daily = block.iloc[[0]]
    daily_row = daily.iloc[0]
    best = block.loc[block["gross_hl_sharpe"].idxmax()]
    sharpes = [float(x) for x in block["gross_hl_sharpe"]]
    declining = all(sharpes[i] >= sharpes[i + 1] for i in range(len(sharpes) - 1))
    daily_gs = float(daily_row["gross_hl_sharpe"])
    best_gs = float(best["gross_hl_sharpe"])
    if str(best["grade"]) == "strategy_grade" or str(daily_row["grade"]) == "strategy_grade":
        verdict = "phase_d_strategy_candidate"
        phase_d = True
    elif best_gs >= daily_gs + 0.20:
        verdict = "staggered_helps_keep_standalone_hope"
        phase_d = True
    elif declining or best_gs < 1.0:
        verdict = "feature_grade_only_stop_rescuing_standalone"
        phase_d = False
    else:
        verdict = "inspect"
        phase_d = False
    return {
        "verdict": verdict,
        "phase_d": phase_d,
        "daily_gross_sharpe": daily_gs,
        "best_hold": str(best["hold_label"]),
        "best_gross_sharpe": best_gs,
        "best_net_sharpe": float(best["net_hl_sharpe"]),
        "grade_daily": str(daily_row["grade"]),
        "grade_best": str(best["grade"]),
    }


@dataclass(frozen=True)
class PreparedCandidate:
    name: str
    ma_window: int
    version: str
    direction: int
    signal_eff: pd.DataFrame
    ret: pd.DataFrame
    daily_groups: Dict[int, pd.DataFrame]
    daily_hl: pd.DataFrame
    rank_ic_raw: pd.Series
    rank_ic_eff: pd.Series
    raw_ic_abs: float


def prepare_candidate(
    name: str,
    ma_window: int,
    raw_wide: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
) -> PreparedCandidate:
    smoothed = trailing_mean_wide(raw_wide, ma_window)
    signal_raw, aligned_ret = prepare_factor_signal(
        smoothed,
        start=FULL_START,
        end=FULL_END,
        mask=mask,
        signal_shift=1,
        ret_matrix=ret,
    )
    direction = load_frozen_direction(name)
    signal_eff = signal_raw * float(direction)
    daily_groups, daily_hl = daily_sleeves(signal_eff)
    return PreparedCandidate(
        name=name,
        ma_window=int(ma_window),
        version=version_label(ma_window),
        direction=int(direction),
        signal_eff=signal_eff,
        ret=aligned_ret,
        daily_groups=daily_groups,
        daily_hl=daily_hl,
        rank_ic_raw=compute_rank_ic(signal_raw, aligned_ret),
        rank_ic_eff=compute_rank_ic(signal_eff, aligned_ret),
        raw_ic_abs=abs(load_baseline_h1(name)),
    )


def evaluate_hold(
    prepared: PreparedCandidate, hold: int
) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series, pd.Series]:
    group_w = {
        group: stagger_weights(w, hold)
        for group, w in prepared.daily_groups.items()
    }
    hl_w = stagger_weights(prepared.daily_hl, hold)
    valid = hl_w.notna().any(axis=1) & (hl_w.abs().sum(axis=1) > 0)
    hl_w = hl_w.loc[valid]
    aligned_ret = prepared.ret.reindex(index=hl_w.index, columns=hl_w.columns)

    group_pnl = pd.DataFrame(index=hl_w.index)
    for group, w in group_w.items():
        group_pnl[str(group)] = book_pnl(w.reindex_like(hl_w), aligned_ret)
    hl_pnl = book_pnl(hl_w, aligned_ret)
    group_pnl["H-L"] = hl_pnl
    to = l1_turnover(hl_w).reindex(hl_pnl.index)
    net_pnl = hl_pnl - to.fillna(0.0) * FEE_RATE_L1

    ranks = pd.Series(np.arange(1, N_GROUPS + 1), dtype=float)
    decile_annu = np.array(
        [calAnnuRet(group_pnl[str(g)]) for g in range(1, N_GROUPS + 1)],
        dtype=float,
    )
    mono = float(ranks.corr(pd.Series(decile_annu), method="spearman"))
    violations = int(np.sum(decile_annu[1:] < decile_annu[:-1]))
    sign_frac, same_years, n_years = yearly_sign_consistency(prepared.rank_ic_eff)
    mdd_g, _ = calMDD(hl_pnl)
    mdd_n, _ = calMDD(net_pnl)
    ic_mean = float(prepared.rank_ic_eff.mean())
    ic_std = float(prepared.rank_ic_eff.std())
    icir = ic_mean / ic_std * (250 ** 0.5) if ic_std > 0 else float("nan")
    g1 = group_pnl["1"]
    g10 = group_pnl[str(N_GROUPS)]
    g1_annu = float(calAnnuRet(g1))
    g10_annu = float(calAnnuRet(g10))
    denom = g10_annu - g1_annu
    long_share = float(g10_annu / denom) if abs(denom) > 1e-12 else float("nan")
    to_mean = float(to.mean())
    ic_ret = (
        abs(ic_mean) / prepared.raw_ic_abs
        if prepared.raw_ic_abs and prepared.raw_ic_abs > 0
        else float("nan")
    )
    row: Dict[str, object] = {
        "factor": prepared.name,
        "version": prepared.version,
        "ma_window": int(prepared.ma_window),
        "hold_days": int(hold),
        "hold_label": hold_label(hold),
        "factor_direction": int(prepared.direction),
        "rank_ic_raw": float(prepared.rank_ic_raw.mean()),
        "rank_ic_effective": ic_mean,
        "icir_effective": float(icir),
        "ic_retention": float(ic_ret),
        "gross_hl_annu": float(calAnnuRet(hl_pnl)),
        "gross_hl_sharpe": float(calSharpe(hl_pnl)),
        "gross_hl_mdd": float(mdd_g),
        "avg_hl_turnover_l1": to_mean,
        "implied_annu_fee": float(to_mean * FEE_RATE_L1 * 250),
        "net_hl_annu": float(calAnnuRet(net_pnl)),
        "net_hl_sharpe": float(calSharpe(net_pnl)),
        "net_hl_mdd": float(mdd_n),
        "decile_mono_spearman": mono,
        "adjacent_violations": violations,
        "yearly_sign_consistency": sign_frac,
        "same_sign_years": same_years,
        "n_years": n_years,
        "n_days": int(len(hl_pnl.dropna())),
        "n_names_avg": float(prepared.signal_eff.notna().sum(axis=1).mean()),
        "g1_annu": g1_annu,
        "g10_annu": g10_annu,
        "g10_excess_sharpe": float(calSharpe(g10)),
        "long_share_of_hl": long_share,
        "short_share_of_hl": (
            float(1.0 - long_share) if np.isfinite(long_share) else float("nan")
        ),
        "family": "R3" if prepared.name in R3_PHASE_B else "depth_recovery",
    }
    row["economic_strategy_pass"] = economic_strategy_pass(row)
    row["feature_pass"] = feature_pass(row)
    row["grade"] = assign_grade(row)
    return row, group_pnl, to, net_pnl


def h1_parity_row(
    row: Dict[str, object], phase_c: Optional[pd.DataFrame]
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "factor": row["factor"],
        "version": row["version"],
        "phase_b_net_sharpe": row["net_hl_sharpe"],
        "phase_b_turnover": row["avg_hl_turnover_l1"],
        "phase_b_gross_sharpe": row["gross_hl_sharpe"],
        "phase_c_net_sharpe": float("nan"),
        "phase_c_turnover": float("nan"),
        "phase_c_gross_sharpe": float("nan"),
        "parity_pass": False,
    }
    if phase_c is None or phase_c.empty:
        return out
    match = phase_c.loc[
        (phase_c["factor"] == row["factor"])
        & (phase_c["version"] == row["version"])
    ]
    if match.empty:
        return out
    c = match.iloc[0]
    out["phase_c_net_sharpe"] = float(c["net_hl_sharpe"])
    out["phase_c_turnover"] = float(c["avg_hl_turnover_l1"])
    out["phase_c_gross_sharpe"] = float(c["gross_hl_sharpe"])
    out["abs_delta_net_sharpe"] = abs(
        float(row["net_hl_sharpe"]) - float(c["net_hl_sharpe"])
    )
    out["abs_delta_turnover"] = abs(
        float(row["avg_hl_turnover_l1"]) - float(c["avg_hl_turnover_l1"])
    )
    out["parity_pass"] = bool(
        parity_ok(
            float(row["gross_hl_sharpe"]),
            float(c["gross_hl_sharpe"]),
            atol=max(H1_PARITY_ATOL, 5e-3),
        )
        and parity_ok(
            float(row["avg_hl_turnover_l1"]),
            float(c["avg_hl_turnover_l1"]),
            atol=max(H1_PARITY_ATOL, 5e-3),
        )
    )
    return out


def plot_phase_b_frontier(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    markers = {"daily": "o", "staggered_5d": "s", "staggered_10d": "^"}
    for name, block in metrics.groupby("factor", sort=False):
        ordered = block.sort_values("hold_days")
        ax.plot(
            ordered["avg_hl_turnover_l1"],
            ordered["net_hl_sharpe"],
            linewidth=1.0,
            alpha=0.5,
        )
        for _, row in ordered.iterrows():
            ax.scatter(
                row["avg_hl_turnover_l1"],
                row["net_hl_sharpe"],
                marker=markers.get(str(row["hold_label"]), "o"),
                s=80,
                label=f"{name} {row['version']} {row['hold_label']}",
            )
    ax.axhline(1.5, color="0.6", linestyle="--", linewidth=0.8)
    ax.axvline(1.0, color="0.6", linestyle="--", linewidth=0.8)
    ax.set_xlabel("H-L L1 turnover on netted book")
    ax.set_ylabel("Net H-L Sharpe (7.5 bp / L1)")
    ax.set_title(f"Phase B staggered holding — {UNIVERSE}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_hold_cum_pnl(
    series_by_hold: Dict[str, pd.Series], title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, series in series_by_hold.items():
        ax.plot(series.index, series.cumsum(), label=label)
    ax.set_title(title)
    ax.set_ylabel("Cumulative H-L (gross)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_phase_b_report(
    metrics: pd.DataFrame,
    selection: Dict[str, object],
    depth: Dict[str, object],
    parity: pd.DataFrame,
    path: Path,
) -> None:
    cols = [
        "factor",
        "version",
        "hold_label",
        "rank_ic_effective",
        "ic_retention",
        "avg_hl_turnover_l1",
        "gross_hl_sharpe",
        "net_hl_annu",
        "net_hl_sharpe",
        "net_hl_mdd",
        "decile_mono_spearman",
        "economic_strategy_pass",
        "grade",
    ]
    phase_d_names = list(selection.get("research_keep") or selection.get("keep") or [])
    if depth.get("phase_d"):
        phase_d_names.append("depth_recovery_5m")
    lines = [
        "# Sprint 16 Phase B — Staggered holding",
        "",
        f"Contract: `{CONTRACT_VERSION}`",
        "",
        "Sleeve: rank **today's** `S_t`, hold that book H days, average active sleeves.",
        "Turnover: L1 on **netted** final weights. Not sum/average of sleeve TO.",
        "Not done: averaging past signals then re-ranking (that is Phase C).",
        "",
        "Diagnosis frozen from Phase C:",
        "",
        "- R3 = holding-period mismatch → keep daily information, slow capital.",
        "- `depth_recovery_5m` = noisy latent-state measurement → MA10 already done.",
        "",
        "## H=1 parity vs Phase C",
        "",
        "```",
        parity.to_string(index=False) if not parity.empty else "(no Phase C metrics)",
        "```",
        "",
        "## Metrics",
        "",
        "```",
        metrics[cols].to_string(index=False),
        "```",
        "",
        "## R3 selection (max 1 if dominated, else max 2)",
        "",
        f"mechanical keep: {selection.get('keep')}  reason: `{selection.get('reason')}`",
        f"research keep (Phase D): {selection.get('research_keep')}  "
        f"reason: `{selection.get('research_reason')}`",
        f"drop: {selection.get('drop')}",
        "",
        "Dropped R3 names do **not** enter Phase D. Research keep is the Phase D list.",
        "",
        "## depth_recovery_5m",
        "",
        f"verdict: `{depth.get('verdict')}`",
        f"daily GrossSR={depth.get('daily_gross_sharpe')}, "
        f"best={depth.get('best_hold')} GrossSR={depth.get('best_gross_sharpe')} "
        f"NetSR={depth.get('best_net_sharpe')}",
        f"grade daily/best: {depth.get('grade_daily')} / {depth.get('grade_best')}",
        "",
        "## Dual gate",
        "",
        "- Strategy-grade: NetAnn>10%, NetSR>1.5, TO<1, ICRetention≥60%, Mono>0.7.",
        "- Feature-grade: |IC|>2%, |ICIR|≥3, Mono>0.7, yearly sign ≥75%.",
        "- `economic_strategy_pass` is the strategy gate **without** mono, because R3",
        "  baseline mono is ~0.4–0.54. Production representative can still be R3.",
        "",
        "## Phase D (not run)",
        "",
        f"Candidates: {phase_d_names}",
        f"Buffer widths: {list(PHASE_D_BUFFER_WIDTHS)} on **new sleeve entry/exit**,",
        "plus asymmetric tail on the G1-dominant short of the R3 winner",
        "(`impact_per_trade` 5D short share ~85%).",
        "`depth_recovery` is not a Phase D strategy input unless standalone hope returns.",
        "Not a new MA. Not a primitive rebuild.",
        "",
        "Phase E L1-penalty optimizer waits until 1–2 production-worthy signals remain.",
        "",
        "## Reading",
        "",
        "R3: daily information was kept (IC identical across holds). Capital slowed.",
        "`impact_per_trade` RAW 5D is the family representative: TO 2.42→0.54, "
        "NetSR 1.40→1.47, GrossSR 3.46→1.97. It did **not** clear NetSR>1.5.",
        "10D over-smoothes capital (NetSR 1.25).",
        "",
        "`signed_amount_impact` MA3 is dominated and frozen.",
        "`signed_sqrt_amount_impact` RAW 5D is a ρ≈0.92 alias with NetSR 0.98; "
        "not worth a Phase D copy.",
        "",
        "`depth_recovery_5m` MA10: GrossSR 0.90→0.64→0.58. Stop rescuing standalone.",
        "Keep as feature-grade (IC 3.56%, mono 0.87, TO already 0.53).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase_b(
    *,
    output_root: Optional[Path] = None,
    candidates: Optional[Sequence[Tuple[str, int]]] = None,
    holds: Sequence[int] = PHASE_B_HOLDS,
    verify_hash: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = Path(output_root) if output_root is not None else OUT_ROOT / "phase_b"
    out.mkdir(parents=True, exist_ok=True)
    specs = list(candidates) if candidates is not None else list(PHASE_B_CANDIDATES)
    allowed = set(PHASE_B_CANDIDATES)
    unknown = [spec for spec in specs if spec not in allowed]
    if unknown:
        raise KeyError(f"Phase B candidates are frozen; refused: {unknown}")
    bad_holds = [h for h in holds if int(h) not in HOLD_LABELS]
    if bad_holds:
        raise KeyError(f"Phase B holds are frozen; refused: {bad_holds}")

    (out / "contract.json").write_text(
        json.dumps(contract_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    mask, ret = load_fast_context("full", verify_hash=verify_hash)
    phase_c = pd.read_csv(PHASE_C_METRICS) if PHASE_C_METRICS.exists() else None
    rows: List[Dict[str, object]] = []
    parity_rows: List[Dict[str, object]] = []
    cum_by_factor: Dict[str, Dict[str, pd.Series]] = {}

    for name, ma_window in specs:
        wide = load_factor_wide(name)
        prepared = prepare_candidate(name, int(ma_window), wide, mask, ret)
        cum_by_factor[name] = {}
        for hold in holds:
            row, group_pnl, to, net_pnl = evaluate_hold(prepared, int(hold))
            rows.append(row)
            dest = (
                out
                / "versions"
                / name
                / str(row["version"])
                / str(row["hold_label"])
            )
            dest.mkdir(parents=True, exist_ok=True)
            group_pnl.to_csv(dest / "group_pnl.csv")
            to.to_csv(dest / "turnover.csv")
            net_pnl.to_csv(dest / "net_pnl.csv")
            (dest / "summary.json").write_text(
                json.dumps(row, indent=2, default=float), encoding="utf-8"
            )
            cum_by_factor[name][str(row["hold_label"])] = group_pnl["H-L"]
            if int(hold) == 1:
                parity_rows.append(h1_parity_row(row, phase_c))

    metrics = pd.DataFrame(rows)
    selection = research_r3_keep(pick_r3(metrics))
    depth = judge_depth_recovery(metrics)
    parity = pd.DataFrame(parity_rows)
    metrics.to_csv(out / "holding_metrics.csv", index=False)
    parity.to_csv(out / "h1_parity.csv", index=False)
    (out / "r3_selection.json").write_text(
        json.dumps(selection, indent=2, default=float), encoding="utf-8"
    )
    (out / "depth_recovery_verdict.json").write_text(
        json.dumps(depth, indent=2, default=float), encoding="utf-8"
    )
    plot_phase_b_frontier(metrics, out / "net_sharpe_turnover_frontier.png")
    for name, series_map in cum_by_factor.items():
        plot_hold_cum_pnl(
            series_map,
            f"{name} gross H-L by holding period",
            out / "figures" / f"{name}_cum_pnl_by_hold.png",
        )
    write_phase_b_report(metrics, selection, depth, parity, out / "report.md")
    return metrics, selection
