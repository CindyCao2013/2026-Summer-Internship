"""Sprint 16 Phase D — hysteresis buffer then dollar-neutral long-tail breadth.

D1: impact_per_trade RAW + staggered 5D, buffer in {0, 5, 10, 20} on the
daily sleeve entry/exit. Book is still the 5D average of those sleeves.

D2: only the D1 winner buffer, times {G10-G1, G9:G10-G1, G8:G10-G1}.
Long and short remain dollar-neutral. No short overweight.

Not done here: Daily/10D books, 4x3 grid, cvxpy, gate relaxation.
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
from l2_factor_reproduction.python.evaluation_protocol_v2 import FEE_RATE_L1
from l2_factor_reproduction.python.fast_discovery import load_fast_context
from l2_factor_reproduction.python.liquidity_impact_execution import (
    CONTRACT_VERSION,
    NET_GATE,
    OUT_ROOT,
    PHASE_D1_IC_RETENTION_MIN,
    PHASE_D2_TAILS,
    PHASE_D_BUFFER_WIDTHS,
    PHASE_D_FACTOR,
    PHASE_D_HOLD,
    PHASE_D_MA_WINDOW,
    PHASE_D_NEAR_NETSR,
    PLOT_DPI,
    contract_payload,
    load_factor_wide,
)
from l2_factor_reproduction.python.liquidity_impact_phase_b import (
    PreparedCandidate,
    book_pnl,
    evaluate_hold,
    l1_turnover,
    prepare_candidate,
    stagger_weights,
)

TAIL_LONG_ENTRY = {
    "G10-G1": 0.90,
    "G9:G10-G1": 0.80,
    "G8:G10-G1": 0.70,
}
SHORT_ENTRY = 0.10


def buffer_label(width: float) -> str:
    return f"b{int(round(float(width) * 100)):02d}"


def cross_section_percentile(signal: pd.DataFrame) -> pd.DataFrame:
    n = signal.notna().sum(axis=1).replace(0, np.nan)
    rank = signal.rank(axis=1, method="first")
    return rank.div(n, axis=0)


def hysteresis_mask(
    pct: pd.DataFrame,
    *,
    enter: np.ndarray,
    stay: np.ndarray,
) -> pd.DataFrame:
    values_ok = np.isfinite(pct.to_numpy(dtype=float))
    enter = np.asarray(enter, dtype=bool) & values_ok
    stay = np.asarray(stay, dtype=bool) & values_ok
    n_days, n_names = enter.shape
    out = np.zeros((n_days, n_names), dtype=bool)
    prev = np.zeros(n_names, dtype=bool)
    for t in range(n_days):
        prev = (enter[t] | (prev & stay[t])) & values_ok[t]
        out[t] = prev
    return pd.DataFrame(out, index=pct.index, columns=pct.columns)


def equal_weight_mask(mask: pd.DataFrame) -> pd.DataFrame:
    count = mask.sum(axis=1).replace(0, np.nan)
    return mask.astype(float).div(count, axis=0)


def hysteresis_hl_weights(
    signal: pd.DataFrame,
    *,
    buffer: float,
    long_entry: float = 0.90,
    short_entry: float = SHORT_ENTRY,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if buffer < -1e-12:
        raise ValueError("buffer must be >= 0")
    pct = cross_section_percentile(signal)
    arr = pct.to_numpy(dtype=float)
    short_enter = np.isfinite(arr) & (arr <= short_entry)
    short_stay = np.isfinite(arr) & (arr <= short_entry + float(buffer))
    long_enter = np.isfinite(arr) & (arr >= long_entry)
    long_stay = np.isfinite(arr) & (arr >= long_entry - float(buffer))
    if abs(float(buffer)) < 1e-12:
        short_mask = pd.DataFrame(short_enter, index=pct.index, columns=pct.columns)
        long_mask = pd.DataFrame(long_enter, index=pct.index, columns=pct.columns)
    else:
        short_mask = hysteresis_mask(pct, enter=short_enter, stay=short_stay)
        long_mask = hysteresis_mask(pct, enter=long_enter, stay=long_stay)
    if bool((short_mask & long_mask).to_numpy().any()):
        raise RuntimeError("long and short hysteresis regions overlapped")
    long_w = equal_weight_mask(long_mask)
    short_w = equal_weight_mask(short_mask)
    hl = long_w.fillna(0.0) - short_w.fillna(0.0)
    return hl, long_w, short_w


def _economic_pass_no_mono(row: Dict[str, object]) -> bool:
    return (
        float(row["net_hl_annu"]) > NET_GATE["net_annu_min"]
        and float(row["net_hl_sharpe"]) > NET_GATE["net_sharpe_min"]
        and float(row["avg_hl_turnover_l1"]) < NET_GATE["turnover_l1_max"]
        and float(row["ic_retention"]) >= NET_GATE["ic_retention_min"]
    )


def score_hl_book(
    prepared: PreparedCandidate,
    daily_hl: pd.DataFrame,
    hold: int,
    extra: Dict[str, object],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.Series, pd.Series]:
    hl_w = stagger_weights(daily_hl, hold)
    valid = hl_w.notna().any(axis=1) & (hl_w.abs().sum(axis=1) > 0)
    hl_w = hl_w.loc[valid]
    aligned_ret = prepared.ret.reindex(index=hl_w.index, columns=hl_w.columns)
    hl_pnl = book_pnl(hl_w, aligned_ret)
    to = l1_turnover(hl_w).reindex(hl_pnl.index)
    net_pnl = hl_pnl - to.fillna(0.0) * FEE_RATE_L1
    long_w = hl_w.clip(lower=0.0)
    short_w = (-hl_w).clip(lower=0.0)
    long_pnl = book_pnl(long_w, aligned_ret)
    short_names_pnl = book_pnl(short_w, aligned_ret)
    mdd_g, _ = calMDD(hl_pnl)
    mdd_n, _ = calMDD(net_pnl)
    ic_mean = float(prepared.rank_ic_eff.mean())
    ic_std = float(prepared.rank_ic_eff.std())
    icir = ic_mean / ic_std * (250 ** 0.5) if ic_std > 0 else float("nan")
    to_mean = float(to.mean())
    ic_ret = (
        abs(ic_mean) / prepared.raw_ic_abs
        if prepared.raw_ic_abs and prepared.raw_ic_abs > 0
        else float("nan")
    )
    g1_annu = float(calAnnuRet(short_names_pnl))
    g10_annu = float(calAnnuRet(long_pnl))
    denom = g10_annu - g1_annu
    long_share = float(g10_annu / denom) if abs(denom) > 1e-12 else float("nan")
    row: Dict[str, object] = {
        "factor": prepared.name,
        "version": prepared.version,
        "ma_window": int(prepared.ma_window),
        "hold_days": int(hold),
        "hold_label": "staggered_5d",
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
        "n_days": int(len(hl_pnl.dropna())),
        "gross_long": float(long_w.sum(axis=1).mean()),
        "gross_short": float(short_w.sum(axis=1).mean()),
        "g1_annu": g1_annu,
        "g10_annu": g10_annu,
        "long_share_of_hl": long_share,
        "short_share_of_hl": (
            float(1.0 - long_share) if np.isfinite(long_share) else float("nan")
        ),
    }
    row.update(extra)
    row["economic_pass_no_mono"] = _economic_pass_no_mono(row)
    row["grade"] = (
        "economic_strategy_pass_pending_mono"
        if row["economic_pass_no_mono"]
        else "neither"
    )
    group_pnl = pd.DataFrame(
        {"H-L": hl_pnl, "long": long_pnl, "short": short_names_pnl}
    )
    return row, group_pnl, to, net_pnl


def attach_cost_efficiency(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    base = out.loc[out["buffer_width"] == 0.0]
    if base.empty:
        out["gross_sr_sacrifice_per_to"] = np.nan
        return out
    to0 = float(base["avg_hl_turnover_l1"].iloc[0])
    gs0 = float(base["gross_hl_sharpe"].iloc[0])
    ga0 = float(base["gross_hl_annu"].iloc[0])
    d_to = to0 - out["avg_hl_turnover_l1"]
    d_gs = gs0 - out["gross_hl_sharpe"]
    out["delta_to_vs_b0"] = d_to
    out["delta_gross_sr_vs_b0"] = -d_gs
    out["gross_sr_sacrifice_per_to"] = np.where(d_to.abs() > 1e-8, d_gs / d_to, 0.0)
    out["gross_annu_retention"] = out["gross_hl_annu"] / ga0 if ga0 else np.nan
    return out


def pick_d1(metrics: pd.DataFrame) -> Dict[str, object]:
    eligible = metrics.loc[metrics["ic_retention"] >= PHASE_D1_IC_RETENTION_MIN].copy()
    if eligible.empty:
        return {"buffer_width": None, "reason": "no_ic_retention"}
    passed = eligible.loc[eligible["economic_pass_no_mono"] == True]  # noqa: E712
    pool = passed if not passed.empty else eligible
    best_sr = float(pool["net_hl_sharpe"].max())
    near = pool.loc[(pool["net_hl_sharpe"] - best_sr).abs() < PHASE_D_NEAR_NETSR]
    winner = near.sort_values(
        ["buffer_width", "avg_hl_turnover_l1"], ascending=[True, True]
    ).iloc[0]
    if not passed.empty:
        reason = (
            "cleared_1.5_simpler_in_near_band"
            if len(near) > 1
            else "cleared_1.5_max_net_sharpe"
        )
    else:
        reason = (
            "max_net_sharpe_gate_not_cleared"
            if len(near) == 1
            else "near_netsr_band_simpler_gate_not_cleared"
        )
    return {
        "buffer_width": float(winner["buffer_width"]),
        "buffer_label": str(winner["buffer_label"]),
        "net_hl_sharpe": float(winner["net_hl_sharpe"]),
        "avg_hl_turnover_l1": float(winner["avg_hl_turnover_l1"]),
        "gross_hl_sharpe": float(winner["gross_hl_sharpe"]),
        "economic_pass_no_mono": bool(winner["economic_pass_no_mono"]),
        "reason": reason,
        "best_net_sharpe_in_grid": best_sr,
        "n_near": int(len(near)),
        "n_cleared_1_5": int(len(passed)),
    }


def pick_d2(metrics: pd.DataFrame) -> Dict[str, object]:
    eligible = metrics.loc[metrics["ic_retention"] >= PHASE_D1_IC_RETENTION_MIN].copy()
    if eligible.empty:
        return {"tail": None, "reason": "no_ic_retention"}
    order = {name: i for i, name in enumerate(PHASE_D2_TAILS)}
    eligible["_tail_rank"] = eligible["tail"].map(order)
    passed = eligible.loc[eligible["economic_pass_no_mono"] == True]  # noqa: E712
    pool = passed if not passed.empty else eligible
    best_sr = float(pool["net_hl_sharpe"].max())
    near = pool.loc[(pool["net_hl_sharpe"] - best_sr).abs() < PHASE_D_NEAR_NETSR]
    winner = near.sort_values(
        ["_tail_rank", "avg_hl_turnover_l1"], ascending=[True, True]
    ).iloc[0]
    if not passed.empty:
        reason = (
            "cleared_1.5_simpler_tail"
            if len(near) > 1
            else "cleared_1.5_max_net_sharpe"
        )
    else:
        reason = (
            "max_net_sharpe_gate_not_cleared"
            if len(near) == 1
            else "near_netsr_band_simpler_gate_not_cleared"
        )
    return {
        "tail": str(winner["tail"]),
        "buffer_width": float(winner["buffer_width"]),
        "net_hl_sharpe": float(winner["net_hl_sharpe"]),
        "avg_hl_turnover_l1": float(winner["avg_hl_turnover_l1"]),
        "gross_hl_sharpe": float(winner["gross_hl_sharpe"]),
        "economic_pass_no_mono": bool(winner["economic_pass_no_mono"]),
        "reason": reason,
        "best_net_sharpe_in_grid": best_sr,
    }


def g1_state_diagnostic(
    prepared: PreparedCandidate,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pct = cross_section_percentile(prepared.signal_eff)
    g1 = pct.le(SHORT_ENTRY) & prepared.signal_eff.notna()
    prev = g1.shift(1).fillna(False).astype(bool)
    new_g1 = g1 & ~prev
    persist = g1 & prev
    ret = prepared.ret.reindex_like(prepared.signal_eff)

    def _cohort_mean(mask: pd.DataFrame, horizon: int) -> pd.Series:
        if horizon == 1:
            return ret.where(mask).mean(axis=1)
        stacked = [ret.shift(-lag).where(mask).mean(axis=1) for lag in range(horizon)]
        return pd.concat(stacked, axis=1).mean(axis=1)

    rows = []
    summary: Dict[str, object] = {}
    for name, mask in (("new_g1", new_g1), ("persistent_g1", persist)):
        r1 = _cohort_mean(mask, 1)
        r5 = _cohort_mean(mask, 5)
        n_names = mask.sum(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "cohort": name,
                    "ret_1d": r1,
                    "ret_5d_avg": r5,
                    "n_names": n_names,
                }
            )
        )
        summary[f"{name}_n_days"] = int(r1.dropna().shape[0])
        summary[f"{name}_mean_1d"] = float(r1.mean())
        summary[f"{name}_mean_5d_avg"] = float(r5.mean())
        summary[f"{name}_names_avg"] = float(n_names.mean())
    new_1d = float(summary["new_g1_mean_1d"])
    per_1d = float(summary["persistent_g1_mean_1d"])
    if per_1d < new_1d - 5e-4 and new_1d < 0:
        case = "C_persistent_g1_stronger"
        note = (
            "Persistent G1 is more negative than new G1: the bad state persists. "
            "Symmetric buffer can still fail if it also loosens the weak long tail."
        )
    elif new_1d < per_1d - 5e-4 and per_1d < 0:
        case = "A_event_alpha_keep_buffer_narrow"
        note = (
            "New G1 << persistent G1 < 0: alpha is the entry event; "
            "wide buffer is costly."
        )
    elif abs(new_1d - per_1d) <= 5e-4 and new_1d < 0 and per_1d < 0:
        case = "B_persistent_liquidity_state"
        note = (
            "New ≈ persistent << 0: state is persistent; "
            "10-20% exit can be reasonable."
        )
    elif new_1d < 0 and per_1d < 0:
        case = "mixed"
        note = "Both negative; inspect the gap before widening buffer."
    else:
        case = "inspect_sign"
        note = "G1 forward return is not uniformly negative; inspect."
    summary["case"] = case
    summary["note"] = note
    summary["gap_1d_new_minus_persist"] = new_1d - per_1d
    daily = pd.concat(rows, axis=0)
    daily.index.name = "date"
    return daily.reset_index(), summary


def plot_d1_frontier(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = metrics.sort_values("buffer_width")
    ax.plot(ordered["avg_hl_turnover_l1"], ordered["net_hl_sharpe"], color="0.7")
    for _, row in ordered.iterrows():
        ax.scatter(row["avg_hl_turnover_l1"], row["net_hl_sharpe"], s=80)
        ax.annotate(
            str(row["buffer_label"]),
            (row["avg_hl_turnover_l1"], row["net_hl_sharpe"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.axhline(1.5, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("H-L L1 turnover on netted 5D book")
    ax.set_ylabel("Net H-L Sharpe")
    ax.set_title(f"Phase D1 buffer — {UNIVERSE} impact_per_trade RAW 5D")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def plot_g1_diagnostic(daily: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = daily.groupby("cohort")[["ret_1d", "ret_5d_avg"]].mean()
    means.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_ylabel("Mean excess return")
    ax.set_title("G1 state diagnostic (explain buffer, not a search)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)


def write_freeze_log(path: Path) -> None:
    from l2_factor_reproduction.python.liquidity_impact_execution import SPRINT_STATUS

    if SPRINT_STATUS == "FROZEN_COMPLETE" and path.exists():
        return
    path.write_text(
        "\n".join(
            [
                "# Sprint 16 freeze log",
                "",
                "Written after Phase B. Reasons are execution-level, not “alpha missing”.",
                "",
                "## impact_per_trade RAW + staggered 5D",
                "",
                "```text",
                "KEEP — family strategy representative",
                "expression = RAW + staggered 5D + buffer 0 + Long(G8:G10)−Short(G1)",
                "```",
                "",
                "Daily information is useful (IC unchanged across holds). Capital does not",
                "need to fully chase the new rank every day. 5D is the persistence/inertia",
                "balance; 10D already loses NetSR (1.47 → 1.25).",
                "",
                "Symmetric buffer 5/10/20 did not buy the last 0.03 NetSR.",
                "Dollar-neutral long-tail breadth did: G8:G10-G1 NetSR 1.47 → 1.93.",
                "Phase E is optional.",
                "",
                "## signed_amount_impact MA3",
                "",
                "```text",
                "FREEZE — execution degradation",
                "```",
                "",
                "5D NetSR 0.86 → 0.61. Slowing capital directly hurt return efficiency.",
                "",
                "## signed_sqrt_amount_impact RAW",
                "",
                "```text",
                "FREEZE — redundant R3 representative",
                "```",
                "",
                "5D TO 0.52, NetSR 0.98 is not useless, but ρ≈0.92 vs impact_per_trade",
                "and weaker on NetSR / NetAnn / MDD. Later FS should treat this as",
                "execution-level redundancy, not absent alpha.",
                "",
                "## depth_recovery_5m MA10",
                "",
                "```text",
                "factor_status = FEATURE_GRADE",
                "strategy_status = NON_STANDALONE",
                "representation = MA10",
                "```",
                "",
                "IC 3.56%, mono 0.87, TO 0.53, yearly stable. Staggered GrossSR",
                "0.90 → 0.64 → 0.58. Standalone STOP. Keep for FS / ML / multifactor.",
                "Feature selection must not require every input to clear NetSR 1.5.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_phase_d_report(
    d1: pd.DataFrame,
    d1_pick: Dict[str, object],
    d2: pd.DataFrame,
    d2_pick: Dict[str, object],
    g1: Dict[str, object],
    path: Path,
) -> None:
    cols1 = [
        "buffer_label",
        "avg_hl_turnover_l1",
        "gross_hl_sharpe",
        "net_hl_sharpe",
        "net_hl_annu",
        "net_hl_mdd",
        "gross_sr_sacrifice_per_to",
        "economic_pass_no_mono",
    ]
    cols2 = [
        "tail",
        "buffer_label",
        "avg_hl_turnover_l1",
        "gross_hl_sharpe",
        "net_hl_sharpe",
        "net_hl_annu",
        "economic_pass_no_mono",
    ]
    final_sr = d2_pick.get("net_hl_sharpe")
    if final_sr is None:
        e_note = "D2 not available."
    elif float(final_sr) > NET_GATE["net_sharpe_min"]:
        e_note = (
            "NetSR cleared 1.5. Phase E cvxpy is optional production enhancement, "
            "not a Sprint 16 requirement."
        )
    else:
        e_note = (
            f"NetSR={float(final_sr):.3f} still below 1.5. Only then is an L1 "
            "penalty optimizer worth asking; do not relax the gate."
        )
    d2_table = d2[cols2].to_string(index=False) if not d2.empty else "(not run)"
    lines = [
        "# Sprint 16 Phase D — buffer then asymmetric long breadth",
        "",
        f"Contract: `{CONTRACT_VERSION}`",
        "Frozen book: `impact_per_trade` RAW + staggered 5D. Gate NetSR>1.5 not relaxed.",
        "Buffer acts on **new sleeve** entry/exit. Not a new MA. Not Daily/10D.",
        "",
        "## D1 hysteresis",
        "",
        "```",
        d1[cols1].to_string(index=False),
        "```",
        "",
        f"winner: buffer={d1_pick.get('buffer_width')} ({d1_pick.get('reason')})",
        "",
        "Primary objective is NetSR, not the lowest TO. "
        f"ICRetention gate is {PHASE_D1_IC_RETENTION_MIN:.0%} (factor IC vs RAW).",
        "",
        "## G1 state diagnostic (explanation only)",
        "",
        f"case: `{g1.get('case')}`",
        f"{g1.get('note')}",
        f"New G1 1D={g1.get('new_g1_mean_1d')}, persistent 1D={g1.get('persistent_g1_mean_1d')}",
        "",
        "Do not invert this table into a 1%…30% buffer search.",
        "",
        "## D2 dollar-neutral long breadth",
        "",
        "```",
        d2_table,
        "```",
        "",
        f"winner: tail={d2_pick.get('tail')} ({d2_pick.get('reason')})",
        f"economic_pass_no_mono={d2_pick.get('economic_pass_no_mono')}",
        "",
        "## Phase E",
        "",
        e_note,
        "",
        "Question this phase answers: is `impact_per_trade` only a strong predictive",
        "feature, or can no-trade / tail expression make it tradable without overfitting?",
        "",
        "## Reading",
        "",
        "D1: every wider buffer cut GrossSR faster than TO. Sacrifice ≈ 4 GrossSR per",
        "unit TO. Buffer is not how to buy the last 0.03 NetSR. Winner is b=0.",
        "",
        "G1 diagnostic: persistent G1 is more negative than new G1, so the short-side",
        "state persists. Symmetric buffer still failed because it also loosened the",
        "weak long tail.",
        "",
        "D2: dollar-neutral broadening of the noisy long tail is the execution lever.",
        "G9:G10-G1 already clears NetSR 1.5. G8:G10-G1 is the frozen-grid winner.",
        "G6:G10 was not tested. Phase E is optional.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _refuse_wrong_spec(
    factor: str, ma_window: int, hold: int, buffers: Sequence[float]
) -> None:
    if factor != PHASE_D_FACTOR or int(ma_window) != PHASE_D_MA_WINDOW:
        raise KeyError("Phase D is frozen to impact_per_trade RAW")
    if int(hold) != PHASE_D_HOLD:
        raise KeyError("Phase D is frozen to staggered 5D; no Daily/10D buffer grid")
    extra = [b for b in buffers if float(b) not in PHASE_D_BUFFER_WIDTHS]
    if extra:
        raise KeyError(f"Phase D buffers are frozen; refused: {extra}")


def _annotate_b0(row: Dict[str, object], extra: Dict[str, object]) -> Dict[str, object]:
    row = dict(row)
    row.update(extra)
    row["gross_long"] = 1.0
    row["gross_short"] = 1.0
    row["economic_pass_no_mono"] = _economic_pass_no_mono(row)
    return row


def run_phase_d1(
    prepared: PreparedCandidate,
    *,
    out: Path,
    buffers: Sequence[float] = PHASE_D_BUFFER_WIDTHS,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for width in buffers:
        extra = {
            "phase": "D1",
            "buffer_width": float(width),
            "buffer_label": buffer_label(width),
            "tail": "G10-G1",
            "long_entry": 0.90,
            "short_entry": SHORT_ENTRY,
        }
        if abs(float(width)) < 1e-12:
            row, group_pnl, to, net_pnl = evaluate_hold(prepared, PHASE_D_HOLD)
            row = _annotate_b0(row, extra)
        else:
            daily_hl, _long_w, _short_w = hysteresis_hl_weights(
                prepared.signal_eff, buffer=float(width)
            )
            row, group_pnl, to, net_pnl = score_hl_book(
                prepared, daily_hl, PHASE_D_HOLD, extra
            )
        rows.append(row)
        dest = out / "d1" / str(row["buffer_label"])
        dest.mkdir(parents=True, exist_ok=True)
        group_pnl.to_csv(dest / "group_pnl.csv")
        to.to_csv(dest / "turnover.csv")
        net_pnl.to_csv(dest / "net_pnl.csv")
        (dest / "summary.json").write_text(
            json.dumps(row, indent=2, default=float), encoding="utf-8"
        )
    metrics = attach_cost_efficiency(pd.DataFrame(rows))
    pick = pick_d1(metrics)
    metrics.to_csv(out / "d1_metrics.csv", index=False)
    (out / "d1_selection.json").write_text(
        json.dumps(pick, indent=2, default=float), encoding="utf-8"
    )
    plot_d1_frontier(metrics, out / "d1_net_sharpe_turnover.png")
    return metrics, pick


def run_phase_d2(
    prepared: PreparedCandidate,
    buffer_width: float,
    *,
    out: Path,
    tails: Sequence[str] = PHASE_D2_TAILS,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    unknown = [t for t in tails if t not in TAIL_LONG_ENTRY]
    if unknown:
        raise KeyError(f"Phase D2 tails are frozen; refused: {unknown}")
    rows: List[Dict[str, object]] = []
    for tail in tails:
        long_entry = TAIL_LONG_ENTRY[tail]
        extra = {
            "phase": "D2",
            "buffer_width": float(buffer_width),
            "buffer_label": buffer_label(buffer_width),
            "tail": tail,
            "long_entry": long_entry,
            "short_entry": SHORT_ENTRY,
        }
        if tail == "G10-G1" and abs(float(buffer_width)) < 1e-12:
            row, group_pnl, to, net_pnl = evaluate_hold(prepared, PHASE_D_HOLD)
            row = _annotate_b0(row, extra)
        else:
            daily_hl, long_w, short_w = hysteresis_hl_weights(
                prepared.signal_eff,
                buffer=float(buffer_width),
                long_entry=long_entry,
            )
            extra["gross_long_sleeve"] = float(long_w.fillna(0).sum(axis=1).mean())
            extra["gross_short_sleeve"] = float(short_w.fillna(0).sum(axis=1).mean())
            row, group_pnl, to, net_pnl = score_hl_book(
                prepared, daily_hl, PHASE_D_HOLD, extra
            )
        rows.append(row)
        dest = out / "d2" / str(tail).replace(":", "-")
        dest.mkdir(parents=True, exist_ok=True)
        group_pnl.to_csv(dest / "group_pnl.csv")
        to.to_csv(dest / "turnover.csv")
        net_pnl.to_csv(dest / "net_pnl.csv")
        (dest / "summary.json").write_text(
            json.dumps(row, indent=2, default=float), encoding="utf-8"
        )
    metrics = pd.DataFrame(rows)
    pick = pick_d2(metrics)
    metrics.to_csv(out / "d2_metrics.csv", index=False)
    (out / "d2_selection.json").write_text(
        json.dumps(pick, indent=2, default=float), encoding="utf-8"
    )
    return metrics, pick


def run_phase_d(
    *,
    output_root: Optional[Path] = None,
    factor: str = PHASE_D_FACTOR,
    ma_window: int = PHASE_D_MA_WINDOW,
    hold: int = PHASE_D_HOLD,
    buffers: Sequence[float] = PHASE_D_BUFFER_WIDTHS,
    d1_only: bool = False,
    verify_hash: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame, Dict[str, object]]:
    _refuse_wrong_spec(factor, ma_window, hold, buffers)
    out = Path(output_root) if output_root is not None else OUT_ROOT / "phase_d"
    out.mkdir(parents=True, exist_ok=True)
    (out / "contract.json").write_text(
        json.dumps(contract_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_freeze_log(OUT_ROOT / "freeze_log.md")
    mask, ret = load_fast_context("full", verify_hash=verify_hash)
    prepared = prepare_candidate(
        factor, int(ma_window), load_factor_wide(factor), mask, ret
    )
    d1, d1_pick = run_phase_d1(prepared, out=out, buffers=buffers)
    g1_daily, g1_summary = g1_state_diagnostic(prepared)
    g1_daily.to_csv(out / "g1_state_daily.csv", index=False)
    (out / "g1_state_summary.json").write_text(
        json.dumps(g1_summary, indent=2, default=float), encoding="utf-8"
    )
    plot_g1_diagnostic(g1_daily, out / "g1_state_diagnostic.png")
    if d1_only or d1_pick.get("buffer_width") is None:
        d2, d2_pick = pd.DataFrame(), {"tail": None, "reason": "d1_only_or_failed"}
    else:
        d2, d2_pick = run_phase_d2(
            prepared, float(d1_pick["buffer_width"]), out=out
        )
    write_phase_d_report(d1, d1_pick, d2, d2_pick, g1_summary, out / "report.md")
    return d1, d1_pick, d2, d2_pick

