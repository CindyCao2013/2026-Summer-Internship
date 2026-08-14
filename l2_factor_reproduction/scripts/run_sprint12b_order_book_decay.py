#!/usr/bin/env python
"""Sprint 12B — ORDER_BOOK Signal Decay & Economic Viability Diagnostics.

DIAGNOSTIC ONLY. No Full Validation / new formulas / optimization / Sprint 13.

Usage:
    python -m l2_factor_reproduction.scripts.run_sprint12b_order_book_decay
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.evaluation_protocol_v2 import (  # noqa: E402
    ANNUALIZATION_DAYS,
    FEE_RATE_L1,
    SIGNAL_SHIFT,
    l1_to_oneway,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    DISCOVERY_END,
    DISCOVERY_START,
    PLOT_DPI,
    _configure_plot_fonts,
    load_fast_context,
)
from l2_factor_reproduction.python.order_book_factors import (  # noqa: E402
    ORDER_BOOK_FACTOR_SPECS,
)

S12A = Path(RESULT_ROOT) / "sprint12_order_book"
OUT = S12A / "decay_diagnostics"
FACTORS_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "order_book_family" / "factors"
)

# Cluster representatives only; exclude relative_spread_mean (alias of book_vwap_gap)
DIAG_FACTORS = [
    "total_depth_volatility",
    "closing_obi_l5",
    "opening_closing_obi_change",
    "depth_slope_asymmetry",
    "obi_sign_persistence",
    "obi_l5_mean",
    "book_vwap_gap",
]

IC_HORIZONS = (1, 2, 3, 5, 10)
PERS_HORIZONS = (1, 2, 3, 5)
LEG_HORIZONS = (1, 2, 3, 5, 10)
N_GROUPS = 10


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(x: float, d: int = 3) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2%}"


def load_factor_wide(factor: str) -> pd.DataFrame:
    path = FACTORS_DIR / factor / "factor_narrow.parquet"
    df = pd.read_parquet(path)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df = df.loc[
        df["tradetime"].between(
            DISCOVERY_START, DISCOVERY_END + pd.Timedelta(hours=23)
        )
    ]
    wide = df.pivot_table(
        index=df["tradetime"].dt.normalize(),
        columns="symbol",
        values="value",
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def apply_mask(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    m = mask.reindex(index=wide.index, columns=wide.columns)
    out = wide.where(m.fillna(False).astype(bool))
    return out


def daily_rank_ic(signal: pd.Series, ret: pd.Series) -> float:
    a = signal.replace([np.inf, -np.inf], np.nan)
    b = ret.replace([np.inf, -np.inf], np.nan)
    valid = a.notna() & b.notna()
    if valid.sum() < 30:
        return float("nan")
    return float(a[valid].corr(b[valid], method="spearman"))


def assign_deciles(row: pd.Series, n: int = N_GROUPS) -> pd.Series:
    s = row.dropna()
    if len(s) < n * 5:
        return pd.Series(index=row.index, dtype=float)
    # rank method first then qcut-like via rank percentiles
    ranks = s.rank(method="first")
    # 1 = lowest, n = highest
    bins = np.ceil(ranks / len(s) * n).clip(1, n).astype(int)
    out = pd.Series(index=row.index, dtype=float)
    out.loc[bins.index] = bins.astype(float)
    return out


# ---------------------------------------------------------------------------
# PART C — freeze
# ---------------------------------------------------------------------------


def build_factor_freeze(summary12a: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid in DIAG_FACTORS:
        spec = ORDER_BOOK_FACTOR_SPECS[fid]
        crow = clusters.loc[clusters["factor_id"] == fid].iloc[0]
        srow = summary12a.loc[summary12a["factor_id"] == fid].iloc[0]
        formula = spec.formula
        rows.append(
            {
                "factor_id": fid,
                "exact_formula": formula,
                "formula_hash": _sha(f"{fid}|{formula}|order_book_v1"),
                "source_primitive": f"order_book_daily / {formula}",
                "raw_direction": int(srow["factor_direction"]),
                "signal_shift": SIGNAL_SHIFT,
                "cluster": crow["cluster_id"],
                "representative_status": "representative"
                if bool(crow["is_representative"])
                else "alias_excluded",
                "mechanism_bucket": srow.get("mechanism_bucket", ""),
                "gate_12a": srow["gate"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART D — IC decay (f_t vs ret at t+h, same observation)
# ---------------------------------------------------------------------------


def compute_ic_decay(
    factors_wide: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    dates = ret.index
    for fid, wide in factors_wide.items():
        aligned = wide.reindex(index=dates, columns=ret.columns)
        ic1 = None
        for h in IC_HORIZONS:
            ics = []
            # f_t vs r_{t+h}
            for i in range(len(dates) - h):
                t = dates[i]
                th = dates[i + h]
                if t not in aligned.index or th not in ret.index:
                    continue
                ic = daily_rank_ic(aligned.loc[t], ret.loc[th])
                if np.isfinite(ic):
                    ics.append(ic)
            arr = np.array(ics, dtype=float)
            mean_ic = float(np.nanmean(arr)) if len(arr) else float("nan")
            std_ic = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan")
            icir = (
                mean_ic / std_ic * np.sqrt(250)
                if std_ic and std_ic > 0
                else float("nan")
            )
            pos = float(np.mean(arr > 0)) if len(arr) else float("nan")
            if h == 1:
                ic1 = mean_ic
            ret_h = (
                mean_ic / ic1
                if ic1 is not None and abs(ic1) > 1e-12 and np.isfinite(mean_ic)
                else float("nan")
            )
            rows.append(
                {
                    "factor_id": fid,
                    "horizon": h,
                    "mean_rank_ic": mean_ic,
                    "ic_std": std_ic,
                    "icir": icir,
                    "positive_ic_fraction": pos,
                    "n_dates": int(len(arr)),
                    "ic_retention_h": ret_h,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART E — rank persistence + decile migration
# ---------------------------------------------------------------------------


def compute_rank_persistence(
    factors_wide: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for fid, wide in factors_wide.items():
        dates = wide.index
        for h in PERS_HORIZONS:
            rhos = []
            for i in range(len(dates) - h):
                a = wide.iloc[i]
                b = wide.iloc[i + h]
                rho = daily_rank_ic(a, b)  # spearman of levels = rank persistence
                if np.isfinite(rho):
                    rhos.append(rho)
            arr = np.array(rhos, dtype=float)
            rows.append(
                {
                    "factor_id": fid,
                    "horizon": h,
                    "mean_spearman": float(np.nanmean(arr)) if len(arr) else float("nan"),
                    "median_spearman": float(np.nanmedian(arr)) if len(arr) else float("nan"),
                    "std_spearman": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else float("nan"),
                    "n_dates": int(len(arr)),
                }
            )
    return pd.DataFrame(rows)


def compute_decile_migration(
    factors_wide: Dict[str, pd.DataFrame],
    directions: Dict[str, int],
) -> pd.DataFrame:
    """Deciles in effective direction: G10=high effective factor."""
    rows = []
    for fid, wide in factors_wide.items():
        direction = int(directions[fid])
        # effective signal: higher = long
        eff = wide * float(direction)
        dates = eff.index
        g10_ret1, g10_ret2, g1_ret1, g1_ret2 = [], [], [], []
        same1, same2, mad1, mad2 = [], [], [], []
        for i in range(len(dates) - 2):
            d0 = assign_deciles(eff.iloc[i])
            d1 = assign_deciles(eff.iloc[i + 1])
            d2 = assign_deciles(eff.iloc[i + 2])
            common1 = d0.dropna().index.intersection(d1.dropna().index)
            common2 = d0.dropna().index.intersection(d2.dropna().index)
            if len(common1) < 50:
                continue
            a0, a1 = d0.loc[common1], d1.loc[common1]
            g10_mask = a0 == N_GROUPS
            g1_mask = a0 == 1
            if g10_mask.any():
                g10_ret1.append(float((a1.loc[g10_mask] == N_GROUPS).mean()))
            if g1_mask.any():
                g1_ret1.append(float((a1.loc[g1_mask] == 1).mean()))
            same1.append(float((a0 == a1).mean()))
            mad1.append(float((a0 - a1).abs().mean()))
            if len(common2) >= 50:
                a2 = d2.loc[common2]
                a0b = d0.loc[common2]
                g10b = a0b == N_GROUPS
                g1b = a0b == 1
                if g10b.any():
                    g10_ret2.append(float((a2.loc[g10b] == N_GROUPS).mean()))
                if g1b.any():
                    g1_ret2.append(float((a2.loc[g1b] == 1).mean()))
                same2.append(float((a0b == a2).mean()))
                mad2.append(float((a0b - a2).abs().mean()))

        def _m(x: List[float]) -> float:
            return float(np.mean(x)) if x else float("nan")

        rows.append(
            {
                "factor_id": fid,
                "G10_retention_t1": _m(g10_ret1),
                "G10_retention_t2": _m(g10_ret2),
                "G1_retention_t1": _m(g1_ret1),
                "G1_retention_t2": _m(g1_ret2),
                "same_decile_retention_t1": _m(same1),
                "same_decile_retention_t2": _m(same2),
                "mean_abs_decile_move_t1": _m(mad1),
                "mean_abs_decile_move_t2": _m(mad2),
                "n_pairs_t1": len(same1),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART F — turnover decomposition
# ---------------------------------------------------------------------------


def turnover_decomposition(
    summary12a: pd.DataFrame,
    cost12a: pd.DataFrame,
    persistence: pd.DataFrame,
    migration: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pers_t1 = persistence.loc[persistence["horizon"] == 1].set_index("factor_id")
    mig = migration.set_index("factor_id")
    cost = cost12a.set_index("factor_id") if "factor_id" in cost12a.columns else cost12a
    for fid in DIAG_FACTORS:
        s = summary12a.loc[summary12a["factor_id"] == fid].iloc[0]
        c = cost.loc[fid] if fid in cost.index else {}
        hl_l1 = float(
            c["daily_hl_l1_traded_notional"]
            if "daily_hl_l1_traded_notional" in c
            else s.get("daily_hl_oneway_turnover", np.nan) * 2
        )
        hl_ow = float(s["daily_hl_oneway_turnover"])
        # long/short one-way from cost file if present
        long_ow = float(c.get("G10_daily_oneway_turnover", np.nan)) if hasattr(c, "get") else float("nan")
        # cost_diagnostics may not have short separately — approximate from discovery if needed
        short_ow = float("nan")
        if fid in cost.index and "daily_short_oneway_turnover" in cost.columns:
            short_ow = float(cost.loc[fid, "daily_short_oneway_turnover"])
            long_ow = float(cost.loc[fid, "daily_long_oneway_turnover"]) if "daily_long_oneway_turnover" in cost.columns else long_ow

        rp1 = float(pers_t1.loc[fid, "mean_spearman"]) if fid in pers_t1.index else float("nan")
        g10r = float(mig.loc[fid, "G10_retention_t1"]) if fid in mig.index else float("nan")
        g1r = float(mig.loc[fid, "G1_retention_t1"]) if fid in mig.index else float("nan")
        mad = float(mig.loc[fid, "mean_abs_decile_move_t1"]) if fid in mig.index else float("nan")

        low_persist = bool(np.isfinite(rp1) and rp1 < 0.50)
        high_churn = bool(
            (np.isfinite(g10r) and g10r < 0.40)
            or (np.isfinite(mad) and mad > 2.0)
        )
        low_to = bool(np.isfinite(hl_ow) and hl_ow < 0.50)
        if low_to and np.isfinite(rp1) and rp1 >= 0.70:
            label = "NOT_TURNOVER_BOUND_PERSISTENT"
        elif low_persist and high_churn:
            label = "BOTH"
        elif low_persist:
            label = "LOW_SIGNAL_PERSISTENCE"
        elif high_churn:
            label = "DECILE_BOUNDARY_CHURN"
        elif np.isfinite(hl_ow) and hl_ow > 1.0 and np.isfinite(rp1) and rp1 >= 0.70:
            label = "UNEXPLAINED_PORTFOLIO_IMPLEMENTATION"
        elif np.isfinite(hl_ow) and hl_ow > 1.0:
            label = "BOTH"
        else:
            label = "NOT_TURNOVER_BOUND_PERSISTENT"

        rows.append(
            {
                "factor_id": fid,
                "daily_hl_l1_traded_notional": hl_l1,
                "daily_hl_oneway_turnover": hl_ow,
                "daily_long_oneway_turnover": long_ow,
                "daily_short_oneway_turnover": short_ow,
                "G10_retention": g10r,
                "G1_retention": g1r,
                "rank_persistence_t1": rp1,
                "mean_abs_decile_move_t1": mad,
                "turnover_association": label,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART G — leg decay (frozen f_t deciles vs forward returns)
# ---------------------------------------------------------------------------


def compute_leg_decay(
    factors_wide: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    directions: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    dates = ret.index
    for fid, wide in factors_wide.items():
        direction = int(directions[fid])
        eff = wide.reindex(index=dates, columns=ret.columns) * float(direction)
        for h in LEG_HORIZONS:
            g10_xs, g1_xs, hl_xs = [], [], []
            for i in range(len(dates) - h):
                t = dates[i]
                th = dates[i + h]
                if t not in eff.index:
                    continue
                dec = assign_deciles(eff.loc[t])
                r = ret.loc[th]
                common = dec.dropna().index.intersection(r.dropna().index)
                if len(common) < 50:
                    continue
                d = dec.loc[common]
                rr = r.loc[common]
                # cross-sectional demean excess
                xs = rr - rr.mean()
                g10 = xs.loc[d == N_GROUPS]
                g1 = xs.loc[d == 1]
                if len(g10) < 5 or len(g1) < 5:
                    continue
                g10m = float(g10.mean())
                g1m = float(g1.mean())
                g10_xs.append(g10m)
                g1_xs.append(g1m)
                hl_xs.append(g10m - g1m)

            def _stat(arr: List[float]) -> Tuple[float, float]:
                a = np.array(arr, dtype=float)
                if len(a) < 5:
                    return float("nan"), float("nan")
                m = float(a.mean())
                se = float(a.std(ddof=1) / np.sqrt(len(a))) if a.std(ddof=1) > 0 else float("nan")
                tstat = m / se if se and se > 0 else float("nan")
                return m, tstat

            g10_m, g10_t = _stat(g10_xs)
            g1_m, g1_t = _stat(g1_xs)
            hl_m, hl_t = _stat(hl_xs)
            # shares of abs HL using mean legs: long=g10, short=-g1
            long_c = g10_m
            short_c = -g1_m if np.isfinite(g1_m) else float("nan")
            denom = (abs(long_c) + abs(short_c)) if np.isfinite(long_c) and np.isfinite(short_c) else float("nan")
            rows.append(
                {
                    "factor_id": fid,
                    "horizon": h,
                    "g10_forward_excess_mean": g10_m,
                    "g10_forward_excess_tstat": g10_t,
                    "g1_forward_excess_mean": g1_m,
                    "g1_forward_excess_tstat": g1_t,
                    "hl_forward_mean": hl_m,
                    "hl_forward_tstat": hl_t,
                    "long_share_of_abs_hl": abs(long_c) / denom if denom and denom > 0 else float("nan"),
                    "short_share_of_abs_hl": abs(short_c) / denom if denom and denom > 0 else float("nan"),
                    "n_dates": len(hl_xs),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART H — frontier
# ---------------------------------------------------------------------------


def build_economic_frontier(summary12a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid in DIAG_FACTORS:
        s = summary12a.loc[summary12a["factor_id"] == fid].iloc[0]
        rows.append(
            {
                "factor_id": fid,
                "gross_hl_sharpe": float(s["gross_hl_sharpe"]),
                "gross_hl_annual": float(s["gross_hl_annual"]),
                "daily_hl_oneway_turnover": float(s["daily_hl_oneway_turnover"]),
                "fee_annualized_at_7p5bps": float(s["fee_annualized_at_7p5bps"]),
                "approx_net_hl_annual": float(s["approx_net_hl_annual"]),
                "approx_net_hl_sharpe": float(s["approx_net_hl_sharpe"]),
                "G10_net_excess_annual": float(s["G10_net_excess_annual"]),
                "dominant_leg": s["dominant_leg"],
                "gate_12a": s["gate"],
            }
        )
    return pd.DataFrame(rows)


def save_frontier_plots(frontier: pd.DataFrame) -> None:
    _configure_plot_fonts()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        frontier["daily_hl_oneway_turnover"],
        frontier["gross_hl_annual"],
        s=80,
        c="#2c7fb8",
        zorder=3,
    )
    for _, r in frontier.iterrows():
        ax.annotate(
            r["factor_id"],
            (r["daily_hl_oneway_turnover"], r["gross_hl_annual"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Daily H-L conventional one-way turnover")
    ax.set_ylabel("Gross H-L annual return")
    ax.set_title("ORDER_BOOK — Gross alpha vs turnover (Discovery 2023–2024)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gross_alpha_vs_turnover.png", dpi=PLOT_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        frontier["daily_hl_oneway_turnover"],
        frontier["approx_net_hl_annual"],
        s=80,
        c="#c44e52",
        zorder=3,
    )
    for _, r in frontier.iterrows():
        ax.annotate(
            r["factor_id"],
            (r["daily_hl_oneway_turnover"], r["approx_net_hl_annual"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Daily H-L conventional one-way turnover")
    ax.set_ylabel("Approx net H-L annual return (7.5bps L1)")
    ax.set_title("ORDER_BOOK — Net alpha vs turnover (Discovery 2023–2024)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "net_alpha_vs_turnover.png", dpi=PLOT_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PART I/J/K — diagnosis + decision
# ---------------------------------------------------------------------------


def approx_half_life(ic_decay: pd.DataFrame, fid: str) -> float:
    """Days until |ic_retention| drops below 0.5; nan if never."""
    sub = ic_decay.loc[ic_decay["factor_id"] == fid].sort_values("horizon")
    for _, r in sub.iterrows():
        if r["horizon"] == 1:
            continue
        ret = abs(float(r["ic_retention_h"])) if np.isfinite(r["ic_retention_h"]) else float("nan")
        if np.isfinite(ret) and ret < 0.5:
            return float(r["horizon"])
    # if still >=0.5 at h=10
    last = sub.loc[sub["horizon"] == 10]
    if not last.empty:
        ret = abs(float(last.iloc[0]["ic_retention_h"]))
        if np.isfinite(ret) and ret >= 0.5:
            return 10.0  # lower bound: >=10d
    return float("nan")


def diagnose_factor(
    fid: str,
    ic_decay: pd.DataFrame,
    persistence: pd.DataFrame,
    migration: pd.DataFrame,
    turnover: pd.DataFrame,
    leg_decay: pd.DataFrame,
    frontier: pd.DataFrame,
) -> Dict[str, Any]:
    ic = ic_decay.loc[ic_decay["factor_id"] == fid].set_index("horizon")
    pers = persistence.loc[persistence["factor_id"] == fid].set_index("horizon")
    mig = migration.loc[migration["factor_id"] == fid].iloc[0]
    to = turnover.loc[turnover["factor_id"] == fid].iloc[0]
    fr = frontier.loc[frontier["factor_id"] == fid].iloc[0]
    leg1 = leg_decay.loc[(leg_decay["factor_id"] == fid) & (leg_decay["horizon"] == 1)].iloc[0]
    leg5 = leg_decay.loc[(leg_decay["factor_id"] == fid) & (leg_decay["horizon"] == 5)]
    leg5 = leg5.iloc[0] if not leg5.empty else None

    hl = approx_half_life(ic_decay, fid)
    ic1 = float(ic.loc[1, "mean_rank_ic"]) if 1 in ic.index else float("nan")
    ic3 = float(ic.loc[3, "mean_rank_ic"]) if 3 in ic.index else float("nan")
    ic5 = float(ic.loc[5, "mean_rank_ic"]) if 5 in ic.index else float("nan")
    ret3 = float(ic.loc[3, "ic_retention_h"]) if 3 in ic.index else float("nan")
    ret5 = float(ic.loc[5, "ic_retention_h"]) if 5 in ic.index else float("nan")
    rp1 = float(pers.loc[1, "mean_spearman"]) if 1 in pers.index else float("nan")
    rp5 = float(pers.loc[5, "mean_spearman"]) if 5 in pers.index else float("nan")

    labels = []
    if np.isfinite(hl) and hl <= 2:
        labels.append("FAST_DECAY_ALPHA")
    if np.isfinite(ret5) and abs(ret5) >= 0.5 and np.isfinite(ic5) and abs(ic5) >= 0.005:
        labels.append("PERSISTENT_ALPHA")
    if float(fr["approx_net_hl_sharpe"]) < 0.3 or float(fr["G10_net_excess_annual"]) < 0:
        labels.append("ECONOMICALLY_NONVIABLE")
    if float(to["rank_persistence_t1"]) < 0.4 or float(mig["G10_retention_t1"]) < 0.35:
        labels.append("HIGH_CHURN_WEAK_SIGNAL")
    # short vs long at h=1
    short_share = float(leg1["short_share_of_abs_hl"])
    long_share = float(leg1["long_share_of_abs_hl"])
    if short_share > 0.65:
        labels.append("SHORT_SIDE_PERSISTENT" if (leg5 is not None and float(leg5["short_share_of_abs_hl"]) > 0.55) else "TRANSIENT_SHORT_ANOMALY")
    if long_share > 0.55 and leg5 is not None and float(leg5["g10_forward_excess_tstat"]) > 1.5:
        labels.append("LONG_SIDE_PERSISTENT")
    if not labels:
        labels.append("HIGH_CHURN_WEAK_SIGNAL")

    return {
        "factor_id": fid,
        "labels": "|".join(dict.fromkeys(labels)),
        "half_life_days": hl,
        "ic_t1": ic1,
        "ic_t3": ic3,
        "ic_t5": ic5,
        "ic_retention_t3": ret3,
        "ic_retention_t5": ret5,
        "rank_pers_t1": rp1,
        "rank_pers_t5": rp5,
        "G10_retention_t1": float(mig["G10_retention_t1"]),
        "turnover_association": to["turnover_association"],
        "approx_net_hl_sharpe": float(fr["approx_net_hl_sharpe"]),
        "G10_net_excess_annual": float(fr["G10_net_excess_annual"]),
        "daily_hl_oneway_turnover": float(fr["daily_hl_oneway_turnover"]),
        "dominant_leg": fr["dominant_leg"],
        "long_share_h1": long_share,
        "short_share_h1": short_share,
    }


def family_decision(diag_rows: List[Dict[str, Any]], frontier: pd.DataFrame) -> str:
    # Any persistent IC at t+3/t+5 with decent retention?
    persistent = [
        d
        for d in diag_rows
        if (
            np.isfinite(d["ic_retention_t5"])
            and abs(d["ic_retention_t5"]) >= 0.45
            and abs(d["ic_t5"]) >= 0.004
        )
        or (
            np.isfinite(d["ic_retention_t3"])
            and abs(d["ic_retention_t3"]) >= 0.55
            and abs(d["ic_t3"]) >= 0.005
        )
    ]
    # Existing factor ready for FV?
    ready = frontier.loc[
        (frontier["approx_net_hl_sharpe"] >= 0.8)
        & (frontier["G10_net_excess_annual"] > 0)
        & (frontier["daily_hl_oneway_turnover"] < 0.6)
        & (frontier["gate_12a"].isin(["strong_candidate", "research_candidate"]))
    ]
    # book_vwap may fail gate but healthy — not READY_FOR_FV under rule B
    if not ready.empty:
        return "ORDER_BOOK_FAMILY_READY_FOR_FV"
    if persistent:
        return "ORDER_BOOK_FAMILY_CONTINUE_MECHANISM_EXPANSION"
    # Check if any economically healthy even below gate
    healthy = frontier.loc[
        (frontier["approx_net_hl_sharpe"] >= 0.5)
        & (frontier["G10_net_excess_annual"] > 0)
        & (frontier["daily_hl_oneway_turnover"] < 0.5)
    ]
    if not healthy.empty and persistent:
        return "ORDER_BOOK_FAMILY_CONTINUE_MECHANISM_EXPANSION"
    if not healthy.empty:
        # healthy but weak gate / unclear persistence → expansion still possible
        return "ORDER_BOOK_FAMILY_CONTINUE_MECHANISM_EXPANSION"
    return "ORDER_BOOK_BASELINE_CLOSE"


def render_report(
    freeze: pd.DataFrame,
    ic_decay: pd.DataFrame,
    persistence: pd.DataFrame,
    migration: pd.DataFrame,
    turnover: pd.DataFrame,
    leg_decay: pd.DataFrame,
    frontier: pd.DataFrame,
    diag_df: pd.DataFrame,
    decision: str,
) -> str:
    lines = [
        "# Sprint 12B — ORDER_BOOK Signal Decay & Economic Viability Diagnostics",
        "",
        "DIAGNOSTIC ONLY. No Full Validation. No new formulas. Protocol/Fast Gate untouched.",
        f"Discovery sample: `{DISCOVERY_START.date()}` ~ `{DISCOVERY_END.date()}`",
        "",
        f"**Family decision:** `{decision}`",
        "",
        "## Diagnostic set (cluster representatives)",
        "",
        "Excluded `relative_spread_mean` as near-alias of representative `book_vwap_gap`.",
        "",
        freeze.to_string(index=False),
        "",
        "## IC decay (same f_t vs ret at t+h)",
        "",
        ic_decay.to_string(index=False),
        "",
        "## Rank persistence",
        "",
        persistence.to_string(index=False),
        "",
        "## Decile migration",
        "",
        migration.to_string(index=False),
        "",
        "## Turnover decomposition",
        "",
        turnover.to_string(index=False),
        "",
        "## Mechanism diagnosis",
        "",
        diag_df.to_string(index=False),
        "",
        "## Economic frontier (from Sprint 12A discovery)",
        "",
        frontier.to_string(index=False),
        "",
        "## PART J — Answers",
        "",
    ]

    # Q1 half-lives
    hl_txt = ", ".join(
        f"{r.factor_id}≈{_fmt(r.half_life_days, 1)}d" for r in diag_df.itertuples()
    )
    lines.append(f"1. Approx alpha half-life (|IC_ret|<0.5): **{hl_txt}**.")

    td = diag_df.loc[diag_df["factor_id"] == "total_depth_volatility"].iloc[0]
    lines.append(
        f"2. `total_depth_volatility` fails economically because of **{td['turnover_association']}** "
        f"(half-life≈{_fmt(td['half_life_days'], 1)}d, rank_pers_t1={_fmt(td['rank_pers_t1'], 3)}, "
        f"G10_ret_t1={_fmt(td['G10_retention_t1'], 3)}, TO_ow={_fmt(td['daily_hl_oneway_turnover'], 3)}, "
        f"netS={_fmt(td['approx_net_hl_sharpe'], 2)}). "
        f"Labels=`{td['labels']}`."
    )

    persist_names = [
        r.factor_id
        for r in diag_df.itertuples()
        if (np.isfinite(r.ic_retention_t3) and abs(r.ic_retention_t3) >= 0.55)
        or (np.isfinite(r.ic_retention_t5) and abs(r.ic_retention_t5) >= 0.45)
    ]
    lines.append(
        "3. Persistent through t+3/t+5? **"
        + (", ".join(persist_names) if persist_names else "NONE clearly")
        + "**."
    )

    long_pers = [
        r.factor_id
        for r in diag_df.itertuples()
        if "LONG_SIDE_PERSISTENT" in str(r.labels)
    ]
    lines.append(
        "4. Materially more persistent LONG alpha? **"
        + (", ".join(long_pers) if long_pers else "NONE clearly")
        + "**."
    )

    short_trans = [
        r.factor_id
        for r in diag_df.itertuples()
        if "TRANSIENT_SHORT_ANOMALY" in str(r.labels) or (
            "SHORT" in str(r.dominant_leg) and "FAST_DECAY_ALPHA" in str(r.labels)
        )
    ]
    lines.append(
        "5. Mainly transient short-leg effects? **"
        + (", ".join(short_trans) if short_trans else "see labels")
        + "**."
    )

    # Q6 best combo score
    score_rows = []
    for r in diag_df.itertuples():
        score = 0.0
        if np.isfinite(r.ic_retention_t5):
            score += abs(r.ic_retention_t5)
        if np.isfinite(r.rank_pers_t1):
            score += r.rank_pers_t1
        if np.isfinite(r.G10_retention_t1):
            score += r.G10_retention_t1
        if np.isfinite(r.daily_hl_oneway_turnover):
            score -= r.daily_hl_oneway_turnover
        if np.isfinite(r.G10_net_excess_annual):
            score += max(r.G10_net_excess_annual, -0.2) * 2
        if np.isfinite(r.approx_net_hl_sharpe):
            score += max(r.approx_net_hl_sharpe, -2) * 0.2
        score_rows.append((r.factor_id, score))
    score_rows.sort(key=lambda x: -x[1])
    best = score_rows[0][0]
    lines.append(
        f"6. Best combo (persistent IC / low churn / lower TO / G10 econ): **{best}** "
        f"(ranked: {', '.join(f'{a}:{b:.2f}' for a,b in score_rows)})."
    )

    bvg = frontier.loc[frontier["factor_id"] == "book_vwap_gap"].iloc[0]
    lines.append(
        f"7. `book_vwap_gap` economically healthier despite failing STRONG gate? **"
        f"{'YES' if float(bvg['approx_net_hl_sharpe']) > 0.5 and float(bvg['G10_net_excess_annual']) > 0 else 'NO'}** "
        f"(netS={_fmt(float(bvg['approx_net_hl_sharpe']), 2)}, "
        f"G10_net={_fmt_pct(float(bvg['G10_net_excess_annual']))}, "
        f"TO_ow={_fmt(float(bvg['daily_hl_oneway_turnover']), 3)}, gate=`{bvg['gate_12a']}`). "
        f"`relative_spread_mean` excluded as near-alias."
    )

    # Q8 family too short-lived?
    median_hl = float(np.nanmedian(diag_df["half_life_days"].to_numpy(dtype=float)))
    high_to = int((frontier["daily_hl_oneway_turnover"] > 1.0).sum())
    lines.append(
        f"8. ORDER_BOOK baseline too short-lived for daily framework? **"
        f"{'YES — largely' if (np.isfinite(median_hl) and median_hl <= 3) or high_to >= 5 else 'MIXED'}** "
        f"(median half-life≈{_fmt(median_hl, 1)}d; {high_to}/{len(frontier)} factors with TO_ow>1.0). "
        f"Decision=`{decision}`."
    )

    lines += [
        "",
        "## STOP",
        "",
        "No Full Validation. No Sprint 13. No formula / parameter / rebalance changes.",
        "",
    ]
    return "\n".join(lines)


def enrich_cost_long_short(cost: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Ensure long/short one-way columns exist; derive from 12A if missing."""
    out = cost.copy()
    if "factor_id" not in out.columns and out.index.name == "factor_id":
        out = out.reset_index()
    # Sprint 12A cost_diagnostics may lack long/short split — leave nan; turnover_decomposition handles it
    return out


def main() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    summary12a = pd.read_csv(S12A / "discovery_summary.csv")
    clusters = pd.read_csv(S12A / "candidate_clusters.csv")
    cost12a = pd.read_csv(S12A / "cost_diagnostics.csv")
    cost12a = enrich_cost_long_short(cost12a, summary12a)

    print("[C] factor freeze", flush=True)
    freeze = build_factor_freeze(summary12a, clusters)
    freeze.to_csv(OUT / "factor_freeze.csv", index=False)

    print("[load] discovery context + factor panels", flush=True)
    mask, ret = load_fast_context("discovery")
    mask.index = pd.to_datetime(mask.index)
    ret.index = pd.to_datetime(ret.index)
    factors_wide: Dict[str, pd.DataFrame] = {}
    directions: Dict[str, int] = {}
    for fid in DIAG_FACTORS:
        print(f"  load {fid}", flush=True)
        w = load_factor_wide(fid)
        w = apply_mask(w, mask)
        # align to ret dates
        w = w.reindex(index=ret.index.intersection(w.index), columns=ret.columns)
        factors_wide[fid] = w
        directions[fid] = int(
            summary12a.loc[summary12a["factor_id"] == fid, "factor_direction"].iloc[0]
        )

    print("[D] IC decay", flush=True)
    ic_decay = compute_ic_decay(factors_wide, ret)
    ic_decay.to_csv(OUT / "ic_decay.csv", index=False)

    print("[E] rank persistence + decile migration", flush=True)
    persistence = compute_rank_persistence(factors_wide)
    persistence.to_csv(OUT / "rank_persistence.csv", index=False)
    migration = compute_decile_migration(factors_wide, directions)
    migration.to_csv(OUT / "decile_migration.csv", index=False)

    print("[F] turnover decomposition", flush=True)
    # Try to add long/short OW from discovery group_to if available via re-reading summary only
    # Fill long/short from cost if we can recompute quickly from stored 12A — optional enhancement:
    # Use approximate split: often long≈short≈hl/2 for L1; but report true if we load from cost.
    # Augment cost with halves as diagnostic fallback labeled in notes
    if "daily_long_oneway_turnover" not in cost12a.columns:
        cost12a = cost12a.copy()
        if "daily_hl_l1_traded_notional" in cost12a.columns:
            # cannot invent accurately — leave empty; migration G10/G1 used instead
            pass
    turnover = turnover_decomposition(summary12a, cost12a, persistence, migration)
    # Fill long/short OW using discovery summary G10 TO if present in cost G10_daily
    for i, row in turnover.iterrows():
        fid = row["factor_id"]
        if not np.isfinite(row["daily_long_oneway_turnover"]):
            if fid in cost12a.set_index("factor_id").index:
                c = cost12a.set_index("factor_id").loc[fid]
                if "G10_daily_oneway_turnover" in c.index and np.isfinite(c["G10_daily_oneway_turnover"]):
                    turnover.at[i, "daily_long_oneway_turnover"] = float(c["G10_daily_oneway_turnover"])
        if not np.isfinite(row["daily_short_oneway_turnover"]):
            # approximate short OW ≈ HL OW - long OW when both known
            lo = turnover.at[i, "daily_long_oneway_turnover"]
            hl = turnover.at[i, "daily_hl_oneway_turnover"]
            if np.isfinite(lo) and np.isfinite(hl):
                turnover.at[i, "daily_short_oneway_turnover"] = max(hl - lo, 0.0)
    turnover.to_csv(OUT / "turnover_decomposition.csv", index=False)

    print("[G] leg decay", flush=True)
    leg_decay = compute_leg_decay(factors_wide, ret, directions)
    leg_decay.to_csv(OUT / "leg_decay.csv", index=False)

    print("[H] economic frontier + plots", flush=True)
    frontier = build_economic_frontier(summary12a)
    frontier.to_csv(OUT / "economic_frontier.csv", index=False)
    save_frontier_plots(frontier)

    print("[I] mechanism diagnosis", flush=True)
    diag_rows = [
        diagnose_factor(
            fid, ic_decay, persistence, migration, turnover, leg_decay, frontier
        )
        for fid in DIAG_FACTORS
    ]
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT / "mechanism_diagnosis.csv", index=False)

    decision = family_decision(diag_rows, frontier)
    print(f"[K] decision = {decision}", flush=True)

    report = render_report(
        freeze,
        ic_decay,
        persistence,
        migration,
        turnover,
        leg_decay,
        frontier,
        diag_df,
        decision,
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "sprint": "Sprint 12B — ORDER_BOOK Decay & Economic Viability Diagnostics",
        "status": "COMPLETE",
        "diagnostic_only": True,
        "discovery_window": ["2023-01-01", "2024-12-31"],
        "diagnostic_factors": DIAG_FACTORS,
        "excluded_alias": "relative_spread_mean (near-alias of book_vwap_gap)",
        "family_decision": decision,
        "no_full_validation": True,
        "no_new_formulas": True,
        "no_parameter_optimization": True,
        "no_sprint13": True,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Update parent sprint12 manifest
    parent = S12A / "manifest.json"
    if parent.exists():
        m = json.loads(parent.read_text(encoding="utf-8"))
        m["sprint12b"] = {
            "status": "COMPLETE",
            "family_decision": decision,
            "path": "decay_diagnostics/",
        }
        m["total_depth_volatility_fv"] = "BLOCKED — economically weak; diagnostic only"
        parent.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n[done] {decision} -> {OUT} ({manifest['elapsed_seconds']}s)", flush=True)
    return manifest


if __name__ == "__main__":
    main()
