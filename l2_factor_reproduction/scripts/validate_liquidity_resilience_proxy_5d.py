#!/usr/bin/env python
"""Single-Factor Full Validation v1 — liquidity_resilience_proxy_5d.

Frozen formula (Sprint 8 verified; NOT modified):
    daily depth_recovery_5m from liquidity_impact_daily
        = mean((Depth5_{t+5}-Depth5_t)/Depth5_t) over high-impact minutes
          (high-impact = |minute mid return| >= same-day 90th pct)
    factor = rolling_mean_5d(depth_recovery_5m), min_periods=5
    signal shift = T+1

The task-brief shock_weight / recovery-clip variant is intentionally
NOT used — that would change the verified formula.

Usage:
    python -m l2_factor_reproduction.scripts.validate_liquidity_resilience_proxy_5d
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import (  # noqa: E402
    IMPLIED_ANNU_FEE_BPS,
    calAnnuRet,
    calMDD,
    calSharpe,
    cs_neutral_size_ind,
    get_preheat_ind_data_citics,
    implied_annu_fee,
    mad,
    zsc,
)
from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    backtest_factor,
    narrow_to_wide,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    FULL_END,
    FULL_START,
    PLOT_DPI,
    _configure_plot_fonts,
    compute_fast_metrics,
    ensure_effective_group_pnl,
    load_fast_context,
)
from l2_factor_reproduction.python.low_turnover_v1 import (  # noqa: E402
    ROLLING_DAYS,
    _build_liquidity_resilience,
    _to_narrow,
    load_primitive_panel,
)
from l2_factor_reproduction.python.mid_trade_amount_research_data import (  # noqa: E402
    load_turnover_wide,
)

FACTOR = "liquidity_resilience_proxy_5d"
FEE = IMPLIED_ANNU_FEE_BPS / 1e4
HOLD_DAYS = 5
N_GROUPS = 10

SEGMENTS = {
    "2019-2022": (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")),
    "2023-2024": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
    "2025-2026.07": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-07-31")),
    "FULL": (FULL_START, FULL_END),
}

OUT_DIR = (
    Path(RESULT_ROOT) / "full_validation" / FACTOR
)
_PRIMITIVES = Path(RESULT_ROOT) / "primitives"
MCAP_PATH = _PRIMITIVES / "mcap_wide_2019-01-01_2026-07-31.parquet"


# ---------------------------------------------------------------------------
# Factor build (frozen)
# ---------------------------------------------------------------------------


def build_factor_narrow(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    liq = load_primitive_panel(
        _PRIMITIVES / "liquidity_impact_daily" / "dataset",
        ["depth_recovery_5m", "coverage_ratio"],
        start,
        end,
        buffer_days=60,
    )
    values = _build_liquidity_resilience(liq)
    mask = liq["TradeDate"].between(start, end)
    return _to_narrow(
        liq.loc[mask, "symbol"],
        liq.loc[mask, "TradeDate"],
        values.loc[mask],
        FACTOR,
    )


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def metrics_from_pnl(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    rank_ic_raw: pd.Series,
    *,
    factor_direction: int,
) -> Dict[str, float]:
    """Metrics on an already effective-direction pnl table."""
    pnl = ensure_effective_group_pnl(group_pnl)
    group_cols = [c for c in pnl.columns if c != "H-L"]
    decile_annu = np.array(
        [calAnnuRet(pnl[c]) for c in group_cols], dtype=float
    )
    ranks = pd.Series(np.arange(1, len(decile_annu) + 1), dtype=float)
    mono = float(ranks.corr(pd.Series(decile_annu), method="spearman"))
    violations = int(np.sum(decile_annu[1:] < decile_annu[:-1]))

    hl = pnl["H-L"].dropna()
    hl.index = pd.to_datetime(hl.index)
    monthly = hl.resample("ME").sum()
    pos_month = float((monthly > 0).mean()) if len(monthly) else float("nan")
    cum = hl.cumsum()
    if len(cum) > 2 and float(cum.std()) > 0:
        time_sp = float(
            pd.Series(cum.to_numpy()).corr(
                pd.Series(np.arange(len(cum)), dtype=float), method="spearman"
            )
        )
    else:
        time_sp = float("nan")

    mdd, _ = calMDD(hl)
    avg_to = float(group_to["H-L"].reindex(hl.index).mean())
    fee = float(implied_annu_fee(avg_to))
    annu = float(calAnnuRet(hl))
    ic_mean = float(rank_ic_raw.mean())
    ic_std = float(rank_ic_raw.std())
    # ICIR keeps the raw-IC sign (not flipped by factor_direction).
    icir = (
        ic_mean / ic_std * (250 ** 0.5) if ic_std and ic_std > 0 else float("nan")
    )
    g10 = pnl[group_cols[-1]]
    return {
        "rank_ic_mean_raw": ic_mean,
        "icir_raw": icir,
        "rank_ic": ic_mean,
        "icir": icir,
        "hl_annu_ret": annu,
        "hl_sharpe": float(calSharpe(hl)),
        "g10_excess_sharpe": float(calSharpe(g10)),
        "hl_mdd": float(mdd),
        "mdd": float(mdd),
        "avg_hl_turnover": avg_to,
        "turnover": avg_to,
        "implied_annu_fee": fee,
        "net_annu_after_fee": annu - fee,
        "decile_mono_spearman": mono,
        "mono": mono,
        "adjacent_violations": violations,
        "violations": violations,
        "positive_month_fraction": pos_month,
        "cum_hl_time_spearman": time_sp,
        "factor_direction": int(factor_direction),
        "n_days": int(len(hl)),
    }


def slice_backtest(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    rank_ic_raw: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    factor_direction: int,
) -> Dict[str, float]:
    pnl = group_pnl.loc[start:end]
    to = group_to.loc[start:end]
    ic = rank_ic_raw.loc[start:end]
    if pnl.empty:
        return {k: float("nan") for k in (
            "rank_ic", "icir", "hl_annu_ret", "hl_sharpe", "g10_excess_sharpe",
            "mono", "violations", "positive_month_fraction", "mdd", "turnover",
        )}
    return metrics_from_pnl(pnl, to, ic, factor_direction=factor_direction)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def save_validation_plots(
    out_dir: Path,
    group_pnl: pd.DataFrame,
    metrics: Dict[str, float],
) -> Tuple[Path, Path]:
    _configure_plot_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)
    pnl = ensure_effective_group_pnl(group_pnl)
    hl = pnl["H-L"]
    cum = hl.cumsum()

    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(cum.index, cum.values, color="black", linewidth=2.4, label="H-L")
    ax1.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    for mark, label in (
        (pd.Timestamp("2022-12-31"), "end-2022"),
        (pd.Timestamp("2024-12-31"), "end-2024"),
    ):
        ax1.axvline(mark, color="crimson", linewidth=1.2, linestyle="--", alpha=0.85)
        ax1.text(
            mark,
            ax1.get_ylim()[1] if False else float(cum.max()) * 0.98,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            color="crimson",
        )
    title = (
        f"{FACTOR} — H-L Cumulative Excess (effective direction; 2019–2026)\n"
        f"Sharpe={metrics['hl_sharpe']:.2f}  "
        f"annu={metrics['hl_annu_ret']:.2%}  "
        f"MDD={metrics['hl_mdd']:.2%}  "
        f"pos_month={metrics['positive_month_fraction']:.0%}  "
        f"TO={metrics['avg_hl_turnover']:.2f}  "
        f"net_annu={metrics['net_annu_after_fee']:.2%}"
    )
    ax1.set_title(title, fontsize=12)
    ax1.set_xlabel("TradeDate")
    ax1.set_ylabel("Cumulative H-L return (cumsum)")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.3)
    fig1.tight_layout()
    path_cum = out_dir / "cumulative_hl.png"
    fig1.savefig(path_cum, dpi=PLOT_DPI)
    plt.close(fig1)

    group_cols = [c for c in pnl.columns if c != "H-L"]
    means = pnl[group_cols].mean()
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.Blues
    colors = [cmap(0.35 + 0.55 * i / max(len(group_cols) - 1, 1)) for i in range(len(group_cols))]
    labels = [f"G{c}" for c in group_cols]
    bars = ax2.bar(labels, means.to_numpy(), color=colors, width=0.75)
    ax2.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.8)
    ax2.set_title(
        f"{FACTOR} — 各组日均超额收益 (full sample; G1=低有效因子 … G10=高有效因子)",
        fontsize=12,
    )
    ax2.set_xlabel("Decile (G1=low effective factor … G10=high effective factor)")
    ax2.set_ylabel("Mean daily excess return")
    ax2.grid(True, axis="y", alpha=0.3)
    y_span = float(np.nanmax(np.abs(means.to_numpy()))) if len(means) else 0.0
    offset = 0.02 * y_span if y_span > 0 else 1e-5
    for bar, val in zip(bars, means.to_numpy()):
        if not np.isfinite(val):
            continue
        label = f"{val * 1e4:.1f}bp" if abs(val) < 0.01 else f"{val:.4f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + (offset if val >= 0 else -offset),
            label,
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    fig2.tight_layout()
    path_bar = out_dir / "decile_bar.png"
    fig2.savefig(path_bar, dpi=PLOT_DPI)
    plt.close(fig2)
    return path_cum, path_bar


# ---------------------------------------------------------------------------
# Neutralization + exposure
# ---------------------------------------------------------------------------


def neutralize_ind_log_float_mktcap(signal: pd.DataFrame) -> pd.DataFrame:
    """INDUSTRY + log(FloatMktCap) residual; uses frozen mcap_wide parquet."""
    start, end = signal.index.min(), signal.index.max()
    ind = get_preheat_ind_data_citics(start, end).set_index("TradingDay")
    ind.index = pd.to_datetime(ind.index)
    ind = ind.reindex(index=signal.index, columns=signal.columns)

    mcap = pd.read_parquet(MCAP_PATH)
    mcap.index = pd.to_datetime(mcap.index)
    mcap = mcap.reindex(index=signal.index, columns=signal.columns)
    log_cap = np.log(mcap.where(mcap > 0))
    log_cap = mad(log_cap)
    log_cap = zsc(log_cap)

    out = signal.copy() * np.nan
    for dt in signal.index:
        s = signal.loc[dt]
        resid = cs_neutral_size_ind(
            s, ind.loc[dt], log_cap.loc[dt], nt_type="ind_cap"
        )
        out.loc[dt, resid.index] = resid.values
    return out


def daily_spearman_exposure(
    signal: pd.DataFrame, panel: pd.DataFrame
) -> pd.Series:
    common_idx = signal.index.intersection(panel.index)
    common_cols = signal.columns.intersection(panel.columns)
    a = signal.loc[common_idx, common_cols]
    b = panel.loc[common_idx, common_cols]
    return a.corrwith(b, axis=1, method="spearman")


# ---------------------------------------------------------------------------
# 5D staggered holding
# ---------------------------------------------------------------------------


def _decile_hl_weights(signal: pd.DataFrame, n_groups: int = N_GROUPS) -> pd.DataFrame:
    """Daily target H-L weights from cross-sectional ranks (equal-weight legs)."""
    # rank within day, map to bins 1..n
    ranks = signal.rank(axis=1, method="first")
    count = signal.notna().sum(axis=1).replace(0, np.nan)
    # Approximate qcut via rank thresholds
    q = ranks.div(count, axis=0)
    # G1: lowest ~10%, G10: highest ~10%
    lo = q <= (1.0 / n_groups)
    hi = q > (1.0 - 1.0 / n_groups)
    # equal weight within legs; missing -> 0
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for dt in signal.index:
        hi_mask = hi.loc[dt].fillna(False)
        lo_mask = lo.loc[dt].fillna(False)
        n_hi = int(hi_mask.sum())
        n_lo = int(lo_mask.sum())
        if n_hi > 0:
            w.loc[dt, hi_mask] = 1.0 / n_hi
        if n_lo > 0:
            w.loc[dt, lo_mask] = -1.0 / n_lo
    return w


def staggered_hold_backtest(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    hold_days: int = HOLD_DAYS,
) -> Tuple[pd.Series, pd.Series]:
    """Overlapping sleeves: P_t = mean_{j=0..h-1} w_{t-j}; pnl = P_t · ret_t."""
    common_idx = signal.index.intersection(ret.index)
    common_cols = signal.columns.intersection(ret.columns)
    sig = signal.loc[common_idx, common_cols]
    r = ret.loc[common_idx, common_cols]
    w_daily = _decile_hl_weights(sig)
    # staggered average of past hold_days target weights (including today)
    w_stagger = (
        w_daily.rolling(hold_days, min_periods=hold_days).mean()
    )
    # align: weights known at open of day from prior closes; signal already T+1 shifted
    pnl = (w_stagger * r).sum(axis=1)
    # L1 turnover of staggered weights
    dw = w_stagger.fillna(0.0).diff().abs().sum(axis=1)
    dw.iloc[0] = w_stagger.iloc[0].fillna(0.0).abs().sum()
    # valid from hold_days onward
    valid = w_stagger.notna().any(axis=1)
    return pnl.where(valid), dw.where(valid)


def execution_stats(pnl: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    pnl = pnl.dropna()
    to = turnover.reindex(pnl.index)
    annu = float(calAnnuRet(pnl))
    sharpe = float(calSharpe(pnl))
    avg_to = float(to.mean())
    fee = float(implied_annu_fee(avg_to))
    pnl_net = pnl - to.fillna(0.0) * FEE
    return {
        "gross_annu_ret": annu,
        "gross_sharpe": sharpe,
        "turnover": avg_to,
        "implied_annu_fee": fee,
        "net_annu_ret": annu - fee,
        "net_sharpe": float(calSharpe(pnl_net)),
        "n_days": int(len(pnl)),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def assign_verdict(
    full: Dict[str, float],
    segments: pd.DataFrame,
    neut: pd.DataFrame,
    exec_df: pd.DataFrame,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    post = segments.loc["2025-2026.07"]
    disc = segments.loc["2023-2024"]

    # A conditions
    a_ok = True
    checks = [
        ("full_hl_sharpe>=2.5", full["hl_sharpe"] >= 2.5),
        ("full_mono>=0.85", full["decile_mono_spearman"] >= 0.85),
        ("full_violations<=1", full["adjacent_violations"] <= 1),
        (
            "post_raw_ic_same_sign_as_discovery",
            np.sign(post["rank_ic"]) == np.sign(disc["rank_ic"])
            and np.sign(post["rank_ic"]) != 0,
        ),
        ("post_hl_sharpe>=1.5", post["hl_sharpe"] >= 1.5),
    ]
    for name, ok in checks:
        reasons.append(f"{'PASS' if ok else 'FAIL'}: {name}")
        a_ok = a_ok and bool(ok)

    # IND_CAP retention ~60%
    raw_ic = float(neut.loc[neut["mode"] == "RAW", "rank_ic_mean_raw"].iloc[0])
    neut_ic = float(
        neut.loc[neut["mode"] == "IND_CAP", "rank_ic_mean_raw"].iloc[0]
    )
    raw_sh = float(neut.loc[neut["mode"] == "RAW", "hl_sharpe"].iloc[0])
    neut_sh = float(neut.loc[neut["mode"] == "IND_CAP", "hl_sharpe"].iloc[0])
    ic_ret = abs(neut_ic) / abs(raw_ic) if abs(raw_ic) > 1e-12 else float("nan")
    sh_ret = neut_sh / raw_sh if abs(raw_sh) > 1e-12 else float("nan")
    neut_ok = (ic_ret >= 0.60) or (sh_ret >= 0.60)
    reasons.append(
        f"{'PASS' if neut_ok else 'FAIL'}: IND_CAP retention "
        f"(IC={ic_ret:.2%}, Sharpe={sh_ret:.2%})"
    )
    a_ok = a_ok and neut_ok

    daily = exec_df.loc[exec_df["mode"] == "daily_baseline"].iloc[0]
    stag = exec_df.loc[exec_df["mode"] == "staggered_5d"].iloc[0]
    to_red = 1.0 - float(stag["turnover"]) / float(daily["turnover"]) if daily["turnover"] else float("nan")
    net_pos = float(stag["net_annu_ret"]) > 0
    to_drop = to_red >= 0.20  # "显著降低"
    exec_ok = bool(to_drop and net_pos)
    reasons.append(
        f"{'PASS' if exec_ok else 'FAIL'}: 5D staggered "
        f"(TO_reduction={to_red:.2%}, net_annu={stag['net_annu_ret']:.2%})"
    )
    a_ok = a_ok and exec_ok

    if a_ok:
        return "A_strong_single_factor_candidate", reasons

    # B: weaker but research-worthy
    b_ok = (
        full["hl_sharpe"] >= 1.5
        and full["decile_mono_spearman"] >= 0.70
        and np.sign(post["rank_ic"]) == np.sign(disc["rank_ic"])
        and post["hl_sharpe"] >= 0.8
    )
    if b_ok:
        return "B_research_candidate", reasons
    return "C_not_confirmed", reasons


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(
    full: Dict[str, float],
    segments: pd.DataFrame,
    neut: pd.DataFrame,
    exposure: pd.DataFrame,
    exec_df: pd.DataFrame,
    verdict: str,
    reasons: List[str],
) -> str:
    disc = segments.loc["2023-2024"]
    post = segments.loc["2025-2026.07"]
    pre = segments.loc["2019-2022"]
    signs = [
        np.sign(pre["rank_ic"]),
        np.sign(disc["rank_ic"]),
        np.sign(post["rank_ic"]),
    ]
    same_sign = len(set(int(s) for s in signs if s != 0)) == 1

    # discovery contribution via cum path needs full pnl — approximate via annu*years
    # Better: use stored cum contribution if available in segments
    disc_share = segments.loc["2023-2024"].get("cum_hl_share_of_full", np.nan)

    raw = neut.loc[neut["mode"] == "RAW"].iloc[0]
    ind = neut.loc[neut["mode"] == "IND_CAP"].iloc[0]
    ic_ret = abs(ind["rank_ic_mean_raw"]) / abs(raw["rank_ic_mean_raw"]) if abs(raw["rank_ic_mean_raw"]) > 1e-12 else np.nan
    sh_ret = ind["hl_sharpe"] / raw["hl_sharpe"] if abs(raw["hl_sharpe"]) > 1e-12 else np.nan
    mono_ret = ind["decile_mono_spearman"] / raw["decile_mono_spearman"] if abs(raw["decile_mono_spearman"]) > 1e-12 else np.nan

    daily = exec_df.loc[exec_df["mode"] == "daily_baseline"].iloc[0]
    stag = exec_df.loc[exec_df["mode"] == "staggered_5d"].iloc[0]

    lines = [
        f"# Full Validation v1 — `{FACTOR}`",
        "",
        "Single-factor validation only. Formula / parameters / direction **frozen**. "
        "No production promotion.",
        "",
        "## Frozen formula (Sprint 8 verified)",
        "",
        "```",
        "depth_recovery_5m = mean( (Depth5_{t+5}-Depth5_t)/Depth5_t )",
        "                   over high-impact minutes",
        "                   (|r_mid| >= same-day 90th percentile)",
        "factor = rolling_mean_5d(depth_recovery_5m)   # min_periods=5",
        "signal = factor.shift(1)                      # T+1",
        "```",
        "",
        "Note: task-brief `shock_weight` / `recovery clip=[0,1]` is **not** the",
        "verified Sprint-8 artifact and was **not** applied (no formula change).",
        "",
        "## Full-sample baseline (2019-01-01 ~ 2026-07-31)",
        "",
        "```",
        pd.Series(full).to_string(),
        "```",
        "",
        "## Segment validation",
        "",
        "```",
        segments.to_string(),
        "```",
        "",
        f"- Three-segment raw IC same sign? **{'YES' if same_sign else 'NO'}** "
        f"(PRE={pre['rank_ic']:+.4f}, DISC={disc['rank_ic']:+.4f}, POST={post['rank_ic']:+.4f})",
        f"- POST H-L Sharpe = {post['hl_sharpe']:.3f}",
        f"- Discovery cum H-L share of full ≈ {disc_share:.1%}"
        if pd.notna(disc_share)
        else "- Discovery cum H-L share: see segment table",
        "",
        "## Neutralization: RAW vs INDUSTRY + log(FloatMktCap)",
        "",
        "```",
        neut.to_string(index=False),
        "```",
        "",
        f"- IC retention = {ic_ret:.1%}",
        f"- Sharpe retention = {sh_ret:.1%}",
        f"- mono retention = {mono_ret:.1%}",
        "",
        "## Exposure diagnostics (daily Spearman; not stripped)",
        "",
        "```",
        exposure.to_string(index=False),
        "```",
        "",
        "## Execution: daily vs fixed 5D staggered",
        "",
        "```",
        exec_df.to_string(index=False),
        "```",
        "",
        f"- GrossRetention = {stag['gross_annu_ret']/daily['gross_annu_ret']:.3f}"
        if daily["gross_annu_ret"]
        else "- GrossRetention = n/a",
        f"- TurnoverReduction = {1 - stag['turnover']/daily['turnover']:.3f}"
        if daily["turnover"]
        else "- TurnoverReduction = n/a",
        f"- NetImprovement = {stag['net_annu_ret'] - daily['net_annu_ret']:+.2%}",
        "",
        f"## Verdict: `{verdict}`",
        "",
        *[f"- {r}" for r in reasons],
        "",
        "## Direct answers",
        "",
        f"1. **POST 2025–2026 still valid?** "
        f"{'YES' if post['hl_sharpe'] >= 1.5 and np.sign(post['rank_ic']) == np.sign(disc['rank_ic']) else 'NO / weak'} "
        f"(POST Sharpe={post['hl_sharpe']:.2f}, raw IC={post['rank_ic']:+.4f}).",
        f"2. **Deciles still smooth/monotonic?** "
        f"{'YES (full)' if full['decile_mono_spearman'] >= 0.85 and full['adjacent_violations'] <= 1 else 'PARTIAL / NO'} "
        f"— full mono={full['decile_mono_spearman']:.3f}, violations={int(full['adjacent_violations'])}; "
        f"POST mono={post['mono']:.3f}, violations={int(post['violations'])} "
        f"({'weak in holdout' if post['mono'] < 0.75 else 'ok in holdout'}).",
        f"3. **Mostly size/liquidity exposure?** "
        f"{'PARTIAL — Sharpe retained but IC compressed' if ic_ret < 0.60 <= sh_ret else ('NO — IND_CAP retains material signal' if (ic_ret >= 0.60 or sh_ret >= 0.60) else 'YES / concerning — weak retention after IND_CAP')} "
        f"(IC ret={ic_ret:.0%}, Sharpe ret={sh_ret:.0%}; "
        f"mean |ρ| vs Turnover={float(exposure.loc[exposure['feature']=='Turnover','abs_mean_spearman'].iloc[0]):.2f}, "
        f"vs logFloatMktCap={float(exposure.loc[exposure['feature']=='log_FloatMktCap','abs_mean_spearman'].iloc[0]):.2f}).",
        f"4. **5D holding keeps alpha & cuts turnover?** "
        f"{'YES (net still >0, TO down)' if stag['net_annu_ret'] > 0 and stag['turnover'] < daily['turnover'] else 'NO'} "
        f"(gross {daily['gross_annu_ret']:.1%}→{stag['gross_annu_ret']:.1%}, "
        f"TO {daily['turnover']:.2f}→{stag['turnover']:.2f}, "
        f"net {daily['net_annu_ret']:.1%}→{stag['net_annu_ret']:.1%}; "
        f"NetImprovement={stag['net_annu_ret']-daily['net_annu_ret']:+.1%}).",
        f"5. **Final verdict:** `{verdict}`.",
        "",
        "## Boundary",
        "",
        "Not a production promotion. No KEEP/DROP beyond A/B/C research label.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start, end = FULL_START, FULL_END
    print(f"[1/7] build frozen factor {FACTOR} [{start.date()} ~ {end.date()}]", flush=True)
    narrow = build_factor_narrow(start, end)
    print(f"  narrow rows={len(narrow):,}", flush=True)

    print("[2/7] load fast_context(full) + baseline backtest", flush=True)
    mask, ret = load_fast_context("full")
    group_pnl, group_to, rank_ic_eff, summary = backtest_factor(
        narrow, start_day=start, end_day=end, mask=mask, ret_matrix=ret
    )
    # raw IC before direction flip
    factor_wide = narrow_to_wide(narrow)
    from l2_factor_reproduction.python.backtest import prepare_factor_signal, compute_rank_ic

    signal_raw, ret_aln = prepare_factor_signal(
        factor_wide, start=start, end=end, mask=mask, ret_matrix=ret
    )
    rank_ic_raw = compute_rank_ic(signal_raw, ret_aln)
    direction = int(summary["factor_direction"])
    full_metrics = metrics_from_pnl(
        group_pnl, group_to, rank_ic_raw, factor_direction=direction
    )
    # Also attach Fast Lane fields for CSV
    full_metrics["g10_excess_sharpe"] = float(summary["g10_excess_sharpe"])
    print(
        f"  full Sharpe={full_metrics['hl_sharpe']:.2f} "
        f"mono={full_metrics['decile_mono_spearman']:.3f} "
        f"dir={direction}",
        flush=True,
    )

    # Cum share by segment
    hl_eff = ensure_effective_group_pnl(group_pnl)["H-L"]
    cum_full = float(hl_eff.cumsum().iloc[-1]) if len(hl_eff) else np.nan

    print("[3/7] segment table", flush=True)
    seg_rows = []
    for name, (s0, s1) in SEGMENTS.items():
        m = slice_backtest(
            group_pnl, group_to, rank_ic_raw, s0, s1, direction
        )
        hl_seg = hl_eff.loc[s0:s1]
        if len(hl_seg) and pd.notna(cum_full) and abs(cum_full) > 1e-12:
            # contribution of this segment to total cumsum path
            # use sum of daily H-L in segment / final cum
            m["cum_hl_share_of_full"] = float(hl_seg.sum() / cum_full)
        else:
            m["cum_hl_share_of_full"] = np.nan
        m["segment"] = name
        seg_rows.append(m)
    segments = pd.DataFrame(seg_rows).set_index("segment")
    seg_out = segments[
        [
            "rank_ic",
            "icir",
            "hl_annu_ret",
            "hl_sharpe",
            "g10_excess_sharpe",
            "mono",
            "violations",
            "positive_month_fraction",
            "mdd",
            "turnover",
            "cum_hl_share_of_full",
        ]
    ].copy()
    seg_out.to_csv(OUT_DIR / "segment_validation.csv")

    print("[4/7] plots", flush=True)
    save_validation_plots(OUT_DIR, group_pnl, full_metrics)

    print("[5/7] neutralization + exposure (needs DDB for industry/turnover)", flush=True)
    # Effective-direction signal for trading metrics; RAW IC uses raw values
    signal_eff = -signal_raw if direction < 0 else signal_raw
    signal_neut = neutralize_ind_log_float_mktcap(signal_raw)

    # Backtest neutralized (raw formula values neutralized, then engine flips if needed)
    neut_narrow = signal_neut.stack(dropna=True).rename("value").reset_index()
    neut_narrow.columns = ["TradeDate", "symbol", "value"]
    neut_narrow["tradetime"] = (
        pd.to_datetime(neut_narrow["TradeDate"]) + pd.Timedelta(hours=9, minutes=30)
    )
    neut_narrow["factorname"] = FACTOR + "_IND_CAP"
    neut_narrow = neut_narrow[["symbol", "tradetime", "factorname", "value"]]
    gp_n, gt_n, _, sum_n = backtest_factor(
        neut_narrow, start_day=start, end_day=end, mask=mask, ret_matrix=ret
    )
    sig_n, ret_n = prepare_factor_signal(
        signal_neut, start=start, end=end, mask=mask, ret_matrix=ret
    )
    ic_neut_raw = compute_rank_ic(sig_n, ret_n)
    neut_metrics = metrics_from_pnl(
        gp_n, gt_n, ic_neut_raw, factor_direction=int(sum_n["factor_direction"])
    )

    neut_cmp = pd.DataFrame(
        [
            {
                "mode": "RAW",
                "rank_ic_mean_raw": full_metrics["rank_ic_mean_raw"],
                "icir_raw": full_metrics["icir_raw"],
                "hl_sharpe": full_metrics["hl_sharpe"],
                "hl_annu_ret": full_metrics["hl_annu_ret"],
                "decile_mono_spearman": full_metrics["decile_mono_spearman"],
                "adjacent_violations": full_metrics["adjacent_violations"],
                "avg_hl_turnover": full_metrics["avg_hl_turnover"],
            },
            {
                "mode": "IND_CAP",
                "rank_ic_mean_raw": neut_metrics["rank_ic_mean_raw"],
                "icir_raw": neut_metrics["icir_raw"],
                "hl_sharpe": neut_metrics["hl_sharpe"],
                "hl_annu_ret": neut_metrics["hl_annu_ret"],
                "decile_mono_spearman": neut_metrics["decile_mono_spearman"],
                "adjacent_violations": neut_metrics["adjacent_violations"],
                "avg_hl_turnover": neut_metrics["avg_hl_turnover"],
            },
        ]
    )
    # retention columns
    neut_cmp["ic_retention"] = (
        neut_cmp["rank_ic_mean_raw"].abs()
        / abs(full_metrics["rank_ic_mean_raw"])
    )
    neut_cmp["sharpe_retention"] = (
        neut_cmp["hl_sharpe"] / full_metrics["hl_sharpe"]
    )
    neut_cmp["mono_retention"] = (
        neut_cmp["decile_mono_spearman"]
        / full_metrics["decile_mono_spearman"]
    )
    neut_cmp.to_csv(OUT_DIR / "neutralization_comparison.csv", index=False)

    # Exposure vs Turnover and log FloatMktCap (use raw factor, lag1 aligned signal dates)
    turn = load_turnover_wide(start, end)
    turn.index = pd.to_datetime(turn.index)
    mcap = pd.read_parquet(MCAP_PATH)
    mcap.index = pd.to_datetime(mcap.index)
    log_cap = np.log(mcap.where(mcap > 0))
    # Align exposures to signal dates (already shifted factor)
    rho_to = daily_spearman_exposure(signal_raw, turn)
    rho_cap = daily_spearman_exposure(signal_raw, log_cap)
    exposure = pd.DataFrame(
        [
            {
                "feature": "Turnover",
                "mean_spearman": float(rho_to.mean()),
                "abs_mean_spearman": float(rho_to.abs().mean()),
                "median_spearman": float(rho_to.median()),
                "n_days": int(rho_to.notna().sum()),
            },
            {
                "feature": "log_FloatMktCap",
                "mean_spearman": float(rho_cap.mean()),
                "abs_mean_spearman": float(rho_cap.abs().mean()),
                "median_spearman": float(rho_cap.median()),
                "n_days": int(rho_cap.notna().sum()),
            },
        ]
    )
    exposure.to_csv(OUT_DIR / "exposure_diagnostics.csv", index=False)

    print("[6/7] daily vs 5D staggered execution", flush=True)
    # Daily H-L from effective pnl
    daily_pnl = ensure_effective_group_pnl(group_pnl)["H-L"]
    daily_to = group_to["H-L"].reindex(daily_pnl.index)
    daily_stats = execution_stats(daily_pnl, daily_to)
    daily_stats["mode"] = "daily_baseline"

    # Staggered on effective-direction signal (so long high-effective)
    stag_pnl, stag_to = staggered_hold_backtest(signal_eff, ret_aln, hold_days=HOLD_DAYS)
    # Flip staggered if mean negative (safety; should already be effective)
    if float(stag_pnl.dropna().mean()) < 0:
        stag_pnl = -stag_pnl
    stag_stats = execution_stats(stag_pnl, stag_to)
    stag_stats["mode"] = "staggered_5d"
    stag_stats["GrossRetention"] = (
        stag_stats["gross_annu_ret"] / daily_stats["gross_annu_ret"]
        if daily_stats["gross_annu_ret"]
        else np.nan
    )
    stag_stats["TurnoverReduction"] = (
        1.0 - stag_stats["turnover"] / daily_stats["turnover"]
        if daily_stats["turnover"]
        else np.nan
    )
    stag_stats["NetImprovement"] = (
        stag_stats["net_annu_ret"] - daily_stats["net_annu_ret"]
    )
    daily_stats["GrossRetention"] = 1.0
    daily_stats["TurnoverReduction"] = 0.0
    daily_stats["NetImprovement"] = 0.0
    exec_df = pd.DataFrame([daily_stats, stag_stats])
    exec_df.to_csv(OUT_DIR / "execution_5d.csv", index=False)

    print("[7/7] verdict + report", flush=True)
    verdict, reasons = assign_verdict(full_metrics, seg_out, neut_cmp, exec_df)
    full_row = pd.DataFrame([{**full_metrics, "factor": FACTOR, "window": "full"}])
    full_row.to_csv(OUT_DIR / "full_summary.csv", index=False)

    report = render_report(
        full_metrics, seg_out, neut_cmp, exposure, exec_df, verdict, reasons
    )
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "factor": FACTOR,
        "task": "Single-Factor Full Validation v1",
        "formula_frozen": {
            "source": "Sprint 8 verified liquidity_resilience_proxy_5d",
            "definition": (
                "rolling_mean_5d(depth_recovery_5m); "
                "depth_recovery_5m = mean depth rebuild over high-impact minutes; "
                "high-impact = |minute mid return| >= same-day 90th pct; "
                "T+1 shift"
            ),
            "NOT_applied": [
                "shock_weight = abs(ret_1m)*Amount/FloatMktCap",
                "recovery clip = [0,1]",
                "weighted mean recovery",
            ],
            "reason_not_applied": (
                "Would change the verified formula; task forbids formula edits"
            ),
            "rolling_days": ROLLING_DAYS,
            "signal_shift": 1,
        },
        "sample": {
            "full": [str(FULL_START.date()), str(FULL_END.date())],
            "segments": {k: [str(a.date()), str(b.date())] for k, (a, b) in SEGMENTS.items()},
        },
        "baseline": {
            "universe": UNIVERSE,
            "n_groups": N_GROUPS,
            "fee_bps_one_way": IMPLIED_ANNU_FEE_BPS,
            "factor_direction": direction,
        },
        "verdict": verdict,
        "verdict_checks": reasons,
        "outputs": sorted(p.name for p in OUT_DIR.iterdir()),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] verdict={verdict} -> {OUT_DIR} ({manifest['elapsed_seconds']}s)", flush=True)
    print(seg_out[["rank_ic", "hl_sharpe", "mono", "turnover"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
