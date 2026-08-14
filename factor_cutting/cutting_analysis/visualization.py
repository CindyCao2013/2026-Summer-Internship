"""Factor Cutting Visualization Layer — researcher-facing evidence pack.

Answers: did cutting purify information? Is the knife real? Is it tradable?

Plot families:
  A. IC decomposition (original / high / low / spread)
  B. Cumulative IC (high vs low persistence)
  C. Knife bucket analysis (knife quantile → fwd return)
  D. Decile portfolio + long-short curve
  E. Raw vs residual IC + neutralization / universe
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_d4_expansion_stack import daily_rank_ic_series, decile_group_means, icir_from_daily
from alpha_dimension_density import residual_ic_stats
from alpha_research_report import monotonicity_score, save_cumulative_decile_figure
from factor_attribution import align_signal, cs_zscore, hl_sharpe_from_composite
from factor_cutting.cutting_analysis.knife_ic import ic_stats
from factor_cutting.cutting_analysis.leg_analysis import decompose_legs, legs_ic_timeseries
from factor_cutting.cutting_analysis.neutralization import neutralization_ladder
from factor_cutting.engine import knife_quantile_mechanism
from liquidity_normalization import panel_cross_sectional_residual

# Consistent palette
C_HIGH = "#C0392B"
C_LOW = "#7F8C8D"
C_SPREAD = "#1A5276"
C_ORIG = "#27AE60"
C_BAR = "#2E86AB"


def _ensure(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A. IC decomposition
# ---------------------------------------------------------------------------

def plot_ic_decomposition_bar(
    ic_table: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "IC decomposition — cutting vs original",
) -> None:
    """Bar chart: original / high / low / spread RankIC."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ic_table["name"].tolist()
    vals = ic_table["rank_ic"].astype(float).tolist()
    colors = []
    for n in labels:
        if n in ("high", "M_high", "V_high"):
            colors.append(C_HIGH)
        elif n in ("low", "M_low", "V_low"):
            colors.append(C_LOW)
        elif n in ("spread", "M", "V", "ideal_reversal", "ideal_amplitude"):
            colors.append(C_SPREAD)
        else:
            colors.append(C_ORIG)
    ax.bar(labels, vals, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("RankIC")
    ax.set_title(title)
    for i, v in enumerate(vals):
        if pd.isna(v):
            continue
        ax.text(i, v, f"{v:.2%}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    _savefig(fig, out_path)


def plot_ic_timeseries(
    ic_daily: Dict[str, pd.Series],
    out_path: Path,
    *,
    ma: int = 60,
    title: str = "Daily RankIC (smoothed)",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    style = {
        "high": (C_HIGH, "-"),
        "low": (C_LOW, "-"),
        "spread": (C_SPREAD, "-"),
        "original": (C_ORIG, "--"),
    }
    for name, s in ic_daily.items():
        if s is None or s.dropna().empty:
            continue
        color, ls = style.get(name, (None, "-"))
        sm = s.rolling(ma, min_periods=max(10, ma // 3)).mean()
        ax.plot(sm.index, sm.values, label=f"{name} ({ma}d MA)", color=color, ls=ls, lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("RankIC")
    ax.legend(loc="best", fontsize=8)
    _savefig(fig, out_path)


def plot_cumulative_ic(
    ic_daily: Dict[str, pd.Series],
    out_path: Path,
    *,
    title: str = "Cumulative RankIC — high vs low leg",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    style = {
        "high": C_HIGH,
        "low": C_LOW,
        "spread": C_SPREAD,
        "original": C_ORIG,
    }
    for name, s in ic_daily.items():
        if s is None or s.dropna().empty:
            continue
        cum = s.fillna(0).cumsum()
        ax.plot(cum.index, cum.values, label=name, color=style.get(name), lw=1.6)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative RankIC")
    ax.legend(loc="best")
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# C. Knife bucket
# ---------------------------------------------------------------------------

def plot_knife_bucket_return(
    bucket_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Knife quantile → mean forward return",
) -> None:
    """bucket_df columns: q, mean_ret."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bucket_df["q"].astype(int), bucket_df["mean_ret"], color=C_BAR, edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Knife quantile (1=low … 10=high)")
    ax.set_ylabel("Mean forward return")
    ax.set_title(title)
    _savefig(fig, out_path)


# ---------------------------------------------------------------------------
# D. Portfolio
# ---------------------------------------------------------------------------

def plot_decile_returns(
    group_means: pd.Series,
    out_path: Path,
    *,
    title: str = "Decile mean forward returns",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = group_means.index.astype(int) if not isinstance(group_means.index[0], str) else range(1, len(group_means) + 1)
    ax.bar(x, group_means.values, color=C_BAR, edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Decile (1=low factor … 10=high)")
    ax.set_ylabel("Mean fwd return")
    ax.set_title(title)
    mono = monotonicity_score(group_means)
    ax.text(0.02, 0.95, f"monotonicity={mono:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    _savefig(fig, out_path)


def compute_group_cum_pnl(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Silent groupTest → cumulative group + H-L PnL."""
    import Factor_Dev_Lib

    sig = align_signal(cs_zscore(signal), 1)
    r = ret.reindex_like(sig)
    old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        _, group_pnl_df, _ = Factor_Dev_Lib.groupTest(sig, r, n=n, fee=0, info="silent")
    finally:
        plt.show = old_show
    return group_pnl_df.cumsum()


# ---------------------------------------------------------------------------
# E. Robustness
# ---------------------------------------------------------------------------

def plot_universe_compare(
    uni_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "RankIC by universe",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(uni_df["universe"], uni_df["rank_ic"], color=C_BAR, edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("RankIC")
    ax.set_title(title)
    for i, row in uni_df.reset_index(drop=True).iterrows():
        ax.text(i, row["rank_ic"], f"{row['rank_ic']:.2%}", ha="center", va="bottom" if row["rank_ic"] >= 0 else "top", fontsize=8)
    _savefig(fig, out_path)


def plot_neutralization(
    ladder: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Neutralization ladder — RankIC retention",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ladder["mode"], ladder["rank_ic"], color=C_SPREAD, edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("RankIC")
    ax.set_title(title)
    for i, row in ladder.reset_index(drop=True).iterrows():
        retn = row.get("ic_retention_vs_raw", np.nan)
        label = f"{row['rank_ic']:.2%}"
        if pd.notna(retn):
            label += f"\n({retn:.0%})"
        ax.text(i, row["rank_ic"], label, ha="center", va="bottom" if row["rank_ic"] >= 0 else "top", fontsize=8)
    _savefig(fig, out_path)


def plot_raw_vs_residual_ic(
    raw_ic: pd.Series,
    resid_ic: pd.Series,
    out_path: Path,
    *,
    title: str = "Raw IC vs residual IC (vs Base3)",
    ma: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(raw_ic.rolling(ma, min_periods=20).mean(), label=f"raw ({ma}d MA)", color=C_ORIG, lw=1.5)
    ax.plot(resid_ic.rolling(ma, min_periods=20).mean(), label=f"residual ({ma}d MA)", color=C_SPREAD, lw=1.5)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("RankIC")
    ax.legend()
    _savefig(fig, out_path)


def residual_daily_ic(
    factor: pd.DataFrame,
    ret: pd.DataFrame,
    anchors: List[pd.DataFrame],
    signal_shift: int = 1,
) -> pd.Series:
    """Daily IC of factor residualized vs anchors (equal-z Base3 combo as single anchor preferred)."""
    f = cs_zscore(factor)
    if len(anchors) == 1:
        resid = panel_cross_sectional_residual(f, [cs_zscore(anchors[0])])
    else:
        from factor_attribution import combine_equal_weight

        combo = combine_equal_weight([cs_zscore(a) for a in anchors])
        resid = panel_cross_sectional_residual(f, [combo])
    return daily_rank_ic_series(resid, ret, signal_shift=signal_shift)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_ic_comparison_table(
    *,
    original: Optional[pd.DataFrame],
    high: pd.DataFrame,
    low: pd.DataFrame,
    spread: pd.DataFrame,
    ret: pd.DataFrame,
    original_name: str = "Ret20",
    high_name: str = "M_high",
    low_name: str = "M_low",
    spread_name: str = "M (spread)",
) -> pd.DataFrame:
    rows = []
    panels = [
        (original_name, original),
        (high_name, high),
        (low_name, low),
        (spread_name, spread),
    ]
    for name, panel in panels:
        if panel is None:
            continue
        st = ic_stats(panel, ret)
        rows.append({"name": name, "rank_ic": st["rank_ic"], "icir": st["icir"], "n_days": st["n_days"]})
    return pd.DataFrame(rows)


def generate_cutting_viz_pack(
    *,
    factor_name: str,
    out_root: Path,
    original: Optional[pd.DataFrame],
    high: pd.DataFrame,
    low: pd.DataFrame,
    spread: pd.DataFrame,
    ret: pd.DataFrame,
    knife: Optional[pd.DataFrame] = None,
    knife_name: str = "",
    universe_df: Optional[pd.DataFrame] = None,
    neut_ladder: Optional[pd.DataFrame] = None,
    base3_panels: Optional[List[pd.DataFrame]] = None,
    original_name: str = "Ret20",
    high_name: str = "M_high",
    low_name: str = "M_low",
    spread_name: str = "M (spread)",
) -> dict:
    """
    Write full researcher-facing plot pack under::

        {out_root}/
          summary.md
          ic_analysis/
          mechanism/
          portfolio/
          robustness/
    """
    out_root = _ensure(out_root)
    ic_dir = _ensure(out_root / "ic_analysis")
    mech_dir = _ensure(out_root / "mechanism")
    port_dir = _ensure(out_root / "portfolio")
    rob_dir = _ensure(out_root / "robustness")

    # --- IC table + bars ---
    ic_table = build_ic_comparison_table(
        original=original,
        high=high,
        low=low,
        spread=spread,
        ret=ret,
        original_name=original_name,
        high_name=high_name,
        low_name=low_name,
        spread_name=spread_name,
    )
    ic_table.to_csv(ic_dir / "ic_comparison.csv", index=False)
    plot_ic_decomposition_bar(ic_table, ic_dir / "rank_ic_bar.png", title=f"{factor_name} — IC decomposition")

    # daily IC series
    ic_map = {
        "high": daily_rank_ic_series(high, ret),
        "low": daily_rank_ic_series(low, ret),
        "spread": daily_rank_ic_series(spread, ret),
    }
    if original is not None:
        ic_map["original"] = daily_rank_ic_series(original, ret)
    pd.DataFrame(ic_map).to_csv(ic_dir / "ic_daily.csv")
    plot_ic_timeseries(ic_map, ic_dir / "rank_ic_timeseries.png", title=f"{factor_name} RankIC timeseries")
    plot_cumulative_ic(ic_map, ic_dir / "cumulative_ic.png", title=f"{factor_name} cumulative RankIC")

    # --- mechanism: legs + knife buckets ---
    plot_cumulative_ic(
        {"high": ic_map["high"], "low": ic_map["low"]},
        mech_dir / "high_low_leg_ic.png",
        title=f"{factor_name} — high leg vs low leg cumulative IC",
    )
    ic_h = float(ic_table.loc[ic_table["name"] == high_name, "rank_ic"].iloc[0])
    ic_l = float(ic_table.loc[ic_table["name"] == low_name, "rank_ic"].iloc[0])
    ic_s = float(ic_table.loc[ic_table["name"] == spread_name, "rank_ic"].iloc[0])
    sep = ic_h - ic_l
    purity = abs(ic_s) / max(abs(ic_h), 1e-12)

    if knife is not None:
        bucket = knife_quantile_mechanism(ret, knife.reindex_like(ret), n_quantiles=10)
        bucket.to_csv(mech_dir / "knife_bucket_return.csv", index=False)
        plot_knife_bucket_return(
            bucket,
            mech_dir / "knife_bucket_return.png",
            title=f"Knife `{knife_name}` quantile → fwd return (CS)",
        )

    # --- portfolio ---
    gmeans = decile_group_means(spread, ret)
    gmeans.to_csv(port_dir / "decile_means.csv", header=["mean_ret"])
    plot_decile_returns(gmeans, port_dir / "decile_return.png", title=f"{factor_name} decile returns")
    sharpe, ann, direction = hl_sharpe_from_composite(spread, ret)
    try:
        import Factor_Dev_Lib
        from factor_runner import compute_group_stats, format_group_stats_title

        sig = align_signal(cs_zscore(spread), 1)
        r = ret.reindex_like(sig)
        old_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            _, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(sig, r, n=10, fee=0, info="silent")
        finally:
            plt.show = old_show
        cum = group_pnl_df.cumsum()
        cum.to_csv(port_dir / "group_cum_pnl.csv")
        stats = compute_group_stats(sig, r, group_pnl_df, group_to_df)
        save_cumulative_decile_figure(
            cum,
            factor_name=factor_name,
            stats_title=format_group_stats_title(stats),
            out_path=port_dir / "long_short_curve.png",
        )
        sharpe = stats["hl_sharpe"]
        ann = stats["hl_annu_ret"]
        direction = stats["direction"]
    except Exception as exc:
        (port_dir / "long_short_error.txt").write_text(str(exc))

    # --- robustness ---
    if universe_df is not None and not universe_df.empty:
        plot_universe_compare(universe_df, rob_dir / "universe_compare.png")
        universe_df.to_csv(rob_dir / "universe.csv", index=False)

    if neut_ladder is not None and not neut_ladder.empty:
        plot_neutralization(neut_ladder, rob_dir / "neutralization.png")
        neut_ladder.to_csv(rob_dir / "neutralization.csv", index=False)

    resid_meta = {}
    if base3_panels:
        raw_ic = ic_map["spread"]
        resid_ic = residual_daily_ic(spread, ret, base3_panels)
        resid_ic.to_csv(rob_dir / "residual_ic_daily.csv", header=["rank_ic"])
        plot_raw_vs_residual_ic(raw_ic, resid_ic, rob_dir / "raw_vs_residual.png")
        # summary stats vs combo
        from factor_attribution import combine_equal_weight

        combo = combine_equal_weight([cs_zscore(p) for p in base3_panels])
        resid_meta = residual_ic_stats(cs_zscore(spread), ret, combo)

    # --- summary.md ---
    write_summary_md(
        out_root / "summary.md",
        factor_name=factor_name,
        knife_name=knife_name,
        ic_table=ic_table,
        separation=sep,
        purity=purity,
        sharpe=sharpe,
        ann=ann,
        direction=direction,
        resid_meta=resid_meta,
        mono=monotonicity_score(gmeans),
    )
    return {
        "ic_table": ic_table,
        "separation": sep,
        "purity": purity,
        "hl_sharpe": sharpe,
        "residual": resid_meta,
    }


def write_summary_md(
    path: Path,
    *,
    factor_name: str,
    knife_name: str,
    ic_table: pd.DataFrame,
    separation: float,
    purity: float,
    sharpe: float,
    ann: float,
    direction: int,
    resid_meta: dict,
    mono: float,
) -> None:
    lines = [
        f"# Cutting Validation — {factor_name}",
        "",
        f"Knife: `{knife_name}`",
        "",
        "## Validation chain",
        "",
        "```",
        "original factor (mixed)",
        "        ↓",
        "knife partitions states",
        "        ↓",
        "high leg has alpha / low leg ~ noise",
        "        ↓",
        "spread purifies + stabilizes",
        "        ↓",
        "survives neutralization / residual vs Base3",
        "        ↓",
        "tradable (decile / H-L)",
        "```",
        "",
        "## IC decomposition",
        "",
        "| Name | RankIC | ICIR |",
        "|------|--------|------|",
    ]
    for _, r in ic_table.iterrows():
        lines.append(f"| {r['name']} | {r['rank_ic']:.4f} | {r['icir']:.2f} |")
    lines += [
        "",
        f"**Separation** (IC_high − IC_low): `{separation:.4f}`",
        f"**Purity** (|IC_spread| / |IC_high|): `{purity:.3f}`",
        "",
        "Purity high ⇒ low leg is near-noise; knife actually found the information locus.",
        "",
        "## Portfolio",
        "",
        f"- H-L Sharpe: `{sharpe:.2f}` (direction={direction})",
        f"- H-L ann return: `{ann:.1%}`",
        f"- Decile monotonicity: `{mono:.2f}`",
        "",
        "## Residual vs Base3",
        "",
    ]
    if resid_meta:
        lines.append(
            f"- residual IC mean `{resid_meta.get('residual_ic_mean', float('nan')):.4f}` · "
            f"t=`{resid_meta.get('residual_ic_t', float('nan')):.2f}` · "
            f"ICIR=`{resid_meta.get('residual_icir', float('nan')):.2f}`"
        )
    else:
        lines.append("- (not computed)")
    lines += [
        "",
        "## Plots",
        "",
        "- `ic_analysis/rank_ic_bar.png` — cutting before/after",
        "- `ic_analysis/cumulative_ic.png` — persistence of legs",
        "- `mechanism/high_low_leg_ic.png` — core cutting claim",
        "- `mechanism/knife_bucket_return.png` — knife separation",
        "- `portfolio/decile_return.png` + `long_short_curve.png`",
        "- `robustness/neutralization.png` + `raw_vs_residual.png` + `universe_compare.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
