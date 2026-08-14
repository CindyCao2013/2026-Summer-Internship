"""Publication-quality figures for Extreme Return Study."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import cumulative_return, monthly_returns, rolling_sharpe

# Visual style — clean quant note, avoid purple/cream AI defaults
PALETTE = {
    "bottom10": "#1B4F72",
    "top10": "#C0392B",
    "long_short": "#117A65",
    "csi300": "#7F8C8D",
}

STRATEGY_LABELS = {
    "bottom10": "Extreme Losers (Bottom10)",
    "top10": "Extreme Winners (Top10)",
    "long_short": "Long-Short (Losers−Winners)",
    "csi300": "CSI300",
}


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_cumulative_returns(
    series_map: Dict[str, pd.Series],
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Figure 1: cumulative return curves."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, s in series_map.items():
        if s is None or s.dropna().empty:
            continue
        cum = cumulative_return(s.dropna())
        ax.plot(
            cum.index,
            cum.values,
            label=STRATEGY_LABELS.get(key, key),
            color=PALETTE.get(key, None),
            lw=1.8 if key != "csi300" else 1.4,
            ls="-" if key != "csi300" else "--",
        )
    ax.axhline(0.0, color="#BDC3C7", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_holding_period_bars(
    comparison: pd.DataFrame,
    *,
    metric: str = "annu_ret",
    title: str,
    out_path: Path,
) -> Path:
    """Figure 2: bar chart across holding periods."""
    _setup_style()
    df = comparison.copy()
    order = ["bottom10", "top10", "long_short"]
    df = df[df["name"].isin(order)]
    pivot = df.pivot(index="name", columns="hold_days", values=metric)
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.columns))
    width = 0.25
    for i, name in enumerate(pivot.index):
        vals = pivot.loc[name].values.astype(float)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=STRATEGY_LABELS.get(name, name),
            color=PALETTE.get(name),
        )
    ax.axhline(0.0, color="#BDC3C7", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}D" for h in pivot.columns])
    ylab = {
        "annu_ret": "Annualized Return",
        "sharpe": "Sharpe",
        "mean_daily": "Mean Daily Return",
    }.get(metric, metric)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_monthly_heatmap(
    pnl: pd.Series,
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Figure 3: monthly return heatmap (year × month)."""
    _setup_style()
    m = monthly_returns(pnl)
    if m.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    df = m.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    mat = df.pivot(index="year", columns="month", values="ret")
    mat = mat.reindex(columns=list(range(1, 13)))

    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.55 * len(mat) + 1.5)))
    sns.heatmap(
        mat * 100.0,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        cbar_kws={"label": "Monthly Return (%)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rolling_sharpe(
    series_map: Dict[str, pd.Series],
    *,
    window: int = 60,
    title: str,
    out_path: Path,
) -> Path:
    """Figure 4: rolling Sharpe."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, s in series_map.items():
        if s is None or s.dropna().empty:
            continue
        rs = rolling_sharpe(s, window=window)
        ax.plot(
            rs.index,
            rs.values,
            label=STRATEGY_LABELS.get(key, key),
            color=PALETTE.get(key),
            lw=1.5,
        )
    ax.axhline(0.0, color="#BDC3C7", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(f"Rolling Sharpe ({window}d)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ic_bars(
    ic_table: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Mean RankIC by forward horizon."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = ic_table["horizon"].astype(int)
    ax.bar(x.astype(str) + "D", ic_table["mean_ic"], color="#1B4F72", width=0.55)
    ax.axhline(0.0, color="#BDC3C7", lw=0.8)
    for i, row in ic_table.iterrows():
        ax.text(
            i,
            row["mean_ic"] + (0.002 if row["mean_ic"] >= 0 else -0.004),
            f"ICIR={row['icir']:.2f}",
            ha="center",
            va="bottom" if row["mean_ic"] >= 0 else "top",
            fontsize=9,
        )
    ax.set_ylabel("Mean RankIC")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_regime_bars(
    regime_df: pd.DataFrame,
    *,
    metric: str = "sharpe",
    title: str,
    out_path: Path,
) -> Path:
    """Regime comparison bars."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = regime_df["name"].tolist()
    vals = regime_df[metric].astype(float).tolist()
    colors = ["#117A65" if (pd.notna(v) and v > 0) else "#C0392B" for v in vals]
    ax.bar(names, vals, color=colors, width=0.55)
    ax.axhline(0.0, color="#BDC3C7", lw=0.8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
