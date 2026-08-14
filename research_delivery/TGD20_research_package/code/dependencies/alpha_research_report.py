"""Alpha Research Report v1 — restore the original research deliverable.

Per-factor output:
  - IC / ICIR / sign consistency
  - Decile (quantile) mean returns + monotonicity score
  - H-L cumulative curve + Sharpe / turnover
  - IC time series + IC decay (1/5/10/20d)
  - Universe stability (CSI300/500/1000/ALL)
  - Markdown report + figures

This layer sits ON TOP of harness/attribution — it answers:
  "Does this factor make money? Show me the charts."
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import factor_config as cfg
from factor_attribution import rank_ic_by_horizon
from factor_runner import compute_group_stats, format_group_stats_title, prepare_signal

DEFAULT_TIER_A: List[str] = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
    "cn_cancel_shock",
    "quality_composite",
    "value_composite",
]

FACTOR_META: Dict[str, Dict[str, str]] = {
    "low_vol_liquidity_quality_60d": {
        "dimension": "D1",
        "role": "base",
        "hypothesis": "Low vol × liquidity stability — quality liquidity state",
    },
    "winner_sentiment_reversal_5d": {
        "dimension": "D4",
        "role": "base",
        "hypothesis": "Short-horizon winner exhaustion / sentiment reversal",
    },
    "upside_fragility_20d": {
        "dimension": "D5",
        "role": "base",
        "hypothesis": "Upside tail fragility — crash-prone winners",
    },
    "cn_cancel_shock": {
        "dimension": "L2",
        "role": "enhancer",
        "hypothesis": "Cancel withdrawal shock — trade-flow state",
    },
    "quality_composite": {
        "dimension": "D7",
        "role": "enhancer",
        "hypothesis": "Equal-z(roe_stability, GP/A, CFO/NI)",
    },
    "value_composite": {
        "dimension": "D6",
        "role": "enhancer",
        "hypothesis": "Equal-z(EP, BP, CFP) ind+size neutral",
    },
}


@dataclass
class FactorReport:
    factor_name: str
    dimension: str
    role: str
    hypothesis: str
    universe_stats: pd.DataFrame
    ic_decay: pd.DataFrame
    quantile_returns: pd.DataFrame
    hl_cum_pnl: pd.Series
    rank_ic_daily: pd.Series
    group_cum_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    aggregate: Dict[str, float] = field(default_factory=dict)


def _df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Simple markdown table without tabulate dependency."""
    if df.empty:
        return "_empty_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def monotonicity_score(group_means: pd.Series) -> float:
    """Fraction of adjacent decile pairs with correct ordering (higher group → higher return)."""
    vals = group_means.dropna().values
    if len(vals) < 3:
        return np.nan
    diffs = np.diff(vals)
    return float(np.mean(diffs > 0))


def ic_positive_ratio(rank_ic_daily: pd.Series) -> float:
    s = rank_ic_daily.dropna()
    if s.empty:
        return np.nan
    return float((s > 0).mean())


def run_universe_backtest(
    factor_panel: pd.DataFrame,
    factor_name: str,
    *,
    start_day: dt.datetime,
    end_day: dt.datetime,
    session,
    df_not_limit,
    df_not_st,
    df_trade_status,
    universes: Dict[str, Optional[str]],
    get_ret_matrix,
    n_groups: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Run groupTest across universes; return stats + ALL-universe decile/H-L artifacts."""
    import Factor_Dev_Lib

    rows = []
    all_group_means = None
    all_hl_cum = None
    all_ic_daily = None
    all_group_cum = None
    all_stats_title = ""

    for uni_name, idx in universes.items():
        ret = get_ret_matrix(start_day, end_day, idx)
        signal = prepare_signal(
            factor_panel.loc[start_day:end_day],
            idx,
            df_not_limit,
            df_not_st,
            df_trade_status,
            session,
            start_day,
            end_day,
        )
        if signal.empty:
            continue
        _, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(
            signal, ret, n=n_groups, info="silent"
        )
        plt.close("all")
        stats = compute_group_stats(signal, ret, group_pnl_df, group_to_df)
        rank_ic_daily = signal.corrwith(ret, axis=1, method="spearman")
        rows.append(
            {
                "factor_name": factor_name,
                "universe": uni_name,
                "rank_ic_mean": stats["rank_ic_mean"],
                "abs_rank_ic_mean": stats["abs_rank_ic_mean"],
                "icir": stats["icir"],
                "ic_positive_ratio": ic_positive_ratio(rank_ic_daily),
                "hl_sharpe": stats["hl_sharpe"],
                "hl_annu_ret": stats["hl_annu_ret"],
                "hl_mdd": stats["hl_mdd"],
                "hl_avg_turnover": stats["hl_avg_turnover"],
                "implied_annu_fee": stats["implied_annu_fee"],
                "direction": stats["direction"],
            }
        )
        if uni_name == "ALL":
            all_group_means = group_pnl_df.mean()
            daily_hl = group_pnl_df["H-L"] * stats["direction"]
            all_hl_cum = daily_hl.cumsum()
            all_ic_daily = rank_ic_daily
            all_group_cum = group_pnl_df.cumsum()
            all_stats_title = format_group_stats_title(stats)

    stats_df = pd.DataFrame(rows)
    quantile_df = (
        all_group_means.rename("mean_daily_return").reset_index()
        if all_group_means is not None
        else pd.DataFrame()
    )
    return stats_df, quantile_df, all_hl_cum, all_ic_daily, all_group_cum, all_stats_title


def build_factor_report(
    factor_name: str,
    factor_panel: pd.DataFrame,
    close: pd.DataFrame,
    *,
    start_day: dt.datetime,
    end_day: dt.datetime,
    session,
    df_not_limit,
    df_not_st,
    df_trade_status,
    universes: Dict[str, Optional[str]],
    get_ret_matrix: Callable,
) -> FactorReport:
    meta = FACTOR_META.get(factor_name, {})
    uni_stats, quantile_df, hl_cum, ic_daily, group_cum, stats_title = run_universe_backtest(
        factor_panel,
        factor_name,
        start_day=start_day,
        end_day=end_day,
        session=session,
        df_not_limit=df_not_limit,
        df_not_st=df_not_st,
        df_trade_status=df_trade_status,
        universes=universes,
        get_ret_matrix=get_ret_matrix,
    )
    ic_decay = rank_ic_by_horizon(factor_panel, close)

    all_row = uni_stats[uni_stats["universe"] == "ALL"]
    agg = {}
    if len(all_row):
        agg = all_row.iloc[0].to_dict()
    if len(quantile_df):
        gm = quantile_df.set_index(quantile_df.columns[0])["mean_daily_return"]
        agg["monotonicity_score"] = monotonicity_score(gm)

    return FactorReport(
        factor_name=factor_name,
        dimension=meta.get("dimension", ""),
        role=meta.get("role", ""),
        hypothesis=meta.get("hypothesis", ""),
        universe_stats=uni_stats,
        ic_decay=ic_decay,
        quantile_returns=quantile_df,
        hl_cum_pnl=hl_cum if hl_cum is not None else pd.Series(dtype=float),
        group_cum_pnl=group_cum if group_cum is not None else pd.DataFrame(),
        rank_ic_daily=ic_daily if ic_daily is not None else pd.Series(dtype=float),
        aggregate={**agg, "stats_title": stats_title},
    )


def save_cumulative_decile_figure(
    cum_pnl_df: pd.DataFrame,
    *,
    factor_name: str,
    stats_title: str = "",
    out_path: Path,
) -> None:
    """10 decile groups + H-L cumulative curves (matches Factor_Dev_Lib.groupTest style)."""
    if cum_pnl_df is None or cum_pnl_df.empty:
        return
    fig, ax = plt.subplots(figsize=(20, 12))
    for col_name, y in cum_pnl_df.items():
        ax.plot(y.index, y, label=str(col_name))
        ax.text(
            y.index[-1],
            y.iloc[-1],
            str(col_name),
            fontsize=12,
            verticalalignment="bottom",
        )
    ax.legend(loc="upper left")
    if stats_title:
        ax.set_xlabel(stats_title, fontsize=11)
    ax.set_title(f"{factor_name} — decile groups + H-L (ALL)")
    ax.set_ylabel("Cumulative return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_report_figures(report: FactorReport, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = report.factor_name

    if len(report.quantile_returns):
        fig, ax = plt.subplots(figsize=(8, 5))
        q = report.quantile_returns.copy()
        col = q.columns[0]
        q.set_index(col)["mean_daily_return"].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{name} — decile mean daily return (ALL)")
        ax.set_xlabel("Quantile group")
        ax.set_ylabel("Mean daily return")
        fig.tight_layout()
        fig.savefig(fig_dir / "quantile_return.png", dpi=150)
        plt.close(fig)

    if len(report.group_cum_pnl):
        save_cumulative_decile_figure(
            report.group_cum_pnl,
            factor_name=name,
            stats_title=str(report.aggregate.get("stats_title", "")),
            out_path=fig_dir / "cumulative_long_short.png",
        )
    elif len(report.hl_cum_pnl):
        # fallback if only H-L series available
        fig, ax = plt.subplots(figsize=(10, 5))
        report.hl_cum_pnl.plot(ax=ax, color="darkgreen", label="H-L")
        ax.legend()
        ax.set_title(f"{name} — cumulative H-L (ALL, direction-adjusted)")
        ax.set_ylabel("Cumulative return")
        fig.tight_layout()
        fig.savefig(fig_dir / "cumulative_long_short.png", dpi=150)
        plt.close(fig)

    if len(report.rank_ic_daily):
        fig, ax = plt.subplots(figsize=(10, 4))
        report.rank_ic_daily.plot(ax=ax, alpha=0.4, linewidth=0.8, label="daily IC")
        report.rank_ic_daily.rolling(20).mean().plot(ax=ax, color="red", label="20d MA")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{name} — rank IC time series (ALL)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "ic_timeseries.png", dpi=150)
        plt.close(fig)

    if len(report.ic_decay):
        fig, ax = plt.subplots(figsize=(6, 4))
        report.ic_decay.plot(x="horizon_days", y="rank_ic", kind="bar", ax=ax, legend=False)
        ax.set_title(f"{name} — IC decay")
        ax.set_ylabel("Mean rank IC")
        fig.tight_layout()
        fig.savefig(fig_dir / "ic_decay.png", dpi=150)
        plt.close(fig)


def render_markdown_report(report: FactorReport) -> str:
    lines = [
        f"# Alpha Report: `{report.factor_name}`",
        "",
        f"- **Dimension**: {report.dimension}",
        f"- **Role**: {report.role}",
        f"- **Hypothesis**: {report.hypothesis}",
        "",
        "## IC Statistics (by universe)",
        "",
    ]
    if len(report.universe_stats):
        cols = [
            "universe",
            "rank_ic_mean",
            "icir",
            "ic_positive_ratio",
            "hl_sharpe",
            "hl_annu_ret",
            "hl_avg_turnover",
        ]
        sub = report.universe_stats[[c for c in cols if c in report.universe_stats.columns]]
        lines.append(_df_to_markdown(sub))
    else:
        lines.append("_No universe stats._")

    lines.extend(["", "## ALL-universe summary", ""])
    for k, v in report.aggregate.items():
        if isinstance(v, float):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")

    if len(report.ic_decay):
        lines.extend(["", "## IC decay", ""])
        lines.append(_df_to_markdown(report.ic_decay[["horizon_days", "rank_ic"]]))

    if len(report.quantile_returns):
        lines.extend(["", "## Decile returns (ALL)", ""])
        lines.append(_df_to_markdown(report.quantile_returns, floatfmt=".6f"))
        mono = report.aggregate.get("monotonicity_score")
        if mono is not None and not pd.isna(mono):
            lines.append(f"\nMonotonicity score: **{mono:.2%}**")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `quantile_return.png`",
            "- `cumulative_long_short.png`",
            "- `ic_timeseries.png`",
            "- `ic_decay.png`",
            "",
        ]
    )
    return "\n".join(lines)


def publish_factor_report(report: FactorReport, out_root: Path) -> Path:
    factor_dir = out_root / report.factor_name
    report_dir = factor_dir / "report"
    fig_dir = factor_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    if len(report.universe_stats):
        report.universe_stats.to_csv(report_dir / "universe_stats.csv", index=False)
    if len(report.ic_decay):
        report.ic_decay.to_csv(report_dir / "ic_decay.csv", index=False)
    if len(report.quantile_returns):
        report.quantile_returns.to_csv(report_dir / "quantile_returns.csv", index=False)
    if len(report.rank_ic_daily):
        report.rank_ic_daily.to_csv(report_dir / "rank_ic_daily.csv", header=["rank_ic"])
    if len(report.hl_cum_pnl):
        report.hl_cum_pnl.to_csv(report_dir / "hl_cum_pnl.csv", header=["cum_hl"])
    if len(report.group_cum_pnl):
        report.group_cum_pnl.to_csv(report_dir / "group_cum_pnl.csv")

    save_report_figures(report, fig_dir)
    md_path = report_dir / "report.md"
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return md_path


def regenerate_markdown_from_csv(report_root: Path) -> pd.DataFrame:
    """Rebuild report.md + tier summary from saved CSV artifacts (no re-backtest)."""
    summary_rows = []
    for factor_dir in sorted(report_root.iterdir()):
        if not factor_dir.is_dir():
            continue
        report_dir = factor_dir / "report"
        if not (report_dir / "universe_stats.csv").exists():
            continue
        name = factor_dir.name
        meta = FACTOR_META.get(name, {})
        uni_stats = pd.read_csv(report_dir / "universe_stats.csv")
        ic_decay = pd.read_csv(report_dir / "ic_decay.csv") if (report_dir / "ic_decay.csv").exists() else pd.DataFrame()
        quantile = pd.read_csv(report_dir / "quantile_returns.csv") if (report_dir / "quantile_returns.csv").exists() else pd.DataFrame()
        hl_cum = pd.read_csv(report_dir / "hl_cum_pnl.csv", index_col=0) if (report_dir / "hl_cum_pnl.csv").exists() else pd.DataFrame()
        group_cum = pd.read_csv(report_dir / "group_cum_pnl.csv", index_col=0) if (report_dir / "group_cum_pnl.csv").exists() else pd.DataFrame()
        ic_daily = pd.read_csv(report_dir / "rank_ic_daily.csv", index_col=0) if (report_dir / "rank_ic_daily.csv").exists() else pd.DataFrame()

        agg = {}
        all_u = uni_stats[uni_stats["universe"] == "ALL"]
        if len(all_u):
            agg = all_u.iloc[0].to_dict()
        if len(quantile):
            col = quantile.columns[0]
            gm = quantile.set_index(col).iloc[:, 0]
            agg["monotonicity_score"] = monotonicity_score(gm)

        report = FactorReport(
            factor_name=name,
            dimension=meta.get("dimension", ""),
            role=meta.get("role", ""),
            hypothesis=meta.get("hypothesis", ""),
            universe_stats=uni_stats,
            ic_decay=ic_decay,
            quantile_returns=quantile,
            hl_cum_pnl=hl_cum.iloc[:, 0] if len(hl_cum.columns) else pd.Series(dtype=float),
            group_cum_pnl=group_cum,
            rank_ic_daily=ic_daily.iloc[:, 0] if len(ic_daily.columns) else pd.Series(dtype=float),
            aggregate=agg,
        )
        (report_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
        save_report_figures(report, factor_dir / "figures")
        summary_rows.append(report_summary_row(report))
    return pd.DataFrame(summary_rows)


def report_summary_row(report: FactorReport) -> dict:
    row = {
        "factor_name": report.factor_name,
        "dimension": report.dimension,
        "role": report.role,
    }
    if len(report.universe_stats):
        all_u = report.universe_stats[report.universe_stats["universe"] == "ALL"]
        if len(all_u):
            for c in ["rank_ic_mean", "icir", "hl_sharpe", "hl_annu_ret", "hl_avg_turnover"]:
                if c in all_u.columns:
                    row[c] = all_u.iloc[0][c]
        row["universe_ic_min"] = report.universe_stats["abs_rank_ic_mean"].min()
        row["universe_ic_max"] = report.universe_stats["abs_rank_ic_mean"].max()
    row["monotonicity_score"] = report.aggregate.get("monotonicity_score")
    row["ic_positive_ratio"] = report.aggregate.get("ic_positive_ratio")
    return row
