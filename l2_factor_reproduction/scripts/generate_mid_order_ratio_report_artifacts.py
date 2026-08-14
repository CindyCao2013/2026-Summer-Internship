#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate verified artifacts for the mid_order_ratio Research Pack.

This script does not rebuild raw tick data. It reuses the ClickHouse cumulative
amount-bucket cache, reconstructs the frozen L4w/H20w factor, then applies
point-in-time universe masks and the project's standard daily evaluation.
The cache must use strict execution records and a per-date regular-session filter.

Outputs
-------
research/reports/factors/mid_order_ratio/
├── artifacts/
│   ├── universe_comparison.csv
│   ├── neutralization_comparison.csv
│   ├── neutralization_by_universe.csv
│   ├── second_neutralization_comparison.csv
│   ├── parameter_sensitivity_csi1000_pit.csv
│   ├── state_dependence_summary.csv
│   ├── state_dependence_daily_ic.csv
│   ├── time_stability_monthly_ic.csv
│   ├── csi1000_decile_index_excess_daily.csv
│   ├── csi1000_decile_summary.csv
│   └── artifact_manifest.json
└── figures/
    └── report figures

Metric conventions
------------------
- Raw RankIC: daily cross-sectional Spearman of T-1 factor vs T raw c2c return.
- ICIR: mean/std(ddof=1)*sqrt(250).
- H-L: effective direction is frozen at -1 before evaluation;
  decile H-L = G10-G1, equal weighted.
- ALL artifact key: SSE/SZSE A-shares only; BSE is excluded.
- Universe comparison group returns: exact valid-universe EW excess.
- Main decile figures: CSI1000 index-excess c2c returns (explicitly labelled).
- MDD: compounded H-L return path; annual return: arithmetic mean*250.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import (  # noqa: E402
    calAnnuRet,
    calMDD,
    calSharpe,
    get_EOD_Not_Limit,
    get_EOD_Not_ST,
    get_Ret_Matrix,
    get_TradeStatus,
    get_index_member_mask,
    groupTest,
    panel_neutral_size_ind,
)
from l2_factor_reproduction.python.neutralization import neutralize_again  # noqa: E402
from l2_factor_reproduction.scripts.test_double_neutralization import (  # noqa: E402
    _get_turnover_wide,
)


DEFAULT_CACHE = (
    PROJ_ROOT
    / "research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity"
    / "tick_bucketed_strict_trade_2023-01-01_2024-06-30.parquet"
)
DEFAULT_OUTPUT = PROJ_ROOT / "research/reports/factors/mid_order_ratio"
RESULT_ROOT = PROJ_ROOT / "research/results/l2_reproduction/mid_order_ratio"

UNIVERSES = {
    "ALL": None,
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
}
UNIVERSE_DISPLAY = {
    "ALL": "SSE/SZSE A",
    "CSI300": "CSI300",
    "CSI500": "CSI500",
    "CSI1000": "CSI1000",
}
L_GRID = [20_000, 30_000, 40_000, 50_000, 60_000]
H_GRID = [100_000, 150_000, 200_000, 250_000, 300_000]
ORDER_BOUNDS = [20_000, 30_000, 40_000, 50_000, 60_000, 100_000, 150_000, 200_000, 250_000, 300_000]
FROZEN_EFFECTIVE_DIRECTION = -1
STYLE_WARMUP_CALENDAR_DAYS = 60
STRICT_SESSION = "09:30:00 <= ExchTime < 15:00:01 on every TradeDate"
STRICT_SSE_FILTER = "Type='T'"
STRICT_SZSE_FILTER = "Type='011' AND BidOrderNo>0 AND AskOrderNo>0"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "axes.unicode_minus": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _fmt_date(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _savefig(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _label_bars(
    ax: plt.Axes,
    bars,
    labels: Iterable[str],
    *,
    fontsize: int = 8,
    rotation: int = 0,
) -> None:
    """Matplotlib<3.4 compatible replacement for ``Axes.bar_label``."""
    y0, y1 = ax.get_ylim()
    offset = max(abs(y1 - y0) * 0.012, 1e-9)
    for bar, label in zip(bars, labels):
        height = float(bar.get_height())
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=fontsize,
            rotation=rotation,
        )


def _is_a_share_symbol(series: pd.Series) -> pd.Series:
    """Conservative SSE/SZSE A-share filter before intersecting Wind returns."""
    return series.astype(str).str.match(r"^[036]\d{5}\.(SH|SZ)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_strict_bucket_cache(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, object]:
    """Verify that a bucket cache has matching strict-construction metadata."""
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Strict cache metadata is required but missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    expected = {
        "session": STRICT_SESSION,
        "sse_trade_filter": STRICT_SSE_FILTER,
        "szse_trade_filter": STRICT_SZSE_FILTER,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Strict cache metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )

    boundaries = set(map(int, metadata.get("boundaries_rmb", [])))
    missing_boundaries = sorted(set(ORDER_BOUNDS) - boundaries)
    if missing_boundaries:
        raise ValueError(
            f"Strict cache metadata is missing boundaries: {missing_boundaries}"
        )

    requested_start = pd.Timestamp(metadata["requested_start"])
    requested_end = pd.Timestamp(metadata["requested_end"])
    if requested_start > start or requested_end < end:
        raise ValueError(
            "Strict cache date coverage does not contain the requested report sample: "
            f"{requested_start.date()}~{requested_end.date()} vs "
            f"{start.date()}~{end.date()}"
        )

    metadata_output = Path(str(metadata.get("output", ""))).resolve()
    if metadata_output != path.resolve():
        raise ValueError(
            f"Strict cache metadata output mismatch: {metadata_output} != {path.resolve()}"
        )

    actual_sha256 = _sha256(path)
    if metadata.get("sha256") != actual_sha256:
        raise ValueError(
            "Strict cache SHA256 mismatch: "
            f"{metadata.get('sha256')} != {actual_sha256}"
        )

    return {
        **metadata,
        "metadata_path": str(metadata_path.resolve()),
        "validated_sha256": actual_sha256,
    }


def load_bucket_cache(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = ["symbol", "TradeDate", "TotalAmount"]
    required += [f"cum_{b}" for b in ORDER_BOUNDS]
    df = pd.read_parquet(path, columns=required)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
    df = df[df["TradeDate"].between(start, end)]
    df = df[_is_a_share_symbol(df["symbol"])].copy()
    for col in required[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def factor_wide_from_bucket(bucket: pd.DataFrame, lower: int, upper: int) -> pd.DataFrame:
    med = bucket[f"cum_{upper}"] - bucket[f"cum_{lower}"]
    value = med / bucket["TotalAmount"].replace(0, np.nan)
    long = pd.DataFrame(
        {
            "Date": bucket["TradeDate"],
            "symbol": bucket["symbol"].astype(str),
            "value": value,
        }
    ).dropna(subset=["value"])
    wide = long.pivot_table(index="Date", columns="symbol", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index).normalize()
    return wide.sort_index()


def write_construction_crosscheck(factor_wide: pd.DataFrame, artifacts: Path) -> None:
    """Quantify how the corrected strict-trade panel differs from the legacy panel."""
    source = RESULT_ROOT / "factor_narrow.parquet"
    base = pd.read_parquet(source, columns=["symbol", "tradetime", "value"])
    base["TradeDate"] = pd.to_datetime(base["tradetime"]).dt.normalize()
    strict = factor_wide.stack(dropna=True).rename("strict_value").reset_index()
    strict.columns = ["TradeDate", "symbol", "strict_value"]
    merged = base.rename(columns={"value": "legacy_value"}).merge(
        strict, on=["symbol", "TradeDate"], how="inner"
    )
    merged["abs_diff"] = (merged["legacy_value"] - merged["strict_value"]).abs()
    diff = merged["abs_diff"]
    summary = {
        "legacy_source_factor_narrow": str(source),
        "legacy_rows": int(len(base)),
        "matched_rows": int(len(merged)),
        "match_share": float(len(merged) / len(base)) if len(base) else np.nan,
        "pearson": float(merged["legacy_value"].corr(merged["strict_value"])),
        "spearman": float(
            merged["legacy_value"].corr(merged["strict_value"], method="spearman")
        ),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(diff.median()),
        "p99_abs_diff": float(diff.quantile(0.99)),
        "max_abs_diff": float(diff.max()),
        "exact_within_1e_12_share": float((diff <= 1e-12).mean()),
        "definition": "(cum_200000-cum_40000)/TotalAmount",
        "interpretation": (
            "This is an impact audit, not an equivalence check. The strict panel applies "
            "the regular-session predicate to every date and identifies SZSE executions "
            "with Type='011' plus both order numbers. Report metrics use only the strict cache."
        ),
    }
    (artifacts / "construction_crosscheck.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    merged.nlargest(100, "abs_diff")[
        ["symbol", "TradeDate", "legacy_value", "strict_value", "abs_diff"]
    ].to_csv(artifacts / "construction_crosscheck_top100.csv", index=False)
    strict["strict_value"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).to_frame("value").to_csv(artifacts / "factor_value_distribution_summary.csv")


def align_core_panels(
    factor: pd.DataFrame,
    ret_raw: pd.DataFrame,
    tradable: pd.DataFrame,
    member: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply day-T factor-date masks, then shift signal to predict day T+1."""
    idx = factor.index.intersection(ret_raw.index).intersection(tradable.index)
    idx = idx[(idx >= start) & (idx <= end)]
    cols = factor.columns.intersection(ret_raw.columns).intersection(tradable.columns)
    if member is not None:
        idx = idx.intersection(member.index)
        cols = cols.intersection(member.columns)

    factor_a = factor.reindex(index=idx, columns=cols)
    ret_a = ret_raw.reindex(index=idx, columns=cols)
    tradable_a = tradable.reindex(index=idx, columns=cols)
    signal_unshifted = factor_a.where(tradable_a == 1)
    if member is not None:
        member_a = member.reindex(index=idx, columns=cols)
        signal_unshifted = signal_unshifted.where(member_a == 1)

    signal = signal_unshifted.shift(1)
    valid_count = (signal.notna() & ret_a.notna()).sum(axis=1)
    keep_days = valid_count >= 20
    signal = signal.loc[keep_days]
    ret_a = ret_a.reindex_like(signal)
    keep_cols = (signal.notna() & ret_a.notna()).any(axis=0)
    return signal.loc[:, keep_cols], ret_a.loc[:, keep_cols]


def evaluate_prepared(
    signal_raw: pd.DataFrame,
    ret_raw: pd.DataFrame,
    effective_direction: int = FROZEN_EFFECTIVE_DIRECTION,
) -> Dict[str, object]:
    """Evaluate a pre-shifted signal using a direction frozen before evaluation."""
    if effective_direction not in (-1, 1):
        raise ValueError("effective_direction must be -1 or 1")

    rank_ic = signal_raw.corrwith(ret_raw, axis=1, method="spearman").dropna()
    ic_mean = float(rank_ic.mean())
    ic_std = float(rank_ic.std(ddof=1))
    icir = ic_mean / ic_std * math.sqrt(250) if ic_std > 0 else np.nan
    factor_direction = int(effective_direction)
    signal_eff = signal_raw * factor_direction

    valid = signal_eff.notna() & ret_raw.notna()
    universe_ew = ret_raw.where(valid).mean(axis=1)
    ret_ew_excess = ret_raw.sub(universe_ew, axis=0)
    _, pnl, turnover = groupTest(signal_eff, ret_ew_excess, n=10, info="silent")

    hl = pnl["H-L"].dropna()
    mdd, _ = calMDD(hl)

    # Frozen delivery diagnostic: effective G10 vs exact valid-universe EW.
    ranks = signal_eff.rank(axis=1, pct=True, method="first")
    g10_mask = valid & (ranks > 0.9)
    g10_raw = ret_raw.where(g10_mask).mean(axis=1)
    g10_excess = g10_raw - universe_ew
    g10_mdd, _ = calMDD(g10_excess)

    valid_dates = rank_ic.index
    row = {
        "date_start": _fmt_date(valid_dates.min()),
        "date_end": _fmt_date(valid_dates.max()),
        "n_days": int(rank_ic.shape[0]),
        "n_names_avg": float(valid.sum(axis=1).mean()),
        "n_names_min": int(valid.sum(axis=1).min()),
        "n_names_max": int(valid.sum(axis=1).max()),
        "rank_ic": ic_mean,
        "rank_ic_std": ic_std,
        "rank_ic_tstat": float(ic_mean / (ic_std / math.sqrt(len(rank_ic)))) if ic_std > 0 else np.nan,
        "icir": float(icir),
        "effective_direction": int(factor_direction),
        "effective_ic_positive_day_share": float((rank_ic * factor_direction > 0).mean()),
        "effective_hl_mean_positive": bool(hl.mean() > 0),
        "hl_annu_ret": float(calAnnuRet(hl)),
        "hl_sharpe": float(calSharpe(hl)),
        "hl_mdd": float(mdd),
        "hl_turnover": float(turnover["H-L"].mean()),
        "g10_exact_ew_excess_annu_ret": float(calAnnuRet(g10_excess)),
        "g10_exact_ew_excess_sharpe": float(calSharpe(g10_excess)),
        "g10_exact_ew_excess_mdd": float(g10_mdd),
        "g10_count_avg": float(g10_mask.sum(axis=1).mean()),
    }
    return {
        "summary": row,
        "signal_raw": signal_raw,
        "signal_effective": signal_eff,
        "ret_raw": ret_raw,
        "ret_ew_excess": ret_ew_excess,
        "universe_ew": universe_ew,
        "rank_ic": rank_ic,
        "group_pnl": pnl,
        "group_turnover": turnover,
        "g10_excess": g10_excess,
    }


def evaluate_factor(
    factor: pd.DataFrame,
    ret_raw: pd.DataFrame,
    tradable: pd.DataFrame,
    member: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, object]:
    signal, ret = align_core_panels(factor, ret_raw, tradable, member, start, end)
    return evaluate_prepared(signal, ret)


def plot_pipeline(path: Path) -> None:
    steps = [
        "L2 Tick Data\nSSE / SZSE",
        "Transaction Filtering\nregular session + true executions",
        "Trade-Size Bucket\nRMB (40k, 200k]",
        "Daily Factor\nbucket amount / total amount",
        "Signal Lag\nshift(1)",
        "Next-Day Return\nclose-to-close / RankIC",
    ]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, text in enumerate(steps):
        x = i + 0.5
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.55", "fc": "#eef3f8", "ec": "#4c6478", "lw": 1.1},
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(i + 1.08, 0.55),
                xytext=(i + 0.92, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#4c6478"},
            )
    ax.set_title(
        "mid_order_ratio — Factor Construction and Validation Alignment",
        fontsize=14,
        pad=18,
    )
    _savefig(
        fig,
        path,
        "Source: project code audit · strict ClickHouse execution records · "
        "daily factor frozen after market close · T-1 signal evaluated on T return.",
    )


def plot_universe_table(df: pd.DataFrame, path: Path) -> None:
    show = pd.DataFrame(
        {
            "Universe": df["universe"].map(UNIVERSE_DISPLAY).fillna(df["universe"]),
            "Avg names": df["n_names_avg"].map(lambda x: f"{x:,.0f}"),
            "Date range": df["date_start"] + "\n" + df["date_end"],
            "RankIC": df["rank_ic"].map(lambda x: f"{x:.2%}"),
            "ICIR": df["icir"].map(lambda x: f"{x:.2f}"),
            "H-L Sharpe": df["hl_sharpe"].map(lambda x: f"{x:.2f}"),
            "H-L MDD": df["hl_mdd"].map(lambda x: f"{x:.2%}"),
            "H-L TO": df["hl_turnover"].map(lambda x: f"{x:.2f}"),
        }
    )
    fig, ax = plt.subplots(figsize=(13.5, 3.4))
    ax.axis("off")
    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.65)
    for j in range(len(show.columns)):
        table[(0, j)].set_facecolor("#dce7f0")
        table[(0, j)].set_text_props(weight="bold")
    ax.set_title(
        "mid_order_ratio — Point-in-Time Universe Comparison\n"
        "RankIC is raw direction; H-L is effective direction and exact-universe-EW excess",
        fontsize=13,
        pad=14,
    )
    _savefig(
        fig,
        path,
        "Source: CH strict-trade L4w/H20w cache + Wind PIT membership/returns · daily c2c · T-1 signal · 2023-01-01 to 2024-06-30.",
    )


def plot_order_size_distribution(bucket: pd.DataFrame, member: pd.DataFrame, path: Path) -> pd.DataFrame:
    member_long = (
        member.stack(dropna=True)
        .rename("member")
        .reset_index()
        .rename(columns={member.index.name or "level_0": "TradeDate", member.columns.name or "level_1": "symbol"})
    )
    # Handle unnamed index/column axes across pandas versions.
    member_long.columns = ["TradeDate", "symbol", "member"]
    member_long["TradeDate"] = pd.to_datetime(member_long["TradeDate"]).dt.normalize()
    member_long = member_long[member_long["member"] == 1]
    use_cols = ["TradeDate", "symbol", "TotalAmount"] + [f"cum_{b}" for b in ORDER_BOUNDS]
    selected = bucket[use_cols].merge(
        member_long[["TradeDate", "symbol"]], on=["TradeDate", "symbol"], how="inner"
    )

    rows = []
    previous = pd.Series(0.0, index=selected.index)
    previous_bound = 0
    for bound in ORDER_BOUNDS:
        current = selected[f"cum_{bound}"]
        amount = float((current - previous).clip(lower=0).sum())
        label = f"≤{bound/1000:.0f}k" if previous_bound == 0 else f"({previous_bound/1000:.0f}k,{bound/1000:.0f}k]"
        rows.append({"order_amount_bin_rmb": label, "traded_amount": amount})
        previous = current
        previous_bound = bound
    tail = float((selected["TotalAmount"] - selected[f"cum_{ORDER_BOUNDS[-1]}"]).clip(lower=0).sum())
    rows.append({"order_amount_bin_rmb": ">300k", "traded_amount": tail})
    out = pd.DataFrame(rows)
    out["amount_share"] = out["traded_amount"] / out["traded_amount"].sum()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(out["order_amount_bin_rmb"], out["amount_share"] * 100, color="#4c78a8")
    _label_bars(ax, bars, [f"{v:.1f}%" for v in out["amount_share"] * 100])
    ax.set_title("CSI1000 PIT — Amount-Weighted Trade-Print Size Distribution")
    ax.set_xlabel("Single-print RMB amount bin")
    ax.set_ylabel("Share of total traded amount (%)")
    ax.tick_params(axis="x", rotation=35)
    _savefig(
        fig,
        path,
        "Source: ClickHouse strict-execution cumulative amount buckets · CSI1000 PIT stock-days · 2023-01-03 to 2024-06-28 · amount-weighted.",
    )
    return out


def plot_decile_index_excess(
    signal_effective: pd.DataFrame,
    ret_index_excess: pd.DataFrame,
    out_artifacts: Path,
    out_figures: Path,
    effective_direction: int,
) -> None:
    effective_name = "-mid_order_ratio" if effective_direction == -1 else "mid_order_ratio"
    ret = ret_index_excess.reindex_like(signal_effective)
    _, pnl, turnover = groupTest(signal_effective, ret, n=10, info="silent")
    pnl.to_csv(out_artifacts / "csi1000_decile_index_excess_daily.csv")
    turnover.to_csv(out_artifacts / "csi1000_decile_turnover_daily.csv")

    order = [str(i) for i in range(1, 11)] + ["H-L"]
    pnl_plot = pnl.copy()
    pnl_plot.columns = [str(c) for c in pnl_plot.columns]
    decile_cols = [str(i) for i in range(1, 11)]
    annu_fraction = pnl_plot[decile_cols].mean() * 250
    monotonicity = pd.Series(
        range(1, 11), index=decile_cols, dtype=float
    ).corr(annu_fraction, method="spearman")
    decile_summary = pd.DataFrame(
        {
            "effective_decile": [f"G{i}" for i in range(1, 11)],
            "annualized_csi1000_index_excess_return": annu_fraction.to_numpy(),
            "full_sample_monotonicity_spearman": float(monotonicity),
            "g10_minus_g1_annualized_return": float(
                annu_fraction["10"] - annu_fraction["1"]
            ),
        }
    )
    decile_summary.to_csv(
        out_artifacts / "csi1000_decile_summary.csv", index=False
    )

    cumulative = pnl_plot[order].cumsum() * 100
    fig, ax = plt.subplots(figsize=(14, 7.5))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 10))
    for i, col in enumerate(order[:-1]):
        ax.plot(cumulative.index, cumulative[col], color=colors[i], lw=1.0, label=f"G{col}")
    ax.plot(cumulative.index, cumulative["H-L"], color="black", lw=2.4, label="H-L (effective)")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_title(
        "CSI1000 PIT — Decile Performance Using CSI1000 Index-Excess Return\n"
        f"Cumulative sum of daily c2c excess returns; effective factor = {effective_name}"
    )
    ax.set_xlabel("Return date")
    ax.set_ylabel("Cumulative sum of daily index-excess return (%)")
    ax.legend(ncol=4, fontsize=8)
    _savefig(
        fig,
        out_figures / "04_decile_cumulative_csi1000_index_excess.png",
        "Source: strict-trade factor + Wind c2c stock return minus 000852.SH c2c index return · PIT members · T-1 signal · fee=0.",
    )

    annu = annu_fraction * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar([f"G{i}" for i in range(1, 11)], annu.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    _label_bars(ax, bars, [f"{v:.1f}%" for v in annu.values])
    ax.set_title(
        "CSI1000 PIT — Decile Annualized CSI1000 Index-Excess Return\n"
        f"Arithmetic mean daily excess return × 250; effective factor = {effective_name}"
    )
    ax.set_xlabel("Effective-signal decile (G1 low, G10 high)")
    ax.set_ylabel("Annualized index-excess return (%)")
    _savefig(
        fig,
        out_figures / "05_decile_annualized_csi1000_index_excess.png",
        "Source: same daily series as cumulative decile figure · arithmetic annualization · fee=0.",
    )


def plot_ic_suite(rank_ic: pd.Series, out_artifacts: Path, out_figures: Path) -> None:
    ic = rank_ic.sort_index().dropna()
    ic.to_frame("rank_ic_raw").to_csv(out_artifacts / "csi1000_rank_ic_daily.csv")
    monthly = ic.groupby(ic.index.to_period("M")).agg(["mean", "std", "count"])
    monthly.index = monthly.index.astype(str)
    monthly.to_csv(out_artifacts / "time_stability_monthly_ic.csv")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(ic.index, ic.values, lw=0.65, color="#7f8c8d", alpha=0.8, label="Daily raw RankIC")
    ax.axhline(float(ic.mean()), color="#c44e52", lw=1.6, label=f"Mean = {ic.mean():.2%}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("CSI1000 PIT — Daily Raw RankIC (T-1 Mid Order Ratio vs T Return)")
    ax.set_xlabel("Return date")
    ax.set_ylabel("Spearman RankIC")
    ax.legend()
    _savefig(
        fig,
        out_figures / "06_ic_time_series.png",
        "Source: strict-trade raw mid_order_ratio and raw stock c2c returns · PIT members · "
        "effective direction frozen at -1 before evaluation.",
    )

    rolling = ic.rolling(63, min_periods=21).mean()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(ic.index, ic.values, lw=0.35, color="#b8b8b8", alpha=0.55, label="Daily RankIC")
    ax.plot(rolling.index, rolling.values, lw=2.0, color="#4c78a8", label="63-trading-day mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("CSI1000 PIT — Raw RankIC with 63-Trading-Day Rolling Mean")
    ax.set_xlabel("Return date")
    ax.set_ylabel("Spearman RankIC")
    ax.legend()
    _savefig(
        fig,
        out_figures / "07_rolling_ic.png",
        "Source: same daily RankIC series · 63-day rolling mean, minimum 21 observations.",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(ic.values, bins=45, color="#4c78a8", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.axvline(ic.mean(), color="#c44e52", lw=1.8, label=f"Mean = {ic.mean():.2%}")
    ax.set_title("CSI1000 PIT — Distribution of Daily Raw RankIC")
    ax.set_xlabel("Daily Spearman RankIC")
    ax.set_ylabel("Number of trading days")
    ax.legend()
    _savefig(
        fig,
        out_figures / "13_ic_distribution.png",
        "Source: same daily RankIC series · negative values support the sign-flipped factor orientation.",
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    values = monthly["mean"] * 100
    colors = ["#4c78a8" if v < 0 else "#c44e52" for v in values]
    bars = ax.bar(monthly.index, values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    _label_bars(ax, bars, [f"{v:.1f}%" for v in values], fontsize=7, rotation=90)
    ax.set_title("CSI1000 PIT — Monthly Mean Raw RankIC")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly mean RankIC (%)")
    ax.tick_params(axis="x", rotation=45)
    _savefig(
        fig,
        out_figures / "14_monthly_ic.png",
        "Source: daily raw RankIC aggregated by calendar month · blue = hypothesized negative direction.",
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=False)
    axes[0].plot(
        ic.index,
        ic.values,
        lw=0.55,
        color="#7f8c8d",
        alpha=0.75,
        label="Daily raw RankIC",
    )
    axes[0].axhline(float(ic.mean()), color="#c44e52", lw=1.5, label=f"Mean = {ic.mean():.2%}")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_ylabel("Daily RankIC")
    axes[0].legend(loc="upper right")

    axes[1].plot(ic.index, ic.values, lw=0.3, color="#b8b8b8", alpha=0.45)
    axes[1].plot(
        rolling.index,
        rolling.values,
        lw=2.0,
        color="#4c78a8",
        label="63-trading-day mean",
    )
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Rolling RankIC")
    axes[1].legend(loc="upper right")

    axes[2].bar(monthly.index, values, color=colors)
    axes[2].axhline(0, color="black", lw=0.7)
    axes[2].set_ylabel("Monthly mean (%)")
    axes[2].set_xlabel("Return month")
    axes[2].tick_params(axis="x", rotation=45)
    fig.suptitle(
        "CSI1000 PIT — Raw RankIC Stability Diagnostics\n"
        "Daily observations, 63-trading-day rolling mean, and monthly averages",
        fontsize=14,
    )
    _savefig(
        fig,
        out_figures / "07b_ic_stability_combined.png",
        "Source: strict-trade raw mid_order_ratio · CSI1000 PIT members · T-1 signal · "
        "daily Spearman RankIC; negative is the raw-factor predictive direction.",
    )


def plot_neutralization_comparison(df: pd.DataFrame, path: Path) -> None:
    labels = df["method"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("rank_ic", "Raw-direction RankIC (%)", 100),
        ("icir", "Annualized ICIR", 1),
        ("hl_sharpe", "Effective H-L gross Sharpe", 1),
        ("hl_mdd", "Effective H-L MDD (%)", 100),
    ]
    for ax, (col, title, scale) in zip(axes.flat, specs):
        values = df[col] * scale
        bars = ax.bar(labels, values, color=["#4c78a8", "#72a0c1", "#9ab8cf", "#c1d1df"])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        fmt = "{:.2f}%" if scale == 100 else "{:.2f}"
        _label_bars(ax, bars, [fmt.format(v) for v in values])
    fig.suptitle(
        "CSI1000 PIT — Neutralization Comparison\n"
        "Known industry/size components are removed by daily cross-sectional OLS",
        fontsize=13,
    )
    _savefig(
        fig,
        path,
        "Source: strict-trade raw / industry / market-cap / industry+cap panels · PIT CSI1000 · exact-universe-EW H-L · T-1 signal.",
    )


def plot_universe_neutralization_matrix(df: pd.DataFrame, path: Path) -> None:
    """Plot 4 universes × 4 within-universe neutralization variants."""
    universes = ["ALL", "CSI300", "CSI500", "CSI1000"]
    methods = ["raw", "industry", "market_cap", "industry+market_cap"]
    method_labels = ["Raw", "Industry", "Cap", "Industry+Cap"]
    specs = [
        ("rank_ic", "Raw-direction RankIC (%)", 100, "coolwarm_r", "{:.2f}"),
        ("icir", "Annualized ICIR", 1, "coolwarm_r", "{:.2f}"),
        ("hl_sharpe", "Effective H-L gross Sharpe", 1, "viridis", "{:.2f}"),
        (
            "g10_exact_ew_excess_sharpe",
            "Effective G10 exact-EW excess Sharpe",
            1,
            "viridis",
            "{:.2f}",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (metric, title, scale, cmap, value_fmt) in zip(axes.flat, specs):
        pivot = (
            df.pivot(index="universe", columns="method", values=metric)
            .reindex(index=universes, columns=methods)
            * scale
        )
        values = pivot.to_numpy(dtype=float)
        image = ax.imshow(values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels, rotation=20, ha="right")
        ax.set_yticks(range(len(universes)))
        ax.set_yticklabels([UNIVERSE_DISPLAY.get(name, name) for name in universes])
        ax.set_title(title)
        midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                color = "white" if value < midpoint else "black"
                ax.text(
                    col,
                    row,
                    value_fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Within-Universe PIT Neutralization Robustness Matrix\n"
        "Each daily regression is re-estimated inside its own stock universe",
        fontsize=13,
    )
    _savefig(
        fig,
        path,
        "Source: strict-trade factor · local daily OLS within each PIT universe · "
        "industry / log-cap controls · exact-universe-EW returns · T-1 signal · fee=0.",
    )


def plot_second_neutralization(df: pd.DataFrame, out_csv: Path, path: Path) -> None:
    df = df.copy()
    for col in ["rank_ic", "icir", "abs_icir", "sharpe"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4c78a8", "#72a0c1", "#9ab8cf", "#c1d1df"]
    values = df["icir"]
    bars = axes[0].bar(df["variant"], values, color=colors[: len(df)])
    axes[0].axhline(0, color="black", lw=0.7)
    _label_bars(axes[0], bars, [f"{v:.2f}" for v in values])
    axes[0].set_title("Annualized ICIR (raw direction)")
    axes[0].set_ylabel("ICIR; more negative = stronger inverse relation")
    axes[0].tick_params(axis="x", rotation=25)

    values_ic = df["rank_ic"] * 100
    bars = axes[1].bar(df["variant"], values_ic, color=colors[: len(df)])
    axes[1].axhline(0, color="black", lw=0.7)
    _label_bars(axes[1], bars, [f"{v:.2f}%" for v in values_ic])
    axes[1].set_title("Mean RankIC (raw direction)")
    axes[1].set_ylabel("RankIC (%)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle(
        "Second-Order Neutralization after Industry + Market Cap\n"
        "A decline in |IC| indicates information overlap, not factor invalidity",
        fontsize=13,
    )
    _savefig(
        fig,
        path,
        "Source: strict-trade factor · CSI1000 PIT · 20-day momentum, volatility and "
        "log-turnover controls with pre-sample warmup · T-1 signal · matched-sample "
        "baselines stored in the CSV.",
    )


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    path: Path,
    value_fmt: str,
    cmap: str,
) -> None:
    pivot = df.pivot(index="L_wan", columns="H_wan", values=value_col).sort_index()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.0f}w" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.0f}w" for x in pivot.index])
    ax.set_xlabel("Upper bound H (RMB ten-thousand)")
    ax.set_ylabel("Lower bound L (RMB ten-thousand)")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, value_fmt.format(pivot.iloc[i, j]), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    _savefig(
        fig,
        path,
        "Source: ClickHouse strict-execution cumulative amount buckets over CSI1000 PIT members · daily c2c · T-1 signal · 2023-01-01 to 2024-06-30.",
    )


def run_parameter_grid(
    bucket: pd.DataFrame,
    member: pd.DataFrame,
    ret_raw: pd.DataFrame,
    tradable: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    symbols = set(member.columns)
    use = bucket[bucket["symbol"].isin(symbols)].copy()
    rows = []
    for lower in L_GRID:
        for upper in H_GRID:
            factor = factor_wide_from_bucket(use, lower, upper)
            result = evaluate_factor(factor, ret_raw, tradable, member, start, end)
            s = result["summary"]
            rows.append(
                {
                    "L_wan": lower / 10_000,
                    "H_wan": upper / 10_000,
                    "combo": f"L{lower//10_000}w_H{upper//10_000}w",
                    "rank_ic": s["rank_ic"],
                    "icir": s["icir"],
                    "hl_annu_ret": s["hl_annu_ret"],
                    "hl_sharpe": s["hl_sharpe"],
                    "hl_mdd": s["hl_mdd"],
                    "hl_turnover": s["hl_turnover"],
                    "n_names_avg": s["n_names_avg"],
                }
            )
            print(
                f"grid L={lower//10000}w H={upper//10000}w "
                f"IC={s['rank_ic']:.4f} ICIR={s['icir']:.2f} Sharpe={s['hl_sharpe']:.2f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def compute_state_dependence(
    signal_raw: pd.DataFrame,
    ret_raw: pd.DataFrame,
    turnover_state: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    state = turnover_state.shift(1).reindex_like(signal_raw)
    rows = []
    for dt in signal_raw.index:
        s = signal_raw.loc[dt]
        r = ret_raw.loc[dt]
        t = state.loc[dt]
        valid = s.notna() & r.notna() & t.notna()
        if int(valid.sum()) < 60:
            continue
        sv, rv, tv = s[valid], r[valid], t[valid]
        q1, q2 = tv.quantile(1 / 3), tv.quantile(2 / 3)
        groups = {
            "Low": tv.index[tv <= q1],
            "Mid": tv.index[(tv > q1) & (tv <= q2)],
            "High": tv.index[tv > q2],
        }
        row = {"date": dt}
        for name, idx in groups.items():
            row[name] = sv.loc[idx].corr(rv.loc[idx], method="spearman") if len(idx) >= 20 else np.nan
        rows.append(row)
    daily = pd.DataFrame(rows).set_index("date")
    summary_rows = []
    for name in ["Low", "Mid", "High"]:
        values = daily[name].dropna()
        std = values.std(ddof=1)
        summary_rows.append(
            {
                "turnover_tercile": name,
                "rank_ic": float(values.mean()),
                "rank_ic_std": float(std),
                "icir": float(values.mean() / std * math.sqrt(250)) if std > 0 else np.nan,
                "negative_ic_day_share": float((values < 0).mean()),
                "n_days": int(len(values)),
            }
        )
    return daily, pd.DataFrame(summary_rows)


def plot_state_dependence(daily: pd.DataFrame, summary: pd.DataFrame, out_figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = summary["rank_ic"] * 100
    colors = ["#8db3d3", "#5f8fb5", "#2f648d"]
    bars = ax.bar(summary["turnover_tercile"], vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    lower = min(float(vals.min()) * 1.22, -0.25)
    upper = max(float(vals.max()) * 0.15, 0.15)
    ax.set_ylim(lower, upper)
    labels = [
        f"IC {ic:.2f}%\nICIR {ir:.2f}" for ic, ir in zip(vals, summary["icir"])
    ]
    _label_bars(ax, bars, labels)
    ax.set_title(
        "CSI1000 PIT — Predictive RankIC by Lagged 20-Day Turnover Tercile\n"
        "Tests when mid_order_ratio works better"
    )
    ax.set_xlabel("Cross-sectional turnover tercile (state known at T-1)")
    ax.set_ylabel("Mean raw-direction RankIC (%)")
    _savefig(
        fig,
        out_figures / "11_turnover_vs_ic_relationship.png",
        "Source: strict-trade factor + Wind S_DQ_TURN, 20-day mean then log, lagged one day · within-tercile daily Spearman IC · 2023-01-01 to 2024-06-30.",
    )

    fig, ax = plt.subplots(figsize=(9, 5.8))
    data = [daily[c].dropna().values for c in ["Low", "Mid", "High"]]
    bp = ax.boxplot(data, labels=["Low turnover", "Mid turnover", "High turnover"], showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_title("CSI1000 PIT — Daily Raw RankIC Distribution by Lagged Turnover Regime")
    ax.set_xlabel("Lagged 20-day turnover state")
    ax.set_ylabel("Daily within-regime Spearman RankIC")
    _savefig(
        fig,
        out_figures / "12_high_low_turnover_regime_comparison.png",
        "Source: same state-dependence daily IC panel · box = interquartile range · triangle = mean.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--bucket-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    output_root = args.output_root.resolve()
    artifacts = output_root / "artifacts"
    figures = output_root / "figures"
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    cache_path = args.bucket_cache.resolve()
    print(f"Validating strict bucket cache: {cache_path}", flush=True)
    cache_metadata = validate_strict_bucket_cache(cache_path, start, end)
    print(
        f"Validated cache SHA256: {cache_metadata['validated_sha256']}",
        flush=True,
    )
    bucket = load_bucket_cache(cache_path, start, end)
    factor_raw = factor_wide_from_bucket(bucket, 40_000, 200_000)
    print(
        f"Factor panel: {factor_raw.shape}, {factor_raw.index.min().date()}~{factor_raw.index.max().date()}",
        flush=True,
    )
    write_construction_crosscheck(factor_raw, artifacts)

    print("Loading Wind returns and tradability masks from DolphinDB...", flush=True)
    style_warmup_start = start - pd.Timedelta(days=STYLE_WARMUP_CALENDAR_DAYS)
    ret_raw_with_warmup = get_Ret_Matrix(
        style_warmup_start, end, method="c2c", base_index=None
    )
    ret_raw = ret_raw_with_warmup.loc[
        (ret_raw_with_warmup.index >= start)
        & (ret_raw_with_warmup.index <= end)
    ]
    ret_csi1000_excess = get_Ret_Matrix(start, end, method="c2c", base_index="000852.SH")
    tradable = get_EOD_Not_Limit(start, end) * get_EOD_Not_ST(start, end) * get_TradeStatus(start, end)
    members = {
        name: (None if code is None else get_index_member_mask(code, start, end))
        for name, code in UNIVERSES.items()
    }

    print("Running point-in-time universe comparison...", flush=True)
    universe_results: Dict[str, Dict[str, object]] = {}
    universe_rows = []
    for name in UNIVERSES:
        result = evaluate_factor(factor_raw, ret_raw, tradable, members[name], start, end)
        universe_results[name] = result
        row = {"universe": name, "benchmark": "exact_valid_universe_EW", **result["summary"]}
        universe_rows.append(row)
        result["rank_ic"].to_frame("rank_ic_raw").to_csv(artifacts / f"rank_ic_{name.lower()}.csv")
        result["group_pnl"].to_csv(artifacts / f"group_pnl_{name.lower()}_ew_excess.csv")
        result["group_turnover"].to_csv(artifacts / f"group_turnover_{name.lower()}.csv")
        print(
            f"{name}: n={row['n_names_avg']:.0f} IC={row['rank_ic']:.4f} "
            f"ICIR={row['icir']:.2f} Sharpe={row['hl_sharpe']:.2f}",
            flush=True,
        )
    universe_df = pd.DataFrame(universe_rows)
    universe_df.to_csv(artifacts / "universe_comparison.csv", index=False)

    plot_pipeline(figures / "01_pipeline_architecture.png")
    order_dist = plot_order_size_distribution(bucket, members["CSI1000"], figures / "02_order_size_distribution.png")
    order_dist.to_csv(artifacts / "order_size_distribution.csv", index=False)
    plot_universe_table(universe_df, figures / "03_universe_comparison_table.png")

    csi = universe_results["CSI1000"]
    plot_decile_index_excess(
        csi["signal_effective"],
        ret_csi1000_excess,
        artifacts,
        figures,
        int(csi["summary"]["effective_direction"]),
    )
    plot_ic_suite(csi["rank_ic"], artifacts, figures)

    print("Loading turnover state for style and regime analysis...", flush=True)
    turnover_state = _get_turnover_wide(style_warmup_start, end)

    print("Running 4-universe × 4-method PIT neutralization matrix...", flush=True)
    method_to_type = {
        "industry": "ind",
        "market_cap": "cap",
        "industry+market_cap": "ind_cap",
    }
    neutral_matrix_rows = []
    neutral_factors_by_universe = {}
    neutral_results_by_universe = {}
    for universe_name in UNIVERSES:
        member = members[universe_name]
        if member is None:
            factor_universe = factor_raw.copy()
        else:
            idx = factor_raw.index.intersection(member.index)
            cols = factor_raw.columns.intersection(member.columns)
            factor_universe = factor_raw.reindex(index=idx, columns=cols).where(
                member.reindex(index=idx, columns=cols) == 1
            )
        print(
            f"  {universe_name}: local daily OLS on {factor_universe.shape[1]} columns",
            flush=True,
        )
        variants = {"raw": factor_universe}
        for method, neutral_type in method_to_type.items():
            variants[method] = panel_neutral_size_ind(
                factor_universe,
                del_limit=False,
                del_st=False,
                nt_type=neutral_type,
            ).astype(float)
        results = {}
        for method, factor in variants.items():
            result = (
                universe_results[universe_name]
                if method == "raw"
                else evaluate_factor(
                    factor,
                    ret_raw,
                    tradable,
                    member,
                    start,
                    end,
                )
            )
            results[method] = result
            neutral_matrix_rows.append(
                {"universe": universe_name, "method": method, **result["summary"]}
            )
            print(
                f"    {method}: IC={result['summary']['rank_ic']:.4f} "
                f"ICIR={result['summary']['icir']:.2f} "
                f"H-L={result['summary']['hl_sharpe']:.2f} "
                f"G10={result['summary']['g10_exact_ew_excess_sharpe']:.2f}",
                flush=True,
            )
        neutral_factors_by_universe[universe_name] = variants
        neutral_results_by_universe[universe_name] = results

    neutral_matrix_df = pd.DataFrame(neutral_matrix_rows)
    neutral_matrix_df.to_csv(
        artifacts / "neutralization_by_universe.csv", index=False
    )
    plot_universe_neutralization_matrix(
        neutral_matrix_df,
        figures / "08b_universe_neutralization_matrix.png",
    )
    neutral_df = (
        neutral_matrix_df.loc[neutral_matrix_df["universe"] == "CSI1000"]
        .drop(columns="universe")
        .reset_index(drop=True)
    )
    neutral_df.to_csv(artifacts / "neutralization_comparison.csv", index=False)
    plot_neutralization_comparison(neutral_df, figures / "08_neutralization_comparison.png")

    print("Running strict PIT second-order neutralization...", flush=True)
    neutral_factors = neutral_factors_by_universe["CSI1000"]
    neutral_results = neutral_results_by_universe["CSI1000"]
    first_stage = neutral_factors["industry+market_cap"]
    style_panels = {
        "momentum": ret_raw_with_warmup.rolling(20, min_periods=10)
        .sum()
        .reindex_like(first_stage),
        "volatility": ret_raw_with_warmup.rolling(20, min_periods=10)
        .std()
        .reindex_like(first_stage),
        "turnover": turnover_state.reindex_like(first_stage),
    }
    second_rows = []
    baseline = neutral_results["industry+market_cap"]["summary"]
    second_rows.append(
        {
            "variant": "Baseline (ind+cap)",
            "n_days": baseline["n_days"],
            "n_names_avg": baseline["n_names_avg"],
            "rank_ic": baseline["rank_ic"],
            "matched_baseline_rank_ic": baseline["rank_ic"],
            "abs_ic_retained_vs_matched": 1.0,
            "icir": baseline["icir"],
            "abs_icir": abs(baseline["icir"]),
            "sharpe": baseline["hl_sharpe"],
            "gross_annu": baseline["hl_annu_ret"],
            "mdd": baseline["hl_mdd"],
        }
    )
    for style_name, style_panel in style_panels.items():
        second_factor = neutralize_again(first_stage, {style_name: style_panel})
        matched_first_stage = first_stage.where(second_factor.notna())
        matched_baseline_result = evaluate_factor(
            matched_first_stage,
            ret_raw,
            tradable,
            members["CSI1000"],
            start,
            end,
        )
        result = evaluate_factor(
            second_factor, ret_raw, tradable, members["CSI1000"], start, end
        )
        matched_baseline = matched_baseline_result["summary"]
        summary = result["summary"]
        if (
            summary["date_start"] != matched_baseline["date_start"]
            or summary["date_end"] != matched_baseline["date_end"]
            or summary["n_days"] != matched_baseline["n_days"]
        ):
            raise RuntimeError(
                f"Matched-sample mismatch for second neutralization: {style_name}"
            )
        retained = abs(summary["rank_ic"]) / abs(matched_baseline["rank_ic"])
        second_rows.append(
            {
                "variant": f"+{style_name}",
                "n_days": summary["n_days"],
                "n_names_avg": summary["n_names_avg"],
                "rank_ic": summary["rank_ic"],
                "matched_baseline_rank_ic": matched_baseline["rank_ic"],
                "abs_ic_retained_vs_matched": retained,
                "icir": summary["icir"],
                "abs_icir": abs(summary["icir"]),
                "sharpe": summary["hl_sharpe"],
                "gross_annu": summary["hl_annu_ret"],
                "mdd": summary["hl_mdd"],
            }
        )
    second_df = pd.DataFrame(second_rows)
    plot_second_neutralization(
        second_df,
        artifacts / "second_neutralization_comparison.csv",
        figures / "09_second_neutralization_comparison.png",
    )

    print("Running CSI1000 PIT threshold grid...", flush=True)
    grid = run_parameter_grid(
        bucket,
        members["CSI1000"],
        ret_raw,
        tradable,
        start,
        end,
    )
    grid.to_csv(artifacts / "parameter_sensitivity_csi1000_pit.csv", index=False)
    plot_heatmap(
        grid,
        "icir",
        "CSI1000 PIT — Parameter Sensitivity: Annualized Raw-Direction ICIR",
        "ICIR",
        figures / "10a_parameter_sensitivity_icir.png",
        "{:.2f}",
        "coolwarm_r",
    )
    plot_heatmap(
        grid,
        "hl_sharpe",
        "CSI1000 PIT — Parameter Sensitivity: Effective H-L Gross Sharpe",
        "H-L Sharpe",
        figures / "10b_parameter_sensitivity_sharpe.png",
        "{:.2f}",
        "viridis",
    )

    print("Computing lagged-turnover regime IC...", flush=True)
    state_daily, state_summary = compute_state_dependence(
        csi["signal_raw"], csi["ret_raw"], turnover_state
    )
    state_daily.to_csv(artifacts / "state_dependence_daily_ic.csv")
    state_summary.to_csv(artifacts / "state_dependence_summary.csv", index=False)
    plot_state_dependence(state_daily, state_summary, figures)

    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "sample_requested": {"start": _fmt_date(start), "end": _fmt_date(end)},
        "factor_observed": {
            "start": _fmt_date(factor_raw.index.min()),
            "end": _fmt_date(factor_raw.index.max()),
            "n_stock_days_cache_after_sse_szse_a_share_filter": int(len(bucket)),
            "n_factor_symbols": int(factor_raw.shape[1]),
        },
        "factor_definition": {
            "name": "mid_order_ratio",
            "lower_rmb_exclusive": 40_000,
            "upper_rmb_inclusive": 200_000,
            "formula": "(cum_200000 - cum_40000) / TotalAmount",
            "effective_direction": FROZEN_EFFECTIVE_DIRECTION,
            "direction_policy": "frozen before evaluation; never inferred from evaluated returns",
            "regular_session": STRICT_SESSION,
            "sse_trade_filter": STRICT_SSE_FILTER,
            "szse_trade_filter": STRICT_SZSE_FILTER,
        },
        "sources": {
            "bucket_cache": str(cache_path),
            "bucket_cache_metadata": cache_metadata["metadata_path"],
            "bucket_cache_sha256": cache_metadata["validated_sha256"],
            "raw_results": str(RESULT_ROOT),
            "returns": "DolphinDB dfs://WIND.ASHAREEODPRICES",
            "index_membership": "DolphinDB Wind daily weight tables",
            "tick_upstream": "ClickHouse cmds.SSE_AL_TICK_EXG / SZSE_AL_TICK_EXG",
        },
        "conventions": {
            "signal_lag_days": 1,
            "rank_ic": "daily cross-sectional Spearman; raw factor direction",
            "icir": "mean/std(ddof=1)*sqrt(250)",
            "all_universe_scope": "SSE/SZSE A-shares matching ^[036]\\d{5}\\.(SH|SZ)$; excludes BSE",
            "universe_comparison_group_benchmark": "exact valid-universe equal-weight",
            "main_decile_figure_benchmark": "CSI1000 index c2c return",
            "neutralization_scope": "daily OLS re-estimated within each PIT universe",
            "second_neutralization_comparison": "pre-sample style warmup and matched-cell baseline",
            "style_warmup_calendar_days": STYLE_WARMUP_CALENDAR_DAYS,
            "h_l": "G10-G1 after frozen effective_direction=-1; gross fee=0",
            "annualization": 250,
        },
        "artifact_files": sorted(str(p.relative_to(output_root)) for p in artifacts.iterdir()),
        "figure_files": sorted(str(p.relative_to(output_root)) for p in figures.iterdir()),
    }
    (artifacts / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done: {output_root}", flush=True)


if __name__ == "__main__":
    main()

