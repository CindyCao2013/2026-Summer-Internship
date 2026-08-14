#!/usr/bin/env python3
"""Plot qualified frozen factors in the standard decile/H-L research format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.intraday_metrics import (  # noqa: E402
    build_group_excess_panel,
    build_hl_panel,
)
from research.freeze_intraday_alpha_v1 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_FREEZE,
    verify_spec,
)
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    _connect,
    _evaluate,
)
from research.run_intraday_alpha_oos_v1 import (  # noqa: E402
    _build_factor_chunked,
    _filter_slots,
)
from research.run_intraday_evaluation_v2 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_EVALUATION,
    _aggregate_exact_group_and_market,
)

DEFAULT_OOS = (
    ROOT
    / "research/results/intraday_alpha_oos_v1/intraday_alpha_oos_v1.csv"
)
DEFAULT_FIGURES = DEFAULT_EVALUATION / "figures"
MIN_HL_SHARPE = 3.0
MIN_MONOTONIC_SPEARMAN = 0.8
MIN_ADJACENT_STEP_RATIO = 2.0 / 3.0


def _group_number(group: pd.Series) -> pd.Series:
    return pd.to_numeric(
        group.astype(str).str.extract(r"^G(\d+)$")[0],
        errors="coerce",
    )


def select_qualified_factors(
    performance: pd.DataFrame,
    candidates: pd.DataFrame,
    oos: pd.DataFrame,
    *,
    min_hl_sharpe: float = MIN_HL_SHARPE,
    min_monotonic_spearman: float = MIN_MONOTONIC_SPEARMAN,
    min_adjacent_step_ratio: float = MIN_ADJACENT_STEP_RATIO,
) -> pd.DataFrame:
    """Apply OOS survival, H-L strength, and robust decile monotonicity gates."""
    conclusion = oos[["factor", "conclusion"]].drop_duplicates("factor")
    base = candidates.merge(conclusion, on="factor", how="left")
    rows = []
    for candidate in base.itertuples():
        groups = performance[
            (performance["factor"] == candidate.factor)
            & (performance["period"] == "train_2024H1")
            & (
                performance["bartime"].astype(str)
                == str(candidate.bartime)
            )
            & (performance["return_window"] == candidate.horizon)
            & (
                performance["metric_scope"]
                == "cross_sectional_group"
            )
        ].copy()
        groups["group_number"] = _group_number(groups["group"])
        groups = groups.dropna(subset=["group_number"]).sort_values(
            "group_number"
        )
        if len(groups) != 10:
            continue
        oriented = (
            int(candidate.direction)
            * groups["group_return_excess"].to_numpy(dtype=float)
        )
        group_numbers = groups["group_number"].to_numpy(dtype=float)
        monotonic_spearman = float(
            pd.Series(group_numbers).corr(
                pd.Series(oriented),
                method="spearman",
            )
        )
        adjacent_step_ratio = float(np.mean(np.diff(oriented) > 0))
        qualified = bool(
            candidate.conclusion == "retain"
            and float(candidate.hl_sharpe) > min_hl_sharpe
            and monotonic_spearman >= min_monotonic_spearman
            and adjacent_step_ratio >= min_adjacent_step_ratio
        )
        rows.append(
            {
                "factor": candidate.factor,
                "bartime": str(candidate.bartime),
                "horizon": candidate.horizon,
                "direction": int(candidate.direction),
                "oos_conclusion": candidate.conclusion,
                "hl_sharpe": float(candidate.hl_sharpe),
                "monotonic_spearman": monotonic_spearman,
                "adjacent_step_ratio": adjacent_step_ratio,
                "qualified": qualified,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["qualified", "hl_sharpe"],
        ascending=[False, False],
    )


def _daily_paths(output: Path, factor_name: str) -> tuple[Path, Path]:
    data_dir = output / "plot_data"
    return (
        data_dir / f"{factor_name}_group_daily.csv",
        data_dir / f"{factor_name}_hl_daily.csv",
    )


def _fetch_daily_panels(
    session,
    freeze: dict,
    row,
    output: Path,
    *,
    chunk_months: int,
) -> None:
    group_path, hl_path = _daily_paths(output, row.factor)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    signal = _build_factor_chunked(
        row.factor,
        freeze["train_period"]["start"],
        freeze["train_period"]["end"],
        chunk_months,
    )
    signal = _filter_slots(signal, {str(row.bartime)})
    filtered, _, _, _ = _evaluate(
        session,
        f"plotv2_{row.factor}_train_2024H1",
        signal,
        apply_limit_filter=True,
    )
    groups, market = _aggregate_exact_group_and_market(
        session,
        f"plotv2_{row.factor}_train_2024H1",
        filtered,
        [str(row.horizon)],
    )
    panel = build_group_excess_panel(groups, market)
    panel = panel[
        (panel["Bartime"].astype(str) == str(row.bartime))
        & (panel["return_window"] == str(row.horizon))
    ].copy()
    hl = build_hl_panel(panel, direction=int(row.direction))
    panel.to_csv(group_path, index=False)
    hl.to_csv(hl_path, index=False)


def _load_daily_panels(
    output: Path,
    factor_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_path, hl_path = _daily_paths(output, factor_name)
    groups = pd.read_csv(group_path, parse_dates=["Date"])
    hl = pd.read_csv(hl_path, parse_dates=["Date"])
    return groups, hl


def _axes_grid(count: int, columns: int = 2):
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(18, 6.8 * rows),
        squeeze=False,
    )
    return fig, axes.ravel()


def _plot_cumulative(
    qualified: pd.DataFrame,
    output: Path,
) -> Path:
    fig, axes = _axes_grid(len(qualified))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 10))
    for ax, row in zip(axes, qualified.itertuples()):
        groups, hl = _load_daily_panels(output, row.factor)
        pivot = groups.pivot(
            index="Date",
            columns="group",
            values="group_return_excess",
        )
        ordered = [f"G{i}" for i in range(1, 11)]
        pivot = pivot.reindex(columns=ordered).sort_index()
        for index, group_name in enumerate(ordered):
            series = pivot[group_name].fillna(0).cumsum()
            ax.plot(
                series.index,
                series.values,
                color=colors[index],
                linewidth=1.1,
                alpha=0.85,
                label=group_name,
            )
        hl_series = (
            hl.sort_values("Date")
            .set_index("Date")["hl_return"]
            .fillna(0)
            .cumsum()
        )
        ax.plot(
            hl_series.index,
            hl_series.values,
            color="black",
            linewidth=2.8,
            label="Directed H-L",
        )
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(
            f"{row.factor} · {row.bartime}/{row.horizon} · "
            f"H-L S={row.hl_sharpe:.2f}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative market-excess return")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=4, fontsize=8, loc="upper left")
    for ax in axes[len(qualified) :]:
        ax.remove()
    fig.suptitle(
        "Qualified intraday factors: decile and directed H-L cumulative return",
        fontsize=16,
        y=1.002,
    )
    fig.text(
        0.5,
        0.006,
        "Exact constituent-EW market benchmark · 2024H1 frozen tuple · "
        "OOS retain · H-L Sharpe > 3 · monotonicity gate",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    path = output / "qualified_decile_hl_cumulative.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_decile_bars(
    qualified: pd.DataFrame,
    output: Path,
) -> Path:
    fig, axes = _axes_grid(len(qualified))
    for ax, row in zip(axes, qualified.itertuples()):
        groups, hl = _load_daily_panels(output, row.factor)
        means = (
            groups.assign(group_number=_group_number(groups["group"]))
            .groupby("group_number")["group_return_excess"]
            .mean()
            .reindex(range(1, 11))
            * 1e4
        )
        colors = ["#9ecae1"] * 10
        colors[0] = "#3182bd"
        colors[-1] = "#de2d26"
        ax.bar(
            [f"G{i}" for i in range(1, 11)],
            means.values,
            color=colors,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        hl_bps = float(hl["hl_return"].mean() * 1e4)
        ax.set_title(
            f"{row.factor} · monotonic ρ={row.monotonic_spearman:.2f} · "
            f"directed H-L={hl_bps:.2f} bps/day"
        )
        ax.set_xlabel("Raw-factor decile (G1 low → G10 high)")
        ax.set_ylabel("Mean market-excess return (bps/day)")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(qualified) :]:
        ax.remove()
    fig.suptitle(
        "Qualified intraday factors: mean market-excess return by decile",
        fontsize=16,
        y=1.002,
    )
    fig.text(
        0.5,
        0.006,
        "Bars preserve raw-factor decile order; frozen direction is applied "
        "only to the reported H-L spread.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    path = output / "qualified_decile_mean_bar.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--oos", type=Path, default=DEFAULT_OOS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    freeze = verify_spec(args.freeze)
    performance = pd.read_csv(args.evaluation / "performance_all_v2.csv")
    candidates = pd.read_csv(
        args.evaluation / "intraday_alpha_library_v3_candidates.csv"
    )
    oos = pd.read_csv(args.oos)
    selection = select_qualified_factors(performance, candidates, oos)
    qualified = selection[selection["qualified"]].copy()
    if qualified.empty:
        raise RuntimeError("No factors passed the configured plotting gates")

    args.output.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output / "plot_factor_selection.csv", index=False)
    session = None
    for row in qualified.itertuples():
        group_path, hl_path = _daily_paths(args.output, row.factor)
        if (
            args.refresh_data
            or not group_path.exists()
            or not hl_path.exists()
        ):
            if session is None:
                session = _connect()
            print(f"[PLOT DATA] {row.factor}", flush=True)
            _fetch_daily_panels(
                session,
                freeze,
                row,
                args.output,
                chunk_months=args.chunk_months,
            )

    cumulative_path = _plot_cumulative(qualified, args.output)
    bar_path = _plot_decile_bars(qualified, args.output)
    manifest = {
        "period": freeze["train_period"],
        "oos_gate": "conclusion == retain",
        "min_hl_sharpe": MIN_HL_SHARPE,
        "monotonicity": {
            "definition": (
                "Spearman(decile, direction*mean_excess_return) and "
                "positive adjacent-step ratio"
            ),
            "min_spearman": MIN_MONOTONIC_SPEARMAN,
            "min_adjacent_step_ratio": MIN_ADJACENT_STEP_RATIO,
        },
        "qualified_factors": qualified["factor"].tolist(),
        "figures": [cumulative_path.name, bar_path.name],
    }
    (args.output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        "Qualified factors: "
        + ", ".join(qualified["factor"].tolist()),
        flush=True,
    )
    print(f"Cumulative plot → {cumulative_path}", flush=True)
    print(f"Decile bar plot → {bar_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
