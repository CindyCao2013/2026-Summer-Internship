#!/usr/bin/env python
"""Sprint 3: expand the daily Trade Flow Family candidate pool.

Scope:
- build frozen factor-narrow files from l2_primitive_trade_flow_daily;
- run the existing daily T+1 decile backtest without optimization;
- export IC/ICIR, turnover/cost, yearly stability, and redundancy diagnostics.

Explicitly out of scope: neutralization search, weekly optimization, pruning,
factor blending, and portfolio construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    _save_backtest_outputs,
    backtest_factor,
)
from l2_factor_reproduction.python.trade_flow_factors import (  # noqa: E402
    TRADE_FLOW_FACTOR_NAMES,
    TRADE_FLOW_FACTOR_SPECS,
    build_trade_flow_feature_frame,
    feature_to_narrow,
    registry_frame,
)


DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
DEFAULT_PRIMITIVE = (
    Path(RESULT_ROOT)
    / "primitives"
    / "trade_flow_daily"
    / "trade_flow_daily_2019-01-01_2026-07-31.parquet"
)
POOL_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "trade_flow_family"


def _parse_factor_names(value: str) -> List[str]:
    names = (
        list(TRADE_FLOW_FACTOR_NAMES)
        if value.strip().lower() in {"", "all"}
        else [x.strip() for x in value.split(",") if x.strip()]
    )
    unknown = sorted(set(names).difference(TRADE_FLOW_FACTOR_SPECS))
    if unknown:
        raise ValueError(
            f"unknown factors={unknown}; valid={list(TRADE_FLOW_FACTOR_NAMES)}"
        )
    if not names:
        raise ValueError("factor list is empty")
    return names


def _yearly_ic_table(rank_ic: pd.Series) -> pd.DataFrame:
    ic = rank_ic.dropna()
    table = ic.groupby(ic.index.year).agg(["mean", "std", "count"])
    table["icir_annualized"] = (
        table["mean"] / table["std"] * np.sqrt(250)
    )
    table.index.name = "year"
    return table


def _load_rank_ic(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return pd.Series(dtype=float, name="rank_ic")
    out = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    out.index = pd.to_datetime(out.index)
    out.name = "rank_ic"
    return out


def _stability_fields(
    yearly_raw: pd.DataFrame,
    full_raw_ic: float,
) -> Dict[str, object]:
    valid = yearly_raw.loc[yearly_raw["count"] >= 30].copy()
    if valid.empty:
        return {
            "n_years": 0,
            "same_sign_years": 0,
            "sign_consistency": np.nan,
            "positive_ic_years": 0,
            "negative_ic_years": 0,
            "yearly_ic_min": np.nan,
            "yearly_ic_max": np.nan,
        }
    full_sign = np.sign(full_raw_ic)
    yearly_sign = np.sign(valid["mean"])
    same = (
        int((yearly_sign == full_sign).sum())
        if full_sign != 0
        else int((yearly_sign == 0).sum())
    )
    return {
        "n_years": int(len(valid)),
        "same_sign_years": same,
        "sign_consistency": float(same / len(valid)),
        "positive_ic_years": int((valid["mean"] > 0).sum()),
        "negative_ic_years": int((valid["mean"] < 0).sum()),
        "yearly_ic_min": float(valid["mean"].min()),
        "yearly_ic_max": float(valid["mean"].max()),
    }


def _decile_monotonicity(summary: Dict[str, object]) -> float:
    values = summary.get("group_mean_annu", {})
    if not isinstance(values, dict) or len(values) < 3:
        return float("nan")
    pairs = sorted(
        ((int(group), float(value)) for group, value in values.items()),
        key=lambda x: x[0],
    )
    groups = pd.Series([x[0] for x in pairs], dtype=float)
    returns = pd.Series([x[1] for x in pairs], dtype=float)
    return float(groups.corr(returns, method="spearman"))


def _mean_daily_cross_sectional_spearman(
    features: pd.DataFrame,
    names: Iterable[str],
    min_names: int = 100,
) -> pd.DataFrame:
    """Full-sample mean of daily cross-sectional Spearman matrices."""
    names = list(names)
    total = pd.DataFrame(0.0, index=names, columns=names)
    count = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for _, block in features.groupby("TradeDate", sort=True):
        corr = block[names].corr(method="spearman", min_periods=min_names)
        valid = corr.notna()
        total = total.add(corr.fillna(0.0), fill_value=0.0)
        count = count.add(valid.astype(int), fill_value=0)
    return total.divide(count.where(count > 0))


def _correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    names = list(corr.index)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            rho = float(corr.loc[left, right])
            abs_rho = abs(rho)
            if abs_rho >= 0.95:
                band = "near_alias"
            elif abs_rho >= 0.80:
                band = "high"
            elif abs_rho >= 0.50:
                band = "moderate"
            else:
                band = "low"
            rows.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "mean_daily_spearman": rho,
                    "abs_mean_daily_spearman": abs_rho,
                    "redundancy_band": band,
                }
            )
    columns = [
        "factor_left",
        "factor_right",
        "mean_daily_spearman",
        "abs_mean_daily_spearman",
        "redundancy_band",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "abs_mean_daily_spearman", ascending=False
    )


def _redundancy_annotations(
    corr: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Connected components under an absolute-correlation threshold.

    This is descriptive metadata only; it does not prune the candidate pool.
    """
    names = list(corr.index)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if abs(float(corr.loc[left, right])) >= threshold:
                union(left, right)

    root_to_cluster: Dict[str, str] = {}
    rows = []
    for name in names:
        root = find(name)
        if root not in root_to_cluster:
            root_to_cluster[root] = f"R{len(root_to_cluster) + 1}"
        peers = corr.loc[name].drop(index=name).dropna()
        if peers.empty:
            max_peer, max_corr = None, np.nan
        else:
            max_peer = str(peers.abs().idxmax())
            max_corr = float(peers.loc[max_peer])
        rows.append(
            {
                "factor": name,
                "redundancy_cluster_080": root_to_cluster[root],
                "max_corr_peer": max_peer,
                "max_abs_corr": abs(max_corr),
                "near_alias_observed": bool(abs(max_corr) >= 0.95),
            }
        )
    return pd.DataFrame(rows)


def _write_registry(names: List[str]) -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    registry = registry_frame(names)
    registry.to_csv(POOL_DIR / "factor_registry.csv", index=False)
    with open(
        POOL_DIR / "factor_registry.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(
            registry.to_dict("records"),
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _build_narrows(
    primitive_path: Path,
    names: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    compute_correlation: bool,
) -> Dict[str, int]:
    print(f"[build] primitive={primitive_path}", flush=True)
    flow = pd.read_parquet(primitive_path)
    flow["TradeDate"] = pd.to_datetime(flow["TradeDate"])
    flow = flow.loc[flow["TradeDate"].between(start, end)]
    print(
        f"[build] raw rows={len(flow):,}; "
        f"dates={flow['TradeDate'].min().date()}~{flow['TradeDate'].max().date()}",
        flush=True,
    )
    features = build_trade_flow_feature_frame(flow)
    del flow

    coverage: Dict[str, int] = {}
    for position, name in enumerate(names, 1):
        narrow = feature_to_narrow(features, name)
        out_dir = Path(RESULT_ROOT) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        narrow.to_parquet(out_dir / "factor_narrow.parquet", index=False)
        coverage[name] = int(len(narrow))
        print(
            f"[build] {position}/{len(names)} {name}: "
            f"{len(narrow):,} rows",
            flush=True,
        )

    pd.DataFrame(
        [
            {"factor": name, "n_factor_rows": rows}
            for name, rows in coverage.items()
        ]
    ).to_csv(POOL_DIR / "factor_coverage.csv", index=False)

    if compute_correlation:
        print(
            "[redundancy] full-sample mean daily cross-sectional Spearman ...",
            flush=True,
        )
        corr = _mean_daily_cross_sectional_spearman(features, names)
        corr.to_csv(POOL_DIR / "factor_correlation_spearman.csv")
        pairs = _correlation_pairs(corr)
        pairs.to_csv(POOL_DIR / "factor_correlation_pairs.csv", index=False)
        print(
            "[redundancy] strongest pairs:\n"
            + pairs.head(10).to_string(index=False),
            flush=True,
        )
    return coverage


def _run_or_reuse_backtest(
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    force: bool,
) -> Tuple[Dict[str, object], pd.Series]:
    out_dir = Path(RESULT_ROOT) / name
    summary_path = out_dir / "summary.json"
    rank_ic_path = out_dir / "rank_ic.csv"
    if not force and summary_path.exists() and rank_ic_path.exists():
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        rank_ic_effective = _load_rank_ic(rank_ic_path)
        print(f"[backtest] reuse {name}", flush=True)
        return summary, rank_ic_effective

    narrow_path = out_dir / "factor_narrow.parquet"
    narrow = pd.read_parquet(narrow_path)
    group_pnl, group_to, rank_ic_effective, summary = backtest_factor(
        narrow,
        start_day=start,
        end_day=end,
    )
    summary["net_annu_after_fee"] = (
        float(summary["hl_annu_ret_flipped"])
        - float(summary["implied_annu_fee"])
    )
    _save_backtest_outputs(
        str(out_dir),
        group_pnl,
        group_to,
        rank_ic_effective,
        summary,
        factor_name=name,
    )
    return summary, rank_ic_effective


def _write_summary_report(summary: pd.DataFrame) -> None:
    display_columns = [
        "factor",
        "rank_ic_raw",
        "icir_raw",
        "hl_annu_ret",
        "hl_sharpe",
        "avg_hl_turnover",
        "implied_annu_fee",
        "net_annu_after_fee",
        "sign_consistency",
        "decile_mono_spearman",
        "redundancy_cluster_080",
        "max_abs_corr",
    ]
    try:
        metrics_text = summary[display_columns].to_markdown(index=False)
    except ImportError:
        metrics_text = (
            "```\n"
            + summary[display_columns].to_string(index=False)
            + "\n```"
        )
    cluster_lines = []
    for cluster, members in summary.groupby(
        "redundancy_cluster_080", sort=True
    ):
        cluster_lines.append(
            f"- `{cluster}`: "
            + ", ".join(f"`{name}`" for name in members["factor"])
        )

    report = [
        "# L2 Candidate Pool v1 — Trade Flow Family",
        "",
        "范围：纯候选扩展；未做参数优化、中性化搜索、周度优化、筛选或组合。",
        "",
        "公式见 `factor_registry.csv`；冗余关系见 "
        "`factor_correlation_spearman.csv`。",
        "",
        "这里不做 KEEP/DROP 决策；相关簇仅用于标记候选的经验重复度。",
        "",
        f"- 冻结候选公式：{len(summary)}",
        f"- 经验相关簇（|日截面 Spearman| ≥ 0.80）："
        f"{summary['redundancy_cluster_080'].nunique()}",
        "",
        "## Baseline screening metrics",
        "",
        metrics_text,
        "",
        "注：`rank_ic_raw/icir_raw` 保留冻结公式原方向；H-L 指标使用统一"
        "有效方向。`sign_consistency` 为年度原始 IC 与全样本原始 IC 同号比例。",
        "",
        "## Empirical redundancy map",
        "",
        *cluster_lines,
        "",
        "关键重复关系：`buy_dominance` 与 `net_buy_ratio` 近乎完全相同；"
        "`avg_buy_trade_size` 与 `avg_sell_trade_size` 高度相关；"
        "20 日 z-score 仍高度保留金额方向截面排序。它们是合法基础 feature，"
        "但不能按独立 alpha 数量重复计数。修正 UInt64 计数下溢后，"
        "`net_buy_count_ratio` 与 `flow_concentration` 不再属于同一高相关簇。",
        "",
        "## Research reading constraints",
        "",
        "- `avg_buy_trade_size` / `avg_sell_trade_size` 的 baseline IC 最强且"
        "年度同号，但两者高度相关，并天然承载股票规模、价格和流动性暴露；"
        "在行业/市值暴露审计前不能解释为独立 L2 alpha。",
        "- `trade_size_asymmetry` 相关性更低，但 H-L 单调性和成本表现明显"
        "弱于两个平均规模腿；当前只保留为候选。",
        "- `flow_zscore_20d` 与 `flow_acceleration` 没有在 baseline 中产生"
        "更强或更稳定的新方向，仍保留用于后续跨家族 taxonomy。",
        "- 本报告不作晋级结论。下一批应继续扩展 Order Size Family，"
        "而不是进入组合层。",
        "",
    ]
    (POOL_DIR / "report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive", type=Path, default=DEFAULT_PRIMITIVE)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument(
        "--factors",
        default="all",
        help="comma-separated names, or 'all'",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="write factor narrow files and stop",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="reuse existing factor narrow files",
    )
    parser.add_argument(
        "--force-backtest",
        action="store_true",
        help="rerun even when summary.json exists",
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="skip candidate redundancy matrix",
    )
    args = parser.parse_args(argv)
    if args.build_only and args.backtest_only:
        parser.error("--build-only and --backtest-only are mutually exclusive")

    names = _parse_factor_names(args.factors)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    _write_registry(names)

    coverage: Dict[str, int] = {}
    if not args.backtest_only:
        coverage = _build_narrows(
            args.primitive,
            names,
            start,
            end,
            compute_correlation=not args.skip_correlation,
        )
    elif (POOL_DIR / "factor_coverage.csv").exists():
        previous = pd.read_csv(POOL_DIR / "factor_coverage.csv")
        if {"factor", "n_factor_rows"}.issubset(previous.columns):
            coverage = {
                str(row["factor"]): int(row["n_factor_rows"])
                for _, row in previous.dropna(
                    subset=["n_factor_rows"]
                ).iterrows()
            }
    if args.build_only:
        return 0

    rows = []
    yearly_raw_parts = []
    yearly_effective_parts = []
    for position, name in enumerate(names, 1):
        print(
            f"[backtest] {position}/{len(names)} {name}",
            flush=True,
        )
        summary, rank_ic_effective = _run_or_reuse_backtest(
            name,
            start,
            end,
            force=args.force_backtest,
        )
        direction = int(summary.get("factor_direction", 1))
        rank_ic_raw = rank_ic_effective * direction
        yearly_effective = _yearly_ic_table(rank_ic_effective)
        yearly_raw = _yearly_ic_table(rank_ic_raw)

        out_dir = Path(RESULT_ROOT) / name
        yearly_effective.to_csv(out_dir / "yearly_ic.csv")
        yearly_raw.to_csv(out_dir / "yearly_ic_raw.csv")
        yearly_raw.assign(factor=name).reset_index().to_csv(
            out_dir / "yearly_stability.csv",
            index=False,
        )
        yearly_effective_parts.append(
            yearly_effective.assign(factor=name).reset_index()
        )
        yearly_raw_parts.append(
            yearly_raw.assign(factor=name).reset_index()
        )

        raw_mean = float(
            summary.get("rank_ic_mean_raw", rank_ic_raw.mean())
        )
        raw_std = float(rank_ic_raw.std())
        raw_icir = (
            raw_mean / raw_std * np.sqrt(250)
            if raw_std > 0
            else float("nan")
        )
        net_annu = (
            float(summary["hl_annu_ret_flipped"])
            - float(summary["implied_annu_fee"])
        )
        row = {
            "factor": name,
            "mechanism": TRADE_FLOW_FACTOR_SPECS[name].mechanism,
            "lookback_days": TRADE_FLOW_FACTOR_SPECS[name].lookback_days,
            "factor_direction": direction,
            "rank_ic_raw": raw_mean,
            "icir_raw": raw_icir,
            "rank_ic_effective": float(summary["rank_ic_mean"]),
            "icir_effective": float(summary["rank_icir"]),
            "hl_annu_ret": float(summary["hl_annu_ret_flipped"]),
            "hl_sharpe": float(summary["hl_sharpe_flipped"]),
            "hl_mdd": float(summary["hl_mdd_flipped"]),
            "avg_hl_turnover": float(summary["avg_hl_turnover"]),
            "implied_annu_fee": float(summary["implied_annu_fee"]),
            "net_annu_after_fee": net_annu,
            "decile_mono_spearman": _decile_monotonicity(summary),
            "n_days": int(summary["n_days"]),
            "n_names_avg": float(summary["n_names_avg"]),
            "n_factor_rows": coverage.get(name),
            **_stability_fields(yearly_raw, raw_mean),
        }
        rows.append(row)
        print(
            f"[result] {name}: raw IC={raw_mean:+.4f}, "
            f"raw ICIR={raw_icir:+.2f}, "
            f"H-L Sharpe={row['hl_sharpe']:.2f}, "
            f"turnover={row['avg_hl_turnover']:.2f}, "
            f"year-sign={row['same_sign_years']}/{row['n_years']}",
            flush=True,
        )

    summary_frame = pd.DataFrame(rows)
    corr_path = POOL_DIR / "factor_correlation_spearman.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
        annotations = _redundancy_annotations(
            corr.loc[names, names],
            threshold=0.80,
        )
        annotations.to_csv(
            POOL_DIR / "redundancy_annotations.csv",
            index=False,
        )
        summary_frame = summary_frame.merge(
            annotations,
            on="factor",
            how="left",
            validate="one_to_one",
        )
    else:
        summary_frame["redundancy_cluster_080"] = None
        summary_frame["max_corr_peer"] = None
        summary_frame["max_abs_corr"] = np.nan
        summary_frame["near_alias_observed"] = False
    summary_frame.to_csv(POOL_DIR / "candidate_summary.csv", index=False)
    pd.concat(yearly_raw_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_raw.csv",
        index=False,
    )
    pd.concat(yearly_effective_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_effective.csv",
        index=False,
    )
    _write_summary_report(summary_frame)

    manifest = {
        "version": "trade_flow_candidate_pool_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "primitive": str(args.primitive),
        "factors": names,
        "n_candidates": len(names),
        "scope_exclusions": [
            "parameter_optimization",
            "neutralization_search",
            "weekly_optimization",
            "candidate_pruning",
            "factor_combination",
        ],
    }
    with open(POOL_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(
        f"[done] candidate pool -> {POOL_DIR}\n"
        + summary_frame[
            [
                "factor",
                "rank_ic_raw",
                "icir_raw",
                "hl_sharpe",
                "avg_hl_turnover",
                "sign_consistency",
            ]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
