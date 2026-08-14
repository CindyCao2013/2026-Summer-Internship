#!/usr/bin/env python
"""Sprint 4: build and screen Order Size Family v1 candidates.

Scope is formula-layer discovery only: frozen thresholds, T+1 daily baseline,
yearly stability, cost/turnover, and empirical redundancy. No optimization,
neutralization search, pruning, combination, or portfolio construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
from l2_factor_reproduction.python.candidate_pool import (  # noqa: E402
    correlation_pairs,
    decile_monotonicity,
    load_rank_ic,
    mean_daily_cross_sectional_spearman,
    redundancy_annotations,
    stability_fields,
    yearly_ic_table,
)
from l2_factor_reproduction.python.order_size_factors import (  # noqa: E402
    ORDER_SIZE_FACTOR_NAMES,
    ORDER_SIZE_FACTOR_SPECS,
    build_order_size_feature_frame,
    feature_to_narrow,
    registry_frame,
)


DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
DEFAULT_PRIMITIVE = (
    Path(RESULT_ROOT)
    / "primitives"
    / "order_size_distribution_daily"
    / "order_size_distribution_daily_2019-01-01_2026-07-31.parquet"
)
POOL_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "order_size_family"


def _parse_names(value: str) -> List[str]:
    names = (
        list(ORDER_SIZE_FACTOR_NAMES)
        if value.strip().lower() in {"", "all"}
        else [item.strip() for item in value.split(",") if item.strip()]
    )
    unknown = sorted(set(names).difference(ORDER_SIZE_FACTOR_SPECS))
    if unknown:
        raise ValueError(
            f"unknown factors={unknown}; "
            f"valid={list(ORDER_SIZE_FACTOR_NAMES)}"
        )
    if not names:
        raise ValueError("factor list is empty")
    return names


def _write_registry(names: List[str]) -> None:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    registry = registry_frame(names)
    registry.to_csv(POOL_DIR / "factor_registry.csv", index=False)
    (POOL_DIR / "factor_registry.json").write_text(
        json.dumps(
            registry.to_dict("records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
    primitive = pd.read_parquet(primitive_path)
    primitive["TradeDate"] = pd.to_datetime(primitive["TradeDate"])
    primitive = primitive.loc[
        primitive["TradeDate"].between(start, end)
    ]
    print(
        f"[build] rows={len(primitive):,}; "
        f"dates={primitive['TradeDate'].min().date()}~"
        f"{primitive['TradeDate'].max().date()}",
        flush=True,
    )
    features = build_order_size_feature_frame(primitive)
    del primitive

    coverage: Dict[str, int] = {}
    for position, name in enumerate(names, 1):
        narrow = feature_to_narrow(features, name)
        out_dir = Path(RESULT_ROOT) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        narrow.to_parquet(
            out_dir / "factor_narrow.parquet",
            index=False,
        )
        coverage[name] = int(len(narrow))
        print(
            f"[build] {position}/{len(names)} {name}: "
            f"{len(narrow):,} rows",
            flush=True,
        )
        del narrow

    pd.DataFrame(
        [
            {"factor": name, "n_factor_rows": rows}
            for name, rows in coverage.items()
        ]
    ).to_csv(POOL_DIR / "factor_coverage.csv", index=False)

    if compute_correlation:
        print(
            "[redundancy] mean daily cross-sectional Spearman ...",
            flush=True,
        )
        corr = mean_daily_cross_sectional_spearman(
            features, names
        )
        corr.to_csv(POOL_DIR / "factor_correlation_spearman.csv")
        pairs = correlation_pairs(corr)
        pairs.to_csv(
            POOL_DIR / "factor_correlation_pairs.csv",
            index=False,
        )
        print(
            "[redundancy] strongest pairs:\n"
            + pairs.head(15).to_string(index=False),
            flush=True,
        )
    return coverage


def _run_or_reuse(
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
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[backtest] reuse {name}", flush=True)
        return summary, load_rank_ic(rank_ic_path)

    narrow = pd.read_parquet(out_dir / "factor_narrow.parquet")
    group_pnl, group_turnover, rank_ic, summary = backtest_factor(
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
        group_turnover,
        rank_ic,
        summary,
        factor_name=name,
    )
    return summary, rank_ic


def _write_report(summary: pd.DataFrame) -> None:
    columns = [
        "factor",
        "rank_ic_raw",
        "icir_raw",
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
        table = summary[columns].to_markdown(index=False)
    except ImportError:
        table = "```\n" + summary[columns].to_string(index=False) + "\n```"
    cluster_lines = [
        f"- `{cluster}`: "
        + ", ".join(f"`{name}`" for name in block["factor"])
        for cluster, block in summary.groupby(
            "redundancy_cluster_080", sort=True
        )
    ]
    indexed = summary.set_index("factor")

    def value(name: str, column: str) -> float:
        return float(indexed.loc[name, column])

    positive_net = summary.loc[
        summary["net_annu_after_fee"] > 0, "factor"
    ].tolist()
    r1_count = int(
        (summary["redundancy_cluster_080"] == "R1").sum()
    )
    lines = [
        "# L2 Candidate Pool v1 — Order Size Family",
        "",
        "范围：冻结公式的 baseline discovery；未做参数优化、中性化搜索、"
        "周度优化、筛选或组合。",
        "",
        f"- 候选公式：{len(summary)}",
        f"- 经验相关簇（|日截面 Spearman| ≥ 0.80）："
        f"{summary['redundancy_cluster_080'].nunique()}",
        "- 固定金额边界：1万 / 4万 / 5万 / 20万 / 100万",
        "",
        "## Baseline metrics",
        "",
        table,
        "",
        "注：IC/ICIR 保留冻结公式原方向；H-L 使用统一有效方向。"
        "这里不作 KEEP/DROP 决策。",
        "",
        "## Evidence-based reading",
        "",
        "1. **订单规模 level 信号跨年稳定，但高度冗余。** "
        f"`small_order_ratio_1w` 原始 IC={value('small_order_ratio_1w', 'rank_ic_raw'):+.2%}、"
        f"ICIR={value('small_order_ratio_1w', 'icir_raw'):+.2f}；"
        f"`mid_order_ratio_4w_20w` 原始 IC={value('mid_order_ratio_4w_20w', 'rank_ic_raw'):+.2%}、"
        f"ICIR={value('mid_order_ratio_4w_20w', 'icir_raw'):+.2f}，两者均 8/8 年同号。"
        f"但 R1 通过相关边连接了 {r1_count} 个公式，不能重复计为独立 alpha。",
        "",
        "2. **`super_large_order_ratio_100w` 是最清晰的新 level 方向。** "
        f"它位于独立 R2，原始 IC={value('super_large_order_ratio_100w', 'rank_ic_raw'):+.2%}、"
        f"ICIR={value('super_large_order_ratio_100w', 'icir_raw'):+.2f}、8/8 年同号；"
        f"但 decile 单调性仅 {value('super_large_order_ratio_100w', 'decile_mono_spearman'):.2f}，"
        f"日频费后年化 {value('super_large_order_ratio_100w', 'net_annu_after_fee'):+.1%}，"
        "尚不能晋级。",
        "",
        "3. **20 日 shock 提供动态维度，但实现成本过高。** "
        f"`large_order_shock_20d` 原始 IC={value('large_order_shock_20d', 'rank_ic_raw'):+.2%}、"
        f"ICIR={value('large_order_shock_20d', 'icir_raw'):+.2f}、7/8 年同号；"
        f"日均 H-L 换手 {value('large_order_shock_20d', 'avg_hl_turnover'):.2f}，"
        f"基准费后年化 {value('large_order_shock_20d', 'net_annu_after_fee'):+.1%}。",
        "",
        "4. **不能把 `super_large_order_pressure` 的高 H-L Sharpe 当成稳定线性 alpha。** "
        f"其 H-L Sharpe={value('super_large_order_pressure', 'hl_sharpe'):.2f}，"
        f"但原始 IC={value('super_large_order_pressure', 'rank_ic_raw'):+.3%}、"
        f"年度同号率={value('super_large_order_pressure', 'sign_consistency'):.0%}、"
        f"decile 单调性={value('super_large_order_pressure', 'decile_mono_spearman'):.2f}、"
        f"费后年化={value('super_large_order_pressure', 'net_annu_after_fee'):+.1%}。"
        "这是端点/非线性分组效应，不是已验证 pressure alpha。",
        "",
        "5. **日频成本诊断严格。** 当前仅 "
        + ", ".join(f"`{name}`" for name in positive_net)
        + " 的基准费后年化为正；本 Sprint 按约束不据此做调仓优化。",
        "",
        "## Empirical redundancy map",
        "",
        *cluster_lines,
        "",
        "关键近别名：`entropy` vs `concentration` = -0.9774；"
        "`small_order_ratio_1w` vs `tail_share` = +0.9753；"
        "4w/20w vs 5w/20w 中单 = +0.9518。保留基础 feature 不等于"
        "重复计为独立 alpha。",
        "",
        "## Primitive audit",
        "",
        "长样本 primitive 共 8,449,391 个唯一 A 股股票日、5,438 只股票，"
        "观测期 2019-01-02 至 2026-07-31。五档成交金额占比为 "
        "17.88% / 37.76% / 27.65% / 13.63% / 3.08%；累计单调性、"
        "分档求和、方向分类上界和重复键均为 0 违规。详见 "
        "`../../primitives/order_size_distribution_daily/manifest.json`。",
        "",
        "## Research boundary",
        "",
        "强 baseline 必须在后续统一 exposure audit 中检查行业、市值、股价、"
        "换手率和流动性暴露。本 Sprint 只扩展 feature space，不晋级、"
        "不优化、不组合。",
        "",
    ]
    (POOL_DIR / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitive", type=Path, default=DEFAULT_PRIMITIVE)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument("--factors", default="all")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--force-backtest", action="store_true")
    parser.add_argument("--skip-correlation", action="store_true")
    args = parser.parse_args(argv)
    if args.build_only and args.backtest_only:
        parser.error("--build-only and --backtest-only are mutually exclusive")

    names = _parse_names(args.factors)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
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
        summary, rank_ic_effective = _run_or_reuse(
            name,
            start,
            end,
            force=args.force_backtest,
        )
        direction = int(summary.get("factor_direction", 1))
        rank_ic_raw = rank_ic_effective * direction
        yearly_effective = yearly_ic_table(rank_ic_effective)
        yearly_raw = yearly_ic_table(rank_ic_raw)
        out_dir = Path(RESULT_ROOT) / name
        yearly_effective.to_csv(out_dir / "yearly_ic.csv")
        yearly_raw.to_csv(out_dir / "yearly_ic_raw.csv")
        yearly_raw.assign(factor=name).reset_index().to_csv(
            out_dir / "yearly_stability.csv", index=False
        )
        yearly_raw_parts.append(
            yearly_raw.assign(factor=name).reset_index()
        )
        yearly_effective_parts.append(
            yearly_effective.assign(factor=name).reset_index()
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
        row = {
            "factor": name,
            "mechanism": ORDER_SIZE_FACTOR_SPECS[name].mechanism,
            "lookback_days": ORDER_SIZE_FACTOR_SPECS[name].lookback_days,
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
            "net_annu_after_fee": (
                float(summary["hl_annu_ret_flipped"])
                - float(summary["implied_annu_fee"])
            ),
            "decile_mono_spearman": decile_monotonicity(summary),
            "n_days": int(summary["n_days"]),
            "n_names_avg": float(summary["n_names_avg"]),
            "n_factor_rows": coverage.get(name),
            **stability_fields(yearly_raw, raw_mean),
        }
        rows.append(row)
        print(
            f"[result] {name}: raw IC={raw_mean:+.4f}, "
            f"ICIR={raw_icir:+.2f}, "
            f"H-L Sharpe={row['hl_sharpe']:.2f}, "
            f"turnover={row['avg_hl_turnover']:.2f}, "
            f"year-sign={row['same_sign_years']}/{row['n_years']}",
            flush=True,
        )

    summary_frame = pd.DataFrame(rows)
    corr_path = POOL_DIR / "factor_correlation_spearman.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0).loc[names, names]
        annotations = redundancy_annotations(corr, threshold=0.80)
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

    summary_frame.to_csv(
        POOL_DIR / "candidate_summary.csv", index=False
    )
    pd.concat(yearly_raw_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_raw.csv", index=False
    )
    pd.concat(yearly_effective_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_effective.csv", index=False
    )
    _write_report(summary_frame)

    manifest = {
        "version": "order_size_candidate_pool_v1",
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
    (POOL_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
