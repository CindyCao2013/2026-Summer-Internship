#!/usr/bin/env python
"""Run the unified daily baseline for frozen Order Book Family v1."""

from __future__ import annotations

import argparse
import gc
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
    load_backtest_context,
)
from l2_factor_reproduction.python.candidate_pool import (  # noqa: E402
    decile_monotonicity,
    load_rank_ic,
    stability_fields,
    yearly_ic_table,
)
from l2_factor_reproduction.python.order_book_factors import (  # noqa: E402
    ORDER_BOOK_FACTOR_NAMES,
    ORDER_BOOK_FACTOR_SPECS,
    registry_frame,
)


DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
POOL_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "order_book_family"
FACTOR_ROOT = POOL_DIR / "factors"
PRIMITIVE_DIR = Path(RESULT_ROOT) / "primitives" / "order_book_daily"


def _parse_names(value: str) -> List[str]:
    if value.strip().lower() in {"", "all"}:
        return list(ORDER_BOOK_FACTOR_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(ORDER_BOOK_FACTOR_SPECS))
    if unknown:
        raise ValueError(f"Unknown Order Book factors: {unknown}")
    return names


def _write_registry(names: List[str]) -> None:
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


def _run_or_reuse(
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    mask: pd.DataFrame,
    ret_matrix: pd.DataFrame,
    force: bool,
) -> Tuple[Dict[str, object], pd.Series]:
    output = FACTOR_ROOT / name
    summary_path = output / "summary.json"
    rank_ic_path = output / "rank_ic.csv"
    if not force and summary_path.exists() and rank_ic_path.exists():
        print(f"[backtest] reuse {name}", flush=True)
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            load_rank_ic(rank_ic_path),
        )
    narrow_path = output / "factor_narrow.parquet"
    narrow = pd.read_parquet(
        narrow_path,
        columns=["symbol", "tradetime", "value"],
    )
    group_pnl, group_turnover, rank_ic, summary = backtest_factor(
        narrow,
        start_day=start,
        end_day=end,
        signal_shift=1,
        mask=mask,
        ret_matrix=ret_matrix,
    )
    summary["net_annu_after_fee"] = (
        float(summary["hl_annu_ret_flipped"])
        - float(summary["implied_annu_fee"])
    )
    _save_backtest_outputs(
        str(output),
        group_pnl,
        group_turnover,
        rank_ic,
        summary,
        factor_name=name,
    )
    del narrow
    gc.collect()
    return summary, rank_ic


def _criterion_list(frame: pd.DataFrame, expression: pd.Series) -> str:
    names = frame.loc[expression.fillna(False), "factor"].tolist()
    return ", ".join(f"`{name}`" for name in names) if names else "无"


def _read_cross_reference(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _write_report(summary: pd.DataFrame) -> None:
    primitive_manifest_path = PRIMITIVE_DIR / "manifest.json"
    primitive_manifest = (
        json.loads(primitive_manifest_path.read_text(encoding="utf-8"))
        if primitive_manifest_path.exists()
        else {}
    )
    primitive_rows = int(primitive_manifest.get("row_count", 0))
    eligible_rows = int(primitive_manifest.get("eligible_row_count", 0))
    primitive_coverage = float(
        primitive_manifest.get("coverage_statistics", {}).get(
            "mean", np.nan
        )
    )
    missing_slope = int(
        primitive_manifest.get("invalid_row_counts", {}).get(
            "missing_slope_symbol_days", 0
        )
    )
    clusters_path = POOL_DIR / "redundancy_clusters_080.csv"
    clusters = (
        pd.read_csv(clusters_path)
        if clusters_path.exists()
        else pd.DataFrame()
    )
    n_clusters = (
        int(clusters["redundancy_cluster_080"].nunique())
        if len(clusters)
        else 0
    )
    g10_over_three = summary["g10_excess_sharpe"] > 3
    hl_over_three = summary["hl_sharpe"] > 3
    all_quality = (
        summary["rank_ic_raw"].abs().ge(0.02)
        & summary["icir_raw"].abs().ge(3)
        & summary["sign_consistency"].ge(0.75)
        & summary["decile_mono_spearman"].ge(0.8)
    )
    endpoint = (
        (g10_over_three | hl_over_three)
        & summary["decile_mono_spearman"].lt(0.8)
    )
    trade_cross = _read_cross_reference(
        POOL_DIR / "order_book_vs_trade_flow_corr.csv"
    )
    size_cross = _read_cross_reference(
        POOL_DIR / "order_book_vs_order_size_corr.csv"
    )
    cross_ready = len(trade_cross) > 0 and len(size_cross) > 0
    if cross_ready:
        cross = pd.concat([trade_cross, size_cross], ignore_index=True)
        low_corr = (
            cross.groupby("order_book_factor")[
                "abs_mean_daily_spearman"
            ]
            .max()
            .sort_values()
        )
        low_corr_text = ", ".join(
            f"`{name}` ({value:.2f})"
            for name, value in low_corr.head(10).items()
        )
        cross_max = float(low_corr.max())
        cross_median = float(low_corr.median())
        cross_low_count = int((low_corr < 0.50).sum())
    else:
        low_corr_text = "跨家族相关任务尚未落盘"
        cross_max = float("nan")
        cross_median = float("nan")
        cross_low_count = 0

    display_columns = [
        "factor",
        "rank_ic_raw",
        "icir_raw",
        "g10_excess_sharpe",
        "hl_sharpe",
        "avg_hl_turnover",
        "net_annu_after_fee",
        "sign_consistency",
        "decile_mono_spearman",
        "redundancy_cluster_080",
    ]
    try:
        table = summary[display_columns].to_markdown(index=False)
    except ImportError:
        table = (
            "```\n"
            + summary[display_columns].to_string(index=False)
            + "\n```"
        )
    lines = [
        "# L2 Candidate Pool v1 — Order Book Family",
        "",
        "范围：冻结公式 baseline discovery 与 evidence-based taxonomy。"
        "未做参数优化、二次中性化、周度优化、组合或 KEEP/DROP。",
        "",
        f"- 冻结公式数：{len(summary)}",
        f"- |日截面 Spearman| ≥ 0.80 经验簇：{n_clusters}",
        "- IC/ICIR：冻结原方向；分组收益：统一 effective direction",
        "",
        "## Unified baseline",
        "",
        table,
        "",
        "## Required questions",
        "",
        "1. **SSL2 覆盖**：SSE 2015-01-05 起、SZSE 2008-01-02 起；"
        "两市均覆盖本次 2019-01-01~2026-07-31 目标区间。审计时两表"
        "最新均为 2026-08-04。",
        "",
        "2. **两市十档一致性**：价量十档可统一；SSE 原始数组恒为十档，"
        "SZSE 必须过滤 `Type=010` 的空/一档伴随行。SZSE 无十档 Nums，"
        "且 Bid/AskVWAP 不可用，因此 v1 不用 Nums、统一自算十档 VWAP。",
        "",
        "3. **更新频率偏差**：存在。活跃股票约 3 秒一条，但每股日内"
        "Snapshot 数差异显著，raw-row 平均会按更新次数加权。",
        "",
        "4. **固定分钟抽样效果**：2024-06 整月 96,733 个 symbol-day "
        "硬检查全通过；固定分钟与 raw-row OBI5 日截面 Spearman=0.992，"
        "但平均绝对差=1.51%，且差异与 Snapshot 数相关=-0.42。"
        "因此它保留主截面结构并实质移除了更新次数权重。",
        "",
        f"5. **冻结公式**：{len(summary)} 个。",
        "",
        f"6. **经验相关簇**：{n_clusters} 个。公式数不等于独立 alpha 数。",
        "",
        "7. **G10 Excess Sharpe > 3**："
        + _criterion_list(summary, g10_over_three)
        + "。",
        "",
        "8. **H-L Sharpe > 3**："
        + _criterion_list(summary, hl_over_three)
        + "。",
        "",
        "9. **质量阈值**：",
        "- |RankIC| ≥ 2%："
        + _criterion_list(summary, summary["rank_ic_raw"].abs() >= 0.02),
        "- |ICIR| ≥ 3："
        + _criterion_list(summary, summary["icir_raw"].abs() >= 3),
        "- 年度同号率 ≥ 75%："
        + _criterion_list(summary, summary["sign_consistency"] >= 0.75),
        "- decile monotonicity ≥ 0.8："
        + _criterion_list(
            summary, summary["decile_mono_spearman"] >= 0.8
        ),
        "- 四项同时满足：" + _criterion_list(summary, all_quality),
        "",
        "10. **高 Sharpe 但非单调端点效应**："
        + _criterion_list(summary, endpoint)
        + "。该标签只表示需要警惕，不作 DROP。",
        "",
        "11. **与 Trade Flow / Order Size 低相关**："
        + low_corr_text
        + "。该参考按本族表现前 10 加各簇代表、已有家族各簇代表计算，"
        "不是全 32×29 矩阵；选择清单见 `cross_family_selection.csv`。",
        "",
        "12. **是否扩展新维度**："
        + (
            f"跨家族参考已完成；各 Order Book 代表对已有参考的最大 |ρ| "
            f"中位数为 {cross_median:.2f}，其中 {cross_low_count} 个低于 "
            f"0.50（最坏上界 {cross_max:.2f}）。这支持至少存在新的盘口"
            "信息方向，而不是宣称全部公式独立。"
            if cross_ready
            else "需等跨家族只读相关文件完成后作最终证据判断。"
        ),
        "",
        "13. **质量/覆盖/容量限制**：源表约 895 亿 raw rows；"
        "SZSE 原始空数组与重复时间戳必须先过滤；coverage<0.80 不出"
        f"因子。日频 primitive={primitive_rows:,} 行、eligible="
        f"{eligible_rows:,} 行、平均 coverage={primitive_coverage:.2%}、"
        f"斜率缺失={missing_slope:,} 行；少量斜率缺失来自有效价格距离"
        "不足。primitive 按季度"
        "分区、zstd 压缩，不缓存 raw/minute 明细。",
        "",
        "14. **下一 family**：优先 Price Formation。原因是两市都能统一"
        "使用价格、价差、microprice 与成交结果，而 Cancellation 原生撤单"
        "字段仅 SSE 具备，直接扩展会产生结构性单市场缺失。该建议不涉及"
        "本 family 的生产晋级。",
        "",
        "## Boundaries",
        "",
        "高 Sharpe、低相关或稳定 IC 均只是在冻结 baseline 下的证据。"
        "本报告不作正式 KEEP/DROP、生产晋级或组合结论。",
        "",
    ]
    (POOL_DIR / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument("--factors", default="all")
    parser.add_argument("--force-backtest", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    names = _parse_names(args.factors)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        _write_report(pd.read_csv(POOL_DIR / "candidate_summary.csv"))
        print(f"[done] refreshed {POOL_DIR / 'report.md'}", flush=True)
        return 0
    _write_registry(names)
    coverage = pd.read_csv(POOL_DIR / "factor_coverage.csv").set_index(
        "factor"
    )
    print("[backtest] loading shared return/mask context once", flush=True)
    mask, ret_matrix = load_backtest_context(start, end)

    rows = []
    yearly_parts = []
    for position, name in enumerate(names, start=1):
        print(f"[backtest {position}/{len(names)}] {name}", flush=True)
        summary, rank_ic_effective = _run_or_reuse(
            name,
            start,
            end,
            mask=mask,
            ret_matrix=ret_matrix,
            force=args.force_backtest,
        )
        direction = int(summary.get("factor_direction", 1))
        rank_ic_raw = rank_ic_effective * direction
        yearly_raw = yearly_ic_table(rank_ic_raw)
        yearly_effective = yearly_ic_table(rank_ic_effective)
        factor_dir = FACTOR_ROOT / name
        yearly_raw.to_csv(factor_dir / "yearly_ic.csv")
        yearly_effective.to_csv(factor_dir / "yearly_ic_effective.csv")
        yearly_raw.assign(factor=name).reset_index().to_csv(
            factor_dir / "yearly_stability.csv", index=False
        )
        yearly_parts.append(yearly_raw.assign(factor=name).reset_index())

        raw_mean = float(
            summary.get("rank_ic_mean_raw", rank_ic_raw.mean())
        )
        raw_std = float(rank_ic_raw.std())
        raw_icir = (
            raw_mean / raw_std * np.sqrt(250)
            if raw_std > 0
            else float("nan")
        )
        coverage_row = coverage.loc[name]
        row = {
            "factor": name,
            "category": ORDER_BOOK_FACTOR_SPECS[name].category,
            "mechanism": ORDER_BOOK_FACTOR_SPECS[name].mechanism,
            "lookback_days": ORDER_BOOK_FACTOR_SPECS[name].lookback_days,
            "n_factor_rows": int(coverage_row["n_factor_rows"]),
            "date_min": coverage_row["date_min"],
            "date_max": coverage_row["date_max"],
            "n_symbols": int(coverage_row["n_symbols"]),
            "factor_direction": direction,
            "direction_flip": bool(direction < 0),
            "rank_ic_raw": raw_mean,
            "icir_raw": raw_icir,
            "rank_ic_std": raw_std,
            "positive_ic_fraction": float(
                (rank_ic_raw.dropna() > 0).mean()
            ),
            "rank_ic_effective": float(summary["rank_ic_mean"]),
            "icir_effective": float(summary["rank_icir"]),
            "hl_annu_ret": float(summary["hl_annu_ret_flipped"]),
            "hl_sharpe": float(summary["hl_sharpe_flipped"]),
            "g10_excess_annu_ret": float(
                summary["g10_excess_annu_ret"]
            ),
            "g10_excess_sharpe": float(
                summary["g10_excess_sharpe"]
            ),
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
            "group_pnl_saved_direction": summary[
                "group_pnl_saved_direction"
            ],
            **stability_fields(yearly_raw, raw_mean),
        }
        rows.append(row)
        print(
            f"[result] raw IC={raw_mean:+.4f} ICIR={raw_icir:+.2f} "
            f"G10={row['g10_excess_sharpe']:.2f} "
            f"H-L={row['hl_sharpe']:.2f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(rows)
    clusters_path = POOL_DIR / "redundancy_clusters_080.csv"
    if clusters_path.exists():
        annotations = pd.read_csv(clusters_path)[
            [
                "factor",
                "redundancy_cluster_080",
                "max_corr_peer",
                "max_abs_corr",
                "near_alias_observed",
            ]
        ]
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
    pd.concat(yearly_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_all.csv", index=False
    )
    _write_report(summary_frame)
    manifest = {
        "version": "order_book_candidate_pool_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "primitive": str(PRIMITIVE_DIR),
        "n_candidates": len(names),
        "factors": names,
        "signal_shift": 1,
        "cost_bps": 7.5,
        "group_direction": "effective",
        "rank_ic_direction": "raw frozen formula",
        "scope_exclusions": [
            "parameter_optimization",
            "factor_combination",
            "machine_learning",
            "weekly_rebalance_optimization",
            "second_neutralization",
            "cross_family_keep_drop",
        ],
    }
    (POOL_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] baseline -> {POOL_DIR}\n"
        + summary_frame[
            [
                "factor",
                "rank_ic_raw",
                "icir_raw",
                "g10_excess_sharpe",
                "hl_sharpe",
            ]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
