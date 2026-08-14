#!/usr/bin/env python
"""Run the frozen daily baseline for Liquidity / Price Impact Family v1."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from l2_factor_reproduction.config.settings import (  # noqa: E402
    RESULT_ROOT,
    UNIVERSE,
)
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
from l2_factor_reproduction.python.liquidity_impact_factors import (  # noqa: E402
    LIQUIDITY_IMPACT_FACTOR_NAMES,
    LIQUIDITY_IMPACT_FACTOR_SPECS,
    registry_frame,
)


DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
POOL_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "liquidity_impact_family"
)
FACTOR_ROOT = POOL_DIR / "factors"
PRIMITIVE_DIR = Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily"
CROSS_FAMILIES = (
    "trade_flow",
    "order_size",
    "order_book",
    "price_formation",
)


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_names(value: str) -> List[str]:
    if value.strip().lower() in {"", "all"}:
        return list(LIQUIDITY_IMPACT_FACTOR_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(
        set(names).difference(LIQUIDITY_IMPACT_FACTOR_SPECS)
    )
    if unknown:
        raise ValueError(f"Unknown Liquidity/Impact factors: {unknown}")
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
    narrow = pd.read_parquet(
        output / "factor_narrow.parquet",
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


def _cross_family_summary() -> Tuple[str, str]:
    frames = []
    for family in CROSS_FAMILIES:
        path = POOL_DIR / f"liquidity_impact_vs_{family}_corr.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return "跨家族只读相关尚未落盘。", "pending"
    cross = pd.concat(frames, ignore_index=True)
    maxima = (
        cross.groupby("liquidity_impact_factor")[
            "abs_mean_daily_spearman"
        ]
        .max()
        .sort_values()
    )
    low = ", ".join(
        f"`{name}` ({value:.2f})"
        for name, value in maxima.head(10).items()
    )
    text = (
        f"各 Liquidity/Impact 代表对既有四族参考的最大 |ρ| 中位数="
        f"{maxima.median():.2f}；最低 10 个为 {low}。"
        "选择清单见 `cross_family_selection.csv`，该结果不用于筛选。"
    )
    return text, "read_only_references_complete"


def _write_report(summary: pd.DataFrame) -> None:
    primitive_manifest_path = PRIMITIVE_DIR / "manifest.json"
    primitive_manifest = json.loads(
        primitive_manifest_path.read_text(encoding="utf-8")
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
    ic_threshold = summary["rank_ic_raw"].abs() >= 0.02
    icir_threshold = summary["icir_raw"].abs() >= 3
    yearly_threshold = summary["sign_consistency"] >= 0.75
    monotonicity_threshold = summary["decile_mono_spearman"] >= 0.8
    all_quality = (
        ic_threshold
        & icir_threshold
        & yearly_threshold
        & monotonicity_threshold
    )
    proxy_flags = summary["factor"].map(
        lambda name: LIQUIDITY_IMPACT_FACTOR_SPECS[name].proxy_note != ""
    )
    exact_names = summary.loc[~proxy_flags, "factor"].tolist()
    proxy_names = summary.loc[proxy_flags, "factor"].tolist()
    cross_text, _cross_status = _cross_family_summary()
    display_columns = [
        "factor",
        "category",
        "rank_ic_raw",
        "icir_raw",
        "g10_excess_sharpe",
        "hl_sharpe",
        "hl_mdd",
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
    family_categories = summary["category"].nunique()
    quality = pd.read_csv(PRIMITIVE_DIR / "primitive_quality.csv")
    coverage_mean = float(quality["coverage_mean"].mean())
    universe_note = primitive_manifest.get("universe_limitation", "")
    lines = [
        "# L2 Candidate Pool v1 — Liquidity / Price Impact Family",
        "",
        "范围：冻结 level 公式 baseline discovery、年度稳定性与只读冗余"
        "证据。未做参数搜索、窗口比较、组合、机器学习或 KEEP/DROP。",
        "",
        "## Research accounting",
        "",
        f"- Formula count：{len(summary)}",
        f"- Empirical clusters（|mean daily Spearman| ≥ 0.80）：{n_clusters}",
        f"- Independent mechanism directions（研究分类，不等同独立 alpha）："
        f"{family_categories}",
        "- Production-ready factors：本 Sprint 不判定（not assessed）",
        f"- Exact metrics：{len(exact_names)} 个公式直接由冻结 primitive"
        "字段计算；Proxy metrics："
        f"{len(proxy_names)} 个（`{'`, `'.join(proxy_names)}`）为分钟级"
        "近似，已在 primitive manifest `proxy_limitations` 中披露，"
        "不得冒充逐笔 prevailing-quote 口径。",
        "- Long-leg alpha 与 short-leg spread alpha 分开呈现：G10 excess"
        " 列度量多头腿超额；H-L 列含空头腿，两者不可混用。",
        "- Candidate Pool 计数使用 formula count；不把 rank/z-score/符号克隆"
        "重复计数。",
        "",
        "## Canonical primitive",
        "",
        f"- Source：`{primitive_manifest['canonical_source']}`",
        f"- Rows：{int(primitive_manifest['row_count']):,}",
        f"- Coverage mean（跨分区平均）：{coverage_mean:.2%}",
        "- Continuous grid：09:30–11:29、13:00–14:59；集合竞价分钟剔除；"
        "forward mid return 仅在当日分钟网格内计算，不跨日。",
        "- SSE 主动方向用 BSFlag；SZSE 用冻结 BidOrderNo/AskOrderNo 规则；"
        "中性成交以 neutral_trade_share 记录。",
        f"- Universe 限制：{universe_note}",
        "- ClickHouse KLIN 只用于 2024-06 validation month parity；"
        "没有静默拼接。",
        "",
        "## Unified baseline",
        "",
        f"- T+1 `signal.shift(1)`；benchmark `{UNIVERSE}`；7.5 bps implied fee。",
        "- Raw IC 保留冻结方向；effective direction 仅用于分组展示；"
        "production direction 未决定。",
        "- ClickHouse L2 覆盖为全市场子集：因子值缺失的股票不参与当日"
        "截面排名，回测在可覆盖子宇宙上进行，该限制已在 manifest 披露。",
        "",
        table,
        "",
        "## Required threshold lists",
        "",
        "- G10 Excess Sharpe > 3："
        + _criterion_list(summary, g10_over_three),
        "- H-L Sharpe > 3："
        + _criterion_list(summary, hl_over_three),
        "- |RankIC| ≥ 2%：" + _criterion_list(summary, ic_threshold),
        "- |ICIR| ≥ 3：" + _criterion_list(summary, icir_threshold),
        "- yearly consistency ≥ 75%："
        + _criterion_list(summary, yearly_threshold),
        "- monotonicity ≥ 0.8："
        + _criterion_list(summary, monotonicity_threshold),
        "- 四项 IC 质量阈值同时满足："
        + _criterion_list(summary, all_quality),
        "",
        "阈值结果没有反向用于修改公式或窗口。",
        "",
        "## Redundancy",
        "",
        f"- {len(summary)} 个冻结公式形成 {n_clusters} 个经验簇；"
        "公式数不等于独立 alpha 数。",
        "- 族内矩阵：`factor_correlation_spearman.csv`；高相关 pair："
        "`high_corr_pairs.csv`；cluster：`redundancy_clusters_080.csv`；"
        "heatmap：`correlation_heatmap.png`。",
        "",
        "## Cross-family read-only reference",
        "",
        cross_text,
        "",
        "## Boundary",
        "",
        "高 Sharpe、稳定 IC、单调性或低相关都只是冻结 baseline 下的"
        "研究证据。本报告不作生产晋级、组合或 KEEP/DROP。",
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
    yearly_raw_parts = []
    yearly_effective_parts = []
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
        yearly_raw.to_csv(factor_dir / "yearly_ic_raw.csv")
        yearly_effective.to_csv(
            factor_dir / "yearly_ic_effective.csv"
        )
        yearly_raw.assign(factor=name).reset_index().to_csv(
            factor_dir / "yearly_stability.csv", index=False
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
        coverage_row = coverage.loc[name]
        spec = LIQUIDITY_IMPACT_FACTOR_SPECS[name]
        row = {
            "factor": name,
            "category": spec.category,
            "mechanism": spec.mechanism,
            "lookback_days": spec.lookback_days,
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
    pd.concat(yearly_raw_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_raw.csv", index=False
    )
    pd.concat(yearly_effective_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_effective.csv", index=False
    )
    _write_report(summary_frame)

    primitive_manifest = PRIMITIVE_DIR / "manifest.json"
    formula_module = (
        PROJ_ROOT
        / "l2_factor_reproduction/python/liquidity_impact_factors.py"
    )
    evaluation_module = (
        PROJ_ROOT / "l2_factor_reproduction/python/backtest.py"
    )
    registry_path = POOL_DIR / "factor_registry.csv"
    _cross_text, cross_status = _cross_family_summary()
    manifest = {
        "version": "liquidity_impact_candidate_pool_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "primitive": str(PRIMITIVE_DIR),
        "primitive_manifest_sha256": _sha256(primitive_manifest),
        "formula_module_sha256": _sha256(formula_module),
        "evaluation_module_sha256": _sha256(evaluation_module),
        "registry_sha256": _sha256(registry_path),
        "n_candidates": len(names),
        "factors": names,
        "benchmark": UNIVERSE,
        "signal_shift": 1,
        "cost_bps": 7.5,
        "group_direction": "effective display only",
        "rank_ic_direction": "raw frozen formula",
        "production_direction": "not decided",
        "cross_family_correlation": cross_status,
        "scope_exclusions": [
            "parameter_optimization",
            "window_comparison",
            "factor_combination",
            "machine_learning",
            "second_neutralization",
            "keep_drop",
            "size_bucket_reoptimization",
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
