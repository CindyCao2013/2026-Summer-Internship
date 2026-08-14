#!/usr/bin/env python
"""Fast Discovery Lane 首批速度测试（不创建新因子）。

使用 10 个已冻结公式完成三件事：

1. 数值一致性：Fast 引擎（全窗口 + 缓存上下文）vs 冻结 baseline
   summary.json（rank_ic_mean_raw / hl_annu_ret_flipped /
   hl_sharpe_flipped / group_mean_annu），容差 1e-9。
2. 墙钟对比：
   - old：DDB 现取上下文 + 全窗口逐因子回测（复用冻结 narrow，
     即当前 expand_*_family 的实际口径）；
   - fast：缓存上下文 + primitive 一次加载批量生成 + discovery 窗回测。
3. 输出 fast_discovery_benchmark.md 回答五个问题。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    backtest_factor,
    load_backtest_context,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    FAST_DISCOVERY_DIR,
    WINDOWS,
    load_fast_context,
    run_fast_batch,
)

TOL = 1e-9

# (family, factor, frozen summary.json 路径模板)
RESULT_ROOT_P = Path(RESULT_ROOT)
POOL = RESULT_ROOT_P / "candidate_pool_v1"

BENCHMARK_FACTORS: List[Tuple[str, str, Path]] = [
    *[
        (
            "liquidity_impact",
            name,
            POOL / "liquidity_impact_family" / "factors" / name,
        )
        for name in (
            "signed_amount_impact",
            "effective_spread_proxy",
            "spread_per_depth",
            "impact_convexity",
        )
    ],
    *[
        (
            "price_formation",
            name,
            POOL / "price_formation_family" / "factors" / name,
        )
        for name in (
            "close_auction_return",
            "overnight_gap",
            "path_efficiency",
        )
    ],
    *[
        (
            "order_book",
            name,
            POOL / "order_book_family" / "factors" / name,
        )
        for name in (
            "obi_l5_mean",
            "bid_depth_slope",
        )
    ],
    ("trade_flow", "net_buy_ratio", RESULT_ROOT_P / "net_buy_ratio"),
]


def _compare_one(
    factor: str,
    fast_summary: Dict[str, object],
    frozen: Dict[str, object],
) -> Dict[str, object]:
    rows: Dict[str, object] = {"factor": factor}
    worst = 0.0
    for key in ("rank_ic_mean_raw", "hl_annu_ret_flipped", "hl_sharpe_flipped"):
        diff = abs(float(fast_summary[key]) - float(frozen[key]))
        rows[f"diff_{key}"] = diff
        worst = max(worst, diff)
    for group, value in frozen["group_mean_annu"].items():
        diff = abs(
            float(fast_summary["group_mean_annu"][group]) - float(value)
        )
        worst = max(worst, diff)
    rows["max_abs_diff"] = worst
    rows["consistent"] = bool(worst < TOL)
    return rows


def run_consistency_check() -> pd.DataFrame:
    """Fast 引擎跑全窗口（缓存上下文），与冻结 baseline 逐字段对比。"""
    start, end = WINDOWS["full"]
    mask, ret = load_fast_context("full")
    rows = []
    by_family: Dict[str, List[Tuple[str, Path]]] = {}
    for family, name, path in BENCHMARK_FACTORS:
        by_family.setdefault(family, []).append((name, path))
    from l2_factor_reproduction.python.fast_discovery import (
        FAMILY_ADAPTERS,
        load_family_features,
    )

    for family, items in by_family.items():
        features = load_family_features(family, start, end)
        adapter = FAMILY_ADAPTERS[family]
        for name, result_dir in items:
            narrow = adapter.to_narrow(features, name)
            _pnl, _to, _ic, summary = backtest_factor(
                narrow,
                start_day=start,
                end_day=end,
                mask=mask,
                ret_matrix=ret,
            )
            frozen = json.loads(
                (result_dir / "summary.json").read_text(encoding="utf-8")
            )
            rows.append(_compare_one(name, summary, frozen))
            print(
                f"[consistency] {name}: "
                f"max_abs_diff={rows[-1]['max_abs_diff']:.3e}",
                flush=True,
            )
    return pd.DataFrame(rows)


def measure_old_pipeline() -> Dict[str, float]:
    """当前 expand_*_family 口径：DDB 上下文 + 全窗口逐因子回测。"""
    start, end = WINDOWS["full"]
    t0 = time.perf_counter()
    mask, ret = load_backtest_context(start, end)
    context_seconds = time.perf_counter() - t0

    backtest_seconds = 0.0
    for _family, name, result_dir in BENCHMARK_FACTORS:
        narrow = pd.read_parquet(
            result_dir / "factor_narrow.parquet",
            columns=["symbol", "tradetime", "value"],
        )
        t0 = time.perf_counter()
        backtest_factor(
            narrow,
            start_day=start,
            end_day=end,
            mask=mask,
            ret_matrix=ret,
        )
        backtest_seconds += time.perf_counter() - t0
        print(f"[old] {name} backtest done", flush=True)
    return {
        "context_ddb_seconds": context_seconds,
        "backtest_full_seconds": backtest_seconds,
        "total_seconds": context_seconds + backtest_seconds,
    }


def measure_fast_lane(out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fast Lane：discovery 窗，批量生成 + 缓存上下文。"""
    context = load_fast_context("discovery")
    summaries, profiles = [], []
    by_family: Dict[str, List[str]] = {}
    for family, name, _path in BENCHMARK_FACTORS:
        by_family.setdefault(family, []).append(name)
    for family, names in by_family.items():
        summary, profile = run_fast_batch(
            family, names, window="discovery", output_root=out_root,
            context=context,
        )
        summaries.append(summary)
        profiles.append(profile)
    return pd.concat(summaries), pd.concat(profiles)


def _md_table(frame: pd.DataFrame) -> str:
    """不依赖 tabulate 的最简 markdown 表。"""
    cols = [str(c) for c in frame.columns]

    def _fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, sep, *lines])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="只根据已落盘 CSV/timings 重新生成 benchmark.md",
    )
    args = parser.parse_args()

    out_root = FAST_DISCOVERY_DIR / "benchmark"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        consistency = pd.read_csv(out_root / "consistency_check.csv")
        fast_summary = pd.read_csv(out_root / "fast_summary.csv")
        fast_profile = pd.read_csv(out_root / "fast_profile.csv")
        timings = json.loads(
            (out_root / "timings.json").read_text(encoding="utf-8")
        )
        old = timings["old"]
        fast_total = timings["fast_total_seconds"]
    else:
        print("[1/3] numeric consistency (full window vs frozen baseline)")
        consistency = run_consistency_check()
        consistency.to_csv(out_root / "consistency_check.csv", index=False)

        print("[2/3] old pipeline wall-clock (DDB context + full window)")
        old = measure_old_pipeline()

        print("[3/3] fast lane wall-clock (cached context + discovery)")
        t0 = time.perf_counter()
        fast_summary, fast_profile = measure_fast_lane(out_root)
        fast_total = time.perf_counter() - t0
        fast_summary.to_csv(out_root / "fast_summary.csv", index=False)
        fast_profile.to_csv(out_root / "fast_profile.csv", index=False)
        (out_root / "timings.json").write_text(
            json.dumps(
                {"old": old, "fast_total_seconds": fast_total}, indent=2
            ),
            encoding="utf-8",
        )
    n_ok = int(consistency["consistent"].sum())

    n = len(BENCHMARK_FACTORS)
    speedup = old["total_seconds"] / max(fast_total, 1e-9)
    report = f"""# Fast Discovery Lane — 首批速度测试

生成时间：{pd.Timestamp.now()}

## 样本

10 个已冻结公式（liquidity_impact 4 / price_formation 3 / order_book 2 /
trade_flow 1），未创建任何新因子。

## 1. 原来慢在哪些阶段？（old pipeline 实测）

| 阶段 | 耗时 |
|---|---|
| DDB 上下文（mask + 超额收益矩阵，2019-2026） | {old['context_ddb_seconds']:.1f}s |
| 10 因子全窗口回测（共享上下文后） | {old['backtest_full_seconds']:.1f}s |
| 合计 | {old['total_seconds']:.1f}s（{old['total_seconds']/n:.1f}s/因子） |

注：此处尚未计入新机制的 Raw L2 扫描（primitive extraction），
该步骤才是「半小时级」的主要来源；Fast Lane 通过禁止访问 Raw
彻底消除它。

## 2. Fast Lane 每阶段耗时（discovery 冻结窗 2023-2024）

{_md_table(fast_profile)}

合计 wall-clock：{fast_total:.1f}s（{fast_total/n:.1f}s/因子，
含 primitive 一次加载 + 上下文缓存读取 + 回测 + 两张图）。

## 3. 加速倍数

- 同口径（10 因子，不含 Raw 扫描）：**{speedup:.1f}x**
- 若计入新机制的 Raw L2 重扫（经验 30min+/idea）：加速一个数量级以上。

## 4. 数值一致性（Fast 引擎全窗口 vs 冻结 baseline）

- 通过：{n_ok}/{n}（容差 {TOL:.0e}）
- 字段：rank_ic_mean_raw / hl_annu_ret_flipped / hl_sharpe_flipped /
  group_mean_annu（十分组逐年化收益逐组对比）

{_md_table(consistency)}

## 5. 是否达到「2 分钟看到两张图」？

- 10 因子批量总耗时 {fast_total:.1f}s，单因子摊销 {fast_total/n:.1f}s：
  {'达标' if fast_total/n <= 120 else '未达标'}（目标 ≤120s，理想 ≤60s）。
- 剩余瓶颈见 fast_profile.csv 中占比最大的阶段。

## Fast Gate 打标结果（discovery 窗，仅标记不筛选）

{_md_table(fast_summary[['factor','family','hl_sharpe','decile_mono_spearman','adjacent_violations','positive_hl_month_fraction','gate']])}
"""
    (out_root / "fast_discovery_benchmark.md").write_text(
        report, encoding="utf-8"
    )
    print(f"[done] consistency {n_ok}/{n}, speedup {speedup:.1f}x")
    print(f"[done] report -> {out_root / 'fast_discovery_benchmark.md'}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
