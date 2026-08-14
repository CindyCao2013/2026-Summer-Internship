#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方向三：时序稳定性拆解——负 IC 是均匀分布还是集中在极端月份？

基于已有 rank_ic.csv（原始日频 Rank IC，未翻向）。

产出（research/results/l2_reproduction/mid_order_ratio/analysis/time_stability/）：
- monthly_ic_bar.png：月度 IC 条形图（正/负分色）
- rolling_3m_ic.png：滚动 3 个月 IC 均值
- ic_hist.png：日 IC 直方图 + 正态拟合
- monthly_summary.csv：月度统计 + 总体描述

用法:
  python l2_factor_reproduction/scripts/analyze_time_stability.py [--factor mid_order_ratio]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="mid_order_ratio")
    args = parser.parse_args()
    factor_name = args.factor

    result_root = Path(RESULT_ROOT)
    ic_path = result_root / factor_name / "rank_ic.csv"
    if not ic_path.exists():
        raise FileNotFoundError(ic_path)
    out_dir = result_root / factor_name / "analysis" / "time_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    ic = pd.read_csv(ic_path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    print(f"{factor_name}: {len(ic)} 日 RankIC, {ic.index[0].date()} ~ {ic.index[-1].date()}")

    monthly = ic.groupby(ic.index.to_period("M")).agg(["mean", "std", "count"])
    monthly.index = monthly.index.astype(str)
    monthly.to_csv(out_dir / "monthly_ic.csv")

    neg_share = float((monthly["mean"] < 0).mean())
    stats = {
        "daily_ic_mean": round(float(ic.mean()), 5),
        "daily_ic_std": round(float(ic.std()), 5),
        "n_months": int(len(monthly)),
        "months_ic_neg": int((monthly["mean"] < 0).sum()),
        "months_ic_neg_share": round(neg_share, 3),
        "monthly_ic_mean_avg": round(float(monthly["mean"].mean()), 5),
        "monthly_ic_mean_std": round(float(monthly["mean"].std()), 5),
        "worst_month": str(monthly["mean"].idxmax()),
        "best_month": str(monthly["mean"].idxmin()),
        "jan2024_monthly_ic": round(float(monthly.loc["2024-01", "mean"]), 5) if "2024-01" in monthly.index else None,
        "ex_jan2024_daily_ic_mean": round(float(ic[ic.index.to_period("M").astype(str) != "2024-01"].mean()), 5),
    }
    pd.DataFrame([stats]).to_csv(out_dir / "monthly_summary.csv", index=False)
    print(pd.DataFrame([stats]).T.to_string(header=False))

    # --- 图1：月度 IC 条形图 ---
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in monthly["mean"]]
    ax.bar(monthly.index, monthly["mean"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{factor_name} — Monthly mean RankIC (blue=negative/effective)")
    ax.set_ylabel("Monthly mean RankIC")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_ic_bar.png", dpi=120)
    plt.close(fig)

    # --- 图2：滚动 3 个月 IC 均值 ---
    roll3 = ic.rolling(63, min_periods=21).mean()
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ic.index, ic.values, color="gray", linewidth=0.4, alpha=0.5, label="daily")
    ax.plot(roll3.index, roll3.values, color="#1f77b4", linewidth=2, label="rolling 3M mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.set_title(f"{factor_name} — RankIC with rolling 3-month mean")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_3m_ic.png", dpi=120)
    plt.close(fig)

    # --- 图3：直方图 + 正态拟合 ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(ic.values, bins=60, density=True, alpha=0.6, color="#1f77b4", label="daily IC")
    xs = np.linspace(ic.min(), ic.max(), 300)
    mu, sigma = ic.mean(), ic.std()
    ax.plot(xs, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2),
            color="#d62728", linewidth=2, label=f"Normal(mu={mu:.4f}, sd={sigma:.4f})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.set_title(f"{factor_name} — Daily RankIC distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_hist.png", dpi=120)
    plt.close(fig)

    print(f"\n输出目录: {out_dir}/")
    print(f"负 IC 月份占比: {neg_share:.1%}（{stats['months_ic_neg']}/{stats['n_months']}）")
    if neg_share >= 0.7:
        print("=> 负 IC 月份占比 >= 70%：信号时序稳定 ✅")
    else:
        print("=> 负 IC 月份占比 < 70%：信号稳定性不足，需谨慎 ⚠️")


if __name__ == "__main__":
    main()
