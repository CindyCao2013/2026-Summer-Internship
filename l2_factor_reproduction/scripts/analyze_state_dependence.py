#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方向一：状态依赖性检验——mid_order_ratio 的 IC 是否依赖于流动性状态？

每日按换手率（S_DQ_TURN，原始日值）将股票等数量三分（Low/Medium/High），
分别计算因子在组内的 Rank IC（vs 次日 c2c 超额收益）。

产出（research/results/l2_reproduction/mid_order_ratio/analysis/state_dependence/）：
- ic_series.png：三组每日 IC 叠加
- ic_rolling.png：三组 IC 滚动 20 日均值
- ic_boxplot.png：三组 IC 分布盒图
- summary.csv：三组 IC Mean/Std/ICIR/IC<0 天数占比

用法:
  python l2_factor_reproduction/scripts/analyze_state_dependence.py [--factor mid_order_ratio]
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

from Factor_Dev_Lib import (  # noqa: E402
    get_EOD_Not_Limit,
    get_EOD_Not_ST,
    get_TradeStatus,
)
from l2_factor_reproduction.config.settings import (  # noqa: E402
    END_DAY,
    RESULT_ROOT,
    START_DAY,
)
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    narrow_to_wide,
    prepare_factor_signal,
)
from l2_factor_reproduction.scripts.test_double_neutralization import (  # noqa: E402
    _get_turnover_wide,
    WINDOW,
)


def _group_ic(signal: pd.DataFrame, ret: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """每日按 state（换手率）三等分，组内 spearman IC。返回 Date x {Low,Mid,High}。"""
    ics = {"Low": [], "Mid": [], "High": []}
    dates = []
    for dt in signal.index:
        s = signal.loc[dt]
        r = ret.loc[dt]
        t = state.loc[dt] if dt in state.index else pd.Series(dtype=float)
        valid = s.notna() & r.notna() & t.notna()
        if valid.sum() < 60:
            continue
        sv, rv, tv = s[valid], r[valid], t[valid]
        q1, q2 = tv.quantile(1 / 3), tv.quantile(2 / 3)
        grp = pd.Series("Mid", index=tv.index)
        grp[tv <= q1] = "Low"
        grp[tv > q2] = "High"
        dates.append(dt)
        for g in ("Low", "Mid", "High"):
            idx = grp[grp == g].index
            if len(idx) < 20:
                ics[g].append(np.nan)
            else:
                ics[g].append(sv[idx].corr(rv[idx], method="spearman"))
    out = pd.DataFrame(ics, index=pd.DatetimeIndex(dates))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="mid_order_ratio")
    args = parser.parse_args()
    factor_name = args.factor

    result_root = Path(RESULT_ROOT)
    out_dir = result_root / factor_name / "analysis" / "state_dependence"
    out_dir.mkdir(parents=True, exist_ok=True)

    narrow = pd.read_parquet(result_root / factor_name / "factor_narrow.parquet")
    factor_wide = narrow_to_wide(narrow).loc[START_DAY:END_DAY]
    mask = (
        get_EOD_Not_Limit(START_DAY, END_DAY)
        * get_EOD_Not_ST(START_DAY, END_DAY)
        * get_TradeStatus(START_DAY, END_DAY)
    )
    signal, ret = prepare_factor_signal(
        factor_wide, start=START_DAY, end=END_DAY, mask=mask, signal_shift=1
    )

    print("获取每日换手率（状态变量）...")
    # _get_turnover_wide 返回 20 日均值 log 换手率——流动性状态用平滑值更稳
    turnover = _get_turnover_wide(START_DAY, END_DAY)
    turnover = turnover.reindex(index=signal.index, columns=signal.columns)

    print("按换手率三分组计算组内 RankIC ...")
    ic_df = _group_ic(signal, ret, turnover)
    ic_df.to_csv(out_dir / "group_ic_daily.csv")
    print(f"有效天数: {len(ic_df)}")

    # --- 汇总表 ---
    rows = []
    for g in ("Low", "Mid", "High"):
        s = ic_df[g].dropna()
        icir = s.mean() / s.std() * (250 ** 0.5) if s.std() > 0 else np.nan
        rows.append({
            "group": g,
            "ic_mean": round(float(s.mean()), 5),
            "ic_std": round(float(s.std()), 5),
            "icir_annu": round(float(icir), 2),
            "pct_days_ic_neg": round(float((s < 0).mean()), 3),
            "n_days": int(len(s)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))

    # --- 图1：每日 IC 叠加 ---
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = {"Low": "#1f77b4", "Mid": "#7f7f7f", "High": "#d62728"}
    for g in ("Low", "Mid", "High"):
        ax.plot(ic_df.index, ic_df[g], label=g, color=colors[g], linewidth=0.6, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.set_title(f"{factor_name} — Daily RankIC by turnover tercile (raw)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_series.png", dpi=120)
    plt.close(fig)

    # --- 图2：滚动 20 日均值 ---
    roll = ic_df.rolling(20, min_periods=10).mean()
    fig, ax = plt.subplots(figsize=(16, 6))
    for g in ("Low", "Mid", "High"):
        ax.plot(roll.index, roll[g], label=g, color=colors[g], linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.set_title(f"{factor_name} — RankIC rolling 20d mean by turnover tercile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_rolling.png", dpi=120)
    plt.close(fig)

    # --- 图3：盒图 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [ic_df[g].dropna().values for g in ("Low", "Mid", "High")]
    bp = ax.boxplot(data, labels=["Low TO", "Mid TO", "High TO"], showmeans=True, patch_artist=True)
    for patch, g in zip(bp["boxes"], ("Low", "Mid", "High")):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{factor_name} — RankIC distribution by turnover tercile")
    ax.set_ylabel("Daily RankIC")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_boxplot.png", dpi=120)
    plt.close(fig)

    print(f"\n输出目录: {out_dir}/")
    low, high = summary.iloc[0], summary.iloc[2]
    if low["ic_mean"] < high["ic_mean"] and abs(low["icir_annu"]) > abs(high["icir_annu"]):
        print("=> 低换手组 IC 更强：验证【流动性条件 Alpha】假说 ✅")
    elif abs(low["ic_mean"] - high["ic_mean"]) < 0.005:
        print("=> 三组差异不显著：流动性可能不是条件变量")
    else:
        print("=> 高换手组更强：因子更像流动性择时信号，定位需调整")


if __name__ == "__main__":
    main()
