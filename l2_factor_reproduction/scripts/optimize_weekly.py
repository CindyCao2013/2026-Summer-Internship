#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务2：一键测试周度调仓对因子净收益的改善效果。

口径（与日频框架一致，便于直接对比）：
- 日频信号（全日聚合 -> shift(1)）在每周第一个交易日采样，周内 ffill 持仓
- 收益仍用日频 c2c 超额（vs UNIVERSE），年化基数 250
- 换手自然集中在调仓日，日均 H-L 换手 * 7.5bps * 250 即年化成本

用法:
  python l2_factor_reproduction/scripts/optimize_weekly.py [--factor avg_outflow_ratio] [--raw]
  默认使用中性化因子（factor_neutralized.parquet）；--raw 使用原始窄表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    get_TradeStatus,
    groupTest,
    implied_annu_fee,
)
from l2_factor_reproduction.config.settings import (  # noqa: E402
    END_DAY,
    RESULT_ROOT,
    START_DAY,
    UNIVERSE,
)
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    compute_rank_ic,
    narrow_to_wide,
    prepare_factor_signal,
    save_group_plots,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="avg_outflow_ratio")
    parser.add_argument("--raw", action="store_true", help="使用原始因子（默认用中性化因子）")
    args = parser.parse_args()
    factor_name = args.factor
    use_neutralized = not args.raw

    result_root = Path(RESULT_ROOT)
    fname = "factor_neutralized.parquet" if use_neutralized else "factor_narrow.parquet"
    narrow_path = result_root / factor_name / fname
    if not narrow_path.exists():
        raise FileNotFoundError(f"因子窄表不存在: {narrow_path}（先跑 test_neutralization.py？）")

    variant = "neutralized" if use_neutralized else "raw"
    print(f"因子: {factor_name} ({variant}) | 基准: {UNIVERSE} | "
          f"区间: {START_DAY.date()} ~ {END_DAY.date()} | 调仓: 周度")

    narrow = pd.read_parquet(narrow_path)
    factor_wide = narrow_to_wide(narrow).loc[START_DAY:END_DAY]

    not_limit = get_EOD_Not_Limit(START_DAY, END_DAY)
    not_st = get_EOD_Not_ST(START_DAY, END_DAY)
    trade_status = get_TradeStatus(START_DAY, END_DAY)
    mask = not_limit * not_st * trade_status

    # 日频信号（已 shift(1)）
    signal_daily, ret = prepare_factor_signal(
        factor_wide, start=START_DAY, end=END_DAY, mask=mask, signal_shift=1
    )

    # 每周第一个交易日为调仓日；周内持仓 ffill
    idx = signal_daily.index
    week_key = idx.to_period("W")
    is_rebalance = pd.Series(week_key, index=idx).ne(
        pd.Series(week_key, index=idx).shift(1)
    )
    rebalance_days = idx[is_rebalance.values]
    signal_weekly = signal_daily.where(
        pd.Series(idx.isin(rebalance_days), index=idx), other=np.nan
    ).ffill(limit=10)
    # 调仓日之前无信号的行仍为 NaN，会被 groupTest 忽略

    print(f"调仓日数量: {len(rebalance_days)} / 交易日 {len(idx)}")

    _, group_pnl, group_to = groupTest(signal_weekly, ret, n=10, info="silent")
    rank_ic = compute_rank_ic(signal_weekly, ret)

    hl = group_pnl["H-L"]
    direction = 1 if hl.mean() > 0 else -1
    hl_adj = hl * direction

    annu_ret = float(calAnnuRet(hl_adj))            # 日频序列，年化 250
    sharpe = float(calSharpe(hl_adj))
    mdd, _ = calMDD(hl_adj)
    avg_to = float(group_to["H-L"].mean())          # 日均（调仓日集中）
    cost_annual = float(implied_annu_fee(avg_to))   # TO * 7.5bps * 250
    net_annu = annu_ret - cost_annual
    ic_mean = float(rank_ic.mean())
    ic_std = float(rank_ic.std())
    icir = ic_mean / ic_std * (250 ** 0.5) if ic_std > 0 else float("nan")

    print("\n" + "=" * 60)
    print(f"【周度调仓结果】{factor_name} ({variant})")
    print(f"方向翻转: {'是(-H-L)' if direction < 0 else '否'}")
    print(f"RankIC(日频): {ic_mean:.4f} | ICIR: {icir:.2f}")
    print(f"毛年化收益: {annu_ret:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {float(mdd):.2%}")
    print(f"日均换手倍数: {avg_to:.2f} (≈周均 {avg_to * 5:.2f})")
    print(f"年化成本(7.5bps): {cost_annual:.2%}")
    print(f"净年化收益(扣费后): {net_annu:.2%}")
    print("=" * 60)

    # 落盘
    out_dir = result_root / factor_name / f"weekly_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    group_pnl.to_csv(out_dir / "group_pnl.csv")
    group_to.to_csv(out_dir / "group_turnover.csv")
    summary = {
        "variant": f"weekly_{variant}",
        "n_rebalance_days": int(len(rebalance_days)),
        "hl_direction_flip": int(direction),
        "rank_ic_mean": ic_mean,
        "rank_icir": float(icir),
        "hl_annu_ret_flipped": annu_ret,
        "hl_sharpe_flipped": sharpe,
        "hl_mdd_flipped": float(mdd),
        "avg_hl_turnover": avg_to,
        "implied_annu_fee": cost_annual,
        "net_annu_after_fee": net_annu,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    save_group_plots(
        str(out_dir), f"{factor_name}_weekly_{variant}", group_pnl,
        group_to=group_to, rank_ic=rank_ic, direction=direction,
    )
    print(f"输出目录: {out_dir}/")


if __name__ == "__main__":
    main()
