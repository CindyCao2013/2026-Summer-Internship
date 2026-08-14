#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""二次中性化变量的前向贪心筛选。

规则（与约定一致）：
1. 基线 = 第一步中性化（默认 ind_cap）的 |ICIR|
2. 逐个加入候选风格变量（动量/波动率/换手率），|ICIR| 变大 -> 保留，变小 -> 剔除
3. 在已保留集合上继续尝试加入剩余变量，直到无提升为止
4. 打印完整对比表（基线 / 单变量 / 贪心路径 / 终选组合）

用法:
  python l2_factor_reproduction/scripts/screen_second_pass.py --factor mid_order_ratio
  python l2_factor_reproduction/scripts/screen_second_pass.py --factor mid_order_ratio --first ind
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import (  # noqa: E402
    get_EOD_Not_Limit,
    get_EOD_Not_ST,
    get_TradeStatus,
    groupTest,
    panel_neutral_size_ind,
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
    summarize_backtest,
)
from l2_factor_reproduction.python.neutralization import neutralize_again  # noqa: E402
from l2_factor_reproduction.scripts.test_double_neutralization import (  # noqa: E402
    build_extra_factors,
)

CANDIDATES = ["momentum", "volatility", "turnover"]


def _quick_summary(factor_wide, mask):
    """快速回测，只取关键指标。"""
    signal, ret = prepare_factor_signal(
        factor_wide, start=START_DAY, end=END_DAY, mask=mask, signal_shift=1
    )
    rank_ic = compute_rank_ic(signal, ret)
    _, group_pnl, group_to = groupTest(signal, ret, n=10, info="silent")
    s = summarize_backtest(signal, ret, group_pnl, group_to, rank_ic)
    s["net_annu_after_fee"] = s["hl_annu_ret_flipped"] - s["implied_annu_fee"]
    return s


def _row(tag, s):
    return {
        "组合": tag,
        "RankIC": f"{s['rank_ic_mean']:.4f}",
        "ICIR": f"{s['rank_icir']:.2f}",
        "|ICIR|": round(abs(s["rank_icir"]), 4),
        "夏普": f"{s['hl_sharpe_flipped']:.2f}",
        "毛年化": f"{s['hl_annu_ret_flipped']:.2%}",
        "MDD": f"{s['hl_mdd_flipped']:.2%}",
        "净年化": f"{s['net_annu_after_fee']:.2%}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="mid_order_ratio")
    parser.add_argument("--first", default="ind_cap", choices=["ind_cap", "ind", "cap"])
    args = parser.parse_args()
    factor_name = args.factor

    result_root = Path(RESULT_ROOT)
    narrow = pd.read_parquet(result_root / factor_name / "factor_narrow.parquet")
    factor_raw = narrow_to_wide(narrow).loc[START_DAY:END_DAY]

    mask = (
        get_EOD_Not_Limit(START_DAY, END_DAY)
        * get_EOD_Not_ST(START_DAY, END_DAY)
        * get_TradeStatus(START_DAY, END_DAY)
    )

    print(f"因子: {factor_name} | 第一步: {args.first} | 候选变量: {CANDIDATES}")

    print("\n--- 第一步中性化（基线）---")
    factor_s1 = panel_neutral_size_ind(
        signal=factor_raw, del_limit=False, del_st=False, nt_type=args.first
    ).astype(float)

    print("--- 构建候选风格因子（动量/波动率/换手率）---")
    extras_all = build_extra_factors(CANDIDATES, factor_s1.index, factor_s1.columns)

    rows = []
    s_base = _quick_summary(factor_s1, mask)
    rows.append(_row(f"基线({args.first})", s_base))
    base_abs_icir = abs(s_base["rank_icir"])
    print(f"基线 |ICIR| = {base_abs_icir:.2f}")

    # --- 单变量测试 ---
    single_scores = {}
    for name in CANDIDATES:
        f2 = neutralize_again(factor_s1, {name: extras_all[name]})
        s = _quick_summary(f2, mask)
        rows.append(_row(f"+{name}", s))
        single_scores[name] = (abs(s["rank_icir"]), f2, s)
        verdict = "保留候选" if abs(s["rank_icir"]) > base_abs_icir else "剔除"
        print(f"+{name:<10} |ICIR| = {abs(s['rank_icir']):.2f} -> {verdict}")

    # --- 前向贪心：只从单变量已提升的集合里继续加 ---
    kept = [n for n in CANDIDATES if single_scores[n][0] > base_abs_icir]
    kept.sort(key=lambda n: single_scores[n][0], reverse=True)

    if not kept:
        print("\n没有任何单变量能提升 |ICIR|，终选 = 基线（不做二次中性化）")
        final_names, final_factor, final_s = [], factor_s1, s_base
        best_abs = base_abs_icir
    else:
        final_names = [kept[0]]
        final_factor = single_scores[kept[0]][1]
        final_s = single_scores[kept[0]][2]
        best_abs = single_scores[kept[0]][0]
        print(f"\n贪心起点: +{kept[0]} (|ICIR|={best_abs:.2f})")
        for name in kept[1:]:
            trial_names = final_names + [name]
            f_trial = neutralize_again(factor_s1, {n: extras_all[n] for n in trial_names})
            s_trial = _quick_summary(f_trial, mask)
            rows.append(_row("+" + "+".join(trial_names), s_trial))
            if abs(s_trial["rank_icir"]) > best_abs:
                final_names, final_factor, final_s = trial_names, f_trial, s_trial
                best_abs = abs(s_trial["rank_icir"])
                print(f"+{name:<10} |ICIR| = {abs(s_trial['rank_icir']):.2f} -> 保留")
            else:
                print(f"+{name:<10} |ICIR| = {abs(s_trial['rank_icir']):.2f} -> 剔除")

    cmp_df = pd.DataFrame(rows)
    print("\n" + "=" * 84)
    print("【二次中性化变量筛选】")
    print(cmp_df.drop(columns=["|ICIR|"]).to_string(index=False))
    print("=" * 84)
    final_tag = f"{args.first}" + ("+" + "+".join(final_names) if final_names else "（不二次中性化）")
    print(f"\n终选组合: {final_tag} | |ICIR| = {best_abs:.2f} "
          f"Sharpe = {final_s['hl_sharpe_flipped']:.2f} 净年化 = {final_s['net_annu_after_fee']:.2%}")

    out_csv = result_root / factor_name / f"second_pass_screen_{args.first}.csv"
    cmp_df.to_csv(out_csv, index=False)
    print(f"对比表: {out_csv}")


if __name__ == "__main__":
    main()
