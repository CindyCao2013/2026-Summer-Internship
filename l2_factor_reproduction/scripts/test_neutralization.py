#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务1：一键测试因子中性化（市值/行业，多口径可选）的效果。

对比原始与各中性化口径（ind_cap / ind / cap）的 Rank IC / 毛年化 / 夏普 / 换手 / 扣费后净年化。

口径与 Phase 1 基线完全一致（复用 backtest.prepare_factor_signal）：
- 因子为当日全日聚合 -> signal.shift(1) 后对当日 c2c 超额收益（vs UNIVERSE）
- 过滤涨跌停 / ST / 停牌
- H-L 若均值为负则翻向展示（direction_flip=-1）

用法:
  python l2_factor_reproduction/scripts/test_neutralization.py --factor avg_outflow_ratio
  python l2_factor_reproduction/scripts/test_neutralization.py --factor mid_order_ratio \
      --neutral_types ind_cap ind cap
"""

from __future__ import annotations

import argparse
import json
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
    save_group_plots,
    summarize_backtest,
)

CMP_KEYS = [
    ("rank_ic_mean", "RankIC", "{:.4f}"),
    ("rank_icir", "ICIR", "{:.2f}"),
    ("hl_annu_ret_flipped", "毛年化(翻向)", "{:.2%}"),
    ("hl_sharpe_flipped", "夏普(翻向)", "{:.2f}"),
    ("hl_mdd_flipped", "MDD(翻向)", "{:.2%}"),
    ("avg_hl_turnover", "日均换手倍数", "{:.2f}"),
    ("implied_annu_fee", "年化成本(7.5bps)", "{:.2%}"),
    ("net_annu_after_fee", "净年化(扣费)", "{:.2%}"),
]


def _run_leg(tag: str, factor_wide: pd.DataFrame, mask: pd.DataFrame) -> dict:
    """一条腿：mask -> shift(1) -> groupTest -> 汇总（与 Phase 1 同口径）。"""
    signal, ret = prepare_factor_signal(
        factor_wide, start=START_DAY, end=END_DAY, mask=mask, signal_shift=1
    )
    rank_ic = compute_rank_ic(signal, ret)
    _, group_pnl, group_to = groupTest(signal, ret, n=10, info="silent")
    summary = summarize_backtest(signal, ret, group_pnl, group_to, rank_ic)
    summary["net_annu_after_fee"] = summary["hl_annu_ret_flipped"] - summary["implied_annu_fee"]
    print(
        f"[{tag}] RankIC={summary['rank_ic_mean']:.4f} ICIR={summary['rank_icir']:.2f} | "
        f"毛年化(翻向)={summary['hl_annu_ret_flipped']:.2%} Sharpe={summary['hl_sharpe_flipped']:.2f} "
        f"MDD={summary['hl_mdd_flipped']:.2%} | 日均换手={summary['avg_hl_turnover']:.2f} "
        f"年化成本={summary['implied_annu_fee']:.2%} 净年化={summary['net_annu_after_fee']:.2%}"
    )
    return {"summary": summary, "group_pnl": group_pnl, "group_to": group_to, "rank_ic": rank_ic}


def _dump_leg(out_dir: Path, name: str, leg: dict, factor_wide: pd.DataFrame) -> None:
    """落盘一条腿：窄表 + group_pnl/summary/三张图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = factor_wide.stack().dropna().reset_index()
    stack.columns = ["tradetime", "symbol", "value"]
    stack["factorname"] = name
    stack.to_parquet(out_dir / "factor_narrow.parquet")

    leg["group_pnl"].to_csv(out_dir / "group_pnl.csv")
    leg["group_to"].to_csv(out_dir / "group_turnover.csv")
    leg["rank_ic"].to_frame("rank_ic").to_csv(out_dir / "rank_ic.csv")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(leg["summary"], fh, ensure_ascii=False, indent=2, default=str)
    save_group_plots(
        str(out_dir), name, leg["group_pnl"],
        group_to=leg["group_to"], rank_ic=leg["rank_ic"],
        direction=int(leg["summary"]["hl_direction_flip"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="avg_outflow_ratio")
    parser.add_argument(
        "--neutral_types", nargs="+", default=["ind_cap"],
        choices=["ind_cap", "ind", "cap"],
        help="中性化口径，可多选：ind_cap=市值+行业, ind=仅行业, cap=仅市值",
    )
    args = parser.parse_args()
    factor_name = args.factor

    result_root = Path(RESULT_ROOT)
    narrow_path = result_root / factor_name / "factor_narrow.parquet"
    if not narrow_path.exists():
        raise FileNotFoundError(f"窄表不存在: {narrow_path}")

    print(f"因子: {factor_name} | 基准: {UNIVERSE} | 区间: {START_DAY.date()} ~ {END_DAY.date()}")

    narrow = pd.read_parquet(narrow_path)
    factor_raw = narrow_to_wide(narrow).loc[START_DAY:END_DAY]

    not_limit = get_EOD_Not_Limit(START_DAY, END_DAY)
    not_st = get_EOD_Not_ST(START_DAY, END_DAY)
    trade_status = get_TradeStatus(START_DAY, END_DAY)
    mask = not_limit * not_st * trade_status

    print("\n--- [原始因子] 回测 ---")
    legs = {"raw": ("原始因子", _run_leg("raw", factor_raw, mask))}

    for nt in args.neutral_types:
        print(f"\n--- [中性化: {nt}] panel_neutral_size_ind 进行中 ---")
        factor_neu = panel_neutral_size_ind(
            signal=factor_raw, del_limit=False, del_st=False, nt_type=nt
        ).astype(float)
        legs[nt] = (f"中性化({nt})", _run_leg(f"neutralized_{nt}", factor_neu, mask))
        _dump_leg(
            result_root / factor_name / f"neutralized_{nt}",
            f"{factor_name}_neutralized_{nt}", legs[nt][1], factor_neu,
        )
        # 兼容旧路径：ind_cap 结果同时写 factor_neutralized.parquet / neutralized/
        if nt == "ind_cap":
            legacy = result_root / factor_name
            legacy_stack = factor_neu.stack().dropna().reset_index()
            legacy_stack.columns = ["tradetime", "symbol", "value"]
            legacy_stack["factorname"] = factor_name + "_neutralized"
            legacy_stack.to_parquet(legacy / "factor_neutralized.parquet")
            _dump_leg(legacy / "neutralized", factor_name + "_neutralized", legs[nt][1], factor_neu)

    # --- 汇总对比 ---
    rows = []
    for key, label, fmt in CMP_KEYS:
        row = {"指标": label}
        for leg_key, (leg_label, leg) in legs.items():
            row[leg_label] = fmt.format(leg["summary"][key])
        rows.append(row)
    cmp_df = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    print("【中性化口径对比】")
    print(cmp_df.to_string(index=False))
    print("=" * 76)

    cmp_df.to_csv(result_root / factor_name / "neutralization_comparison.csv", index=False)
    print(f"\n对比表: {result_root / factor_name / 'neutralization_comparison.csv'}")
    print("各口径输出目录: " + ", ".join(
        str(result_root / factor_name / f"neutralized_{nt}") for nt in args.neutral_types
    ))


if __name__ == "__main__":
    main()
