#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案二：两步顺序中性化测试。

第一步：panel_neutral_size_ind（默认 ind_cap，市值+行业）
第二步：neutralize_again 继续剥离额外风格因子（动量 / 波动率 / 换手率，可组合）

额外风格因子全部由 WIND 日频数据现算：
- momentum：20 日累计收益（原始 c2c，未做超额）
- volatility：20 日收益标准差
- turnover：20 日平均换手率（S_DQ_TURN，取 log）

用法:
  python l2_factor_reproduction/scripts/test_double_neutralization.py --factor mid_order_ratio
  python l2_factor_reproduction/scripts/test_double_neutralization.py --factor mid_order_ratio \
      --first ind --extras momentum turnover
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from COMMON_CONST import DATA_DB_CONN  # noqa: E402
from Factor_Dev_Lib import (  # noqa: E402
    get_EOD_Not_Limit,
    get_EOD_Not_ST,
    get_Ret_Matrix,
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
from l2_factor_reproduction.python.neutralization import neutralize_again  # noqa: E402

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

WINDOW = 20


def _get_turnover_wide(start, end) -> pd.DataFrame:
    """WIND 日换手率宽表（%），取 20 日均值的 log。"""
    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    start_str = start.strftime("%Y.%m.%d")
    end_str = end.strftime("%Y.%m.%d")
    df = s.run(
        f"""
        t = select S_INFO_WINDCODE, TRADE_DT, S_DQ_TURN
            from loadTable('dfs://WIND.ASHAREEODDERIVATIVEINDICATOR', 'data')
            where TRADE_DT >= {start_str} and TRADE_DT <= {end_str}
            context by TRADE_DT, S_INFO_WINDCODE csort OPDATE limit 1
        select * from t
        """
    )
    s.close()
    wide = df.pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values="S_DQ_TURN")
    wide.index = pd.to_datetime(wide.index)
    return np.log(wide.sort_index().rolling(WINDOW, min_periods=10).mean())


def build_extra_factors(names, ref_index, ref_columns) -> dict:
    """按名称现算额外风格宽表，并对齐到因子宽表的 index/columns。"""
    extras = {}
    need_ret = any(n in ("momentum", "volatility") for n in names)
    ret_raw = None
    if need_ret:
        ret_raw = get_Ret_Matrix(START_DAY, END_DAY, method="c2c")  # 原始收益（非超额）
    for n in names:
        if n == "momentum":
            w = ret_raw.rolling(WINDOW, min_periods=10).sum()
        elif n == "volatility":
            w = ret_raw.rolling(WINDOW, min_periods=10).std()
        elif n == "turnover":
            w = _get_turnover_wide(START_DAY, END_DAY)
        else:
            raise ValueError(f"未知额外因子: {n}（可选 momentum/volatility/turnover）")
        extras[n] = w.reindex(index=ref_index, columns=ref_columns)
    return extras


def _run_leg(tag, factor_wide, mask):
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
    parser.add_argument("--factor", default="mid_order_ratio")
    parser.add_argument("--first", default="ind_cap", choices=["ind_cap", "ind", "cap"],
                        help="第一步中性化口径")
    parser.add_argument("--extras", nargs="+", default=["momentum", "volatility", "turnover"],
                        choices=["momentum", "volatility", "turnover"],
                        help="第二步继续剥离的风格因子")
    args = parser.parse_args()
    factor_name = args.factor

    result_root = Path(RESULT_ROOT)
    narrow_path = result_root / factor_name / "factor_narrow.parquet"
    if not narrow_path.exists():
        raise FileNotFoundError(f"窄表不存在: {narrow_path}")

    print(f"因子: {factor_name} | 基准: {UNIVERSE} | 区间: {START_DAY.date()} ~ {END_DAY.date()}")
    print(f"第一步: {args.first} | 第二步剥离: {args.extras}")

    narrow = pd.read_parquet(narrow_path)
    factor_raw = narrow_to_wide(narrow).loc[START_DAY:END_DAY]

    not_limit = get_EOD_Not_Limit(START_DAY, END_DAY)
    not_st = get_EOD_Not_ST(START_DAY, END_DAY)
    trade_status = get_TradeStatus(START_DAY, END_DAY)
    mask = not_limit * not_st * trade_status

    legs = {}

    print("\n--- [原始因子] 回测 ---")
    legs["raw"] = ("原始因子", _run_leg("raw", factor_raw, mask))

    print(f"\n--- [第一步: {args.first}] panel_neutral_size_ind ---")
    factor_s1 = panel_neutral_size_ind(
        signal=factor_raw, del_limit=False, del_st=False, nt_type=args.first
    ).astype(float)
    legs["step1"] = (f"一步({args.first})", _run_leg(f"step1_{args.first}", factor_s1, mask))

    print(f"\n--- [第二步] 继续剥离: {args.extras} ---")
    extras = build_extra_factors(args.extras, factor_s1.index, factor_s1.columns)
    factor_s2 = neutralize_again(factor_s1, extras)
    tag2 = f"step2_{args.first}+{'+'.join(args.extras)}"
    legs["step2"] = (f"二步({args.first}+{','.join(args.extras)})", _run_leg(tag2, factor_s2, mask))

    rows = []
    for key, label, fmt in CMP_KEYS:
        row = {"指标": label}
        for _, (leg_label, leg) in legs.items():
            row[leg_label] = fmt.format(leg["summary"][key])
        rows.append(row)
    cmp_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("【两步顺序中性化对比】")
    print(cmp_df.to_string(index=False))
    print("=" * 80)

    out_dir = result_root / factor_name / f"double_neutralized_{args.first}"
    _dump_leg(out_dir, f"{factor_name}_{tag2}", legs["step2"][1], factor_s2)
    cmp_df.to_csv(out_dir / "double_neutralization_comparison.csv", index=False)
    print(f"\n二步中性化输出: {out_dir}/")
    print(f"对比表: {out_dir / 'double_neutralization_comparison.csv'}")


if __name__ == "__main__":
    main()
