#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试单个因子。

用法:
  python l2_factor_reproduction/scripts/run_single_factor.py --factor avg_outflow_ratio
  python l2_factor_reproduction/scripts/run_single_factor.py --factor avg_outflow_ratio \\
      --start 2024-01-02 --end 2024-03-29 --no-backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.factor_builder import FACTOR_BUILDERS
from l2_factor_reproduction.python.factor_runner import run_single_factor
from l2_factor_reproduction.python.utils.date_utils import to_datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single L2 reproduction factor")
    parser.add_argument("--factor", required=True, help="因子名称，见 FACTOR_BUILDERS")
    parser.add_argument("--start", default=None, help="覆盖开始日 YYYY-MM-DD / YYYYMMDD")
    parser.add_argument("--end", default=None, help="覆盖结束日 YYYY-MM-DD / YYYYMMDD")
    parser.add_argument("--universe", default=None, help="覆盖股票池，如 000852.SH")
    parser.add_argument("--no-backtest", action="store_true", help="只算因子不回测")
    parser.add_argument("--no-save", action="store_true", help="不落盘 parquet")
    args = parser.parse_args()

    if args.factor not in FACTOR_BUILDERS:
        print(f"因子 {args.factor} 未注册。可选: {sorted(FACTOR_BUILDERS)}")
        sys.exit(1)

    start_day = to_datetime(args.start) if args.start else None
    end_day = to_datetime(args.end) if args.end else None

    df = run_single_factor(
        args.factor,
        save=not args.no_save,
        start_day=start_day,
        end_day=end_day,
        universe=args.universe,
    )
    if df is None or df.empty:
        print("因子返回空")
        sys.exit(2)

    print(f"因子数据行数: {len(df)}")
    print(df.head())

    if args.no_backtest:
        return

    group_pnl, group_to, rank_ic, summary = backtest_factor(
        df,
        start_day=start_day,
        end_day=end_day,
        universe=args.universe,
    )
    from l2_factor_reproduction.config.settings import RESULT_ROOT
    from l2_factor_reproduction.python.backtest import _save_backtest_outputs
    import os

    out_dir = os.path.join(RESULT_ROOT, args.factor)
    _save_backtest_outputs(out_dir, group_pnl, group_to, rank_ic, summary, factor_name=args.factor)
    print("回测完成，分组收益前5行:")
    print(group_pnl.head())
    print(
        f"RankIC(有效方向)={summary['rank_ic_mean']:.4f} ICIR={summary['rank_icir']:.2f} | "
        f"H-L年化={summary['hl_annu_ret_flipped']:.2%} Sharpe={summary['hl_sharpe_flipped']:.2f} | "
        f"factor_direction={summary.get('factor_direction', 1)}（-1 表示生产应取 -factor）"
    )
    print("日均换手:", float(group_to["H-L"].mean()) if "H-L" in group_to.columns else "N/A")
    print("RankIC 前5:\n", rank_ic.head())
    print(f"图表已保存: {out_dir}/cum_pnl.png , {out_dir}/decile_bar.png")


if __name__ == "__main__":
    main()
