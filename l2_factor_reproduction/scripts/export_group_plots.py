#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从已有 group_pnl.csv 重新生成标准图（有效方向）：

- cum_pnl.png：G1..G10 + H-L 累计（高因子组在上，H-L 上行）
- decile_bar.png：单调性柱状图

方向判定：读取同目录 summary.json——
- 含 ``group_pnl_saved_direction == "effective"``（新管线）：CSV 已是有效方向
- 否则（旧 CSV 为原始方向）：用 ``hl_direction_flip``（-1 表示需整体倒转展示）

用法:
  python l2_factor_reproduction/scripts/export_group_plots.py
  python l2_factor_reproduction/scripts/export_group_plots.py --factor mid_order_ratio
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

from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.python.backtest import save_group_plots


def _direction_from_summary(d: Path) -> int:
    sj = d / "summary.json"
    if sj.exists():
        try:
            with open(sj, encoding="utf-8") as fh:
                s = json.load(fh)
            if s.get("group_pnl_saved_direction") == "effective":
                return 1
            return int(s.get("hl_direction_flip", s.get("factor_direction", 1)))
        except Exception:  # noqa: BLE001
            pass
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default=None, help="只画某一个因子；默认扫 RESULT_ROOT")
    args = parser.parse_args()

    root = Path(RESULT_ROOT)
    if args.factor:
        dirs = sorted(root.glob(f"{args.factor}/**/")) + [root / args.factor]
    else:
        dirs = sorted(p for p in root.rglob("") if p.is_dir())

    for d in dirs:
        pnl_path = d / "group_pnl.csv"
        if not pnl_path.exists():
            continue
        pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
        to = None
        ic = None
        if (d / "group_turnover.csv").exists():
            to = pd.read_csv(d / "group_turnover.csv", index_col=0, parse_dates=True)
        if (d / "rank_ic.csv").exists():
            ic = pd.read_csv(d / "rank_ic.csv", index_col=0, parse_dates=True).iloc[:, 0]
        direction = _direction_from_summary(d)
        cum, bar = save_group_plots(
            str(d), d.name, pnl, group_to=to, rank_ic=ic, direction=direction
        )
        # 旧版翻转图已废弃，清理避免误导
        stale = d / "cum_hl_flipped.png"
        if stale.exists():
            stale.unlink()
        print(f"{d.relative_to(root)}: direction={direction} -> {Path(cum).name} | {Path(bar).name}")


if __name__ == "__main__":
    main()
