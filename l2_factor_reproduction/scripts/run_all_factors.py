#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键计算全部配置因子并回测。

用法:
  python l2_factor_reproduction/scripts/run_all_factors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.backtest import backtest_all_factors
from l2_factor_reproduction.python.factor_runner import run_all_factors


def main() -> None:
    print("开始计算所有因子...")
    results = run_all_factors()
    print(f"因子计算完成: {list(results.keys())}")
    print("开始回测...")
    backtest_all_factors()
    print("全部完成！结果目录: research/results/l2_reproduction/")


if __name__ == "__main__":
    main()
