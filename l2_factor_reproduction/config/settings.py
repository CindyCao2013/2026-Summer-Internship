"""项目专属配置：日期、标的池、因子清单、路径。

复用全局 ``factor_config.START_DAY / END_DAY``，与主研究管线保持一致。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 仓库根目录（factor_dev），用于导入 Factor_Dev_Lib / intraday_lib 等
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from factor_config import END_DAY, PREHEAT_CALENDAR_DAYS, START_DAY  # noqa: E402

# 回测区间（与全局保持一致；可在本文件覆盖）
START_DAY = START_DAY
END_DAY = END_DAY
PREHEAT_CALENDAR_DAYS = PREHEAT_CALENDAR_DAYS

# 分钟数据预热日历天数（滚动窗口 / 日收益差分需要）
PREHEAT_DAYS = 40

# 标的池：中证1000；可改为 '000300.SH' / '000905.SH'
UNIVERSE = "000852.SH"

# 分组回测参数
N_GROUPS = 10
BACKTEST_SILENT = True  # groupTest(info='silent')，批量跑时不弹图

# 本次要计算的因子清单
FACTOR_LIST = [
    "mid_order_ratio",
]

# Phase 2 占位（逐笔成交）
PHASE2_FACTOR_LIST = [
    "mid_order_ratio",
    "small_order_ratio",
]

# 绝对路径：DDB run() 必须能解析
L2_ROOT = Path(__file__).resolve().parents[1]
DDB_SCRIPT_ROOT = str(L2_ROOT / "ddb_scripts")

# 结果统一落到 research/results，便于与其他研究对齐
RESULT_ROOT = str(PROJ_ROOT / "research" / "results" / "l2_reproduction")

# 大单阈值（分钟内按 AvgOrderAmt 降序取前 frac）
BIG_ORDER_TOP_FRAC = 0.2
BIG_ORDER_DRIVE_TOP_FRAC = 0.3

# 盘口因子默认使用的本地快照库（由定时任务从 CH 同步）
ORDER_BOOK_DB = "dfs://L2_Snapshot_Daily"
ORDER_BOOK_TABLE = "snapshot"

STREAMING_MODE = False
LOG_LEVEL = os.environ.get("L2_REPRO_LOG_LEVEL", "INFO")
