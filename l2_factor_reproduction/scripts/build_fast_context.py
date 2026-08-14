#!/usr/bin/env python
"""构建 Fast Discovery 永久回测上下文缓存。

一次性从 DolphinDB 提取并落盘（之后 Fast Lane 不再访问 DDB）：

fast_context/<window>/
    ret_matrix.parquet       相对 UNIVERSE 的超额 c2c 收益宽表
    universe_mask.parquet    可投资 mask（非涨跌停 × 非ST × 交易状态）
    benchmark_return.parquet 基准指数自身 c2c 收益（参考列）
    trading_dates.parquet    交易日序列
    context_manifest.json    sha256 / 参数 / 生成时间 / 模块哈希

用法：

    python -m l2_factor_reproduction.scripts.build_fast_context \
        --windows discovery full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from COMMON_CONST import DATA_DB_CONN  # noqa: E402
from l2_factor_reproduction.config.settings import UNIVERSE  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    load_backtest_context,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    WINDOWS,
    _sha256,
    context_paths,
)


def _module_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_benchmark_return(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """基准指数自身 c2c 收益（小查询，仅用于参考列）。"""
    import dolphindb as ddb

    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    try:
        table = session.loadTable(
            dbPath="dfs://WIND.AINDEXEODPRICES", tableName="data"
        )
        frame = (
            table.where(
                f"TRADE_DT>= {start.strftime('%Y.%m.%d')} "
                f"and TRADE_DT <= {end.strftime('%Y.%m.%d')} "
                f"and S_INFO_WINDCODE = '{UNIVERSE}'"
            )
            .select(
                "TRADE_DT as Date, "
                "(S_DQ_CLOSE/S_DQ_PRECLOSE-1) as benchmark_ret"
            )
            .toDF()
        )
    finally:
        session.close()
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.set_index("Date")["benchmark_ret"].sort_index()


def build_window(window: str) -> None:
    start, end = WINDOWS[window]
    paths = context_paths(window)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)

    print(f"[context] {window}: {start.date()}..{end.date()} querying DDB")
    mask, ret = load_backtest_context(start, end)
    benchmark = _fetch_benchmark_return(start, end)
    dates = pd.DataFrame({"TradeDate": ret.index})

    mask.to_parquet(paths["universe_mask"])
    ret.to_parquet(paths["ret_matrix"])
    benchmark.to_frame().to_parquet(paths["benchmark_return"])
    dates.to_parquet(paths["trading_dates"])

    manifest = {
        "window": window,
        "start": str(start.date()),
        "end": str(end.date()),
        "universe": UNIVERSE,
        "method": "c2c benchmark-relative excess return",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "module_sha256": {
            "backtest": _module_sha256(
                PROJ_ROOT / "l2_factor_reproduction" / "python" / "backtest.py"
            ),
            "factor_dev_lib": _module_sha256(
                PROJ_ROOT / "Factor_Dev_Lib.py"
            ),
        },
        "shape": {
            "ret_matrix": list(ret.shape),
            "universe_mask": list(mask.shape),
        },
        "files": {
            key: {
                "path": str(paths[key].name),
                "sha256": _sha256(paths[key]),
            }
            for key in (
                "ret_matrix",
                "universe_mask",
                "benchmark_return",
                "trading_dates",
            )
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[context] {window}: ret{ret.shape} mask{mask.shape} -> "
        f"{paths['manifest'].parent}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        nargs="+",
        default=list(WINDOWS),
        choices=list(WINDOWS),
    )
    args = parser.parse_args()
    for window in args.windows:
        build_window(window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
