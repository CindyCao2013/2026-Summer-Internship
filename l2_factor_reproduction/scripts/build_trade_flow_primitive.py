#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sprint 1 / 阶段1：构建 l2_primitive_trade_flow_daily 原语层（不做回测）。

季度分块 CH 聚合 + parquet 缓存（可断点续跑），输出：
- chunks/chunk_*.parquet：每季度一块
- trade_flow_daily_<start>_<end>.parquet：合并全量
- coverage_report.csv：每块 rows / 股票数 / 日期范围 / 方向字段完整性

用法:
  python l2_factor_reproduction/scripts/build_trade_flow_primitive.py
  python l2_factor_reproduction/scripts/build_trade_flow_primitive.py \
      --start 2019-01-01 --end 2026-07-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.ch_tick import fetch_trade_flow_daily  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-07-31")
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    out_dir = Path(RESULT_ROOT) / "primitives" / "trade_flow_daily"
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    bounds = pd.date_range(start, end, freq="QS")
    edges = [start] + [b for b in bounds if start < b <= end] + [end + pd.Timedelta(days=1)]

    parts = []
    coverage = []
    for c_start, c_end_ex in zip(edges[:-1], edges[1:]):
        c_end = c_end_ex - pd.Timedelta(days=1)
        cp = chunk_dir / f"chunk_{c_start.date()}_{c_end.date()}.parquet"
        if cp.exists():
            part = pd.read_parquet(cp)
            print(f"缓存: {c_start.date()} ~ {c_end.date()} rows={len(part)}", flush=True)
        else:
            print(f"CH 查询: {c_start.date()} ~ {c_end.date()} ...", flush=True)
            part = fetch_trade_flow_daily(c_start, c_end)
            part.to_parquet(cp)
            print(f"  rows={len(part)}", flush=True)
        parts.append(part)
        part["TradeDate"] = pd.to_datetime(part["TradeDate"])
        coverage.append({
            "chunk": f"{c_start.date()}~{c_end.date()}",
            "rows": len(part),
            "n_symbols": int(part["symbol"].nunique()),
            "date_min": str(part["TradeDate"].min().date()) if len(part) else None,
            "date_max": str(part["TradeDate"].max().date()) if len(part) else None,
            "buy_amt_share": round(
                float(part["active_buy_amt"].sum() / part["total_amt"].sum()), 4
            ) if len(part) and part["total_amt"].sum() > 0 else None,
            "sell_amt_share": round(
                float(part["active_sell_amt"].sum() / part["total_amt"].sum()), 4
            ) if len(part) and part["total_amt"].sum() > 0 else None,
        })

    df = pd.concat(parts, ignore_index=True)
    merged = out_dir / f"trade_flow_daily_{start.date()}_{end.date()}.parquet"
    df.to_parquet(merged)
    cov = pd.DataFrame(coverage)
    cov.to_csv(out_dir / "coverage_report.csv", index=False)

    print("\n=== coverage ===")
    print(cov.to_string(index=False))
    print(f"\n合并: {merged} rows={len(df)}")
    print(f"覆盖: {df['TradeDate'].min().date()} ~ {df['TradeDate'].max().date()}, "
          f"symbols={df['symbol'].nunique()}")
    buy = df["active_buy_amt"].sum() / df["total_amt"].sum()
    sell = df["active_sell_amt"].sum() / df["total_amt"].sum()
    print(f"方向金额占比: buy={buy:.2%} sell={sell:.2%} (合计应≈100%, 差=SSE中性单)")


if __name__ == "__main__":
    main()
