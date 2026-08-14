#!/usr/bin/env python
"""Prefetch active_pressure bricks, then run Factor_Test_Process for APM_ActiveV2.

Usage:
  OMP_NUM_THREADS=1 python run_apm_active_v2_backtest.py
  OMP_NUM_THREADS=1 python run_apm_active_v2_backtest.py --prefetch-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
from core.l2_features.bricks.active_pressure import ensure_active_pressure_daily_bricks

REPO = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    start = cfg.START_DAY - dt.timedelta(days=40)
    end = cfg.END_DAY
    print(
        f"=== Prefetch active_pressure bricks {start.date()}..{end.date()} "
        f"(track={cfg.TRACK}) ===",
        flush=True,
    )
    daily = ensure_active_pressure_daily_bricks(
        start, end, use_cache=True, refresh_cache=args.refresh_cache
    )
    print(
        f"daily brick rows={len(daily):,} symbols={daily['symbol'].nunique()} "
        f"days={daily['date'].nunique()}",
        flush=True,
    )
    if args.prefetch_only:
        return

    print("=== Launch Factor_Test_Process.py ===", flush=True)
    sys.path.insert(0, str(REPO))
    runpy.run_path(str(REPO / "Factor_Test_Process.py"), run_name="__main__")


if __name__ == "__main__":
    main()
