#!/usr/bin/env python3
"""Signal and unchanged-evaluation-layer parity for minute_skew."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTOR = "minute_skew"
FLAG = "INTRADAY_MINUTE_SKEW_USE_DDB"
PARITY_ROOT = ROOT / "result" / "intraday" / "_parity_minute_skew"
SAMPLE_START = "2024-05-01"
SAMPLE_END = "2024-05-31"


def _build_signal(use_ddb: bool, start: str, end: str):
    import factor_config as cfg
    from intraday_formulas import build_intraday_narrow_table
    from minute_bar_store import get_default_store

    store = get_default_store(start_date=start)
    with unittest.mock.patch.object(cfg, FLAG, use_ddb):
        return build_intraday_narrow_table(FACTOR, start, end, store=store)


def run_signal_parity(start: str, end: str):
    from factors.intraday.minute_skew.test_compare import assert_consistency

    py = _build_signal(False, start, end)
    db = _build_signal(True, start, end)
    metrics = assert_consistency(
        py.rename(columns={"tradetime": "bartime"}),
        db.rename(columns={"tradetime": "bartime"}),
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def run_backtest_parity(start: str, end: str):
    from factors.intraday.volume_front_loading import run_backtest_parity as harness

    harness.FACTOR = FACTOR
    harness.FLAG = FLAG
    harness.PARITY_ROOT = PARITY_ROOT
    py_dir = harness._run_backtest_once("python", False, eval_start=start, eval_end=end)
    db_dir = harness._run_backtest_once("ddb", True, eval_start=start, eval_end=end)
    return harness.compare_backtests(py_dir, db_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--end", default=SAMPLE_END)
    args = parser.parse_args()
    if not any((args.signal, args.backtest, args.all)):
        parser.print_help()
        return 0
    if os.environ.get("RUN_DDB_TESTS") != "1":
        raise RuntimeError("Set RUN_DDB_TESTS=1 for live DDB validation")
    if args.signal or args.all:
        run_signal_parity(args.start, args.end)
    if args.backtest or args.all:
        run_backtest_parity(args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
