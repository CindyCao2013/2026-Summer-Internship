#!/usr/bin/env python3
"""Production, signal, and backtest parity for ``ofi_persistence``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unittest.mock
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTOR = "ofi_persistence"
FLAG = "INTRADAY_OFI_PERSISTENCE_USE_DDB"
PARITY_ROOT = ROOT / "result" / "intraday" / "_parity_ofi_persistence"
SAMPLE_START = "2024-05-01"
SAMPLE_END = "2024-05-31"


def trace_production_path() -> None:
    """Verify that configured production dispatch reaches shared DDB compute."""
    from intraday_formulas import build_intraday_narrow_table

    fake = pd.DataFrame(
        {
            "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
            "symbol": ["600000.SH"],
            "factorname": [FACTOR],
            "value": [0.6],
        }
    )
    with unittest.mock.patch(
        "factors.intraday.discovery_v1.ddb_version",
        return_value=fake,
    ) as mock_ddb:
        with unittest.mock.patch.dict(os.environ, {FLAG: "true"}):
            out = build_intraday_narrow_table(
                FACTOR, SAMPLE_START, SAMPLE_END, store=None
            )
    mock_ddb.assert_called_once()
    if mock_ddb.call_args.args[0] != FACTOR:
        raise AssertionError("Production dispatch selected the wrong factor")
    assert list(out.columns) == ["tradetime", "symbol", "factorname", "value"]
    print(
        "TRACE OK: build_intraday_narrow_table → "
        "compute_ofi_persistence → discovery_v1.ddb_version",
        flush=True,
    )


def _build_signal(use_ddb: bool, start: str, end: str) -> pd.DataFrame:
    from intraday_formulas import build_intraday_narrow_table
    from minute_bar_store import get_default_store

    store = get_default_store(start_date=start)
    with unittest.mock.patch.dict(
        os.environ, {FLAG: "true" if use_ddb else "false"}
    ):
        return build_intraday_narrow_table(FACTOR, start, end, store=store)


def run_signal_parity(start: str, end: str) -> Dict[str, Any]:
    from factors.intraday.ofi_persistence.test_compare import assert_consistency

    print(f"SIGNAL parity {start} → {end} ...", flush=True)
    py = _build_signal(False, start, end)
    db = _build_signal(True, start, end)
    metrics = assert_consistency(
        py.rename(columns={"tradetime": "bartime"}),
        db.rename(columns={"tradetime": "bartime"}),
    )
    print("SIGNAL PASS", json.dumps(metrics, indent=2, default=str), flush=True)
    return metrics


def run_backtest_parity(
    eval_start: Optional[str],
    eval_end: Optional[str],
) -> Dict[str, Any]:
    from factors.intraday.volume_front_loading import (
        run_backtest_parity as harness,
    )

    harness.FACTOR = FACTOR
    harness.FLAG = FLAG
    harness.PARITY_ROOT = PARITY_ROOT
    py_dir = harness._run_backtest_once(
        "python", False, eval_start=eval_start, eval_end=eval_end
    )
    db_dir = harness._run_backtest_once(
        "ddb", True, eval_start=eval_start, eval_end=eval_end
    )
    return harness.compare_backtests(py_dir, db_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--end", default=SAMPLE_END)
    parser.add_argument("--backtest-start", default=SAMPLE_START)
    parser.add_argument("--backtest-end", default=SAMPLE_END)
    args = parser.parse_args()

    if not any((args.trace, args.signal, args.backtest, args.all)):
        parser.print_help()
        return 0
    if args.trace or args.all:
        trace_production_path()
    if args.signal or args.all:
        if os.environ.get("RUN_DDB_TESTS") != "1":
            raise RuntimeError("Set RUN_DDB_TESTS=1 for live DDB validation")
        run_signal_parity(args.start, args.end)
    if args.backtest or args.all:
        if os.environ.get("RUN_DDB_TESTS") != "1":
            raise RuntimeError("Set RUN_DDB_TESTS=1 for live DDB validation")
        run_backtest_parity(args.backtest_start, args.backtest_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
