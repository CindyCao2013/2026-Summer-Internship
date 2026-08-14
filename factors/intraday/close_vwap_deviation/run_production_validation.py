#!/usr/bin/env python3
"""Production validation: Python vs DDB close_vwap_deviation before enabling flag.

Usage::

    # Parity only (needs DDB)
    RUN_DDB_TESTS=1 python factors/intraday/close_vwap_deviation/run_production_validation.py

    # Full intraday backtest — run twice manually:
    #   INTRADAY_CLOSE_VWAP_USE_DDB=False  →  python Intraday_Factor_Test_Process.py
    #   INTRADAY_CLOSE_VWAP_USE_DDB=True   →  python Intraday_Factor_Test_Process.py
    # Compare result/intraday/close_vwap_deviation/group_performance_summary.csv

Compare metrics: IC, ICIR, H-L Sharpe (excess + raw), turnover, bartime set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factors.intraday.close_vwap_deviation.test_compare import (  # noqa: E402
    assert_consistency,
)
from factors.intraday.close_vwap_deviation.compute import (  # noqa: E402
    ddb_version,
    python_version,
)


def main() -> int:
    if os.environ.get("RUN_DDB_TESTS") != "1":
        print(
            "Set RUN_DDB_TESTS=1 to run live DDB parity validation.\n"
            "For full backtest compare, toggle factor_config.INTRADAY_CLOSE_VWAP_USE_DDB "
            "and run Intraday_Factor_Test_Process.py twice.",
        )
        return 0

    start = os.environ.get("CVWAP_VAL_START", "2024-05-01")
    end = os.environ.get("CVWAP_VAL_END", "2024-05-31")
    print(f"Validating close_vwap_deviation {start} → {end} ...", flush=True)
    py = python_version(start, end)
    db = ddb_version(start, end)
    metrics = assert_consistency(py, db)
    print("PASS", metrics, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
