#!/usr/bin/env python
"""Locate the exact integer-column NA events in a date range.

Day-by-day fetch with the frozen NA policy collector: every NA event is
recorded (date, column, na_count, na_share, policy, affected_symbols);
hard-fail events are caught and recorded too so the scan completes.

Output: integer_na_fill_report rows appended to the probe CSV.

Usage:
    python probe_integer_na.py --start 2025-02-06 --end 2025-02-28 \
        --out probe_na_2025-02.csv
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    fetch_ddb_snapshot_daily,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = []
    errors = []
    days = pd.date_range(args.start, args.end, freq="D")
    for day in days:
        d = str(day.date())
        try:
            frame = fetch_ddb_snapshot_daily(d, d, na_report=report)
            if frame.empty:
                print(f"[{d}] non-trading or empty", flush=True)
                continue
            print(f"[{d}] rows={len(frame):,}", flush=True)
        except ValueError as exc:
            errors.append({"date": d, "error": str(exc)[:500]})
            print(f"[{d}] HARD FAIL: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - probe must not stop
            errors.append(
                {"date": d, "error": traceback.format_exc()[-500:]}
            )
            print(f"[{d}] ERROR: {exc}", flush=True)
        # flush partial report after every day
        if report:
            pd.DataFrame(report).to_csv(args.out, index=False)
    if errors:
        err_path = Path(args.out).with_suffix(".errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
    if report:
        out = pd.DataFrame(report)
        out.to_csv(args.out, index=False)
        print("\n== NA events ==")
        print(out.to_string(index=False))
    else:
        print("\n== no integer NA events in range ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
