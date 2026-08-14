#!/usr/bin/env python
"""LR-0 feasibility audit: session/PIT/shock coverage from shared L2 minutes.

Does not compute RankIC. Does not retune BDL. Optional --smoke for one date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.discovery_lite.candidate_matrix import (  # noqa: E402
    load_trading_calendar,
)
from l2_factor_reproduction.discovery_lite.contracts import (  # noqa: E402
    LITE_END,
    LITE_START,
    lite_trading_dates,
)
from l2_factor_reproduction.liquidity_resilience.contracts import LR0_DIR  # noqa: E402
from l2_factor_reproduction.liquidity_resilience.materialize import (  # noqa: E402
    FeasibilityAccumulator,
    materialize_lite_dates,
    write_lr0_artifacts,
)
from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="single Lite date (2024-06-28 if present)")
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--output-dir", default=str(LR0_DIR))
    args = parser.parse_args()

    cal = load_trading_calendar("discovery")
    dates = lite_trading_dates(cal, start=LITE_START, end=LITE_END)
    if args.smoke:
        target = pd.Timestamp("2024-06-28")
        dates = pd.DatetimeIndex([d for d in dates if d == target] or [dates[min(40, len(dates) - 1)]])
    elif args.max_dates and args.max_dates > 0:
        dates = dates[: int(args.max_dates)]

    print(
        f"[lr0] dates={len(dates)} {dates[0].date()} → {dates[-1].date()} "
        f"scans={len(dates)*2} candidates=24 (coverage only, no RankIC)",
        flush=True,
    )
    client = connect_hf_client()
    acc = FeasibilityAccumulator()
    _panel, acc = materialize_lite_dates(
        dates, client=client, out_dir=Path(args.output_dir) / "_smoke_mat", acc=acc
    )
    verdict = write_lr0_artifacts(acc, Path(args.output_dir))
    print(f"[lr0] verdict={verdict} wrote {args.output_dir}", flush=True)
    return 0 if verdict in {"A", "B"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
