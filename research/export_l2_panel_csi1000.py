#!/usr/bin/env python3
"""Export full CSI1000 SSL2 Phase-2 panels (resume-safe, day checkpoints)."""

from __future__ import annotations

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_THREAD = threading.local()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from research.l2_alpha.export_l2_intraday_panel import export_day  # noqa: E402
from research.l2_alpha.l2_factor_registry import (  # noqa: E402
    DEFAULT_BARTIMES,
    UNIVERSE_INDEX,
)

DEFAULT_OUTPUT = ROOT / "research/results/l2_factor_panel_csi1000"


def _membership_by_day(start: str, end: str) -> Dict[str, List[str]]:
    """Point-in-time CSI1000 members per trade date (≈1000 names/day)."""
    import Factor_Dev_Lib as fdl

    mask = fdl.get_index_member_mask(UNIVERSE_INDEX, start, end)
    if mask is None or mask.empty:
        raise RuntimeError("CSI1000 membership mask is empty")
    out: Dict[str, List[str]] = {}
    for dt, row in mask.iterrows():
        day_s = pd.Timestamp(dt).strftime("%Y-%m-%d")
        members = [str(c) for c, v in row.items() if pd.notna(v) and float(v) > 0]
        out[day_s] = sorted(members)
    return out


def _export_one(
    day_s: str,
    symbols: List[str],
    output_dir: Path,
    bartimes: List[str],
    client=None,
) -> tuple:
    t0 = time.time()
    # Holidays / non-trade days have no membership row.
    if not symbols:
        path = output_dir / f"{pd.Timestamp(day_s).strftime('%Y%m%d')}.parquet"
        pd.DataFrame(
            columns=[
                "date",
                "bartime",
                "symbol",
                "factor",
                "value",
                "source",
                "aggregation",
            ]
        ).to_parquet(path, index=False)
        return day_s, 0, time.time() - t0, str(path)

    own = client is None
    client = client or connect_hf_client()
    try:
        path = export_day(
            day_s,
            symbols=symbols,
            output_dir=output_dir,
            bartimes=bartimes,
            client=client,
        )
    finally:
        if own:
            client.close()
    n = 0
    if path.exists():
        n = len(pd.read_parquet(path, columns=["symbol"]))
    return day_s, n, time.time() - t0, str(path)


def _thread_client():
    client = getattr(_THREAD, "client", None)
    if client is None:
        client = connect_hf_client()
        _THREAD.client = client
    return client


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-08-18")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bartimes",
        default=",".join(DEFAULT_BARTIMES),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel day exports (≤10). Keep low to avoid CH contention.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if parquet exists",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1000,
        help="Rebuild existing files with fewer than this many rows",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 10:
        raise ValueError("workers must be in 1..10")

    bartimes = [b.strip() for b in args.bartimes.split(",") if b.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    membership = _membership_by_day(args.start, args.end)
    days = list(pd.bdate_range(args.start, args.end))
    todo = []
    for d in days:
        day_s = d.strftime("%Y-%m-%d")
        path = args.output / f"{d.strftime('%Y%m%d')}.parquet"
        if args.force or not path.exists():
            todo.append(day_s)
            continue
        # Replace sparse/smoke leftovers.
        try:
            n = len(pd.read_parquet(path, columns=["symbol"]))
        except Exception:  # noqa: BLE001
            todo.append(day_s)
            continue
        if day_s in membership and n < args.min_rows:
            todo.append(day_s)

    avg_n = (
        int(sum(len(v) for v in membership.values()) / max(len(membership), 1))
        if membership
        else 0
    )
    print(
        f"[panel] trade_days={len(membership)} avg_members={avg_n} "
        f"bdays={len(days)} todo={len(todo)} workers={args.workers} "
        f"→ {args.output}",
        flush=True,
    )
    if not todo:
        print("[panel] nothing to do", flush=True)
        return 0

    def _job(day_s: str):
        return _export_one(
            day_s,
            membership.get(day_s, []),
            args.output,
            bartimes,
            client=_thread_client(),
        )

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, day_s) for day_s in todo]
        for fut in as_completed(futs):
            day_s, n, dt, path = fut.result()
            done += 1
            print(
                f"[panel] {done}/{len(todo)} {day_s} rows={n} "
                f"syms={len(membership.get(day_s, []))} "
                f"{dt:.1f}s → {path}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
