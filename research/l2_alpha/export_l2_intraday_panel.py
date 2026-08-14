#!/usr/bin/env python3
"""Export CSI1000 SSL2 minute agg panels (ClickHouse → parquet)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.l2_alpha.clickhouse_ssl2 import (  # noqa: E402
    connect_hf_client,
    extract_minute_agg_wide,
)
from research.l2_alpha.l2_factor_panel import (  # noqa: E402
    filter_bartimes,
    minute_wide_to_long,
)
from research.l2_alpha.l2_factor_registry import (  # noqa: E402
    DEFAULT_BARTIMES,
    L2_PHASE2_FACTORS,
    UNIVERSE_INDEX,
)

DEFAULT_OUTPUT = ROOT / "research/results/l2_factor_panel"


def _csi1000_symbols(start: str, end: str) -> List[str]:
    import Factor_Dev_Lib as fdl

    mask = fdl.get_index_member_mask(UNIVERSE_INDEX, start, end)
    if mask is None or mask.empty:
        raise RuntimeError("CSI1000 membership mask is empty")
    return sorted(str(c) for c in mask.columns)


def _limit_symbols_balanced(symbols: List[str], limit: int) -> List[str]:
    """Keep SH/SZ mix so SSE-only cancel_pressure is not dropped in smoke runs."""
    if limit <= 0 or len(symbols) <= limit:
        return symbols
    sh = [s for s in symbols if s.endswith(".SH")]
    sz = [s for s in symbols if s.endswith(".SZ")]
    n_sh = min(len(sh), limit // 2)
    n_sz = min(len(sz), limit - n_sh)
    # Top up if one side is short.
    if n_sh + n_sz < limit:
        n_sh = min(len(sh), limit - n_sz)
    return sorted(sh[:n_sh] + sz[:n_sz])


def _agg_map() -> dict:
    return {name: spec["aggregation"] for name, spec in L2_PHASE2_FACTORS.items()}


def export_day(
    day: str,
    *,
    symbols: Optional[List[str]],
    output_dir: Path,
    bartimes: List[str],
    client=None,
) -> Path:
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wide = extract_minute_agg_wide(
        day,
        next_day,
        symbols=symbols,
        client=client,
        bartimes=bartimes,
    )
    long = minute_wide_to_long(
        wide,
        factor_columns=list(L2_PHASE2_FACTORS.keys()),
        aggregation_map=_agg_map(),
    )
    long = filter_bartimes(long, bartimes)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pd.Timestamp(day).strftime('%Y%m%d')}.parquet"
    if long.empty:
        # Still write empty schema for idempotent downstream loops.
        long = long.reindex(columns=list(minute_wide_to_long(
            pd.DataFrame(), factor_columns=[]
        ).columns))
    long.to_parquet(path, index=False)
    return path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-06-03")
    parser.add_argument("--end", default="2024-06-03")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bartimes",
        default=",".join(DEFAULT_BARTIMES),
        help="Comma-separated HH:MM slots",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Do not restrict to CSI1000 (expensive)",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=0,
        help="Optional cap for smoke exports",
    )
    args = parser.parse_args(argv)

    bartimes = [b.strip() for b in args.bartimes.split(",") if b.strip()]
    days = pd.bdate_range(args.start, args.end)
    symbols = None
    if not args.all_symbols:
        symbols = _csi1000_symbols(args.start, args.end)
        if args.limit_symbols > 0:
            symbols = _limit_symbols_balanced(symbols, args.limit_symbols)
        print(f"[export] CSI1000 symbols={len(symbols)}", flush=True)

    client = connect_hf_client()
    try:
        for day in days:
            day_s = day.strftime("%Y-%m-%d")
            path = export_day(
                day_s,
                symbols=symbols,
                output_dir=args.output,
                bartimes=bartimes,
                client=client,
            )
            n = len(pd.read_parquet(path)) if path.exists() else 0
            print(f"[export] {day_s} rows={n} → {path}", flush=True)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
