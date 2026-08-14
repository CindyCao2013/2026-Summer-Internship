#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the corrected mid_order_ratio cumulative-bucket cache.

The legacy cache included all positive-price/volume SZSE rows and only applied
09:30/15:00 bounds to the range endpoints. This builder uses the corrected
``ch_tick.fetch_tick_bucketed`` path:

- regular-session predicate on every row/date;
- SSE executions: Type='T';
- SZSE executions: Type='011' with both order numbers present;
- candidate symbol list from the Wind EOD return panel; Tick coverage is SSE/SZSE.

Existing legacy caches are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import get_Ret_Matrix  # noqa: E402
from l2_factor_reproduction.python.ch_tick import fetch_tick_bucketed  # noqa: E402

BOUNDARIES = [
    20_000,
    30_000,
    40_000,
    50_000,
    60_000,
    100_000,
    150_000,
    200_000,
    250_000,
    300_000,
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJ_ROOT
            / "research/results/l2_reproduction/mid_order_ratio/analysis/param_sensitivity"
        ),
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{start.date()}_{end.date()}"
    output = out_dir / f"tick_bucketed_strict_trade_{stem}.parquet"
    metadata_path = output.with_suffix(".metadata.json")
    if output.exists():
        print(f"Strict cache already exists: {output}")
        return

    print("Loading candidate symbol list from Wind EOD returns...", flush=True)
    returns = get_Ret_Matrix(start, end, method="c2c", base_index=None)
    symbols = sorted(map(str, returns.columns))
    del returns
    print(f"Wind candidate symbols: {len(symbols)}", flush=True)

    chunk_dir = out_dir / f"strict_trade_chunks_{stem}"
    chunk_dir.mkdir(exist_ok=True)
    quarter_starts = pd.date_range(start, end, freq="QS")
    edges = [start] + [x for x in quarter_starts if start < x <= end]
    edges += [end + pd.Timedelta(days=1)]

    parts = []
    chunk_files = []
    for chunk_start, chunk_end_exclusive in zip(edges[:-1], edges[1:]):
        chunk_end = chunk_end_exclusive - pd.Timedelta(days=1)
        chunk_path = chunk_dir / f"chunk_{chunk_start.date()}_{chunk_end.date()}.parquet"
        chunk_files.append(chunk_path)
        if chunk_path.exists():
            print(f"Reuse strict chunk: {chunk_path.name}", flush=True)
            part = pd.read_parquet(chunk_path)
        else:
            print(
                f"CH strict trade query: {chunk_start.date()} ~ {chunk_end.date()}",
                flush=True,
            )
            part = fetch_tick_bucketed(
                chunk_start,
                chunk_end,
                boundaries=BOUNDARIES,
                symbols=symbols,
            )
            part.to_parquet(chunk_path, index=False)
            print(f"  rows={len(part)}", flush=True)
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)
    combined["TradeDate"] = pd.to_datetime(combined["TradeDate"]).dt.normalize()
    duplicates = int(combined.duplicated(["symbol", "TradeDate"]).sum())
    if duplicates:
        raise RuntimeError(f"strict cache has {duplicates} duplicate symbol-days")
    combined = combined.sort_values(["TradeDate", "symbol"]).reset_index(drop=True)
    combined.to_parquet(output, index=False)

    metadata = {
        "output": str(output),
        "sha256": _sha256(output),
        "created_at": pd.Timestamp.now().isoformat(),
        "requested_start": str(start.date()),
        "requested_end": str(end.date()),
        "observed_start": str(combined["TradeDate"].min().date()),
        "observed_end": str(combined["TradeDate"].max().date()),
        "rows": int(len(combined)),
        "symbols": int(combined["symbol"].nunique()),
        "input_a_share_symbols": int(len(symbols)),
        "boundaries_rmb": BOUNDARIES,
        "session": "09:30:00 <= ExchTime < 15:00:01 on every TradeDate",
        "sse_trade_filter": "Type='T'",
        "szse_trade_filter": "Type='011' AND BidOrderNo>0 AND AskOrderNo>0",
        "chunk_files": [str(path) for path in chunk_files],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Strict cache: {output}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    print(f"SHA256: {metadata['sha256']}", flush=True)


if __name__ == "__main__":
    main()

