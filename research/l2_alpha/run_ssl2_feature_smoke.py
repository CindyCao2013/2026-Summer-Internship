#!/usr/bin/env python3
"""Smoke: extract one day of SSL2 minute features from ClickHouse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.l2_alpha.clickhouse_ssl2 import (  # noqa: E402
    connect_hf_client,
    extract_minute_features,
)
from research.l2_alpha.schema import FACTOR_NAMES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-06-03")
    parser.add_argument("--limit-symbols", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "research/results/l2_ssl2_feature_extractor_v1/smoke_minute_features.csv",
    )
    args = parser.parse_args()

    start = args.date
    # half-open end = next calendar day
    end = str(pd_next_day(args.date))

    client = connect_hf_client()
    try:
        # Pick a few liquid-looking symbols from each exchange that day.
        symbols = []
        for table in ("SSE_AL_SSL2_EXG", "SZSE_AL_SSL2_EXG"):
            n = max(1, args.limit_symbols // 2)
            # Prefer A-share equity prefixes; avoid repo/bond codes that dominate amount.
            prefix = (
                "(Symbol LIKE '6%' OR Symbol LIKE '688%')"
                if table.startswith("SSE")
                else "(Symbol LIKE '0%' OR Symbol LIKE '3%')"
            )
            q = f"""
            SELECT Symbol
            FROM cmds.`{table}`
            WHERE ExchTime >= toDateTime64('{start} 09:30:00', 6, 'Asia/Shanghai')
              AND ExchTime <  toDateTime64('{start} 09:35:00', 6, 'Asia/Shanghai')
              AND length(BidVolumes) >= 10
              AND {prefix}
            GROUP BY Symbol
            ORDER BY count() DESC
            LIMIT {n}
            """
            rows = client.query(q).result_rows
            symbols.extend(r[0] for r in rows)

        print(f"[smoke] date={start} symbols={symbols}", flush=True)
        panel = extract_minute_features(
            start, end, symbols=symbols, client=client
        )
    finally:
        client.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output, index=False)
    print(
        f"[smoke] rows={len(panel)} factors={sorted(panel['factorname'].unique()) if len(panel) else []} "
        f"names={panel['symbol'].nunique() if len(panel) else 0} "
        f"minutes={panel['tradetime'].nunique() if len(panel) else 0}",
        flush=True,
    )
    print(f"[smoke] wrote {args.output}", flush=True)
    if panel.empty:
        return 2
    missing = set(FACTOR_NAMES) - set(panel["factorname"].unique())
    if missing:
        print(f"[smoke] missing factors: {sorted(missing)}", flush=True)
        return 3
    return 0


def pd_next_day(date_str: str) -> str:
    import pandas as pd

    return (pd.Timestamp(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


if __name__ == "__main__":
    raise SystemExit(main())
