"""Export Feature Factory panels (CH derived + CS ranks)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from research.l2_alpha.export_l2_intraday_panel import (  # noqa: E402
    _csi1000_symbols,
    _limit_symbols_balanced,
)
from research.l2_alpha.feature_factory.extract import (  # noqa: E402
    extract_derived_wide,
)
from research.l2_alpha.feature_factory.registry import (  # noqa: E402
    CS_RANK_SOURCES,
    L2_FF_ALL_COLUMNS,
    L2_FF_DERIVED_COLUMNS,
)
from research.l2_alpha.l2_factor_panel import (  # noqa: E402
    filter_bartimes,
    minute_wide_to_long,
)
from research.l2_alpha.l2_factor_registry import DEFAULT_BARTIMES  # noqa: E402

DEFAULT_OUTPUT = ROOT / "research/results/l2_feature_factory_v1/panel"


def add_cross_section_ranks(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-minute cross-sectional rank in [0, 1] for selected sources."""
    if wide.empty:
        return wide
    out = wide.copy()
    for src in CS_RANK_SOURCES:
        if src not in out.columns:
            continue
        rank_col = f"{src}_rank"
        out[rank_col] = out.groupby("minute_time", group_keys=False)[src].transform(
            lambda s: s.rank(method="average", pct=True)
        )
    return out


def export_day(
    day: str,
    *,
    symbols: Optional[List[str]],
    output_dir: Path,
    bartimes: List[str],
    client=None,
) -> Path:
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wide = extract_derived_wide(
        day,
        next_day,
        symbols=symbols,
        bartimes=bartimes,
        client=client,
    )
    wide = add_cross_section_ranks(wide)
    long = minute_wide_to_long(
        wide,
        factor_columns=list(L2_FF_ALL_COLUMNS),
        aggregation_map={c: "derived" for c in L2_FF_ALL_COLUMNS},
    )
    long = filter_bartimes(long, bartimes)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pd.Timestamp(day).strftime('%Y%m%d')}.parquet"
    if long.empty:
        long = pd.DataFrame(
            columns=[
                "date",
                "bartime",
                "symbol",
                "factor",
                "value",
                "source",
                "aggregation",
            ]
        )
    long.to_parquet(path, index=False)
    return path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-06-03")
    parser.add_argument("--end", default="2024-06-03")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bartimes", default=",".join(DEFAULT_BARTIMES))
    parser.add_argument("--limit-symbols", type=int, default=50)
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Newline-separated symbols; overrides CSI1000/limit sampling",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip days whose parquet already exists (resume-safe)",
    )
    args = parser.parse_args(argv)

    bartimes = [b.strip() for b in args.bartimes.split(",") if b.strip()]
    if args.symbols_file is not None:
        symbols = [
            s.strip()
            for s in args.symbols_file.read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
    else:
        symbols = _csi1000_symbols(args.start, args.end)
        if args.limit_symbols > 0:
            symbols = _limit_symbols_balanced(symbols, args.limit_symbols)
    print(
        f"[ff_export] symbols={len(symbols)} factors={len(L2_FF_DERIVED_COLUMNS)}+ranks",
        flush=True,
    )
    client = connect_hf_client()
    try:
        for day in pd.bdate_range(args.start, args.end):
            day_s = day.strftime("%Y-%m-%d")
            path = args.output / f"{day.strftime('%Y%m%d')}.parquet"
            if args.skip_existing and path.exists():
                n = len(pd.read_parquet(path))
                print(f"[ff_export] skip {day_s} rows={n}", flush=True)
                continue
            path = export_day(
                day_s,
                symbols=symbols,
                output_dir=args.output,
                bartimes=bartimes,
                client=client,
            )
            n = len(pd.read_parquet(path)) if path.exists() else 0
            print(f"[ff_export] {day_s} rows={n} → {path}", flush=True)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
