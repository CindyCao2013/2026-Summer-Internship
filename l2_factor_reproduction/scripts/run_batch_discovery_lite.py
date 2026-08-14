#!/usr/bin/env python3
"""Batch Discovery Lite runner.

Examples:

    python l2_factor_reproduction/scripts/run_batch_discovery_lite.py \\
        --family trade_flow \\
        --registry path/to/candidate_registry.csv \\
        --window discovery_lite

    python l2_factor_reproduction/scripts/run_batch_discovery_lite.py \\
        --dry-run-existing --window discovery_lite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.discovery_lite.candidate_matrix import (  # noqa: E402
    load_candidate_registry,
)
from l2_factor_reproduction.discovery_lite.contracts import (  # noqa: E402
    OUTPUT_ROOT,
)
from l2_factor_reproduction.discovery_lite.pipeline import (  # noqa: E402
    load_dry_run_registry,
    run_batch_discovery_lite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=None, help="family name (filter registry)")
    parser.add_argument("--registry", default=None, help="candidate registry CSV")
    parser.add_argument(
        "--window",
        default="discovery_lite",
        choices=["discovery_lite", "discovery"],
        help="Lite window alias (maps to Fast Discovery discovery context)",
    )
    parser.add_argument(
        "--dry-run-existing",
        action="store_true",
        help="Engineering dry-run on a frozen subset of existing pool factors",
    )
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "materialized", "primitives"],
        help="candidate exposure source (default auto)",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    if args.window in {"discovery_lite", "discovery"}:
        context_window = "discovery"
    else:
        raise ValueError(args.window)

    if args.dry_run_existing:
        if args.registry:
            print("note: --dry-run-existing ignores --registry and uses the frozen pool subset")
        registry = load_dry_run_registry()
        run_name = "dry_run_existing"
        source = "materialized"
    else:
        if not args.registry:
            parser.error("--registry is required unless --dry-run-existing")
        registry = load_candidate_registry(Path(args.registry))
        if args.family:
            registry = registry.loc[
                registry["family"].astype(str) == str(args.family)
            ].copy()
            if registry.empty:
                raise SystemExit(f"no registry rows for family {args.family!r}")
        run_name = str(args.family or "custom_registry")
        source = args.source

    out_dir = (
        Path(args.output_root)
        if args.output_root
        else OUTPUT_ROOT / run_name
    )
    result = run_batch_discovery_lite(
        registry=registry,
        out_dir=out_dir,
        window=context_window,
        source=source,
        dry_run=bool(args.dry_run_existing),
        verify_hash=True,
    )
    ranking = result["ranking"]
    n_full = int(result["counts"]["full_discovery_survivors"])
    print(
        f"[bdl] wrote {out_dir} | candidates={len(registry)} "
        f"FULL_DISCOVERY_SURVIVOR={n_full} "
        f"total={result['timings']['total']:.1f}s"
    )
    if not ranking.empty:
        print(ranking[["factor", "family", "final_status", "rank_ic_lite"]].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
