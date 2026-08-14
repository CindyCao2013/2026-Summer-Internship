#!/usr/bin/env python3
"""Unified Factor Research entry point — Milestone 1A skeleton.

Examples:
  python run_factor_research.py --factor TGD20 --mode research
  python run_factor_research.py --factor TGD20 --mode production
  python run_factor_research.py --factor TGD20 --mode production --dry-run
  python run_factor_research.py --list-adapters

Does NOT migrate packs, upgrade registry, or modify factor formulas.
"""

from __future__ import annotations

import argparse
import json
import sys

from factor_research_harness import (
    list_registered_adapters,
    protocol_path,
    run_factor_research,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Factor Research Harness (Protocol v1) — Milestone 1A",
    )
    p.add_argument("--factor", type=str, help="factor_id, e.g. TGD20")
    p.add_argument(
        "--mode",
        choices=("research", "production"),
        default="production",
        help="Dual Benchmark track (Protocol v1)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve benchmark + load spec only",
    )
    p.add_argument(
        "--list-adapters",
        action="store_true",
        help="List registered factor adapters and exit",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full HarnessResult as JSON",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_adapters:
        print("Registered adapters:", ", ".join(list_registered_adapters()) or "(none)")
        print("Protocol:", protocol_path())
        return 0

    if not args.factor:
        print("error: --factor is required (or use --list-adapters)", file=sys.stderr)
        return 2

    result = run_factor_research(args.factor, args.mode, dry_run=args.dry_run)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"factor={result.factor_id} mode={result.mode} ok={result.ok}")
        print(
            f"benchmark_version={result.benchmark.get('benchmark_version')} "
            f"universe={result.benchmark.get('universe')} "
            f"horizon={result.benchmark.get('horizon_days')}D "
            f"neut={result.benchmark.get('neutralization')}"
        )
        for stage in result.stages:
            print(f"  [{stage.status:16}] {stage.name}: {stage.message}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
