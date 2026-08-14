#!/usr/bin/env python
"""Sprint 8 — Low-Turnover L2 Discovery v1 runner.

    python -m l2_factor_reproduction.scripts.run_low_turnover_v1

Outputs under research/results/l2_reproduction/fast_discovery/low_turnover_v1/:
    primitive_capability.csv
    candidate_summary.csv
    fast_profile.csv
    report.md
    figures/<factor>/{cumulative_hl,decile_bar}.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.low_turnover_v1 import (  # noqa: E402
    OUT_DIR,
    run_low_turnover_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=None,
        help="default: fast_discovery/low_turnover_v1",
    )
    parser.add_argument(
        "--capability-only",
        action="store_true",
        help="Phase 0 only: write primitive_capability.csv and exit",
    )
    args = parser.parse_args()
    out_root = Path(args.output_root) if args.output_root else OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    if args.capability_only:
        from l2_factor_reproduction.python.low_turnover_v1 import (
            build_primitive_capability,
        )

        capability = build_primitive_capability()
        path = out_root / "primitive_capability.csv"
        capability.to_csv(path, index=False)
        print(capability.to_string(index=False))
        print(f"[done] capability -> {path}")
        return 0

    t0 = time.perf_counter()
    summary, profile, capability = run_low_turnover_v1(output_root=out_root)
    wall = time.perf_counter() - t0
    n_run = int((summary["gate"] != "unavailable").sum())
    print(
        f"[done] {n_run} available / {len(summary)} total in {wall:.1f}s "
        f"-> {out_root}"
    )
    print(summary[["factor", "gate", "hl_sharpe", "decile_mono_spearman", "avg_hl_turnover"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
