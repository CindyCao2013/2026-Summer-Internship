#!/usr/bin/env python
"""Sprint 16 Phase A — Liquidity Impact RankIC horizon / decay.

    python -m l2_factor_reproduction.scripts.run_sprint16_liquidity_impact_horizon

Does not modify frozen liquidity-impact formulas. Phase B/C/D are not run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.liquidity_impact_execution import (  # noqa: E402
    OUT_ROOT,
    run_phase_a,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip fast_context sha256 verify (debug only)",
    )
    args = parser.parse_args()
    out_root = Path(args.output_root) if args.output_root else OUT_ROOT / "phase_a"
    t0 = time.perf_counter()
    summary, _detail, parity = run_phase_a(
        output_root=out_root,
        verify_hash=not args.skip_hash,
    )
    wall = time.perf_counter() - t0
    print(parity.to_string(index=False))
    print()
    print(
        summary[
            [
                "factor",
                "rank_ic_h1",
                "half_life_days",
                "decay_class",
                "recommended_next_phase",
            ]
        ].to_string(index=False)
    )
    print(f"[done] Phase A {len(summary)} factors in {wall:.1f}s -> {out_root}")
    if not bool(parity["parity_pass"].all()):
        print("[warn] H=1 parity failed vs candidate-pool summary.json")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
