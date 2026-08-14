#!/usr/bin/env python
"""Sprint 16 Phase B — staggered holding on frozen Phase C versions.

    python -m l2_factor_reproduction.scripts.run_sprint16_phase_b

Does not expand the MA grid. Does not run Phase D / E.
After this run, R3 is reduced to at most two names.
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
)
from l2_factor_reproduction.python.liquidity_impact_phase_b import (  # noqa: E402
    run_phase_b,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    metrics, selection = run_phase_b(verify_hash=not args.skip_hash)
    cols = [
        "factor",
        "version",
        "hold_label",
        "avg_hl_turnover_l1",
        "gross_hl_sharpe",
        "net_hl_annu",
        "net_hl_sharpe",
        "decile_mono_spearman",
        "economic_strategy_pass",
        "grade",
    ]
    print(metrics[cols].to_string(index=False))
    print()
    print(f"R3 mechanical keep={selection.get('keep')} drop={selection.get('drop')}")
    print(f"R3 research keep={selection.get('research_keep')} ({selection.get('research_reason')})")
    print(
        f"[done] Phase B {len(metrics)} rows in "
        f"{time.perf_counter() - t0:.1f}s -> {OUT_ROOT / 'phase_b'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
