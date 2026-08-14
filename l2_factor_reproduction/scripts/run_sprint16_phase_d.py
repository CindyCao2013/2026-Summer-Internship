#!/usr/bin/env python
"""Sprint 16 Phase D — D1 hysteresis then D2 long-tail breadth.

    python -m l2_factor_reproduction.scripts.run_sprint16_phase_d

Frozen: impact_per_trade RAW + staggered 5D. Does not relax NetSR>1.5.
Does not run Daily/10D, 4x3 grids, or cvxpy.
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
from l2_factor_reproduction.python.liquidity_impact_phase_d import (  # noqa: E402
    run_phase_d,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--d1-only", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    d1, d1_pick, d2, d2_pick = run_phase_d(
        verify_hash=not args.skip_hash,
        d1_only=args.d1_only,
    )
    cols = [
        "buffer_label",
        "avg_hl_turnover_l1",
        "gross_hl_sharpe",
        "net_hl_sharpe",
        "net_hl_annu",
        "gross_sr_sacrifice_per_to",
        "economic_pass_no_mono",
    ]
    print(d1[cols].to_string(index=False))
    print()
    print(f"D1 winner buffer={d1_pick.get('buffer_width')} ({d1_pick.get('reason')})")
    if not d2.empty:
        print()
        print(
            d2[
                [
                    "tail",
                    "buffer_label",
                    "avg_hl_turnover_l1",
                    "gross_hl_sharpe",
                    "net_hl_sharpe",
                    "economic_pass_no_mono",
                ]
            ].to_string(index=False)
        )
        print(
            f"D2 winner tail={d2_pick.get('tail')} "
            f"NetSR={d2_pick.get('net_hl_sharpe')} ({d2_pick.get('reason')})"
        )
    print(
        f"[done] Phase D in {time.perf_counter() - t0:.1f}s -> {OUT_ROOT / 'phase_d'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
