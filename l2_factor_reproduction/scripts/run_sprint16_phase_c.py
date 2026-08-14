#!/usr/bin/env python
"""Sprint 16 Phase C + Horizon ICIR audit.

    python -m l2_factor_reproduction.scripts.run_sprint16_phase_c

SLOW-group smoothing only. Does not modify frozen formulas.
Does not run Phase B / D. Does not apply production TO<1 gate.
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
    run_horizon_icir_audit,
)
from l2_factor_reproduction.python.liquidity_impact_phase_c import (  # noqa: E402
    run_phase_c,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hash", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--phase-c-only", action="store_true")
    args = parser.parse_args()
    verify = not args.skip_hash
    code = 0

    if not args.phase_c_only:
        t0 = time.perf_counter()
        audit = run_horizon_icir_audit(verify_hash=verify)
        print(audit.to_string(index=False))
        print(f"[done] ICIR audit in {time.perf_counter() - t0:.1f}s")

    if not args.audit_only:
        t0 = time.perf_counter()
        metrics, promotions, parity = run_phase_c(verify_hash=verify)
        print(parity.to_string(index=False))
        print()
        show = metrics.merge(promotions, on=["factor", "version"])
        cols = [
            "factor",
            "version",
            "ic_retention",
            "avg_hl_turnover_l1",
            "turnover_reduction",
            "gross_hl_sharpe",
            "net_hl_sharpe",
            "decile_mono_spearman",
            "decision",
        ]
        print(show[cols].to_string(index=False))
        print(
            f"[done] Phase C {len(metrics)} versions in "
            f"{time.perf_counter() - t0:.1f}s -> {OUT_ROOT / 'phase_c'}"
        )
        if not bool(parity["parity_pass"].all()):
            print("[warn] RAW parity failed vs candidate-pool summary.json")
            code = 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
