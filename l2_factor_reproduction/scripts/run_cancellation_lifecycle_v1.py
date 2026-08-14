#!/usr/bin/env python
"""Sprint 10 — Cancellation / Order Lifecycle Family v1.

Usage:
    python -m l2_factor_reproduction.scripts.run_cancellation_lifecycle_v1
    python -m l2_factor_reproduction.scripts.run_cancellation_lifecycle_v1 --monthly-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.cancellation_lifecycle_v1 import (  # noqa: E402
    OUT_ROOT,
    render_discovery_report,
    run_fast_discovery,
    run_monthly_gate,
    select_next,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monthly-only",
        action="store_true",
        help="Stop after monthly gate (do not run discovery)",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("===== Sprint 10 Phase 1: Monthly Gate 2024-06 =====", flush=True)
    gate = run_monthly_gate()
    if not gate["gate_pass"]:
        (OUT_ROOT / "SPRINT10_MONTHLY_GATE_FAIL").write_text(
            "MONTHLY_GATE_FAIL — discovery build blocked\n", encoding="utf-8"
        )
        print("[STOP] monthly gate FAIL — no discovery build", flush=True)
        return 1

    if args.monthly_only:
        print("[done] monthly-only", flush=True)
        return 0

    print("\n===== Sprint 10 Phase 2: Discovery Fast Lane =====", flush=True)
    summary = run_fast_discovery(gate)
    selection = select_next(summary)
    (OUT_ROOT / "report.md").write_text(
        render_discovery_report(summary, gate, selection), encoding="utf-8"
    )
    if selection["status"] == "HAS_STRONG":
        (OUT_ROOT / "next_full_validation_candidate.md").write_text(
            selection["next_md"], encoding="utf-8"
        )
    else:
        (OUT_ROOT / "SPRINT10_NO_STRONG_CANDIDATE").write_text(
            "SPRINT10_NO_STRONG_CANDIDATE\n", encoding="utf-8"
        )

    manifest = {
        "sprint": "Sprint 10 — Cancellation / Order Lifecycle Family v1",
        "monthly_gate_pass": gate["gate_pass"],
        "selection": selection["status"],
        "next_candidate": selection.get("factor"),
        "structure_risk_flagged": gate.get("structure_risk_flagged"),
        "strong_pool_eligible": gate.get("strong_pool_eligible"),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "protocol_ref": "evaluation_protocol_v2.0 (untouched)",
        "full_history_built": False,
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n===== Sprint 10 summary =====")
    print(
        summary[
            [
                "factor",
                "gate",
                "hl_sharpe",
                "decile_mono_spearman",
                "adjacent_violations",
                "g10_gross_excess_annual",
                "EXCHANGE_STRUCTURE_RISK",
            ]
        ].to_string(index=False)
    )
    print(f"selection={selection['status']} next={selection.get('factor')}")
    print(f"artifacts -> {OUT_ROOT} ({manifest['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
