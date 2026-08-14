#!/usr/bin/env python
"""Stage B full-universe novelty for LR BDL survivors only.

Does not rerun BDL, Full Fast Discovery, FS, or ML.
Does not mutate candidate_pool_v1.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.liquidity_resilience.full_novelty import (  # noqa: E402
    FULL_NOVELTY_CSV,
    STATUS_ALIAS,
    STATUS_PASS,
    STATUS_REVIEW,
    run_full_universe_novelty,
)


def main() -> int:
    frame = run_full_universe_novelty()
    print(frame.to_string(index=False), flush=True)
    n_pass = int((frame["novelty_status"] == STATUS_PASS).sum())
    n_review = int((frame["novelty_status"] == STATUS_REVIEW).sum())
    n_alias = int((frame["novelty_status"] == STATUS_ALIAS).sum())
    eligible = n_pass + n_review
    print(
        f"[lr-novelty] PASS_INDEPENDENT={n_pass} "
        f"REVIEW_MODERATE_OVERLAP={n_review} "
        f"REJECT_HIDDEN_ALIAS={n_alias} "
        f"eligible_for_full_discovery={eligible}",
        flush=True,
    )
    print(f"[lr-novelty] artifact={FULL_NOVELTY_CSV}", flush=True)
    print("[lr-novelty] STOP. Do not start Full Fast Discovery or FS/ML.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
