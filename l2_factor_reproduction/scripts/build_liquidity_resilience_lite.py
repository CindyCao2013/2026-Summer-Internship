#!/usr/bin/env python
"""LR-1: freeze formulas, materialize Lite exposures, run existing BDL, STOP.

Does not run Full Fast Discovery, ML, or retune BDL thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.discovery_lite.candidate_matrix import (  # noqa: E402
    load_trading_calendar,
)
from l2_factor_reproduction.discovery_lite.contracts import (  # noqa: E402
    DRY_RUN_NOVELTY_REFERENCE,
    LITE_END,
    LITE_START,
    OUTPUT_ROOT,
    lite_trading_dates,
)
from l2_factor_reproduction.discovery_lite.pipeline import run_batch_discovery_lite  # noqa: E402
from l2_factor_reproduction.liquidity_resilience.contracts import (  # noqa: E402
    BDL_LINK_DIR,
    FROZEN_CANDIDATE_NAMES,
    LR0_DIR,
    LR1_MAT_DIR,
    LR_RESULT_ROOT,
)
from l2_factor_reproduction.liquidity_resilience.materialize import (  # noqa: E402
    FeasibilityAccumulator,
    materialize_lite_dates,
    write_formula_audit,
    write_freeze_manifest,
    write_frozen_registry,
    write_lr0_artifacts,
)
from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--skip-bdl", action="store_true")
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args()

    LR_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    registry_path = LR_RESULT_ROOT / "lr1_candidate_registry.csv"
    registry, digest = write_frozen_registry(registry_path)
    write_formula_audit(LR_RESULT_ROOT / "lr1_formula_audit.csv")
    manifest = write_freeze_manifest(
        LR_RESULT_ROOT / "lr1_freeze_manifest.json",
        extra={"registry_path": str(registry_path)},
    )
    print(
        f"[lr1] FROZEN {manifest['n_candidates']} formulas "
        f"registry_sha256={digest} formula_sha256={manifest['formula_sha256']}",
        flush=True,
    )
    print("[lr1] RankIC/ICIR/decile have not been inspected. Formulas are immutable for LR v1.", flush=True)

    cal = load_trading_calendar("discovery")
    dates = lite_trading_dates(cal, start=LITE_START, end=LITE_END)
    if args.max_dates and args.max_dates > 0:
        dates = dates[: int(args.max_dates)]

    acc = FeasibilityAccumulator()
    if not args.skip_materialize:
        client = connect_hf_client()
        panel, acc = materialize_lite_dates(dates, client=client, out_dir=LR1_MAT_DIR, acc=acc)
        print(f"[lr1] materialized panel rows={len(panel)} dates={acc.n_dates} scans={acc.n_db_scans}", flush=True)
    else:
        panel_path = LR1_MAT_DIR / "panel.parquet"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
            print(f"[lr1] reused {panel_path} rows={len(panel)}", flush=True)

    verdict = write_lr0_artifacts(acc, LR0_DIR)
    print(f"[lr0] verdict={verdict}", flush=True)
    if verdict == "C":
        print("[lr1] STOP: LR-0 not feasible; BDL not started", flush=True)
        return 2

    if args.skip_bdl:
        print("[lr1] skip BDL as requested", flush=True)
        return 0

    out_dir = OUTPUT_ROOT / "liquidity_resilience"
    print("[lr1] handing frozen registry to existing BDL (thresholds unchanged)", flush=True)
    result = run_batch_discovery_lite(
        registry=registry,
        out_dir=out_dir,
        window="discovery",
        source="materialized",
        dry_run=False,
        novelty_names=DRY_RUN_NOVELTY_REFERENCE,
        verify_hash=True,
    )
    BDL_LINK_DIR.mkdir(parents=True, exist_ok=True)
    pointer = {
        "bdl_output": str(out_dir),
        "novelty_limitation": "NOVELTY_REFERENCE_LIMITED",
        "novelty_reference": list(DRY_RUN_NOVELTY_REFERENCE),
        "n_candidates": len(FROZEN_CANDIDATE_NAMES),
        "counts": result.get("counts"),
        "note": "BDL artifacts live in discovery_lite/; this pointer is not a duplicate of Gate files.",
    }
    (BDL_LINK_DIR / "run_pointer.json").write_text(
        json.dumps(pointer, indent=2, default=str) + "\n", encoding="utf-8"
    )
    ranking = result.get("ranking")
    n_full = int(result["counts"]["full_discovery_survivors"])
    print(
        f"[lr1] BDL done FULL_DISCOVERY_SURVIVOR={n_full} out={out_dir}. "
        "STOP. Do not start Full Fast Discovery.",
        flush=True,
    )
    if ranking is not None and not ranking.empty:
        cols = [c for c in ("factor", "family", "final_status", "rank_ic_lite") if c in ranking.columns]
        print(ranking[cols].head(15).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
