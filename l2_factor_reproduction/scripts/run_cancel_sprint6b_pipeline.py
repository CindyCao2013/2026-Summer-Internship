#!/usr/bin/env python
"""Sprint 6B end-to-end orchestrator.

Waits for the 4 cancel primitive workers to finish, then runs:
  partition audit → narrow ×7 → force-backtest baseline →
  post-baseline gate → A.6 pool refresh → candidate_expansion_final_manifest.md

Usage:
    python run_cancel_sprint6b_pipeline.py [--poll-seconds 120]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

PY = "/opt/conda/anaconda3/envs/base_93/bin/python"
PRIM = (
    PROJ_ROOT / "research/results/l2_reproduction/primitives"
    / "cancel_lifecycle_daily"
)
DATASET = PRIM / "dataset"
POOL = (
    PROJ_ROOT / "research/results/l2_reproduction/candidate_pool_v1"
)
FAMILY = POOL / "cancel_lifecycle_family"
LOG_DIR = PROJ_ROOT / "logs" / "cancel_build"
EXPECTED_PARTITIONS = 31


def _run(cmd: list, log_path: Path) -> int:
    print(f"[run] {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as handle:
        proc = subprocess.run(
            cmd, cwd=str(PROJ_ROOT), stdout=handle, stderr=subprocess.STDOUT,
        )
    print(f"[exit {proc.returncode}] {log_path}", flush=True)
    if proc.returncode != 0:
        print(log_path.read_text()[-2000:], flush=True)
    return proc.returncode


def _workers_alive() -> list:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "build_cancel_lifecycle_primitive"], text=True
        )
    except subprocess.CalledProcessError:
        return []
    return [line for line in out.splitlines()
            if "build_cancel_lifecycle_primitive.py" in line
            and "pgrep" not in line]


def _partition_count() -> int:
    return len(list(DATASET.glob("year=*/cancel_daily_*.parquet")))


def wait_for_build(poll: int) -> bool:
    print(f"[wait] for {EXPECTED_PARTITIONS} quarterly partitions "
          f"(poll={poll}s)", flush=True)
    while True:
        alive = _workers_alive()
        n = _partition_count()
        print(f"  [{datetime.now():%H:%M:%S}] partitions={n}/{EXPECTED_PARTITIONS} "
              f"workers={len(alive)}", flush=True)
        if n >= EXPECTED_PARTITIONS and not alive:
            return True
        if not alive and n < EXPECTED_PARTITIONS:
            print("[FAIL] workers exited before completing all partitions",
                  flush=True)
            for log in sorted(LOG_DIR.glob("*.log")):
                print(f"--- {log.name} ---")
                print(log.read_text()[-800:])
            return False
        time.sleep(poll)


def write_final_manifest() -> None:
    """candidate_expansion_final_manifest.md — Candidate Expansion CLOSED."""
    families = {
        "trade_flow": POOL / "trade_flow_family",
        "order_size": POOL / "order_size_family",
        "order_book": POOL / "order_book_family",
        "price_formation": POOL / "price_formation_family",
        "liquidity_impact": POOL / "liquidity_impact_family",
        "ddb_reference_snapshot": POOL / "ddb_reference_snapshot_family",
        "cancel_lifecycle": FAMILY,
    }
    summaries = []
    for name, path in families.items():
        csv = path / "candidate_summary.csv"
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        frame["family"] = name
        summaries.append(frame)
    all_s = pd.concat(summaries, ignore_index=True)
    # bridge
    bridge = Path(
        PROJ_ROOT / "research/results/l2_reproduction"
        / "net_buy_amount_mcap" / "summary.json"
    )
    bridge_count = 1 if bridge.exists() else 0

    formula_count = len(all_s) + bridge_count
    # empirical clusters: sum of per-family unique R* labels
    cluster_count = 0
    for name, path in families.items():
        csv = path / "redundancy_clusters_080.csv"
        if csv.exists():
            cluster_count += pd.read_csv(csv)[
                "redundancy_cluster_080"].nunique()
        elif (path / "candidate_summary.csv").exists():
            s = pd.read_csv(path / "candidate_summary.csv")
            if "redundancy_cluster_080" in s.columns:
                cluster_count += s["redundancy_cluster_080"].nunique()

    g10 = all_s.loc[all_s["g10_excess_sharpe"] > 3, "factor"].tolist()
    hl = all_s.loc[all_s["hl_sharpe"] > 3, "factor"].tolist()
    ic2 = all_s.loc[all_s["rank_ic_raw"].abs() >= 0.02, "factor"].tolist()
    icir3 = all_s.loc[all_s["icir_raw"].abs() >= 3, "factor"].tolist()
    mono = all_s.loc[
        all_s.get("decile_mono_spearman", pd.Series(dtype=float)) >= 0.8,
        "factor"].tolist() if "decile_mono_spearman" in all_s.columns else []
    yc = all_s.loc[
        all_s.get("sign_consistency", pd.Series(dtype=float)) >= 0.75,
        "factor"].tolist() if "sign_consistency" in all_s.columns else []

    cancel = all_s.loc[all_s["family"] == "cancel_lifecycle"]
    cancel_table = "```\n" + cancel[[
        c for c in [
            "factor", "rank_ic_raw", "icir_raw", "g10_excess_sharpe",
            "hl_sharpe", "avg_hl_turnover", "net_annu_after_fee",
            "sign_consistency", "decile_mono_spearman",
        ] if c in cancel.columns
    ]].to_string(index=False) + "\n```"

    lines = [
        "# Candidate Expansion Final Manifest",
        "",
        f"generated {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Status",
        "",
        "**Candidate Expansion Phase CLOSED.**",
        "",
        "Sprint 6B (Cancellation / Order Lifecycle Family) is the last "
        "family admitted to the L2 Candidate Pool. No further family "
        "additions are permitted.",
        "",
        "## Downstream stages (fixed order)",
        "",
        "1. Global Taxonomy",
        "2. Exposure Audit",
        "3. Execution Validation",
        "4. Incremental Alpha",
        "5. A/B/C Screening",
        "",
        "## Pool totals",
        "",
        f"- **formula_count**: {formula_count}",
        f"- **empirical_cluster_count** (|ρ|≥0.80, sum of family-local "
        f"clusters): {cluster_count}",
        f"- **families**: {len(families)} + mcap bridge",
        "",
        "## Threshold hits (all families, discovery baseline)",
        "",
        f"- G10 Excess Sharpe > 3: {g10 or 'none'}",
        f"- H-L Sharpe > 3: {hl or 'none'}",
        f"- |IC| ≥ 2%: {ic2 or 'none'}",
        f"- |ICIR| ≥ 3: {icir3 or 'none'}",
        f"- monotonicity ≥ 0.8: {mono or 'none'}",
        f"- yearly consistency ≥ 75%: {yc or 'none'}",
        "",
        "## Sprint 6B Cancellation Family baseline",
        "",
        cancel_table,
        "",
        "## Boundaries carried forward",
        "",
        "- Daily baseline remains the signal-discovery standard, not a "
        "production rebalance mandate.",
        "- No KEEP/DROP, no combination, no parameter / weekly / "
        "neutralization search in this phase.",
        "- Cross-family correlation is taxonomy reference only.",
        "",
    ]
    out = POOL / "candidate_expansion_final_manifest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    if not args.skip_wait:
        if not wait_for_build(args.poll_seconds):
            return 1

    steps = [
        ([PY, "l2_factor_reproduction/scripts/build_cancel_lifecycle_narrow.py"],
         LOG_DIR / "narrow.log"),
        ([PY, "-u",
          "l2_factor_reproduction/scripts/expand_cancel_lifecycle_family.py",
          "--force-backtest"],
         LOG_DIR / "baseline.log"),
        ([PY, "l2_factor_reproduction/scripts/post_baseline_gate_cancel.py"],
         LOG_DIR / "post_gate.log"),
        ([PY, "l2_factor_reproduction/scripts/build_candidate_pool_index.py"],
         LOG_DIR / "pool_index.log"),
    ]
    for cmd, log in steps:
        code = _run(cmd, log)
        if code != 0:
            print(f"[pipeline FAIL] at {log.name}", flush=True)
            return code

    write_final_manifest()
    print("[pipeline DONE] Candidate Expansion CLOSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
