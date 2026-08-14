#!/usr/bin/env python
"""Sprint 6A final gate 1 — partition version consistency audit.

For every ddb_reference_snapshot quarterly partition:
  partition path, row count, date_min/date_max, schema hash,
  primitive module hash, formula module hash (current code), file mtime.

Partitions whose mtime predates a known code-fix timestamp are flagged
needs_parity_check. For each flagged partition, one representative trading
day is recomputed with the CURRENT code and compared column-by-column
(parity). Partitions passing parity are accepted in the manifest with
legacy_chunk_accepted_by_parity=true; failing partitions must be rebuilt.

Usage:
    python audit_partition_version.py            # inventory + flags
    python audit_partition_version.py --parity 2019-03-29 --chunk <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    DAILY_COLUMNS,
    fetch_ddb_snapshot_daily,
)

DATASET_DIR = (
    Path(RESULT_ROOT) / "primitives" / "ddb_reference_snapshot" / "dataset"
)
MODULES = {
    "primitive_module": PROJ_ROOT
    / "l2_factor_reproduction/scripts/build_ddb_snapshot_primitive.py",
    "formula_module": PROJ_ROOT
    / "l2_factor_reproduction/python/ch_ddb_snapshot.py",
}
# code-fix timestamps (local wall time of the code edit): partitions written
# BEFORE these were built by older code. tie_flag: all surviving chunks are
# post-fix by construction (pre-fix code crashed on tie NA). whitelist: the
# whole long-sample fleet runs the interim 0.5%-tolerance build (never
# restarted after the strict-policy edit), so EVERY chunk counts as
# pre-strict-policy and requires parity acceptance before the dataset is final.
CODE_FIXES = {
    "tie_flag_ifnull_fix": "2026-08-06T14:00:00",
    "integer_na_whitelist_policy": "2099-01-01T00:00:00",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(str(pq.read_schema(path)).encode()).hexdigest()[:16]


def inventory() -> pd.DataFrame:
    module_hashes = {
        key: _sha256(path) for key, path in MODULES.items()
    }
    rows = []
    for path in sorted(DATASET_DIR.glob("year=*/ddb_snapshot_daily_*.parquet")):
        meta = pq.read_metadata(path)
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        frame_head = pd.read_parquet(path, columns=["TradeDate"])
        built_before = [
            name for name, ts in CODE_FIXES.items()
            if mtime < datetime.fromisoformat(ts)
        ]
        rows.append(
            {
                "partition": str(path.relative_to(DATASET_DIR)),
                "row_count": meta.num_rows,
                "date_min": str(frame_head["TradeDate"].min().date()),
                "date_max": str(frame_head["TradeDate"].max().date()),
                "schema_hash": _schema_hash(path),
                "primitive_module_hash": module_hashes["primitive_module"][:16],
                "formula_module_hash": module_hashes["formula_module"][:16],
                "file_mtime": mtime.isoformat(timespec="seconds"),
                "built_before_fixes": ",".join(built_before),
                "needs_parity_check": bool(built_before),
            }
        )
    out = pd.DataFrame(rows)
    out_path = DATASET_DIR.parent / "partition_version_audit.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\n[inventory] -> {out_path}")
    flagged = out.loc[out["needs_parity_check"], "partition"].tolist()
    if flagged:
        print(f"[parity] partitions needing parity check: {flagged}")
    return out


def parity(chunk_path: Path, day: str) -> bool:
    """Column-by-column parity of one trading day: stored partition vs
    recomputation with the current code."""
    stored = pd.read_parquet(chunk_path)
    stored["TradeDate"] = pd.to_datetime(stored["TradeDate"])
    stored_day = stored.loc[
        stored["TradeDate"] == pd.Timestamp(day)
    ].sort_values("symbol").reset_index(drop=True)
    if stored_day.empty:
        raise ValueError(f"{chunk_path.name}: no rows for {day}")
    fresh = fetch_ddb_snapshot_daily(day, day).sort_values(
        "symbol"
    ).reset_index(drop=True)
    if len(stored_day) != len(fresh):
        print(
            f"[parity FAIL] row count differs: stored={len(stored_day)} "
            f"fresh={len(fresh)}"
        )
        return False
    # ClickHouse aggregation order is not deterministic, so identical SQL run
    # twice can differ at the last ulp. Hard diffs (rtol/atol) separate that
    # FP noise from real version drift; NA-pattern changes always fail.
    hard, noise = [], []
    for column in DAILY_COLUMNS:
        left = stored_day[column]
        right = fresh[column]
        if pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right):
            both_na = left.isna() & right.isna()
            diff = (left.astype("float64") - right.astype("float64")).abs()
            tol = 1e-9 + 1e-9 * right.astype("float64").abs()
            bad = ((diff > tol) | (left.isna() != right.isna())) & ~both_na
            fp = ((diff > 0) & ~bad) & ~both_na
            if bad.any():
                hard.append((column, int(bad.sum()),
                             float(diff.loc[bad].max())))
            elif fp.any():
                noise.append((column, int(fp.sum()),
                              float(diff.loc[fp].max())))
        else:
            bad = left.fillna(-1) != right.fillna(-1)
            if bad.any():
                hard.append((column, int(bad.sum()), None))
    if hard:
        print(f"[parity FAIL] {chunk_path.name} @ {day}:")
        for column, n, mx in hard:
            print(f"  {column}: {n} rows differ (max abs diff {mx})")
        return False
    tag = "identical" if not noise else f"fp-noise only: {noise}"
    print(f"[parity OK] {chunk_path.name} @ {day}: {len(fresh)} rows, "
          f"{len(DAILY_COLUMNS)} columns {tag}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", metavar="YYYY-MM-DD", default=None,
                        help="representative trading day for parity check")
    parser.add_argument("--chunk", default=None,
                        help="partition path for the parity check")
    args = parser.parse_args()
    report = inventory()
    if args.parity and args.chunk:
        ok = parity(Path(args.chunk), args.parity)
        manifest_path = DATASET_DIR.parent / "partition_parity_manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest[Path(args.chunk).name] = {
            "parity_day": args.parity,
            "legacy_chunk_accepted_by_parity": bool(ok),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[manifest] -> {manifest_path}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
