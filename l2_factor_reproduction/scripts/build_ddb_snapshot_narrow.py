#!/usr/bin/env python
"""Materialize factor narrow files for the DDB reference snapshot family.

Reads the partitioned daily primitive dataset and writes one
factor_narrow.parquet per frozen formula into the candidate pool layout
(candidate_pool_v1/ddb_reference_snapshot_family/factors/<name>/), plus
factor_coverage.csv consumed by the unified baseline runner.

Usage:
    python build_ddb_snapshot_narrow.py [--start 2019-01-01] [--end 2026-07-31]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    COVERAGE_THRESHOLD,
    FACTOR_NAMES,
    FORMULA_VERSION,
    SCHEMA_VERSION,
)
from l2_factor_reproduction.python.ddb_snapshot_factors import (  # noqa: E402
    primitive_to_narrow,
    registry_frame,
)

PRIMITIVE_DIR = (
    Path(RESULT_ROOT) / "primitives" / "ddb_reference_snapshot" / "dataset"
)
POOL_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "ddb_reference_snapshot_family"
)
FACTOR_ROOT = POOL_DIR / "factors"


def _load_dataset(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(
        PRIMITIVE_DIR.glob("year=*/ddb_snapshot_daily_*.parquet")
    ):
        parts = path.stem.split("_")
        chunk_start = pd.Timestamp(parts[-2])
        chunk_end = pd.Timestamp(parts[-1])
        if chunk_end < start or chunk_start > end:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no primitive chunks under {PRIMITIVE_DIR}")
    frame = pd.concat(frames, ignore_index=True)
    mask = frame["TradeDate"].between(start, end)
    return frame.loc[mask].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-07-31")
    args = parser.parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    FACTOR_ROOT.mkdir(parents=True, exist_ok=True)
    frame = _load_dataset(start, end)
    eligible = frame.loc[frame["coverage_ratio"] >= COVERAGE_THRESHOLD]
    print(
        f"[load] rows={len(frame):,} eligible(coverage>={COVERAGE_THRESHOLD})"
        f"={len(eligible):,}",
        flush=True,
    )

    coverage_rows = []
    for name in FACTOR_NAMES:
        narrow_all = primitive_to_narrow(frame, name)
        narrow = primitive_to_narrow(eligible, name)
        output = FACTOR_ROOT / name
        output.mkdir(parents=True, exist_ok=True)
        narrow.to_parquet(
            output / "factor_narrow.parquet",
            index=False,
            compression="zstd",
        )
        coverage_rows.append(
            {
                "factor": name,
                "n_factor_rows": int(len(narrow)),
                "n_rows_unfiltered": int(len(narrow_all)),
                "date_min": str(narrow["tradetime"].min().date()),
                "date_max": str(narrow["tradetime"].max().date()),
                "n_symbols": int(narrow["symbol"].nunique()),
                "n_dates": int(narrow["tradetime"].nunique()),
                "bytes": int(
                    (output / "factor_narrow.parquet").stat().st_size
                ),
            }
        )
        print(
            f"[narrow] {name}: rows={len(narrow):,} "
            f"symbols={coverage_rows[-1]['n_symbols']:,}",
            flush=True,
        )
    coverage = pd.DataFrame(coverage_rows).set_index("factor")
    coverage.to_csv(POOL_DIR / "factor_coverage.csv")
    registry_frame().to_csv(POOL_DIR / "factor_registry.csv", index=False)
    (POOL_DIR / "factor_registry.json").write_text(
        json.dumps(
            registry_frame().to_dict("records"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    narrow_hashes = {}
    for name in FACTOR_NAMES:
        path = FACTOR_ROOT / name / "factor_narrow.parquet"
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        narrow_hashes[name] = digest.hexdigest()
    manifest = {
        "version": "ddb_reference_snapshot_family_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "primitive": str(PRIMITIVE_DIR),
        "primitive_schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "n_factors": len(FACTOR_NAMES),
        "factors": list(FACTOR_NAMES),
        "narrow_sha256": narrow_hashes,
        "narrow_format": "symbol, tradetime(TradeDate+09:30), factorname, value",
        "output_tier": "guarded_formula_output",
        "guard_spec": (
            "NOT raw official formula output: SQL applies nullIf(denominator,0) "
            "for time_weighted_order_slope and a std floor 1e-7 for wavg_soir; "
            "small-denominator (<1e-6) and |x|>clip-guard events are monitored "
            "only (never modified). winsorized/trimmed variants exist only in "
            "the twos stability audit, not in this narrow."
        ),
        "note": (
            "official DolphinDB formulas replicated on company ClickHouse; "
            "effective formulas only; no parameter search"
        ),
    }
    (POOL_DIR / "narrow_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] narrow factors -> {FACTOR_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
