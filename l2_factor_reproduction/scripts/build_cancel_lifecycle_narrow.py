#!/usr/bin/env python
"""Materialize 7 frozen Cancellation Family factor narrows (Sprint 6B).

Reads partitioned cancel_lifecycle_daily dataset/, applies the frozen
candidate formulas (candidate_registry_v1.csv), including the two 20d
shocks with history-excluding-today standardization, and writes
factor_narrow.parquet under
candidate_pool_v1/cancel_lifecycle_family/factors/<name>/.

Does NOT modify formulas, windows, guards or direction.
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
from l2_factor_reproduction.python.ch_cancel_lifecycle import (  # noqa: E402
    PRIMITIVE_VERSION,
    build_candidates,
    shock_20d,
)

PRIMITIVE_DIR = (
    Path(RESULT_ROOT) / "primitives" / "cancel_lifecycle_daily" / "dataset"
)
POOL_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "cancel_lifecycle_family"
FACTOR_ROOT = POOL_DIR / "factors"
REGISTRY_SRC = (
    Path(RESULT_ROOT) / "primitives" / "cancel_lifecycle_daily"
    / "candidate_registry_v1.csv"
)

CANDIDATES = [
    "cancel_value_pressure",
    "cancel_count_pressure",
    "cancel_value_intensity",
    "cancel_qty_intensity",
    "relative_cancel_order_size",
    "cancel_pressure_shock_20d",
    "cancel_intensity_shock_20d",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(PRIMITIVE_DIR.glob("year=*/cancel_daily_*.parquet")):
        parts = path.stem.split("_")
        chunk_start = pd.Timestamp(parts[-2])
        chunk_end = pd.Timestamp(parts[-1])
        if chunk_end < start or chunk_start > end:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no primitive chunks under {PRIMITIVE_DIR}")
    frame = pd.concat(frames, ignore_index=True)
    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
    mask = frame["TradeDate"].between(start, end)
    return frame.loc[mask].reset_index(drop=True)


def _to_narrow(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    out = panel[["symbol", "TradeDate", name]].dropna(subset=[name]).copy()
    out = out.rename(columns={"TradeDate": "tradetime", name: "value"})
    out["tradetime"] = (
        pd.to_datetime(out["tradetime"]).dt.normalize()
        + pd.Timedelta(hours=9, minutes=30)
    )
    out["factorname"] = name
    return out[["symbol", "tradetime", "factorname", "value"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-07-31")
    args = parser.parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    FACTOR_ROOT.mkdir(parents=True, exist_ok=True)
    prim = _load_dataset(start, end)
    print(f"[load] rows={len(prim):,} "
          f"{prim['TradeDate'].min().date()}.."
          f"{prim['TradeDate'].max().date()}", flush=True)

    cand = build_candidates(prim)
    cand = cand.sort_values(["symbol", "TradeDate"]).reset_index(drop=True)
    cand["cancel_pressure_shock_20d"] = cand.groupby("symbol")[
        "cancel_value_pressure"].transform(shock_20d)
    cand["cancel_intensity_shock_20d"] = cand.groupby("symbol")[
        "cancel_value_intensity"].transform(shock_20d)

    coverage_rows = []
    hashes = {}
    for name in CANDIDATES:
        narrow = _to_narrow(cand, name)
        out_dir = FACTOR_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "factor_narrow.parquet"
        narrow.to_parquet(path, index=False, compression="zstd")
        hashes[name] = _sha256(path)
        coverage_rows.append({
            "factor": name,
            "n_factor_rows": int(len(narrow)),
            "date_min": str(narrow["tradetime"].min().date()),
            "date_max": str(narrow["tradetime"].max().date()),
            "n_symbols": int(narrow["symbol"].nunique()),
            "n_dates": int(narrow["tradetime"].nunique()),
            "bytes": int(path.stat().st_size),
        })
        print(f"[narrow] {name}: rows={len(narrow):,} "
              f"symbols={coverage_rows[-1]['n_symbols']:,}", flush=True)

    pd.DataFrame(coverage_rows).to_csv(
        POOL_DIR / "factor_coverage.csv", index=False)
    registry = pd.read_csv(REGISTRY_SRC)
    registry.to_csv(POOL_DIR / "factor_registry.csv", index=False)
    (POOL_DIR / "factor_registry.json").write_text(
        json.dumps(registry.to_dict("records"), indent=2), encoding="utf-8",
    )
    manifest = {
        "version": "cancel_lifecycle_family_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "primitive": str(PRIMITIVE_DIR),
        "primitive_schema_version": PRIMITIVE_VERSION,
        "formula_version": PRIMITIVE_VERSION,
        "n_factors": len(CANDIDATES),
        "factors": CANDIDATES,
        "narrow_sha256": hashes,
        "narrow_format": "symbol, tradetime(TradeDate+09:30), factorname, value",
        "note": (
            "frozen cancel_lifecycle_v1 formulas; shock_20d uses "
            "shift(1).rolling(20) excluding today; no parameter search"
        ),
    }
    (POOL_DIR / "narrow_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    print(f"[done] {len(CANDIDATES)} narrows -> {POOL_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
