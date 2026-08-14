#!/usr/bin/env python
"""Build cancel_lifetime_daily from ClickHouse (Sprint 15B).

Day-by-day × symbol IN-batches. NO alpha / discovery / FV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python import cancel_lifetime_daily as cld  # noqa: E402
from l2_factor_reproduction.scripts.build_liquidity_impact_primitive import (  # noqa: E402
    quarter_ranges,
)

OUT_DIR = Path(RESULT_ROOT) / "primitives" / "cancel_lifetime_daily"
DATASET_DIR = OUT_DIR / "dataset"
SWEEP_DS = Path(RESULT_ROOT) / "primitives" / "sweep_penetration_daily" / "dataset"
LIQ_DS = Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily" / "dataset"


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _module_sha256() -> str:
    import inspect

    return hashlib.sha256(inspect.getsource(cld).encode()).hexdigest()


def _load_calendar_and_symbols() -> pd.DataFrame:
    """TradeDate × symbol panel from existing primitive (calendar + universe)."""
    frames = []
    for root in (SWEEP_DS, LIQ_DS):
        files = sorted(root.glob("quarter=*/**/*.parquet"))
        if not files:
            files = sorted(root.glob("**/*.parquet"))
        for path in files:
            try:
                df = pd.read_parquet(path, columns=["symbol", "TradeDate"])
            except Exception:  # noqa: BLE001
                continue
            frames.append(df)
        if frames:
            break
    if not frames:
        raise FileNotFoundError("no sweep/liquidity primitives for calendar")
    panel = pd.concat(frames, ignore_index=True)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    panel["symbol"] = panel["symbol"].astype(str)
    panel = panel.drop_duplicates(["symbol", "TradeDate"])
    # Compact for fast day lookup
    return panel.sort_values(["TradeDate", "symbol"], kind="stable").reset_index(
        drop=True
    )


def _codes_for_day(panel: pd.DataFrame, day: str) -> Tuple[List[str], List[str]]:
    d = pd.Timestamp(day)
    sub = panel.loc[panel["TradeDate"] == d, "symbol"]
    sse = sorted({s.split(".")[0] for s in sub if s.endswith(".SH")})
    szse = sorted({s.split(".")[0] for s in sub if s.endswith(".SZ")})
    return sse, szse


def _run_day(
    client, day: str, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    day_end = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sse, szse = _codes_for_day(panel, day)
    frames = []
    hashes: Dict[str, str] = {}
    for exchange, codes in (("sse", sse), ("szse", szse)):
        if not codes:
            continue
        for i, batch in enumerate(cld.chunked(codes, cld.SYMBOL_BATCH_SIZE)):
            sql = cld.daily_sql(exchange, day, day_end, batch)
            hashes[f"{exchange}_{i}"] = cld.query_sha256(sql)
            frame = client.query_df(sql)
            frames.append(frame)
    daily = cld.finalize_daily(frames)
    return cld.prepare_cancel_lifetime_daily(daily), hashes


def _run_period(
    client, start: str, end: str, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) - pd.Timedelta(days=1)
    days = sorted(
        panel.loc[panel["TradeDate"].between(s, e), "TradeDate"].unique()
    )
    days = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in days]
    print(f"[days] {start}..{end} n={len(days)}", flush=True)
    frames = []
    last_hashes: Dict[str, str] = {}
    for i, day in enumerate(days):
        t0 = time.time()
        frame, hashes = _run_day(client, day, panel)
        last_hashes = hashes
        if len(frame):
            frames.append(frame)
        med = (
            float(frame["cancel_age_median_ms"].median())
            if len(frame) and frame["cancel_age_median_ms"].notna().any()
            else float("nan")
        )
        neg = int(frame["negative_lifetime_count"].fillna(0).sum()) if len(frame) else 0
        print(
            f"  [{i+1}/{len(days)}] {day} rows={len(frame)} "
            f"med_age={med:.0f} neg={neg} {time.time()-t0:.1f}s",
            flush=True,
        )
    if not frames:
        return pd.DataFrame(columns=list(cld.DAILY_COLUMNS)), {
            "query_sha256": last_hashes,
            "n_days": 0,
        }
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["TradeDate", "source_exchange", "symbol"], kind="stable"
    ).reset_index(drop=True)
    return out, {"query_sha256_sample": last_hashes, "n_days": len(days)}


def _quality_row(frame: pd.DataFrame, tag: str) -> Dict[str, object]:
    return {
        "partition": tag,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if len(frame) else 0,
        "actual_date_min": str(frame["TradeDate"].min().date()) if len(frame) else None,
        "actual_date_max": str(frame["TradeDate"].max().date()) if len(frame) else None,
        "mean_eligible": float(frame["eligible_order_count"].mean()) if len(frame) else None,
        "mean_cancel_age_median_ms": float(
            frame["cancel_age_median_ms"].dropna().mean()
        )
        if len(frame) and frame["cancel_age_median_ms"].notna().any()
        else float("nan"),
        "mean_censored_share": float(frame["censored_order_share"].dropna().mean())
        if len(frame) and frame["censored_order_share"].notna().any()
        else float("nan"),
        "sum_negative_lifetime": int(frame["negative_lifetime_count"].fillna(0).sum())
        if len(frame)
        else 0,
        "sse_rows": int((frame["source_exchange"] == "SSE").sum()) if len(frame) else 0,
        "szse_rows": int((frame["source_exchange"] == "SZSE").sum()) if len(frame) else 0,
    }


def _write_partition(frame: pd.DataFrame, quarter: str) -> Dict[str, object]:
    directory = DATASET_DIR / f"quarter={quarter}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cancel_lifetime_daily_{quarter}.parquet"
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return {
        "quarter": quarter,
        "path": str(path.relative_to(PROJ_ROOT)),
        "rows": int(len(frame)),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = connect_hf_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("[calendar] load symbols/dates from existing primitive", flush=True)
    panel = _load_calendar_and_symbols()
    print(
        f"  panel rows={len(panel)} dates={panel['TradeDate'].nunique()} "
        f"symbols={panel['symbol'].nunique()}",
        flush=True,
    )

    if args.smoke:
        ranges = [("smoke", args.start, args.end)]
    else:
        ranges = quarter_ranges(args.start, args.end)

    quality_rows = []
    for quarter, start, end in ranges:
        out_path = (
            DATASET_DIR
            / f"quarter={quarter}"
            / f"cancel_lifetime_daily_{quarter}.parquet"
        )
        if out_path.exists() and not args.force and not args.smoke:
            print(f"[skip] {quarter} exists", flush=True)
            continue
        frame, meta = _run_period(client, start, end, panel)
        if len(frame) == 0:
            print(f"[warn] {quarter} empty", flush=True)
            continue
        part = _write_partition(frame, quarter)
        part.update(meta)
        quality_rows.append(_quality_row(frame, quarter))
        print(f"[done] {quarter} rows={len(frame)}", flush=True)

    all_parts = []
    for path in sorted(DATASET_DIR.glob("quarter=*/cancel_lifetime_daily_*.parquet")):
        quarter = path.parent.name.split("=", 1)[1]
        all_parts.append(
            {
                "quarter": quarter,
                "path": str(path.relative_to(PROJ_ROOT)),
                "rows": int(pd.read_parquet(path, columns=["symbol"]).shape[0]),
                "sha256": _sha256(path),
            }
        )

    dates = []
    n_rows = 0
    for p in all_parts:
        df = pd.read_parquet(PROJ_ROOT / p["path"], columns=["TradeDate"])
        dates.append(pd.to_datetime(df["TradeDate"]).min())
        dates.append(pd.to_datetime(df["TradeDate"]).max())
        n_rows += p["rows"]

    manifest = {
        "primitive_name": "l2_primitive_cancel_lifetime_daily",
        "schema_version": cld.SCHEMA_VERSION,
        "formula_version": cld.FORMULA_VERSION,
        "canonical_source": cld.CANONICAL_SOURCE,
        "module_sha256": _module_sha256(),
        "host": platform.node(),
        "requested_start": args.start,
        "requested_end": args.end,
        "actual_min": str(min(dates).date()) if dates else None,
        "actual_max": str(max(dates).date()) if dates else None,
        "row_count": n_rows,
        "symbol_batch_size": cld.SYMBOL_BATCH_SIZE,
        "formulas": {
            "cancel_age_ms": "cancel_time - order_add_time (cancel-terminated only)",
            "partial_fill_then_cancel": "cancel_qty + eps < order_size (residual)",
            "universe": "continuous auction posted orders (SSE Type=A; SZSE Cat1/2)",
            "censored_order_share_v1": (
                "non-cancel eligible share (FULL_FILL ∪ SESSION_END_CENSORED); "
                "not separated without fill join"
            ),
            "primary_object": "cancel-lifetime / order commitment",
            "no_short_lived_threshold": True,
        },
        "daily_columns": list(cld.DAILY_COLUMNS),
        "event_level_note": (
            "Full event-level history not persisted (storage). "
            "Daily cancel-age aggregates via single-scan GROUP BY + symbol batches."
        ),
        "partition_checksums": all_parts,
        "build_quality": quality_rows,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    if quality_rows:
        pd.DataFrame(quality_rows).to_csv(OUT_DIR / "primitive_quality.csv", index=False)
    print(f"[manifest] {OUT_DIR / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
