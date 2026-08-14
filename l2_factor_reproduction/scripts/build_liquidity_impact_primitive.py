#!/usr/bin/env python
"""Build l2_primitive_liquidity_impact_daily from ClickHouse Tick + SSL2.

Pipeline:
1. 2024-06 full-month validation gate (invariants + KLIN amount parity).
2. Quarterly partitions 2019Q1 .. 2026Q3 (2026-07 is the final partial
   quarter) written as zstd parquet, one file per quarter. No combined
   parquet, no raw/minute panel.
3. manifest.json records sources, engines, query hashes, direction rules,
   time grid, amount units, join rules, proxy limitations, checksums,
   module hashes, environment and the benchmark/cost/shift policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python import (  # noqa: E402
    liquidity_impact_daily as lid,
)

SCHEMA_VERSION = "liquidity_impact_daily_v1"
FORMULA_VERSION = "frozen_v1"
OUT_DIR = Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily"
DATASET_DIR = OUT_DIR / "dataset"
VALIDATION_MONTH = ("2024-06-01", "2024-07-01")

KLIN_TABLES = {
    "sse": ("cmds.LOCAL_SSE_AL_KLIN_EXG", ".SH"),
    "szse": ("cmds.LOCAL_SZSE_AL_KLIN_CMD", ".SZ"),
}


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _module_sha256() -> str:
    import inspect

    return _text_sha256(inspect.getsource(lid))


def _environment() -> Dict[str, str]:
    import dolphindb as ddb

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "dolphindb": ddb.__version__,
        "platform": platform.platform(),
    }


def _git_head() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def quarter_ranges(start: str, end: str) -> List[Tuple[str, str, str]]:
    frame = pd.DataFrame(
        {"start": pd.date_range(start, end, freq="QS")}
    )
    if pd.Timestamp(start) < pd.Timestamp(frame["start"].iloc[0]):
        frame = pd.concat(
            [pd.DataFrame({"start": [pd.Timestamp(start)]}), frame]
        )
    frame["end"] = frame["start"].shift(-1).fillna(pd.Timestamp(end))
    rows = []
    for _, row in frame.iterrows():
        quarter = f"{row['start'].year}Q{row['start'].quarter}"
        rows.append(
            (
                quarter,
                row["start"].strftime("%Y-%m-%d"),
                row["end"].strftime("%Y-%m-%d"),
            )
        )
    return rows


def _quality_row(frame: pd.DataFrame, tag: str) -> Dict[str, object]:
    return {
        "partition": tag,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "actual_date_min": str(frame["TradeDate"].min().date()),
        "actual_date_max": str(frame["TradeDate"].max().date()),
        "coverage_mean": float(frame["coverage_ratio"].mean()),
        "coverage_median": float(frame["coverage_ratio"].median()),
        "low_coverage_rows": int(
            (frame["coverage_ratio"] < lid.COVERAGE_THRESHOLD).sum()
        ),
        "zero_amount_rows": int((frame["daily_amount"] <= 0).sum()),
        "impact_null_share": float(
            frame["signed_amount_impact"].isna().mean()
        ),
        "recovery_null_share": float(
            frame["spread_recovery_5m"].isna().mean()
        ),
    }


def _run_period(
    client, start: str, end: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    frames = []
    hashes = {}
    for exchange in lid.EXCHANGES:
        sql = lid.daily_sql(exchange, start, end)
        hashes[exchange] = lid.query_sha256(sql)
        frame = client.query_df(sql)
        frames.append(frame)
        print(
            f"[build] {exchange} {start}..{end} rows={len(frame)}",
            flush=True,
        )
    daily = lid.finalize_daily(frames, start=start, end=end)
    return lid.prepare_liquidity_impact_daily(daily), hashes


def _klin_amount_parity(
    client, daily: pd.DataFrame, sample_symbols: int = 20
) -> Dict[str, object]:
    """Cross-source sanity: tick-summed daily amount vs KLIN AccAmount."""
    rng = np.random.default_rng(7)
    candidates = daily.loc[daily["daily_amount"] > 0, "symbol"].unique()
    picks = list(
        rng.choice(candidates, size=min(sample_symbols, len(candidates)), replace=False)
    )
    rows = []
    for exchange, (table, suffix) in KLIN_TABLES.items():
        symbols = [p.replace(suffix, "") for p in picks if p.endswith(suffix)]
        if not symbols:
            continue
        in_list = ", ".join(f"'{s}'" for s in symbols)
        klin = client.query_df(
            f"""
            SELECT Symbol AS symbol_raw, toDate(ExchTime) AS TradeDate,
                sum(toFloat64(Amount)) AS klin_amount
            FROM {table}
            WHERE ExchTime >= toDateTime64('2024-06-01 00:00:00', 6,
                'Asia/Shanghai')
                AND ExchTime < toDateTime64('2024-07-01 00:00:00', 6,
                'Asia/Shanghai')
                AND Symbol IN ({in_list})
                AND Type = '1MIN'
            GROUP BY symbol_raw, TradeDate
            """
        )
        klin["symbol"] = klin["symbol_raw"] + suffix
        merged = daily.merge(
            klin[["symbol", "TradeDate", "klin_amount"]],
            on=["symbol", "TradeDate"],
            how="inner",
        )
        merged["rel_diff"] = (
            (merged["daily_amount"] - merged["klin_amount"]).abs()
            / merged["klin_amount"].clip(lower=1.0)
        )
        rows.append(merged)
    if not rows:
        return {"checked": 0, "median_rel_diff": np.nan, "p95_rel_diff": np.nan}
    frame = pd.concat(rows, ignore_index=True)
    return {
        "checked": int(len(frame)),
        "median_rel_diff": float(frame["rel_diff"].median()),
        "p95_rel_diff": float(frame["rel_diff"].quantile(0.95)),
        "max_rel_diff": float(frame["rel_diff"].max()),
    }


def _write_partition(frame: pd.DataFrame, quarter: str) -> Dict[str, object]:
    directory = DATASET_DIR / f"quarter={quarter}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"liquidity_impact_daily_{quarter}.parquet"
    frame.to_parquet(
        path, engine="pyarrow", compression="zstd", index=False
    )
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
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip the 2024-06 validation gate (must have passed before)",
    )
    args = parser.parse_args()

    client = connect_hf_client()
    quality_rows: List[Dict[str, object]] = []
    partitions: List[Dict[str, object]] = []

    # --- 1. validation month gate ---------------------------------------------
    validation_report: Dict[str, object] = {}
    validation_path = OUT_DIR / "validation_month_2024-06.json"
    if args.skip_validation and validation_path.exists():
        validation_report = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        if not validation_report.get("passed"):
            raise RuntimeError("stored validation month result is failing")
        print("[gate] stored 2024-06 validation reused", flush=True)
    else:
        print("[gate] 2024-06 validation month build", flush=True)
        daily, hashes = _run_period(client, *VALIDATION_MONTH)
        quality = _quality_row(daily, "validation_2024-06")
        parity = _klin_amount_parity(client, daily)
        trading_days = int(daily["TradeDate"].nunique())
        # KLIN daily amount includes auction bars (09:25/15:00) while the
        # frozen tick grid excludes them, so a systematic shortfall of a
        # few percent is expected; the gate bounds dispersion, not zero.
        passed = (
            trading_days >= 18
            and quality["coverage_mean"] > 0.85
            and quality["impact_null_share"] < 0.10
            and parity.get("median_rel_diff", 1.0) < 0.10
            and parity.get("p95_rel_diff", 1.0) < 0.15
        )
        validation_report = {
            "passed": bool(passed),
            "month": VALIDATION_MONTH,
            "trading_days": trading_days,
            "quality": quality,
            "klin_amount_parity": parity,
            "query_hashes": hashes,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            json.dumps(validation_report, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            f"[gate] validation passed={passed} parity={parity}",
            flush=True,
        )
        if not passed:
            raise RuntimeError(
                "2024-06 validation month failed; quarterly build aborted"
            )

    # --- 2. quarterly build -----------------------------------------------------
    for quarter, start, end in quarter_ranges(args.start, args.end):
        part_dir = DATASET_DIR / f"quarter={quarter}"
        part_file = (
            part_dir / f"liquidity_impact_daily_{quarter}.parquet"
        )
        if part_file.exists():
            print(f"[skip] {quarter} exists", flush=True)
            existing = pd.read_parquet(part_file)
            quality_rows.append(_quality_row(existing, quarter))
            partitions.append(
                {
                    "quarter": quarter,
                    "path": str(part_file.relative_to(PROJ_ROOT)),
                    "rows": int(len(existing)),
                    "sha256": _sha256(part_file),
                    "reused": True,
                }
            )
            continue
        daily, hashes = _run_period(client, start, end)
        quality_rows.append(_quality_row(daily, quarter))
        info = _write_partition(daily, quarter)
        info["query_sha256"] = hashes
        partitions.append(info)
        print(f"[done] {quarter} rows={len(daily)}", flush=True)

    # --- 3. manifest -------------------------------------------------------------
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT_DIR / "primitive_quality.csv", index=False)
    engines = client.query_df(
        "SELECT name, engine FROM system.tables WHERE database='cmds' "
        "AND name LIKE 'LOCAL_%AL_TICK_EXG' OR name LIKE 'LOCAL_%AL_SSL2_EXG'"
    )
    manifest = {
        "primitive_name": "l2_primitive_liquidity_impact_daily",
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "canonical_source": lid.CANONICAL_SOURCE,
        "source_tables": {
            exchange: {
                "tick": cfg["tick_table"],
                "book": cfg["book_table"],
                "symbol_filter": cfg["symbol_filter"],
                "trade_filter": cfg["trade_filter"],
            }
            for exchange, cfg in lid.EXCHANGES.items()
        },
        "table_engines": dict(zip(engines["name"], engines["engine"])),
        "distributed_join": "never; LOCAL MergeTree per-exchange queries",
        "date_coverage": {
            "requested_start": args.start,
            "requested_end": args.end,
            "actual_min": (
                str(quality["actual_date_min"].min())
                if len(quality)
                else None
            ),
            "actual_max": (
                str(quality["actual_date_max"].max())
                if len(quality)
                else None
            ),
        },
        "row_count": int(quality["rows"].sum()) if len(quality) else 0,
        "time_grid": {
            "sessions": ["09:30-11:29", "13:00-14:59"],
            "expected_continuous_minutes": lid.EXPECTED_CONTINUOUS_MINUTES,
            "auction_minutes_excluded": True,
        },
        "direction_rules": {
            "sse": "BSFlag 'B'/'S'/'N' on Type='T'",
            "szse": (
                "Type='011' AND Category='F'; BidOrderNo > AskOrderNo -> "
                "active buy, '<' -> active sell, '=' -> neutral"
            ),
            "neutral_recording": (
                "neutral_amount aggregated daily; neutral_trade_share field"
            ),
        },
        "amount_units": {
            "amount": "CNY (SSE Amount column; SZSE Price*Volume)",
            "volume": "shares",
            "depth": "shares summed over book levels 1-5",
            "prices": "unadjusted; returns within-day only",
        },
        "minute_join_rules": (
            "tick minute aggregate FULL OUTER JOIN minute-last book state "
            "on (symbol, TradeDate, minute); no-trade minutes keep zero "
            "trade fields; forward mid returns require exact consecutive "
            "minute keys inside the same day"
        ),
        "size_buckets_cny": lid.SIZE_BUCKETS,
        "high_impact_definition": (
            "abs(minute_return) >= per-symbol-day 90th percentile "
            "(frozen top-10%)"
        ),
        "proxy_limitations": [
            "effective_spread_proxy / realized_spread_proxy_5m are minute"
            " approximations (minute signed direction + minute-last"
            " midquote), not per-trade prevailing-quote spreads",
            "size-conditioned impacts bucket trades at tick level but"
            " measure response with minute forward mid returns",
            "buy/sell price impacts are amount-weighted mean forward mid"
            " returns, not per-trade measurements",
        ],
        "universe_limitation": (
            "ClickHouse L2 covers a documented symbol subset (~1.2k-1.8k"
            " names, growing over time); DolphinDB has no Tick/L2 source."
            " Uncovered names carry NaN factor values."
        ),
        "coverage_threshold": lid.COVERAGE_THRESHOLD,
        "validation_month": validation_report,
        "partition_checksums": partitions,
        "module_sha256": _module_sha256(),
        "query_hashes": {
            item["quarter"]: item.get("query_sha256", "reused")
            for item in partitions
        },
        "lineage": {"git_head": _git_head()},
        "environment": _environment(),
        "storage": {
            "format": "quarterly partitioned parquet",
            "compression": "zstd",
            "dataset_path": str(DATASET_DIR.relative_to(PROJ_ROOT)),
            "combined_parquet_written": False,
            "raw_minute_panel_written": False,
        },
        "benchmark_definition": {
            "benchmark": "000852.SH",
            "return": "benchmark-relative daily close-to-close",
            "cost_bps": 7.5,
            "signal_shift": 1,
        },
        "direction_policy": {
            "raw_ic": "frozen formula direction",
            "effective_direction": "display grouping only",
            "production_direction": "not decided in Sprint 7",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[done] primitive manifest -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
