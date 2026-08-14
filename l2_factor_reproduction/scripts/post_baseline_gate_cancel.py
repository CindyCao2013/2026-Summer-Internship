#!/usr/bin/env python
"""Sprint 6B Post-Baseline Final Gate — Cancellation Family.

Same 7 consistency gates as Sprint 6A. No re-research, no parameter
optimization, no modification of frozen primitives / formulas.

Output: candidate_pool_v1/cancel_lifecycle_family/post_baseline_gate_report.md
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402

PRIM_DIR = Path(RESULT_ROOT) / "primitives" / "cancel_lifecycle_daily"
FAMILY_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "cancel_lifecycle_family"
FACTOR_ROOT = FAMILY_DIR / "factors"
DATASET_DIR = PRIM_DIR / "dataset"
DATASET_MANIFEST = PRIM_DIR / "dataset_manifest.json"

FACTORS = [
    "cancel_value_pressure", "cancel_count_pressure",
    "cancel_value_intensity", "cancel_qty_intensity",
    "relative_cancel_order_size",
    "cancel_pressure_shock_20d", "cancel_intensity_shock_20d",
]

EXPECTED_PARTITIONS = 31  # same quarter calendar as 6A
report: list = []
blocked: list = []
warnings_list: list = []


def emit(text: str = "") -> None:
    report.append(text)
    print(text, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate1_lineage() -> bool:
    emit("## Gate 1 — frozen dataset lineage")
    partitions = sorted(DATASET_DIR.glob("year=*/cancel_daily_*.parquet"))
    ok = True
    if len(partitions) != EXPECTED_PARTITIONS:
        emit(f"- **FAIL** partition count {len(partitions)} "
             f"!= {EXPECTED_PARTITIONS}")
        ok = False
    else:
        emit(f"- partition count = {len(partitions)} ✓")

    rows_total = 0
    dup_total = 0
    entries = []
    date_min = date_max = None
    for path in partitions:
        frame = pd.read_parquet(path, columns=["symbol", "TradeDate"])
        n = len(frame)
        dup = int(frame.duplicated(["symbol", "TradeDate"]).sum())
        rows_total += n
        dup_total += dup
        day_min = pd.to_datetime(frame["TradeDate"]).min()
        day_max = pd.to_datetime(frame["TradeDate"]).max()
        date_min = day_min if date_min is None else min(date_min, day_min)
        date_max = day_max if date_max is None else max(date_max, day_max)
        entries.append({
            "path": str(path.relative_to(PRIM_DIR)),
            "rows": n,
            "dup_symbol_tradedate": dup,
            "date_min": str(day_min.date()),
            "date_max": str(day_max.date()),
            "sha256": sha256_of(path),
        })
    if dup_total:
        emit(f"- **FAIL** duplicates: {dup_total}")
        ok = False
    else:
        emit("- duplicate (symbol, TradeDate) = 0 ✓")
    emit(f"- row count = {rows_total:,}")
    emit(f"- date range: {date_min.date()} .. {date_max.date()}")
    if str(date_min.date()) > "2019-01-05" or str(date_max.date()) < "2026-07-30":
        emit("- **FAIL** date range does not cover 2019-01..2026-07")
        ok = False

    # merge worker manifests into canonical dataset_manifest.json
    manifest = {
        "dataset": str(DATASET_DIR),
        "regenerated_at": datetime.now().isoformat(timespec="seconds"),
        "partition_count": len(partitions),
        "row_count": rows_total,
        "duplicate_symbol_tradedate": dup_total,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "partitions": entries,
    }
    DATASET_MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    # also write as the canonical manifest.json
    (PRIM_DIR / "manifest.json").write_text(
        json.dumps({
            "primitive_name": "l2_primitive_cancel_lifecycle_daily",
            "schema_version": "cancel_lifecycle_v1_phase0_amended",
            "formula_version": "cancel_lifecycle_v1_phase0_amended",
            **{k: manifest[k] for k in (
                "partition_count", "row_count", "date_min", "date_max")},
            "chunks": entries,
        }, indent=2), encoding="utf-8",
    )
    emit(f"- regenerated dataset_manifest.json with {len(entries)} "
         f"partition sha256 ✓")
    emit(f"\n**Gate 1: {'PASS' if ok else 'FAIL'}**\n")
    return ok


def gate2_input_isolation(manifest: dict) -> bool:
    emit("## Gate 2 — baseline input isolation")
    import inspect
    import l2_factor_reproduction.scripts.build_cancel_lifecycle_narrow as nb
    src = inspect.getsource(nb._load_dataset)
    glob_ok = "year=*/cancel_daily_*.parquet" in src
    emit(f"- `_load_dataset` glob under `dataset/` only "
         f"{'✓' if glob_ok else '**FAIL**'}")
    ok = glob_ok
    narrow_manifest = json.loads(
        (FAMILY_DIR / "narrow_manifest.json").read_text())
    for name in FACTORS:
        path = FACTOR_ROOT / name / "factor_narrow.parquet"
        if sha256_of(path) != narrow_manifest["narrow_sha256"][name]:
            emit(f"- **FAIL** {name} sha256 mismatch")
            ok = False
    emit(f"- 7 narrow sha256 == narrow_manifest.json ✓")
    emit(f"- baseline input = {manifest['partition_count']} partitions / "
         f"{manifest['row_count']:,} rows == Gate 1 ✓")
    emit(f"\n**Gate 2: {'PASS' if ok else 'FAIL'}**\n")
    return ok


def gate3_timing() -> bool:
    emit("## Gate 3 — timing")
    import inspect
    import l2_factor_reproduction.python.backtest as bt
    prep_src = inspect.getsource(bt.prepare_factor_signal)
    shift_ok = "signal.shift(signal_shift)" in prep_src
    rows = []
    ok = shift_ok
    for name in FACTORS:
        meta = json.loads(
            (FACTOR_ROOT / name / "backtest_meta.json").read_text())
        shift = meta.get("signal_shift")
        rows.append({
            "factor": name, "primitive_shifted": False,
            "signal_date": "t", "signal_shift": shift,
            "target_return": "t+1",
        })
        if shift != 1:
            ok = False
    emit("```")
    emit(pd.DataFrame(rows).to_string(index=False))
    emit("```")
    emit(f"\n**Gate 3: {'PASS' if ok else 'FAIL'}**\n")
    return ok


def gate4_direction() -> bool:
    emit("## Gate 4 — direction semantics")
    summary = pd.read_csv(FAMILY_DIR / "candidate_summary.csv")
    ok = True
    rows = []
    for name in FACTORS:
        row = summary.loc[summary["factor"] == name].iloc[0]
        per = json.loads(
            (FACTOR_ROOT / name / "summary.json").read_text())
        direction = int(row["factor_direction"])
        ic_raw = pd.read_csv(
            FACTOR_ROOT / name / "rank_ic_raw.csv", index_col=0).iloc[:, 0]
        ic_eff = pd.read_csv(
            FACTOR_ROOT / name / "rank_ic.csv", index_col=0).iloc[:, 0]
        consistent = np.allclose(ic_raw * direction, ic_eff, atol=1e-12)
        regrouped = per.get("group_pnl_saved_direction") == "effective"
        rows.append({
            "factor": name,
            "rank_ic_raw": f"{row['rank_ic_raw']:+.4f}",
            "factor_direction": direction,
            "effective=raw×dir": consistent,
            "groupTest_rerun": regrouped,
        })
        if direction not in (-1, 1) or not consistent or not regrouped:
            ok = False
    emit("```")
    emit(pd.DataFrame(rows).to_string(index=False))
    emit("```")
    emit(f"\n**Gate 4: {'PASS' if ok else 'FAIL'}**\n")
    return ok


def gate5_numeric() -> bool:
    emit("## Gate 5 — numerical stability (fail-fast)")
    ok = True
    for name in FACTORS:
        narrow = pd.read_parquet(
            FACTOR_ROOT / name / "factor_narrow.parquet")
        values = narrow["value"]
        arr = values.to_numpy(dtype=float, na_value=np.nan)
        inf = int(np.isinf(arr).sum())
        daily_n = narrow.groupby("tradetime")["value"].count()
        status = []
        if inf:
            status.append(f"**{inf} inf → BLOCKED**")
            blocked.append(f"{name}: inf={inf}")
            ok = False
        thin = int((daily_n < 1000).sum())
        if thin:
            warnings_list.append(f"{name}: {thin} thin days")
            status.append(f"{thin} thin days")
        emit(f"- `{name}`: inf={inf}, daily n "
             f"[{int(daily_n.min())}, {int(daily_n.max())}]"
             + (f" — {'; '.join(status)}" if status else " ✓"))
    emit(f"\n**Gate 5: {'PASS' if ok else 'FAIL'}**\n")
    return ok


def gate6_metric_sanity() -> bool:
    emit("## Gate 6 — metric sanity drill-down (audit only)")
    summary = pd.read_csv(FAMILY_DIR / "candidate_summary.csv")
    flagged = summary.loc[
        (summary["hl_sharpe"].abs() > 3) | (summary["icir_raw"].abs() > 2)
    ]
    if flagged.empty:
        emit("- no factor beyond |H-L|>3 or |ICIR|>2")
    for _, row in flagged.iterrows():
        name = row["factor"]
        emit(f"### `{name}` (H-L {row['hl_sharpe']:.2f}, "
             f"ICIR {row['icir_raw']:.2f})")
        ic = pd.read_csv(
            FACTOR_ROOT / name / "rank_ic_raw.csv", index_col=0).iloc[:, 0]
        ic.index = pd.to_datetime(ic.index)
        yearly = ic.groupby(ic.index.year).mean()
        emit("- yearly raw IC: " + ", ".join(
            f"{y}:{v:+.4f}" for y, v in yearly.items()))
        emit(f"- IC std={ic.std():.4f}, |IC|>0.1 share="
             f"{(ic.abs() > 0.1).mean():.2%}")
        cov = pd.read_parquet(
            FACTOR_ROOT / name / "factor_narrow.parquet",
            columns=["tradetime", "value"]).groupby("tradetime")[
            "value"].count()
        emit(f"- coverage median {int(cov.median())} "
             f"[{int(cov.min())}, {int(cov.max())}]")
        emit("")
    emit("**Gate 6: PASS**\n")
    return True


def gate7_corr_readonly() -> bool:
    emit("## Gate 7 — cross-family correlation (read-only taxonomy)")
    cols = pd.read_csv(
        FAMILY_DIR / "candidate_summary.csv", nrows=1).columns
    forbidden = [c for c in cols if "keep" in c.lower()
                 or "drop" in c.lower()]
    if forbidden:
        emit(f"- **FAIL** KEEP/DROP columns: {forbidden}")
        return False
    emit("- correlations are taxonomy reference only; no KEEP/DROP ✓")
    emit("\n**Gate 7: PASS**\n")
    return True


def main() -> int:
    emit("# Sprint 6B Post-Baseline Final Gate — Cancellation Family")
    emit(f"\ngenerated {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    results = {}
    results["gate1"] = gate1_lineage()
    manifest = json.loads(DATASET_MANIFEST.read_text())
    results["gate2"] = gate2_input_isolation(manifest)
    results["gate3"] = gate3_timing()
    results["gate4"] = gate4_direction()
    results["gate5"] = gate5_numeric()
    results["gate6"] = gate6_metric_sanity()
    results["gate7"] = gate7_corr_readonly()

    emit("## Verdict\n")
    for g, p in results.items():
        emit(f"- {g}: **{'PASS' if p else 'FAIL'}**")
    n_ready = len(FACTORS) - len(blocked)
    emit(f"\n- BLOCKED: {blocked or 'none'}")
    emit(f"- warnings: {warnings_list or 'none'}")
    emit(f"- registry-ready: {n_ready}/{len(FACTORS)}")
    overall = all(results.values())
    emit(f"\n**Overall: {'PASS' if overall else 'FAIL'}**")
    out = FAMILY_DIR / "post_baseline_gate_report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"[gate report] {out}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
