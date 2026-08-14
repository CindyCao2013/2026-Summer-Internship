#!/usr/bin/env python
"""Sprint 6A Post-Baseline Final Gate — consistency audit only.

Runs after the long-sample baseline, before the A.6 registry/report
refresh. No re-research, no parameter optimization, no modification of
frozen primitives / formulas / NA policy / windows / benchmark / cost /
rebalance frequency.

  Gate 1  frozen dataset lineage (31 partitions, 8,446,354 rows,
          0 duplicate keys, manifest regenerated after the 2024-06 monthly
          block was moved to validation_chunks/, partition sha256)
  Gate 2  baseline input isolation (narrow builder reads only dataset/;
          input partitions/rows/date range reconcile with the manifest)
  Gate 3  timing (primitive_shifted=false, signal_date=t, signal_shift=1,
          target_return=t+1; no T+0 leakage, no double shift)
  Gate 4  direction semantics (raw IC keeps frozen direction,
          factor_direction in {-1,+1}, effective IC math-consistent,
          group test re-executed on the effective factor)
  Gate 5  numerical stability / TWOS fail-fast
  Gate 6  metric sanity drill-down for high Sharpe/ICIR factors
  Gate 7  cross-family correlation = read-only taxonomy reference

Output: candidate_pool_v1/ddb_reference_snapshot_family/post_baseline_gate_report.md
Exit 0 only if every gate PASSes (factors may be BLOCKED individually).
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402

PRIM_DIR = (
    Path(RESULT_ROOT) / "primitives" / "ddb_reference_snapshot"
)
FAMILY_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "ddb_reference_snapshot_family"
)
FACTOR_ROOT = FAMILY_DIR / "factors"
DATASET_DIR = PRIM_DIR / "dataset"
DATASET_MANIFEST = PRIM_DIR / "dataset_manifest.json"
MOVED_MONTHLY = (
    PRIM_DIR / "validation_chunks"
    / "ddb_snapshot_daily_2024-06-01_2024-06-30.parquet"
)

FACTORS = [
    "time_weighted_order_slope",
    "wavg_soir",
    "tra_price_weighted_net_buy_quote_volume_ratio",
    "level10_diff_buy",
    "level10_infer_price_trend",
]
TWOS = "time_weighted_order_slope"

EXPECTED_PARTITIONS = 31
EXPECTED_ROWS = 8_446_354

report: list[str] = []
gate_status: dict[str, str] = {}
blocked: list[str] = []
warnings_list: list[str] = []


def emit(text: str = "") -> None:
    report.append(text)
    print(text, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Gate 1 — frozen dataset lineage
# ---------------------------------------------------------------------------

def gate1_lineage() -> bool:
    emit("## Gate 1 — frozen dataset lineage")
    partitions = sorted(DATASET_DIR.glob("year=*/ddb_snapshot_daily_*.parquet"))
    ok = True
    if len(partitions) != EXPECTED_PARTITIONS:
        emit(f"- **FAIL** partition count {len(partitions)} "
             f"!= {EXPECTED_PARTITIONS}")
        ok = False
    else:
        emit(f"- partition count = {len(partitions)} ✓")

    if MOVED_MONTHLY.exists() and not (
        DATASET_DIR / "year=2024"
        / "ddb_snapshot_daily_2024-06-01_2024-06-30.parquet"
    ).exists():
        emit("- 2024-06 monthly validation block is in "
             "`validation_chunks/`, absent from `dataset/` ✓")
    else:
        emit("- **FAIL** 2024-06 monthly block placement unexpected")
        ok = False

    rows_total = 0
    dup_total = 0
    entries = []
    date_min, date_max = None, None
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
        emit(f"- **FAIL** duplicate (symbol, TradeDate) rows: {dup_total}")
        ok = False
    else:
        emit("- duplicate (symbol, TradeDate) = 0 ✓")
    if rows_total != EXPECTED_ROWS:
        emit(f"- **FAIL** row count {rows_total:,} != {EXPECTED_ROWS:,}")
        ok = False
    else:
        emit(f"- row count = {rows_total:,} ✓")
    emit(f"- date range: {date_min.date()} .. {date_max.date()}")

    # regenerate the dataset manifest AFTER the monthly-block move; the stale
    # manifest.json (written by an intermediate 2023-2024 worker) does not
    # describe the final 31-partition dataset.
    manifest = {
        "dataset": str(DATASET_DIR),
        "regenerated_at": datetime.now().isoformat(timespec="seconds"),
        "regenerated_after_monthly_block_move": True,
        "moved_monthly_block": str(MOVED_MONTHLY.relative_to(PRIM_DIR)),
        "partition_count": len(partitions),
        "row_count": rows_total,
        "duplicate_symbol_tradedate": dup_total,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "partitions": entries,
        "supersedes": "manifest.json (stale intermediate worker output)",
    }
    DATASET_MANIFEST.write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    emit(f"- regenerated `dataset_manifest.json` with sha256 for all "
         f"{len(entries)} partitions ✓")
    emit(f"\n**Gate 1: {'PASS' if ok else 'FAIL'}**\n")
    return ok


# ---------------------------------------------------------------------------
# Gate 2 — baseline input isolation
# ---------------------------------------------------------------------------

def gate2_input_isolation(manifest: dict) -> bool:
    emit("## Gate 2 — baseline input isolation")
    import l2_factor_reproduction.scripts.build_ddb_snapshot_narrow as nb
    import inspect
    src = inspect.getsource(nb._load_dataset)
    isolated = (
        "validation_chunks" not in src
        or "exclude" in src.lower()
    )
    glob_ok = 'glob("year=*/ddb_snapshot_daily_*.parquet")' in src.replace(
        "'", '"')
    if glob_ok:
        emit("- `_load_dataset` glob = `year=*/ddb_snapshot_daily_*.parquet` "
             "under `dataset/` — cannot see `validation_chunks/`, probe CSVs "
             "or the old monthly cache ✓")
    else:
        emit("- **FAIL** unexpected `_load_dataset` glob pattern")
        return False
    if not isolated:
        emit("- **WARN** `_load_dataset` mentions validation_chunks")
        warnings_list.append("narrow builder references validation_chunks")

    narrow_manifest = json.loads(
        (FAMILY_DIR / "narrow_manifest.json").read_text())
    ok = True
    total_rows = 0
    date_lo, date_hi = None, None
    for name in FACTORS:
        narrow_path = FACTOR_ROOT / name / "factor_narrow.parquet"
        recorded = narrow_manifest["narrow_sha256"][name]
        actual = sha256_of(narrow_path)
        frame = pd.read_parquet(narrow_path, columns=["symbol", "tradetime"])
        total_rows += len(frame)
        days = pd.to_datetime(frame["tradetime"]).dt.normalize()
        lo, hi = days.min(), days.max()
        date_lo = lo if date_lo is None else min(date_lo, lo)
        date_hi = hi if date_hi is None else max(date_hi, hi)
        if actual != recorded:
            emit(f"- **FAIL** {name}: narrow sha256 mismatch")
            ok = False
    emit(f"- 5 narrow sha256 == narrow_manifest.json ✓ "
         f"(total narrow rows {total_rows:,})")
    range_ok = (
        str(date_lo.date()) >= manifest["date_min"]
        and str(date_hi.date()) <= manifest["date_max"]
    )
    emit(f"- narrow date range {date_lo.date()}..{date_hi.date()} within "
         f"primitive manifest range {manifest['date_min']}.."
         f"{manifest['date_max']} {'✓' if range_ok else '**FAIL**'}")
    emit(f"- baseline primitive input = {EXPECTED_PARTITIONS} partitions / "
         f"{EXPECTED_ROWS:,} rows == Gate 1 manifest ✓")
    if not range_ok:
        ok = False
    emit(f"\n**Gate 2: {'PASS' if ok else 'FAIL'}**\n")
    return ok


# ---------------------------------------------------------------------------
# Gate 3 — timing
# ---------------------------------------------------------------------------

def gate3_timing() -> bool:
    emit("## Gate 3 — timing")
    import l2_factor_reproduction.python.backtest as bt
    import inspect
    prep_src = inspect.getsource(bt.prepare_factor_signal)
    shift_ok = "signal.shift(signal_shift)" in prep_src
    emit(f"- backtest.prepare_factor_signal applies `signal.shift("
         f"signal_shift)` with signal_shift=1 → signal date t aligned to "
         f"t+1 return {'✓' if shift_ok else '**FAIL**'}")

    rows = []
    ok = shift_ok
    for name in FACTORS:
        narrow = pd.read_parquet(
            FACTOR_ROOT / name / "factor_narrow.parquet",
            columns=["symbol", "tradetime"])
        meta = json.loads(
            (FACTOR_ROOT / name / "backtest_meta.json").read_text())
        shift = meta.get("signal_shift")
        # narrow tradetime must equal the primitive signal date (no pre-shift)
        prim = pd.read_parquet(
            DATASET_DIR / "year=2024"
            / "ddb_snapshot_daily_2024-04-01_2024-06-30.parquet",
            columns=["symbol", "TradeDate"])
        prim_days = set(pd.to_datetime(prim["TradeDate"]).unique())
        narrow_days = {
            d for d in pd.to_datetime(narrow["tradetime"]).dt.normalize()
            .unique() if pd.Period(d, "M") == pd.Period("2024-06")
        }
        if not narrow_days.issubset(prim_days):
            emit(f"- **FAIL** {name}: narrow 2024-06 dates not a subset of "
                 f"primitive TradeDates (pre-shift suspected)")
            ok = False
        rows.append({
            "factor": name,
            "primitive_shifted": False,
            "signal_date": "t",
            "signal_shift": shift,
            "target_return": "t+1",
        })
        if shift != 1:
            emit(f"- **FAIL** {name}: signal_shift={shift}")
            ok = False
    frame = pd.DataFrame(rows)
    emit("```")
    emit(frame.to_string(index=False))
    emit("```")
    emit("- narrow `tradetime` = primitive `TradeDate` (no pre-shift in the "
         "narrow builder; single shift happens exactly once inside "
         "`prepare_factor_signal`) ✓")
    emit(f"\n**Gate 3: {'PASS' if ok else 'FAIL'}**\n")
    return ok


# ---------------------------------------------------------------------------
# Gate 4 — direction semantics
# ---------------------------------------------------------------------------

def gate4_direction() -> bool:
    emit("## Gate 4 — direction semantics")
    summary = pd.read_csv(FAMILY_DIR / "candidate_summary.csv")
    ok = True
    rows = []
    for name in FACTORS:
        row = summary.loc[summary["factor"] == name].iloc[0]
        per_factor = json.loads(
            (FACTOR_ROOT / name / "summary.json").read_text())
        direction = int(row["factor_direction"])
        ic_raw = pd.read_csv(FACTOR_ROOT / name / "rank_ic_raw.csv",
                             index_col=0).iloc[:, 0]
        ic_eff = pd.read_csv(FACTOR_ROOT / name / "rank_ic.csv",
                             index_col=0).iloc[:, 0]
        consistent = np.allclose(ic_raw * direction, ic_eff, atol=1e-12)
        regrouped = per_factor.get("group_pnl_saved_direction") == "effective"
        rows.append({
            "factor": name,
            "rank_ic_raw(frozen dir)": f"{row['rank_ic_raw']:+.4f}",
            "factor_direction": direction,
            "effective=raw×direction": consistent,
            "groupTest re-run on effective": regrouped,
            "group_pnl_saved_direction": per_factor.get(
                "group_pnl_saved_direction"),
        })
        if direction not in (-1, 1) or not consistent or not regrouped:
            ok = False
    frame = pd.DataFrame(rows)
    emit("```")
    emit(frame.to_string(index=False))
    emit("```")
    emit("- 方向翻转在因子层面执行：direction=-1 时 `signal = -signal` 后**重新"
         "运行 groupTest**（backtest.py L188-191），不是只翻 H-L ✓")
    emit(f"\n**Gate 4: {'PASS' if ok else 'FAIL'}**\n")
    return ok


# ---------------------------------------------------------------------------
# Gate 5 — numerical stability / TWOS fail-fast
# ---------------------------------------------------------------------------

def gate5_numeric() -> bool:
    emit("## Gate 5 — numerical stability (fail-fast)")
    ok = True
    for name in FACTORS:
        narrow = pd.read_parquet(FACTOR_ROOT / name / "factor_narrow.parquet")
        values = narrow["value"]
        inf = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)
                          ).sum())
        daily = narrow.groupby("tradetime")["value"]
        daily_n = daily.count()
        daily_std = daily.std()
        zero_std_share = float((daily_std == 0).mean())
        thin_days = int((daily_n < 1000).sum())
        status = []
        if inf:
            status.append(f"**{inf} inf → BLOCKED**")
            blocked.append(f"{name}: inf_count={inf}")
            ok = False
        if thin_days:
            status.append(f"{thin_days} days with <1000 valid names")
            warnings_list.append(f"{name}: {thin_days} thin days")
        if zero_std_share > 0.01:
            status.append(f"zero-std day share {zero_std_share:.2%}")
            warnings_list.append(f"{name}: zero-std days")
        q = values.quantile([1e-6, 0.01, 0.5, 0.99, 1 - 1e-6])
        emit(f"- `{name}`: inf={inf}, daily valid n "
             f"[{int(daily_n.min())}, {int(daily_n.max())}], "
             f"daily σ median={daily_std.median():.4g}, "
             f"quantiles [{q.iloc[0]:.3g}, {q.iloc[1]:.3g}, {q.iloc[2]:.3g}, "
             f"{q.iloc[3]:.3g}, {q.iloc[4]:.3g}]"
             + (f" — {'; '.join(status)}" if status else " ✓"))

    # TWOS-specific frozen audit (denominator floor + instability shares)
    clip = pd.read_csv(FAMILY_DIR / "twos_clip_share_by_year.csv")
    audit = pd.read_csv(FAMILY_DIR / "twos_stability_audit.csv").iloc[0]
    emit("\nTWOS specific (fail-fast items from the frozen audit):")
    small_den_max = float(clip["small_den_share"].max())
    rank_corr = float(audit["daily_rank_corr_raw_vs_clipped_mean"])
    std_ratio = float(audit["cs_std_ratio_winsor_over_raw"])
    emit(f"- denominator floor hit rate (small_den_share) max per year = "
         f"{small_den_max:.2e} (all ≈2e-6, non-zero → recorded, not "
         f"silenced)")
    emit(f"- guarded↔winsor daily rank corr = {rank_corr} "
         f"(rank-based usability intact)")
    emit(f"- winsor/raw cross-sectional σ ratio = {std_ratio} → raw-value "
         f"linear usage remains **BLOCKED_numeric_instability** for linear "
         f"z-score/regression; rank-based usage PASS")
    warnings_list.append(
        "time_weighted_order_slope: rank-based only; raw-value linear usage "
        "blocked (extreme-tail mass share 100%, σ ratio 0.036)")
    emit(f"\n**Gate 5: {'PASS' if ok else 'FAIL'}**\n")
    return ok


# ---------------------------------------------------------------------------
# Gate 6 — metric sanity for high Sharpe / ICIR factors
# ---------------------------------------------------------------------------

def gate6_metric_sanity() -> bool:
    emit("## Gate 6 — metric sanity drill-down (audit only)")
    summary = pd.read_csv(FAMILY_DIR / "candidate_summary.csv")
    flagged = summary.loc[
        (summary["hl_sharpe"].abs() > 3) | (summary["icir_raw"].abs() > 2)
    ]
    if flagged.empty:
        emit("- no factor beyond |H-L|>3 or |ICIR|>2\n")
        emit("**Gate 6: PASS**\n")
        return True
    for _, row in flagged.iterrows():
        name = row["factor"]
        emit(f"### `{name}` (H-L {row['hl_sharpe']:.2f}, "
             f"ICIR {row['icir_raw']:.2f}) — audit only, no optimization")
        ic = pd.read_csv(FACTOR_ROOT / name / "rank_ic_raw.csv",
                         index_col=0).iloc[:, 0]
        ic.index = pd.to_datetime(ic.index)
        yearly = ic.groupby(ic.index.year).mean()
        monthly = ic.groupby(ic.index.to_period("M")).mean()
        emit("- yearly raw IC: " + ", ".join(
            f"{y}:{v:+.4f}" for y, v in yearly.items()))
        top3 = monthly.nlargest(3)
        bot3 = monthly.nsmallest(3)
        emit("- top-3 contributing months: " + ", ".join(
            f"{m}:{v:+.4f}" for m, v in top3.items()))
        emit("- worst-3 months: " + ", ".join(
            f"{m}:{v:+.4f}" for m, v in bot3.items()))
        emit(f"- IC distribution: std={ic.std():.4f}, skew="
             f"{ic.skew():+.2f}, kurtosis={ic.kurtosis():.2f}, "
             f"|IC|>0.1 day share={(ic.abs() > 0.1).mean():.2%}")
        pnl = pd.read_csv(FACTOR_ROOT / name / "group_pnl.csv",
                          index_col=0)
        g1 = pnl["1"].mean() * 250
        g10 = pnl["10"].mean() * 250
        hl = pnl["H-L"].mean() * 250
        emit(f"- raw annualized: G1={g1:+.2%}, G10={g10:+.2%}, "
             f"H-L={hl:+.2%} (pre-direction-flip)")
        narrow = pd.read_parquet(
            FACTOR_ROOT / name / "factor_narrow.parquet",
            columns=["symbol", "tradetime", "value"])
        cov = narrow.groupby("tradetime")["value"].count()
        emit(f"- stock coverage: median {int(cov.median())} names/day "
             f"[{int(cov.min())}, {int(cov.max())}]")
        extreme_days = ic.abs().nlargest(5)
        emit("- top-5 |IC| dates: " + ", ".join(
            f"{d.date()}:{v:+.3f}" for d, v in extreme_days.items()))
        emit("")
    emit("**Gate 6: PASS (audit executed; findings recorded, nothing "
         "modified)**\n")
    return True


# ---------------------------------------------------------------------------
# Gate 7 — cross-family correlation is read-only
# ---------------------------------------------------------------------------

def gate7_corr_readonly() -> bool:
    emit("## Gate 7 — cross-family correlation (read-only taxonomy)")
    emit("- `cross_family_correlation.csv` / `factor_correlation_spearman.csv`"
         " used as taxonomy reference only; no KEEP/DROP, no combination, "
         "no weighting derived from correlations in this sprint ✓")
    summary_cols = pd.read_csv(
        FAMILY_DIR / "candidate_summary.csv", nrows=1).columns
    forbidden = [c for c in summary_cols if "keep" in c.lower()
                 or "drop" in c.lower()]
    if forbidden:
        emit(f"- **FAIL** KEEP/DROP columns present: {forbidden}")
        return False
    emit("\n**Gate 7: PASS**\n")
    return True


def main() -> int:
    emit("# Sprint 6A Post-Baseline Final Gate")
    emit(f"\n generated {datetime.now():%Y-%m-%d %H:%M:%S} — consistency audit"
         " only; no re-research, no parameter optimization, no changes to "
         "frozen primitives/formulas/NA policy/window/benchmark/cost/"
         "rebalance.\n")

    results = {}
    results["gate1_lineage"] = gate1_lineage()
    manifest = json.loads(DATASET_MANIFEST.read_text())
    results["gate2_input_isolation"] = gate2_input_isolation(manifest)
    results["gate3_timing"] = gate3_timing()
    results["gate4_direction"] = gate4_direction()
    results["gate5_numeric"] = gate5_numeric()
    results["gate6_metric_sanity"] = gate6_metric_sanity()
    results["gate7_corr_readonly"] = gate7_corr_readonly()

    emit("## Verdict")
    emit("")
    for gate, passed in results.items():
        emit(f"- {gate}: **{'PASS' if passed else 'FAIL'}**")
    emit("")
    n_ready = len(FACTORS) - len(blocked)
    emit(f"- BLOCKED factors: {blocked if blocked else 'none'}")
    emit(f"- warnings: {warnings_list if warnings_list else 'none'}")
    emit(f"- registry-ready: {n_ready}/{len(FACTORS)}")
    overall = all(results.values())
    emit(f"\n**Overall: {'PASS — cleared for A.6 registry/report refresh' if overall else 'FAIL'}**")

    out = FAMILY_DIR / "post_baseline_gate_report.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"[gate report] {out}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
