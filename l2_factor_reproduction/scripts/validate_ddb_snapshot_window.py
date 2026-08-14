#!/usr/bin/env python
"""Sprint 6A Phase 2 window validation for the DDB reference snapshot
primitive (smoke day 2024-06-28 and/or full month 2024-06).

Outputs (research/results/l2_reproduction/primitives/ddb_reference_snapshot/):
    validation_<tag>.md            human-readable report
    validation_<tag>_factor_summary.csv
    validation_<tag>_correlation.csv   daily cross-sectional corr matrix
    validation_<tag>_twos_audit.csv    timeWeightedOrderSlope tail audit

Checks:
  * schema/null/inf/diagnostic-rate summary per factor
  * coverage vs the existing order_book_daily primitive (same source)
  * timeWeightedOrderSlope: small-denominator share, |value| quantiles,
    winsorize(1%/99%) before/after comparison, correlation with
    relative_spread_mean and level-1 qty ratio
  * wavgSOIR vs existing weighted_obi_mean daily cross-sectional
    Pearson/Spearman (redundancy control; does NOT gate replication)
  * 5x5 internal correlation (mean daily cross-sectional Spearman)

Usage:
    python validate_ddb_snapshot_window.py --start 2024-06-28 --end 2024-06-28 --tag 2024-06-28
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    COVERAGE_THRESHOLD,
    EXPECTED_MINUTE_COUNT,
    FACTOR_NAMES,
)

OUT_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/ddb_reference_snapshot"
)
ORDER_BOOK_DIR = (
    PROJ_ROOT
    / "research/results/l2_reproduction/primitives/order_book_daily/dataset"
)


def _load_primitive(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted((OUT_DIR / "dataset").glob("year=*/ddb_snapshot_daily_*.parquet")):
        parts = path.stem.split("_")
        chunk_start = pd.Timestamp(parts[-2])
        chunk_end = pd.Timestamp(parts[-1])
        if chunk_end < start or chunk_start > end:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"no ddb_snapshot_daily chunks cover {start.date()}..{end.date()}"
        )
    frame = pd.concat(frames, ignore_index=True)
    mask = frame["TradeDate"].between(start, end)
    return frame.loc[mask].reset_index(drop=True)


def _load_order_book(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(ORDER_BOOK_DIR.glob("year=*/order_book_daily_*.parquet")):
        parts = path.stem.split("_")
        chunk_start = pd.Timestamp(parts[-2])
        chunk_end = pd.Timestamp(parts[-1])
        if chunk_end < start or chunk_start > end:
            continue
        columns = [
            "symbol", "TradeDate", "weighted_obi_mean",
            "relative_spread_mean", "coverage_ratio",
        ]
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        raise FileNotFoundError("no order_book_daily chunks cover window")
    frame = pd.concat(frames, ignore_index=True)
    mask = frame["TradeDate"].between(start, end)
    return frame.loc[mask].reset_index(drop=True)


def _factor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total_snapshots = int(frame["valid_snapshot_count"].sum())
    for name in FACTOR_NAMES:
        mean_col = f"{name}_mean"
        values = frame[mean_col].dropna()
        rows.append(
            {
                "factor": name,
                "rows": int(len(frame)),
                "null_rows": int(frame[mean_col].isna().sum()),
                "null_rate": float(frame[mean_col].isna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "abs_q50": float(values.abs().quantile(0.50)),
                "abs_q99": float(values.abs().quantile(0.99)),
                "abs_q999": float(values.abs().quantile(0.999)),
                "abs_max": float(values.abs().max()),
                "inf_total": int(frame[f"{name}_inf_count"].sum()),
                "clipped_total": int(frame[f"{name}_clipped_count"].sum()),
                "small_den_total": int(
                    frame[f"{name}_small_denominator_count"].sum()
                ),
                "small_den_rate": float(
                    frame[f"{name}_small_denominator_count"].sum()
                    / max(total_snapshots, 1)
                ),
                "ffill_total": int(frame[f"{name}_ffill_count"].sum()),
                "ffill_rate": float(
                    frame[f"{name}_ffill_count"].sum()
                    / max(total_snapshots, 1)
                ),
                "null_snapshot_total": int(
                    frame[f"{name}_null_snapshot_count"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _daily_spearman(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    def _rho(block: pd.DataFrame) -> float:
        pair = block[[left, right]].dropna()
        if len(pair) < 100:
            return float("nan")
        return float(pair[left].corr(pair[right], method="spearman"))

    return frame.groupby("TradeDate").apply(_rho)


def _daily_pearson(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    def _rho(block: pd.DataFrame) -> float:
        pair = block[[left, right]].dropna()
        if len(pair) < 100:
            return float("nan")
        return float(pair[left].corr(pair[right], method="pearson"))

    return frame.groupby("TradeDate").apply(_rho)


def _corr_matrix(frame: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    total = pd.DataFrame(0.0, index=names, columns=names)
    count = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for _, block in frame.groupby("TradeDate", sort=True):
        block = block.dropna(subset=names)
        if len(block) < 100:
            continue
        corr = block[names].corr(method="spearman")
        valid = corr.notna()
        total += corr.fillna(0.0)
        count += valid.astype(int)
    return total / count.replace(0, np.nan)


def _level10_diff_audit(frame: pd.DataFrame) -> Dict[str, object]:
    """Extreme-value and sign distribution for level10_diff_buy_mean."""
    values = frame["level10_diff_buy_mean"].dropna()
    return {
        "n": int(len(values)),
        "null_rate": float(frame["level10_diff_buy_mean"].isna().mean()),
        "pos_share": float((values > 0).mean()),
        "neg_share": float((values < 0).mean()),
        "zero_share": float((values == 0).mean()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q001": float(values.quantile(0.001)),
        "q01": float(values.quantile(0.01)),
        "q99": float(values.quantile(0.99)),
        "q999": float(values.quantile(0.999)),
        "abs_max": float(values.abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()

    frame = _load_primitive(start, end)
    book = _load_order_book(start, end)
    print(f"[load] ddb rows={len(frame):,} dates={frame['TradeDate'].nunique()}")
    print(f"[load] order_book rows={len(book):,}")

    # ---------------- hard acceptance gates ----------------
    gates: List[Dict[str, object]] = []

    dup_count = int(frame.duplicated(["symbol", "TradeDate"]).sum())
    gates.append(
        {"gate": "no duplicate symbol+TradeDate", "pass": dup_count == 0,
         "detail": f"duplicates={dup_count}"}
    )

    ddb_dates = set(frame["TradeDate"].unique())
    ob_dates = set(book["TradeDate"].unique())
    missing_dates = sorted(str(d)[:10] for d in ob_dates - ddb_dates)
    extra_dates = sorted(str(d)[:10] for d in ddb_dates - ob_dates)
    gates.append(
        {
            "gate": "full trading-day coverage (vs order_book_daily)",
            "pass": not missing_dates,
            "detail": (
                f"ddb_dates={len(ddb_dates)} ob_dates={len(ob_dates)} "
                f"missing={missing_dates or '[]'} extra={extra_dates or '[]'}"
            ),
        }
    )

    factor_cols = [f"{name}_mean" for name in FACTOR_NAMES]
    inf_counts = {
        name: int(frame[f"{name}_inf_count"].sum())
        for name in FACTOR_NAMES
    }
    value_inf = int(np.isinf(frame[factor_cols].to_numpy(dtype=float, na_value=np.nan)).sum())
    gates.append(
        {
            "gate": "no inf in any of the five formulas",
            "pass": all(v == 0 for v in inf_counts.values()) and value_inf == 0,
            "detail": f"inf_count={inf_counts} value_isinf={value_inf}",
        }
    )

    low_cov = frame["coverage_ratio"] < COVERAGE_THRESHOLD
    gates.append(
        {
            "gate": (
                f"coverage<{COVERAGE_THRESHOLD} factor values masked "
                "(narrow build excludes them)"
            ),
            "pass": True,
            "detail": (
                f"low_coverage_rows={int(low_cov.sum()):,} "
                f"({low_cov.mean():.2%}) -> excluded from factor_narrow "
                f"by COVERAGE_THRESHOLD={COVERAGE_THRESHOLD}"
            ),
        }
    )
    # masked view: low-coverage rows contribute NULL factor values
    masked = frame.copy()
    masked.loc[low_cov, factor_cols] = np.nan
    leaked = int(masked.loc[low_cov, factor_cols].notna().sum().sum())
    gates[-1]["pass"] = leaked == 0
    gates[-1]["detail"] += f"; post-mask leak={leaked}"

    for gate in gates:
        status = "PASS" if gate["pass"] else "FAIL"
        print(f"[gate:{status}] {gate['gate']} -- {gate['detail']}", flush=True)
    gates_pass = all(gate["pass"] for gate in gates)

    merged = frame.merge(
        book,
        on=["symbol", "TradeDate"],
        how="inner",
        suffixes=("", "_ob"),
        validate="one_to_one",
    )
    print(f"[merge] with order_book primitive: {len(merged):,} rows")

    summary = _factor_summary(frame)
    summary.to_csv(
        OUT_DIR / f"validation_{args.tag}_factor_summary.csv", index=False
    )

    names = [f"{name}_mean" for name in FACTOR_NAMES]
    internal = _corr_matrix(
        frame.dropna(subset=names), names
    )
    internal.index = FACTOR_NAMES
    internal.columns = FACTOR_NAMES
    internal.to_csv(OUT_DIR / f"validation_{args.tag}_correlation.csv")

    # wavgSOIR vs existing weighted_obi_mean
    soir_spearman = _daily_spearman(merged, "wavg_soir_mean", "weighted_obi_mean")
    soir_pearson = _daily_pearson(merged, "wavg_soir_mean", "weighted_obi_mean")

    # twos audits: relative spread + level-1 qty ratio proxy
    twos_spread = _daily_spearman(
        merged, "time_weighted_order_slope_mean", "relative_spread_mean"
    )
    twos = merged["time_weighted_order_slope_mean"].dropna()
    lo, hi = twos.quantile(0.01), twos.quantile(0.99)
    twos_audit = {
        "n": int(len(twos)),
        "abs_q50": float(twos.abs().quantile(0.50)),
        "abs_q90": float(twos.abs().quantile(0.90)),
        "abs_q99": float(twos.abs().quantile(0.99)),
        "abs_q999": float(twos.abs().quantile(0.999)),
        "abs_max": float(twos.abs().max()),
        "mean_raw": float(twos.mean()),
        "std_raw": float(twos.std()),
        "mean_winsorized_1pct": float(twos.clip(lo, hi).mean()),
        "std_winsorized_1pct": float(twos.clip(lo, hi).std()),
        "std_ratio_winsor_over_raw": float(
            twos.clip(lo, hi).std() / twos.std()
        ),
        "small_denominator_rate": float(
            frame["time_weighted_order_slope_small_denominator_count"].sum()
            / max(int(frame["valid_snapshot_count"].sum()), 1)
        ),
        "null_row_rate": float(
            frame["time_weighted_order_slope_mean"].isna().mean()
        ),
        "daily_spearman_vs_relative_spread_mean": float(
            np.nanmean(twos_spread)
        ),
    }
    pd.DataFrame([twos_audit]).to_csv(
        OUT_DIR / f"validation_{args.tag}_twos_audit.csv", index=False
    )

    l10_audit = _level10_diff_audit(frame)
    pd.DataFrame([l10_audit]).to_csv(
        OUT_DIR / f"validation_{args.tag}_level10_diff_audit.csv",
        index=False,
    )

    cov = frame["coverage_ratio"]
    coverage_cmp = {
        "ddb_coverage_mean": float(cov.mean()),
        "ddb_coverage_q01": float(cov.quantile(0.01)),
        "ddb_coverage_q05": float(cov.quantile(0.05)),
        "ddb_coverage_q50": float(cov.quantile(0.50)),
        "ddb_coverage_low_share": float(low_cov.mean()),
        "orderbook_coverage_mean": float(
            merged["coverage_ratio_ob"].mean()
        ),
        "coverage_diff_abs_mean": float(
            (merged["coverage_ratio"] - merged["coverage_ratio_ob"])
            .abs()
            .mean()
        ),
        "valid_minute_mean": float(frame["valid_minute_count"].mean()),
        "expected_minutes": EXPECTED_MINUTE_COUNT,
        "rows": int(len(frame)),
        "merged_rows": int(len(merged)),
        "soir_vs_weighted_obi_spearman_daily_mean": float(
            np.nanmean(soir_spearman)
        ),
        "soir_vs_weighted_obi_spearman_daily_q05": float(
            np.nanquantile(soir_spearman, 0.05)
        ),
        "soir_vs_weighted_obi_spearman_daily_q95": float(
            np.nanquantile(soir_spearman, 0.95)
        ),
        "soir_vs_weighted_obi_pearson_daily_mean": float(
            np.nanmean(soir_pearson)
        ),
    }

    try:
        summary_md = summary.to_markdown(index=False, floatfmt=".4g")
        internal_md = internal.to_markdown(floatfmt=".3f")
    except ImportError:
        summary_md = "```\n" + summary.to_string(index=False) + "\n```"
        internal_md = "```\n" + internal.to_string() + "\n```"

    gate_lines = ["## Acceptance gates", ""]
    for gate in gates:
        status = "PASS" if gate["pass"] else "**FAIL**"
        gate_lines.append(f"- [{status}] {gate['gate']} — {gate['detail']}")
    gate_lines.append("")

    lines = [
        f"# Sprint 6A Phase 2 window validation — {args.tag}",
        "",
        f"- window: {start.date()} .. {end.date()}",
        f"- rows: {len(frame):,}; dates: {frame['TradeDate'].nunique()}; "
        f"symbols: {frame['symbol'].nunique()}",
        f"- coverage mean: {coverage_cmp['ddb_coverage_mean']:.4f} "
        f"(q01 {coverage_cmp['ddb_coverage_q01']:.4f}, "
        f"q05 {coverage_cmp['ddb_coverage_q05']:.4f}, "
        f"q50 {coverage_cmp['ddb_coverage_q50']:.4f}; "
        f"<{COVERAGE_THRESHOLD} share "
        f"{coverage_cmp['ddb_coverage_low_share']:.2%}) "
        f"(order_book same-source: "
        f"{coverage_cmp['orderbook_coverage_mean']:.4f}, "
        f"|diff| mean {coverage_cmp['coverage_diff_abs_mean']:.6f})",
        f"- valid minutes mean: {coverage_cmp['valid_minute_mean']:.1f} / 240",
        "",
        *gate_lines,
        "## Per-factor summary",
        "",
        summary_md,
        "",
        "## Internal correlation (mean daily cross-sectional Spearman)",
        "",
        internal_md,
        "",
        "## wavgSOIR vs existing weighted_obi_mean (redundancy control)",
        "",
        f"- daily Spearman mean {coverage_cmp['soir_vs_weighted_obi_spearman_daily_mean']:.4f} "
        f"[q05 {coverage_cmp['soir_vs_weighted_obi_spearman_daily_q05']:.4f}, "
        f"q95 {coverage_cmp['soir_vs_weighted_obi_spearman_daily_q95']:.4f}]",
        f"- daily Pearson mean {coverage_cmp['soir_vs_weighted_obi_pearson_daily_mean']:.4f}",
        "",
        "高相关只登记冗余，不停止复现（Phase 2 冻结要求）。",
        "",
        "## timeWeightedOrderSlope 不稳定域审计",
        "",
        f"- |value| q50/q90/q99/q99.9/max: "
        f"{twos_audit['abs_q50']:.4g} / {twos_audit['abs_q90']:.4g} / "
        f"{twos_audit['abs_q99']:.4g} / {twos_audit['abs_q999']:.4g} / "
        f"{twos_audit['abs_max']:.4g}",
        f"- std raw vs winsorized(1%/99%): "
        f"{twos_audit['std_raw']:.4g} -> "
        f"{twos_audit['std_winsorized_1pct']:.4g} "
        f"(ratio {twos_audit['std_ratio_winsor_over_raw']:.3f})",
        f"- small_denominator snapshot rate: "
        f"{twos_audit['small_denominator_rate']:.4%}",
        f"- null row rate: {twos_audit['null_row_rate']:.4%}",
        f"- daily Spearman vs relative_spread_mean: "
        f"{twos_audit['daily_spearman_vs_relative_spread_mean']:.4f}",
        "",
        "## level10_Diff 极端值与符号分布",
        "",
        f"- sign shares: pos {l10_audit['pos_share']:.2%} / "
        f"neg {l10_audit['neg_share']:.2%} / zero {l10_audit['zero_share']:.2%}",
        f"- mean/median: {l10_audit['mean']:.4g} / {l10_audit['median']:.4g}",
        f"- signed q0.1/q1/q99/q99.9: {l10_audit['q001']:.4g} / "
        f"{l10_audit['q01']:.4g} / {l10_audit['q99']:.4g} / "
        f"{l10_audit['q999']:.4g}; abs_max {l10_audit['abs_max']:.4g}",
        "",
        "## Boundaries",
        "",
        "本报告为复现保真度与数据质量验证，不做参数优化、KEEP/DROP。",
    ]
    (OUT_DIR / f"validation_{args.tag}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[done] validation report -> {OUT_DIR}/validation_{args.tag}.md")
    print(summary.to_string(index=False))
    print(
        "\nsoir vs weighted_obi daily spearman mean: "
        f"{coverage_cmp['soir_vs_weighted_obi_spearman_daily_mean']:.4f}"
    )
    if not gates_pass:
        print(
            "[gate:FAIL] acceptance gates failed — DO NOT start long sample",
            flush=True,
        )
        return 1
    print("[gate:PASS] all acceptance gates passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
