#!/usr/bin/env python
"""Sprint 6B — unified Candidate Pool baseline for Cancellation Family.

7 frozen formulas. Mirrors expand_ddb_snapshot_family conventions:
T+1 signal.shift(1), benchmark 000852.SH, 7.5bps, raw IC keeps frozen
direction, effective direction re-runs groupTest.

不做：参数优化、周频优化、中性化搜索、组合、KEEP/DROP。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    _save_backtest_outputs,
    backtest_factor,
    load_backtest_context,
)
from l2_factor_reproduction.python.candidate_pool import (  # noqa: E402
    correlation_pairs,
    decile_monotonicity,
    load_rank_ic,
    mean_daily_cross_sectional_spearman,
    redundancy_annotations,
    stability_fields,
    yearly_ic_table,
)
from l2_factor_reproduction.python.candidate_pool_registry import (  # noqa: E402
    BASELINE_POLICY,
)
from l2_factor_reproduction.python.ch_cancel_lifecycle import (  # noqa: E402
    PRIMITIVE_VERSION,
)

DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
POOL_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "cancel_lifecycle_family"
FACTOR_ROOT = POOL_DIR / "factors"
POOL_ROOT = Path(RESULT_ROOT) / "candidate_pool_v1"
ORDER_BOOK_POOL = POOL_ROOT / "order_book_family"
LIQUIDITY_POOL = POOL_ROOT / "liquidity_impact_family"

CANDIDATES = [
    "cancel_value_pressure",
    "cancel_count_pressure",
    "cancel_value_intensity",
    "cancel_qty_intensity",
    "relative_cancel_order_size",
    "cancel_pressure_shock_20d",
    "cancel_intensity_shock_20d",
]

REGISTRY = pd.read_csv(
    Path(RESULT_ROOT) / "primitives" / "cancel_lifecycle_daily"
    / "candidate_registry_v1.csv"
).set_index("name")

PEER_FAMILIES: Tuple[Tuple[str, Path, bool], ...] = (
    ("order_book", ORDER_BOOK_POOL / "factors", False),
    ("trade_flow", Path(RESULT_ROOT), True),
    ("order_size", Path(RESULT_ROOT), True),
    ("price_formation", POOL_ROOT / "price_formation_family" / "factors", False),
    ("liquidity_impact", LIQUIDITY_POOL / "factors", False),
    ("ddb_reference_snapshot",
     POOL_ROOT / "ddb_reference_snapshot_family" / "factors", False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_meta(start, end, narrow_path: Path) -> Dict[str, object]:
    return {
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "signal_shift": 1,
        "benchmark": BASELINE_POLICY["benchmark"],
        "factor_narrow_sha256": _sha256(narrow_path),
        "formula_version": PRIMITIVE_VERSION,
        "primitive_schema_version": PRIMITIVE_VERSION,
    }


def _meta_valid(directory: Path, expected: Dict[str, object]) -> bool:
    meta_path = directory / "backtest_meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mismatches = [
        k for k, v in expected.items() if meta.get(k) != v
    ]
    if mismatches:
        print(f"[reuse-reject] {directory.name}: {mismatches}", flush=True)
        return False
    return True


def _run_or_reuse(name, start, end, *, mask, ret_matrix, force):
    output = FACTOR_ROOT / name
    summary_path = output / "summary.json"
    rank_ic_path = output / "rank_ic.csv"
    narrow_path = output / "factor_narrow.parquet"
    expected = _expected_meta(start, end, narrow_path)
    if (not force and summary_path.exists() and rank_ic_path.exists()
            and _meta_valid(output, expected)):
        print(f"[backtest] reuse {name} (meta verified)", flush=True)
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            load_rank_ic(rank_ic_path),
        )
    narrow = pd.read_parquet(
        narrow_path, columns=["symbol", "tradetime", "value"])
    group_pnl, group_to, rank_ic, summary = backtest_factor(
        narrow, start_day=start, end_day=end, signal_shift=1,
        mask=mask, ret_matrix=ret_matrix,
    )
    summary["net_annu_after_fee"] = (
        float(summary["hl_annu_ret_flipped"])
        - float(summary["implied_annu_fee"])
    )
    _save_backtest_outputs(
        str(output), group_pnl, group_to, rank_ic, summary,
        factor_name=name,
    )
    meta = dict(expected)
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    (output / "backtest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, rank_ic


def _peer_dirs(label: str, root: Path, external: bool):
    if external:
        registry = POOL_ROOT / f"{label}_family" / "factor_registry.csv"
        if not registry.exists():
            warnings.warn(f"peer {label}: registry missing; skipped")
            return []
        names = pd.read_csv(registry)["name"].tolist()
        return [
            (n, root / n / "factor_narrow.parquet")
            for n in names
            if (root / n / "factor_narrow.parquet").exists()
        ]
    if not root.is_dir():
        warnings.warn(f"peer {label}: root missing; skipped")
        return []
    return [
        (p.name, p / "factor_narrow.parquet")
        for p in sorted(root.iterdir())
        if (p / "factor_narrow.parquet").exists()
    ]


def _load_narrow_wide(names, start, end):
    frames = []
    for name in names:
        n = pd.read_parquet(
            FACTOR_ROOT / name / "factor_narrow.parquet",
            columns=["symbol", "tradetime", "value"],
        )
        n = n.rename(columns={"value": name})
        n["TradeDate"] = pd.to_datetime(n["tradetime"]).dt.normalize()
        n = n.loc[n["TradeDate"].between(start, end),
                  ["symbol", "TradeDate", name]]
        frames.append(n)
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["symbol", "TradeDate"], how="outer")
    return out


def _cross_family(wide_new, names, start, end):
    wide = wide_new.copy()
    wide["year"] = wide["TradeDate"].dt.year
    year_blocks = {
        int(y): b.drop(columns="year") for y, b in wide.groupby("year")
    }
    del wide
    rows = []
    selection = []
    for label, root, external in PEER_FAMILIES:
        peers = _peer_dirs(label, root, external)
        print(f"[corr] cross-family vs {len(peers)} {label}", flush=True)
        for i, (peer, path) in enumerate(peers, start=1):
            selection.append({"peer_family": label, "peer_factor": peer,
                              "path": str(path)})
            narrow = pd.read_parquet(
                path, columns=["symbol", "tradetime", "value"])
            block = narrow.rename(columns={"value": peer})
            block["TradeDate"] = pd.to_datetime(
                block["tradetime"]).dt.normalize()
            block["year"] = block["TradeDate"].dt.year
            rhos = {name: [] for name in names}
            for year, new_block in year_blocks.items():
                peer_year = block.loc[
                    block["year"] == year, ["symbol", "TradeDate", peer]]
                if peer_year.empty:
                    continue
                merged = new_block.merge(
                    peer_year, on=["symbol", "TradeDate"], how="inner")
                for name in names:
                    daily = merged.groupby("TradeDate").apply(
                        lambda g, n=name: g[n].corr(g[peer], method="spearman")
                        if len(g) >= 100 else np.nan
                    )
                    rhos[name].extend(daily.dropna().tolist())
            for name in names:
                series = pd.Series(rhos[name])
                rows.append({
                    "cancel_factor": name,
                    "peer_family": label,
                    "peer_factor": peer,
                    "mean_daily_spearman": float(series.mean())
                    if len(series) else np.nan,
                    "n_days": int(len(series)),
                })
            if i % 16 == 0:
                print(f"  [corr] {label} {i}/{len(peers)}", flush=True)
            del narrow, block
            gc.collect()
    pd.DataFrame(selection).to_csv(
        POOL_DIR / "cross_family_peer_selection.csv", index=False)
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame) -> None:
    cols = [
        "factor", "rank_ic_raw", "icir_raw", "g10_excess_sharpe",
        "hl_sharpe", "avg_hl_turnover", "net_annu_after_fee",
        "sign_consistency", "decile_mono_spearman",
        "redundancy_cluster_080",
    ]
    view = summary[[c for c in cols if c in summary.columns]]
    try:
        table = view.to_markdown(index=False)
    except ImportError:
        table = "```\n" + view.to_string(index=False) + "\n```"

    g10 = summary.loc[summary["g10_excess_sharpe"] > 3, "factor"].tolist()
    hl = summary.loc[summary["hl_sharpe"] > 3, "factor"].tolist()
    ic2 = summary.loc[summary["rank_ic_raw"].abs() >= 0.02, "factor"].tolist()
    icir3 = summary.loc[summary["icir_raw"].abs() >= 3, "factor"].tolist()
    mono = summary.loc[
        summary["decile_mono_spearman"] >= 0.8, "factor"].tolist()
    yc = summary.loc[
        summary["sign_consistency"] >= 0.75, "factor"].tolist()
    n_clusters = (
        summary["redundancy_cluster_080"].nunique()
        if "redundancy_cluster_080" in summary.columns else float("nan")
    )

    lines = [
        "# L2 Candidate Pool v1 — Cancellation / Order Lifecycle Family",
        "",
        "Sprint 6B：撤单生命周期家族（冻结 cancel_lifecycle_v1）。"
        "未做参数优化、二次中性化、周度优化、组合或 KEEP/DROP。",
        "",
        f"- 冻结公式数：{len(summary)}",
        f"- 经验簇数（|ρ|≥0.80）：{n_clusters}",
        "- IC/ICIR：冻结原方向；分组收益：统一 effective direction",
        "- 日频调仓仅用于统一 discovery baseline",
        "",
        "## Unified baseline",
        "",
        table,
        "",
        "## Required thresholds",
        "",
        f"1. **G10 Excess Sharpe > 3**：{g10 or '无'}",
        f"2. **H-L Sharpe > 3**：{hl or '无'}",
        f"3. **|IC| ≥ 2%**：{ic2 or '无'}",
        f"4. **|ICIR| ≥ 3**：{icir3 or '无'}",
        f"5. **monotonicity ≥ 0.8**：{mono or '无'}",
        f"6. **yearly consistency ≥ 75%**：{yc or '无'}",
        "",
        "## Boundaries",
        "",
        "本报告不作正式 KEEP/DROP、生产晋级或组合结论。"
        "cross-family correlation 仅作 taxonomy reference。",
        "",
    ]
    (POOL_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument("--force-backtest", action="store_true")
    parser.add_argument("--skip-corr", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        _write_report(pd.read_csv(POOL_DIR / "candidate_summary.csv"))
        return 0

    coverage = pd.read_csv(POOL_DIR / "factor_coverage.csv").set_index(
        "factor")
    print("[backtest] loading shared return/mask context once", flush=True)
    mask, ret_matrix = load_backtest_context(start, end)

    rows = []
    yearly_parts = []
    for i, name in enumerate(CANDIDATES, start=1):
        print(f"[backtest {i}/{len(CANDIDATES)}] {name}", flush=True)
        summary, rank_ic_eff = _run_or_reuse(
            name, start, end, mask=mask, ret_matrix=ret_matrix,
            force=args.force_backtest,
        )
        direction = int(summary.get("factor_direction", 1))
        rank_ic_raw = rank_ic_eff * direction
        yearly_raw = yearly_ic_table(rank_ic_raw)
        yearly_eff = yearly_ic_table(rank_ic_eff)
        factor_dir = FACTOR_ROOT / name
        yearly_raw.to_csv(factor_dir / "yearly_ic.csv")
        yearly_eff.to_csv(factor_dir / "yearly_ic_effective.csv")
        yearly_raw.assign(factor=name).reset_index().to_csv(
            factor_dir / "yearly_stability.csv", index=False)
        yearly_parts.append(yearly_raw.assign(factor=name).reset_index())

        raw_mean = float(summary.get("rank_ic_mean_raw", rank_ic_raw.mean()))
        raw_std = float(rank_ic_raw.std())
        raw_icir = (
            raw_mean / raw_std * np.sqrt(250) if raw_std > 0 else float("nan")
        )
        cov = coverage.loc[name]
        spec = REGISTRY.loc[name]
        row = {
            "factor": name,
            "category": "cancel_lifecycle",
            "mechanism": spec["mechanism"],
            "lookback_days": int(spec["lookback_days"]),
            "n_factor_rows": int(cov["n_factor_rows"]),
            "date_min": cov["date_min"],
            "date_max": cov["date_max"],
            "n_symbols": int(cov["n_symbols"]),
            "factor_direction": direction,
            "direction_flip": bool(direction < 0),
            "rank_ic_raw": raw_mean,
            "icir_raw": raw_icir,
            "rank_ic_std": raw_std,
            "positive_ic_fraction": float(
                (rank_ic_raw.dropna() > 0).mean()),
            "rank_ic_effective": float(summary["rank_ic_mean"]),
            "icir_effective": float(summary["rank_icir"]),
            "hl_annu_ret": float(summary["hl_annu_ret_flipped"]),
            "hl_sharpe": float(summary["hl_sharpe_flipped"]),
            "g10_excess_annu_ret": float(summary["g10_excess_annu_ret"]),
            "g10_excess_sharpe": float(summary["g10_excess_sharpe"]),
            "hl_mdd": float(summary["hl_mdd_flipped"]),
            "avg_hl_turnover": float(summary["avg_hl_turnover"]),
            "implied_annu_fee": float(summary["implied_annu_fee"]),
            "net_annu_after_fee": (
                float(summary["hl_annu_ret_flipped"])
                - float(summary["implied_annu_fee"])
            ),
            "decile_mono_spearman": decile_monotonicity(summary),
            "n_days": int(summary["n_days"]),
            "n_names_avg": float(summary["n_names_avg"]),
            "group_pnl_saved_direction": summary[
                "group_pnl_saved_direction"],
            **stability_fields(yearly_raw, raw_mean),
        }
        rows.append(row)
        print(
            f"[result] raw IC={raw_mean:+.4f} ICIR={raw_icir:+.2f} "
            f"G10={row['g10_excess_sharpe']:.2f} "
            f"H-L={row['hl_sharpe']:.2f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(POOL_DIR / "candidate_summary.csv", index=False)
    pd.concat(yearly_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_all.csv", index=False)

    if not args.skip_corr:
        print("[corr] intra-family daily spearman", flush=True)
        wide = _load_narrow_wide(CANDIDATES, start, end)
        intra = mean_daily_cross_sectional_spearman(wide, CANDIDATES)
        intra.to_csv(POOL_DIR / "factor_correlation_spearman.csv")
        correlation_pairs(intra).to_csv(
            POOL_DIR / "intra_family_correlation_pairs.csv", index=False)
        annotations = redundancy_annotations(intra, threshold=0.80)
        annotations.to_csv(
            POOL_DIR / "redundancy_clusters_080.csv", index=False)
        summary_frame = summary_frame.merge(
            annotations[[
                "factor", "redundancy_cluster_080", "max_corr_peer",
                "max_abs_corr", "near_alias_observed",
            ]],
            on="factor", how="left", validate="one_to_one",
        )
        summary_frame.to_csv(POOL_DIR / "candidate_summary.csv", index=False)
        cross = _cross_family(wide, CANDIDATES, start, end)
        cross.to_csv(POOL_DIR / "cross_family_correlation.csv", index=False)
        del wide
        gc.collect()

    _write_report(summary_frame)
    manifest = {
        "version": "cancel_lifecycle_candidate_pool_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "n_candidates": len(CANDIDATES),
        "baseline_policy": BASELINE_POLICY,
        "formula_version": PRIMITIVE_VERSION,
    }
    (POOL_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] baseline -> {POOL_DIR}", flush=True)
    print(summary_frame[[
        "factor", "rank_ic_raw", "icir_raw",
        "g10_excess_sharpe", "hl_sharpe",
    ]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
