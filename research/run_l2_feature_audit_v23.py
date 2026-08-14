#!/usr/bin/env python3
"""Phase 2.3 — L2 Feature Factory family / correlation audit.

Does not expand features. Deduplicates CS-rank clones (evaluator already ranks).
Outputs correlation tables and a Phase-3 research candidate set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from research.l2_alpha.feature_factory.registry import (  # noqa: E402
    L2_FF_ALL_COLUMNS,
    expand_derived_names,
)

DEFAULT_PANEL = PROJECT / "research/results/l2_feature_factory_v1/panel"
DEFAULT_OUTPUT = PROJECT / "research/results/l2_feature_factory_v1"
DEFAULT_DECISIONS = DEFAULT_OUTPUT / "l2_ff_decisions.csv"
DEFAULT_METRICS = DEFAULT_OUTPUT / "l2_ff_metrics.csv"

# Mechanism families for overlap report.
FAMILY_MAP = {
    "woi": "woi",
    "depth_imb": "depth",
    "micro_bias": "microprice",
    "spread": "liquidity",
    "cancel": "cancel",
}

# Research candidates for Phase 3 (not production KEEP only).
PHASE3_RESEARCH = (
    "woi_mean10",
    "depth_imb_mean10",
    "woi_delta30",
    "woi_std20",
)


def _family(factor: str) -> str:
    for prefix, fam in FAMILY_MAP.items():
        if factor.startswith(prefix):
            return fam
    return "other"


def _is_rank_clone(factor: str) -> bool:
    return factor.endswith("_rank")


def _unique_factors(factors: List[str]) -> List[str]:
    """Drop *_rank clones — evaluation already cross-sectionally ranks."""
    return [f for f in factors if not _is_rank_clone(f)]


def _load_panel(panel_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for day in pd.bdate_range(start, end):
        path = panel_dir / f"{day.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(out["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates
    return out


def _mean_cs_feature_corr(
    panel: pd.DataFrame,
    factors: List[str],
    *,
    bartime: Optional[str] = None,
    max_slots: int = 400,
) -> pd.DataFrame:
    """Average cross-sectional Spearman corr of factor values over date×bartime."""
    sub = panel[panel["factor"].isin(factors)].copy()
    if bartime:
        sub = sub[sub["bartime"] == bartime]
    keys = sub.groupby(["date", "bartime"], sort=True).ngroup()
    # Cap for speed on large panels.
    unique_keys = keys.drop_duplicates().to_numpy()
    if len(unique_keys) > max_slots:
        rng = np.random.default_rng(42)
        keep = set(rng.choice(unique_keys, size=max_slots, replace=False))
        sub = sub[keys.isin(keep)]

    corrs = []
    for (_, _), g in sub.groupby(["date", "bartime"], sort=False):
        wide = g.pivot_table(
            index="symbol", columns="factor", values="value", aggfunc="last"
        )
        wide = wide.reindex(columns=factors)
        if wide.dropna(how="all").shape[0] < 30:
            continue
        c = wide.corr(method="spearman")
        corrs.append(c)
    if not corrs:
        return pd.DataFrame(index=factors, columns=factors, dtype=float)
    stack = np.stack([c.to_numpy(dtype=float) for c in corrs], axis=0)
    mean = np.nanmean(stack, axis=0)
    return pd.DataFrame(mean, index=factors, columns=factors)


def _ic_vector_corr(metrics: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
    """Correlate factors by their train RankIC vectors across bartime×horizon."""
    frame = metrics[metrics["factor"].isin(factors)].copy()
    if frame.empty:
        return pd.DataFrame()
    # Use signed IC aligned with frozen direction from best? Use raw rank_ic as stored.
    pivot = frame.pivot_table(
        index=["bartime", "return_window"],
        columns="factor",
        values="rank_ic",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=factors)
    return pivot.corr(method="spearman")


def _family_overlap(feat_corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    factors = list(feat_corr.columns)
    for i, a in enumerate(factors):
        for b in factors[i + 1 :]:
            rows.append(
                {
                    "factor_a": a,
                    "factor_b": b,
                    "family_a": _family(a),
                    "family_b": _family(b),
                    "same_family": _family(a) == _family(b),
                    "spearman": float(feat_corr.loc[a, b])
                    if a in feat_corr.index and b in feat_corr.columns
                    else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("spearman", ascending=False).reset_index(drop=True)


def _write_report(
    path: Path,
    *,
    decisions: pd.DataFrame,
    unique_keep: List[str],
    phase3: pd.DataFrame,
    high_corr: pd.DataFrame,
    meta: dict,
) -> None:
    lines = [
        "# Phase 2.3 — L2 Feature Family Audit",
        "",
        "## Verdict",
        "",
        "- Feature Factory works; do **not** treat `*_rank` as separate alphas.",
        "- Production KEEP (unique): "
        + (", ".join(unique_keep) if unique_keep else "_none_"),
        "- Phase 3 research set expands beyond strict KEEP for robustness.",
        "",
        "## Rank clone note",
        "",
        "Evaluator already cross-sectionally ranks signals. Therefore",
        "`woi_mean10` and `woi_mean10_rank` (and other `*_rank` pairs) share",
        "identical IC / H-L metrics and count as **one** candidate.",
        "",
        "## Meta",
        "",
        "```json",
        json.dumps(meta, indent=2),
        "```",
        "",
        "## Phase 3 research candidates",
        "",
    ]
    try:
        lines.append(phase3.to_markdown(index=False))
    except Exception:  # noqa: BLE001
        lines.append("```\n" + phase3.to_string(index=False) + "\n```")
    lines += [
        "",
        "## Highest feature-value correlations (|ρ|≥0.7)",
        "",
    ]
    if high_corr.empty:
        lines.append("_None above threshold._")
    else:
        try:
            lines.append(high_corr.head(30).to_markdown(index=False))
        except Exception:  # noqa: BLE001
            lines.append(
                "```\n" + high_corr.head(30).to_string(index=False) + "\n```"
            )
    lines += [
        "",
        "## Next",
        "",
        "Phase 3: freeze tuples on 2024H1, OOS 2024H2+2025, then residual vs",
        "RV/CVWAP/Amihud, then cost ladder. Do not expand feature count yet.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument(
        "--anchor-bartime",
        default="14:29",
        help="Primary bartime for feature-value corr focus",
    )
    parser.add_argument("--max-slots", type=int, default=300)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_csv(args.decisions)
    metrics = pd.read_csv(args.metrics)

    all_factors = _unique_factors(list(L2_FF_ALL_COLUMNS))
    # Prefer derived list if present in decisions
    present = sorted(set(decisions["factor"]) & set(all_factors))
    if not present:
        present = all_factors

    unique_keep = sorted(
        {
            f.replace("_rank", "") if f.endswith("_rank") else f
            for f in decisions.loc[decisions["decision"] == "KEEP", "factor"]
        }
    )
    # Map back to non-rank names only
    unique_keep = [f for f in unique_keep if not f.endswith("_rank")]

    print("[p23] load panel …", flush=True)
    panel = _load_panel(args.panel_dir, args.start, args.end)
    if panel.empty:
        print("[p23] empty panel", flush=True)
        return 2

    print(
        f"[p23] panel rows={len(panel)} unique_factors={len(present)}",
        flush=True,
    )
    feat_corr = _mean_cs_feature_corr(
        panel,
        present,
        bartime=args.anchor_bartime,
        max_slots=args.max_slots,
    )
    feat_corr.to_csv(args.output / "l2_feature_corr.csv")

    # Also full-grid mean corr (sampled) for family report
    feat_corr_all = _mean_cs_feature_corr(
        panel, present, bartime=None, max_slots=args.max_slots
    )
    feat_corr_all.to_csv(args.output / "l2_feature_corr_all_slots.csv")

    ic_corr = _ic_vector_corr(metrics, present)
    if not ic_corr.empty:
        ic_corr.to_csv(args.output / "l2_ic_corr.csv")

    overlap = _family_overlap(feat_corr)
    overlap.to_csv(args.output / "l2_feature_pair_overlap.csv", index=False)
    high = overlap[overlap["spearman"].abs() >= 0.7].copy()

    # Phase-3 candidate table: train-best + frozen 14:29/Ret_30 view
    frozen_bt, frozen_h = "14:29", "Ret_30"
    cand_rows = []
    for name in PHASE3_RESEARCH:
        row = decisions[decisions["factor"] == name]
        if row.empty:
            continue
        r = row.iloc[0]
        frozen = metrics[
            (metrics["factor"] == name)
            & (metrics["bartime"] == frozen_bt)
            & (metrics["return_window"] == frozen_h)
        ]
        fr = frozen.iloc[0] if not frozen.empty else None
        cand_rows.append(
            {
                "factor": name,
                "family": _family(name),
                "role": (
                    "production_candidate"
                    if name in unique_keep
                    else "research_candidate"
                ),
                "best_bartime": r["bartime"],
                "best_horizon": r["horizon"],
                "best_direction": int(r["direction"]),
                "best_icir": float(r["icir"]),
                "best_hl_sharpe": float(r["hl_sharpe"])
                if pd.notna(r["hl_sharpe"])
                else np.nan,
                "best_mono_spearman": float(r["mono_spearman"]),
                "best_mono_pass": bool(r["mono_pass"]),
                "train_decision": r["decision"],
                "frozen_bartime": frozen_bt,
                "frozen_horizon": frozen_h,
                "frozen_icir": float(fr["annualized_icir"])
                if fr is not None
                else np.nan,
                "frozen_hl_sharpe": float(fr["hl_sharpe"])
                if fr is not None and pd.notna(fr["hl_sharpe"])
                else np.nan,
                "frozen_rank_ic": float(fr["rank_ic"]) if fr is not None else np.nan,
                "frozen_mono": float(fr["decile_mono_spearman"])
                if fr is not None and pd.notna(fr["decile_mono_spearman"])
                else np.nan,
            }
        )
    phase3 = pd.DataFrame(cand_rows)
    phase3.to_csv(args.output / "l2_phase3_candidates.csv", index=False)

    # Deduped decisions view
    dedup = decisions[~decisions["factor"].map(_is_rank_clone)].copy()
    dedup.to_csv(args.output / "l2_ff_decisions_dedup.csv", index=False)

    meta = {
        "phase": "2.3_feature_audit",
        "anchor_bartime": args.anchor_bartime,
        "n_unique_factors": len(present),
        "n_rank_clones_excluded": int(
            sum(_is_rank_clone(f) for f in decisions["factor"])
        ),
        "unique_keep": unique_keep,
        "phase3_research": list(PHASE3_RESEARCH),
        "derived_only_universe": list(expand_derived_names()),
    }
    (args.output / "l2_feature_family_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        args.output / "l2_feature_family_report.md",
        decisions=decisions,
        unique_keep=unique_keep,
        phase3=phase3,
        high_corr=high,
        meta=meta,
    )
    print(phase3.to_string(index=False), flush=True)
    print(f"[p23] done → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
