#!/usr/bin/env python
"""Sprint FS-1 — Build L2 ML Feature Panel (dataset contract only).

Gates:
  1. Freeze feature inventory from candidate_registry
  2. Canonical PIT (TradeDate, Symbol) spine from fast_context mask
  3. aligned_raw + processed_ind_cap_z_v1 panels
  4. HUATAI_STYLE_IND_CAP_Z_V1 preprocessing contract
  5. Integrity audits → A/B/C verdict

No selectors, learners, labels, or forward returns.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.feature_selection.contracts import (  # noqa: E402
    FS1_OUT_ROOT,
    PARITY_FACTORS,
    PREPROCESS_CONTRACT_ID,
)
from l2_factor_reproduction.feature_selection.panel import (  # noqa: E402
    align_quarter_panel,
    audit_key_integrity,
    audit_label_contamination,
    audit_source_parity,
    build_feature_inventory,
    build_spine_from_fast_context,
    compute_family_coverage,
    compute_feature_coverage,
    finalize_eligibility_with_spine_coverage,
    iter_quarters,
    write_panel_schema,
    write_partitioned_panel,
)
from l2_factor_reproduction.feature_selection.preprocessing import (  # noqa: E402
    apply_huatai_style_ind_cap_z_v1,
    audit_preprocessing_stages,
    load_log_mcap,
    load_or_cache_industry,
    preprocess_contract_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fs1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--window",
        choices=("discovery", "full"),
        default="full",
        help="fast_context window for PIT spine (default: full)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: first 5 spine dates + parity factors only",
    )
    p.add_argument("--smoke-n-dates", type=int, default=5)
    p.add_argument(
        "--max-factors",
        type=int,
        default=0,
        help="If >0, limit to first N eligible factors (after inventory)",
    )
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Build inventory + aligned_raw + audits only (no IND_CAP_Z)",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Override output root (default: feature_selection/fs1_feature_panel)",
    )
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip parquet probe in inventory (use summary stats)",
    )
    return p.parse_args()


def _select_factors(
    inventory: pd.DataFrame,
    *,
    smoke: bool,
    max_factors: int,
) -> List[str]:
    if smoke:
        names = [f for f, _ in PARITY_FACTORS]
        # keep only those present
        have = set(inventory["factor"])
        return [f for f in names if f in have]
    mats = inventory.loc[inventory["materialized"], "factor"].tolist()
    if max_factors > 0:
        return mats[:max_factors]
    return mats


def _render_report(
    out_root: Path,
    verdict: str,
    gates: Dict[str, bool],
    inventory: pd.DataFrame,
    extras: Dict[str, object],
) -> str:
    n_reg = len(inventory)
    n_mat = int(inventory["materialized"].sum())
    n_elig = int(inventory["eligible_for_fs"].sum())
    excl = inventory.loc[~inventory["eligible_for_fs"], ["factor", "family", "ineligible_reason"]]

    lines = [
        "# Sprint FS-1 — L2 ML Feature Panel",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        f"- Preprocess contract: `{PREPROCESS_CONTRACT_ID}`",
        f"- Registry rows: {n_reg}",
        f"- Materialized: {n_mat}",
        f"- Eligible for FS: {n_elig}",
        f"- Exclusions: {n_reg - n_elig}",
        "",
        "## Gate checklist",
        "",
    ]
    for k, v in gates.items():
        lines.append(f"- [{'PASS' if v else 'FAIL'}] {k}")
    lines += ["", "## Exclusions (if any)", ""]
    if excl.empty:
        lines.append("(none)")
    else:
        lines.append("| factor | family | reason |")
        lines.append("|---|---|---|")
        for _, r in excl.head(50).iterrows():
            lines.append(
                f"| {r['factor']} | {r['family']} | {r['ineligible_reason']} |"
            )
        if len(excl) > 50:
            lines.append(f"| ... | ... | (+{len(excl)-50} more) |")
    lines += ["", "## Notes", ""]
    for k, v in extras.items():
        lines.append(f"- **{k}:** {v}")
    lines += [
        "",
        "## Non-goals (confirmed)",
        "",
        "- No F / MI / FDR / L1 / Tree selectors",
        "- No Logistic / XGBoost",
        "- No `l2_ml_score`",
        "- No forward-return / label columns",
        "- Fast Discovery RAW path untouched",
        "",
    ]
    text = "\n".join(lines)
    (out_root / "report.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    args = _parse_args()
    t0 = time.time()

    out_root = Path(args.out_root) if args.out_root else FS1_OUT_ROOT
    # Keep discovery panel intact when building the full-history panel.
    if not args.out_root and args.window == "full":
        out_root = out_root.parent / (out_root.name + "_full")
    if args.smoke:
        out_root = out_root.parent / (out_root.name + "_smoke")
    out_root.mkdir(parents=True, exist_ok=True)
    aligned_root = out_root / "aligned_raw"
    processed_root = out_root / "processed_ind_cap_z_v1"
    coverage_dir = out_root / "coverage"
    audits_dir = out_root / "audits"
    for d in (aligned_root, processed_root, coverage_dir, audits_dir):
        d.mkdir(parents=True, exist_ok=True)

    gates: Dict[str, bool] = {
        "canonical_pit_universe": False,
        "duplicate_keys_zero": False,
        "source_factor_parity": False,
        "preprocessing_audit": False,
        "missingness_provenance": False,
        "feature_inventory_frozen": False,
        "no_label_contamination": False,
    }

    # ------------------------------------------------------------------ Gate 1
    logger.info("[Gate 1] feature inventory")
    inv_path = out_root / "feature_inventory.csv"
    inventory = build_feature_inventory(
        out_path=inv_path,
        probe_parquet=not args.no_probe,
    )
    gates["feature_inventory_frozen"] = inv_path.exists() and len(inventory) > 0
    logger.info(
        "  registered=%d materialized=%d",
        len(inventory),
        int(inventory["materialized"].sum()),
    )

    factors = _select_factors(
        inventory, smoke=args.smoke, max_factors=args.max_factors
    )
    logger.info("  building panel for %d factors", len(factors))

    # ------------------------------------------------------------------ Gate 2
    logger.info("[Gate 2] PIT spine from fast_context/%s", args.window)
    spine = build_spine_from_fast_context(
        args.window,
        smoke_n_dates=args.smoke_n_dates if args.smoke else None,
    )
    spine_path = out_root / "spine.parquet"
    spine.to_parquet(spine_path, index=False)
    gates["canonical_pit_universe"] = (
        spine.duplicated(["TradeDate", "Symbol"]).sum() == 0 and len(spine) > 0
    )
    logger.info(
        "  spine rows=%d dates=%d symbols≈%.0f",
        len(spine),
        spine["TradeDate"].nunique(),
        spine.groupby("TradeDate")["Symbol"].nunique().mean(),
    )

    # ------------------------------------------------------------------ Gate 3
    logger.info("[Gate 3] align raw factors onto spine (quarterly)")
    dates = sorted(spine["TradeDate"].unique())
    quarters = iter_quarters(dates)
    miss_parts: List[pd.DataFrame] = []

    inv_sub = inventory[inventory["factor"].isin(factors)].copy()

    for year, quarter, q_start, q_end in quarters:
        spine_q = spine[
            (spine["TradeDate"] >= q_start) & (spine["TradeDate"] <= q_end)
        ]
        if spine_q.empty:
            continue
        logger.info(
            "  align %dQ%d rows=%d factors=%d",
            year,
            quarter,
            len(spine_q),
            len(factors),
        )
        panel, miss = align_quarter_panel(spine_q, inv_sub, factors=factors)
        write_partitioned_panel(panel, aligned_root, year, quarter)
        if not miss.empty:
            miss_parts.append(miss)

    if miss_parts:
        miss_summary = pd.concat(miss_parts, ignore_index=True)
        miss_summary.to_csv(coverage_dir / "missingness_summary.csv", index=False)
        gates["missingness_provenance"] = True
    else:
        gates["missingness_provenance"] = False

    # coverage + finalize eligibility
    feat_cov = compute_feature_coverage(inv_sub, spine, aligned_root)
    feat_cov.to_csv(coverage_dir / "feature_coverage.csv", index=False)
    fam_cov = compute_family_coverage(feat_cov)
    fam_cov.to_csv(coverage_dir / "family_coverage.csv", index=False)

    inventory = finalize_eligibility_with_spine_coverage(inventory, feat_cov)
    # for smoke / subset, mark non-selected as excluded from this build only
    if args.smoke or args.max_factors > 0:
        selected = set(factors)
        mask_sel = inventory["factor"].isin(selected)
        inventory.loc[~mask_sel, "eligible_for_fs"] = False
        inventory.loc[~mask_sel, "ineligible_reason"] = inventory.loc[
            ~mask_sel, "ineligible_reason"
        ].apply(
            lambda x: (
                (str(x) + "|NOT_IN_THIS_BUILD").strip("|")
                if str(x)
                else "NOT_IN_THIS_BUILD"
            )
        )
    inventory.to_csv(inv_path, index=False)

    eligible_factors = inventory.loc[
        inventory["eligible_for_fs"], "factor"
    ].tolist()
    # keep intersection with built factors
    eligible_factors = [f for f in factors if f in set(eligible_factors)]
    logger.info("  eligible_for_fs (this build)=%d", len(eligible_factors))

    # ------------------------------------------------------------------ Gate 4
    stages_for_audit = None
    if args.skip_preprocess:
        logger.info("[Gate 4] SKIPPED (--skip-preprocess)")
        gates["preprocessing_audit"] = False
    else:
        logger.info("[Gate 4] %s", PREPROCESS_CONTRACT_ID)
        ind_cache = (
            Path(RESULT_ROOT)
            / "primitives"
            / f"citics_industry_wide_{args.window}.parquet"
        )
        d0, d1 = spine["TradeDate"].min(), spine["TradeDate"].max()
        industry = load_or_cache_industry(d0, d1, ind_cache)
        symbols = sorted(spine["Symbol"].unique())
        log_cap = load_log_mcap(d0, d1, columns=symbols)

        # audit sample dates/factors
        sample_dates = sorted(spine["TradeDate"].unique())[:5]
        sample_factors = eligible_factors[:10]

        for year, quarter, q_start, q_end in quarters:
            part = (
                aligned_root
                / f"year={year}"
                / f"quarter={quarter}"
                / "part.parquet"
            )
            if not part.exists():
                continue
            raw = pd.read_parquet(part)
            # only eligible columns
            cols = [c for c in eligible_factors if c in raw.columns]
            if not cols:
                continue
            logger.info(
                "  preprocess %dQ%d rows=%d cols=%d",
                year,
                quarter,
                len(raw),
                len(cols),
            )
            want_stages = any(
                (q_start <= pd.Timestamp(d) <= q_end) for d in sample_dates
            )
            processed, stages = apply_huatai_style_ind_cap_z_v1(
                raw,
                cols,
                industry=industry,
                log_cap=log_cap,
                inventory=inventory,
                return_stages=want_stages,
            )
            write_partitioned_panel(processed, processed_root, year, quarter)
            if stages is not None:
                stages_for_audit = stages

        # transform audit
        if stages_for_audit is not None:
            pre_audit = audit_preprocessing_stages(
                stages_for_audit,
                eligible_factors,
                industry,
                log_cap,
                sample_dates,
                sample_factors,
            )
            pre_audit.to_csv(audits_dir / "preprocessing_audit.csv", index=False)
            gates["preprocessing_audit"] = bool(
                len(pre_audit) > 0 and pre_audit["pass"].mean() >= 0.6
            )
        else:
            gates["preprocessing_audit"] = False

        (out_root / "preprocess_contract.json").write_text(
            json.dumps(preprocess_contract_manifest(), indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ Gate 5
    logger.info("[Gate 5] integrity audits")
    key_df = audit_key_integrity(aligned_root)
    key_df.to_csv(audits_dir / "key_integrity.csv", index=False)
    gates["duplicate_keys_zero"] = bool(len(key_df) > 0 and key_df["pass"].all())

    lab_df = audit_label_contamination(aligned_root)
    if not args.skip_preprocess and any(processed_root.glob("year=*/quarter=*/part.parquet")):
        lab_df = pd.concat(
            [lab_df, audit_label_contamination(processed_root)],
            ignore_index=True,
        )
    lab_df.to_csv(audits_dir / "label_contamination.csv", index=False)
    gates["no_label_contamination"] = bool(
        len(lab_df) > 0 and lab_df["pass"].all()
    )

    parity = audit_source_parity(inventory, aligned_root, factors=PARITY_FACTORS)
    parity.to_csv(audits_dir / "source_parity.csv", index=False)
    # only require parity for factors that were built
    built = set(factors)
    parity_built = parity[parity["factor"].isin(built)]
    gates["source_factor_parity"] = bool(
        len(parity_built) > 0 and parity_built["pass"].all()
    )

    pit_md = [
        "# PIT audit (FS-1)",
        "",
        "- Spine source: `fast_context/{}/universe_mask.parquet` where mask==1".format(
            args.window
        ),
        "- Factor values joined on (TradeDate, Symbol) with left join (no forward fill across days).",
        "- **No forward returns / labels are joined in this sprint.**",
        "- Fast Discovery remains RAW + signal.shift(1); this panel is a separate ML adapter path.",
        "",
        f"- Spine dates: {spine['TradeDate'].min().date()} → {spine['TradeDate'].max().date()}",
        f"- Spine rows: {len(spine)}",
        "",
    ]
    (audits_dir / "pit_audit.md").write_text("\n".join(pit_md), encoding="utf-8")

    write_panel_schema(
        out_root,
        extra={
            "window": args.window,
            "smoke": bool(args.smoke),
            "n_spine_rows": int(len(spine)),
            "n_factors_built": len(factors),
            "n_eligible": len(eligible_factors),
            "out_root": str(out_root),
        },
    )

    # Verdict
    required = [
        "canonical_pit_universe",
        "duplicate_keys_zero",
        "source_factor_parity",
        "missingness_provenance",
        "feature_inventory_frozen",
        "no_label_contamination",
    ]
    if not args.skip_preprocess:
        required.append("preprocessing_audit")

    all_pass = all(gates[k] for k in required)
    n_excl = int((~inventory["eligible_for_fs"]).sum())
    if all_pass and n_excl == 0 and not args.smoke and args.max_factors <= 0:
        verdict = "A. FS1_PANEL_READY"
    elif all_pass:
        verdict = "B. FS1_PANEL_READY_WITH_EXCLUSIONS"
    else:
        verdict = "C. FS1_PANEL_NOT_READY"

    manifest = {
        "verdict": verdict,
        "gates": gates,
        "window": args.window,
        "smoke": bool(args.smoke),
        "n_registry": int(len(inventory)),
        "n_materialized": int(inventory["materialized"].sum()),
        "n_eligible_for_fs": int(inventory["eligible_for_fs"].sum()),
        "n_factors_built": len(factors),
        "elapsed_sec": round(time.time() - t0, 1),
        "out_root": str(out_root),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _render_report(
        out_root,
        verdict,
        gates,
        inventory,
        extras={
            "window": args.window,
            "smoke": args.smoke,
            "elapsed_sec": manifest["elapsed_sec"],
            "out_root": str(out_root),
            "failed_gates": [k for k in required if not gates[k]],
        },
    )

    logger.info("VERDICT %s (elapsed=%.1fs)", verdict, time.time() - t0)
    logger.info("Gates: %s", json.dumps(gates))
    print(json.dumps(manifest, indent=2))
    return 0 if verdict.startswith(("A.", "B.")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
