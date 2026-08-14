"""Phase A + Phase B skeleton runner.

Safe: reads existing registries/CSVs only. Does not load the 17GB FS-1 panel,
does not fit models, does not scan ClickHouse / DolphinDB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.l2_ai_stock_selection.contracts import (  # noqa: E402
    AI_CONTRACT_VERSION,
    data_contract_dict,
)
from l2_factor_reproduction.l2_ai_stock_selection.model_contract import (  # noqa: E402
    COMPARISON_METRICS,
    COMPARISON_MODELS,
    model_contract_dict,
)
from l2_factor_reproduction.l2_ai_stock_selection.fs_jury import (  # noqa: E402
    EVIDENCE_METHODS,
)
from l2_factor_reproduction.l2_ai_stock_selection.inventory import (  # noqa: E402
    family_summary,
    load_factor_inventory,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (  # noqa: E402
    CANDIDATE_DISCOVERY,
    FEATURE_ENGINEERING,
    FEATURE_SELECTION,
    PROJECT_DIR,
    REPORTS,
    ensure_layout,
)
from l2_factor_reproduction.l2_ai_stock_selection.ratio_catalog import (  # noqa: E402
    EXISTING_RATIO_ALIASES,
    build_ratio_candidate_registry,
)
from l2_factor_reproduction.l2_ai_stock_selection.style_controls import (  # noqa: E402
    style_control_catalog,
)


def _write_empty_schema(path: Path, columns, overwrite: bool = False) -> None:
    if overwrite or (not path.exists()):
        pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Override result root (tests).",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()

    if args.out_root:
        # tests only — keep production default otherwise
        from l2_factor_reproduction.l2_ai_stock_selection import paths as _paths

        _paths.PROJECT_DIR = Path(args.out_root)
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        for sub in (
            "candidate_discovery",
            "feature_engineering",
            "feature_selection",
            "models/baseline_linear",
            "models/lightgbm",
            "models/xgboost",
            "models/random_forest",
            "attribution",
            "frequency",
            "portfolio",
            "reports",
        ):
            (out_root / sub).mkdir(parents=True, exist_ok=True)
        inventory_path = out_root / "factor_inventory_by_family.csv"
        family_path = out_root / "family_summary.csv"
        ratio_path = out_root / "feature_engineering" / "ratio_candidate_registry.csv"
        contract_path = out_root / "data_contract.json"
        style_path = out_root / "attribution" / "style_control_catalog.csv"
        disc = out_root / "candidate_discovery"
        fs_dir = out_root / "feature_selection"
        reports = out_root / "reports"
    else:
        ensure_layout()
        inventory_path = PROJECT_DIR / "factor_inventory_by_family.csv"
        family_path = PROJECT_DIR / "family_summary.csv"
        ratio_path = FEATURE_ENGINEERING / "ratio_candidate_registry.csv"
        contract_path = PROJECT_DIR / "data_contract.json"
        style_path = PROJECT_DIR / "attribution" / "style_control_catalog.csv"
        disc = CANDIDATE_DISCOVERY
        fs_dir = FEATURE_SELECTION
        reports = REPORTS

    inventory = load_factor_inventory()
    summary = family_summary(inventory)
    ratios = build_ratio_candidate_registry(inventory)
    styles = style_control_catalog()
    contract = data_contract_dict()

    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(family_path, index=False)
    ratios.to_csv(ratio_path, index=False)
    styles.to_csv(style_path, index=False)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    model_contract_path = (
        Path(args.out_root) / "model_contract.json"
        if args.out_root
        else PROJECT_DIR / "model_contract.json"
    )
    model_contract_path.write_text(
        json.dumps(model_contract_dict(), indent=2), encoding="utf-8"
    )
    comparison_path = (
        Path(args.out_root) / "model_comparison.csv"
        if args.out_root
        else PROJECT_DIR / "model_comparison.csv"
    )
    cmp = pd.DataFrame({"model": list(COMPARISON_MODELS)})
    for col in COMPARISON_METRICS:
        cmp[col] = pd.NA
    cmp.to_csv(comparison_path, index=False)

    _write_empty_schema(
        disc / "candidate_incremental_alpha.csv",
        [
            "factor",
            "family",
            "raw_rank_ic",
            "residual_rank_ic",
            "incremental_rank_ic",
            "raw_mi",
            "residual_mi",
            "incremental_mi",
            "residual_bin_spread",
            "n_train_dates",
        ],
    )
    _write_empty_schema(
        disc / "candidate_non_linear_diagnostics.csv",
        ["factor", "family", "raw_rank_ic", "residual_mi", "bin_spread", "review_nonlinear"],
    )
    _write_empty_schema(
        disc / "factor_clusters.csv",
        ["factor", "family", "redundancy_cluster_080", "max_candidate_corr", "representative"],
    )
    _write_empty_schema(
        FEATURE_ENGINEERING / "factor_horizon_profile.csv"
        if not args.out_root
        else Path(args.out_root) / "feature_engineering" / "factor_horizon_profile.csv",
        ["factor", "family"]
        + ["IC_{}D".format(h) for h in (1, 3, 5, 10, 20)]
        + ["peak_horizon", "peak_ic", "sign_stability", "approx_half_life"],
    )
    _write_empty_schema(
        FEATURE_ENGINEERING / "ic_horizon_matrix.csv"
        if not args.out_root
        else Path(args.out_root) / "feature_engineering" / "ic_horizon_matrix.csv",
        ["factor", "IC_1D", "IC_3D", "IC_5D", "IC_10D", "IC_20D"],
    )
    for h in (1, 3, 5, 10, 20):
        _write_empty_schema(
            fs_dir / "feature_selection_{}d.csv".format(h),
            [
                "factor",
                "family",
                "selection_count",
                "selection_methods",
                "selected",
                "nonlinear_keep_override",
                "tree_gain_without_confirmation",
            ]
            + list(EVIDENCE_METHODS),
            overwrite=True,
        )

    elapsed = time.perf_counter() - t0
    n_elig = int(inventory["eligible_for_fs"].astype(str).str.lower().isin(("true", "1")).sum())
    n_proposed = int((ratios["status"] == "PROPOSED").sum())
    print("L2 AI Stock Selection v1 — Phase A inventory")
    print("  contract:", AI_CONTRACT_VERSION)
    print("  n_formulas:", len(inventory))
    print("  n_eligible_for_fs:", n_elig)
    print("  n_ratio_already_in_pool:", int((ratios["status"] == "ALREADY_IN_POOL").sum()))
    print("  n_ratio_proposed:", n_proposed)
    print("  existing_huatai_aliases:", len(EXISTING_RATIO_ALIASES))
    print("  core_models: B0 equal-weight | B1 Ridge/ElasticNet | B2 LightGBM | B3 XGBoost")
    print("  optional_model: RandomForest")
    print("  elapsed_sec: {:.2f}".format(elapsed))
    print("  out:", inventory_path.parent)
    print("  db_scans: 0")
    print("  panel_rows_loaded: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
