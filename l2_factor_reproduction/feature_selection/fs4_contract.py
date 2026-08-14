"""FS-4 Fast Track contracts and FS-3 artifact loaders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.feature_selection.walkforward import EXPECTED_FEATURE_SCHEMA_HASH

FS3_ROOT = Path(RESULT_ROOT) / "feature_selection" / "fs3_walkforward_selection"
FS1_FULL = Path(RESULT_ROOT) / "feature_selection" / "fs1_feature_panel_full"
FS1_DISC = Path(RESULT_ROOT) / "feature_selection" / "fs1_feature_panel"
FS4_ROOT = Path(RESULT_ROOT) / "feature_selection" / "fs4_fast_track"
FS5_ROOT = Path(RESULT_ROOT) / "feature_selection" / "fs5_fast_validation"

HORIZON = 5
TRAINING_WINDOW_MONTHS = 24
MAX_NAMES_PER_TRAIN_DATE = 500
SCREEN_END = pd.Timestamp("2024-12-31")
CONFIRM_START = pd.Timestamp("2025-01-01")
REFIT_EVERY_N = 3
RANDOM_SEED = 42
SCHEMA_HASH = EXPECTED_FEATURE_SCHEMA_HASH

ROUTES_STAGE1 = ("ALL_127", "F_KBEST_60", "MI_KBEST_60", "TREE_SELECTED")
SELECTOR_FOR_ROUTE = {
    "ALL_127": None,
    "F_KBEST_60": "F_REGRESSION_KBEST",
    "MI_KBEST_60": "MI_REGRESSION_KBEST",
    "TREE_SELECTED": "TREE_IMPORTANCE_REGRESSION",
}

RIDGE_PARAMS = {"alpha": 1.0, "fit_intercept": True, "solver": "lsqr"}
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "max_bin": 128,
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
    "early_stopping_rounds": 30,
}
XGB_VAL_MONTHS = 2


def fast_track_contract() -> Dict[str, object]:
    return {
        "contract_version": "fs4_fast_track_v1",
        "horizon": HORIZON,
        "training_window_months": TRAINING_WINDOW_MONTHS,
        "screen_end": str(SCREEN_END.date()),
        "confirm_start": str(CONFIRM_START.date()),
        "model_refit_frequency": f"every_{REFIT_EVERY_N}_monthly_anchors",
        "max_names_per_train_date": MAX_NAMES_PER_TRAIN_DATE,
        "feature_routes_stage1": list(ROUTES_STAGE1),
        "excluded_routes": ["F_REGRESSION_FPR", "F_REGRESSION_FDR", "L1_REGRESSION"],
        "stage1_learner": {"name": "RIDGE_FAST_V1", "params": RIDGE_PARAMS},
        "stage2_learner": {"name": "XGB_FAST_V1", "params": XGB_PARAMS},
        "xgb_val_months": XGB_VAL_MONTHS,
        "survivor_rule_A_delta_rankic": 0.002,
        "survivor_rule_B_rankic_tol": 0.002,
        "survivor_rule_B_max_feature_frac": 0.60,
        "survivor_pos_ic_tol": 0.05,
        "max_selected_survivors": 2,
        "schema_hash": SCHEMA_HASH,
        "random_seed": RANDOM_SEED,
        "benchmark": UNIVERSE,
        "missing_policy": "complete-case on route feature subset; no zero-fill",
    }


def contract_hash(contract: Dict[str, object]) -> str:
    payload = json.dumps(contract, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def load_ordered_features() -> Tuple[List[str], Dict[str, str]]:
    inv = pd.read_csv(FS1_DISC / "feature_inventory.csv")
    elig = inv.loc[inv["eligible_for_fs"] == True]  # noqa: E712
    names = elig["factor"].tolist()
    fam = dict(zip(elig["factor"], elig["family"].astype(str)))
    return names, fam


def load_y5_ok_windows() -> pd.DataFrame:
    split = pd.read_csv(FS3_ROOT / "audits" / "split_integrity.csv")
    w = split.loc[(split["horizon"] == HORIZON) & (split["status"] == "OK")].copy()
    w["oos_anchor"] = pd.to_datetime(w["oos_anchor"]).dt.normalize()
    w["train_start"] = pd.to_datetime(w["train_start"]).dt.normalize()
    w["train_end"] = pd.to_datetime(w["train_end"]).dt.normalize()
    w["train_label_end_max"] = pd.to_datetime(w["train_label_end_max"]).dt.normalize()
    return w.sort_values("oos_anchor").reset_index(drop=True)


def build_refit_schedule(windows: pd.DataFrame) -> pd.DataFrame:
    """Every REFIT_EVERY_N-th OK monthly OOS is a model refit; others score-only."""
    rows = []
    for i, r in windows.iterrows():
        is_refit = (i % REFIT_EVERY_N) == 0
        refit_idx = (i // REFIT_EVERY_N) * REFIT_EVERY_N
        refit_anchor = windows.iloc[refit_idx]["oos_anchor"]
        rows.append(
            {
                "oos_anchor": r["oos_anchor"],
                "train_start": windows.iloc[refit_idx]["train_start"] if is_refit else pd.NaT,
                "train_end": windows.iloc[refit_idx]["train_end"] if is_refit else pd.NaT,
                "train_label_end_max": windows.iloc[refit_idx]["train_label_end_max"]
                if is_refit
                else pd.NaT,
                "is_refit": is_refit,
                "refit_anchor": pd.Timestamp(refit_anchor).normalize(),
                "period": "SCREEN"
                if pd.Timestamp(r["oos_anchor"]) <= SCREEN_END
                else "CONFIRM",
            }
        )
    # fill train bounds for score-only from their refit row
    df = pd.DataFrame(rows)
    refit_map = (
        df.loc[df["is_refit"], ["refit_anchor", "train_start", "train_end", "train_label_end_max"]]
        .drop_duplicates("refit_anchor")
        .set_index("refit_anchor")
    )
    for col in ("train_start", "train_end", "train_label_end_max"):
        df[col] = df.apply(
            lambda r, c=col: r[c]
            if r["is_refit"]
            else refit_map.loc[r["refit_anchor"], c],
            axis=1,
        )
    return df


def load_selected_mask(selector: Optional[str], oos_anchor: pd.Timestamp, all_features: Sequence[str]) -> List[str]:
    if selector is None:
        return list(all_features)
    oos = pd.Timestamp(oos_anchor).normalize()
    year = oos.year
    path = (
        FS3_ROOT
        / "walkforward_runs"
        / f"horizon={HORIZON}"
        / f"selector={selector}"
        / f"year={year}"
        / f"oos={oos.strftime('%Y%m%d')}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"FS-3 mask missing: {path}")
    tab = pd.read_parquet(path)
    sel = tab.loc[tab["selected"] == True, "feature"].tolist()  # noqa: E712
    # preserve frozen order
    order = {f: i for i, f in enumerate(all_features)}
    sel = sorted(sel, key=lambda f: order.get(f, 10**9))
    return sel


def deterministic_sample_symbols(
    symbols: Sequence[str],
    *,
    k: int = MAX_NAMES_PER_TRAIN_DATE,
    seed: int = RANDOM_SEED,
) -> List[str]:
    """Stable subsample of symbols (not process-randomized hash)."""
    syms = sorted(map(str, symbols))
    if len(syms) <= k:
        return syms
    # hash each symbol with seed via sha1 for stability across processes
    scored = []
    for s in syms:
        h = hashlib.sha1(f"{seed}|{s}".encode()).hexdigest()
        scored.append((h, s))
    scored.sort()
    return [s for _, s in scored[:k]]
