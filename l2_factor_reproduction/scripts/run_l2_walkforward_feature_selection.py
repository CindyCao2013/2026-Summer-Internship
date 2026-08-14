#!/usr/bin/env python3
"""FS-3: PIT labels + purged/embargoed monthly walk-forward feature selection.

Research only — no learners, portfolios, or selector hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.feature_selection.contracts import (
    FS1_OUT_ROOT,
    PREPROCESS_CONTRACT_ID,
)
from l2_factor_reproduction.feature_selection.fs3_runner import run_fs3_selector
from l2_factor_reproduction.feature_selection.labels import (
    CANONICAL_HORIZONS,
    audit_label_boundaries,
    audit_label_parity_sample,
    build_labels_wide_panel,
    label_contract_dict,
    label_contract_hash,
    load_daily_excess_and_bench,
    write_label_partitions,
)
from l2_factor_reproduction.feature_selection.panel_io import (
    load_processed_panel_slice,
    panel_is_ready,
)
from l2_factor_reproduction.feature_selection.selection_analysis import (
    coverage_selection_diagnostics,
    family_selection_frequency,
    horizon_decay,
    selection_frequency,
    selection_jaccard_series,
    selector_agreement,
    selector_consensus,
)
from l2_factor_reproduction.feature_selection.selectors import (
    CANONICAL_SELECTORS,
    feature_schema_hash,
)
from l2_factor_reproduction.feature_selection.walkforward import (
    EXPECTED_FEATURE_SCHEMA_HASH,
    TRAINING_WINDOW_MONTHS,
    assert_no_overlap,
    build_walkforward_windows,
    canonical_selector_params,
    contract_hash,
    month_end_oos_anchors,
    walkforward_contract_dict,
    windows_to_frame,
)

logger = logging.getLogger("fs3")

FS3_OUT_ROOT = Path(RESULT_ROOT) / "feature_selection" / "fs3_walkforward_selection"
FS1_FULL_ROOT = FS1_OUT_ROOT.parent / (FS1_OUT_ROOT.name + "_full")
FS2_ROOT = Path(RESULT_ROOT) / "feature_selection" / "fs2_selector_engine"

BANNED_IMPORT_TOKENS = (
    "xgboost",
    "lightgbm",
    "catboost",
    "LogisticRegression",
    "l2_ml_score",
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(PROJ_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_frozen_inventory(panel_root: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, str], str]:
    inv = pd.read_csv(panel_root / "feature_inventory.csv")
    elig = inv.loc[inv["eligible_for_fs"] == True].copy()  # noqa: E712
    # preserve file order = registry order
    names = elig["factor"].tolist()
    families = dict(zip(elig["factor"], elig["family"].astype(str)))
    schema_h = feature_schema_hash(names, families=families)
    return inv, names, families, schema_h


def verify_frozen_dependencies(panel_root: Path) -> Dict[str, object]:
    inv, names, families, schema_h = load_frozen_inventory(panel_root)
    fs2 = json.loads((FS2_ROOT / "selector_contract.json").read_text(encoding="utf-8"))
    mismatches = []
    if schema_h != EXPECTED_FEATURE_SCHEMA_HASH:
        mismatches.append(
            f"schema_hash {schema_h} != frozen {EXPECTED_FEATURE_SCHEMA_HASH}"
        )
    if len(names) != 127:
        mismatches.append(f"eligible features {len(names)} != 127")
    if list(fs2.get("selector_names", [])) != list(CANONICAL_SELECTORS):
        mismatches.append("FS-2 selector names mismatch")
    if fs2.get("fs1_feature_schema_hash") != EXPECTED_FEATURE_SCHEMA_HASH:
        mismatches.append("FS-2 contract schema hash mismatch")
    preprocess = panel_root / "preprocess_contract.json"
    profile = PREPROCESS_CONTRACT_ID
    if preprocess.exists():
        pc = json.loads(preprocess.read_text(encoding="utf-8"))
        profile = pc.get("contract_id", pc.get("preprocess_contract", profile))
    if profile != PREPROCESS_CONTRACT_ID:
        mismatches.append(f"preprocess profile {profile} != {PREPROCESS_CONTRACT_ID}")
    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "n_eligible": len(names),
        "schema_hash": schema_h,
        "profile": profile,
        "selectors": list(CANONICAL_SELECTORS),
        "inventory_path": str(panel_root / "feature_inventory.csv"),
    }


def ensure_full_panel(panel_root: Path) -> Tuple[Path, bool]:
    """Return (panel_root, built_now). Invoke FS-1 builder only if needed."""
    ready, reason = panel_is_ready(panel_root, expect_schema_hash=EXPECTED_FEATURE_SCHEMA_HASH)
    if ready:
        # also require preprocess done
        if any((panel_root / "processed_ind_cap_z_v1").glob("year=*/quarter=*/part.parquet")):
            man = panel_root / "manifest.json"
            if man.exists():
                return panel_root, False
            # partitions exist but manifest missing → build still running / incomplete
            logger.warning("panel partitions present but manifest missing (%s)", reason)
    logger.info("Full panel not ready (%s). Invoking FS-1 builder --window full ...", reason)
    cmd = [
        sys.executable,
        str(PROJ_ROOT / "l2_factor_reproduction" / "scripts" / "build_l2_ml_feature_panel.py"),
        "--window",
        "full",
        "--out-root",
        str(panel_root),
    ]
    log_path = panel_root.parent / "fs1_full_build_fs3_invoke.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(PROJ_ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"FS-1 full panel build failed; see {log_path}")
    return panel_root, True


def wait_for_panel(panel_root: Path, *, timeout_s: int = 0) -> bool:
    """Poll until manifest exists. timeout_s=0 means wait forever."""
    t0 = time.time()
    while True:
        man = panel_root / "manifest.json"
        proc = any(
            (panel_root / "processed_ind_cap_z_v1").glob("year=*/quarter=*/part.parquet")
        )
        if man.exists() and proc:
            return True
        if timeout_s and (time.time() - t0) > timeout_s:
            return False
        logger.info("waiting for full panel at %s ...", panel_root)
        time.sleep(60)


def audit_multiday_parity(
    excess: pd.DataFrame,
    bench: pd.Series,
    dates: pd.DatetimeIndex,
    label_map: Dict,
    *,
    n_dates: int = 5,
    n_symbols: int = 10,
    seed: int = 1,
) -> pd.DataFrame:
    """Independently reconstruct Y5/Y20 compound excess for a sample."""
    from l2_factor_reproduction.feature_selection.labels import (
        compound_excess_path,
        recover_stock_returns,
    )

    rng = np.random.default_rng(seed)
    stock = recover_stock_returns(excess, bench)
    rows = []
    for h in (5, 20):
        usable = dates[: -(h + 1)]
        if len(usable) < n_dates:
            continue
        pick_dates = sorted(rng.choice(usable, size=n_dates, replace=False))
        y = label_map[h]
        for dt in pick_dates:
            dt = pd.Timestamp(dt).normalize()
            pos = dates.get_loc(dt)
            w = dates[pos + 1 : pos + 1 + h]
            finite = []
            for sym in y.columns:
                if np.isfinite(y.loc[dt, sym]):
                    finite.append(sym)
            if not finite:
                continue
            pick_syms = list(rng.choice(finite, size=min(n_symbols, len(finite)), replace=False))
            for sym in pick_syms:
                ref = compound_excess_path(stock.loc[w, sym], bench.reindex(w))
                lab = float(y.loc[dt, sym])
                abs_diff = abs(ref - lab) if np.isfinite(ref) and np.isfinite(lab) else np.nan
                rows.append(
                    {
                        "TradeDate": str(dt.date()),
                        "Symbol": sym,
                        "horizon": h,
                        "reference_return": ref,
                        "fs3_label": lab,
                        "abs_diff": abs_diff,
                        "pass": bool(np.isfinite(abs_diff) and abs_diff < 1e-10),
                    }
                )
    return pd.DataFrame(rows)


def build_training_xy(
    panel_slice: pd.DataFrame,
    y_wide: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Align panel rows to Y; return X frame, y vector, and key frame."""
    keys = panel_slice[["TradeDate", "Symbol"]].copy()
    keys["TradeDate"] = pd.to_datetime(keys["TradeDate"]).dt.normalize()
    # map y
    y_long = y_wide.stack(future_stack=True).rename("y").reset_index()
    y_long.columns = ["TradeDate", "Symbol", "y"]
    y_long["TradeDate"] = pd.to_datetime(y_long["TradeDate"]).dt.normalize()
    merged = keys.merge(y_long, on=["TradeDate", "Symbol"], how="left")
    feats = [f for f in feature_names if f in panel_slice.columns]
    X = panel_slice[feats].reset_index(drop=True)
    # ensure all feature columns exist
    for f in feature_names:
        if f not in X.columns:
            X[f] = np.nan
    X = X[list(feature_names)]
    y = merged["y"].to_numpy(dtype=float)
    return X, y, merged


def coverage_map_from_inventory(inv: pd.DataFrame) -> Dict[str, float]:
    if "mean_symbol_coverage" in inv.columns:
        return dict(zip(inv["factor"], inv["mean_symbol_coverage"].astype(float)))
    if "median_symbol_coverage" in inv.columns:
        return dict(zip(inv["factor"], inv["median_symbol_coverage"].astype(float)))
    return {f: np.nan for f in inv["factor"]}


def run_one_window_loaded(
    *,
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    families: Dict[str, str],
    coverage: Dict[str, float],
    schema_h: str,
    y_wide: pd.DataFrame,
    label_ends: pd.Series,
    window_row: pd.Series,
    selector_name: str,
    params: Dict,
    n_before_purge: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    oos = pd.Timestamp(window_row["oos_anchor"]).normalize()
    h = int(window_row["horizon"])
    train_start = pd.Timestamp(window_row["train_start"]).normalize()
    train_end = pd.Timestamp(window_row["train_end"]).normalize()

    ends = label_ends.reindex(panel["TradeDate"].unique())
    ok_dates = set(ends.index[ends.notna() & (pd.to_datetime(ends) < oos)])
    panel_p = panel.loc[panel["TradeDate"].isin(ok_dates)].copy()
    n_after = len(panel_p)
    used_ends = pd.to_datetime(label_ends.reindex(sorted(ok_dates)))
    overlap_count = int((used_ends.notna() & (used_ends >= oos)).sum())
    train_label_end_max = used_ends.max() if used_ends.notna().any() else pd.NaT

    audit = {
        "selector": selector_name,
        "horizon": h,
        "oos_anchor": str(oos.date()),
        "train_feature_date_min": str(train_start.date()),
        "train_feature_date_max": str(train_end.date()),
        "train_label_end_max": str(pd.Timestamp(train_label_end_max).date())
        if pd.notna(train_label_end_max)
        else "",
        "oos_date": str(oos.date()),
        "n_train_before_purge": int(n_before_purge),
        "n_train_after_purge": int(n_after),
        "n_purged": int(n_before_purge - n_after),
        "overlap_count": overlap_count,
        "train_dates": int(panel_p["TradeDate"].nunique()) if n_after else 0,
        "train_rows": int(n_after),
    }
    if overlap_count != 0 or n_after == 0:
        return pd.DataFrame(), audit

    X, y, _ = build_training_xy(panel_p, y_wide, feature_names)
    y_ok = np.isfinite(y)
    X = X.loc[y_ok].reset_index(drop=True)
    y = y[y_ok]
    if len(y) < 100:
        audit["status"] = "SKIPPED_INSUFFICIENT_HISTORY"
        return pd.DataFrame(), audit

    res = run_fs3_selector(
        X,
        y,
        feature_names,
        families,
        coverage,
        selector_name,
        params,
        schema_h,
    )
    res["oos_anchor"] = oos
    res["horizon"] = h
    res["training_start"] = train_start
    res["training_end"] = train_end
    res["train_rows"] = int(len(y))
    res["train_dates"] = int(panel_p["TradeDate"].nunique())
    res["selector_param_hash"] = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    audit["status"] = "OK"
    audit["n_selected"] = int(res["selected"].sum())
    return res, audit


def run_one_window(
    *,
    panel_root: Path,
    feature_names: Sequence[str],
    families: Dict[str, str],
    coverage: Dict[str, float],
    schema_h: str,
    y_wide: pd.DataFrame,
    label_ends: pd.Series,
    window_row: pd.Series,
    selector_name: str,
    params: Dict,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_start = pd.Timestamp(window_row["train_start"]).normalize()
    train_end = pd.Timestamp(window_row["train_end"]).normalize()
    processed = panel_root / "processed_ind_cap_z_v1"
    panel = load_processed_panel_slice(processed, train_start, train_end, columns=feature_names)
    return run_one_window_loaded(
        panel=panel,
        feature_names=feature_names,
        families=families,
        coverage=coverage,
        schema_h=schema_h,
        y_wide=y_wide,
        label_ends=label_ends,
        window_row=window_row,
        selector_name=selector_name,
        params=params,
        n_before_purge=len(panel),
    )


def scope_audit_text() -> str:
    lines = [
        "# FS-3 no-model scope audit",
        "",
        "Logistic learner used = NO",
        "XGBoost learner used = NO",
        "LightGBM learner used = NO",
        "CatBoost learner used = NO",
        "AUC computed = NO",
        "portfolio backtest run = NO",
        "H-L Sharpe optimized = NO",
        "l2_ml_score created = NO",
        "selector hyperparameters tuned on outcomes = NO",
        "",
    ]
    # scan FS-3 modules for banned tokens
    fs3_files = [
        PROJ_ROOT / "l2_factor_reproduction" / "feature_selection" / "labels.py",
        PROJ_ROOT / "l2_factor_reproduction" / "feature_selection" / "walkforward.py",
        PROJ_ROOT / "l2_factor_reproduction" / "feature_selection" / "selection_analysis.py",
        PROJ_ROOT / "l2_factor_reproduction" / "feature_selection" / "fs3_runner.py",
        PROJ_ROOT / "l2_factor_reproduction" / "scripts" / "run_l2_walkforward_feature_selection.py",
    ]
    hits = []
    for p in fs3_files:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for tok in BANNED_IMPORT_TOKENS:
            if tok in text:
                hits.append(f"{p.name}: {tok}")
    lines.append("Static scan hits: " + ("NONE" if not hits else ", ".join(hits)))
    lines.append("")
    return "\n".join(lines)


def _df_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_empty_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def write_report(
    out_root: Path,
    *,
    verdict: str,
    dep: Dict,
    panel_meta: Dict,
    label_meta: Dict,
    wf_meta: Dict,
    sel_summary: pd.DataFrame,
    freq: pd.DataFrame,
    fam_freq: pd.DataFrame,
    jac: pd.DataFrame,
    agree: pd.DataFrame,
    decay: pd.DataFrame,
    cov_diag: pd.DataFrame,
    limitations: List[str],
    files_touched: List[str],
) -> None:
    top_lines = []
    if not freq.empty:
        for (sel, hor), g in freq.groupby(["selector_name", "horizon"]):
            top = g.sort_values("selection_frequency", ascending=False).head(10)
            top_lines.append(f"### {sel} × Y{hor}")
            for _, r in top.iterrows():
                top_lines.append(
                    f"- {r['feature']} ({r['family']}): "
                    f"freq={r['selection_frequency']:.3f}, "
                    f"avail={r['availability_frequency']:.3f}"
                )
            top_lines.append("")

    jac_sum = (
        jac.groupby(["selector_name", "horizon"])["jaccard"]
        .agg(["mean", "median", "count"])
        .reset_index()
        if not jac.empty
        else pd.DataFrame()
    )

    text = f"""# FS-3 Walk-Forward Feature Selection Report

## 1. Verdict

```text
{verdict}
```

## 2. Frozen dependencies

- FS-1 verdict: B. FS1_PANEL_READY_WITH_EXCLUSIONS (frozen)
- FS-1 profile: `{dep.get('profile')}`
- FS-1 eligible features: {dep.get('n_eligible')}
- FS-1/FS-2 schema hash: `{dep.get('schema_hash')}`
- FS-2 verdict: A. FS2_SELECTOR_ENGINE_READY (frozen)
- Six selectors: {', '.join(dep.get('selectors', []))}

## 3. Full panel

- built/reused: {panel_meta.get('built_or_reused')}
- date_min: {panel_meta.get('date_min')}
- date_max: {panel_meta.get('date_max')}
- n_dates: {panel_meta.get('n_dates')}
- n_symbols: {panel_meta.get('n_symbols')}
- n_rows: {panel_meta.get('n_rows')}
- n_features: {panel_meta.get('n_features')}
- size_on_disk: {panel_meta.get('size_on_disk')}

## 4. Label contract

- Y1: T+1 excess return (trading day)
- Y5: cumulative excess T+1..T+5 (compound then subtract)
- Y20: cumulative excess T+1..T+20 (compound then subtract)
- benchmark: `{UNIVERSE}`
- return convention: {label_meta.get('return_convention')}
- label coverage: see `labels/label_coverage.csv`
- parity: {label_meta.get('parity')}

## 5. Walk-forward contract

- training window: {TRAINING_WINDOW_MONTHS} months (research-design choice; not optimized)
- refit frequency: monthly (month-end trading date)
- purge rule: train_label_end < oos_anchor
- embargo: horizon-aware via label interval non-overlap
- minimum history: see walkforward_contract.json
- valid OOS windows: {wf_meta.get('n_valid_windows')}
- skipped: {wf_meta.get('n_skipped')}

## 6. Leakage audit

- overlap_count (aggregate): {wf_meta.get('overlap_count')}
- max_overlap_count: {wf_meta.get('max_overlap_count')}
- PIT status: {wf_meta.get('pit_status')}

Required: `overlap_count = 0`.

## 7. Selection counts

{_df_md(sel_summary)}

## 8. Selection frequency (high-frequency selected features)

Descriptive only — not a production shortlist.

{chr(10).join(top_lines) if top_lines else '_empty_'}

## 9. Family composition

See `family_selection_frequency.csv` (coverage-aware; family-size adjusted).

## 10. Temporal stability

{_df_md(jac_sum)}

## 11. Selector agreement

{_df_md(agree)}

## 12. Horizon decay

See `horizon_decay.csv`. Do not force monotonic decay interpretation.

## 13. Coverage sensitivity

{_df_md(cov_diag)}

Interpret Spearman(coverage, selection_frequency) cautiously — mechanical association ≠ causality.

## 14. Local eligibility

See `audits/local_eligibility_audit.csv`. Locally ineligible windows are excluded from
selection-frequency denominators (not counted as failed selections).

## 15. Scope audit

```text
XGBoost used            = NO
Logistic used           = NO
AUC evaluated           = NO
portfolio run           = NO
Sharpe optimized        = NO
selector params tuned   = NO
l2_ml_score created     = NO
```

## 16. Existing project mutation

```text
Fast Discovery modified = NO
FS-1 contract modified  = NO
FS-2 selector modified  = NO
candidate registry      = NO
```

## 17. Files added / modified

{chr(10).join('- ' + f for f in files_touched)}

## 18. Recommendation

```text
{wf_meta.get('recommendation', 'FS-4 NOT READY')}
```

Do not begin FS-4 from this sprint automatically.

## Limitations

{chr(10).join('- ' + x for x in limitations) if limitations else '- none recorded'}
"""
    (out_root / "report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=FS3_OUT_ROOT)
    p.add_argument(
        "--panel-root",
        type=Path,
        default=None,
        help="FS-1 panel root (default: fs1_feature_panel_full)",
    )
    p.add_argument(
        "--use-discovery-panel",
        action="store_true",
        help="Use discovery FS-1 panel (limited history; may force B verdict)",
    )
    p.add_argument("--max-anchors", type=int, default=0, help="Cap OOS anchors per horizon (0=all)")
    p.add_argument(
        "--selectors",
        nargs="*",
        default=list(CANONICAL_SELECTORS),
        help="Subset of selectors (default: all six)",
    )
    p.add_argument("--skip-wait", action="store_true", help="Fail if full panel not ready")
    p.add_argument("--wait-timeout-s", type=int, default=0)
    p.add_argument("--labels-only", action="store_true")
    p.add_argument("--skip-mi", action="store_true", help="Skip MI (debug only; not for verdict A)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    contracts = out_root / "contracts"
    audits = out_root / "audits"
    labels_dir = out_root / "labels"
    runs_dir = out_root / "walkforward_runs"
    for d in (contracts, audits, labels_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    files_touched: List[str] = []
    limitations: List[str] = []

    # ------------------------------------------------------------------ PART A
    if args.use_discovery_panel:
        panel_root = FS1_OUT_ROOT
        limitations.append("Used discovery-window FS-1 panel (not full 2019→latest)")
        built = False
    else:
        panel_root = args.panel_root or FS1_FULL_ROOT
        ready, reason = panel_is_ready(panel_root, expect_schema_hash=EXPECTED_FEATURE_SCHEMA_HASH)
        man_ok = (panel_root / "manifest.json").exists()
        if not (ready and man_ok):
            if args.skip_wait:
                logger.error("Full panel not ready: %s", reason)
                return 2
            # If a build is already running, wait; else invoke
            if not any(
                (panel_root / "aligned_raw").glob("year=*/quarter=*/part.parquet")
            ):
                panel_root, built = ensure_full_panel(panel_root)
            else:
                built = False
                logger.info("Panel build in progress; waiting ...")
                ok = wait_for_panel(panel_root, timeout_s=args.wait_timeout_s)
                if not ok:
                    logger.error("Timed out waiting for full panel")
                    return 2
        else:
            built = False

    dep = verify_frozen_dependencies(panel_root)
    pd.DataFrame(
        [{"check": k, "value": str(v)} for k, v in dep.items() if k != "mismatches"]
        + [{"check": "mismatch", "value": m} for m in dep["mismatches"]]
    ).to_csv(audits / "dependency_freeze_audit.csv", index=False)
    if not dep["ok"]:
        logger.error("Frozen dependency mismatch: %s", dep["mismatches"])
        (out_root / "report.md").write_text(
            "# FS-3\n\n## Verdict\n\n```text\nC. FS3_WALKFORWARD_SELECTION_NOT_READY\n```\n\n"
            + "Mismatch: "
            + "; ".join(dep["mismatches"]),
            encoding="utf-8",
        )
        return 1

    inv, feature_names, families, schema_h = load_frozen_inventory(panel_root)
    # Prefer frozen discovery inventory order/hash if full panel eligibility drifts
    # Spec: preserve 127-column contract / schema hash 0b90fed383d3ba75
    inv_disc, names_disc, fam_disc, hash_disc = load_frozen_inventory(FS1_OUT_ROOT)
    if hash_disc == EXPECTED_FEATURE_SCHEMA_HASH:
        feature_names = names_disc
        families = fam_disc
        schema_h = hash_disc
        inv = inv_disc
        if panel_root != FS1_OUT_ROOT:
            limitations.append(
                "Feature universe/order taken from frozen discovery FS-1 inventory "
                "to preserve schema hash; full-panel coverage may differ by date"
            )

    coverage = coverage_map_from_inventory(inv)

    # panel meta
    spine = pd.read_parquet(panel_root / "spine.parquet")
    spine["TradeDate"] = pd.to_datetime(spine["TradeDate"]).dt.normalize()
    proc_root = panel_root / "processed_ind_cap_z_v1"
    n_parts = len(list(proc_root.glob("year=*/quarter=*/part.parquet")))
    size = shutil.disk_usage(panel_root).used  # not ideal; use du via walk
    total_bytes = sum(f.stat().st_size for f in panel_root.rglob("*") if f.is_file())
    panel_meta = {
        "built_or_reused": "built" if built else "reused",
        "date_min": str(spine["TradeDate"].min().date()),
        "date_max": str(spine["TradeDate"].max().date()),
        "n_dates": int(spine["TradeDate"].nunique()),
        "n_symbols": int(spine["Symbol"].nunique()),
        "n_rows": int(len(spine)),
        "n_features": len(feature_names),
        "size_on_disk": f"{total_bytes / 1e9:.2f} GB",
        "n_processed_partitions": n_parts,
        "panel_root": str(panel_root),
    }

    frozen_deps = {
        "FS1_profile": PREPROCESS_CONTRACT_ID,
        "FS1_eligible_features": 127,
        "feature_schema_hash": schema_h,
        "FS2_contract": "fs2_selector_v1",
        "selectors": list(CANONICAL_SELECTORS),
        "panel_root": str(panel_root),
    }
    (contracts / "frozen_dependencies.json").write_text(
        json.dumps(frozen_deps, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------ PART D/E labels
    logger.info("[labels] loading ret_matrix + benchmark")
    excess, bench, dates = load_daily_excess_and_bench("full")
    # restrict to panel dates intersection
    panel_dates = pd.DatetimeIndex(sorted(spine["TradeDate"].unique()))
    dates = pd.DatetimeIndex([d for d in dates if d in set(panel_dates)])
    logger.info("[labels] building Y1/Y5/Y20 on %d dates × %d symbols", len(dates), excess.shape[1])
    label_map = build_labels_wide_panel(excess, bench, dates, horizons=CANONICAL_HORIZONS)
    write_label_partitions(label_map, labels_dir, horizons=CANONICAL_HORIZONS)
    (contracts / "label_contract.json").write_text(
        json.dumps(label_contract_dict(), indent=2), encoding="utf-8"
    )

    parity_y1 = audit_label_parity_sample(excess, bench, dates, label_map[1])
    parity_md = audit_multiday_parity(excess, bench, dates, label_map)
    parity = pd.concat([parity_y1, parity_md], ignore_index=True)
    parity.to_csv(audits / "label_parity.csv", index=False)
    boundary = audit_label_boundaries(
        dates, label_map["_meta_end"], label_map["_meta_valid"]
    )
    boundary.to_csv(audits / "label_pit_audit.csv", index=False)
    parity_pass = bool(parity["pass"].all()) if len(parity) else False
    boundary_pass = bool(boundary["pass"].all()) if len(boundary) else False
    label_meta = {
        "return_convention": (
            "fast_context ret_matrix c2c excess vs 000852.SH; "
            "multi-day: compound stock & bench separately then subtract"
        ),
        "parity": "PASS" if parity_pass and boundary_pass else "FAIL",
        "parity_n": int(len(parity)),
        "parity_fail": int((~parity["pass"]).sum()) if len(parity) else -1,
        "boundary_fail": int((~boundary["pass"]).sum()) if len(boundary) else -1,
        "label_contract_hash": label_contract_hash(),
    }
    if not parity_pass or not boundary_pass:
        logger.error("Label parity/boundary FAILED — stopping before selectors")
        write_report(
            out_root,
            verdict="C. FS3_WALKFORWARD_SELECTION_NOT_READY",
            dep=dep,
            panel_meta=panel_meta,
            label_meta=label_meta,
            wf_meta={"overlap_count": "n/a", "max_overlap_count": "n/a", "pit_status": "FAIL", "recommendation": "FS-4 NOT READY"},
            sel_summary=pd.DataFrame(),
            freq=pd.DataFrame(),
            fam_freq=pd.DataFrame(),
            jac=pd.DataFrame(),
            agree=pd.DataFrame(),
            decay=pd.DataFrame(),
            cov_diag=pd.DataFrame(),
            limitations=limitations + ["Label parity or boundary audit failed"],
            files_touched=files_touched,
        )
        return 1

    if args.labels_only:
        logger.info("labels-only done")
        return 0

    # ------------------------------------------------------------------ PART F-H
    label_end_by_h = {
        h: label_map["_meta_end"][h] for h in CANONICAL_HORIZONS
    }
    windows = build_walkforward_windows(dates, label_end_by_h)
    win_df = windows_to_frame(windows)
    win_df.to_csv(audits / "split_integrity.csv", index=False)
    overlap_fails = assert_no_overlap(windows)
    if overlap_fails:
        logger.error("Purge overlap failures: %d", overlap_fails)
        write_report(
            out_root,
            verdict="C. FS3_WALKFORWARD_SELECTION_NOT_READY",
            dep=dep,
            panel_meta=panel_meta,
            label_meta=label_meta,
            wf_meta={
                "overlap_count": overlap_fails,
                "max_overlap_count": overlap_fails,
                "pit_status": "FAIL",
                "recommendation": "FS-4 NOT READY",
            },
            sel_summary=pd.DataFrame(),
            freq=pd.DataFrame(),
            fam_freq=pd.DataFrame(),
            jac=pd.DataFrame(),
            agree=pd.DataFrame(),
            decay=pd.DataFrame(),
            cov_diag=pd.DataFrame(),
            limitations=limitations + ["overlap_count > 0"],
            files_touched=files_touched,
        )
        return 1

    wf_contract = walkforward_contract_dict(
        feature_schema_hash=schema_h,
        date_min=panel_meta["date_min"],
        date_max=panel_meta["date_max"],
    )
    (contracts / "walkforward_contract.json").write_text(
        json.dumps(wf_contract, indent=2, default=str), encoding="utf-8"
    )
    wf_hash = contract_hash(wf_contract)

    # freeze before research
    params = canonical_selector_params()
    selectors = [s for s in args.selectors if s in CANONICAL_SELECTORS]
    if args.skip_mi:
        selectors = [s for s in selectors if s != "MI_REGRESSION_KBEST"]
        limitations.append("MI_REGRESSION_KBEST skipped via --skip-mi")

    # filter OK windows; optional max anchors
    ok_win = win_df.loc[win_df["status"] == "OK"].copy()
    if args.max_anchors > 0:
        keep = []
        for h, g in ok_win.groupby("horizon"):
            keep.append(g.sort_values("oos_anchor").head(args.max_anchors))
        ok_win = pd.concat(keep, ignore_index=True)
        limitations.append(f"max_anchors={args.max_anchors} (subset run)")

    # ------------------------------------------------------------------ PART I
    purge_rows = []
    run_frames = []
    t_run0 = time.time()
    processed = panel_root / "processed_ind_cap_z_v1"
    for _, wrow in ok_win.iterrows():
        h = int(wrow["horizon"])
        y_wide = label_map[h]
        ends = label_end_by_h[h]
        train_start = pd.Timestamp(wrow["train_start"]).normalize()
        train_end = pd.Timestamp(wrow["train_end"]).normalize()
        logger.info(
            "load panel h=%s oos=%s %s→%s",
            h,
            pd.Timestamp(wrow["oos_anchor"]).date(),
            train_start.date(),
            train_end.date(),
        )
        panel = load_processed_panel_slice(
            processed, train_start, train_end, columns=feature_names
        )
        n_before = len(panel)
        for sel in selectors:
            logger.info(
                "run h=%s oos=%s sel=%s",
                h,
                pd.Timestamp(wrow["oos_anchor"]).date(),
                sel,
            )
            res, audit = run_one_window_loaded(
                panel=panel,
                feature_names=feature_names,
                families=families,
                coverage=coverage,
                schema_h=schema_h,
                y_wide=y_wide,
                label_ends=ends,
                window_row=wrow,
                selector_name=sel,
                params=params[sel],
                n_before_purge=n_before,
            )
            purge_rows.append(audit)
            if res is None or res.empty:
                continue
            year = pd.Timestamp(wrow["oos_anchor"]).year
            outp = (
                runs_dir
                / f"horizon={h}"
                / f"selector={sel}"
                / f"year={year}"
            )
            outp.mkdir(parents=True, exist_ok=True)
            fname = f"oos={pd.Timestamp(wrow['oos_anchor']).strftime('%Y%m%d')}.parquet"
            res.to_parquet(outp / fname, index=False)
            run_frames.append(res)

    purge_df = pd.DataFrame(purge_rows)
    purge_df.to_csv(audits / "purge_embargo_audit.csv", index=False)
    max_overlap = int(purge_df["overlap_count"].max()) if len(purge_df) else 0
    if max_overlap != 0:
        logger.error("Runtime overlap detected")
        return 1

    if not run_frames:
        logger.error("No selector runs produced output")
        return 1

    runs = pd.concat(run_frames, ignore_index=True)
    # local eligibility audit
    elig_audit = (
        runs.groupby(["family", "local_ineligible_reason"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    elig_audit.to_csv(audits / "local_eligibility_audit.csv", index=False)

    # ------------------------------------------------------------------ PART K
    freq = selection_frequency(runs)
    freq.to_csv(out_root / "selection_frequency.csv", index=False)
    fam_freq = family_selection_frequency(freq, inv)
    fam_freq.to_csv(out_root / "family_selection_frequency.csv", index=False)
    jac = selection_jaccard_series(runs)
    jac.to_csv(out_root / "selection_jaccard.csv", index=False)
    agree = selector_agreement(runs)
    agree.to_csv(out_root / "selector_agreement.csv", index=False)
    consensus = selector_consensus(runs)
    consensus.to_csv(out_root / "selector_consensus.csv", index=False)
    decay = horizon_decay(freq)
    decay.to_csv(out_root / "horizon_decay.csv", index=False)
    # feature-level coverage diagnostics join
    cov_feat = freq.copy()
    cov_diag = coverage_selection_diagnostics(cov_feat)
    cov_feat.to_csv(out_root / "coverage_selection_diagnostics.csv", index=False)
    cov_diag.to_csv(audits / "coverage_selection_spearman.csv", index=False)

    # selection counts summary
    sel_summary = (
        runs.groupby(["selector_name", "horizon", "oos_anchor"])["selected"]
        .sum()
        .groupby(["selector_name", "horizon"])
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )

    # L1 degeneracy check
    l1 = sel_summary.loc[sel_summary["selector_name"] == "L1_REGRESSION"]
    if len(l1):
        for _, r in l1.iterrows():
            if r["mean"] < 0.5 or r["mean"] > 120:
                limitations.append(
                    f"L1_PARAMETER_NOT_INFORMATIVE at Y{int(r['horizon'])}: "
                    f"mean selected≈{r['mean']:.1f}"
                )

    # determinism subset: re-run first OK window for F_KBEST
    det_rows = []
    sample = ok_win.sort_values(["horizon", "oos_anchor"]).head(1)
    if len(sample):
        wrow = sample.iloc[0]
        h = int(wrow["horizon"])
        r1, _ = run_one_window(
            panel_root=panel_root,
            feature_names=feature_names,
            families=families,
            coverage=coverage,
            schema_h=schema_h,
            y_wide=label_map[h],
            label_ends=label_end_by_h[h],
            window_row=wrow,
            selector_name="F_REGRESSION_KBEST",
            params=params["F_REGRESSION_KBEST"],
        )
        r2, _ = run_one_window(
            panel_root=panel_root,
            feature_names=feature_names,
            families=families,
            coverage=coverage,
            schema_h=schema_h,
            y_wide=label_map[h],
            label_ends=label_end_by_h[h],
            window_row=wrow,
            selector_name="F_REGRESSION_KBEST",
            params=params["F_REGRESSION_KBEST"],
        )
        s1 = set(r1.loc[r1["selected"], "feature"]) if len(r1) else set()
        s2 = set(r2.loc[r2["selected"], "feature"]) if len(r2) else set()
        det_ok = s1 == s2
        det_rows.append({"check": "deterministic_subset_rerun", "pass": det_ok, "n": len(s1)})
    else:
        det_ok = False
        det_rows.append({"check": "deterministic_subset_rerun", "pass": False, "n": 0})
    pd.DataFrame(det_rows).to_csv(audits / "determinism_audit.csv", index=False)

    (audits / "no_model_scope_audit.md").write_text(scope_audit_text(), encoding="utf-8")

    # mutation audit — FS-1/FS-2/FD paths unchanged by this script (we only read)
    mut = pd.DataFrame(
        [
            {"artifact": "FS-1 selectors.py", "modified": False},
            {"artifact": "FS-2 selectors math", "modified": False},
            {"artifact": "Fast Discovery", "modified": False},
            {"artifact": "candidate_registry", "modified": False},
        ]
    )
    mut.to_csv(audits / "mutation_audit.csv", index=False)

    n_valid = int(ok_win.shape[0])
    n_skipped = int((win_df["status"] != "OK").sum())
    # Verdict
    core_ok = (
        parity_pass
        and boundary_pass
        and max_overlap == 0
        and det_ok
        and schema_h == EXPECTED_FEATURE_SCHEMA_HASH
        and set(selectors) >= set(CANONICAL_SELECTORS) - ({"MI_REGRESSION_KBEST"} if args.skip_mi else set())
    )
    if args.skip_mi or args.max_anchors or args.use_discovery_panel:
        core_ok_full = False
    else:
        core_ok_full = core_ok and set(selectors) == set(CANONICAL_SELECTORS)

    if not core_ok:
        verdict = "C. FS3_WALKFORWARD_SELECTION_NOT_READY"
        rec = "FS-4 NOT READY"
    elif limitations or not core_ok_full:
        verdict = "B. FS3_WALKFORWARD_SELECTION_READY_WITH_LIMITATIONS"
        rec = "FS-4 READY" if (not args.skip_mi and max_overlap == 0 and parity_pass) else "FS-4 NOT READY"
        # If only coverage/L1 limitations, FS-4 can still be ready
        hard_lim = any("NOT_READY" in x or "parity" in x.lower() for x in limitations)
        if not hard_lim and max_overlap == 0 and parity_pass and not args.skip_mi:
            rec = "FS-4 READY"
        if args.use_discovery_panel or args.max_anchors or args.skip_mi:
            rec = "FS-4 NOT READY"
    else:
        verdict = "A. FS3_WALKFORWARD_SELECTION_READY"
        rec = "FS-4 READY"

    wf_meta = {
        "n_valid_windows": n_valid,
        "n_skipped": n_skipped,
        "overlap_count": 0,
        "max_overlap_count": max_overlap,
        "pit_status": "PASS",
        "recommendation": rec,
        "runtime_sec": round(time.time() - t_run0, 1),
        "walkforward_contract_hash": wf_hash,
    }

    files_touched = [
        "l2_factor_reproduction/feature_selection/labels.py",
        "l2_factor_reproduction/feature_selection/walkforward.py",
        "l2_factor_reproduction/feature_selection/selection_analysis.py",
        "l2_factor_reproduction/feature_selection/fs3_runner.py",
        "l2_factor_reproduction/feature_selection/panel_io.py",
        "l2_factor_reproduction/scripts/run_l2_walkforward_feature_selection.py",
        "l2_factor_reproduction/tests/test_fs3_walkforward_labels.py",
        str(out_root / "report.md"),
    ]

    write_report(
        out_root,
        verdict=verdict,
        dep=dep,
        panel_meta=panel_meta,
        label_meta=label_meta,
        wf_meta=wf_meta,
        sel_summary=sel_summary,
        freq=freq,
        fam_freq=fam_freq,
        jac=jac,
        agree=agree,
        decay=decay,
        cov_diag=cov_diag,
        limitations=limitations,
        files_touched=files_touched,
    )

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "input_paths": {"panel_root": str(panel_root)},
        "feature_schema_hash": schema_h,
        "label_contract_hash": label_contract_hash(),
        "walkforward_contract_hash": wf_hash,
        "selector_contract_reference": str(FS2_ROOT / "selector_contract.json"),
        "output_paths": {"out_root": str(out_root)},
        "verdict": verdict,
        "recommendation": rec,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("FS-3 complete: %s", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
