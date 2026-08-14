#!/usr/bin/env python3
"""FS-4 Fast Track — cheap Ridge screen → XGB survivors → 2025+ holdout.

Consumes FS-3 masks/labels/walk-forward. No selector reruns. No tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.feature_selection.fs4_contract import (
    CONFIRM_START,
    FS1_FULL,
    FS3_ROOT,
    FS4_ROOT,
    HORIZON,
    ROUTES_STAGE1,
    SCHEMA_HASH,
    SCREEN_END,
    SELECTOR_FOR_ROUTE,
    build_refit_schedule,
    contract_hash,
    fast_track_contract,
    load_ordered_features,
    load_selected_mask,
    load_y5_ok_windows,
)
from l2_factor_reproduction.feature_selection.fs4_fast_cache import (
    complete_case_matrix,
    monthly_rank_ic,
    sample_training_frame,
    split_train_val_by_date,
    summarize_ic,
    survivor_gate,
)
from l2_factor_reproduction.feature_selection.learners import (
    fit_ridge,
    fit_xgb,
    predict_ridge,
    predict_xgb,
    xgb_available,
)
from l2_factor_reproduction.feature_selection.panel_io import load_processed_panel_slice
from l2_factor_reproduction.feature_selection.selectors import feature_schema_hash

logger = logging.getLogger("fs4")


def load_y5_wide() -> Tuple[pd.DataFrame, pd.Series]:
    y = pd.read_parquet(FS3_ROOT / "labels" / f"horizon={HORIZON}" / "y_wide.parquet")
    y.index = pd.to_datetime(y.index).normalize()
    bounds = pd.read_parquet(FS3_ROOT / "labels" / f"horizon={HORIZON}" / "label_bounds.parquet")
    bounds["TradeDate"] = pd.to_datetime(bounds["TradeDate"]).dt.normalize()
    ends = bounds.set_index("TradeDate")[f"label_end_{HORIZON}d"]
    return y, ends


def score_dates_for_month(
    schedule: pd.DataFrame,
    month_anchor: pd.Timestamp,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Score trading days (prev_anchor, curr_anchor]."""
    anchors = schedule["oos_anchor"].tolist()
    idx = anchors.index(pd.Timestamp(month_anchor).normalize())
    start = anchors[idx - 1] + pd.Timedelta(days=1) if idx > 0 else schedule.iloc[idx]["train_end"] + pd.Timedelta(days=1)
    end = anchors[idx]
    return pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()


def fit_predict_route(
    *,
    route: str,
    features: List[str],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    keys_tr: pd.DataFrame,
    X_oos: np.ndarray,
    learner: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    meta: Dict[str, object] = {"route": route, "learner": learner, "n_features": len(features)}
    if learner == "ridge":
        model = fit_ridge(X_tr, y_tr)
        pred = predict_ridge(model, X_oos)
        meta["model"] = "Ridge"
        return pred, meta
    if learner == "xgb":
        Xa, ya, Xv, yv = split_train_val_by_date(keys_tr, X_tr, y_tr)
        if len(ya) < 1000 or len(yv) < 200:
            # fallback: train all, no early stop path
            model, m2 = fit_xgb(X_tr, y_tr, X_tr[-5000:], y_tr[-5000:])
        else:
            model, m2 = fit_xgb(Xa, ya, Xv, yv)
        meta.update(m2)
        pred = predict_xgb(model, X_oos)
        return pred, meta
    raise ValueError(learner)


def run_period(
    *,
    schedule: pd.DataFrame,
    period: str,
    routes: Sequence[str],
    learner: str,
    features_all: List[str],
    y_wide: pd.DataFrame,
    processed_root: Path,
    out_pred: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Quarterly refit / monthly score for given period and routes."""
    sub = schedule.loc[schedule["period"] == period].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    pred_frames: List[pd.DataFrame] = []
    audit_rows: List[Dict] = []
    runtime_rows: List[Dict] = []

    # group by refit_anchor
    for refit_anchor, g in sub.groupby("refit_anchor", sort=True):
        refit_anchor = pd.Timestamp(refit_anchor).normalize()
        refit_row = schedule.loc[
            (schedule["oos_anchor"] == refit_anchor) & (schedule["is_refit"])
        ].iloc[0]
        train_start = pd.Timestamp(refit_row["train_start"]).normalize()
        train_end = pd.Timestamp(refit_row["train_end"]).normalize()
        label_end_max = pd.Timestamp(refit_row["train_label_end_max"]).normalize()
        if not (label_end_max < refit_anchor):
            raise RuntimeError(f"leakage: label_end_max {label_end_max} >= refit {refit_anchor}")

        t0 = time.time()
        logger.info(
            "[%s/%s] load train %s→%s for refit %s",
            period,
            learner,
            train_start.date(),
            train_end.date(),
            refit_anchor.date(),
        )
        panel_tr = load_processed_panel_slice(
            processed_root, train_start, train_end, columns=features_all
        )
        X_all, y_all, keys_all, samp_meta = sample_training_frame(
            panel_tr, y_wide, features_all
        )
        load_train_s = time.time() - t0

        # score date span covering all months under this refit
        score_months = sorted(pd.to_datetime(g["oos_anchor"]).tolist())
        score_start, _ = score_dates_for_month(schedule, score_months[0])
        _, score_end = score_dates_for_month(schedule, score_months[-1])
        t1 = time.time()
        panel_oos = load_processed_panel_slice(
            processed_root, score_start, score_end, columns=features_all
        )
        load_oos_s = time.time() - t1
        if panel_oos.empty:
            logger.warning("empty OOS panel for refit %s", refit_anchor.date())
            continue
        panel_oos = panel_oos.copy()
        panel_oos["TradeDate"] = pd.to_datetime(panel_oos["TradeDate"]).dt.normalize()
        y_long = y_wide.stack(future_stack=True).rename("y_5d").reset_index()
        y_long.columns = ["TradeDate", "Symbol", "y_5d"]
        y_long["TradeDate"] = pd.to_datetime(y_long["TradeDate"]).dt.normalize()
        oos_merged = panel_oos.merge(y_long, on=["TradeDate", "Symbol"], how="left")

        for route in routes:
            sel_name = SELECTOR_FOR_ROUTE[route]
            # feature mask frozen at refit_anchor (FS-3)
            feats = load_selected_mask(sel_name, refit_anchor, features_all)
            if not feats:
                logger.warning("empty mask route=%s refit=%s", route, refit_anchor.date())
                continue
            Xtr_m, ytr_m, mask_tr = complete_case_matrix(X_all, y_all, feats)
            keys_tr = keys_all.loc[mask_tr].reset_index(drop=True)
            if len(ytr_m) < 5000:
                logger.warning(
                    "insufficient train rows route=%s refit=%s n=%d",
                    route,
                    refit_anchor.date(),
                    len(ytr_m),
                )
                continue

            # OOS matrix for route features
            for f in feats:
                if f not in oos_merged.columns:
                    oos_merged[f] = np.nan
            Xo = oos_merged[feats].to_numpy(dtype=np.float32)
            # predict only finite-X rows; leave others NaN
            row_ok = np.all(np.isfinite(Xo), axis=1)
            pred = np.full(len(oos_merged), np.nan, dtype=float)
            t2 = time.time()
            if row_ok.sum() == 0:
                continue
            p_ok, meta = fit_predict_route(
                route=route,
                features=feats,
                X_tr=Xtr_m,
                y_tr=ytr_m,
                keys_tr=keys_tr,
                X_oos=Xo[row_ok],
                learner=learner,
            )
            fit_s = time.time() - t2
            pred[row_ok] = p_ok

            out = pd.DataFrame(
                {
                    "TradeDate": oos_merged["TradeDate"].to_numpy(),
                    "Symbol": oos_merged["Symbol"].to_numpy(),
                    "route": route,
                    "learner": learner,
                    "prediction": pred,
                    "y_5d": oos_merged["y_5d"].to_numpy(),
                    "refit_anchor": refit_anchor,
                    "n_features": len(feats),
                    "period": period,
                }
            )
            # keep only score months' date ranges
            keep = pd.Series(False, index=out.index)
            for m_anchor in score_months:
                s0, s1 = score_dates_for_month(schedule, m_anchor)
                keep |= (out["TradeDate"] >= s0) & (out["TradeDate"] <= s1)
            out = out.loc[keep].reset_index(drop=True)
            pred_frames.append(out)

            audit_rows.append(
                {
                    "period": period,
                    "learner": learner,
                    "route": route,
                    "refit_anchor": str(refit_anchor.date()),
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "train_label_end_max": str(label_end_max.date()),
                    "overlap_ok": bool(label_end_max < refit_anchor),
                    "n_train_sampled": int(samp_meta["n_before_y"]),
                    "n_train_complete_case": int(len(ytr_m)),
                    "n_oos_rows": int(len(out)),
                    "n_features": len(feats),
                    "n_score_months": len(score_months),
                    "load_train_sec": round(load_train_s, 2),
                    "load_oos_sec": round(load_oos_s, 2),
                    "fit_predict_sec": round(fit_s, 2),
                }
            )
            runtime_rows.append(
                {
                    "period": period,
                    "learner": learner,
                    "route": route,
                    "refit_anchor": str(refit_anchor.date()),
                    "fit_predict_sec": round(fit_s, 2),
                    "n_train": int(len(ytr_m)),
                    "n_features": len(feats),
                }
            )
            logger.info(
                "  route=%s n_feat=%d train=%d oos=%d fit=%.1fs",
                route,
                len(feats),
                len(ytr_m),
                len(out),
                fit_s,
            )

    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if not preds.empty:
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out_pred, index=False)

    # metrics by route: aggregate RankIC by month-end (month of TradeDate)
    metric_rows = []
    if not preds.empty:
        tmp = preds.copy()
        tmp["month"] = pd.to_datetime(tmp["TradeDate"]).dt.to_period("M").astype(str)
        for route, gr in tmp.groupby("route"):
            # daily IC then mean within month, then across months — or IC on each day then mean
            ic_daily = monthly_rank_ic(gr.rename(columns={}))  # uses TradeDate groups
            # collapse to month-level by averaging daily ICs in month
            if ic_daily.empty:
                continue
            ic_daily["month"] = pd.to_datetime(ic_daily["TradeDate"]).dt.to_period("M").astype(str)
            ic_m = (
                ic_daily.groupby("month", as_index=False)
                .agg(rank_ic=("rank_ic", "mean"), pearson_ic=("pearson_ic", "mean"), n=("n", "mean"))
            )
            # rename for summarize
            ic_m = ic_m.rename(columns={"month": "TradeDate"})
            nfeat = float(gr["n_features"].mean())
            summ = summarize_ic(ic_m, n_features=nfeat)
            summ["route"] = route
            summ["learner"] = learner
            summ["period"] = period
            metric_rows.append(summ)
    metrics = pd.DataFrame(metric_rows)
    return preds, metrics, audit_rows


def write_stage_report_bits():
    pass


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=FS4_ROOT)
    ap.add_argument("--panel-root", type=Path, default=FS1_FULL)
    ap.add_argument("--skip-xgb", action="store_true")
    ap.add_argument("--skip-holdout", action="store_true")
    args = ap.parse_args()
    out = args.out_root
    for d in (
        out / "contracts",
        out / "cache",
        out / "stage1_ridge",
        out / "stage2_xgb",
        out / "holdout",
        out / "audits",
    ):
        d.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    contract = fast_track_contract()
    (out / "contracts" / "fast_track_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    ch = contract_hash(contract)

    features_all, families = load_ordered_features()
    sch = feature_schema_hash(features_all, families=families)
    if sch != SCHEMA_HASH:
        logger.error("schema mismatch %s != %s", sch, SCHEMA_HASH)
        return 1

    windows = load_y5_ok_windows()
    schedule = build_refit_schedule(windows)
    schedule.to_csv(out / "audits" / "refit_schedule.csv", index=False)

    n_refits_screen = int(((schedule.period == "SCREEN") & schedule.is_refit).sum())
    n_refits_confirm = int(((schedule.period == "CONFIRM") & schedule.is_refit).sum())
    n_score_screen = int((schedule.period == "SCREEN").sum())
    n_score_confirm = int((schedule.period == "CONFIRM").sum())
    plan = {
        "n_refits_screen": n_refits_screen,
        "n_score_months_screen": n_score_screen,
        "n_refits_confirm": n_refits_confirm,
        "n_score_months_confirm": n_score_confirm,
        "routes_stage1": list(ROUTES_STAGE1),
        "estimated_train_rows_per_refit": "~250k (500 names × ~500 dates, then complete-case)",
        "ridge_fits_stage1": n_refits_screen * len(ROUTES_STAGE1),
        "xgb_fits_stage2_max": n_refits_screen * 3,
        "holdout_fits_max": n_refits_confirm * 2,
    }
    logger.info("EXECUTION PLAN:\n%s", json.dumps(plan, indent=2))
    (out / "audits" / "execution_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if plan["ridge_fits_stage1"] > 200:
        logger.error("RUNTIME_BUDGET_EXCEEDED: too many ridge fits")
        return 2

    y_wide, _ = load_y5_wide()
    processed = args.panel_root / "processed_ind_cap_z_v1"

    # ---------------- Stage 1
    t1 = time.time()
    preds1, metrics1, audits1 = run_period(
        schedule=schedule,
        period="SCREEN",
        routes=list(ROUTES_STAGE1),
        learner="ridge",
        features_all=features_all,
        y_wide=y_wide,
        processed_root=processed,
        out_pred=out / "stage1_ridge" / "monthly_predictions.parquet",
    )
    metrics1.to_csv(out / "stage1_ridge" / "route_metrics.csv", index=False)
    pd.DataFrame(audits1).to_csv(out / "audits" / "training_sample_audit.csv", index=False)
    stage1_verdict, survivors, decision = survivor_gate(metrics1)
    decision.to_csv(out / "stage1_ridge" / "survivor_decision.csv", index=False)
    logger.info("Stage1 verdict=%s survivors=%s (%.1fs)", stage1_verdict, survivors, time.time() - t1)

    # ---------------- Stage 2
    stage2_routes: List[str] = []
    metrics2 = pd.DataFrame()
    preds2 = pd.DataFrame()
    xgb_note = ""
    if args.skip_xgb:
        xgb_note = "skipped via --skip-xgb"
    elif not xgb_available():
        xgb_note = "XGBOOST_UNAVAILABLE"
    elif stage1_verdict.startswith("B."):
        xgb_note = "no selected survivors; skip selected XGB (ALL-only control optional)"
        # still run ALL for baseline comparison in screen with XGB
        stage2_routes = ["ALL_127"]
    else:
        stage2_routes = survivors  # ALL + up to 2 selected

    t2 = time.time()
    if stage2_routes and not args.skip_xgb and xgb_available():
        preds2, metrics2, audits2 = run_period(
            schedule=schedule,
            period="SCREEN",
            routes=stage2_routes,
            learner="xgb",
            features_all=features_all,
            y_wide=y_wide,
            processed_root=processed,
            out_pred=out / "stage2_xgb" / "monthly_predictions.parquet",
        )
        metrics2.to_csv(out / "stage2_xgb" / "route_metrics.csv", index=False)
        pd.DataFrame(audits2).to_csv(out / "audits" / "stage2_training_audit.csv", index=False)
        # pick best selected vs ALL
        if not metrics2.empty:
            all_m = metrics2.loc[metrics2.route == "ALL_127"]
            sel_m = metrics2.loc[metrics2.route != "ALL_127"].copy()
            winner = "ALL_127"
            if not sel_m.empty:
                sel_m = sel_m.sort_values(["mean_rank_ic", "icir"], ascending=False)
                # competitive if within -0.002 or better
                best = sel_m.iloc[0]
                if all_m.empty or float(best["mean_rank_ic"]) >= float(all_m.iloc[0]["mean_rank_ic"]) - 0.002:
                    winner = str(best["route"])
            pd.DataFrame(
                [{"winner": winner, "routes": ",".join(stage2_routes), "note": xgb_note}]
            ).to_csv(out / "stage2_xgb" / "survivor_decision.csv", index=False)
        else:
            winner = "ALL_127"
            pd.DataFrame([{"winner": winner, "note": "empty metrics"}]).to_csv(
                out / "stage2_xgb" / "survivor_decision.csv", index=False
            )
    else:
        winner = survivors[1] if len(survivors) > 1 else "ALL_127"
        pd.DataFrame(
            [{"winner": winner if stage1_verdict.startswith("A") else "ALL_127", "note": xgb_note or "ridge-only"}]
        ).to_csv(out / "stage2_xgb" / "survivor_decision.csv", index=False)
        if stage1_verdict.startswith("A") and len(survivors) > 1:
            winner = survivors[1]
        else:
            winner = "ALL_127"
    logger.info("Stage2 winner=%s note=%s (%.1fs)", winner, xgb_note, time.time() - t2)

    # ---------------- Stage 3 holdout
    holdout_routes = ["ALL_127"]
    if winner != "ALL_127":
        holdout_routes.append(winner)
    use_learner = "xgb" if (xgb_available() and not args.skip_xgb and not metrics2.empty) else "ridge"
    # If we ran XGB stage2, confirm with XGB; else Ridge
    if use_learner == "xgb" and winner not in stage2_routes and winner != "ALL_127":
        use_learner = "ridge"

    preds_h = pd.DataFrame()
    metrics_h = pd.DataFrame()
    if not args.skip_holdout:
        t3 = time.time()
        preds_h, metrics_h, audits_h = run_period(
            schedule=schedule,
            period="CONFIRM",
            routes=holdout_routes,
            learner=use_learner,
            features_all=features_all,
            y_wide=y_wide,
            processed_root=processed,
            out_pred=out / "holdout" / "predictions.parquet",
        )
        metrics_h.to_csv(out / "holdout" / "predictive_metrics.csv", index=False)
        pd.DataFrame(audits_h).to_csv(out / "audits" / "holdout_training_audit.csv", index=False)
        logger.info("Holdout done (%.1fs)", time.time() - t3)

    # leakage audit aggregate
    audits_all = pd.read_csv(out / "audits" / "training_sample_audit.csv")
    if (out / "audits" / "holdout_training_audit.csv").exists():
        audits_all = pd.concat(
            [audits_all, pd.read_csv(out / "audits" / "holdout_training_audit.csv")],
            ignore_index=True,
        )
    leak = pd.DataFrame(
        [
            {
                "n_refits_audited": int(audits_all["refit_anchor"].nunique()) if len(audits_all) else 0,
                "overlap_count": int((~audits_all["overlap_ok"]).sum()) if len(audits_all) else -1,
                "max_overlap_count": 0
                if len(audits_all) and bool(audits_all["overlap_ok"].all())
                else int((~audits_all["overlap_ok"]).sum()) if len(audits_all) else -1,
            }
        ]
    )
    leak.to_csv(out / "audits" / "leakage_audit.csv", index=False)

    # FS-4 verdict
    if leak["overlap_count"].iloc[0] != 0:
        fs4_verdict = "C. FS4_FAST_TRACK_NOT_READY"
    elif stage1_verdict.startswith("A.") and winner != "ALL_127" and not metrics_h.empty:
        # check holdout selected still competitive
        all_h = metrics_h.loc[metrics_h.route == "ALL_127"]
        sel_h = metrics_h.loc[metrics_h.route == winner]
        if not sel_h.empty and not all_h.empty:
            if float(sel_h.iloc[0]["mean_rank_ic"]) >= float(all_h.iloc[0]["mean_rank_ic"]) - 0.002:
                fs4_verdict = "A. FS4_FAST_TRACK_READY_FOR_FS5"
            else:
                fs4_verdict = "B. FS4_FAST_TRACK_ALL_FEATURES_ONLY"
                winner = "ALL_127"
        else:
            fs4_verdict = "B. FS4_FAST_TRACK_ALL_FEATURES_ONLY"
    elif stage1_verdict.startswith("B."):
        fs4_verdict = "B. FS4_FAST_TRACK_ALL_FEATURES_ONLY"
        winner = "ALL_127"
    else:
        fs4_verdict = "A. FS4_FAST_TRACK_READY_FOR_FS5" if winner != "ALL_127" else "B. FS4_FAST_TRACK_ALL_FEATURES_ONLY"

    # save holdout scores for FS-5 (wide-ready long)
    if not preds_h.empty:
        for route in holdout_routes:
            sub = preds_h.loc[preds_h.route == route, ["TradeDate", "Symbol", "prediction"]].rename(
                columns={"prediction": "ml_score"}
            )
            sub.to_parquet(out / "holdout" / f"ml_score_{route}.parquet", index=False)

    # report
    def _fmt_metrics(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_empty_"
        cols = [c for c in ["route", "mean_rank_ic", "icir", "pos_ic_frac", "mean_features", "n_months"] if c in df.columns]
        return "```\n" + df[cols].to_string(index=False) + "\n```"

    report = f"""# FS-4 Fast Track Report

## Verdict

```text
{fs4_verdict}
```

## Frozen setup

- horizon: Y{HORIZON}
- training window: 24m
- refit frequency: every {contract['model_refit_frequency']}
- train sample cap: {contract['max_names_per_train_date']} names/date
- screen period: OOS <= {SCREEN_END.date()}
- confirmation period: OOS >= {CONFIRM_START.date()}
- schema hash: `{SCHEMA_HASH}`
- contract hash: `{ch}`

## Execution plan

```json
{json.dumps(plan, indent=2)}
```

## Stage 1 Ridge (SCREEN)

{stage1_verdict}

{_fmt_metrics(metrics1)}

Survivors: {survivors}

## Stage 2 XGBoost (SCREEN)

- note: {xgb_note or 'ran'}
- routes: {stage2_routes}
- winner: {winner}

{_fmt_metrics(metrics2)}

## 2025+ confirmation ({use_learner})

{_fmt_metrics(metrics_h)}

## Leakage

- overlap_count: {int(leak['overlap_count'].iloc[0])}

## Runtime

- total_sec: {round(time.time() - t_all, 1)}

## Scope audit

- FPR/FDR rerun: NO
- L1 rerun: NO
- selector tuning: NO
- XGB tuning: NO
- Y1/Y20: NO
- 6m/72m: NO
- new portfolio engine: NO
- FS-1/2/3 mutation: NO
"""
    (out / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "schema_hash": SCHEMA_HASH,
        "contract_hash": ch,
        "fs4_verdict": fs4_verdict,
        "stage1_verdict": stage1_verdict,
        "survivors": survivors,
        "winner": winner,
        "holdout_learner": use_learner,
        "xgb_note": xgb_note,
        "runtime_sec": round(time.time() - t_all, 1),
        "plan": plan,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # freeze winner for FS-5 script
    (out / "holdout" / "fs5_routes.json").write_text(
        json.dumps(
            {
                "fs4_verdict": fs4_verdict,
                "learner": use_learner,
                "routes": holdout_routes if fs4_verdict.startswith("A") else ["ALL_127"],
                "winner_selected": winner if winner != "ALL_127" else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("FS-4 complete: %s", fs4_verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
