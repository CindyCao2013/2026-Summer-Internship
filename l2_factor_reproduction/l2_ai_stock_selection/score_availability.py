"""FS-4 score-date availability adapter. Does not rewrite frozen FS-4 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.leakage import (
    filter_available_score_dates,
    first_allowed_score_date,
    score_dates_are_available,
)


FS4_AUDIT_CSV = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs4_fast_track"
    / "audits"
    / "holdout_training_audit.csv"
)
FS4_PREDICTIONS = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs4_fast_track"
    / "holdout"
    / "predictions.parquet"
)


def audit_refit_window(
    *,
    train_start,
    train_end,
    train_label_end_max,
    score_dates: Sequence,
    trading_dates: Sequence,
    route: str = "",
    learner: str = "",
    refit_anchor=None,
) -> dict:
    allowed = first_allowed_score_date(train_label_end_max, trading_dates)
    dates = pd.DatetimeIndex(pd.to_datetime(list(score_dates))).normalize().unique().sort_values()
    audit = score_dates_are_available(dates, train_label_end_max)
    n_invalid = int((~audit["available"]).sum())
    filtered = filter_available_score_dates(dates, train_label_end_max)
    actual_first = dates.min() if len(dates) else pd.NaT
    actual_first_after = filtered.min() if len(filtered) else pd.NaT
    n_invalid_after = int((~score_dates_are_available(filtered, train_label_end_max)["available"]).sum()) if len(filtered) else 0
    return {
        "route": route,
        "learner": learner,
        "refit_anchor": pd.Timestamp(refit_anchor).normalize() if refit_anchor is not None else pd.NaT,
        "train_start": pd.Timestamp(train_start).normalize(),
        "train_end": pd.Timestamp(train_end).normalize(),
        "train_label_end_max": pd.Timestamp(train_label_end_max).normalize(),
        "first_allowed_score_date": allowed,
        "actual_first_score_date": actual_first,
        "n_invalid_score_dates": n_invalid,
        "n_score_dates": int(len(dates)),
        "actual_first_score_date_after_ai_v1_filter": actual_first_after,
        "n_invalid_after_ai_v1_filter": n_invalid_after,
        "n_score_dates_after_ai_v1_filter": int(len(filtered)),
    }


def audit_fs4_holdout(
    *,
    trading_dates: Sequence,
    audit_csv: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Read frozen FS-4 holdout; do not mutate it. Apply AI-v1 filter in-memory."""
    audit_csv = Path(audit_csv or FS4_AUDIT_CSV)
    predictions_path = Path(predictions_path or FS4_PREDICTIONS)
    windows = pd.read_csv(audit_csv)
    for col in ("refit_anchor", "train_start", "train_end", "train_label_end_max"):
        windows[col] = pd.to_datetime(windows[col])
    pred = pd.read_parquet(predictions_path, columns=["TradeDate", "route", "learner", "refit_anchor"])
    pred["TradeDate"] = pd.to_datetime(pred["TradeDate"]).dt.normalize()
    pred["refit_anchor"] = pd.to_datetime(pred["refit_anchor"]).dt.normalize()
    rows = []
    keys = windows[["route", "learner", "refit_anchor", "train_start", "train_end", "train_label_end_max"]].drop_duplicates()
    for _, w in keys.iterrows():
        m = (
            (pred["route"] == w["route"])
            & (pred["learner"] == w["learner"])
            & (pred["refit_anchor"] == pd.Timestamp(w["refit_anchor"]).normalize())
        )
        score_dates = pred.loc[m, "TradeDate"].unique()
        rows.append(
            audit_refit_window(
                train_start=w["train_start"],
                train_end=w["train_end"],
                train_label_end_max=w["train_label_end_max"],
                score_dates=score_dates,
                trading_dates=trading_dates,
                route=str(w["route"]),
                learner=str(w["learner"]),
                refit_anchor=w["refit_anchor"],
            )
        )
    return pd.DataFrame(rows)
