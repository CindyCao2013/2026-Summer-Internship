"""FS-3 purged / embargoed monthly walk-forward splitter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.feature_selection.labels import CANONICAL_HORIZONS
from l2_factor_reproduction.feature_selection.selectors import (
    CANONICAL_SELECTORS,
    SELECTOR_REGISTRY,
)

WALKFORWARD_CONTRACT_VERSION = "fs3_walkforward_v1"
EXPECTED_FEATURE_SCHEMA_HASH = "0b90fed383d3ba75"
TRAINING_WINDOW_MONTHS = 24
REFIT_FREQUENCY = "monthly"
# ~20 trading days/month × 24m ≈ 480; require near-full window (not truncated early history)
MIN_TRAIN_DATES = 400
MIN_EFFECTIVE_N = 5_000
K_BEST = 60
ALPHA = 0.05


def canonical_selector_params() -> Dict[str, Dict]:
    """Ex-ante frozen research parameters (not optimized on outcomes)."""
    return {
        "F_REGRESSION_KBEST": {"k": K_BEST},
        "MI_REGRESSION_KBEST": {
            "k": K_BEST,
            "n_neighbors": 3,
            "random_state": 42,
            "max_samples": 50000,
        },
        "F_REGRESSION_FPR": {"alpha": ALPHA},
        "F_REGRESSION_FDR": {"alpha": ALPHA},
        "L1_REGRESSION": dict(
            SELECTOR_REGISTRY["L1_REGRESSION"]["default_test_parameters"]
        ),  # reuse FS-2 fixture alpha=0.15 (ex-ante; not outcome-tuned)

        "TREE_IMPORTANCE_REGRESSION": {
            **dict(SELECTOR_REGISTRY["TREE_IMPORTANCE_REGRESSION"]["default_test_parameters"]),
            "max_samples": 100000,
        },
    }


def walkforward_contract_dict(
    *,
    feature_schema_hash: str,
    date_min: str,
    date_max: str,
) -> Dict[str, object]:
    return {
        "contract_version": WALKFORWARD_CONTRACT_VERSION,
        "training_window_months": TRAINING_WINDOW_MONTHS,
        "refit_frequency": REFIT_FREQUENCY,
        "horizons": list(CANONICAL_HORIZONS),
        "selector_parameters": canonical_selector_params(),
        "minimum_train_dates": MIN_TRAIN_DATES,
        "minimum_effective_n": MIN_EFFECTIVE_N,
        "benchmark": "000852.SH",
        "return_convention": "c2c excess vs 000852.SH; multi-day compound then subtract",
        "purge_rule": "train_label_end < oos_anchor (trading-date interval non-overlap)",
        "embargo_rule": {
            1: ">=1 trading day via label_end < oos",
            5: ">=5 trading days via label_end < oos",
            20: ">=20 trading days via label_end < oos",
        },
        "feature_schema_hash": feature_schema_hash,
        "expected_feature_schema_hash": EXPECTED_FEATURE_SCHEMA_HASH,
        "FS1_profile": "HUATAI_STYLE_IND_CAP_Z_V1",
        "FS2_contract_version": "fs2_selector_v1",
        "panel_date_min": date_min,
        "panel_date_max": date_max,
        "multivariate_missing_policy": (
            "complete-case on locally-eligible feature subset only; "
            "no universal 127-feature dropna; no zero-fill"
        ),
        "local_eligibility": {
            "min_finite_obs": 500,
            "not_constant": True,
        },
        "mi_max_samples": 50000,
        "mi_max_samples_note": (
            "Ex-ante computational subsample for MI only; "
            "seeded by random_state; not tuned on selection quality"
        ),
        "tree_max_samples": 100000,
        "tree_max_samples_note": (
            "Ex-ante computational subsample of complete-case rows for RF fit; "
            "seeded; not tuned on selection quality"
        ),
    }


def contract_hash(contract: Dict[str, object]) -> str:
    payload = json.dumps(contract, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def month_end_oos_anchors(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Final trading date of each calendar month present in ``dates``."""
    s = pd.Series(1, index=pd.DatetimeIndex(dates).normalize())
    # group by year-month, take last index
    keys = s.index.to_period("M")
    anchors = s.groupby(keys).apply(lambda x: x.index.max())
    return pd.DatetimeIndex(sorted(pd.to_datetime(anchors.to_numpy()).normalize()))


def training_date_bounds(
    dates: pd.DatetimeIndex,
    oos_anchor: pd.Timestamp,
    *,
    training_window_months: int = TRAINING_WINDOW_MONTHS,
    horizon: int,
    label_end_by_feature_date: pd.Series,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], str]:
    """Return (train_start, train_end_feature_date, status).

    train_end_feature_date is the latest feature date whose label_end < oos_anchor.
    train_start is ~training_window_months before that end (trading-date calendar approx
    via timestamp offset, then snap to available dates).
    """
    oos_anchor = pd.Timestamp(oos_anchor).normalize()
    dates = pd.DatetimeIndex(dates).normalize()
    # eligible feature dates by purge
    ends = label_end_by_feature_date.reindex(dates)
    ok = ends.notna() & (pd.to_datetime(ends) < oos_anchor)
    eligible = dates[ok.to_numpy()]
    if len(eligible) == 0:
        return None, None, "SKIPPED_NO_PURGED_DATES"
    train_end = eligible.max()
    # approximate start: train_end - N months, then first date >= that among eligible
    start_target = train_end - pd.DateOffset(months=training_window_months)
    eligible_in_window = eligible[eligible >= start_target]
    if len(eligible_in_window) == 0:
        return None, None, "SKIPPED_INSUFFICIENT_HISTORY"
    train_start = eligible_in_window.min()
    n_dates = int((eligible_in_window).shape[0])
    if n_dates < MIN_TRAIN_DATES:
        return train_start, train_end, "SKIPPED_INSUFFICIENT_HISTORY"
    # Also require the window not to be heavily left-truncated vs the 24m target
    # (early sample where panel starts mid-window).
    if train_start > start_target + pd.Timedelta(days=40):
        return train_start, train_end, "SKIPPED_INSUFFICIENT_HISTORY"
    return train_start, train_end, "OK"


@dataclass
class WalkForwardWindow:
    oos_anchor: pd.Timestamp
    horizon: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_label_end_max: pd.Timestamp
    status: str
    n_train_dates: int


def build_walkforward_windows(
    dates: pd.DatetimeIndex,
    label_end_by_horizon: Dict[int, pd.Series],
    horizons: Sequence[int] = CANONICAL_HORIZONS,
    training_window_months: int = TRAINING_WINDOW_MONTHS,
) -> List[WalkForwardWindow]:
    """Build monthly OOS windows with purge/embargo for each horizon.

    ``label_end_by_horizon[h]`` must be a Series indexed by feature TradeDate
    giving the last trading day included in that horizon's label interval.
    """
    anchors = month_end_oos_anchors(dates)
    out: List[WalkForwardWindow] = []
    for h in horizons:
        label_end_by_feature_date = label_end_by_horizon[h]
        for oos in anchors:
            start, end, status = training_date_bounds(
                dates,
                oos,
                training_window_months=training_window_months,
                horizon=h,
                label_end_by_feature_date=label_end_by_feature_date,
            )
            if status != "OK" or start is None or end is None:
                out.append(
                    WalkForwardWindow(
                        oos_anchor=pd.Timestamp(oos),
                        horizon=h,
                        train_start=pd.Timestamp(start) if start is not None else pd.NaT,
                        train_end=pd.Timestamp(end) if end is not None else pd.NaT,
                        train_label_end_max=pd.NaT,
                        status=status,
                        n_train_dates=0,
                    )
                )
                continue
            mask = (dates >= start) & (dates <= end)
            ends = pd.to_datetime(label_end_by_feature_date.reindex(dates[mask]))
            # ensure all ends < oos (fundamental Gate)
            if (ends.notna() & (ends >= oos)).any():
                status = "FAIL_OVERLAP"
            train_label_end_max = ends.max() if ends.notna().any() else pd.NaT
            n_dates = int(mask.sum())
            out.append(
                WalkForwardWindow(
                    oos_anchor=pd.Timestamp(oos),
                    horizon=h,
                    train_start=pd.Timestamp(start),
                    train_end=pd.Timestamp(end),
                    train_label_end_max=pd.Timestamp(train_label_end_max)
                    if pd.notna(train_label_end_max)
                    else pd.NaT,
                    status=status,
                    n_train_dates=n_dates,
                )
            )
    return out


def windows_to_frame(windows: Sequence[WalkForwardWindow]) -> pd.DataFrame:
    rows = []
    for w in windows:
        rows.append(
            {
                "oos_anchor": w.oos_anchor,
                "horizon": w.horizon,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "train_label_end_max": w.train_label_end_max,
                "status": w.status,
                "n_train_dates": w.n_train_dates,
                "overlap_count": int(
                    0
                    if w.status != "FAIL_OVERLAP"
                    else 1
                ),
            }
        )
    return pd.DataFrame(rows)


def assert_no_overlap(windows: Sequence[WalkForwardWindow]) -> int:
    """Return total overlap failures."""
    return sum(1 for w in windows if w.status == "FAIL_OVERLAP")


def jaccard(a: Iterable[str], b: Iterable[str]) -> Tuple[float, int, int, bool]:
    sa, sb = set(a), set(b)
    empty_both = (not sa) and (not sb)
    if empty_both:
        return float("nan"), 0, 0, True
    inter = len(sa & sb)
    union = len(sa | sb)
    return (inter / union if union else float("nan")), inter, union, False
