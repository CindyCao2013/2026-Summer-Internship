"""AI-v1 label builder. Reuses FS-3 economics; does not mutate FS-3 files."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.feature_selection.labels import (
    audit_label_boundaries,
    build_labels_wide_panel,
    load_daily_excess_and_bench,
)
from l2_factor_reproduction.l2_ai_stock_selection.contracts import (
    FS3_LABEL_ROOT,
    NEW_HORIZONS,
    PRODUCTION_LABEL_STATUS,
    TIMING_VERDICT,
)

PARITY_TOL = 1e-12
FS3_HORIZONS = (1, 5, 20)


def compare_wide(new: pd.DataFrame, frozen: pd.DataFrame, *, horizon: int) -> dict:
    common_idx = new.index.intersection(frozen.index)
    common_cols = new.columns.intersection(frozen.columns)
    a = new.loc[common_idx, common_cols].to_numpy(dtype=float)
    b = frozen.loc[common_idx, common_cols].to_numpy(dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    a_only = np.isfinite(a) & ~np.isfinite(b)
    b_only = np.isfinite(b) & ~np.isfinite(a)
    if both.any():
        max_abs = float(np.max(np.abs(a[both] - b[both])))
        mean_abs = float(np.mean(np.abs(a[both] - b[both])))
    else:
        max_abs = float("nan")
        mean_abs = float("nan")
    passed = bool(np.isfinite(max_abs) and max_abs <= PARITY_TOL and int(a_only.sum()) == 0 and int(b_only.sum()) == 0)
    return {
        "horizon": int(horizon),
        "n_common_dates": int(len(common_idx)),
        "n_common_symbols": int(len(common_cols)),
        "n_both_finite": int(both.sum()),
        "n_new_only_finite": int(a_only.sum()),
        "n_frozen_only_finite": int(b_only.sum()),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "tolerance": PARITY_TOL,
        "pass": passed,
    }


def load_frozen_fs3_wide(horizon: int) -> pd.DataFrame:
    path = FS3_LABEL_ROOT / "horizon={}".format(horizon) / "y_wide.parquet"
    y = pd.read_parquet(path)
    y.index = pd.to_datetime(y.index).normalize()
    return y


def run_fs3_parity(
    *,
    horizons: Sequence[int] = FS3_HORIZONS,
    rebuilt: Dict[int, pd.DataFrame] = None,
) -> pd.DataFrame:
    if rebuilt is None:
        excess, bench, dates = load_daily_excess_and_bench("full")
        rebuilt = build_labels_wide_panel(excess, bench, dates, horizons=list(horizons) + list(NEW_HORIZONS))
    rows = []
    for h in horizons:
        frozen = load_frozen_fs3_wide(h)
        rows.append(compare_wide(rebuilt[h], frozen, horizon=h))
    return pd.DataFrame(rows)


def tail_truncation_rows(
    label_map: Dict,
    dates: pd.DatetimeIndex,
    horizons: Sequence[int],
) -> pd.DataFrame:
    return audit_label_boundaries(
        dates,
        label_map["_meta_end"],
        label_map["_meta_valid"],
        horizons=horizons,
    )


def label_status_payload(*, parity_ok: bool) -> dict:
    production = bool(parity_ok) and TIMING_VERDICT == "C2C_TPLUS1_EXECUTABLE"
    return {
        "timing_verdict": TIMING_VERDICT,
        "parity_ok": bool(parity_ok),
        "status": "PRODUCTION" if production else PRODUCTION_LABEL_STATUS,
        "y3_y10_role": "production" if production else "diagnostic_only",
        "note": (
            "Y3/Y10 use the same FS-3 c2c economics (feature T -> T+1..T+h). "
            "They are not an AI-v1 production contract while T+1 c2c is not executable."
        ),
    }
