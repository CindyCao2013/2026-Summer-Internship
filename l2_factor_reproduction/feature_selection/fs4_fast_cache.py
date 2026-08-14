"""FS-4 Fast Track: cache builders, metrics, survivor gates."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.feature_selection.fs4_contract import (
    MAX_NAMES_PER_TRAIN_DATE,
    XGB_VAL_MONTHS,
    deterministic_sample_symbols,
)
from l2_factor_reproduction.feature_selection.panel_io import load_processed_panel_slice


def sample_training_frame(
    panel: pd.DataFrame,
    y_wide: pd.DataFrame,
    features: Sequence[str],
    *,
    max_names: int = MAX_NAMES_PER_TRAIN_DATE,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, object]]:
    """Deterministic per-date symbol subsample + join Y; no zero-fill."""
    panel = panel.copy()
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    # sample symbols per date
    parts = []
    for dt, g in panel.groupby("TradeDate", sort=True):
        pick = deterministic_sample_symbols(g["Symbol"].tolist(), k=max_names)
        parts.append(g.loc[g["Symbol"].isin(pick)])
    sampled = pd.concat(parts, ignore_index=True) if parts else panel.iloc[0:0]
    y_long = y_wide.stack(future_stack=True).rename("y").reset_index()
    y_long.columns = ["TradeDate", "Symbol", "y"]
    y_long["TradeDate"] = pd.to_datetime(y_long["TradeDate"]).dt.normalize()
    merged = sampled.merge(y_long, on=["TradeDate", "Symbol"], how="left")
    feats = [f for f in features if f in merged.columns]
    for f in features:
        if f not in merged.columns:
            merged[f] = np.nan
    X = merged[list(features)]
    y = merged["y"].to_numpy(dtype=float)
    meta = {
        "n_before_y": int(len(merged)),
        "n_finite_y": int(np.isfinite(y).sum()),
        "n_dates": int(merged["TradeDate"].nunique()),
        "n_symbols_sampled_mean": float(merged.groupby("TradeDate")["Symbol"].nunique().mean())
        if len(merged)
        else 0.0,
    }
    keys = merged[["TradeDate", "Symbol"]].copy()
    return X, y, keys, meta


def complete_case_matrix(
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete-case on route features only (FS-3 multivariate policy)."""
    cols = [f for f in features if f in X.columns]
    Xm = X[cols].to_numpy(dtype=np.float32)
    mask = np.isfinite(y) & np.all(np.isfinite(Xm), axis=1)
    return Xm[mask], y[mask].astype(np.float64), mask


def monthly_rank_ic(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional Spearman RankIC per TradeDate; aggregate helpers."""
    rows = []
    for dt, g in pred_df.groupby("TradeDate"):
        m = g["prediction"].notna() & g["y_5d"].notna()
        if m.sum() < 30:
            continue
        ric = g.loc[m, "prediction"].corr(g.loc[m, "y_5d"], method="spearman")
        pic = g.loc[m, "prediction"].corr(g.loc[m, "y_5d"], method="pearson")
        rows.append(
            {
                "TradeDate": dt,
                "rank_ic": float(ric) if pd.notna(ric) else np.nan,
                "pearson_ic": float(pic) if pd.notna(pic) else np.nan,
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic_df: pd.DataFrame, *, n_features: float) -> Dict[str, float]:
    if ic_df.empty:
        return {
            "mean_rank_ic": np.nan,
            "std_rank_ic": np.nan,
            "icir": np.nan,
            "pos_ic_frac": np.nan,
            "mean_pearson_ic": np.nan,
            "n_months": 0,
            "mean_features": float(n_features),
        }
    ric = ic_df["rank_ic"].astype(float)
    mean = float(ric.mean())
    std = float(ric.std())
    icir = mean / std * np.sqrt(12) if std > 0 else np.nan  # monthly series
    return {
        "mean_rank_ic": mean,
        "std_rank_ic": std,
        "icir": float(icir) if pd.notna(icir) else np.nan,
        "pos_ic_frac": float((ric > 0).mean()),
        "mean_pearson_ic": float(ic_df["pearson_ic"].mean()),
        "n_months": int(len(ic_df)),
        "mean_features": float(n_features),
    }


def survivor_gate(metrics: pd.DataFrame) -> Tuple[str, List[str], pd.DataFrame]:
    """Apply frozen Stage-1 survivor rules. Always keep ALL_127."""
    base = metrics.loc[metrics["route"] == "ALL_127"].iloc[0]
    all_ric = float(base["mean_rank_ic"])
    all_pos = float(base["pos_ic_frac"])
    rows = []
    survivors = ["ALL_127"]
    selected = []
    for _, r in metrics.iterrows():
        route = r["route"]
        if route == "ALL_127":
            rows.append({**r.to_dict(), "survive": True, "rule": "BASELINE"})
            continue
        ric = float(r["mean_rank_ic"])
        pos = float(r["pos_ic_frac"])
        nfeat = float(r["mean_features"])
        rule_a = ric >= all_ric + 0.002
        rule_b = (ric >= all_ric - 0.002) and (nfeat <= 0.60 * 127)
        pos_ok = pos >= all_pos - 0.05
        ok = (rule_a or rule_b) and pos_ok and np.isfinite(ric)
        rule = "A" if rule_a and pos_ok else ("B" if rule_b and pos_ok else "FAIL")
        rows.append({**r.to_dict(), "survive": bool(ok), "rule": rule})
        if ok:
            selected.append(route)
    # keep at most 2 selected, ranked by RankIC, ICIR, fewer features
    sel_df = pd.DataFrame([x for x in rows if x["route"] in selected])
    if not sel_df.empty:
        sel_df = sel_df.sort_values(
            ["mean_rank_ic", "icir", "mean_features"],
            ascending=[False, False, True],
        )
        selected = sel_df["route"].head(2).tolist()
    for row in rows:
        if row["route"] not in survivors and row["route"] not in selected:
            row["survive"] = False if row["route"] != "ALL_127" else True
        if row["route"] in selected:
            row["survive"] = True
    decision = pd.DataFrame(rows)
    if selected:
        verdict = "A. FAST_SCREEN_HAS_SURVIVORS"
    else:
        verdict = "B. FAST_SCREEN_ALL_ONLY"
    return verdict, survivors + selected, decision


def split_train_val_by_date(
    keys: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_months: int = XGB_VAL_MONTHS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Last ``val_months`` calendar months of training keys → validation."""
    dates = pd.to_datetime(keys["TradeDate"]).dt.to_period("M")
    months = sorted(dates.unique())
    if len(months) <= val_months + 1:
        # tiny fallback: last 20% rows by date order
        order = np.argsort(pd.to_datetime(keys["TradeDate"]).to_numpy())
        cut = int(len(order) * 0.8)
        tr, va = order[:cut], order[cut:]
        return X[tr], y[tr], X[va], y[va]
    val_set = set(months[-val_months:])
    m = dates.isin(val_set).to_numpy()
    return X[~m], y[~m], X[m], y[m]
