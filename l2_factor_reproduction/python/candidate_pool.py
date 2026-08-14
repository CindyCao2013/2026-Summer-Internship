"""Shared evaluation utilities for L2 candidate-family discovery screens."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def yearly_ic_table(rank_ic: pd.Series) -> pd.DataFrame:
    ic = rank_ic.dropna()
    table = ic.groupby(ic.index.year).agg(["mean", "std", "count"])
    table["icir_annualized"] = (
        table["mean"] / table["std"] * np.sqrt(250)
    )
    table.index.name = "year"
    return table


def load_rank_ic(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.empty:
        return pd.Series(dtype=float, name="rank_ic")
    result = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    result.index = pd.to_datetime(result.index)
    result.name = "rank_ic"
    return result


def stability_fields(
    yearly_raw: pd.DataFrame,
    full_raw_ic: float,
) -> Dict[str, object]:
    valid = yearly_raw.loc[yearly_raw["count"] >= 30].copy()
    if valid.empty:
        return {
            "n_years": 0,
            "same_sign_years": 0,
            "sign_consistency": np.nan,
            "positive_ic_years": 0,
            "negative_ic_years": 0,
            "yearly_ic_min": np.nan,
            "yearly_ic_max": np.nan,
        }
    full_sign = np.sign(full_raw_ic)
    yearly_sign = np.sign(valid["mean"])
    same = (
        int((yearly_sign == full_sign).sum())
        if full_sign != 0
        else int((yearly_sign == 0).sum())
    )
    return {
        "n_years": int(len(valid)),
        "same_sign_years": same,
        "sign_consistency": float(same / len(valid)),
        "positive_ic_years": int((valid["mean"] > 0).sum()),
        "negative_ic_years": int((valid["mean"] < 0).sum()),
        "yearly_ic_min": float(valid["mean"].min()),
        "yearly_ic_max": float(valid["mean"].max()),
    }


def decile_monotonicity(summary: Dict[str, object]) -> float:
    values = summary.get("group_mean_annu", {})
    if not isinstance(values, dict) or len(values) < 3:
        return float("nan")
    pairs = sorted(
        ((int(group), float(value)) for group, value in values.items()),
        key=lambda item: item[0],
    )
    groups = pd.Series([item[0] for item in pairs], dtype=float)
    returns = pd.Series([item[1] for item in pairs], dtype=float)
    return float(groups.corr(returns, method="spearman"))


def mean_daily_cross_sectional_spearman(
    features: pd.DataFrame,
    names: Iterable[str],
    min_names: int = 100,
) -> pd.DataFrame:
    """Mean of full-sample daily cross-sectional Spearman matrices."""
    selected = list(names)
    total = pd.DataFrame(0.0, index=selected, columns=selected)
    count = pd.DataFrame(
        0, index=selected, columns=selected, dtype=int
    )
    for _, block in features.groupby("TradeDate", sort=True):
        corr = block[selected].corr(
            method="spearman", min_periods=min_names
        )
        valid = corr.notna()
        total = total.add(corr.fillna(0.0), fill_value=0.0)
        count = count.add(valid.astype(int), fill_value=0)
    return total.divide(count.where(count > 0))


def correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "factor_left",
        "factor_right",
        "mean_daily_spearman",
        "abs_mean_daily_spearman",
        "redundancy_band",
    ]
    rows = []
    names = list(corr.index)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            rho = float(corr.loc[left, right])
            absolute = abs(rho)
            if absolute >= 0.95:
                band = "near_alias"
            elif absolute >= 0.80:
                band = "high"
            elif absolute >= 0.50:
                band = "moderate"
            else:
                band = "low"
            rows.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "mean_daily_spearman": rho,
                    "abs_mean_daily_spearman": absolute,
                    "redundancy_band": band,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "abs_mean_daily_spearman", ascending=False
    )


def redundancy_annotations(
    corr: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Connected components under |rho| threshold; no pruning decision."""
    names = list(corr.index)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            if abs(float(corr.loc[left, right])) >= threshold:
                union(left, right)

    roots: Dict[str, str] = {}
    rows = []
    for name in names:
        root = find(name)
        if root not in roots:
            roots[root] = f"R{len(roots) + 1}"
        peers = corr.loc[name].drop(index=name).dropna()
        if peers.empty:
            max_peer, max_corr = None, np.nan
        else:
            max_peer = str(peers.abs().idxmax())
            max_corr = float(peers.loc[max_peer])
        rows.append(
            {
                "factor": name,
                "redundancy_cluster_080": roots[root],
                "max_corr_peer": max_peer,
                "max_abs_corr": abs(max_corr),
                "near_alias_observed": bool(abs(max_corr) >= 0.95),
            }
        )
    return pd.DataFrame(rows)
