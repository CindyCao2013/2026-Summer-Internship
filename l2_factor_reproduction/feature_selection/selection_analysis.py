"""FS-3 selection research aggregations (frequency, Jaccard, agreement, decay)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from l2_factor_reproduction.feature_selection.walkforward import jaccard


def selection_frequency(run_df: pd.DataFrame) -> pd.DataFrame:
    """Eligibility-conditioned selection frequency.

    Expects columns: feature, family, selector_name, horizon, oos_anchor,
    selected, locally_eligible, effective_n, coverage_ratio
    """
    d = run_df.copy()
    d["locally_eligible"] = d["locally_eligible"].astype(bool)
    d["selected"] = d["selected"].astype(bool)
    rows = []
    gcols = ["feature", "family", "selector_name", "horizon"]
    for keys, g in d.groupby(gcols, dropna=False):
        feat, fam, sel, hor = keys
        elig = g.loc[g["locally_eligible"]]
        n_elig = int(len(elig))
        n_sel = int(elig["selected"].sum()) if n_elig else 0
        n_all = int(g["oos_anchor"].nunique())
        rows.append(
            {
                "feature": feat,
                "family": fam,
                "selector_name": sel,
                "horizon": hor,
                "n_eligible_windows": n_elig,
                "n_selected_windows": n_sel,
                "selection_frequency": (n_sel / n_elig) if n_elig else np.nan,
                "availability_frequency": (n_elig / n_all) if n_all else np.nan,
                "n_total_windows": n_all,
                "mean_effective_n": float(elig["effective_n"].mean()) if n_elig else np.nan,
                "median_coverage_ratio": float(elig["coverage_ratio"].median())
                if n_elig and "coverage_ratio" in elig
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def family_selection_frequency(
    freq: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    inv = inventory.copy()
    if "feature" not in inv.columns and "factor" in inv.columns:
        inv = inv.rename(columns={"factor": "feature"})
    # family sizes from inventory eligible
    if "eligible_for_fs" in inv.columns:
        elig_inv = inv.loc[inv["eligible_for_fs"] == True]  # noqa: E712
    else:
        elig_inv = inv
    fam_size = elig_inv.groupby("family")["feature"].nunique()
    rows = []
    for (fam, sel, hor), g in freq.groupby(["family", "selector_name", "horizon"]):
        n_in_fam = int(fam_size.get(fam, g["feature"].nunique()))
        rows.append(
            {
                "family": fam,
                "selector_name": sel,
                "horizon": hor,
                "n_features_in_family": n_in_fam,
                "mean_feature_selection_frequency": float(g["selection_frequency"].mean()),
                "median_feature_selection_frequency": float(
                    g["selection_frequency"].median()
                ),
                "selected_feature_share": float(
                    (g["selection_frequency"].fillna(0) > 0).mean()
                ),
                "available_feature_share": float(g["availability_frequency"].mean()),
                "mean_availability_frequency": float(g["availability_frequency"].mean()),
                "coverage_adjusted_selection_rate": float(
                    (
                        g["selection_frequency"] * g["availability_frequency"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def selection_jaccard_series(run_df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-anchor Jaccard of selected sets."""
    rows = []
    for (sel, hor), g in run_df.groupby(["selector_name", "horizon"]):
        anchors = sorted(g["oos_anchor"].unique())
        prev_set = None
        prev_anchor = None
        for a in anchors:
            sub = g.loc[(g["oos_anchor"] == a) & (g["selected"]), "feature"]
            cur = set(sub.tolist())
            if prev_set is not None:
                jac, inter, union, empty_both = jaccard(prev_set, cur)
                rows.append(
                    {
                        "selector_name": sel,
                        "horizon": hor,
                        "oos_anchor_prev": prev_anchor,
                        "oos_anchor_curr": a,
                        "n_prev": len(prev_set),
                        "n_curr": len(cur),
                        "intersection": inter,
                        "union": union,
                        "jaccard": jac,
                        "empty_both": empty_both,
                    }
                )
            prev_set = cur
            prev_anchor = a
    return pd.DataFrame(rows)


def selector_agreement(run_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise selected-set Jaccard within common locally-eligible universe."""
    selectors = sorted(run_df["selector_name"].unique())
    rows = []
    for hor, gh in run_df.groupby("horizon"):
        for a, ga in gh.groupby("oos_anchor"):
            for i, s1 in enumerate(selectors):
                for s2 in selectors[i + 1 :]:
                    g1 = ga.loc[ga["selector_name"] == s1]
                    g2 = ga.loc[ga["selector_name"] == s2]
                    # common locally eligible
                    e1 = set(g1.loc[g1["locally_eligible"], "feature"])
                    e2 = set(g2.loc[g2["locally_eligible"], "feature"])
                    common = e1 & e2
                    sel1 = set(g1.loc[g1["selected"], "feature"]) & common
                    sel2 = set(g2.loc[g2["selected"], "feature"]) & common
                    jac, inter, union, empty_both = jaccard(sel1, sel2)
                    rows.append(
                        {
                            "horizon": hor,
                            "oos_anchor": a,
                            "selector_a": s1,
                            "selector_b": s2,
                            "n_common_eligible": len(common),
                            "intersection": inter,
                            "union": union,
                            "jaccard": jac,
                            "empty_both": empty_both,
                        }
                    )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    agg = (
        detail.groupby(["horizon", "selector_a", "selector_b"], as_index=False)
        .agg(
            mean_jaccard=("jaccard", "mean"),
            median_jaccard=("jaccard", "median"),
            n_windows=("oos_anchor", "nunique"),
        )
    )
    return agg


def selector_consensus(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (a, hor, feat), g in run_df.groupby(["oos_anchor", "horizon", "feature"]):
        n_elig = int(g["locally_eligible"].sum())
        n_sel = int(g.loc[g["locally_eligible"], "selected"].sum())
        fam = g["family"].iloc[0]
        rows.append(
            {
                "oos_anchor": a,
                "horizon": hor,
                "feature": feat,
                "family": fam,
                "n_selectors_eligible": n_elig,
                "n_selectors_selected": n_sel,
                "consensus_ratio": (n_sel / n_elig) if n_elig else np.nan,
            }
        )
    return pd.DataFrame(rows)


def horizon_decay(freq: pd.DataFrame) -> pd.DataFrame:
    piv = freq.pivot_table(
        index=["feature", "family", "selector_name"],
        columns="horizon",
        values="selection_frequency",
        aggfunc="first",
    )
    piv = piv.rename(columns={1: "selection_freq_y1", 5: "selection_freq_y5", 20: "selection_freq_y20"})
    for c in ("selection_freq_y1", "selection_freq_y5", "selection_freq_y20"):
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv.reset_index()
    piv["delta_y1_to_y5"] = piv["selection_freq_y5"] - piv["selection_freq_y1"]
    piv["delta_y5_to_y20"] = piv["selection_freq_y20"] - piv["selection_freq_y5"]
    piv["delta_y1_to_y20"] = piv["selection_freq_y20"] - piv["selection_freq_y1"]
    return piv


def coverage_selection_diagnostics(freq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = freq.copy()
    # per feature row already has median_coverage_ratio
    for (sel, hor), g in base.groupby(["selector_name", "horizon"]):
        x = g["median_coverage_ratio"].astype(float)
        y = g["selection_frequency"].astype(float)
        m = x.notna() & y.notna()
        spear = float(x[m].corr(y[m], method="spearman")) if m.sum() >= 5 else np.nan
        rows.append(
            {
                "selector_name": sel,
                "horizon": hor,
                "n_features": int(len(g)),
                "spearman_coverage_vs_selection_freq": spear,
                "mean_selection_frequency": float(y.mean()),
                "mean_median_coverage": float(x.mean()),
            }
        )
    return pd.DataFrame(rows)
