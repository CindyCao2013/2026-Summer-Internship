"""Knife family attribution — participation vs trader_structure vs liquidity.

Do not replace the paper knife with the top scorer. Ask:
  - same Object, different Knife → same or different mechanism?
  - corr(cut_a, cut_b) low + residual IC high → independent alpha sources.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha_dimension_density import residual_ic_stats
from alpha_information_space import correlation_matrix
from factor_attribution import cs_zscore
from factor_cutting.cutting_analysis.knife_evaluator import evaluate_knives
from factor_cutting.cutting_analysis.knife_ic import ic_stats
from factor_cutting.w_cut import w_cut

# Mechanism families (extensible for Alpha Factory knife search)
KNIFE_FAMILIES: Dict[str, List[str]] = {
    "participation": ["volume", "amount", "turnover", "turnover_proxy"],
    "trader_structure": ["ats_trade_count", "ats_volume", "trade_count"],
    "liquidity": ["amihud", "volatility_state"],
}

FAMILY_OF: Dict[str, str] = {
    name: fam for fam, members in KNIFE_FAMILIES.items() for name in members
}


def family_of(knife: str) -> str:
    return FAMILY_OF.get(knife, "other")


def build_cut_factors(
    object_panel: pd.DataFrame,
    knives: Dict[str, pd.DataFrame],
    *,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """W-cut spread panel per knife."""
    out = {}
    for name, knife in knives.items():
        out[name] = w_cut(object_panel, knife.reindex_like(object_panel), window=window)
    return out


def knife_factor_corr(
    cut_factors: Dict[str, pd.DataFrame],
    *,
    min_overlap: int = 100,
) -> pd.DataFrame:
    """Pairwise correlation of cut factor panels (stacked CS observations).

    Default min_overlap=100: ATS panels start ~2018-11 and subsampled
    date×symbol indices rarely share 5000+ rows with full-history knives.
    """
    z = {k: cs_zscore(v) for k, v in cut_factors.items()}
    return correlation_matrix(z, min_overlap=min_overlap)


def pairwise_residual_ic(
    cut_factors: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
) -> pd.DataFrame:
    """Full residual-IC matrix: factor residualized vs each other knife."""
    names = list(cut_factors.keys())
    z = {k: cs_zscore(v) for k, v in cut_factors.items()}
    corr = knife_factor_corr(cut_factors)
    rows = []
    for name in names:
        raw = ic_stats(z[name], ret)
        for other in names:
            if other == name:
                continue
            st = residual_ic_stats(z[name], ret, z[other])
            c = (
                abs(corr.loc[name, other])
                if name in corr.index and other in corr.columns
                else np.nan
            )
            rows.append(
                {
                    "knife": name,
                    "family": family_of(name),
                    "vs_knife": other,
                    "vs_family": family_of(other),
                    "raw_ic": raw["rank_ic"],
                    "raw_icir": raw["icir"],
                    "abs_corr": c,
                    "residual_ic": st.get("residual_ic_mean", np.nan),
                    "residual_ic_t": st.get("residual_ic_t", np.nan),
                    "residual_icir": st.get("residual_icir", np.nan),
                }
            )
    return pd.DataFrame(rows)


def synthesize_dual_knife(
    cut_a: pd.DataFrame,
    cut_b: pd.DataFrame,
    *,
    mode: str = "equal_z",
) -> pd.DataFrame:
    """Combine two W-cut factors.

    Modes:
      equal_z     — 0.5·z(A) + 0.5·z(B), then re-z
      residual_add — z(A) + resid(z(B)|z(A)), then re-z
    """
    from liquidity_normalization import panel_cross_sectional_residual

    za = cs_zscore(cut_a)
    zb = cs_zscore(cut_b.reindex_like(cut_a))
    if mode == "equal_z":
        return cs_zscore(0.5 * za + 0.5 * zb)
    if mode == "residual_add":
        resid_b = panel_cross_sectional_residual(zb, [za])
        return cs_zscore(za + resid_b)
    raise ValueError(f"Unknown synth mode: {mode}")


def dual_knife_ic_table(
    cut_factors: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """IC of single knives vs dual-knife syntheses."""
    names = list(cut_factors.keys())
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    rows = []
    for name, panel in cut_factors.items():
        st = ic_stats(cs_zscore(panel), ret)
        rows.append(
            {
                "label": name,
                "kind": "single",
                "knife_a": name,
                "knife_b": None,
                "mode": None,
                "rank_ic": st["rank_ic"],
                "icir": st["icir"],
                "n_days": st.get("n_days", np.nan),
            }
        )

    for a, b in pairs:
        if a not in cut_factors or b not in cut_factors:
            continue
        for mode in ("equal_z", "residual_add"):
            synth = synthesize_dual_knife(cut_factors[a], cut_factors[b], mode=mode)
            st = ic_stats(synth, ret)
            rows.append(
                {
                    "label": f"{a}+{b}:{mode}",
                    "kind": "dual",
                    "knife_a": a,
                    "knife_b": b,
                    "mode": mode,
                    "rank_ic": st["rank_ic"],
                    "icir": st["icir"],
                    "n_days": st.get("n_days", np.nan),
                }
            )
    return pd.DataFrame(rows)


def incremental_ic_table(
    cut_factors: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    corr_indep_threshold: float = 0.50,
) -> pd.DataFrame:
    """
    For each knife cut factor: raw IC + residual IC vs each other knife.

    Independent if residual |t| >= 2 vs best competing knife and |corr| < threshold.
    """
    names = list(cut_factors.keys())
    z = {k: cs_zscore(v) for k, v in cut_factors.items()}
    corr = knife_factor_corr(cut_factors)

    rows = []
    for name in names:
        raw = ic_stats(z[name], ret)
        best_resid_t = np.nan
        best_vs = None
        for other in names:
            if other == name:
                continue
            st = residual_ic_stats(z[name], ret, z[other])
            t = st.get("residual_ic_t", np.nan)
            if pd.isna(best_resid_t) or (pd.notna(t) and abs(t) < abs(best_resid_t)):
                # track the *hardest* competitor: smallest |resid t| (most absorbed)
                # Actually for independence we want resid t vs closest competitor
                pass
            c = abs(corr.loc[name, other]) if name in corr.index and other in corr.columns else np.nan
            rows.append(
                {
                    "knife": name,
                    "family": family_of(name),
                    "vs_knife": other,
                    "vs_family": family_of(other),
                    "raw_ic": raw["rank_ic"],
                    "raw_icir": raw["icir"],
                    "abs_corr": c,
                    "residual_ic": st.get("residual_ic_mean", np.nan),
                    "residual_ic_t": t,
                    "residual_icir": st.get("residual_icir", np.nan),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail

    # Summary: per knife, hardest peer (max |corr|) and resid vs that peer
    summary_rows = []
    for name in names:
        sub = detail[detail["knife"] == name]
        if sub.empty:
            continue
        # peer with highest abs corr
        hard = sub.sort_values("abs_corr", ascending=False).iloc[0]
        indep = (
            pd.notna(hard["residual_ic_t"])
            and abs(hard["residual_ic_t"]) >= 2.0
            and (pd.isna(hard["abs_corr"]) or hard["abs_corr"] < corr_indep_threshold)
        )
        # also independent if corr low even if t mild
        soft_indep = pd.notna(hard["abs_corr"]) and hard["abs_corr"] < 0.35
        summary_rows.append(
            {
                "knife": name,
                "family": family_of(name),
                "raw_ic": hard["raw_ic"],
                "raw_icir": hard["raw_icir"],
                "closest_peer": hard["vs_knife"],
                "abs_corr_to_peer": hard["abs_corr"],
                "resid_ic_vs_peer": hard["residual_ic"],
                "resid_t_vs_peer": hard["residual_ic_t"],
                "independent": bool(indep or (soft_indep and abs(hard.get("residual_ic_t") or 0) >= 1.5)),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("raw_ic", key=lambda s: s.abs(), ascending=False)


def family_attribution_report(
    object_panel: pd.DataFrame,
    knives: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    *,
    window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Returns:
      eval_df     — effectiveness ranking (with family column)
      corr_df     — cut-factor correlation
      indep_df    — incremental / independence summary
      cut_factors — W-cut panels
    """
    eval_df = evaluate_knives(object_panel, knives, ret, window=window)
    eval_df.insert(1, "family", eval_df["knife"].map(family_of))

    cut_factors = build_cut_factors(object_panel, knives, window=window)
    corr_df = knife_factor_corr(cut_factors)
    indep_df = incremental_ic_table(cut_factors, ret)
    return eval_df, corr_df, indep_df, cut_factors


def write_family_markdown(
    path,
    eval_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    indep_df: pd.DataFrame,
) -> None:
    lines = [
        "# Knife Family Attribution",
        "",
        "## Effectiveness by family",
        "",
        "| Knife | Family | IC_spread | Separation | Effectiveness |",
        "|-------|--------|-----------|------------|---------------|",
    ]
    for _, r in eval_df.iterrows():
        lines.append(
            f"| `{r['knife']}` | {r['family']} | {r['ic_spread']:.4f} | "
            f"{r['separation']:.4f} | {r['effectiveness']:.4f} |"
        )
    lines += ["", "## Independence (vs closest peer)", ""]
    if indep_df is not None and not indep_df.empty:
        lines += [
            "| Knife | Peer | |corr| | resid_t | Independent? |",
            "|-------|------|-------|---------|--------------|",
        ]
        for _, r in indep_df.iterrows():
            lines.append(
                f"| `{r['knife']}` | `{r['closest_peer']}` | {r['abs_corr_to_peer']:.2f} | "
                f"{r['resid_t_vs_peer']:.2f} | {r['independent']} |"
            )
    lines += [
        "",
        "## Note",
        "",
        "volume (participation) ≠ ATS (trader_structure). High corr → same story; "
        "low corr + residual IC → two alpha sources. Do not auto-replace paper knife.",
        "",
    ]
    if corr_df is not None and not corr_df.empty:
        lines.append("## Correlation matrix")
        lines.append("")
        lines.append("```")
        lines.append(corr_df.round(3).to_string())
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
