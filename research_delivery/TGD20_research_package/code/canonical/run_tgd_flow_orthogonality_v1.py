#!/usr/bin/env python
"""TGD20 ⟂ FlowDensity20 Orthogonality Report v1.

Three cases (confirmation window, size+industry signals):
  A) TGD vs FlowDensity_raw     — tradable alpha overlap?
  B) TGD vs Flow_perp_Amount    — pure flow leftover? (expected weak)
  C) TGD vs Amount              — independent of liquidity anomaly?

Outputs:
  research/reports/factor_orthogonality/TGD20_FlowDensity20/
    correlation.csv
    residual_ic.csv
    composite_probe.csv
    summary.md
    orthogonality_verdict.json
    figures/factor_overlap_matrix.png

Usage:
  OMP_NUM_THREADS=1 python run_tgd_flow_orthogonality_v1.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_dimension_density import DISCOVERY_DAYS, residual_ic_stats, split_discovery_confirmation
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from liquidity_normalization import panel_cross_sectional_residual
from run_flow_density_mechanism_v1 import FACTOR_COL, build_components, mean_cs_corr

OUT = Path("research/reports/factor_orthogonality/TGD20_FlowDensity20")
SIGNAL_SHIFT = 1


def log(msg: str) -> None:
    print(msg, flush=True)


def si_neut(panel: pd.DataFrame, ind: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    return cs_zscore(neutralize_size_industry(panel, ind.reindex_like(panel), mkt.reindex_like(panel)))


def pairwise_corr_matrix(panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(panels.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                mat.loc[a, b] = mat.loc[b, a]
            elif i == j:
                mat.loc[a, b] = 1.0
            else:
                mat.loc[a, b] = mean_cs_corr(panels[a], panels[b])
    return mat


def residual_pair(
    name_y: str,
    y: pd.DataFrame,
    name_x: str,
    x: pd.DataFrame,
    ret: pd.DataFrame,
) -> dict:
    """IC of y after CS residualizing on x; plus raw ICIRs and corr."""
    raw_y = daily_rank_ic_series(y, ret, signal_shift=SIGNAL_SHIFT)
    raw_x = daily_rank_ic_series(x, ret, signal_shift=SIGNAL_SHIFT)
    st = residual_ic_stats(y, ret, x, signal_shift=SIGNAL_SHIFT)
    return {
        "case": f"{name_y}_perp_{name_x}",
        "y": name_y,
        "x": name_x,
        "raw_icir_y": float(icir_from_daily(raw_y)),
        "raw_icir_x": float(icir_from_daily(raw_x)),
        "cs_corr": mean_cs_corr(y, x),
        "residual_ic_mean": st["residual_ic_mean"],
        "residual_icir": st["residual_icir"],
        "residual_ic_t": st["residual_ic_t"],
        "icir_retention": (
            float(st["residual_icir"] / icir_from_daily(raw_y))
            if abs(icir_from_daily(raw_y)) > 1e-9
            else np.nan
        ),
    }


def equal_rank_composite(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ra = a.rank(axis=1, pct=True, method="average")
    rb = b.rank(axis=1, pct=True, method="average")
    return cs_zscore(0.5 * ra + 0.5 * rb)


def probe_composite(
    label: str,
    a_name: str,
    a: pd.DataFrame,
    b_name: str,
    b: pd.DataFrame,
    ret: pd.DataFrame,
) -> dict:
    combo = equal_rank_composite(a, b)
    ic_a = daily_rank_ic_series(a, ret, signal_shift=SIGNAL_SHIFT)
    ic_b = daily_rank_ic_series(b, ret, signal_shift=SIGNAL_SHIFT)
    ic_c = daily_rank_ic_series(combo, ret, signal_shift=SIGNAL_SHIFT)
    ia, ib, ic = icir_from_daily(ic_a), icir_from_daily(ic_b), icir_from_daily(ic_c)
    return {
        "label": label,
        "a": a_name,
        "b": b_name,
        "icir_a": float(ia),
        "icir_b": float(ib),
        "icir_composite": float(ic),
        "beats_max_single": bool(pd.notna(ic) and pd.notna(ia) and pd.notna(ib) and ic > max(ia, ib)),
        "uplift_vs_max": float(ic - max(ia, ib)) if pd.notna(ic) else np.nan,
        "cs_corr": mean_cs_corr(a, b),
    }


def classify_independence(row: dict) -> str:
    corr = abs(row.get("cs_corr") or 0)
    t = abs(row.get("residual_ic_t") or 0)
    ret_ratio = row.get("icir_retention")
    if corr < 0.15 and t >= 2.0:
        return "independent"
    if corr < 0.35 and t >= 2.0 and pd.notna(ret_ratio) and abs(ret_ratio) >= 0.5:
        return "mostly_independent"
    if corr >= 0.60 and t < 2.0:
        return "redundant"
    if t >= 2.0:
        return "partial_overlap"
    return "absorbed_or_weak"


def plot_corr_heatmap(corr: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vals = corr.astype(float).to_numpy()
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    labels = list(corr.index)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Factor Orthogonality Matrix\n(mean daily CS Spearman)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _df_md(df: pd.DataFrame, float_prec: int = 3) -> str:
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_prec}f}")
    # include index as first column for corr matrix
    show = show.reset_index()
    cols = [str(c) for c in show.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in show.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])


def write_summary(
    out: Path,
    corr: pd.DataFrame,
    resid: pd.DataFrame,
    combo: pd.DataFrame,
    verdict: dict,
) -> None:
    lines = [
        "# TGD20 ⟂ FlowDensity20 Orthogonality Report v1",
        "",
        "**Window:** confirmation (post discovery-504)  ",
        "**Signals:** size+industry neutralized, signal_shift=1  ",
        "",
        "## Information taxonomy",
        "",
        "| Factor | Category | Role |",
        "|--------|----------|------|",
        "| TGD20 | temporal_information | Pure temporal timing residual |",
        "| FlowDensity_raw | liquidity_flow_interaction | Flow × Liquidity (tradable) |",
        "| Flow_perp_Amount | flow_only_candidate | Not validated (ICIR flipped neg) |",
        "| Amount | microstructure / liquidity | Anti-activity anomaly |",
        "",
        "## Case design",
        "",
        "- **A** TGD vs FlowDensity_raw — do two tradable alphas overlap?",
        "- **B** TGD vs Flow_perp_Amount — any pure-flow leftover vs TGD?",
        "- **C** TGD vs Amount — is TGD independent of liquidity anomaly?",
        "",
        "## Correlation matrix (mean CS Spearman)",
        "",
        _df_md(corr),
        "",
        "## Residual IC (both directions)",
        "",
        "| Case | Y ⊥ X | CS corr | Raw ICIR(Y) | Resid ICIR | Resid t | Retention | Class |",
        "|------|-------|--------:|------------:|-----------:|--------:|----------:|-------|",
    ]
    for _, r in resid.iterrows():
        lines.append(
            f"| {r['pair_case']} | `{r['case']}` | {r['cs_corr']:.3f} | {r['raw_icir_y']:.2f} | "
            f"{r['residual_icir']:.2f} | {r['residual_ic_t']:.2f} | {r['icir_retention']:.2f} | "
            f"{r['independence']} |"
        )

    lines += [
        "",
        "## Equal-rank composite probe (not production weights)",
        "",
        "| Label | ICIR A | ICIR B | Composite | Beats max? | Uplift | Corr |",
        "|-------|-------:|-------:|----------:|:----------:|-------:|-----:|",
    ]
    for _, r in combo.iterrows():
        lines.append(
            f"| {r['label']} | {r['icir_a']:.2f} | {r['icir_b']:.2f} | {r['icir_composite']:.2f} | "
            f"{'Y' if r['beats_max_single'] else 'N'} | {r['uplift_vs_max']:.2f} | {r['cs_corr']:.3f} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"- **Overall:** `{verdict['overall']}`",
        f"- **Composite readiness:** `{verdict['composite_readiness']}`",
        "",
        verdict["interpretation"],
        "",
        "## Artifacts",
        "",
        "- `correlation.csv`",
        "- `residual_ic.csv`",
        "- `composite_probe.csv`",
        "- `orthogonality_verdict.json`",
        "- `figures/factor_overlap_matrix.png`",
        "",
        "## Next",
        "",
        "- If Case A mostly independent → Composite Alpha Engine v1 (equal rank) is justified",
        "- Keep FlowDensity as interaction factor (do not freeze as pure flow)",
        "- Do not use Flow_perp_Amount as a long-side enhancer",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    log("=== TGD20 ⟂ FlowDensity20 Orthogonality v1 ===")
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2 = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

    float_mkt = enriched.float_mktcap.loc[start:end]
    comps = build_components(l2, float_mkt, enriched.amount.loc[start:end])
    for k in list(comps):
        comps[k] = comps[k].loc[start:end]

    tgd, _ = build_tgd20_wide_from_eod_l2(
        start, end, open_=enriched.open, close=enriched.close, use_cache=True
    )
    tgd = tgd.loc[start:end]

    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    # Align to intersection
    idx = tgd.index.intersection(comps[FACTOR_COL].index)
    cols = tgd.columns.intersection(comps[FACTOR_COL].columns)
    ret_full = ret_full.reindex(index=idx, columns=cols)
    _, ret = split_discovery_confirmation(ret_full, args.discovery_days)
    if ret.empty:
        ret = ret_full
    log(f"Confirmation: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    ind = industry.reindex_like(ret)
    mkt = float_mkt.reindex_like(ret)

    tgd_si = si_neut(tgd.reindex_like(ret), ind, mkt)
    flow_raw = comps[FACTOR_COL].reindex_like(ret)
    amt = comps["amount_mktcap_20d"].reindex_like(ret)
    flow_si = si_neut(flow_raw, ind, mkt)
    amt_si = si_neut(amt, ind, mkt)
    flow_perp = cs_zscore(panel_cross_sectional_residual(flow_si, [amt_si]))

    panels = {
        "TGD20": tgd_si,
        "FlowDensity_raw": flow_si,
        "Flow_perp_Amount": flow_perp,
        "Amount": amt_si,
    }

    log("\n--- Correlation matrix ---")
    corr = pairwise_corr_matrix(panels)
    log(corr.to_string(float_format=lambda x: f"{x:6.3f}"))
    corr.to_csv(OUT / "correlation.csv")
    plot_corr_heatmap(corr, OUT / "figures" / "factor_overlap_matrix.png")

    log("\n--- Residual IC ---")
    pairs: List[Tuple[str, str, str]] = [
        ("A", "TGD20", "FlowDensity_raw"),
        ("A", "FlowDensity_raw", "TGD20"),
        ("B", "TGD20", "Flow_perp_Amount"),
        ("B", "Flow_perp_Amount", "TGD20"),
        ("C", "TGD20", "Amount"),
        ("C", "Amount", "TGD20"),
        # Extra: Flow vs Amount already known, keep for matrix completeness in residual table
        ("X", "FlowDensity_raw", "Amount"),
        ("X", "Amount", "FlowDensity_raw"),
    ]
    resid_rows = []
    for case_id, y_name, x_name in pairs:
        row = residual_pair(y_name, panels[y_name], x_name, panels[x_name], ret)
        row["pair_case"] = case_id
        row["independence"] = classify_independence(row)
        resid_rows.append(row)
        log(
            f"  [{case_id}] {row['case']:40s} corr={row['cs_corr']:.3f} "
            f"resid_ICIR={row['residual_icir']:.2f} t={row['residual_ic_t']:.2f} "
            f"→ {row['independence']}"
        )
    resid = pd.DataFrame(resid_rows)
    resid.to_csv(OUT / "residual_ic.csv", index=False)

    log("\n--- Equal-rank composite probe ---")
    combo_rows = [
        probe_composite("A_TGD_FlowRaw", "TGD20", tgd_si, "FlowDensity_raw", flow_si, ret),
        probe_composite("B_TGD_FlowPerp", "TGD20", tgd_si, "Flow_perp_Amount", flow_perp, ret),
        probe_composite("C_TGD_Amount", "TGD20", tgd_si, "Amount", amt_si, ret),
        probe_composite("X_FlowRaw_Amount", "FlowDensity_raw", flow_si, "Amount", amt_si, ret),
    ]
    combo = pd.DataFrame(combo_rows)
    for _, r in combo.iterrows():
        log(
            f"  {r['label']:20s} ICIR {r['icir_a']:.2f}+{r['icir_b']:.2f}→{r['icir_composite']:.2f} "
            f"beats_max={r['beats_max_single']} uplift={r['uplift_vs_max']:.2f}"
        )
    combo.to_csv(OUT / "composite_probe.csv", index=False)

    # Verdict from Case A / C
    def _row(case: str, y: str, x: str) -> dict:
        sub = resid[(resid.pair_case == case) & (resid.y == y) & (resid.x == x)]
        return sub.iloc[0].to_dict() if len(sub) else {}

    a_tgd = _row("A", "TGD20", "FlowDensity_raw")
    a_flow = _row("A", "FlowDensity_raw", "TGD20")
    c_tgd = _row("C", "TGD20", "Amount")
    a_combo = combo.loc[combo.label == "A_TGD_FlowRaw"].iloc[0].to_dict()
    c_combo = combo.loc[combo.label == "C_TGD_Amount"].iloc[0].to_dict()

    mostly_indep = a_tgd.get("independence") in ("independent", "mostly_independent") and a_flow.get(
        "independence"
    ) in ("independent", "mostly_independent", "partial_overlap")
    tgd_vs_amt_ok = c_tgd.get("independence") in ("independent", "mostly_independent", "partial_overlap")

    if mostly_indep and a_combo.get("beats_max_single"):
        overall = "complementary_tradable_alphas"
        readiness = "ready_for_composite_v1_equal_rank"
    elif mostly_indep:
        overall = "low_overlap_probe_mixed"
        readiness = "composite_v1_worth_testing"
    else:
        overall = "material_overlap"
        readiness = "composite_deferred_pending_exposure_control"

    interpretation = (
        f"Case A: corr(TGD, FlowRaw)={a_tgd.get('cs_corr', float('nan')):.3f}; "
        f"TGD⊥Flow resid ICIR={a_tgd.get('residual_icir', float('nan')):.2f} "
        f"({a_tgd.get('independence')}); "
        f"Flow⊥TGD resid ICIR={a_flow.get('residual_icir', float('nan')):.2f} "
        f"({a_flow.get('independence')}). "
        f"Equal-rank composite ICIR={a_combo.get('icir_composite', float('nan')):.2f} "
        f"(beats max single: {a_combo.get('beats_max_single')}). "
        f"Case C: TGD⊥Amount resid ICIR={c_tgd.get('residual_icir', float('nan')):.2f} "
        f"({c_tgd.get('independence')}) — TGD is "
        f"{'not' if tgd_vs_amt_ok else 'possibly'} absorbed by liquidity anomaly. "
        "Taxonomy: Temporal (TGD) + Liquidity-conditioned Flow (FlowDensity), "
        "not Temporal + Pure Flow. Flow_perp_Amount remains non-candidate."
    )

    verdict = {
        "overall": overall,
        "composite_readiness": readiness,
        "taxonomy": {
            "TGD20": "temporal_information",
            "FlowDensity_raw": "liquidity_flow_interaction",
            "Flow_perp_Amount": "not_validated_pure_flow",
            "Amount": "microstructure_liquidity",
        },
        "case_A": {
            "corr": a_tgd.get("cs_corr"),
            "TGD_perp_Flow": a_tgd.get("independence"),
            "Flow_perp_TGD": a_flow.get("independence"),
            "composite_beats_max": a_combo.get("beats_max_single"),
            "composite_icir": a_combo.get("icir_composite"),
        },
        "case_C": {
            "corr": c_tgd.get("cs_corr"),
            "TGD_perp_Amount": c_tgd.get("independence"),
            "composite_beats_max": c_combo.get("beats_max_single"),
            "composite_icir": c_combo.get("icir_composite"),
        },
        "interpretation": interpretation,
        "next": "Composite Alpha Engine v1 with IC-weighted ranks (equal 0.5/0.5 underperforms TGD alone); keep FlowDensity as interaction candidate",
    }
    (OUT / "orthogonality_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    write_summary(OUT, corr, resid, combo, verdict)
    log(f"\nOverall: {overall} | readiness: {readiness}")
    log(f"Wrote {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
