#!/usr/bin/env python
"""Milestone 1F — Factor Similarity Matrix v1.

Registry factors only: TGD20, FlowDensity20, D1, D4, D5.
Analysis only — no Composite, no Registry schema changes, no formula retune.

Signal book (documented):
  confirmation window (post discovery-504)
  size+industry neutralized + CS z-score for ALL five
  signal_shift=1

Outputs:
  research/reports/factor_similarity_matrix/
    factor_ic_corr.csv
    factor_return_corr.csv
    residual_ic_matrix.csv
    residual_ic_long.csv
    cs_corr_matrix.csv
    factor_clusters.yaml
    similarity_report.md
    similarity_verdict.json
    figures/*.png

Usage:
  OMP_NUM_THREADS=1 python run_factor_similarity_matrix_v1.py
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
import yaml

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_dimension_density import DISCOVERY_DAYS, residual_ic_stats, split_discovery_confirmation
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from run_flow_density_mechanism_v1 import mean_cs_corr

OUT = Path("research/reports/factor_similarity_matrix")
SIGNAL_SHIFT = 1

# Registry factor_id → library panel builder key
FACTORS = [
    ("TGD20", "tgd"),
    ("FlowDensity20", "flow"),
    ("D1_LiquidityQuality60d", "low_vol_liquidity_quality_60d"),
    ("D4_WinnerSentimentReversal5d", "winner_sentiment_reversal_5d"),
    ("D5_UpsideFragility20d", "upside_fragility_20d"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def _md_table(df: pd.DataFrame) -> str:
    d = df.round(3)
    cols = [str(c) for c in d.columns]
    lines = [
        "| index | " + " | ".join(cols) + " |",
        "| --- | " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for idx, row in d.iterrows():
        cells = [f"{v:.3f}" if pd.notna(v) else "" for v in row]
        lines.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def si_neut(panel: pd.DataFrame, ind: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    return cs_zscore(neutralize_size_industry(panel, ind.reindex_like(panel), mkt.reindex_like(panel)))


def build_raw_panels(
    enriched,
    l2_cache,
    pv_cache,
    start: dt.datetime,
    end: dt.datetime,
) -> Dict[str, pd.DataFrame]:
    float_mkt = enriched.float_mktcap.loc[start:end]
    raw: Dict[str, pd.DataFrame] = {}

    log("  build TGD20 ...")
    tgd_wide, _ = build_tgd20_wide_from_eod_l2(
        start,
        end,
        open_=enriched.open,
        close=enriched.close,
        use_cache=True,
        window=20,
    )
    raw["TGD20"] = tgd_wide.loc[start:end]

    log("  build FlowDensity20 ...")
    raw["FlowDensity20"] = build_net_active_flow_mktcap(l2_cache, float_mkt, window=20).loc[start:end]

    for fid, lib in FACTORS:
        if lib in ("tgd", "flow"):
            continue
        log(f"  build {fid} ({lib}) ...")
        raw[fid] = build_eod_engine_factor(lib, pv_cache).loc[start:end]

    return raw


def align_panels(
    panels: Dict[str, pd.DataFrame], ret: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    names = list(panels.keys())
    idx = panels[names[0]].index
    cols = panels[names[0]].columns
    for n in names[1:]:
        idx = idx.intersection(panels[n].index)
        cols = cols.intersection(panels[n].columns)
    idx = idx.intersection(ret.index)
    cols = cols.intersection(ret.columns)
    out = {n: panels[n].reindex(index=idx, columns=cols) for n in names}
    return out, ret.reindex(index=idx, columns=cols)


def ic_corr_matrix(ic_df: pd.DataFrame) -> pd.DataFrame:
    return ic_df.corr(method="pearson")


def pairwise_cs_corr(panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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


def hl_daily_series(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    sig = align_signal(signal, SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    _, pnl, _ = Factor_Dev_Lib.groupTest(sig, r, n=10, fee=0, info="silent")
    return pnl["H-L"]


def residual_matrix(panels: Dict[str, pd.DataFrame], ret: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    names = list(panels.keys())
    resid_icir = pd.DataFrame(index=names, columns=names, dtype=float)
    rows = []
    for y in names:
        for x in names:
            if y == x:
                resid_icir.loc[y, x] = np.nan
                continue
            raw_y = daily_rank_ic_series(panels[y], ret, signal_shift=SIGNAL_SHIFT)
            st = residual_ic_stats(panels[y], ret, panels[x], signal_shift=SIGNAL_SHIFT)
            raw_icir = float(icir_from_daily(raw_y))
            resid = float(st["residual_icir"])
            resid_icir.loc[y, x] = resid
            retention = resid / raw_icir if abs(raw_icir) > 1e-9 else np.nan
            # classify
            t = abs(st.get("residual_ic_t") or 0)
            if abs(resid) < 1.0 or t < 2.0:
                role = "redundant_or_absorbed"
            elif retention >= 0.70 and t >= 3.0:
                role = "independent_source"
            elif retention >= 0.35:
                role = "partial_overlap_enhancer"
            else:
                role = "mostly_redundant"
            rows.append(
                {
                    "y": y,
                    "x": x,
                    "case": f"{y}_perp_{x}",
                    "cs_corr": mean_cs_corr(panels[y], panels[x]),
                    "raw_icir_y": raw_icir,
                    "residual_ic_mean": st["residual_ic_mean"],
                    "residual_icir": resid,
                    "residual_ic_t": st["residual_ic_t"],
                    "icir_retention": retention,
                    "role": role,
                }
            )
            log(f"  {y} ⊥ {x}: resid_ICIR={resid:.2f} retention={retention:.2f} → {role}")
    return resid_icir, pd.DataFrame(rows)


def cluster_factors(ic_corr: pd.DataFrame, resid_long: pd.DataFrame) -> dict:
    """Simple rule-based clusters from |IC corr| + residual roles."""
    names = list(ic_corr.index)
    # average |IC corr| to others
    abs_corr = ic_corr.abs()
    # Seed clusters: TGD temporal; liquidity group if D1-Flow linked; D4 behavioral; D5
    clusters = {
        "temporal_core": ["TGD20"],
        "liquidity_quality": [],
        "flow_liquidity_interaction": [],
        "behavioral_reversal": [],
        "fragility_tail": [],
        "unassigned": [],
    }
    for n in names:
        if n == "TGD20":
            continue
        if n == "FlowDensity20":
            clusters["flow_liquidity_interaction"].append(n)
        elif n == "D1_LiquidityQuality60d":
            clusters["liquidity_quality"].append(n)
        elif n == "D4_WinnerSentimentReversal5d":
            clusters["behavioral_reversal"].append(n)
        elif n == "D5_UpsideFragility20d":
            clusters["fragility_tail"].append(n)
        else:
            clusters["unassigned"].append(n)

    # Independence summary vs TGD / D1
    def best_role(y: str, x: str) -> str:
        sub = resid_long[(resid_long["y"] == y) & (resid_long["x"] == x)]
        return str(sub.iloc[0]["role"]) if len(sub) else "unknown"

    independence = {}
    for n in names:
        if n == "TGD20":
            independence[n] = {
                "vs_TGD20": "self",
                "alpha_role_hint": "core",
            }
            continue
        vs_tgd = best_role(n, "TGD20")
        vs_d1 = best_role(n, "D1_LiquidityQuality60d") if n != "D1_LiquidityQuality60d" else "self"
        if vs_tgd == "independent_source" and (vs_d1 in ("independent_source", "self", "partial_overlap_enhancer")):
            hint = "core_or_satellite"
        elif vs_tgd == "partial_overlap_enhancer" or vs_d1 == "partial_overlap_enhancer":
            hint = "enhancer"
        elif "redundant" in vs_tgd or "redundant" in vs_d1:
            hint = "redundant_risk"
        else:
            hint = "review"
        independence[n] = {
            "vs_TGD20": vs_tgd,
            "vs_D1_LiquidityQuality60d": vs_d1,
            "ic_corr_vs_TGD20": float(ic_corr.loc[n, "TGD20"]) if "TGD20" in ic_corr.columns else None,
            "ic_corr_vs_D1": float(ic_corr.loc[n, "D1_LiquidityQuality60d"])
            if n != "D1_LiquidityQuality60d" and "D1_LiquidityQuality60d" in ic_corr.columns
            else None,
            "alpha_role_hint": hint,
        }

    return {
        "schema_version": "factor_similarity_v1",
        "signal_book": "confirmation_size_industry_cs_z",
        "clusters": {k: v for k, v in clusters.items() if v},
        "independence": independence,
        "notes": [
            "Clusters are taxonomy seeds + residual roles — not ML clustering.",
            "D4/D5 lack Template v2 packs; treat as library_inventory candidates.",
            "No Composite weights produced in 1F.",
        ],
    }


def heatmap(mat: pd.DataFrame, title: str, path: Path, cmap: str = "RdBu_r", vlim: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    data = mat.astype(float)
    im = ax.imshow(data.values, cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels([c.replace("_", "\n") for c in data.columns], fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels([c.replace("_", "\n") for c in data.index], fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    out: Path,
    meta: dict,
    ic_corr: pd.DataFrame,
    ret_corr: pd.DataFrame,
    resid_long: pd.DataFrame,
    clusters: dict,
) -> None:
    lines = [
        "# Factor Similarity Matrix v1",
        "",
        f"**Window:** {meta['start']} → {meta['end']} ({meta['n_days']}d confirmation)",
        f"**Signal book:** size+industry CS-z · signal_shift={SIGNAL_SHIFT}",
        f"**Universe factors:** {', '.join(meta['factors'])}",
        "",
        "## Questions answered",
        "",
        "1. Same alpha source? → IC / CS / H-L correlations",
        "2. Incremental alpha? → residual ICIR after CS residualization",
        "3. Combination value? → independence roles (no Composite weights yet)",
        "",
        "## IC correlation (daily RankIC series, Pearson)",
        "",
        _md_table(ic_corr),
        "",
        "## Factor return correlation (daily H-L)",
        "",
        _md_table(ret_corr),
        "",
        "## Residual IC (Y ⊥ X)",
        "",
        "| Y | X | CS corr | Raw ICIR(Y) | Resid ICIR | Retention | Role |",
        "|---|---|---------|------------:|-----------:|----------:|------|",
    ]
    for _, r in resid_long.sort_values(["y", "x"]).iterrows():
        lines.append(
            f"| {r['y']} | {r['x']} | {r['cs_corr']:.3f} | {r['raw_icir_y']:.2f} | "
            f"{r['residual_icir']:.2f} | {r['icir_retention']:.2f} | {r['role']} |"
        )

    lines += [
        "",
        "## Clusters / role hints",
        "",
        "```yaml",
        yaml.safe_dump(clusters, allow_unicode=True, sort_keys=False).rstrip(),
        "```",
        "",
        "## Verdict (human-readable)",
        "",
    ]

    # Narrative bullets from independence
    ind = clusters.get("independence") or {}
    tgd_flow = ind.get("FlowDensity20", {})
    tgd_d1 = ind.get("D1_LiquidityQuality60d", {})
    lines.append(
        f"- **TGD20** is the temporal core (`alpha_role_hint=core`). "
        f"Flow vs TGD: `{tgd_flow.get('vs_TGD20')}` (IC corr={tgd_flow.get('ic_corr_vs_TGD20')})."
    )
    lines.append(
        f"- **D1** vs TGD: `{tgd_d1.get('vs_TGD20')}` (IC corr={tgd_d1.get('ic_corr_vs_TGD20')}). "
        "Slow liquidity quality — candidate core/satellite, not validated."
    )
    flow_d1 = resid_long[
        (resid_long["y"] == "FlowDensity20") & (resid_long["x"] == "D1_LiquidityQuality60d")
    ]
    if len(flow_d1):
        fr = flow_d1.iloc[0]
        lines.append(
            f"- **Flow ⊥ D1:** resid ICIR={fr['residual_icir']:.2f}, retention={fr['icir_retention']:.2f}, "
            f"role=`{fr['role']}` — critical for whether Flow is new info vs liquidity repackaging."
        )
    for fid in ("D4_WinnerSentimentReversal5d", "D5_UpsideFragility20d"):
        h = ind.get(fid, {})
        lines.append(
            f"- **{fid}**: vs TGD `{h.get('vs_TGD20')}`, role hint `{h.get('alpha_role_hint')}` "
            "(library inventory; pack incomplete)."
        )

    lines += [
        "",
        "## Independent / redundant / enhancer (summary)",
        "",
        "| Factor | Independent alpha source? | Redundant risk? | Enhancer-only risk? |",
        "|--------|---------------------------|-----------------|---------------------|",
    ]
    for fid in meta["factors"]:
        h = ind.get(fid, {})
        role = h.get("alpha_role_hint", "core" if fid == "TGD20" else "review")
        indep = "yes" if role in ("core", "core_or_satellite") else "partial/no"
        red = "yes" if role == "redundant_risk" else "low/monitor"
        enh = "yes" if role == "enhancer" else "no"
        lines.append(f"| {fid} | {indep} | {red} | {enh} |")

    lines += [
        "",
        "## Artifacts",
        "",
        "- `factor_ic_corr.csv`",
        "- `factor_return_corr.csv`",
        "- `cs_corr_matrix.csv`",
        "- `residual_ic_matrix.csv`",
        "- `residual_ic_long.csv`",
        "- `factor_clusters.yaml`",
        "- `similarity_verdict.json`",
        "- `figures/`",
        "",
        "## Explicit non-goals",
        "",
        "- No Composite weights",
        "- No Registry schema changes",
        "- No formula changes",
        "",
        "## Next",
        "",
        "Composite Alpha Engine only after human review of residual roles "
        "(especially Flow⊥D1 and D4/D5 pack completeness).",
        "",
    ]
    (out / "similarity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    log("=== Milestone 1F Factor Similarity Matrix v1 ===")

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
    pv_cache = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )

    raw = build_raw_panels(enriched, l2_cache, pv_cache, start, end)
    float_mkt = enriched.float_mktcap.loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    # Confirmation slice
    panels_full = {}
    for name, panel in raw.items():
        _, conf = split_discovery_confirmation(panel, args.discovery_days)
        panels_full[name] = conf
    _, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    _, ind_conf = split_discovery_confirmation(industry, args.discovery_days)
    _, mkt_conf = split_discovery_confirmation(float_mkt, args.discovery_days)

    panels_raw_aligned, ret = align_panels(panels_full, ret_conf)
    ind = ind_conf.reindex_like(ret)
    mkt = mkt_conf.reindex_like(ret)

    log("Neutralize size+industry ...")
    panels = {n: si_neut(p, ind, mkt) for n, p in panels_raw_aligned.items()}
    names = list(panels.keys())
    log(f"Aligned confirmation: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    # Daily IC series
    log("Daily RankIC series ...")
    ic = {n: daily_rank_ic_series(panels[n], ret, signal_shift=SIGNAL_SHIFT) for n in names}
    ic_df = pd.DataFrame(ic).dropna(how="any")
    ic_corr = ic_corr_matrix(ic_df)
    ic_corr.to_csv(OUT / "factor_ic_corr.csv")

    log("CS correlation matrix ...")
    cs_corr = pairwise_cs_corr(panels)
    cs_corr.to_csv(OUT / "cs_corr_matrix.csv")

    log("H-L return series ...")
    hl = {n: hl_daily_series(panels[n], ret) for n in names}
    hl_df = pd.DataFrame(hl).dropna(how="any")
    ret_corr = hl_df.corr(method="pearson")
    ret_corr.to_csv(OUT / "factor_return_corr.csv")

    log("Residual IC matrix ...")
    resid_mat, resid_long = residual_matrix(panels, ret)
    resid_mat.to_csv(OUT / "residual_ic_matrix.csv")
    resid_long.to_csv(OUT / "residual_ic_long.csv", index=False)

    clusters = cluster_factors(ic_corr, resid_long)
    (OUT / "factor_clusters.yaml").write_text(
        yaml.safe_dump(clusters, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    heatmap(ic_corr, "Daily RankIC correlation", OUT / "figures" / "ic_corr_heatmap.png")
    heatmap(ret_corr, "Daily H-L return correlation", OUT / "figures" / "return_corr_heatmap.png")
    # residual ICIR can exceed ±1 — clip display
    heatmap(
        resid_mat.fillna(0),
        "Residual ICIR (Y ⊥ X)",
        OUT / "figures" / "residual_icir_heatmap.png",
        vlim=12,
    )

    meta = {
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "n_days": int(len(ret)),
        "factors": names,
        "signal_book": "confirmation_size_industry_cs_z",
        "discovery_days": args.discovery_days,
    }
    write_report(OUT, meta, ic_corr, ret_corr, resid_long, clusters)

    verdict = {
        "schema_version": "factor_similarity_v1",
        "meta": meta,
        "ic_corr": ic_corr.round(4).to_dict(),
        "return_corr": ret_corr.round(4).to_dict(),
        "clusters": clusters,
        "composite": "deferred — Milestone Composite after human review",
    }
    (OUT / "similarity_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    log(f"Wrote {OUT / 'similarity_report.md'}")
    log("=== 1F complete (no Composite) ===")


if __name__ == "__main__":
    main()
