#!/usr/bin/env python
"""FlowDensity amount-orthogonalization analysis v1 — attribution, not freeze.

Purpose:
  Decompose FlowDensity into Flow × Liquidity interaction vs pure flow direction.
  Does NOT auto-promote to validated_single_factor.

Builds tradable panels:
  FlowDensity_raw          = net_active_flow_mktcap_20d (size+industry)
  Amount                   = amount_mktcap_20d (size+industry)
  Flow_perp_Amount         = CS residual: Flow ~ Amount → ε  (then size+ind optional)
  Amount_perp_Flow         = CS residual: Amount ~ Flow → ε  (attribution control)

Compares RankIC / ICIR / H-L / mono / Net Sharpe.

Outputs:
  research/reports/l2_flow_density_v1/mechanism/
    mechanism_amount_neutral.csv
    amount_orth_summary.md
    amount_orth_verdict.json

Usage:
  OMP_NUM_THREADS=1 python run_flow_density_amount_orth_v1.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from factor_runner import compute_group_stats
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from liquidity_normalization import panel_cross_sectional_residual

from run_flow_density_mechanism_v1 import (
    FACTOR_COL,
    N_GROUPS,
    OUT,
    SIGNAL_SHIFT,
    build_components,
    eval_panel,
    mean_cs_corr,
)

FACTOR_ID = "FlowDensity20"


def log(msg: str) -> None:
    print(msg, flush=True)


def decile_monotonicity(pnl: pd.DataFrame) -> float:
    means = pnl.mean()
    try:
        order = pd.to_numeric(means.index, errors="coerce").to_numpy(dtype=float)
    except Exception:
        order = np.arange(1, len(means) + 1, dtype=float)
    y = means.to_numpy(dtype=float)
    m = np.isfinite(order) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(pd.Series(order[m]).corr(pd.Series(y[m]), method="spearman"))


def eval_with_mono(
    name: str,
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    family: str,
    note: str,
    masks: dict,
    close: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict:
    row = eval_panel(
        name,
        panel,
        ret,
        family=family,
        note=note,
        masks=masks,
        close=close,
        amount=amount,
    )
    p = cs_zscore(panel.reindex_like(ret))
    sig = align_signal(p, SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    _, pnl, _ = Factor_Dev_Lib.groupTest(sig, r, n=N_GROUPS, fee=0, info="silent")
    row["monotonicity"] = decile_monotonicity(pnl)
    return row


def classify_attribution(rows: pd.DataFrame, corr_flow_amt: float) -> dict:
    def get(sig: str, col: str) -> float:
        sub = rows.loc[rows["signal"] == sig]
        return float(sub[col].iloc[0]) if len(sub) else float("nan")

    raw_icir = get("FlowDensity_raw", "icir")
    perp_icir = get("Flow_perp_Amount", "icir")
    amt_icir = get("Amount", "icir")
    amt_perp_icir = get("Amount_perp_Flow", "icir")
    raw_hl = get("FlowDensity_raw", "hl_sharpe")
    perp_hl = get("Flow_perp_Amount", "hl_sharpe")
    raw_net = get("FlowDensity_raw", "net_sharpe")
    perp_net = get("Flow_perp_Amount", "net_sharpe")

    # Fraction of |ICIR| retained after amount orth
    if pd.notna(raw_icir) and abs(raw_icir) > 1e-9 and pd.notna(perp_icir):
        retain = float(perp_icir / raw_icir)  # signed retention
        retain_abs = float(abs(perp_icir) / abs(raw_icir))
    else:
        retain, retain_abs = float("nan"), float("nan")

    # Case classification (attribution, not freeze gate)
    if pd.notna(perp_icir) and perp_icir >= 2.0:
        case = "case1_pure_flow_survives"
        true_info = "pure_flow_direction_plus_liquidity_interaction"
        note = (
            f"Flow⊥Amount ICIR={perp_icir:.2f} ≥ 2 — independent flow direction exists "
            f"alongside anti-amount loading (corr={corr_flow_amt:.3f})."
        )
    elif pd.notna(perp_icir) and abs(perp_icir) < 1.0:
        case = "case2_mostly_anti_amount"
        true_info = "primarily_anti_amount_liquidity"
        note = (
            f"Flow⊥Amount ICIR={perp_icir:.2f} ≈ 0 — raw FlowDensity alpha is largely "
            f"anti-amount / liquidity. Consider rename LiquidityAdjustedFlow."
        )
    else:
        case = "case_interaction_entangled"
        true_info = "flow_liquidity_interaction"
        note = (
            f"Flow⊥Amount ICIR={perp_icir:.2f} (signed retain={retain:.2f}) — neither pure "
            f"flow nor pure amount. Classify as Flow × Liquidity interaction."
        )

    return {
        "factor_id": FACTOR_ID,
        "source_column": FACTOR_COL,
        "analysis": "amount_orthogonalization_v1",
        "purpose": "attribution_not_freeze",
        "promote_to_validated_single_factor": False,
        "freeze_formula": False,
        "category": "liquidity_flow_interaction",
        "mechanism_class": "flow_liquidity_interaction",
        "attribution_case": case,
        "true_information": true_info,
        "cs_corr_flow_amount": float(corr_flow_amt) if pd.notna(corr_flow_amt) else None,
        "icir": {
            "FlowDensity_raw": raw_icir,
            "Flow_perp_Amount": perp_icir,
            "Amount": amt_icir,
            "Amount_perp_Flow": amt_perp_icir,
            "signed_retention_vs_raw": retain,
            "abs_retention_vs_raw": retain_abs,
        },
        "hl_sharpe": {"FlowDensity_raw": raw_hl, "Flow_perp_Amount": perp_hl},
        "net_sharpe": {"FlowDensity_raw": raw_net, "Flow_perp_Amount": perp_net},
        "status": "validated_single_factor_candidate",
        "interpretation": note,
        "factor_map": {
            "TGD20": "Temporal return structure",
            "FlowDensity_raw": "Flow + liquidity (interaction)",
            "Flow_perp_Amount": "Pure flow candidate (if Case 1)",
            "Amount": "Anti-activity / liquidity premium channel",
        },
    }


def write_summary(out: Path, table: pd.DataFrame, verdict: dict) -> None:
    ic = verdict["icir"]
    lines = [
        "# FlowDensity Amount-Orthogonalization v1",
        "",
        "**Purpose:** factor attribution — *not* a freeze gate.",
        f"**Mechanism class:** `{verdict['mechanism_class']}`",
        f"**Attribution case:** `{verdict['attribution_case']}`",
        f"**Status:** keep `{verdict['status']}` (no auto-promote)",
        "",
        "## Research question",
        "",
        "> After removing anti-amount / low-activity exposure, does pure flow direction still have alpha?",
        "",
        "## Comparison table",
        "",
        "| Signal | Family | RankIC | ICIR | H-L Sharpe | Mono | Net@15bp | Note |",
        "|--------|--------|-------:|-----:|-----------:|-----:|---------:|------|",
    ]
    for _, r in table.iterrows():
        mono = r["monotonicity"] if pd.notna(r.get("monotonicity")) else float("nan")
        lines.append(
            f"| `{r['signal']}` | {r['family']} | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{r['hl_sharpe']:.2f} | {mono:.3f} | {r['net_sharpe']:.2f} | {r['note']} |"
        )

    lines += [
        "",
        "## Attribution summary",
        "",
        f"- cs_corr(Flow, Amount) = **{verdict['cs_corr_flow_amount']:.3f}**",
        f"- Raw ICIR = **{ic['FlowDensity_raw']:.2f}**",
        f"- Flow⊥Amount ICIR = **{ic['Flow_perp_Amount']:.2f}** "
        f"(signed retention {ic['signed_retention_vs_raw']:.2f})",
        f"- Amount ICIR = **{ic['Amount']:.2f}**",
        f"- Amount⊥Flow ICIR = **{ic['Amount_perp_Flow']:.2f}** "
        "(liquidity channel after removing flow)",
        "",
        "## Interpretation",
        "",
        verdict["interpretation"],
        "",
        "## Factor map (working)",
        "",
        "| Factor | True information |",
        "|--------|------------------|",
    ]
    for k, v in verdict["factor_map"].items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Do not",
        "",
        "- Auto-freeze FlowDensity20 from this run",
        "- Jump to TGD×Flow composite before orthogonality + explicit liquidity exposure note",
        "",
        "## Next",
        "",
        "1. Keep raw FlowDensity as `liquidity_flow_interaction` candidate",
        "2. Optionally track `Flow_perp_Amount` as a research satellite (not production rename yet)",
        "3. Run **TGD20 ⟂ FlowDensity** with both raw and perp variants in the independence table",
        "",
        "## Artifacts",
        "",
        "- `mechanism_amount_neutral.csv`",
        "- `amount_orth_verdict.json`",
        "",
    ]
    (out / "amount_orth_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("=== FlowDensity Amount-Orthogonalization v1 (attribution) ===")
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

    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret_full = ret_full.reindex(index=comps[FACTOR_COL].index, columns=comps[FACTOR_COL].columns)
    _, ret = split_discovery_confirmation(ret_full, args.discovery_days)
    if ret.empty:
        ret = ret_full
    log(f"Confirmation: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    d0, d1 = ret.index[0].to_pydatetime(), ret.index[-1].to_pydatetime()
    masks = {
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(d0, d1),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(d0, d1),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(d0, d1),
    }
    close = enriched.close.reindex_like(ret)
    amount_eod = enriched.amount.reindex_like(ret)
    ind = industry.reindex_like(ret)
    mkt = float_mkt.reindex_like(ret)

    flow_raw = comps[FACTOR_COL].reindex_like(ret)
    amt = comps["amount_mktcap_20d"].reindex_like(ret)
    gross = comps["gross_active_mktcap_20d"].reindex_like(ret)

    # Neutralize style first (same book as confirmation), then amount-orth on that space
    flow_si = cs_zscore(neutralize_size_industry(flow_raw, ind, mkt))
    amt_si = cs_zscore(neutralize_size_industry(amt, ind, mkt))
    gross_si = cs_zscore(neutralize_size_industry(gross, ind, mkt))

    flow_perp_amt = cs_zscore(panel_cross_sectional_residual(flow_si, [amt_si]))
    amt_perp_flow = cs_zscore(panel_cross_sectional_residual(amt_si, [flow_si]))
    # Also: amount-orth before style neut (diagnostic)
    flow_perp_amt_rawspace = cs_zscore(
        neutralize_size_industry(
            cs_zscore(panel_cross_sectional_residual(flow_raw, [amt])),
            ind,
            mkt,
        )
    )

    corr_fa = mean_cs_corr(flow_si, amt_si)
    log(f"cs_corr(Flow_si, Amount_si) = {corr_fa:.3f}")

    specs = [
        ("FlowDensity_raw", flow_si, "canonical", "size+ind net flow (confirmation signal)"),
        ("Amount", amt_si, "liquidity_channel", "size+ind amount/mktcap 20d"),
        ("GrossActive", gross_si, "liquidity_channel", "size+ind gross active/mktcap 20d"),
        (
            "Flow_perp_Amount",
            flow_perp_amt,
            "amount_orthogonal",
            "ε from Flow_si ~ Amount_si (tradable residual panel)",
        ),
        (
            "Amount_perp_Flow",
            amt_perp_flow,
            "amount_orthogonal",
            "ε from Amount_si ~ Flow_si (liquidity after removing flow)",
        ),
        (
            "Flow_perp_Amount_then_SI",
            flow_perp_amt_rawspace,
            "amount_orthogonal_alt",
            "ε from Flow~Amount in raw space, then size+ind",
        ),
    ]

    rows = []
    log("\n--- Evaluation ---")
    for name, panel, family, note in specs:
        rows.append(
            eval_with_mono(
                name,
                panel,
                ret,
                family=family,
                note=note,
                masks=masks,
                close=close,
                amount=amount_eod,
            )
        )

    table = pd.DataFrame(rows)
    # Column order for pack
    cols = [
        "signal",
        "family",
        "rank_ic",
        "icir",
        "hl_sharpe",
        "hl_annu_ret",
        "hl_mdd",
        "monotonicity",
        "daily_turnover",
        "net_sharpe",
        "direction",
        "note",
    ]
    table = table[cols]

    verdict = classify_attribution(table, corr_fa)
    log(f"\nCase: {verdict['attribution_case']}")
    log(verdict["interpretation"])

    table.to_csv(OUT / "mechanism_amount_neutral.csv", index=False)
    (OUT / "amount_orth_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    write_summary(OUT, table, verdict)
    log(f"Wrote {OUT / 'mechanism_amount_neutral.csv'}")
    log(f"Wrote {OUT / 'amount_orth_summary.md'}")


if __name__ == "__main__":
    main()
