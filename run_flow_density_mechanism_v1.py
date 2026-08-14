#!/usr/bin/env python
"""FlowDensity20 mechanism validation v1.

Answers:
  Q1: Is alpha just an amount / volume effect?
  Q2: Does net active flow beat buy/sell alone?
  Q3: Does size/industry residual still carry alpha?

Components (20d, / float mktcap unless noted):
  active_buy_mktcap_20d
  active_sell_mktcap_20d
  net_active_flow_mktcap_20d          ← canonical FlowDensity20
  gross_active_mktcap_20d             ← buy+sell (activity, no direction)
  amount_mktcap_20d                   ← total amount / mktcap
  volume_mktcap_20d                   ← L2 volume / mktcap
  active_buy_share_20d
  net_size_resid / net_size_industry_resid
  Flow ⊥ Amount / Volume / Gross / Buy / Sell

Usage:
  OMP_NUM_THREADS=1 python run_flow_density_mechanism_v1.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_dimension_density import (
    DISCOVERY_DAYS,
    residual_ic_stats,
    split_discovery_confirmation,
)
from alpha_investability import DEFAULT_ROUND_TRIP_COST, evaluate_investability
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_l2_flow_p2 import daily_net_active_amt
from factor_formulas_sue import neutralize_size_industry
from factor_runner import compute_group_stats
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from liquidity_normalization import panel_cross_sectional_residual

OUT = Path("research/reports/l2_flow_density_v1/mechanism")
FACTOR_ID = "FlowDensity20"
FACTOR_COL = "net_active_flow_mktcap_20d"
WINDOW = 20
MIN_P = 10
N_GROUPS = 10
SIGNAL_SHIFT = 1

# Soft gates to promote candidate → validated_single_factor
MIN_RESID_ICIR = 2.0
MIN_RESID_T = 2.0


def log(msg: str) -> None:
    print(msg, flush=True)


def roll_sum(x: pd.DataFrame) -> pd.DataFrame:
    return x.rolling(WINDOW, min_periods=MIN_P).sum()


def roll_mean(x: pd.DataFrame) -> pd.DataFrame:
    return x.rolling(WINDOW, min_periods=MIN_P).mean()


def build_components(
    cache,
    float_mkt: pd.DataFrame,
    amount_eod: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    mcap = float_mkt.reindex(index=cache.active_buy_amt.index, columns=cache.active_buy_amt.columns)
    mcap = mcap.replace(0, np.nan)

    buy = cache.active_buy_amt / mcap
    sell = cache.active_sell_amt / mcap
    net = daily_net_active_amt(cache) / mcap
    gross = (cache.active_buy_amt + cache.active_sell_amt) / mcap
    amt_l2 = cache.amount.reindex_like(mcap) / mcap
    vol_l2 = cache.volume.reindex_like(mcap) / mcap
    # EOD amount as alternative liquidity proxy (千元 / 万元 convention cancels in CS z)
    amt_eod = amount_eod.reindex_like(mcap) / mcap
    buy_share = cache.active_buy_amt / (cache.active_buy_amt + cache.active_sell_amt).replace(0, np.nan)

    comps = {
        "active_buy_mktcap_20d": roll_sum(buy),
        "active_sell_mktcap_20d": roll_sum(sell),
        FACTOR_COL: roll_sum(net),
        "gross_active_mktcap_20d": roll_sum(gross),
        "amount_mktcap_20d": roll_sum(amt_l2),
        "amount_eod_mktcap_20d": roll_sum(amt_eod),
        "volume_mktcap_20d": roll_sum(vol_l2),
        "active_buy_share_20d": roll_mean(buy_share),
    }
    return {k: cs_zscore(v) for k, v in comps.items()}


def eval_panel(
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
    p = cs_zscore(panel.reindex_like(ret))
    sig = align_signal(p, SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    _, pnl, to = Factor_Dev_Lib.groupTest(sig, r, n=N_GROUPS, fee=0, info="silent")
    stats = compute_group_stats(sig, r, pnl, to)
    inv = evaluate_investability(
        p,
        ret,
        df_not_limit=masks["df_not_limit"].reindex_like(ret),
        df_not_st=masks["df_not_st"].reindex_like(ret),
        df_trade_status=masks["df_trade_status"].reindex_like(ret),
        close=close,
        amount=amount,
        round_trip_cost=DEFAULT_ROUND_TRIP_COST,
        signal_shift=SIGNAL_SHIFT,
        top_frac=0.10,
        min_listing_days=60 if len(ret) >= 120 else 0,
    )
    row = {
        "signal": name,
        "family": family,
        "rank_ic": stats["rank_ic_mean"],
        "icir": stats["icir"],
        "hl_sharpe": stats["hl_sharpe"],
        "hl_annu_ret": stats["hl_annu_ret"],
        "hl_mdd": stats["hl_mdd"],
        "daily_turnover": stats["hl_avg_turnover"],
        "net_sharpe": inv["net_sharpe_tradable"],
        "direction": stats["direction"],
        "note": note,
    }
    log(
        f"  {name:32s} ICIR={row['icir']:7.2f} RankIC={row['rank_ic']:.4f} "
        f"HL={row['hl_sharpe']:.2f} net={row['net_sharpe']:.2f}"
    )
    return row


def mean_cs_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean daily cross-sectional Spearman corr."""
    aa = a.reindex_like(b)
    corrs = []
    for dt in aa.index:
        x = aa.loc[dt]
        y = b.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() < 30:
            continue
        c = x[m].corr(y[m], method="spearman")
        if pd.notna(c):
            corrs.append(float(c))
    return float(np.mean(corrs)) if corrs else float("nan")


def resid_row(name: str, factor: pd.DataFrame, anchor: pd.DataFrame, ret: pd.DataFrame, note: str) -> dict:
    stats = residual_ic_stats(factor, ret, anchor, signal_shift=SIGNAL_SHIFT)
    corr = mean_cs_corr(factor, anchor)
    raw_ic = daily_rank_ic_series(factor, ret, signal_shift=SIGNAL_SHIFT)
    return {
        "signal": name,
        "family": "residual",
        "rank_ic": stats["residual_ic_mean"],
        "icir": stats["residual_icir"],
        "residual_ic_t": stats["residual_ic_t"],
        "cs_corr_with_anchor": corr,
        "raw_icir": float(icir_from_daily(raw_ic)) if raw_ic.notna().sum() > 20 else np.nan,
        "hl_sharpe": np.nan,
        "net_sharpe": np.nan,
        "direction": 1 if (stats["residual_ic_mean"] or 0) >= 0 else -1,
        "note": note,
    }


def decide_verdict(comp: pd.DataFrame, resid: pd.DataFrame) -> dict:
    def icir_of(sig: str) -> float:
        sub = comp.loc[comp["signal"] == sig]
        return float(sub["icir"].iloc[0]) if len(sub) else np.nan

    net = icir_of(FACTOR_COL)
    buy = icir_of("active_buy_mktcap_20d")
    sell = icir_of("active_sell_mktcap_20d")
    gross = icir_of("gross_active_mktcap_20d")
    amount = icir_of("amount_mktcap_20d")
    buy_share = icir_of("active_buy_share_20d")

    def resid_stats(sig: str) -> Tuple[float, float, float]:
        sub = resid.loc[resid["signal"] == sig]
        if sub.empty:
            return np.nan, np.nan, np.nan
        return (
            float(sub["icir"].iloc[0]),
            float(sub["residual_ic_t"].iloc[0]),
            float(sub["cs_corr_with_anchor"].iloc[0]),
        )

    r_amt_icir, r_amt_t, corr_amt = resid_stats("Flow_perp_Amount")
    r_vol_icir, r_vol_t, corr_vol = resid_stats("Flow_perp_Volume")
    r_gross_icir, r_gross_t, corr_gross = resid_stats("Flow_perp_GrossActive")

    # Q2: direction vs undirected activity — net should flip sign vs buy/sell/gross/amount
    direction_not_activity = (
        pd.notna(net)
        and net >= MIN_RESID_ICIR
        and pd.notna(amount)
        and amount <= -MIN_RESID_ICIR
        and pd.notna(gross)
        and gross <= -MIN_RESID_ICIR
        and pd.notna(buy)
        and buy <= -MIN_RESID_ICIR
    )
    # Buy/sell alone are activity clones; net is not "buy alone"
    net_distinct_from_legs = direction_not_activity and (
        np.sign(net) != np.sign(buy) if pd.notna(buy) and buy != 0 else True
    )

    # Q1 strict: positive residual alpha after removing amount (pure direction)
    residual_positive_vs_amount = (
        pd.notna(r_amt_icir) and r_amt_icir >= MIN_RESID_ICIR and abs(r_amt_t) >= MIN_RESID_T
    )
    residual_positive_vs_volume = (
        pd.notna(r_vol_icir) and r_vol_icir >= MIN_RESID_ICIR and abs(r_vol_t) >= MIN_RESID_T
    )
    residual_positive_vs_gross = (
        pd.notna(r_gross_icir) and r_gross_icir >= MIN_RESID_ICIR and abs(r_gross_t) >= MIN_RESID_T
    )
    # Soft: residual exists but may flip sign → entangled with anti-amount/liquidity
    residual_exists_vs_amount = pd.notna(r_amt_t) and abs(r_amt_t) >= MIN_RESID_T

    pass_strict = bool(
        direction_not_activity
        and residual_positive_vs_amount
        and residual_positive_vs_volume
    )
    entangled = bool(
        direction_not_activity
        and residual_exists_vs_amount
        and not residual_positive_vs_amount
        and (pd.notna(corr_amt) and corr_amt < -0.3)
    )

    if pass_strict:
        verdict = "mechanism_pass"
        interpretation = (
            "Alpha is active-flow direction (net imbalance). Residual IC after Amount/Volume "
            "controls remains positive — not a raw amount/volume effect."
        )
    elif entangled:
        verdict = "mechanism_entangled_with_anti_amount"
        interpretation = (
            "Net flow is NOT undirected activity: buy/sell/gross/amount all have large "
            f"NEGATIVE ICIR (amount={amount:.2f}, gross={gross:.2f}) while net has POSITIVE "
            f"ICIR ({net:.2f}). However Flow⊥Amount residual ICIR={r_amt_icir:.2f} "
            f"(cs_corr={corr_amt:.3f}) — positive net alpha is entangled with anti-amount / "
            "low-activity exposure. Do NOT freeze yet; consider amount-orthogonalized net "
            "or document liquidity entanglement before validated_single_factor."
        )
    else:
        verdict = "mechanism_incomplete"
        interpretation = (
            "Mechanism gates not fully met — keep validated_single_factor_candidate."
        )

    def _b(x) -> bool:
        return bool(x)

    def _f(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return x

    return {
        "verdict": verdict,
        "promote_to_validated_single_factor": _b(pass_strict),
        "freeze_formula": _b(pass_strict),
        "gates": {
            "direction_not_activity": _b(direction_not_activity),
            "net_distinct_from_buy_sell_legs": _b(net_distinct_from_legs),
            "residual_positive_vs_amount": _b(residual_positive_vs_amount),
            "residual_positive_vs_volume": _b(residual_positive_vs_volume),
            "residual_positive_vs_gross": _b(residual_positive_vs_gross),
            "residual_exists_vs_amount": _b(residual_exists_vs_amount),
            "entangled_with_anti_amount": _b(entangled),
            "buy_share_weak": _b(pd.notna(buy_share) and abs(buy_share) < 2.0),
        },
        "icir": {
            "net": _f(net),
            "buy": _f(buy),
            "sell": _f(sell),
            "gross": _f(gross),
            "amount": _f(amount),
            "buy_share": _f(buy_share),
            "Flow_perp_Amount": _f(r_amt_icir),
            "Flow_perp_Volume": _f(r_vol_icir),
            "Flow_perp_GrossActive": _f(r_gross_icir),
            "cs_corr_amount": _f(corr_amt),
            "cs_corr_volume": _f(corr_vol),
            "cs_corr_gross": _f(corr_gross),
        },
        "interpretation": interpretation,
    }


def write_report(out: Path, mech: pd.DataFrame, verdict: dict) -> None:
    g = verdict["gates"]
    ic = verdict["icir"]
    lines = [
        "# FlowDensity20 Mechanism Validation v1",
        "",
        f"**Canonical factor:** `{FACTOR_COL}`",
        f"**Verdict:** `{verdict['verdict']}`",
        f"**Promote to validated_single_factor:** `{verdict['promote_to_validated_single_factor']}`",
        f"**Freeze formula:** `{verdict['freeze_formula']}`",
        "",
        "## Research questions",
        "",
        "1. Is alpha just an amount / volume effect?",
        "2. Does net active flow beat buy/sell alone?",
        "3. Does size / size+industry residual still carry alpha?",
        "",
        "## Component ICIR (size+industry signal unless noted)",
        "",
        "| Signal | Family | RankIC | ICIR | H-L Sharpe | Net@15bp | Note |",
        "|--------|--------|-------:|-----:|-----------:|---------:|------|",
    ]
    for _, r in mech.iterrows():
        hl = r["hl_sharpe"] if pd.notna(r.get("hl_sharpe")) else float("nan")
        ns = r["net_sharpe"] if pd.notna(r.get("net_sharpe")) else float("nan")
        lines.append(
            f"| `{r['signal']}` | {r['family']} | {r['rank_ic']:.4f} | {r['icir']:.2f} | "
            f"{hl:.2f} | {ns:.2f} | {r.get('note', '')} |"
        )

    lines += [
        "",
        "## Gate checklist",
        "",
        f"- direction ≠ undirected activity (net>0, amount/gross/buy≪0): "
        f"**{g['direction_not_activity']}**",
        f"- net distinct from buy/sell legs (sign flip): **{g['net_distinct_from_buy_sell_legs']}**",
        f"- residual POSITIVE vs Amount (strict freeze gate): "
        f"**{g['residual_positive_vs_amount']}** (ICIR={ic['Flow_perp_Amount']:.2f}, "
        f"corr={ic['cs_corr_amount']:.3f})",
        f"- residual POSITIVE vs Volume: **{g['residual_positive_vs_volume']}** "
        f"(ICIR={ic['Flow_perp_Volume']:.2f})",
        f"- residual POSITIVE vs GrossActive: **{g['residual_positive_vs_gross']}** "
        f"(ICIR={ic['Flow_perp_GrossActive']:.2f})",
        f"- entangled with anti-amount: **{g['entangled_with_anti_amount']}**",
        "",
        "## Interpretation",
        "",
        verdict["interpretation"],
        "",
        "## Artifacts",
        "",
        "- `mechanism.csv` — canonical pack file",
        "- `mechanism_components.csv` — full eval table",
        "- `mechanism_residuals.csv` — residual IC table",
        "- `mechanism_verdict.json`",
        "",
        "## Next (if entangled)",
        "",
        "1. Build `net_active_flow_mktcap_20d ⊥ amount_mktcap_20d` as candidate cleaner signal",
        "2. Or document liquidity entanglement and keep combination-only use",
        "3. Then TGD ⟂ Flow orthogonality",
        "",
    ]
    (out / "mechanism_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("=== FlowDensity20 Mechanism Validation v1 ===")
    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2 = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

    float_mkt = enriched.float_mktcap.loc[start:end]
    comps_full = build_components(l2, float_mkt, enriched.amount.loc[start:end])
    # slice to research window
    for k in list(comps_full):
        comps_full[k] = comps_full[k].loc[start:end]

    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret_full = ret_full.reindex(index=comps_full[FACTOR_COL].index, columns=comps_full[FACTOR_COL].columns)
    _, ret = split_discovery_confirmation(ret_full, args.discovery_days)
    if ret.empty:
        ret = ret_full
    log(f"Confirmation: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    d0 = ret.index[0].to_pydatetime()
    d1 = ret.index[-1].to_pydatetime()
    masks = {
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(d0, d1),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(d0, d1),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(d0, d1),
    }
    close = enriched.close.reindex_like(ret)
    amount = enriched.amount.reindex_like(ret)
    ind = industry.reindex_like(ret)
    mkt = float_mkt.reindex_like(ret)

    # Neutralized views of components (primary comparison table)
    rows = []
    log("\n--- Component ladder (size+industry) ---")
    for name, panel in comps_full.items():
        p = panel.reindex_like(ret)
        neut = cs_zscore(neutralize_size_industry(p, ind, mkt))
        family = "canonical" if name == FACTOR_COL else "component"
        rows.append(
            eval_panel(
                name,
                neut,
                ret,
                family=family,
                note="size+industry",
                masks=masks,
                close=close,
                amount=amount,
            )
        )

    # Size / size+industry residual of net (explicit rows)
    log("\n--- Style residuals of net flow ---")
    net_raw = comps_full[FACTOR_COL].reindex_like(ret)
    log_size = np.log(mkt.replace(0, np.nan))
    net_size = cs_zscore(panel_cross_sectional_residual(net_raw, [log_size]))
    net_si = cs_zscore(neutralize_size_industry(net_raw, ind, mkt))
    rows.append(
        eval_panel(
            "net_size_resid",
            net_size,
            ret,
            family="style_residual",
            note="Flow ⊥ size",
            masks=masks,
            close=close,
            amount=amount,
        )
    )
    rows.append(
        eval_panel(
            "net_size_industry_resid",
            net_si,
            ret,
            family="style_residual",
            note="Flow ⊥ size+industry (same as confirmation)",
            masks=masks,
            close=close,
            amount=amount,
        )
    )

    # Also evaluate raw (no neut) canonical for reference
    log("\n--- Canonical raw (no neut) ---")
    rows.append(
        eval_panel(
            f"{FACTOR_COL}|raw",
            net_raw,
            ret,
            family="canonical_raw",
            note="raw cs_z",
            masks=masks,
            close=close,
            amount=amount,
        )
    )

    comp_df = pd.DataFrame(rows)

    # Residual independence tests (on size+industry net vs anchors)
    log("\n--- Residual alpha (Flow ⊥ anchors) ---")
    net_si_panel = net_si
    anchors = {
        "Flow_perp_Amount": (comps_full["amount_mktcap_20d"].reindex_like(ret), "⊥ L2 amount/mktcap 20d"),
        "Flow_perp_AmountEOD": (comps_full["amount_eod_mktcap_20d"].reindex_like(ret), "⊥ EOD amount/mktcap 20d"),
        "Flow_perp_Volume": (comps_full["volume_mktcap_20d"].reindex_like(ret), "⊥ L2 volume/mktcap 20d"),
        "Flow_perp_GrossActive": (
            comps_full["gross_active_mktcap_20d"].reindex_like(ret),
            "⊥ undirected gross active flow",
        ),
        "Flow_perp_Buy": (comps_full["active_buy_mktcap_20d"].reindex_like(ret), "⊥ buy leg"),
        "Flow_perp_Sell": (comps_full["active_sell_mktcap_20d"].reindex_like(ret), "⊥ sell leg"),
    }
    resid_rows = []
    for name, (anchor, note) in anchors.items():
        # residualize neutralized net vs neutralized anchor for fair test
        a_neut = cs_zscore(neutralize_size_industry(cs_zscore(anchor), ind, mkt))
        rr = resid_row(name, net_si_panel, a_neut, ret, note)
        resid_rows.append(rr)
        log(
            f"  {name:28s} resid_ICIR={rr['icir']:.2f} t={rr['residual_ic_t']:.2f} "
            f"cs_corr={rr['cs_corr_with_anchor']:.3f}"
        )

    resid_df = pd.DataFrame(resid_rows)

    # Canonical mechanism.csv for pack (TGD-compatible columns + extras)
    mech = pd.concat(
        [
            comp_df[
                [
                    "signal",
                    "family",
                    "rank_ic",
                    "icir",
                    "hl_sharpe",
                    "net_sharpe",
                    "daily_turnover",
                    "direction",
                    "note",
                ]
            ],
            resid_df[
                [
                    "signal",
                    "family",
                    "rank_ic",
                    "icir",
                    "hl_sharpe",
                    "net_sharpe",
                    "direction",
                    "note",
                ]
            ].assign(daily_turnover=np.nan),
        ],
        ignore_index=True,
    )
    # attach residual_ic_t / corr where available
    mech = mech.merge(
        resid_df[["signal", "residual_ic_t", "cs_corr_with_anchor"]],
        on="signal",
        how="left",
    )

    verdict = decide_verdict(comp_df, resid_df)
    log(f"\nVerdict: {verdict['verdict']} | promote={verdict['promote_to_validated_single_factor']}")

    mech.to_csv(OUT / "mechanism.csv", index=False)
    mech.to_csv(OUT / "mechanism_analysis.csv", index=False)
    comp_df.to_csv(OUT / "mechanism_components.csv", index=False)
    resid_df.to_csv(OUT / "mechanism_residuals.csv", index=False)
    (OUT / "mechanism_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    write_report(OUT, mech, verdict)
    log(f"Wrote {OUT / 'mechanism_summary.md'}")


if __name__ == "__main__":
    main()
