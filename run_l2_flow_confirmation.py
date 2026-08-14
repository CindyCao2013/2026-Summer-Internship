#!/usr/bin/env python
"""P2 confirmation + investability for net_active_flow_mktcap_20d.

Discovery 504d (already done) → Confirmation OOS (~951d) + gates:
  - ICIR > 3.0, residual_t vs Base3 > 2.0
  - Yearly / universe stability
  - Net Sharpe > 0 @ 15bp RT after tradability mask
  - Window ±30% (14/20/26) ICIR sign stable
  - Annu one-way turnover < 100% (soft), capacity note

Usage:
  OMP_NUM_THREADS=1 python run_l2_flow_confirmation.py
  OMP_NUM_THREADS=1 python run_l2_flow_confirmation.py --discovery-days 504
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
from alpha_d4_expansion_stack import daily_rank_ic_series, evaluate_stack_signal, icir_from_daily
from alpha_dimension_density import DISCOVERY_DAYS, residual_ic_stats, split_discovery_confirmation
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    evaluate_investability,
    strip_internal,
    yearly_net_sharpes,
)
from factor_attribution import combine_equal_weight, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_l2_v2 import build_l2_v2_factor
from factor_formulas_sue import neutralize_size_industry
from factor_runner import get_universe_mask
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache

OUT = Path("research/reports/l2_flow_density_v1")
FACTOR = "net_active_flow_mktcap_20d"
BASE3 = [
    "low_vol_liquidity_quality_60d",
    "winner_sentiment_reversal_5d",
    "upside_fragility_20d",
]
STACK_LAMBDAS = [0.0, 0.1, 0.2, 0.3]
PARAM_WINDOWS = [14, 20, 26]  # 20 ± 30%
UNIVERSES = {
    "CSI300": "000300.SH",
    "CSI500": "000905.SH",
    "CSI1000": "000852.SH",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def yearly_ic(panel: pd.DataFrame, ret: pd.DataFrame) -> dict:
    ic = daily_rank_ic_series(panel, ret)
    out = {}
    for year, sub in ic.groupby(ic.index.year):
        s = sub.dropna()
        if len(s) < 40:
            out[str(year)] = {"n": len(s), "ic": np.nan, "icir": np.nan}
        else:
            out[str(year)] = {
                "n": int(len(s)),
                "ic": float(s.mean()),
                "icir": float(s.mean() / s.std() * np.sqrt(250)) if s.std() > 0 else np.nan,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    log("=== P2 Confirmation + Investability ===")
    log(f"Factor={FACTOR} | discovery={args.discovery_days}d")

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    ret_disc, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
    log(
        f"Full {len(ret_full)}d | disc {ret_disc.index[0].date()}->{ret_disc.index[-1].date()} "
        f"({len(ret_disc)}) | conf {ret_conf.index[0].date()}->{ret_conf.index[-1].date()} "
        f"({len(ret_conf)})"
    )

    pv = build_factor_cache(
        df_close=enriched.close,
        df_open=enriched.open,
        df_high=enriched.high,
        df_low=enriched.low,
        df_volume=enriched.volume,
        df_amount=enriched.amount,
        df_turnover=enriched.turnover,
    )
    float_mkt = enriched.float_mktcap.loc[start:end]
    ind = industry.reindex(index=float_mkt.index, columns=float_mkt.columns)

    raw20 = build_net_active_flow_mktcap(l2_cache, float_mkt, window=20).loc[start:end]
    neut20 = cs_zscore(neutralize_size_industry(raw20, ind, float_mkt))

    base3_full = {
        n: build_eod_engine_factor(n, pv).loc[start:end] for n in BASE3
    }
    voi_full = build_l2_v2_factor("cn_voi_shock", l2_cache).reindex_like(neut20)

    def slice_panel(p: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
        return p.reindex(index=ret.index, columns=ret.columns)

    # --- Confirmation core ---
    panel_c = slice_panel(neut20, ret_conf)
    base3_c = [slice_panel(base3_full[n], ret_conf) for n in BASE3]
    base3_combo_c = combine_equal_weight(base3_c)
    voi_c = slice_panel(voi_full, ret_conf)

    ic_daily = daily_rank_ic_series(panel_c, ret_conf)
    icir = icir_from_daily(ic_daily)
    resid_b3 = residual_ic_stats(panel_c, ret_conf, base3_combo_c)
    resid_voi = residual_ic_stats(panel_c, ret_conf, voi_c)

    b, s = cs_zscore(base3_combo_c), cs_zscore(panel_c)
    stack_rows = []
    for lam in STACK_LAMBDAS:
        combo = (1.0 - lam) * b + lam * s if lam > 0 else b
        stack_rows.append({"lambda": lam, **evaluate_stack_signal(combo, ret_conf)})
    base_icir = stack_rows[0]["icir"]
    best = max(stack_rows, key=lambda r: r["icir"] if pd.notna(r.get("icir")) else -1e9)
    uplift = float(best["icir"] - base_icir) if pd.notna(best.get("icir")) and pd.notna(base_icir) else np.nan

    log(
        f"CONF IC={float(ic_daily.mean()):.4f} ICIR={icir:.2f} "
        f"resid_t_B3={resid_b3.get('residual_ic_t', np.nan):.2f} "
        f"resid_t_VOI={resid_voi.get('residual_ic_t', np.nan):.2f} "
        f"stack_uplift={uplift:.3f} best_λ={best.get('lambda')}"
    )

    # --- Yearly ---
    yearly = yearly_ic(panel_c, ret_conf)
    pos_years = [y for y, v in yearly.items() if pd.notna(v.get("ic")) and v["ic"] > 0]
    year_pos_ratio = len(pos_years) / max(len([y for y, v in yearly.items() if pd.notna(v.get("ic"))]), 1)
    log(f"Yearly IC+: {year_pos_ratio:.0%} | {yearly}")

    # --- Universes ---
    uni_rows = []
    for uname, code in UNIVERSES.items():
        try:
            mask = get_universe_mask(session, start, end, code)
            masked = panel_c.where(mask.reindex_like(panel_c) == 1)
            ic_u = daily_rank_ic_series(masked, ret_conf)
            uni_rows.append(
                {
                    "universe": uname,
                    "rank_ic": float(ic_u.mean()),
                    "icir": icir_from_daily(ic_u),
                    "n_ic_days": int(ic_u.dropna().shape[0]),
                }
            )
            log(f"  {uname}: IC={uni_rows[-1]['rank_ic']:.4f} ICIR={uni_rows[-1]['icir']:.2f}")
        except Exception as exc:
            log(f"  {uname} FAIL: {exc}")
            uni_rows.append({"universe": uname, "rank_ic": np.nan, "icir": np.nan, "error": str(exc)})

    # ALL
    uni_rows.append(
        {
            "universe": "ALL",
            "rank_ic": float(ic_daily.mean()),
            "icir": icir,
            "n_ic_days": int(ic_daily.dropna().shape[0]),
        }
    )

    # --- Param stability ---
    param_rows = []
    for w in PARAM_WINDOWS:
        raw_w = build_net_active_flow_mktcap(l2_cache, float_mkt, window=w).loc[start:end]
        neut_w = cs_zscore(neutralize_size_industry(raw_w, ind, float_mkt))
        p = slice_panel(neut_w, ret_conf)
        ic_w = daily_rank_ic_series(p, ret_conf)
        row = {
            "window": w,
            "rank_ic": float(ic_w.mean()),
            "icir": icir_from_daily(ic_w),
            "residual_ic_t_base3": residual_ic_stats(p, ret_conf, base3_combo_c).get("residual_ic_t"),
        }
        param_rows.append(row)
        log(f"  window={w}: IC={row['rank_ic']:.4f} ICIR={row['icir']:.2f} resid_t={row['residual_ic_t_base3']:.2f}")

    param_icirs = [r["icir"] for r in param_rows if pd.notna(r["icir"])]
    param_stable = bool(param_icirs) and all(x > 0 for x in param_icirs)

    # --- Investability ---
    df_not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start, end)
    df_not_st = Factor_Dev_Lib.get_EOD_Not_ST(start, end)
    df_trade_status = Factor_Dev_Lib.get_TradeStatus(start, end)
    trad_kw = dict(
        df_not_limit=df_not_limit.reindex_like(panel_c),
        df_not_st=df_not_st.reindex_like(panel_c),
        df_trade_status=df_trade_status.reindex_like(panel_c),
        close=enriched.close.reindex_like(panel_c),
        amount=enriched.amount.reindex_like(panel_c),
        round_trip_cost=DEFAULT_ROUND_TRIP_COST,
    )
    inv = evaluate_investability(panel_c, ret_conf, **trad_kw)
    inv_clean = strip_internal(inv)
    yearly_net = yearly_net_sharpes(inv["_net_pnl"])
    log(
        f"Investability: net_sharpe={inv['net_sharpe_tradable']:.3f} "
        f"gross={inv['gross_sharpe_tradable']:.3f} "
        f"TO_1way={inv['annu_one_way_turnover']:.1f}% "
        f"capacity≈{inv['capacity_cny_approx']:.3e}"
    )

    # --- Gates ---
    gates = {
        "icir_gt_3": bool(pd.notna(icir) and icir > 3.0),
        "residual_t_base3_gt_2": bool(
            pd.notna(resid_b3.get("residual_ic_t")) and abs(resid_b3["residual_ic_t"]) > 2.0
        ),
        "residual_t_voi_gt_2": bool(
            pd.notna(resid_voi.get("residual_ic_t")) and abs(resid_voi["residual_ic_t"]) > 2.0
        ),
        "stack_uplift_ge_0": bool(pd.notna(uplift) and uplift >= 0),
        "year_ic_pos_ratio_gt_70": bool(year_pos_ratio > 0.70),
        "no_two_consecutive_neg_ic_years": True,  # filled below
        "net_sharpe_gt_0": bool(pd.notna(inv["net_sharpe_tradable"]) and inv["net_sharpe_tradable"] > 0),
        "param_icir_sign_stable": param_stable,
        "turnover_1way_lt_100": bool(
            pd.notna(inv["annu_one_way_turnover"]) and inv["annu_one_way_turnover"] < 100
        ),
    }
    # consecutive negative IC years
    years_sorted = sorted(
        [(int(y), v["ic"]) for y, v in yearly.items() if pd.notna(v.get("ic"))]
    )
    consec_neg = False
    for i in range(len(years_sorted) - 1):
        if years_sorted[i][1] < 0 and years_sorted[i + 1][1] < 0:
            consec_neg = True
            break
    gates["no_two_consecutive_neg_ic_years"] = not consec_neg

    hard_pass = (
        gates["icir_gt_3"]
        and gates["residual_t_base3_gt_2"]
        and gates["stack_uplift_ge_0"]
        and gates["net_sharpe_gt_0"]
        and gates["param_icir_sign_stable"]
    )
    soft_notes = []
    if not gates["turnover_1way_lt_100"]:
        soft_notes.append("turnover_1way >= 100% (soft fence)")
    if not gates["year_ic_pos_ratio_gt_70"]:
        soft_notes.append("yearly IC+ ratio <= 70%")
    if not gates["residual_t_voi_gt_2"]:
        soft_notes.append("residual vs cn_voi_shock t<=2 (still check overlap)")

    if hard_pass and gates["turnover_1way_lt_100"] and gates["year_ic_pos_ratio_gt_70"]:
        verdict = "confirm_pass_enhancer"
    elif hard_pass:
        verdict = "confirm_pass_with_soft_flags"
    elif gates["icir_gt_3"] and gates["residual_t_base3_gt_2"]:
        verdict = "confirm_partial_investability_fail"
    else:
        verdict = "confirm_fail"

    summary = {
        "factor": FACTOR,
        "period_confirmation": f"{ret_conf.index[0].date()} -> {ret_conf.index[-1].date()}",
        "n_confirmation_days": int(len(ret_conf)),
        "discovery_days": args.discovery_days,
        "rank_ic": float(ic_daily.mean()),
        "icir": icir,
        "residual_ic_t_base3": resid_b3.get("residual_ic_t"),
        "residual_ic_t_voi": resid_voi.get("residual_ic_t"),
        "stack_icir_uplift": uplift,
        "best_lambda": best.get("lambda"),
        "year_ic_pos_ratio": year_pos_ratio,
        "net_sharpe_tradable": inv["net_sharpe_tradable"],
        "gross_sharpe_tradable": inv["gross_sharpe_tradable"],
        "annu_one_way_turnover": inv["annu_one_way_turnover"],
        "capacity_cny_approx": inv["capacity_cny_approx"],
        "param_stable": param_stable,
        "verdict": verdict,
    }
    pd.DataFrame([summary]).to_csv(OUT / "confirmation_summary.csv", index=False)
    pd.DataFrame(uni_rows).to_csv(OUT / "confirmation_universe.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT / "confirmation_param_stability.csv", index=False)
    pd.DataFrame(
        [{"year": y, **v} for y, v in yearly.items()]
    ).to_csv(OUT / "confirmation_yearly_ic.csv", index=False)

    payload = {
        "factor": FACTOR,
        "hard_gates": gates,
        "soft_notes": soft_notes,
        "verdict": verdict,
        "confirmation": summary,
        "residual_base3": resid_b3,
        "residual_voi": resid_voi,
        "stack": stack_rows,
        "yearly_ic": yearly,
        "universes": uni_rows,
        "param_windows": param_rows,
        "investability": inv_clean,
        "yearly_net_sharpe": yearly_net,
        "pass_criteria": {
            "icir": ">3.0",
            "residual_t_base3": ">2.0",
            "net_sharpe": ">0 @ 15bp RT",
            "param_±30%": "ICIR sign stable",
            "turnover_1way": "<100% soft",
            "year_ic_pos": ">70%",
        },
        "next_if_pass": [
            "Run C7 combo: C2_D1_0.60 + λ·z(P2) on confirmation",
            "Update production_stack_v3_design.md",
            "Update alpha_library frozen JSON enhancer slot",
        ],
    }
    (OUT / "confirmation_verdict.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    log(f"\nVERDICT: {verdict}")
    log(f"Gates: {json.dumps(gates)}")
    if soft_notes:
        log(f"Soft flags: {soft_notes}")
    log(f"Wrote {OUT / 'confirmation_summary.csv'}")
    log(f"Wrote {OUT / 'confirmation_verdict.json'}")


if __name__ == "__main__":
    main()
