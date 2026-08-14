#!/usr/bin/env python
"""Head-to-head: C2_D1_0.60 vs C6_D1_0.60_state_mod on confirmation.

If C6 ICIR >= C2 and TO<=120 and net>0 → upgrade recommendation to C6.
Otherwise keep C2_D1_0.60 as research freeze.

Usage:
  OMP_NUM_THREADS=1 python run_c6_vs_c2_head_to_head.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import factor_config as cfg
import numpy as np
import pandas as pd

from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import DEFAULT_ROUND_TRIP_COST, evaluate_investability, strip_internal
from factor_attribution import cs_zscore
from run_alpha_combination_v1 import (
    D1,
    D4,
    D5,
    blend_base3,
    load_cancel_and_close,
    load_masks,
    state_modifier_blend,
)
from run_l2_validation import load_context

OUT = Path("research/results/alpha_combination_v1")
COMB_JSON = OUT / "alpha_combination_v1.json"
LIBRARY_PATH = Path("research/alpha_library_v1/alpha_library_v1.0-frozen.json")
POOL_PATH = Path("research/frozen_candidate_pool_v1.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    log("=== C6@0.60 vs C2@0.60 head-to-head ===")
    ctx = load_context(sample_days=9999, engine="v2", build_cluster_reps=False)
    ret_full = ctx["ret"]
    _, ret_conf = split_discovery_confirmation(ret_full, DISCOVERY_DAYS)
    log(f"Confirmation: {ret_conf.index[0].date()} -> {ret_conf.index[-1].date()} ({len(ret_conf)}d)")

    masks = load_masks(cfg.START_DAY, cfg.END_DAY)
    cancel, close, amount = load_cancel_and_close(ctx, cfg.START_DAY, cfg.END_DAY)
    panels = {
        n: ctx["frozen_panels"][n].reindex(index=ret_conf.index, columns=ret_conf.columns)
        for n in [D1, D4, D5]
    }
    cancel_c = cancel.reindex(index=ret_conf.index, columns=ret_conf.columns)
    close_c = close.reindex(index=ret_conf.index, columns=ret_conf.columns)
    amount_c = amount.reindex(index=ret_conf.index, columns=ret_conf.columns)

    trad_kw = dict(
        df_not_limit=masks["df_not_limit"],
        df_not_st=masks["df_not_st"],
        df_trade_status=masks["df_trade_status"],
        close=close_c,
        amount=amount_c,
        round_trip_cost=DEFAULT_ROUND_TRIP_COST,
        apply_tradability=True,
    )

    sig_c2 = blend_base3(panels, 0.60)
    sig_c6, n_stress = state_modifier_blend(
        panels, cancel_c, w_d1=0.60, w_d4=0.20, w_d5=0.20, stress_quantile=0.20
    )
    log(f"C6 stress days: {n_stress}/{len(ret_conf)}")

    inv_c2 = evaluate_investability(sig_c2, ret_conf, **trad_kw)
    inv_c6 = evaluate_investability(sig_c6, ret_conf, **trad_kw)

    rows = []
    for label, inv, extra in [
        ("C2_D1_0.60", inv_c2, {}),
        ("C6_D1_0.60_state_mod", inv_c6, {"n_stress_days": n_stress}),
    ]:
        row = {
            "label": label,
            "IC": inv["rank_ic_tradable"],
            "ICIR": inv["icir_tradable"],
            "gross_Sharpe": inv["gross_sharpe_tradable"],
            "annu_TO_1way": inv["annu_one_way_turnover"],
            "net_Sharpe": inv["net_sharpe_tradable"],
            "feasible": bool(inv["annu_one_way_turnover"] <= 120 and inv["net_sharpe_tradable"] > 0),
            **extra,
        }
        rows.append(row)
        log(
            f"  {label}: IC={row['IC']:.4f} ICIR={row['ICIR']:.3f} "
            f"gross={row['gross_Sharpe']:.3f} TO={row['annu_TO_1way']:.1f} "
            f"net={row['net_Sharpe']:.3f} feasible={row['feasible']}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "c6_vs_c2_060_head_to_head.csv", index=False)

    c2 = df[df["label"] == "C2_D1_0.60"].iloc[0]
    c6 = df[df["label"] == "C6_D1_0.60_state_mod"].iloc[0]

    upgrade = bool(
        c6["feasible"]
        and c6["ICIR"] >= c2["ICIR"]
        and c6["annu_TO_1way"] <= 120
        and c6["net_Sharpe"] > 0
    )
    winner = "C6_D1_0.60_state_mod" if upgrade else "C2_D1_0.60"
    reason = (
        "C6 ICIR >= C2 and passes TO/net fences → upgrade research recommendation to C6"
        if upgrade
        else (
            f"Keep C2_D1_0.60: C6 ICIR={c6['ICIR']:.3f} vs C2={c2['ICIR']:.3f}; "
            f"state_modifier does not improve information efficiency at w1=0.60"
        )
    )
    log(f"\nDecision: {winner}")
    log(f"Reason: {reason}")

    decision = {
        "task": "c6_vs_c2_060_head_to_head",
        "period": "confirmation_951d",
        "c2": c2.to_dict(),
        "c6": c6.to_dict(),
        "upgrade_to_c6": upgrade,
        "research_recommended": winner,
        "reason": reason,
        "production_alternatives": {
            "C2_D1_0.70": "lower turnover compromise (research note only)",
            "C2_D1_0.80": "lowest-turnover compromise (research note only)",
        },
    }
    (OUT / "c6_vs_c2_060_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    # Patch combination / library / pool if needed (keep C2 unless upgrade)
    if COMB_JSON.exists():
        comb = json.loads(COMB_JSON.read_text())
        comb["c6_vs_c2_060"] = decision
        if upgrade:
            comb["recommended"] = {
                "label": "C6_D1_0.60_state_mod",
                "weights": {"D1": 0.6, "D4": 0.2, "D5": 0.2},
                "state_modifier": {
                    "trigger": "cancel mean-z bottom 20%",
                    "effect": "D4 weight 0.2→0.1; redistribute 0.05 to D1 and D5",
                },
                "IC": float(c6["IC"]),
                "ICIR": float(c6["ICIR"]),
                "gross_Sharpe": float(c6["gross_Sharpe"]),
                "annu_one_way_turnover": float(c6["annu_TO_1way"]),
                "net_Sharpe": float(c6["net_Sharpe"]),
                "note": reason,
            }
        else:
            # ensure recommended stays C2 with explicit C6 rejection note
            rec = comb.get("recommended", {})
            rec["c6_decision"] = "not_upgraded"
            rec["c6_note"] = reason
            comb["recommended"] = rec
        COMB_JSON.write_text(json.dumps(comb, indent=2, ensure_ascii=False, default=str) + "\n")

    for path, key in [
        (LIBRARY_PATH, "combination_layer_v1"),
        (POOL_PATH, "combination_v1_recommended"),
    ]:
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        if key == "combination_layer_v1":
            block = obj.get(key, {})
            block["c6_vs_c2_060"] = {
                "upgrade_to_c6": upgrade,
                "research_recommended": winner,
                "reason": reason,
                "c2_ICIR": float(c2["ICIR"]),
                "c6_ICIR": float(c6["ICIR"]),
            }
            block["production_alternatives_note"] = (
                "C2_D1_0.70 / 0.80 are lower-turnover production compromises; "
                "research recommendation remains ICIR-optimal C2_D1_0.60 unless C6 upgrades."
            )
            obj[key] = block
        else:
            block = obj.get(key, {})
            block["c6_decision"] = "upgraded" if upgrade else "not_upgraded"
            block["c6_note"] = reason
            obj[key] = block
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")

    log(f"Published -> {OUT / 'c6_vs_c2_060_decision.json'}")


if __name__ == "__main__":
    main()
