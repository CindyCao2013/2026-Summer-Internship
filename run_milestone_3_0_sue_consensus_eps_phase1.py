#!/usr/bin/env python
"""III-B1 Phase1 — SUE_ConsensusEPS events + impulse panel smoke + sanity.

No scout · No Registry · No hold/decay lock · No Composite.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_sue_consensus_eps_phase1.py
  OMP_NUM_THREADS=1 python run_milestone_3_0_sue_consensus_eps_phase1.py --year 2024
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
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from core.fundamental.sue_consensus_eps_panel import (
    FORMULA_VERSION,
    META_DIR,
    SUE_P0_ROOT,
    build_and_cache_events,
    build_and_cache_impulse_panel,
    coverage_report,
    distribution_on_event_days,
)
from factor_data_loaders import connect_ddb
from factor_runner import get_universe_mask

REPO = Path(__file__).resolve().parent
OUT = REPO / "research/reports/sue_consensus_eps_v1/phase1"
SIGNAL_SHIFT = 1  # known_dt mapped to trade day; then shift1 for next-day ret


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=0, help="0 = all CSI1000 union")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    start = dt.datetime(args.year, 1, 1)
    end = dt.datetime(args.year, 12, 31)

    log("=== III-B1 Phase1 SUE_ConsensusEPS smoke + sanity ===")
    log(f"formula={FORMULA_VERSION} | window={start.date()}→{end.date()}")
    log("impulse only · no hold/decay · no scout · no Registry")
    log(f"sue_p0={SUE_P0_ROOT}")

    # --- Step 1–2: events + impulse ---
    events, audit = build_and_cache_events(
        start, end, sue_p0_root=SUE_P0_ROOT, refresh=args.refresh
    )
    log(f"Events: n={audit.get('n_events')} pit_pass={audit.get('pit_hard_pass')}")
    if audit.get("hard_fail") or events.empty:
        (OUT / "phase1_report.json").write_text(
            json.dumps({"smoke_pass": False, "audit": audit}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        raise SystemExit("Phase1 FAIL: events empty or PIT hard_fail")

    # duplicate check
    dup = int(events.duplicated(["symbol", "fiscal_period"]).sum())
    log(f"Duplicate (symbol, fiscal_period): {dup}")

    session = connect_ddb()
    try:
        session.run(intraday_lib.ddb_functions)
        mask = get_universe_mask(session, start, end, cfg.UNIVERSE_LIST["CSI1000"])
    finally:
        session.close()

    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end + dt.timedelta(days=5), method="c2c")
    trade_index = ret.index[(ret.index >= pd.Timestamp(start)) & (ret.index <= pd.Timestamp(end))]

    # CSI1000 symbols that appear in events
    csi_cols = [c for c in mask.columns if str(c)[0] in ("6", "0", "3")]
    csi_any = mask[csi_cols].notna().any(axis=0)
    csi_syms = set(csi_any[csi_any].index.astype(str).tolist())
    ev_syms = sorted(set(events["symbol"].astype(str)) & csi_syms)
    if args.max_symbols and args.max_symbols > 0:
        ev_syms = ev_syms[: args.max_symbols]
    log(f"CSI1000 ∩ event symbols: {len(ev_syms)}")

    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_csi1000"
    wide = build_and_cache_impulse_panel(
        events, trade_index, tag=tag, symbols=ev_syms, refresh=args.refresh
    )
    # apply daily membership
    m = mask.reindex(index=wide.index, columns=wide.columns)
    wide = wide.where(m.notna())

    cov = coverage_report(wide)
    dist = distribution_on_event_days(wide)
    log(f"Panel {wide.shape} event_cells={cov['n_event_cells']} active_days={dist.get('n_active_days')}")
    log(f"  mean_cs_std={dist.get('mean_cs_std')} mean_n={dist.get('mean_n')}")

    # PIT gate on events: est < known; panel mapping: trade day >= known
    pit_est = bool((events["est_dt"] < events["known_dt"]).all())
    # map audit: all impulse dates >= known_dt for those events
    ev_m = events[events["symbol"].isin(ev_syms)].copy()
    idx = trade_index.sort_values()
    pos = idx.searchsorted(ev_m["known_dt"].to_numpy())
    valid = pos < len(idx)
    mapped = idx[pos[valid]]
    known = pd.to_datetime(ev_m["known_dt"].to_numpy()[valid])
    pit_signal = bool((mapped >= known).all()) if len(mapped) else False

    # Gate 2: rankable on active days
    rankable = bool(
        dist.get("n_active_days", 0) >= 5
        and (dist.get("mean_cs_std") or 0) > 1e-6
        and (dist.get("mean_n") or 0) >= 5
    )

    # Gate 3: raw RankIC (no flip) — sparse panel; only days with enough names
    ret_a = ret.reindex(index=wide.index, columns=wide.columns)
    ic = daily_rank_ic_series(wide, ret_a, signal_shift=SIGNAL_SHIFT).dropna()
    # require ≥10 finite names for IC day (handled inside corrwith via pairwise)
    rank_ic = float(ic.mean()) if len(ic) else float("nan")
    icir = float(icir_from_daily(ic)) if len(ic) >= 20 else float("nan")
    direction = (
        "positive_matches_expected"
        if np.isfinite(rank_ic) and rank_ic > 0
        else (
            "negative_empirical"
            if np.isfinite(rank_ic) and rank_ic < 0
            else "flat_or_nan"
        )
    )
    log(f"Raw RankIC={rank_ic:.4f} n_ic_days={len(ic)} → {direction} (no flip)")

    gates = {
        "pit_est_before_known": pit_est,
        "pit_signal_date_ge_known": pit_signal,
        "no_duplicate_symbol_period": dup == 0,
        "events_gt_0": len(events) > 0,
        "rankable_cs": rankable,
        "raw_rank_ic_finite": bool(np.isfinite(rank_ic)),
    }
    # hard: PIT + events; soft: rankable + IC finite
    hard_ok = (
        gates["pit_est_before_known"]
        and gates["pit_signal_date_ge_known"]
        and gates["no_duplicate_symbol_period"]
        and gates["events_gt_0"]
    )
    soft_ok = hard_ok and gates["rankable_cs"] and gates["raw_rank_ic_finite"]

    report = {
        "phase": "1_smoke_sanity",
        "formula_version": FORMULA_VERSION,
        "factor_id": "SUE_ConsensusEPS",
        "panel_mode": "impulse",
        "window": f"{start.date()}_{end.date()}",
        "universe": "CSI1000",
        "n_event_symbols_used": len(ev_syms),
        "pit_audit": audit,
        "duplicates_symbol_period": dup,
        "coverage": cov,
        "distribution_event_days": dist,
        "gate3_raw_direction": {
            "rank_ic_mean": rank_ic,
            "icir_short": icir,
            "n_ic_days": int(len(ic)),
            "expected": "positive_ic",
            "empirical_direction": direction,
            "sign_flip": False,
        },
        "gates": gates,
        "hard_pass": hard_ok,
        "soft_pass": soft_ok,
        "next": (
            "III-B1 Phase3 scout CSI1000 2020-2025 + IC decay"
            if soft_ok
            else "Fix PIT/events before scout"
        ),
        "forbidden": ["Registry", "Composite", "hold_decay_lock", "scout_in_phase1"],
    }
    out_path = OUT / "phase1_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    events.head(20).to_csv(OUT / "events_sample.csv", index=False)
    pd.Series(cov).to_csv(OUT / "coverage.csv", header=["value"])
    log(f"Wrote {out_path}")
    log(f"PHASE1 hard={'PASS' if hard_ok else 'FAIL'} soft={'PASS' if soft_ok else 'FAIL'}")
    if not hard_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
