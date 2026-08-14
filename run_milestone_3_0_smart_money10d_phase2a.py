#!/usr/bin/env python
"""III-A4.1 Phase 2A — SmartMoney10d CSI1000 scout (2023–2025).

No Registry · No Composite · No formula change · No sign flip on IC.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_phase2a.py
  OMP_NUM_THREADS=1 python run_milestone_3_0_smart_money10d_phase2a.py --prefetch-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, decile_group_means, icir_from_daily
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from core.l2_features.smart_money_panel_builder import (
    FACTOR_PANEL_DIR,
    FORMULA_VERSION,
    coverage_report,
    ensure_minute_feature_months,
    load_minute_feature,
)
from core.l2_features.smart_money_q import compute_daily_smart_money_q_fast
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import cs_zscore
from factor_data_loaders import connect_ddb, load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from factor_runner import get_universe_mask
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache

REPO = Path(__file__).resolve().parent
OUT = REPO / "research/reports/smart_money_v1/phase2a"
SIGNAL_SHIFT = 1
TOP_FRAC = 0.10
COST_RT = 0.0015  # 15bp round-trip
SCOUT_START = dt.datetime(2023, 1, 1)
SCOUT_END = dt.datetime(2025, 12, 31)


def log(msg: str) -> None:
    print(msg, flush=True)


def _month_starts(start: dt.datetime, end: dt.datetime) -> List[Tuple[dt.datetime, dt.datetime]]:
    chunks = []
    cur = dt.datetime(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = dt.datetime(cur.year + 1, 1, 1)
        else:
            nxt = dt.datetime(cur.year, cur.month + 1, 1)
        c0 = max(cur, start)
        c1 = min(nxt - dt.timedelta(days=1), end)
        if c0 <= c1:
            chunks.append((c0, c1))
        cur = nxt
    return chunks


def prefetch_minute_features(start: dt.datetime, end: dt.datetime, refresh: bool = False) -> None:
    """Build L2 smart_score month caches (DDB). Preheat one month before start."""
    feat_start = dt.datetime(start.year, start.month, 1) - dt.timedelta(days=1)
    feat_start = dt.datetime(feat_start.year, feat_start.month, 1)
    log(f"Prefetch L2 minute_feature months {feat_start.date()} → {end.date()} (no full concat)")
    ensure_minute_feature_months(feat_start, end, refresh_cache=refresh)
    log("Prefetch done.")


def csi1000_symbol_set(session, start: dt.datetime, end: dt.datetime) -> List[str]:
    mask = get_universe_mask(session, start, end, cfg.UNIVERSE_LIST["CSI1000"])
    # any membership over window
    cols = [c for c in mask.columns if str(c)[0] in ("6", "0", "3")]
    present = mask[cols].notna().any(axis=0)
    syms = sorted(present[present].index.astype(str).tolist())
    log(f"CSI1000 union symbols: {len(syms)}")
    return syms


def build_q_panel_csi1000_monthly(
    start: dt.datetime,
    end: dt.datetime,
    symbols: List[str],
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Month-chunk Q build to avoid loading all minute years into RAM."""
    FACTOR_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"CSI1000_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    wide_path = FACTOR_PANEL_DIR / f"SmartMoney10d_{tag}.parquet"
    long_path = FACTOR_PANEL_DIR / f"SmartMoney10d_long_{tag}.parquet"
    if wide_path.exists() and long_path.exists() and not refresh:
        log(f"Load cached panel {wide_path}")
        return pd.read_parquet(wide_path)

    sym_set = set(symbols)
    month_long_dir = FACTOR_PANEL_DIR / f"monthly_long_{tag}"
    month_long_dir.mkdir(parents=True, exist_ok=True)

    parts: List[pd.DataFrame] = []
    for c0, c1 in _month_starts(start, end):
        mtag = c0.strftime("%Y%m")
        mpath = month_long_dir / f"q_{mtag}.parquet"
        if mpath.exists() and not refresh:
            parts.append(pd.read_parquet(mpath))
            log(f"  month {mtag}: cache hit rows={len(parts[-1]):,}")
            continue

        # need previous calendar month for 10d lookback
        if c0.month == 1:
            pre0 = dt.datetime(c0.year - 1, 12, 1)
        else:
            pre0 = dt.datetime(c0.year, c0.month - 1, 1)
        log(f"  month {mtag}: load features {pre0.date()}→{c1.date()} ...")
        feat = load_minute_feature(pre0, c1, use_cache=True, refresh_cache=False)
        feat = feat[feat["symbol"].isin(sym_set)]
        log(f"  month {mtag}: feat rows={len(feat):,} → Q ...")
        target_dates = sorted(feat.loc[feat["date"].between(c0, c1), "date"].unique())
        long_m = compute_daily_smart_money_q_fast(
            feat,
            dates=target_dates,
            progress_every=300,
        )
        long_m = long_m[(long_m["date"] >= pd.Timestamp(c0)) & (long_m["date"] <= pd.Timestamp(c1))]
        long_m.to_parquet(mpath, index=False)
        parts.append(long_m)
        log(f"  month {mtag}: Q rows={len(long_m):,}")

    long = pd.concat(parts, ignore_index=True)
    long = long.drop_duplicates(["date", "symbol"], keep="last")
    wide = long.pivot(index="date", columns="symbol", values="Q").sort_index()
    wide = wide.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    wide.to_parquet(wide_path)
    long.to_parquet(long_path, index=False)
    meta = {
        "formula_version": FORMULA_VERSION,
        "universe": "CSI1000",
        "start": str(start.date()),
        "end": str(end.date()),
        "n_days": int(wide.shape[0]),
        "n_symbols": int(wide.shape[1]),
        "coverage": coverage_report(wide),
    }
    (FACTOR_PANEL_DIR / f"SmartMoney10d_{tag}_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    log(f"Panel saved {wide.shape} → {wide_path}")
    return wide


def apply_mask(panel: pd.DataFrame, mask: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mask is None:
        return panel
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m.notna())


def eval_ic_block(signal: pd.DataFrame, ret: pd.DataFrame, label: str) -> dict:
    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    ic = ic.dropna()
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    # signed ICIR (negative expected for SmartMoney)
    icir = float(icir_from_daily(ic)) if len(ic) >= 20 else float("nan")
    pos_frac = float((ic > 0).mean()) if len(ic) else float("nan")
    return {
        "mode": label,
        "rank_ic": mean_ic,
        "abs_rank_ic": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
        "icir": icir,
        "abs_icir": abs(icir) if np.isfinite(icir) else np.nan,
        "ic_pos_frac": pos_frac,
        "n_days": int(len(ic)),
    }


def eval_portfolio_low_minus_high(q: pd.DataFrame, ret: pd.DataFrame) -> dict:
    """H-L = long low-Q − short high-Q (paper economic direction). IC stays on raw Q."""
    # implement via inverted signal for weight builder only
    sig_for_w = -q
    gross, to = daily_hl_pnl_and_turnover(
        sig_for_w, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    net = net_pnl_series(gross, to, COST_RT)
    gperf = series_performance(gross.dropna())
    nperf = series_performance(net.dropna())
    return {
        "hl_construction": "long_lowQ_short_highQ",
        "gross_sharpe": gperf["sharpe"],
        "net_sharpe": nperf["sharpe"],
        "gross_annu_ret": gperf["annu_ret"],
        "net_annu_ret": nperf["annu_ret"],
        "daily_turnover": float(to.mean()),
        "cost_rt": COST_RT,
    }


def yearly_table(q: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(set(q.index.year)):
        qi = q.loc[str(year)]
        ri = ret.reindex(index=qi.index, columns=qi.columns)
        icb = eval_ic_block(qi, ri, f"raw_{year}")
        port = eval_portfolio_low_minus_high(qi, ri)
        rows.append(
            {
                "year": year,
                "rank_ic": icb["rank_ic"],
                "abs_icir": icb["abs_icir"],
                "ic_pos_frac": icb["ic_pos_frac"],
                "hl_net_sharpe": port["net_sharpe"],
                "hl_gross_sharpe": port["gross_sharpe"],
                "daily_turnover": port["daily_turnover"],
                "n_days": icb["n_days"],
                "year_works": bool(
                    np.isfinite(port["net_sharpe"]) and port["net_sharpe"] > 0 and icb["rank_ic"] < 0
                ),
            }
        )
    return pd.DataFrame(rows)


def soft_bars(ic_raw: dict, port: dict, yearly: pd.DataFrame) -> dict:
    n_years = len(yearly)
    n_work = int(yearly["year_works"].sum()) if n_years else 0
    gates = {
        "abs_rank_ic_gt_0p02": bool(ic_raw.get("abs_rank_ic", 0) > 0.02),
        "abs_icir_gt_1p5": bool(ic_raw.get("abs_icir", 0) > 1.5),
        "years_work_ge_2_of_3": n_work >= min(2, n_years),
        "hl_net_sharpe_gt_1": bool(port.get("net_sharpe", 0) > 1.0),
        "turnover_finite": bool(np.isfinite(port.get("daily_turnover", np.nan))),
    }
    return {
        "gates": gates,
        "n_years_work": n_work,
        "n_years": n_years,
        "phase2a_pass": all(gates.values()),
        "phase2a_soft_pass": (
            gates["abs_rank_ic_gt_0p02"]
            and gates["years_work_ge_2_of_3"]
            and gates["turnover_finite"]
            and (gates["abs_icir_gt_1p5"] or gates["hl_net_sharpe_gt_1"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--refresh-feature", action="store_true")
    parser.add_argument("--refresh-panel", action="store_true")
    parser.add_argument("--skip-peers", action="store_true", help="skip TGD/D1/Flow IC corr")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log("=== III-A4.1 Phase 2A SmartMoney10d CSI1000 Scout ===")
    log(f"{SCOUT_START.date()} → {SCOUT_END.date()} | raw Q | cost={COST_RT} | no Registry")

    prefetch_minute_features(SCOUT_START, SCOUT_END, refresh=args.refresh_feature)
    if args.prefetch_only:
        log("Prefetch-only done.")
        return

    session = connect_ddb()
    try:
        session.run(intraday_lib.ddb_functions)
        symbols = csi1000_symbol_set(session, SCOUT_START, SCOUT_END)
        mask = get_universe_mask(session, SCOUT_START, SCOUT_END, cfg.UNIVERSE_LIST["CSI1000"])
    finally:
        session.close()

    q = build_q_panel_csi1000_monthly(
        SCOUT_START, SCOUT_END, symbols, refresh=args.refresh_panel
    )
    q = apply_mask(q, mask)
    q = q.dropna(how="all", axis=1).dropna(how="all", axis=0)

    ret = Factor_Dev_Lib.get_Ret_Matrix(SCOUT_START, SCOUT_END + dt.timedelta(days=5), method="c2c")
    ret = ret.reindex(index=q.index, columns=q.columns)

    # --- IC raw ---
    ic_raw = eval_ic_block(q, ret, "raw")
    log(f"Raw RankIC={ic_raw['rank_ic']:.4f} |ICIR|={ic_raw['abs_icir']:.2f} pos_frac={ic_raw['ic_pos_frac']:.2f}")

    # --- size+industry ---
    log("Load EOD for neutralization ...")
    preheat = SCOUT_START - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, SCOUT_END)
    try:
        session.run(intraday_lib.ddb_functions)
    except Exception:
        pass
    industry = load_citics_industry_panel(SCOUT_START, SCOUT_END)
    float_mkt = enriched.float_mktcap.reindex_like(q)
    ind = industry.reindex_like(q)
    q_si = cs_zscore(neutralize_size_industry(q, ind, float_mkt))
    ic_si = eval_ic_block(q_si, ret, "size_industry")
    log(f"SI RankIC={ic_si['rank_ic']:.4f} |ICIR|={ic_si['abs_icir']:.2f}")

    ic_df = pd.DataFrame([ic_raw, ic_si])
    ic_df.to_csv(OUT / "ic_summary.csv", index=False)

    # --- Decile on raw Q (D1=low Q) ---
    dec = decile_group_means(q, ret, n_groups=10, signal_shift=SIGNAL_SHIFT)
    # decile_group_means typically ranks low=1? check - usually ascending rank so group 1 = low signal
    dec.to_csv(OUT / "decile_return.csv", header=["mean_ret"])
    # H-L bottom-top for decile series if index is 1..10
    if len(dec.dropna()) >= 2:
        hl_dec = float(dec.iloc[0] - dec.iloc[-1])  # low Q − high Q if ascending
    else:
        hl_dec = float("nan")

    # --- Portfolio ---
    port = eval_portfolio_low_minus_high(q, ret)
    port_si = eval_portfolio_low_minus_high(q_si, ret)
    pd.DataFrame(
        [
            {"mode": "raw_Q_low_minus_high", **port},
            {"mode": "si_z_low_minus_high", **{k: v for k, v in port_si.items() if k != "hl_construction"}, "hl_construction": port_si["hl_construction"]},
        ]
    ).to_csv(OUT / "portfolio_summary.csv", index=False)
    log(f"H-L net Sharpe (low-high)={port['net_sharpe']:.2f} TO={port['daily_turnover']:.3f}")

    # --- Yearly ---
    yearly = yearly_table(q, ret)
    yearly.to_csv(OUT / "yearly_stability.csv", index=False)
    log("Yearly:\n" + yearly.to_string(index=False))

    # --- Peer IC correlation (not residual) ---
    peer_corr = {}
    if not args.skip_peers:
        log("Peer IC correlation vs TGD/D1/Flow ...")
        l2 = build_l2_daily_cache(preheat, SCOUT_END, session=session, close=enriched.close)
        pv = build_factor_cache(
            df_close=enriched.close,
            df_open=enriched.open,
            df_high=enriched.high,
            df_low=enriched.low,
            df_volume=enriched.volume,
            df_amount=enriched.amount,
            df_turnover=enriched.turnover,
        )
        tgd, _ = build_tgd20_wide_from_eod_l2(
            SCOUT_START,
            SCOUT_END,
            open_=enriched.open,
            close=enriched.close,
            use_cache=True,
            window=20,
        )
        peers_raw = {
            "SmartMoney10d": apply_mask(q, mask),
            "TGD20": apply_mask(tgd.reindex_like(q), mask),
            "D1": apply_mask(
                build_eod_engine_factor("low_vol_liquidity_quality_60d", pv).reindex_like(q), mask
            ),
            "FlowDensity20": apply_mask(
                build_net_active_flow_mktcap(l2, enriched.float_mktcap, window=20).reindex_like(q),
                mask,
            ),
        }
        # IC series on raw (SM) / peers as-is for scout
        ic_series = {
            k: daily_rank_ic_series(v, ret.reindex_like(v), signal_shift=SIGNAL_SHIFT)
            for k, v in peers_raw.items()
        }
        ic_mat = pd.DataFrame(ic_series)
        peer_corr = ic_mat.corr()
        peer_corr.to_csv(OUT / "peer_ic_series_corr.csv")
        log("Peer IC-series corr:\n" + peer_corr.round(3).to_string())

    session.close()

    bars = soft_bars(ic_raw, port, yearly)
    report = {
        "phase": "2A_csi1000_scout",
        "formula_version": FORMULA_VERSION,
        "universe": "CSI1000",
        "period": f"{SCOUT_START.date()}_{SCOUT_END.date()}",
        "signal": "raw_Q",
        "sign_flip_on_ic": False,
        "hl_book": "long_lowQ_short_highQ",
        "cost_rt": COST_RT,
        "coverage": coverage_report(q),
        "ic_raw": ic_raw,
        "ic_size_industry": ic_si,
        "portfolio_raw": port,
        "portfolio_si": port_si,
        "decile_low_minus_high": hl_dec,
        "yearly": yearly.to_dict(orient="records"),
        "peer_ic_corr_vs_SmartMoney": (
            {k: float(peer_corr.loc["SmartMoney10d", k]) for k in peer_corr.columns if k != "SmartMoney10d"}
            if len(peer_corr)
            else {}
        ),
        "soft_bars": bars,
        "verdict": "PASS_scout" if bars["phase2a_pass"] else (
            "SOFT_PASS_scout" if bars["phase2a_soft_pass"] else "FAIL_scout"
        ),
        "next": (
            "Phase2B/2C pack candidate"
            if bars["phase2a_soft_pass"]
            else "Diagnose / do not pack yet"
        ),
        "forbidden": ["Registry", "Composite", "formula_change", "Active_*"],
    }
    (OUT / "phase2a_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    log(f"Wrote {OUT / 'phase2a_report.json'}")
    log(f"VERDICT: {report['verdict']} gates={bars['gates']}")


if __name__ == "__main__":
    main()
