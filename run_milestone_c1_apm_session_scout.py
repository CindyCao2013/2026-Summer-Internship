#!/usr/bin/env python
"""C1.3 — APM_SessionResidual CSI1000 Scout (2021–2025).

Alpha existence gate — NOT Pack / library / Registry.

Signal: apm_cs (paper object). No sign flip on IC.
Eval: signal.shift(1) → ret_t (cache unshifted).

Usage:
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_scout.py --prefetch-only
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_scout.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
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
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from core.l2_features.apm_session_panel_builder import (
    CACHE_ROOT,
    FORMULA_VERSION,
    build_apm_session_panel,
    formula_meta,
)
from core.l2_features.apm_session_signal import (
    build_apm_stat_panel,
    build_ret20_long,
    cs_residualize_vs_ret20,
)
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import cs_zscore
from factor_data_loaders import connect_ddb, load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from factor_runner import get_universe_mask
from industry_neutral import load_citics_industry_panel, panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual
from l2_data_loaders import build_l2_daily_cache

REPO = Path(__file__).resolve().parent
OUT = REPO / "research/reports/apm_session_v1/scout"
SIGNAL_CACHE = CACHE_ROOT / "signal"
SIGNAL_SHIFT = 1
TOP_FRAC = 0.10
COST_RT = 0.0015
SCOUT_START = dt.datetime(2021, 1, 1)
SCOUT_END = dt.datetime(2025, 12, 31)
PREHEAT_START = dt.datetime(2020, 12, 1)
PAPER_DIRECTION = "positive"


def log(msg: str, fh=None) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if fh is not None:
        fh.write(line)
        fh.flush()


def apply_mask(panel: pd.DataFrame, mask: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mask is None:
        return panel
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m.notna())


def neutralize_size_only(raw: pd.DataFrame, float_mktcap: pd.DataFrame) -> pd.DataFrame:
    log_size = np.log(float_mktcap.replace(0, np.nan))
    log_size = log_size.reindex(index=raw.index, columns=raw.columns)
    return panel_cross_sectional_residual(raw, [log_size])


def eval_ic_block(signal: pd.DataFrame, ret: pd.DataFrame, label: str) -> dict:
    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT).dropna()
    mean_ic = float(ic.mean()) if len(ic) else float("nan")
    icir = float(icir_from_daily(ic)) if len(ic) >= 20 else float("nan")
    return {
        "mode": label,
        "rank_ic": mean_ic,
        "abs_rank_ic": abs(mean_ic) if np.isfinite(mean_ic) else np.nan,
        "icir": icir,
        "abs_icir": abs(icir) if np.isfinite(icir) else np.nan,
        "ic_pos_frac": float((ic > 0).mean()) if len(ic) else np.nan,
        "n_days": int(len(ic)),
        "ic_series": ic,
    }


def hl_book_for_ic(rank_ic: float) -> Tuple[str, float]:
    """PnL book follows empirical IC; IC metrics stay on raw signal."""
    if np.isfinite(rank_ic) and rank_ic < 0:
        return "long_low_short_high", -1.0
    return "long_high_short_low", 1.0


def eval_portfolio(signal: pd.DataFrame, ret: pd.DataFrame, rank_ic: float) -> dict:
    book, sign = hl_book_for_ic(rank_ic)
    sig_w = signal * sign
    gross, to = daily_hl_pnl_and_turnover(
        sig_w, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    net = net_pnl_series(gross, to, COST_RT)
    gperf = series_performance(gross.dropna())
    nperf = series_performance(net.dropna())
    return {
        "hl_construction": book,
        "gross_sharpe": gperf["sharpe"],
        "net_sharpe": nperf["sharpe"],
        "gross_annu_ret": gperf["annu_ret"],
        "net_annu_ret": nperf["annu_ret"],
        "daily_turnover": float(to.mean()) if len(to) else np.nan,
        "cost_rt": COST_RT,
    }


def monotonicity_score(decile: pd.Series, prefer_increasing: bool) -> dict:
    vals = decile.dropna().astype(float)
    if len(vals) < 2:
        return {"mono_frac": np.nan, "prefer_increasing": prefer_increasing, "n_adj": 0}
    diffs = vals.diff().iloc[1:]
    if prefer_increasing:
        ok = (diffs > 0).sum()
    else:
        ok = (diffs < 0).sum()
    n = int(len(diffs))
    return {
        "mono_frac": float(ok / n) if n else np.nan,
        "prefer_increasing": prefer_increasing,
        "n_adj": n,
        "d1_minus_d10": float(vals.iloc[0] - vals.iloc[-1]),
        "d10_minus_d1": float(vals.iloc[-1] - vals.iloc[0]),
    }


def mean_daily_cs_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("nan")
    aa = a.reindex(index=common)
    bb = b.reindex(index=common, columns=aa.columns)
    s = aa.corrwith(bb, axis=1, method="spearman").dropna()
    return float(s.mean()) if len(s) else float("nan")


def build_apm_cs_wide(
    start: dt.datetime,
    end: dt.datetime,
    *,
    refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (apm_cs wide, apm_stat wide) for [start, end], building caches as needed."""
    SIGNAL_CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"CSI1000scout_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    cs_wide_path = SIGNAL_CACHE / f"apm_cs_wide_{tag}.parquet"
    st_wide_path = SIGNAL_CACHE / f"apm_stat_wide_{tag}.parquet"
    if cs_wide_path.exists() and st_wide_path.exists() and not refresh:
        log(f"Load cached wide panels {cs_wide_path.name}")
        cs = pd.read_parquet(cs_wide_path)
        st = pd.read_parquet(st_wide_path)
        cs.index = pd.to_datetime(cs.index)
        st.index = pd.to_datetime(st.index)
        return cs, st

    log(f"Build residual panel {PREHEAT_START.date()} → {end.date()} ...")
    residual, _ = build_apm_session_panel(
        PREHEAT_START, end, use_cache=True, refresh_cache=refresh
    )
    log(f"  residual rows={len(residual):,}")
    log("Build APM_stat + Ret20 CS residual ...")
    apm = build_apm_stat_panel(residual, window=20, min_periods=10)
    ret = Factor_Dev_Lib.get_Ret_Matrix(
        PREHEAT_START - dt.timedelta(days=40), end + dt.timedelta(days=5), method="c2c"
    )
    ret20 = build_ret20_long(ret, window=20)
    apm_cs, _align = cs_residualize_vs_ret20(apm, ret20, signal_col="apm_stat")

    apm_cs = apm_cs[
        (apm_cs["date"] >= pd.Timestamp(start)) & (apm_cs["date"] <= pd.Timestamp(end))
    ]
    apm = apm[(apm["date"] >= pd.Timestamp(start)) & (apm["date"] <= pd.Timestamp(end))]

    cs_wide = apm_cs.pivot(index="date", columns="symbol", values="apm_cs").sort_index()
    st_wide = apm.pivot(index="date", columns="symbol", values="apm_stat").sort_index()
    cs_wide.to_parquet(cs_wide_path)
    st_wide.to_parquet(st_wide_path)
    apm_cs.to_parquet(SIGNAL_CACHE / f"apm_cs_long_{tag}.parquet", index=False)
    log(f"  wide apm_cs shape={cs_wide.shape}")
    return cs_wide, st_wide


def main() -> None:
    parser = argparse.ArgumentParser(description="C1.3 APM CSI1000 scout")
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-peers", action="store_true")
    args = parser.parse_args()

    for sub in (
        OUT,
        OUT / "ic",
        OUT / "neutralization",
        OUT / "quantile",
        OUT / "stability",
        OUT / "similarity",
        OUT / "execution",
    ):
        sub.mkdir(parents=True, exist_ok=True)

    fh = (OUT / "build_log.txt").open("w", encoding="utf-8")
    try:
        log("=== C1.3 APM_SessionResidual CSI1000 SCOUT ===", fh)
        log(f"formula={FORMULA_VERSION} signal=apm_cs identity=adapted_replication", fh)
        log(f"{SCOUT_START.date()} → {SCOUT_END.date()} | no sign flip | cost={COST_RT}", fh)
        log("Forbidden: Pack · factor_library · Registry · Composite · Proxy rename", fh)

        if args.prefetch_only:
            build_apm_session_panel(
                PREHEAT_START, SCOUT_END, use_cache=True, refresh_cache=args.refresh
            )
            log("Prefetch-only done.", fh)
            return

        # --- Universe ---
        session = connect_ddb()
        try:
            session.run(intraday_lib.ddb_functions)
            mask = get_universe_mask(
                session, SCOUT_START, SCOUT_END, cfg.UNIVERSE_LIST["CSI1000"]
            )
        finally:
            session.close()

        # --- Signal ---
        apm_cs, _apm_stat = build_apm_cs_wide(
            SCOUT_START, SCOUT_END, refresh=args.refresh
        )
        apm_cs = apply_mask(apm_cs, mask).dropna(how="all", axis=1).dropna(how="all", axis=0)
        log(f"Masked apm_cs shape={apm_cs.shape}", fh)

        ret = Factor_Dev_Lib.get_Ret_Matrix(
            SCOUT_START, SCOUT_END + dt.timedelta(days=5), method="c2c"
        )
        ret = ret.reindex(index=apm_cs.index, columns=apm_cs.columns)

        # --- A. Raw IC ---
        ic_raw = eval_ic_block(apm_cs, ret, "raw")
        direction = (
            "positive_match"
            if np.isfinite(ic_raw["rank_ic"]) and ic_raw["rank_ic"] > 0
            else "negative_mismatch"
        )
        log(
            f"Raw RankIC={ic_raw['rank_ic']:.4f} ICIR={ic_raw['icir']:.2f} "
            f"direction={direction}",
            fh,
        )

        # --- B. Neutralization ---
        log("Neutralization panels ...", fh)
        preheat = SCOUT_START - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
        enriched, sess = load_eod_enriched_tables(preheat, SCOUT_END)
        try:
            try:
                sess.run(intraday_lib.ddb_functions)
            except Exception:
                pass
            industry = load_citics_industry_panel(SCOUT_START, SCOUT_END)
            float_mkt = enriched.float_mktcap.reindex_like(apm_cs)
            ind = industry.reindex_like(apm_cs)

            sig_size = cs_zscore(neutralize_size_only(apm_cs, float_mkt))
            sig_ind = cs_zscore(panel_industry_demean(apm_cs, ind))
            sig_si = cs_zscore(neutralize_size_industry(apm_cs, ind, float_mkt))

            ic_size = eval_ic_block(sig_size, ret, "size_neutral")
            ic_ind = eval_ic_block(sig_ind, ret, "industry_neutral")
            ic_si = eval_ic_block(sig_si, ret, "size_industry_neutral")
        finally:
            sess.close()

        neut_df = pd.DataFrame(
            [
                {k: v for k, v in ic_raw.items() if k != "ic_series"},
                {k: v for k, v in ic_size.items() if k != "ic_series"},
                {k: v for k, v in ic_ind.items() if k != "ic_series"},
                {k: v for k, v in ic_si.items() if k != "ic_series"},
            ]
        )
        neut_df.to_csv(OUT / "neutralization" / "neutral_ic.csv", index=False)
        neut_df.to_csv(OUT / "ic" / "factor_summary.csv", index=False)
        log("Neutral IC:\n" + neut_df.to_string(index=False), fh)

        # IC curve
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ic_raw["ic_series"].cumsum().plot(ax=ax, label="raw apm_cs cum RankIC")
            ax.axhline(0, color="k", lw=0.8)
            ax.legend()
            ax.set_title("APM_SessionResidual CSI1000 — cumulative RankIC (raw apm_cs)")
            fig.tight_layout()
            fig.savefig(OUT / "ic" / "ic_curve.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            log(f"ic_curve.png skipped: {exc}", fh)

        # --- C. Decile ---
        dec = decile_group_means(apm_cs, ret, n_groups=10, signal_shift=SIGNAL_SHIFT)
        dec.to_csv(OUT / "quantile" / "decile_return.csv", header=["mean_ret"])
        prefer_up = bool(np.isfinite(ic_raw["rank_ic"]) and ic_raw["rank_ic"] > 0)
        mono = monotonicity_score(dec, prefer_increasing=prefer_up)
        (OUT / "quantile" / "monotonicity.json").write_text(
            json.dumps(mono, indent=2) + "\n", encoding="utf-8"
        )
        log(f"Decile mono_frac={mono.get('mono_frac')} prefer_up={prefer_up}", fh)

        # --- D. Execution / H-L ---
        port = eval_portfolio(apm_cs, ret, ic_raw["rank_ic"])
        port_si = eval_portfolio(sig_si, ret, ic_si["rank_ic"])
        exec_df = pd.DataFrame(
            [
                {"mode": "raw_apm_cs", **port},
                {"mode": "si_z_apm_cs", **port_si},
            ]
        )
        exec_df.to_csv(OUT / "execution" / "turnover_summary.csv", index=False)
        log(
            f"H-L gross={port['gross_sharpe']:.2f} net={port['net_sharpe']:.2f} "
            f"TO={port['daily_turnover']:.3f} book={port['hl_construction']}",
            fh,
        )

        # --- E. Yearly ---
        yearly_rows = []
        for year in range(SCOUT_START.year, SCOUT_END.year + 1):
            yi = apm_cs.loc[str(year)]
            if yi.empty:
                continue
            ri = ret.reindex(index=yi.index, columns=yi.columns)
            icb = eval_ic_block(yi, ri, f"raw_{year}")
            pr = eval_portfolio(yi, ri, icb["rank_ic"])
            yearly_rows.append(
                {
                    "year": year,
                    "rank_ic": icb["rank_ic"],
                    "icir": icb["icir"],
                    "abs_icir": icb["abs_icir"],
                    "ic_pos_frac": icb["ic_pos_frac"],
                    "gross_sharpe": pr["gross_sharpe"],
                    "net_sharpe": pr["net_sharpe"],
                    "daily_turnover": pr["daily_turnover"],
                    "n_days": icb["n_days"],
                    "sign_matches_paper_positive": bool(
                        np.isfinite(icb["rank_ic"]) and icb["rank_ic"] > 0
                    ),
                    "gross_works": bool(
                        np.isfinite(pr["gross_sharpe"]) and pr["gross_sharpe"] > 0
                    ),
                }
            )
        yearly = pd.DataFrame(yearly_rows)
        yearly.to_csv(OUT / "stability" / "yearly_ic.csv", index=False)
        log("Yearly:\n" + yearly.to_string(index=False), fh)

        # --- F. Similarity ---
        peer_ic_corr = pd.DataFrame()
        peer_sig_corr = {}
        if not args.skip_peers:
            log("Peer IC + signal corr (Flow priority) ...", fh)
            enriched2, sess2 = load_eod_enriched_tables(preheat, SCOUT_END)
            try:
                try:
                    sess2.run(intraday_lib.ddb_functions)
                except Exception:
                    pass
                l2 = build_l2_daily_cache(
                    preheat, SCOUT_END, session=sess2, close=enriched2.close
                )
                pv = build_factor_cache(
                    df_close=enriched2.close,
                    df_open=enriched2.open,
                    df_high=enriched2.high,
                    df_low=enriched2.low,
                    df_volume=enriched2.volume,
                    df_amount=enriched2.amount,
                    df_turnover=enriched2.turnover,
                )
                tgd, _ = build_tgd20_wide_from_eod_l2(
                    SCOUT_START,
                    SCOUT_END,
                    open_=enriched2.open,
                    close=enriched2.close,
                    use_cache=True,
                    window=20,
                )
                peers = {
                    "APM_SessionResidual": apm_cs,
                    "TGD20": apply_mask(tgd.reindex_like(apm_cs), mask),
                    "D1": apply_mask(
                        build_eod_engine_factor(
                            "low_vol_liquidity_quality_60d", pv
                        ).reindex_like(apm_cs),
                        mask,
                    ),
                    "FlowDensity20": apply_mask(
                        build_net_active_flow_mktcap(
                            l2, enriched2.float_mktcap, window=20
                        ).reindex_like(apm_cs),
                        mask,
                    ),
                }
                sm_dir = REPO / "research/cache/smart_money/factor_panel"
                sm_alt = list(sm_dir.glob("SmartMoney10d_CSI1000_*.parquet")) if sm_dir.exists() else []
                if sm_alt:
                    sm = pd.read_parquet(sorted(sm_alt)[-1])
                    sm.index = pd.to_datetime(sm.index)
                    peers["SmartMoney10d"] = apply_mask(sm.reindex_like(apm_cs), mask)
                    log(f"  SmartMoney peer from {sorted(sm_alt)[-1].name}", fh)
                else:
                    log("  SmartMoney10d peer skipped (no CSI1000 panel cache)", fh)

                ic_series = {
                    k: daily_rank_ic_series(
                        v, ret.reindex_like(v), signal_shift=SIGNAL_SHIFT
                    )
                    for k, v in peers.items()
                }
                peer_ic_corr = pd.DataFrame(ic_series).corr()
                peer_ic_corr.to_csv(OUT / "similarity" / "factor_ic_corr.csv")
                log("IC-series corr:\n" + peer_ic_corr.round(3).to_string(), fh)

                for k, v in peers.items():
                    if k == "APM_SessionResidual":
                        continue
                    peer_sig_corr[k] = mean_daily_cs_corr(apm_cs, v)
                pd.Series(peer_sig_corr, name="mean_daily_cs_spearman").to_csv(
                    OUT / "similarity" / "factor_signal_corr.csv", header=True
                )
                log(f"Signal CS corr vs peers: {peer_sig_corr}", fh)
            finally:
                sess2.close()

        # --- Verdict ---
        n_years = len(yearly)
        n_sign_ok = int(yearly["sign_matches_paper_positive"].sum()) if n_years else 0
        n_gross_ok = int(yearly["gross_works"].sum()) if n_years else 0
        info_pass = bool(
            ic_raw.get("abs_rank_ic", 0) > 0.02 and ic_raw.get("abs_icir", 0) > 1.5
        )
        structure_pass = bool(
            (mono.get("mono_frac") or 0) >= 0.5 and n_gross_ok >= max(3, n_years - 2)
        )
        invest_pass = bool(np.isfinite(port.get("net_sharpe")) and port["net_sharpe"] > 1.0)
        flow_ic_corr = (
            float(peer_ic_corr.loc["APM_SessionResidual", "FlowDensity20"])
            if len(peer_ic_corr) and "FlowDensity20" in peer_ic_corr.columns
            else np.nan
        )
        flow_warn = bool(np.isfinite(flow_ic_corr) and abs(flow_ic_corr) > 0.7)

        if info_pass and structure_pass and invest_pass:
            verdict = "PASS_scout"
        elif info_pass:
            verdict = "PASS_research_FAIL_invest" if not invest_pass else "PASS_research"
        else:
            verdict = "FAIL_scout"

        summary = {
            "milestone": "C1.3",
            "factor_id": "APM_SessionResidual",
            "identity_class": "adapted_replication",
            "formula_version": FORMULA_VERSION,
            "universe": "CSI1000",
            "period": f"{SCOUT_START.date()}_{SCOUT_END.date()}",
            "signal": "apm_cs",
            "sign_flip_on_ic": False,
            "paper_direction": PAPER_DIRECTION,
            "direction": direction,
            "formula_meta": formula_meta(),
            "ic_raw": {k: v for k, v in ic_raw.items() if k != "ic_series"},
            "ic_size": {k: v for k, v in ic_size.items() if k != "ic_series"},
            "ic_industry": {k: v for k, v in ic_ind.items() if k != "ic_series"},
            "ic_size_industry": {k: v for k, v in ic_si.items() if k != "ic_series"},
            "monotonicity": mono,
            "portfolio_raw": port,
            "portfolio_si": port_si,
            "yearly": yearly.to_dict(orient="records"),
            "years_sign_match_paper": n_sign_ok,
            "years_gross_works": n_gross_ok,
            "peer_ic_corr_vs_APM": (
                {
                    k: float(peer_ic_corr.loc["APM_SessionResidual", k])
                    for k in peer_ic_corr.columns
                    if k != "APM_SessionResidual"
                }
                if len(peer_ic_corr)
                else {}
            ),
            "peer_signal_corr_vs_APM": peer_sig_corr,
            "flow_ic_corr": flow_ic_corr,
            "flow_overlap_warning": flow_warn,
            "gates": {
                "info_pass": info_pass,
                "structure_pass": structure_pass,
                "invest_pass": invest_pass,
            },
            "verdict": verdict,
            "forbidden": ["Pack_v1", "factor_library", "Registry", "Composite"],
            "next": (
                "C1 Pack v1"
                if verdict == "PASS_scout"
                else (
                    "document research alpha; optional execution grid; no fake Pack"
                    if verdict.startswith("PASS_research")
                    else "park / diagnose object; do not Pack"
                )
            ),
        }
        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        log(f"Wrote {OUT / 'summary.json'}", fh)
        log(f"VERDICT: {verdict} gates={summary['gates']} direction={direction}", fh)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
