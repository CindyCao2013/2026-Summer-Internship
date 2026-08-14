#!/usr/bin/env python
"""C1.4 — APM_SessionResidual horizon + execution diagnosis.

Uses frozen CSI1000 scout apm_cs panel. No formula change · No Pack · No Registry.

Usage:
  OMP_NUM_THREADS=1 python run_milestone_c1_apm_session_execution.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import icir_from_daily
from alpha_investability import (
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from core.l2_features.apm_session_panel_builder import FORMULA_VERSION
from execution_layer import evaluate_execution
from factor_data_loaders import connect_ddb
from factor_runner import get_universe_mask

REPO = Path(__file__).resolve().parent
PANEL = (
    REPO
    / "research/cache/apm_session/signal"
    / "apm_cs_wide_CSI1000scout_20210101_20251231.parquet"
)
OUT = REPO / "research/reports/apm_session_v1/execution"
SCOUT_START = dt.datetime(2021, 1, 1)
SCOUT_END = dt.datetime(2025, 12, 31)
COST_RT = 0.0015
TOP_FRAC = 0.10
DECAY_HS = [1, 3, 5, 10, 20]
HOLD_HS = [1, 3, 5, 10, 20]


def log(msg: str, fh=None) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if fh is not None:
        fh.write(line)
        fh.flush()


def apply_mask(panel: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m.notna())


def forward_cumret(ret: pd.DataFrame, horizon: int) -> pd.DataFrame:
    parts = [ret.shift(-k) for k in range(1, horizon + 1)]
    out = parts[0].copy()
    for p in parts[1:]:
        out = out.add(p, fill_value=np.nan)
    valid = parts[0].notna()
    for p in parts[1:]:
        valid = valid & p.notna()
    return out.where(valid)


def hl_at_horizon(sig: pd.DataFrame, ret_u: pd.DataFrame, horizon: int) -> dict:
    """Long-high book proxies (positive IC orientation)."""
    fwd = forward_cumret(ret_u, horizon).reindex(index=sig.index, columns=sig.columns)
    gross, to = daily_hl_pnl_and_turnover(
        sig, fwd, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=0
    )
    g_proxy = gross / float(horizon)
    to_proxy = to / float(horizon)
    net_proxy = net_pnl_series(g_proxy, to_proxy, COST_RT)
    gp = series_performance(g_proxy.dropna())
    np_ = series_performance(net_proxy.dropna())
    g_nl = gross.iloc[::horizon]
    to_nl = to.iloc[::horizon]
    net_nl = net_pnl_series(g_nl, to_nl, COST_RT).dropna()
    if len(net_nl) >= 5 and net_nl.std() > 0:
        periods = 250.0 / horizon
        nl_sharpe = float(net_nl.mean() / net_nl.std() * np.sqrt(periods))
        nl_ann = float(net_nl.mean() * periods)
    else:
        nl_sharpe, nl_ann = np.nan, np.nan
    return {
        "horizon": horizon,
        "gross_sharpe_proxy": gp["sharpe"],
        "net_sharpe_proxy": np_["sharpe"],
        "daily_turnover_proxy": float(to_proxy.mean()) if len(to_proxy) else np.nan,
        "net_sharpe_nonoverlap": nl_sharpe,
        "net_annu_ret_nonoverlap": nl_ann,
        "n_rebalances": int(len(net_nl)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-exec-grid", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fh = (OUT / "build_log.txt").open("w", encoding="utf-8")

    try:
        log("=== C1.4 APM_SessionResidual Execution / Horizon ===", fh)
        log(f"formula={FORMULA_VERSION} | book=long_high | cost={COST_RT}", fh)
        log("No Pack · No Registry · No sign flip · No formula change", fh)

        if not PANEL.exists():
            raise FileNotFoundError(f"Missing scout panel: {PANEL}")

        sig = pd.read_parquet(PANEL)
        sig.index = pd.to_datetime(sig.index)
        session = connect_ddb()
        try:
            session.run(intraday_lib.ddb_functions)
            mask = get_universe_mask(
                session, SCOUT_START, SCOUT_END, cfg.UNIVERSE_LIST["CSI1000"]
            )
        finally:
            session.close()
        sig = apply_mask(sig, mask).dropna(how="all", axis=0).dropna(how="all", axis=1)

        ret_full = Factor_Dev_Lib.get_Ret_Matrix(
            SCOUT_START, SCOUT_END + dt.timedelta(days=40), method="c2c"
        )
        ret_full = ret_full.reindex(columns=sig.columns)
        all_idx = sig.index.union(ret_full.index).sort_values()
        ret_u = ret_full.reindex(index=all_idx, columns=sig.columns)
        log(f"Panel {sig.shape[0]}d × {sig.shape[1]} names", fh)

        # --- Horizon IC ---
        log("Horizon IC decay ...", fh)
        decay_rows = []
        for h in DECAY_HS:
            fwd = forward_cumret(ret_u, h).reindex(index=sig.index, columns=sig.columns)
            ic = sig.corrwith(fwd, axis=1, method="spearman").dropna()
            row = {
                "horizon": h,
                "rank_ic": float(ic.mean()) if len(ic) else np.nan,
                "abs_rank_ic": float(abs(ic.mean())) if len(ic) else np.nan,
                "icir": float(icir_from_daily(ic)) if len(ic) >= 20 else np.nan,
                "abs_icir": float(abs(icir_from_daily(ic))) if len(ic) >= 20 else np.nan,
                "ic_pos_frac": float((ic > 0).mean()) if len(ic) else np.nan,
                "n_days": int(len(ic)),
            }
            decay_rows.append(row)
            log(
                f"  H={h}: RankIC={row['rank_ic']:.4f} ICIR={row['abs_icir']:.2f}",
                fh,
            )
        decay_df = pd.DataFrame(decay_rows)
        decay_df.to_csv(OUT / "horizon_ic.csv", index=False)

        # --- Holding proxies ---
        log("Holding-horizon H-L proxies (long high) ...", fh)
        hold_rows = []
        for h in HOLD_HS:
            row = hl_at_horizon(sig, ret_u, h)
            ic_match = decay_df.loc[decay_df["horizon"] == h, "rank_ic"]
            row["rank_ic"] = float(ic_match.iloc[0]) if len(ic_match) else np.nan
            hold_rows.append(row)
            log(
                f"  H={h}: gross~{row['gross_sharpe_proxy']:.2f} "
                f"net_proxy~{row['net_sharpe_proxy']:.2f} "
                f"net_nl~{row['net_sharpe_nonoverlap']:.2f} "
                f"TO~{row['daily_turnover_proxy']:.3f}",
                fh,
            )
        hold_df = pd.DataFrame(hold_rows)
        hold_df.to_csv(OUT / "holding_horizon.csv", index=False)

        # --- Execution grid ---
        best = {}
        exec_df = pd.DataFrame()
        if not args.skip_exec_grid:
            log("Execution grid (rebalance + buffer) ...", fh)
            ret_e = ret_u.reindex(index=sig.index, columns=sig.columns)
            rows = []
            rebalance_schemes = [
                ("daily", dict(rebalance_freq=1, friday_only=False)),
                ("every_3d", dict(rebalance_freq=3, friday_only=False)),
                ("every_5d", dict(rebalance_freq=5, friday_only=False)),
                ("every_10d", dict(rebalance_freq=10, friday_only=False)),
            ]
            for label, kw in rebalance_schemes:
                row = evaluate_execution(
                    sig,
                    ret_e,
                    label=f"highAPM|{label}",
                    stage="E1_rebalance",
                    top_frac=TOP_FRAC,
                    weight_method="ew",
                    round_trip_cost=COST_RT,
                    **kw,
                )
                rows.append(row)
                log(
                    f"  {label}: gross={row['gross_sharpe']:.2f} "
                    f"net={row['net_sharpe']:.2f} TO={row['daily_turnover']:.3f}",
                    fh,
                )

            for blabel, entry, exit_ in [
                ("buffer_5_15", 0.05, 0.15),
                ("buffer_10_30", 0.10, 0.30),
            ]:
                for rb_name, rb_kw in rebalance_schemes:
                    row = evaluate_execution(
                        sig,
                        ret_e,
                        label=f"highAPM|{rb_name}|{blabel}",
                        stage="E2_buffer",
                        entry_frac=entry,
                        exit_frac=exit_,
                        weight_method="ew",
                        round_trip_cost=COST_RT,
                        **rb_kw,
                    )
                    rows.append(row)
                    log(
                        f"  {rb_name}|{blabel}: net={row['net_sharpe']:.2f} "
                        f"TO={row['daily_turnover']:.3f}",
                        fh,
                    )

            exec_df = pd.DataFrame(rows)
            exec_df.to_csv(OUT / "execution_grid.csv", index=False)
            ranked = exec_df.dropna(subset=["net_sharpe"]).sort_values(
                "net_sharpe", ascending=False
            )
            ranked.to_csv(OUT / "execution_ranked.csv", index=False)
            best = ranked.iloc[0].to_dict() if len(ranked) else {}
            log(
                f"Best net: {best.get('label')} net={best.get('net_sharpe')} "
                f"TO={best.get('daily_turnover')}",
                fh,
            )

            # Turnover curve plot
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                e1 = exec_df[exec_df["stage"] == "E1_rebalance"].copy()
                if len(e1):
                    ax.plot(e1["daily_turnover"], e1["net_sharpe"], "o-", label="rebalance")
                    for _, r in e1.iterrows():
                        ax.annotate(
                            str(r["label"]).split("|")[-1],
                            (r["daily_turnover"], r["net_sharpe"]),
                            fontsize=8,
                        )
                ax.axhline(1.0, color="g", ls="--", lw=0.8, label="Net=1")
                ax.axhline(0.5, color="orange", ls=":", lw=0.8, label="Net=0.5")
                ax.set_xlabel("Daily turnover")
                ax.set_ylabel("Net Sharpe @15bp")
                ax.set_title("APM_SessionResidual — execution TO vs Net Sharpe")
                ax.legend()
                fig.tight_layout()
                fig.savefig(OUT / "turnover_curve.png", dpi=120)
                plt.close(fig)
            except Exception as exc:
                log(f"turnover_curve.png skipped: {exc}", fh)

        # --- Classification ---
        ic1 = abs(float(decay_df.loc[decay_df["horizon"] == 1, "rank_ic"].iloc[0]))
        half_life = None
        for h in DECAY_HS:
            aic = abs(float(decay_df.loc[decay_df["horizon"] == h, "rank_ic"].iloc[0]))
            if aic <= 0.5 * ic1:
                half_life = h
                break

        best_net = float(best["net_sharpe"]) if best.get("net_sharpe") is not None else np.nan
        case_a = bool(np.isfinite(best_net) and best_net > 1.0)
        case_a_soft = bool(np.isfinite(best_net) and best_net > 0.5)
        ic_persistent = half_life is None or (half_life is not None and half_life > 5)

        if half_life is None and ic1 > 0:
            horizon_class = "medium_or_longer_persistent"
        elif half_life is not None and half_life <= 3:
            horizon_class = "ultra_short_daily"
        elif half_life is not None and half_life <= 10:
            horizon_class = "short_weekly"
        else:
            horizon_class = "medium_horizon"

        if case_a:
            case = "A"
            status = "testing_candidate"
            next_step = "C1 Pack v1 with documented execution recipe"
        elif case_a_soft and ic_persistent:
            case = "A_soft"
            status = "research_candidate"
            next_step = (
                "optional testing pack with recipe; prefer research_candidate until Net>1"
            )
        elif ic_persistent:
            case = "B"
            status = "research_candidate"
            next_step = "park research_candidate (like SmartMoney); no auto Pack"
        else:
            case = "C"
            status = "closed_or_diagnose"
            next_step = "IC unstable / short half-life — do not Pack"

        verdict = {
            "milestone": "C1.4",
            "factor_id": "APM_SessionResidual",
            "identity_class": "adapted_replication",
            "formula_version": FORMULA_VERSION,
            "universe": "CSI1000",
            "period": f"{SCOUT_START.date()}_{SCOUT_END.date()}",
            "signal": "apm_cs",
            "book": "long_high_short_low",
            "cost_rt": COST_RT,
            "horizon_ic": decay_df.to_dict(orient="records"),
            "holding_horizon": hold_df.to_dict(orient="records"),
            "best_execution": {
                "label": best.get("label"),
                "net_sharpe": best.get("net_sharpe"),
                "gross_sharpe": best.get("gross_sharpe"),
                "daily_turnover": best.get("daily_turnover"),
                "stage": best.get("stage"),
            }
            if best
            else {},
            "half_life_proxy_h": half_life,
            "horizon_class": horizon_class,
            "case": case,
            "case_a_net_gt_1": case_a,
            "case_a_soft_net_gt_0p5": case_a_soft,
            "recommended_status": status,
            "next": next_step,
            "forbidden": ["auto_Pack", "Registry", "Composite", "sign_flip", "formula_change"],
            "interpretation": (
                "Alpha confirmed in C1.3; C1.4 diagnoses whether slower/buffer "
                "execution clears 15bp. Do not Pack unless Case A."
            ),
        }
        (OUT / "execution_verdict.json").write_text(
            json.dumps(verdict, indent=2, default=str) + "\n", encoding="utf-8"
        )
        log(f"Wrote {OUT / 'execution_verdict.json'}", fh)
        log(
            f"CASE={case} horizon_class={horizon_class} "
            f"best_net={best_net} status→{status}",
            fh,
        )
    finally:
        fh.close()


if __name__ == "__main__":
    main()
