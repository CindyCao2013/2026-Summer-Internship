#!/usr/bin/env python
"""TGD paper-replication integrity — mechanism + primitive family (factor frozen).

Does NOT retune TGD20. Builds:
  1) Mechanism: εu / εd / tgd_eps / TGD20
  2) Family: Gu / Gd / τ=Gd−Gu / υ=|Gd−Gu| / TGD20  (MA20 for fair compare)
  3) Machine-readable metrics.json + factor_summary.csv
  4) Feeds research report tables

Usage:
  OMP_NUM_THREADS=1 python run_tgd_replication_integrity.py
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
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import DEFAULT_ROUND_TRIP_COST, evaluate_investability
from factor_attribution import align_signal, cs_zscore
from factor_eval_metrics import (
    metrics_to_summary_row,
    pack_factor_metrics,
    save_metrics,
    schema_table,
)
from factor_runner import compute_group_stats

OUT = Path("research/reports/tgd_v1/replication")
LONG_CACHE = Path("research/cache/tgd_panels/TGD20_long_20200101_20251231_w20.parquet")
NEUT = Path("research/reports/tgd_v1/neutralization/neut_summary.csv")
STAB = Path("research/reports/tgd_v1/stability/yearly_ic.csv")
EXEC_BEST = Path("research/reports/tgd_v1/execution/all_experiments.csv")
N_GROUPS = 10
SIGNAL_SHIFT = 1
MA = 20


def log(msg: str) -> None:
    print(msg, flush=True)


def long_to_wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    w = df.pivot(index="date", columns="symbol", values=col)
    w.index = pd.to_datetime(w.index)
    w.index.name = "Date"
    return w.sort_index()


def ma20_wide(wide: pd.DataFrame, window: int = MA) -> pd.DataFrame:
    return wide.rolling(window, min_periods=window).mean()


def decile_monotonicity(pnl: pd.DataFrame) -> float:
    means = pnl[[c for c in pnl.columns if c != "H-L"]].mean()
    try:
        order = pd.to_numeric(means.index, errors="coerce").to_numpy(dtype=float)
    except Exception:
        order = np.arange(1, len(means) + 1, dtype=float)
    y = means.to_numpy(dtype=float)
    m = np.isfinite(order) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(pd.Series(order[m]).corr(pd.Series(y[m]), method="spearman"))


def eval_factor(
    name: str,
    panel: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    family: str,
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
    mono = decile_monotonicity(pnl)
    row = {
        "factor": name,
        "family": family,
        "rank_ic": stats["rank_ic_mean"],
        "icir": stats["icir"],
        "annu_ic": stats["rank_ic_mean"] * np.sqrt(250),
        "hl_annu_ret": stats["hl_annu_ret"],
        "hl_sharpe": stats["hl_sharpe"],
        "hl_mdd": stats["hl_mdd"],
        "daily_turnover": stats["hl_avg_turnover"],
        "implied_annu_fee": stats["implied_annu_fee"],
        "net_sharpe": inv["net_sharpe_tradable"],
        "gross_sharpe_tradable": inv["gross_sharpe_tradable"],
        "monotonicity": mono,
        "direction": stats["direction"],
    }
    log(
        f"  {name:18s} RankIC={row['rank_ic']:.4f} ICIR={row['icir']:.2f} "
        f"HL={row['hl_sharpe']:.2f} net={row['net_sharpe']:.2f} mono={mono:.3f}"
    )
    return row


def build_family_panels(long: pd.DataFrame) -> dict:
    Gu = long_to_wide(long, "Gu")
    Gd = long_to_wide(long, "Gd")
    tau = Gd - Gu
    upsilon = (Gd - Gu).abs()
    eps_u = long_to_wide(long, "epsilon_u")
    eps_d = long_to_wide(long, "epsilon_d")
    tgd_eps = long_to_wide(long, "tgd_eps")
    tgd20 = long_to_wide(long, "TGD20")
    return {
        # mechanism (daily + MA20)
        "epsilon_u": (eps_u, "mechanism"),
        "epsilon_d": (eps_d, "mechanism"),
        "tgd_eps": (tgd_eps, "mechanism"),
        "epsilon_u_MA20": (ma20_wide(eps_u), "mechanism"),
        "epsilon_d_MA20": (ma20_wide(eps_d), "mechanism"),
        "TGD20": (tgd20, "mechanism"),
        # primitive family (MA20 fair vs TGD20)
        "Gu_MA20": (ma20_wide(Gu), "family"),
        "Gd_MA20": (ma20_wide(Gd), "family"),
        "tau_MA20": (ma20_wide(tau), "family"),
        "upsilon_MA20": (ma20_wide(upsilon), "family"),
        "TGD20_family": (tgd20, "family"),
    }


def assemble_canonical_metrics(out: Path, conf_period: str) -> None:
    """Phase-1 machine-readable pack from Stage-4 + execution best."""
    schema_table().to_csv(out / "factor_metrics_schema.csv", index=False)
    (out / "factor_metrics_schema.json").write_text(
        json.dumps(
            {"schema": schema_table().to_dict(orient="records"), "note": "Research vs production dual score"},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    neut = pd.read_csv(NEUT) if NEUT.exists() else pd.DataFrame()
    payloads = []
    for _, r in neut.iterrows():
        payloads.append(
            pack_factor_metrics(
                factor="TGD20",
                period=conf_period,
                universe="ALL",
                mode=str(r["mode"]),
                rank_ic=r["rank_ic"],
                icir=r["icir"],
                hl_annu_ret=r["hl_annu_ret"],
                hl_sharpe=r["hl_sharpe"],
                hl_mdd=r["hl_mdd"],
                daily_turnover=r["daily_turnover_hl"],
                implied_annu_fee=r["implied_annu_fee"],
                net_sharpe=r["net_sharpe_15bp"],
                monotonicity=r.get("monotonicity_spearman"),
                direction=int(r["direction"]),
                extra={"source": "stage4_neutralization"},
            )
        )

    # execution investable best
    if EXEC_BEST.exists():
        ex = pd.read_csv(EXEC_BEST).dropna(subset=["net_sharpe"])
        if len(ex):
            best = ex.sort_values("net_sharpe", ascending=False).iloc[0]
            payloads.append(
                pack_factor_metrics(
                    factor="TGD20",
                    period=conf_period,
                    universe="ALL",
                    mode="execution_best",
                    rank_ic=best["rank_ic"],
                    icir=best["icir"],
                    hl_annu_ret=best["gross_annu_ret"],
                    hl_sharpe=best["gross_sharpe"],
                    hl_mdd=best["mdd_net"],
                    daily_turnover=best["daily_turnover"],
                    implied_annu_fee=best["implied_annu_fee"],
                    net_sharpe=best["net_sharpe"],
                    monotonicity=None,
                    direction=int(best["direction"]),
                    extra={"label": best["label"], "source": "execution_opt_v1"},
                )
            )

    save_metrics(
        {
            "factor": "TGD20",
            "variants": payloads,
            "stability": pd.read_csv(STAB).to_dict(orient="records") if STAB.exists() else [],
        },
        out / "metrics.json",
    )
    summary = pd.DataFrame([metrics_to_summary_row(p) for p in payloads])
    summary.to_csv(out / "factor_summary.csv", index=False)
    log(f"Wrote {out / 'metrics.json'} and factor_summary.csv")


def write_integrity_md(out: Path, mech: pd.DataFrame, fam: pd.DataFrame) -> None:
    lines = [
        "# TGD Replication Integrity",
        "",
        "Factor definition frozen. This checks **why** TGD works vs simpler primitives.",
        "",
        "## Mechanism decomposition",
        "",
        "| Factor | RankIC | ICIR | H-L Sharpe | Net@15bp | Mono |",
        "|--------|--------|------|------------|----------|------|",
    ]
    for _, r in mech.iterrows():
        lines.append(
            f"| `{r['factor']}` | {r['rank_ic']:.4f} | {r['icir']:.2f} | {r['hl_sharpe']:.2f} | "
            f"{r['net_sharpe']:.2f} | {r['monotonicity']:.3f} |"
        )
    lines += [
        "",
        "Interpretation: if `TGD20` ≫ `epsilon_u` / `epsilon_d` alone, alpha comes from "
        "the εd⊥εu residual (late-selling / early-buying asymmetry), not raw timing centers.",
        "",
        "## Primitive family (MA20-smoothed)",
        "",
        "| Factor | RankIC | ICIR | H-L Sharpe | Net@15bp | Mono |",
        "|--------|--------|------|------------|----------|------|",
    ]
    for _, r in fam.iterrows():
        lines.append(
            f"| `{r['factor']}` | {r['rank_ic']:.4f} | {r['icir']:.2f} | {r['hl_sharpe']:.2f} | "
            f"{r['net_sharpe']:.2f} | {r['monotonicity']:.3f} |"
        )
    lines += [
        "",
        "Interpretation: if `TGD20` beats `tau_MA20` / `upsilon_MA20`, residualization vs "
        "return-structure controls is the incremental alpha source (研报否定的简单 τ).",
        "",
        "## Paper vs framework",
        "",
        "| Item | Paper | This framework |",
        "|------|-------|----------------|",
        "| Data | 1-min returns | Stock_one_minute Close |",
        "| Gu/Gd | √ | √ (`return_timing.py`) |",
        "| Controls | Rū/Rd̄, R1, R2, overnight | √ conditional means |",
        "| Residual | √ | √ CS OLS |",
        "| TGD | εd~εu → MA20 | √ (`tgd.py`) |",
        "| Groups | 5 | **10 + H-L** |",
        "| Neutral | limited | size / industry / both |",
        "| Cost | typically none | 15bp RT + ImpliedFee(7.5bps) |",
        "| Shift | often implicit | **explicit shift-1** |",
        "",
    ]
    (out / "REPLICATION_INTEGRITY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("=== TGD Replication Integrity ===")
    if not LONG_CACHE.exists():
        raise FileNotFoundError(f"Missing {LONG_CACHE} — run Stage-4 panel build first")

    start, end = cfg.START_DAY, cfg.END_DAY
    long = pd.read_parquet(LONG_CACHE)
    long["date"] = pd.to_datetime(long["date"])
    long = long[(long["date"] >= pd.Timestamp(start)) & (long["date"] <= pd.Timestamp(end))]

    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    _, ret = split_discovery_confirmation(ret_full, args.discovery_days)
    if ret.empty:
        ret = ret_full
    conf_period = f"{ret.index[0].date()}_{ret.index[-1].date()}"
    log(f"Confirmation: {conf_period} ({len(ret)}d)")

    # Phase 1 metrics pack (from existing Stage-4 artifacts)
    assemble_canonical_metrics(OUT, conf_period)

    # Load EOD for investability
    from factor_data_loaders import load_eod_enriched_tables

    enriched, _ = load_eod_enriched_tables(start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS), end)
    masks = {
        "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(ret.index[0].to_pydatetime(), ret.index[-1].to_pydatetime()),
        "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(ret.index[0].to_pydatetime(), ret.index[-1].to_pydatetime()),
        "df_trade_status": Factor_Dev_Lib.get_TradeStatus(ret.index[0].to_pydatetime(), ret.index[-1].to_pydatetime()),
    }

    panels = build_family_panels(long)
    rows = []
    log("\n--- Mechanism / family evaluations ---")
    for name, (panel, family) in panels.items():
        # align columns to ret
        panel = panel.reindex(index=ret.index, columns=ret.columns)
        rows.append(
            eval_factor(
                name,
                panel,
                ret,
                family=family,
                masks=masks,
                close=enriched.close,
                amount=enriched.amount,
            )
        )

    tbl = pd.DataFrame(rows)
    mech = tbl[tbl["family"] == "mechanism"].copy()
    fam = tbl[tbl["family"] == "family"].copy()
    # rename TGD20_family display
    fam["factor"] = fam["factor"].replace({"TGD20_family": "TGD20"})
    mech.to_csv(OUT / "mechanism_decomposition.csv", index=False)
    fam.to_csv(OUT / "primitive_family.csv", index=False)
    tbl.to_csv(OUT / "integrity_all.csv", index=False)
    write_integrity_md(OUT, mech, fam)
    log(f"\nWrote {OUT / 'REPLICATION_INTEGRITY.md'}")


if __name__ == "__main__":
    main()
