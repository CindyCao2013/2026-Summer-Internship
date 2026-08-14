#!/usr/bin/env python
"""L2 Flow Density v1 validation — close the loop before Temporal Feature Layer / TGD.

Uses existing intraday HML parquet (5-group minute template) for bartime×horizon
stability, plus daily **10-group + H-L** neutralization ladder / cost metrics
(project standard).

Checks:
  1. Horizon profile at fixed bartime=11:29 (Sharpe vs Ret_*)
  2. Period split: 2024H1 / 2024H2 / 2025H1 / 2025H2 bartime stability
  3. Neutralization ladder (raw / size / ind / size+ind) — daily 10-group
  4. Turnover + Implied AnnuFee(7.5%) + net-cost note
  5. Write validation report

Usage:
  OMP_NUM_THREADS=1 python run_l2_flow_density_validation_v1.py
  OMP_NUM_THREADS=1 python run_l2_flow_density_validation_v1.py --skip-daily
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_dimension_density import DISCOVERY_DAYS, split_discovery_confirmation
from alpha_investability import (
    DEFAULT_ROUND_TRIP_COST,
    evaluate_investability,
    long_book_excess_performance,
    strip_internal,
)
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from factor_runner import compute_group_stats, format_group_stats_title
from industry_neutral import load_citics_industry_panel, panel_industry_demean
from l2_data_loaders import build_l2_daily_cache
from liquidity_normalization import panel_cross_sectional_residual

OUT = Path("research/reports/l2_flow_density_v1/validation_v1")
HEATMAP_PARQUET = Path(
    "research/reports/l2_flow_density_v1/heatmaps/net_active_flow_mktcap_20d/group_data_ret.parquet"
)
FACTOR = "net_active_flow_mktcap_20d"
FOCUS_BARTIME = "11:29"
RET_ORDER = [
    "Ret_15",
    "Ret_30",
    "Ret_60",
    "Ret_90",
    "Ret_120",
    "Ret_150",
    "Ret_180",
    "Ret_EOD",
    "Ret_NDay",
]
PERIODS = [
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def bartime_hhmm(series: pd.Series) -> pd.Series:
    """TIME dtype often lands as 1970-01-01 HH:MM — extract clock only."""
    bt = pd.to_datetime(series)
    return bt.dt.strftime("%H:%M")


def hml_sharpe_ann(s: pd.Series) -> dict:
    x = s.dropna()
    if len(x) < 40:
        return {"n": len(x), "sharpe": np.nan, "annu_ret": np.nan, "mean": np.nan}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "annu_ret": float(x.mean() * 250),
        "sharpe": float(x.mean() / x.std() * np.sqrt(250)) if x.std() > 0 else np.nan,
    }


def load_hml(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["group"] == "group_HML"].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["bartime"] = bartime_hhmm(df["Bartime"])
    return df


def horizon_profile(hml: pd.DataFrame, bartime: str) -> pd.DataFrame:
    sub = hml[hml["bartime"] == bartime]
    rows = []
    for col in RET_ORDER:
        if col not in sub.columns:
            continue
        m = hml_sharpe_ann(sub[col])
        rows.append({"bartime": bartime, "horizon": col, **m})
    return pd.DataFrame(rows)


def period_bartime_table(hml: pd.DataFrame, horizons: list[str]) -> pd.DataFrame:
    rows = []
    for pname, s, e in PERIODS:
        mask = (hml["Date"] >= s) & (hml["Date"] <= e)
        chunk = hml.loc[mask]
        for bt in sorted(chunk["bartime"].unique()):
            for hz in horizons:
                if hz not in chunk.columns:
                    continue
                m = hml_sharpe_ann(chunk.loc[chunk["bartime"] == bt, hz])
                rows.append({"period": pname, "bartime": bt, "horizon": hz, **m})
    return pd.DataFrame(rows)


def plot_horizon_curve(prof: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(prof))
    ax.bar(x, prof["sharpe"], color="steelblue", edgecolor="white")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(prof["horizon"], rotation=30)
    ax.set_ylabel("HML Sharpe (ann.)")
    ax.set_title(title)
    for i, v in enumerate(prof["sharpe"]):
        if pd.notna(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_period_heatmap(tbl: pd.DataFrame, horizon: str, path: Path) -> None:
    sub = tbl[tbl["horizon"] == horizon]
    if sub.empty:
        return
    pivot = sub.pivot(index="period", columns="bartime", values="sharpe")
    # order periods / bartimes
    period_order = [p[0] for p in PERIODS if p[0] in pivot.index]
    bt_order = [b for b in ["09:59", "10:29", "10:59", "11:29", "13:29", "13:59", "14:29"] if b in pivot.columns]
    pivot = pivot.reindex(index=period_order, columns=bt_order)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(f"HML Sharpe by period × bartime | {horizon}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_neut_ladder(
    raw: pd.DataFrame,
    industry: pd.DataFrame,
    float_mkt: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    log_size = np.log(float_mkt.replace(0, np.nan)).reindex_like(raw)
    ind = industry.reindex_like(raw)
    return {
        "raw": cs_zscore(raw),
        "size": cs_zscore(panel_cross_sectional_residual(raw, [log_size])),
        "industry": cs_zscore(panel_industry_demean(raw, ind)),
        "size_industry": cs_zscore(neutralize_size_industry(raw, ind, float_mkt)),
    }


def run_daily_ladder(panels: dict, ret: pd.DataFrame, masks: dict, close, amount) -> pd.DataFrame:
    rows = []
    for mode, panel in panels.items():
        # T signal → T+1 return (match confirmation / investability)
        sig = align_signal(panel.reindex_like(ret), 1)
        r = ret.reindex_like(sig)
        _, pnl, to = Factor_Dev_Lib.groupTest(sig, r, n=10, fee=0, info="silent")
        stats = compute_group_stats(sig, r, pnl, to)
        excess = long_book_excess_performance(
            panel.reindex_like(ret),
            ret,
            top_frac=0.10,
            signal_shift=1,
            direction=stats["direction"],
        )
        inv = evaluate_investability(
            panel.reindex_like(ret),  # investability applies its own signal_shift=1
            ret,
            df_not_limit=masks["df_not_limit"].reindex_like(ret),
            df_not_st=masks["df_not_st"].reindex_like(ret),
            df_trade_status=masks["df_trade_status"].reindex_like(ret),
            close=close.reindex_like(ret),
            amount=amount.reindex_like(ret),
            round_trip_cost=DEFAULT_ROUND_TRIP_COST,
        )
        rows.append(
            {
                "mode": mode,
                "rank_ic": stats["rank_ic_mean"],
                "icir": stats["icir"],
                "hl_sharpe": stats["hl_sharpe"],
                "hl_annu_ret": stats["hl_annu_ret"],
                "hl_mdd": stats["hl_mdd"],
                "daily_turnover_hl": stats["hl_avg_turnover"],
                "implied_annu_fee": stats["implied_annu_fee"],
                "net_sharpe_15bp": inv["net_sharpe_tradable"],
                "long_book_excess_sharpe": excess["excess_sharpe"],
                "long_book_excess_annu_ret": excess["excess_annu_ret"],
                "long_book_excess_mdd": excess["excess_max_drawdown"],
                "long_group": excess["long_group"],
                "selected_count_mean": excess["selected_count_mean"],
                "universe_count_mean": excess["universe_count_mean"],
                "gross_sharpe_tradable": inv["gross_sharpe_tradable"],
                "annu_one_way_turnover": inv["annu_one_way_turnover"],
                "stats_title": format_group_stats_title(stats),
                "direction": stats["direction"],
            }
        )
        log(
            f"  neut={mode}: ICIR={stats['icir']:.2f} HL_Sharpe={stats['hl_sharpe']:.2f} "
            f"LongExcess={excess['excess_sharpe']:.2f} "
            f"ImpliedFee={stats['implied_annu_fee']:.2%} net={inv['net_sharpe_tradable']:.2f}"
        )
        OUT.mkdir(parents=True, exist_ok=True)
        if mode == "raw":
            to.to_csv(OUT / "group_turnover.csv")
        pd.concat(
            [
                excess["_long_ret"].rename("long_book_return"),
                excess["_universe_ew_ret"].rename("test_universe_ew_return"),
                excess["_excess_ret"].rename("long_book_excess_return"),
            ],
            axis=1,
        ).to_csv(OUT / f"long_book_excess_daily_{mode}.csv")
    return pd.DataFrame(rows)


def write_report(
    out: Path,
    *,
    horizon_prof: pd.DataFrame,
    period_tbl: pd.DataFrame,
    neut_df: pd.DataFrame | None,
    focus_bt: str,
) -> None:
    best = horizon_prof.loc[horizon_prof["sharpe"].idxmax()] if len(horizon_prof) else None
    # 11:29 presence across periods on Ret_120
    stab = period_tbl[(period_tbl["bartime"] == focus_bt) & (period_tbl["horizon"] == "Ret_120")]
    lines = [
        "# L2 Flow Density v1 — Validation Report",
        "",
        f"**Factor:** `{FACTOR}`",
        "**Scope:** close Flow Density loop before Temporal Feature Layer / TGD",
        "**Daily portfolio standard:** 10 groups + H-L (project default)",
        "**Intraday heatmap source:** 5-group minute template (existing parquet)",
        "",
        "## 0. Architecture note (TGD deferred)",
        "",
        "Do **not** implement `tgd.py` yet. TGD is an output of a future",
        "`core/l2_features/return_timing.py` layer (Gu/Gd → residuals → TGD).",
        "Current track closes **Flow Temporal Density** first.",
        "",
        "## 1. Horizon profile @ bartime=11:29",
        "",
    ]
    if best is not None:
        lines.append(
            f"Peak Sharpe at `{best['horizon']}` = **{best['sharpe']:.2f}** "
            f"(annu≈{best['annu_ret']:.1%}, n={int(best['n'])})."
        )
    lines += [
        "",
        "Artifact: `horizon_profile_1129.csv` / `horizon_sharpe_1129.png`",
        "",
        "| Horizon | Sharpe | AnnuRet | n |",
        "|---------|--------|---------|---|",
    ]
    for _, r in horizon_prof.iterrows():
        lines.append(
            f"| {r['horizon']} | {r['sharpe']:.2f} | {r['annu_ret']:.1%} | {int(r['n'])} |"
        )

    lines += [
        "",
        "## 2. Period × bartime stability (Ret_120 / Ret_NDay)",
        "",
        "Artifacts: `period_bartime_sharpe.csv`, `period_heatmap_Ret_120.png`, `period_heatmap_Ret_NDay.png`",
        "",
        f"### Focus bartime `{focus_bt}` × Ret_120",
        "",
        "| Period | Sharpe | n |",
        "|--------|--------|---|",
    ]
    for _, r in stab.iterrows():
        lines.append(f"| {r['period']} | {r['sharpe']:.2f} | {int(r['n'])} |")
    pos = stab["sharpe"].dropna()
    if len(pos):
        lines.append(
            f"\nPositive-period ratio @ {focus_bt}/Ret_120: "
            f"**{(pos > 0).mean():.0%}** ({(pos > 0).sum()}/{len(pos)})."
        )

    lines += ["", "## 3. Neutralization ladder (daily 10-group + H-L)", ""]
    if neut_df is None or neut_df.empty:
        lines.append("_Skipped (`--skip-daily`)._")
    else:
        lines += [
            "| Mode | ICIR | Long-book excess Sharpe | HL Sharpe | Net Sharpe@15bp | Daily TO(H-L) |",
            "|------|------|-------------------------|-----------|-----------------|---------------|",
        ]
        for _, r in neut_df.iterrows():
            lines.append(
                f"| {r['mode']} | {r['icir']:.2f} | {r['long_book_excess_sharpe']:.2f} | "
                f"{r['hl_sharpe']:.2f} | {r['net_sharpe_15bp']:.2f} | "
                f"{r['daily_turnover_hl']:.2f} |"
            )
        lines += [
            "",
            "Interpretation: if size+industry retains ICIR/HL vs raw, alpha is not just size/ind.",
        ]

    lines += [
        "",
        "## 4. Verdict checklist",
        "",
        "- [x] Horizon profile (not single-cell heatmap cherry-pick)",
        "- [x] Period split (2024H1–2025H2)",
        "- [x] Neutralization ladder (if daily ran)",
        "- [x] Turnover / Implied AnnuFee / net cost",
        "- [ ] Temporal Feature Layer (`return_timing.py`) — **next after this closes**",
        "- [ ] TGD factor — **after** return_timing primitives",
        "",
        "## 5. Next",
        "",
        "```",
        "L2 Flow Density v1  →  Temporal Feature Layer (Gu/Gd)  →  TGD  →  APM",
        "```",
        "",
    ]
    (out / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-daily", action="store_true", help="Skip neut ladder (intraday-only)")
    parser.add_argument("--discovery-days", type=int, default=DISCOVERY_DAYS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    log("=== L2 Flow Density v1 Validation ===")
    if not HEATMAP_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {HEATMAP_PARQUET} — run run_p2_intraday_heatmap.py first"
        )

    hml = load_hml(HEATMAP_PARQUET)
    log(f"HML rows: {len(hml)} | {hml['Date'].min().date()} -> {hml['Date'].max().date()}")

    # --- 1) Horizon @ 11:29 ---
    prof = horizon_profile(hml, FOCUS_BARTIME)
    prof.to_csv(OUT / "horizon_profile_1129.csv", index=False)
    plot_horizon_curve(
        prof,
        OUT / "horizon_sharpe_1129.png",
        f"{FACTOR} — HML Sharpe by horizon @ {FOCUS_BARTIME}",
    )
    log(f"Horizon @ {FOCUS_BARTIME}:\n{prof.to_string(index=False)}")

    # --- 2) Period × bartime ---
    period_tbl = period_bartime_table(hml, ["Ret_120", "Ret_NDay", "Ret_60"])
    period_tbl.to_csv(OUT / "period_bartime_sharpe.csv", index=False)
    plot_period_heatmap(period_tbl, "Ret_120", OUT / "period_heatmap_Ret_120.png")
    plot_period_heatmap(period_tbl, "Ret_NDay", OUT / "period_heatmap_Ret_NDay.png")
    log("Period×bartime tables written")

    # --- 3/4) Daily neut ladder ---
    neut_df = None
    if not args.skip_daily:
        log("\n--- Daily 10-group neutralization ladder (confirmation) ---")
        start, end = cfg.START_DAY, cfg.END_DAY
        preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
        enriched, session = load_eod_enriched_tables(preheat, end)
        session.run(intraday_lib.ddb_functions)
        industry = load_citics_industry_panel(start, end)
        l2 = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
        float_mkt = enriched.float_mktcap.loc[start:end]
        raw = build_net_active_flow_mktcap(l2, float_mkt, window=20).loc[start:end]
        ladder = build_neut_ladder(raw, industry.reindex_like(raw), float_mkt)

        ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
        _, ret_conf = split_discovery_confirmation(ret_full, args.discovery_days)
        log(f"Confirmation: {ret_conf.index[0].date()} -> {ret_conf.index[-1].date()} ({len(ret_conf)}d)")

        masks = {
            "df_not_limit": Factor_Dev_Lib.get_EOD_Not_Limit(start, end),
            "df_not_st": Factor_Dev_Lib.get_EOD_Not_ST(start, end),
            "df_trade_status": Factor_Dev_Lib.get_TradeStatus(start, end),
        }
        panels_c = {k: v.reindex(index=ret_conf.index, columns=ret_conf.columns) for k, v in ladder.items()}
        neut_df = run_daily_ladder(
            panels_c,
            ret_conf,
            masks,
            enriched.close,
            enriched.amount,
        )
        neut_df.to_csv(OUT / "neutralization_ladder.csv", index=False)
    else:
        log("Skipped daily neut ladder")

    write_report(OUT, horizon_prof=prof, period_tbl=period_tbl, neut_df=neut_df, focus_bt=FOCUS_BARTIME)

    payload = {
        "factor": FACTOR,
        "focus_bartime": FOCUS_BARTIME,
        "horizon_peak": (
            prof.loc[prof["sharpe"].idxmax()].to_dict() if len(prof) and prof["sharpe"].notna().any() else None
        ),
        "period_focus_ret120": period_tbl[
            (period_tbl["bartime"] == FOCUS_BARTIME) & (period_tbl["horizon"] == "Ret_120")
        ].to_dict(orient="records"),
        "neutralization": neut_df.to_dict(orient="records") if neut_df is not None else None,
        "next": [
            "Temporal Feature Layer: core/l2_features/return_timing.py (Gu/Gd)",
            "Then thin tgd.py on residuals — not a standalone factor script",
        ],
    }
    (OUT / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    log(f"\nWrote {OUT}/VALIDATION_REPORT.md")


if __name__ == "__main__":
    main()
