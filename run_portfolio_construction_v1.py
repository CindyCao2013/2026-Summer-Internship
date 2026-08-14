#!/usr/bin/env python
"""Milestone 2.2 — Portfolio Construction Layer v1.

Freeze Alpha Information Topology; build portfolio experiments on B/C only.

  B = TGD + D1          (baseline)
  C = TGD + D1 + Flow   (enhancer)

Tasks:
  1. Position sizing: equal / IC-weighted / vol-scaled exposure
  2. Risk controls: vol targeting, drawdown control
  3. Capacity: turnover, ADV participation, capital grid
  4. Light regime diagnostics (when Flow helps)

Constraints:
  - No new factors / formula changes / Registry schema edits

Outputs:
  research/reports/portfolio_construction_v1/
  docs/milestone_2_2_portfolio_construction.md  (written separately)

Usage:
  OMP_NUM_THREADS=1 python run_portfolio_construction_v1.py
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_d4_expansion_stack import daily_rank_ic_series, icir_from_daily
from alpha_investability import (
    CAPACITY_PARTICIPATION,
    DEFAULT_ROUND_TRIP_COST,
    build_long_short_weights,
    classify_market_regimes,
    estimate_capacity,
    net_pnl_series,
    series_performance,
)
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import align_signal, cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache
from run_composite_alpha_v1 import (
    SIGNAL_SHIFT,
    TOP_FRAC,
    align_all,
    ic_weighted_composite,
    rolling_ic_weights,
    si_neut,
)

OUT = Path("research/reports/portfolio_construction_v1")
TARGET_ANN_VOL = 0.15
VOL_LOOKBACK = 60
DD_THRESH = -0.10
DD_SCALE = 0.5
CAPITAL_GRID = [10e6, 50e6, 100e6, 500e6]  # CNY


def log(msg: str) -> None:
    print(msg, flush=True)


def static_weights(index: pd.Index, weight_map: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({k: float(v) for k, v in weight_map.items()}, index=index)


def build_composite_signals(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    scheme: str,
) -> Dict[str, pd.DataFrame]:
    names_b = ["TGD20", "D1"]
    names_c = ["TGD20", "D1", "FlowDensity20"]
    if scheme == "equal":
        w_b = static_weights(ret.index, {"TGD20": 0.5, "D1": 0.5})
        w_c = static_weights(
            ret.index, {"TGD20": 1 / 3, "D1": 1 / 3, "FlowDensity20": 1 / 3}
        )
    elif scheme == "ic_weighted":
        w_b = rolling_ic_weights({n: panels[n] for n in names_b}, ret, names_b, lookback=60)
        w_c = rolling_ic_weights({n: panels[n] for n in names_c}, ret, names_c, lookback=60)
    else:
        raise ValueError(scheme)
    return {
        "B": ic_weighted_composite({n: panels[n] for n in names_b}, w_b, names_b),
        "C": ic_weighted_composite({n: panels[n] for n in names_c}, w_c, names_c),
    }


def ls_weights_from_signal(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    top_frac: float = TOP_FRAC,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sig = align_signal(signal, SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    w_long, w_short = build_long_short_weights(sig, top_frac=top_frac, bottom_frac=top_frac)
    w = w_long.fillna(0.0) - w_short.fillna(0.0)
    return w, r, sig


def exposure_vol_target(
    gross_unit: pd.Series,
    *,
    target_ann_vol: float = TARGET_ANN_VOL,
    lookback: int = VOL_LOOKBACK,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Causal scale: target_vol / realized_vol(t-1)."""
    rv = gross_unit.rolling(lookback, min_periods=max(20, lookback // 3)).std() * np.sqrt(250)
    scale = (target_ann_vol / rv.shift(1)).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(upper=max_leverage).fillna(1.0)
    # warm-up: unit exposure
    scale = scale.where(rv.shift(1).notna(), 1.0)
    return scale


def exposure_drawdown_control(
    gross_unit: pd.Series,
    *,
    dd_thresh: float = DD_THRESH,
    scale_down: float = DD_SCALE,
) -> pd.Series:
    """Causal: if yesterday DD < thresh, cut exposure."""
    # unit equity from direction-adjusted path for DD measure
    g = gross_unit.copy()
    direction = 1.0 if g.mean() >= 0 else -1.0
    pnl = (g * direction).fillna(0.0)
    eq = (1.0 + pnl).cumprod()
    dd = eq / eq.cummax() - 1.0
    scale = pd.Series(1.0, index=g.index)
    scale = scale.mask(dd.shift(1) < dd_thresh, scale_down)
    return scale.fillna(1.0)


def run_portfolio(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    exposure_mode: str,
    cost_rt: float = DEFAULT_ROUND_TRIP_COST,
    top_frac: float = TOP_FRAC,
    target_ann_vol: float = TARGET_ANN_VOL,
) -> dict:
    """
    exposure_mode:
      none | vol_target | dd_control | vol_dd
      vol_scaled uses same as vol_target (portfolio-level scaling)
    """
    w, r, _ = ls_weights_from_signal(signal, ret, top_frac=top_frac)
    # unit gross (exposure=1)
    gross_unit = w.mul(r).sum(axis=1)
    to_unit = w.diff().abs().sum(axis=1)
    to_unit.iloc[0] = w.iloc[0].abs().sum()

    if exposure_mode in ("none", "equal_book"):
        exp = pd.Series(1.0, index=w.index)
    elif exposure_mode in ("vol_target", "vol_scaled"):
        exp = exposure_vol_target(gross_unit, target_ann_vol=target_ann_vol)
    elif exposure_mode == "dd_control":
        exp = exposure_drawdown_control(gross_unit)
    elif exposure_mode == "vol_dd":
        exp = exposure_vol_target(gross_unit, target_ann_vol=target_ann_vol)
        # apply DD on vol-scaled path unit for threshold, then multiply
        g_vol = gross_unit * exp
        exp = exp * exposure_drawdown_control(g_vol)
    else:
        raise ValueError(exposure_mode)

    w_s = w.mul(exp, axis=0)
    gross = w_s.mul(r).sum(axis=1)
    to = w_s.diff().abs().sum(axis=1)
    to.iloc[0] = w_s.iloc[0].abs().sum()
    net = net_pnl_series(gross, to, cost_rt)
    direction = 1 if gross.mean() >= 0 else -1
    perf_g = series_performance((gross * direction).dropna())
    perf_n = series_performance(net.dropna())
    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    return {
        "rank_ic": float(ic.mean()),
        "rank_icir": float(icir_from_daily(ic)),
        "gross_sharpe": perf_g["sharpe"],
        "net_sharpe": perf_n["sharpe"],
        "net_annu_ret": perf_n["annu_ret"],
        "mdd_net": perf_n["max_drawdown"],
        "daily_turnover": float(to.mean()),
        "mean_exposure": float(exp.mean()),
        "p95_exposure": float(exp.quantile(0.95)),
        "n_days": int(ic.dropna().shape[0]),
        "cost_rt": cost_rt,
        "_net": net,
        "_gross": gross,
        "_to": to,
        "_exp": exp,
        "_w": w_s,
    }


def book_adv_cny(signal: pd.DataFrame, amount: pd.DataFrame, top_frac: float = TOP_FRAC) -> pd.Series:
    """Daily ADV (CNY) of names in LS book — mean name ADV × n_book."""
    sig = align_signal(signal, SIGNAL_SHIFT)
    amt = amount.reindex_like(sig) * 1000.0  # 千元 → 元
    ranks = sig.rank(axis=1, pct=True)
    in_book = (ranks >= 1 - top_frac) | (ranks <= top_frac)
    daily_adv_name = amt.where(in_book).mean(axis=1)
    n_book = in_book.sum(axis=1)
    return daily_adv_name * n_book


def capacity_grid(
    signal: pd.DataFrame,
    amount: pd.DataFrame,
    daily_turnover: float,
    *,
    top_frac: float = TOP_FRAC,
    participation_cap: float = CAPACITY_PARTICIPATION,
) -> List[dict]:
    cap = estimate_capacity(signal, amount, top_frac=top_frac, signal_shift=SIGNAL_SHIFT)
    adv = book_adv_cny(signal, amount, top_frac=top_frac)
    med_adv = float(adv.median()) if adv.notna().any() else np.nan
    rows = []
    for capital in CAPITAL_GRID:
        # Approx daily traded notional (L+S gross): capital * daily_turnover
        # (daily_turnover is L1 of LS weights ≈ 2× one-way fraction of book)
        trade_notional = capital * daily_turnover
        part = trade_notional / med_adv if med_adv and med_adv > 0 else np.nan
        rows.append(
            {
                "capital_cny": capital,
                "capital_m": capital / 1e6,
                "capacity_cny_approx": cap,
                "book_adv_median_cny": med_adv,
                "implied_daily_trade_cny": trade_notional,
                "adv_participation": part,
                "participation_cap": participation_cap,
                "within_cap": bool(part <= participation_cap) if pd.notna(part) else False,
            }
        )
    return rows


def regime_table(net: pd.Series, index_ret: pd.Series) -> pd.DataFrame:
    regime = classify_market_regimes(index_ret.reindex(net.index).fillna(0))
    # vol regimes from rolling index vol
    iv = index_ret.reindex(net.index).rolling(60, min_periods=30).std() * np.sqrt(250)
    vol_med = iv.median()
    rows = []
    for label in ["bull", "bear", "sideways"]:
        mask = regime == label
        sub = net[mask]
        rows.append(
            {
                "regime": label,
                "n_days": int(sub.dropna().shape[0]),
                "net_sharpe": series_performance(sub.dropna())["sharpe"]
                if sub.dropna().shape[0] >= 40
                else np.nan,
            }
        )
    for label, mask in [
        ("high_vol", iv >= vol_med),
        ("low_vol", iv < vol_med),
    ]:
        sub = net[mask.fillna(False)]
        rows.append(
            {
                "regime": label,
                "n_days": int(sub.dropna().shape[0]),
                "net_sharpe": series_performance(sub.dropna())["sharpe"]
                if sub.dropna().shape[0] >= 40
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_nav(results: Dict[str, dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, res in results.items():
        net = res["_net"].dropna()
        nav = (1 + net).cumprod()
        ax.plot(nav.index, nav.values, label=name)
    ax.set_title("Portfolio Construction v1 — Net NAV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_capacity(cap_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, g in cap_df.groupby("model"):
        ax.plot(g["capital_m"], g["adv_participation"] * 100, marker="o", label=model)
    ax.axhline(CAPACITY_PARTICIPATION * 100, color="red", ls="--", label="5% ADV cap")
    ax.set_xlabel("Capital (CNY million)")
    ax.set_ylabel("Implied ADV participation (%)")
    ax.set_title("Capacity diagnostic (gross book turnover)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    out: Path,
    meta: dict,
    sizing: pd.DataFrame,
    risk: pd.DataFrame,
    capacity: pd.DataFrame,
    regime_b: pd.DataFrame,
    regime_c: pd.DataFrame,
) -> None:
    def tbl(df: pd.DataFrame, cols: List[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df[cols].iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}" if pd.notna(v) else "")
                elif isinstance(v, (np.floating,)):
                    cells.append(f"{float(v):.3f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# Portfolio Construction Layer v1",
        "",
        f"**Window:** {meta['start']} → {meta['end']} ({meta['n_days']}d)",
        f"**Universe identity:** China A-share Mid/Small Cap Microstructure Alpha",
        f"**Book:** top_frac={TOP_FRAC} · cost={DEFAULT_ROUND_TRIP_COST} RT · SI CS-z",
        "",
        "## Frozen roles",
        "",
        "| Factor | Role |",
        "|--------|------|",
        "| TGD20 | primary alpha source (generation) |",
        "| D1 | stabilizer / independent source |",
        "| FlowDensity20 | combination enhancer |",
        "",
        "See `docs/alpha_information_topology_v1.md`.",
        "",
        "## 1. Position sizing (no risk overlay)",
        "",
        tbl(
            sizing,
            ["model", "combine", "net_sharpe", "gross_sharpe", "mdd_net", "daily_turnover", "mean_exposure"],
        ),
        "",
        "## 2. Risk controls (IC-weighted combine)",
        "",
        tbl(
            risk,
            [
                "model",
                "exposure_mode",
                "net_sharpe",
                "gross_sharpe",
                "mdd_net",
                "daily_turnover",
                "mean_exposure",
                "p95_exposure",
            ],
        ),
        "",
        "## 3. Capacity (IC-weighted · exposure=none)",
        "",
        tbl(
            capacity,
            [
                "model",
                "capital_m",
                "capacity_cny_approx",
                "adv_participation",
                "within_cap",
                "daily_turnover",
            ],
        ),
        "",
        f"Participation cap = {CAPACITY_PARTICIPATION:.0%} of book ADV.",
        "",
        "## 4. Regime (IC-weighted · none) — Flow when?",
        "",
        "### Model B",
        "",
        tbl(regime_b, ["regime", "n_days", "net_sharpe"]),
        "",
        "### Model C",
        "",
        tbl(regime_c, ["regime", "n_days", "net_sharpe"]),
        "",
        "## Interpretation",
        "",
        "- Prefer **B** as production baseline; **C** as optional enhancer.",
        "- Vol targeting / DD control trade Sharpe vs MDD — report, do not auto-promote.",
        "- Capacity: mid/small microstructure → capital ceiling is binding before CSI300 universe is.",
        "- No new factors. Registry schema unchanged.",
        "",
    ]
    (out / "portfolio_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-vol", type=float, default=TARGET_ANN_VOL)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    log("=== Milestone 2.2 Portfolio Construction Layer v1 ===")
    log("Topology: docs/alpha_information_topology_v1.md (FROZEN)")

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2 = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)
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
    amount = enriched.amount.loc[start:end]

    log("Build panels ...")
    tgd, _ = build_tgd20_wide_from_eod_l2(
        start, end, open_=enriched.open, close=enriched.close, use_cache=True, window=20
    )
    panels_raw = {
        "TGD20": tgd.loc[start:end],
        "D1": build_eod_engine_factor("low_vol_liquidity_quality_60d", pv).loc[start:end],
        "FlowDensity20": build_net_active_flow_mktcap(l2, float_mkt, window=20).loc[start:end],
    }
    ret = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    panels_raw, ret = align_all(panels_raw, ret)
    amount = amount.reindex_like(ret)
    panels = {
        k: si_neut(v, industry.reindex_like(ret), float_mkt.reindex_like(ret))
        for k, v in panels_raw.items()
    }
    log(f"Aligned: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    # Equal-weight market proxy for regimes
    mkt_ret = ret.mean(axis=1)

    # --- 1. Position sizing ---
    log("1. Position sizing ...")
    sizing_rows = []
    nav_keep = {}
    for combine in ["equal", "ic_weighted"]:
        sigs = build_composite_signals(panels, ret, combine)
        for model, sig in sigs.items():
            res = run_portfolio(sig, ret, exposure_mode="none", target_ann_vol=args.target_vol)
            sizing_rows.append(
                {
                    "model": model,
                    "combine": combine,
                    "exposure_mode": "none",
                    **{k: v for k, v in res.items() if not k.startswith("_")},
                }
            )
            if combine == "ic_weighted":
                nav_keep[f"{model}_{combine}_none"] = res
    sizing_df = pd.DataFrame(sizing_rows)
    sizing_df.to_csv(OUT / "position_sizing.csv", index=False)
    log(sizing_df[["model", "combine", "net_sharpe", "daily_turnover"]].to_string(index=False))

    # Vol-scaled as sizing variant (IC combine)
    log("1b. Vol-scaled exposure sizing ...")
    sigs_ic = build_composite_signals(panels, ret, "ic_weighted")
    for model, sig in sigs_ic.items():
        res = run_portfolio(sig, ret, exposure_mode="vol_scaled", target_ann_vol=args.target_vol)
        sizing_rows.append(
            {
                "model": model,
                "combine": "ic_weighted",
                "exposure_mode": "vol_scaled",
                **{k: v for k, v in res.items() if not k.startswith("_")},
            }
        )
        nav_keep[f"{model}_ic_vol_scaled"] = res
    sizing_df = pd.DataFrame(sizing_rows)
    sizing_df.to_csv(OUT / "position_sizing.csv", index=False)

    # --- 2. Risk controls ---
    log("2. Risk controls ...")
    risk_rows = []
    for model, sig in sigs_ic.items():
        for mode in ["none", "vol_target", "dd_control", "vol_dd"]:
            res = run_portfolio(sig, ret, exposure_mode=mode, target_ann_vol=args.target_vol)
            risk_rows.append(
                {
                    "model": model,
                    "combine": "ic_weighted",
                    "exposure_mode": mode,
                    **{k: v for k, v in res.items() if not k.startswith("_")},
                }
            )
            nav_keep[f"{model}_{mode}"] = res
    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(OUT / "risk_controls.csv", index=False)
    log(risk_df[["model", "exposure_mode", "net_sharpe", "mdd_net", "mean_exposure"]].to_string(index=False))

    # --- 3. Capacity ---
    log("3. Capacity diagnostics ...")
    cap_rows = []
    for model, sig in sigs_ic.items():
        res = run_portfolio(sig, ret, exposure_mode="none")
        for row in capacity_grid(sig, amount, res["daily_turnover"]):
            row["model"] = model
            row["daily_turnover"] = res["daily_turnover"]
            cap_rows.append(row)
        log(
            f"  {model}: capacity≈{cap_rows[-1]['capacity_cny_approx']:.3e} CNY, "
            f"TO={res['daily_turnover']:.3f}"
        )
    capacity_df = pd.DataFrame(cap_rows)
    capacity_df.to_csv(OUT / "capacity_diagnostics.csv", index=False)

    # --- 4. Regime ---
    log("4. Regime diagnostics ...")
    res_b = run_portfolio(sigs_ic["B"], ret, exposure_mode="none")
    res_c = run_portfolio(sigs_ic["C"], ret, exposure_mode="none")
    regime_b = regime_table(res_b["_net"], mkt_ret)
    regime_c = regime_table(res_c["_net"], mkt_ret)
    regime_b.assign(model="B").to_csv(OUT / "regime_B.csv", index=False)
    regime_c.assign(model="C").to_csv(OUT / "regime_C.csv", index=False)
    # enhancer delta by regime
    delta = regime_b.merge(regime_c, on="regime", suffixes=("_B", "_C"))
    delta["delta_net_sharpe_C_minus_B"] = delta["net_sharpe_C"] - delta["net_sharpe_B"]
    delta.to_csv(OUT / "regime_enhancer_delta.csv", index=False)

    plot_nav(
        {k: nav_keep[k] for k in ["B_none", "C_none", "B_vol_target", "C_vol_target"] if k in nav_keep},
        OUT / "charts" / "nav_risk_overlays.png",
    )
    plot_capacity(capacity_df, OUT / "charts" / "capacity_participation.png")

    meta = {
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "n_days": int(len(ret)),
        "target_ann_vol": args.target_vol,
        "dd_thresh": DD_THRESH,
        "strategy_identity": "China A-share Mid/Small Cap Microstructure Alpha",
        "topology": "docs/alpha_information_topology_v1.md",
    }
    write_report(OUT, meta, sizing_df, risk_df, capacity_df, regime_b, regime_c)

    verdict = {
        "schema_version": "portfolio_construction_v1",
        "meta": meta,
        "alpha_roles": {
            "TGD20": "source",
            "D1": "stabilizer",
            "FlowDensity20": "enhancer",
        },
        "baseline_model": "B",
        "enhancer_model": "C",
        "sizing": sizing_df.drop(columns=[], errors="ignore").to_dict(orient="records"),
        "risk": risk_df.to_dict(orient="records"),
        "capacity_summary": capacity_df.groupby("model")
        .apply(lambda g: g.loc[g["capital_m"] == 100, "within_cap"].iloc[0] if (g["capital_m"] == 100).any() else None)
        .to_dict(),
    }
    # simplify verdict capacity
    verdict["capacity_at_100m"] = (
        capacity_df[capacity_df["capital_m"] == 100][
            ["model", "adv_participation", "within_cap", "capacity_cny_approx"]
        ].to_dict(orient="records")
    )
    (OUT / "portfolio_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    log(f"Wrote {OUT / 'portfolio_report.md'}")
    log("=== Portfolio Construction 2.2 complete ===")


if __name__ == "__main__":
    main()
