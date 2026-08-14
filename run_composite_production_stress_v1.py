#!/usr/bin/env python
"""Milestone 2.1 — Composite Production Stress Validation.

Stress the Composite v1 engine (A/B/C) — not new factors.

Dimensions:
  A. Universe: CSI300 / CSI500 / CSI1000 / ALL
  B. Cost: 10 / 15 / 20 / 30 / 50 bp round-trip
  C. Weight schemes: static 50/50 (B), rolling IC 60/120, vol-adj IC
  D. Calendar OOS: Discovery / Validation / Test

Does NOT modify Registry or factor formulas. No D4/D5.

Outputs:
  research/reports/composite_production_stress_v1/
    universe_stress.csv
    cost_stress.csv
    weight_stress.csv
    period_stress.csv
    stress_report.md
    stress_verdict.json
    charts/

Usage:
  OMP_NUM_THREADS=1 python run_composite_production_stress_v1.py
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
    DEFAULT_ROUND_TRIP_COST,
    daily_hl_pnl_and_turnover,
    net_pnl_series,
    series_performance,
)
from core.l2_features.tgd_panel_builder import build_tgd20_wide_from_eod_l2
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from factor_runner import get_universe_mask
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

OUT = Path("research/reports/composite_production_stress_v1")

COST_GRID = [0.0010, 0.0015, 0.0020, 0.0030, 0.0050]
# Calendar OOS (data window starts at cfg.START_DAY — typically 2020, not 2018)
PERIODS = [
    ("discovery", None, "2022-12-31"),  # start → 2022
    ("validation", "2023-01-01", "2024-12-31"),
    ("test", "2025-01-01", None),  # → end
]


def log(msg: str) -> None:
    print(msg, flush=True)


def apply_universe_mask(panel: pd.DataFrame, mask: Optional[pd.DataFrame]) -> pd.DataFrame:
    if mask is None:
        return panel
    m = mask.reindex(index=panel.index, columns=panel.columns)
    return panel.where(m.notna() & (m > 0))


def slice_period(
    obj: pd.DataFrame | pd.Series,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame | pd.Series:
    s = pd.Timestamp(start) if start else None
    e = pd.Timestamp(end) if end else None
    if s is not None and e is not None:
        return obj.loc[s:e]
    if s is not None:
        return obj.loc[s:]
    if e is not None:
        return obj.loc[:e]
    return obj


def static_weights(index: pd.Index, weight_map: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({k: float(v) for k, v in weight_map.items()}, index=index)


def vol_adj_ic_weights(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    names: List[str],
    *,
    lookback: int = 60,
) -> pd.DataFrame:
    """Weight ∝ max(0, rolling_IC / rolling_IC_std), causal (shift 1)."""
    ics = {n: daily_rank_ic_series(panels[n], ret, signal_shift=SIGNAL_SHIFT) for n in names}
    scores = {}
    for n in names:
        mu = ics[n].rolling(lookback, min_periods=max(20, lookback // 3)).mean().shift(1)
        sd = ics[n].rolling(lookback, min_periods=max(20, lookback // 3)).std().shift(1)
        scores[n] = (mu / sd.replace(0, np.nan)).clip(lower=0.0)
    w = pd.DataFrame(scores)
    s = w.sum(axis=1)
    equal = 1.0 / len(names)
    bad = (s <= 0) | s.isna()
    w = w.div(s.replace(0, np.nan), axis=0)
    w.loc[bad] = equal
    return w.fillna(equal)


def evaluate_signal(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    cost_rt: float = DEFAULT_ROUND_TRIP_COST,
    top_frac: float = TOP_FRAC,
) -> dict:
    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    gross, to = daily_hl_pnl_and_turnover(
        signal, ret, top_frac=top_frac, bottom_frac=top_frac, signal_shift=SIGNAL_SHIFT
    )
    net = net_pnl_series(gross, to, cost_rt)
    direction = 1 if gross.mean() >= 0 else -1
    perf_g = series_performance((gross * direction).dropna())
    perf_n = series_performance(net.dropna())
    return {
        "rank_ic": float(ic.mean()) if ic.notna().any() else np.nan,
        "rank_icir": float(icir_from_daily(ic)),
        "gross_sharpe": perf_g["sharpe"],
        "net_sharpe": perf_n["sharpe"],
        "net_annu_ret": perf_n["annu_ret"],
        "mdd_net": perf_n["max_drawdown"],
        "daily_turnover": float(to.mean()) if len(to) else np.nan,
        "n_days": int(ic.dropna().shape[0]),
        "cost_rt": cost_rt,
        "cost_bp": int(round(cost_rt * 1e4)),
    }


def build_model_signals(
    panels: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    scheme: str,
) -> Dict[str, pd.DataFrame]:
    """Return {A, B, C} signals for a weight scheme."""
    names_b = ["TGD20", "D1"]
    names_c = ["TGD20", "D1", "FlowDensity20"]
    out: Dict[str, pd.DataFrame] = {"A": panels["TGD20"]}

    if scheme == "static_50_50":
        w_b = static_weights(ret.index, {"TGD20": 0.5, "D1": 0.5})
        w_c = static_weights(ret.index, {"TGD20": 1 / 3, "D1": 1 / 3, "FlowDensity20": 1 / 3})
    elif scheme == "rolling_ic_60":
        w_b = rolling_ic_weights({n: panels[n] for n in names_b}, ret, names_b, lookback=60)
        w_c = rolling_ic_weights({n: panels[n] for n in names_c}, ret, names_c, lookback=60)
    elif scheme == "rolling_ic_120":
        w_b = rolling_ic_weights({n: panels[n] for n in names_b}, ret, names_b, lookback=120)
        w_c = rolling_ic_weights({n: panels[n] for n in names_c}, ret, names_c, lookback=120)
    elif scheme == "vol_adj_ic_60":
        w_b = vol_adj_ic_weights({n: panels[n] for n in names_b}, ret, names_b, lookback=60)
        w_c = vol_adj_ic_weights({n: panels[n] for n in names_c}, ret, names_c, lookback=60)
    else:
        raise ValueError(scheme)

    out["B"] = ic_weighted_composite({n: panels[n] for n in names_b}, w_b, names_b)
    out["C"] = ic_weighted_composite({n: panels[n] for n in names_c}, w_c, names_c)
    return out


def plot_cost_curve(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, g in df.groupby("model"):
        g = g.sort_values("cost_bp")
        ax.plot(g["cost_bp"], g["net_sharpe"], marker="o", label=model)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Round-trip cost (bp)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Cost sensitivity (ALL · rolling IC 60)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_universe_bars(df: pd.DataFrame, path: Path) -> None:
    piv = df.pivot(index="universe", columns="model", values="net_sharpe")
    order = [u for u in ["CSI300", "CSI500", "CSI1000", "ALL"] if u in piv.index]
    piv = piv.reindex(order)
    ax = piv.plot(kind="bar", figsize=(9, 4.5))
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Universe robustness (15bp · rolling IC 60)")
    ax.axhline(0, color="gray", lw=0.8)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    out: Path,
    meta: dict,
    universe: pd.DataFrame,
    cost: pd.DataFrame,
    weight: pd.DataFrame,
    period: pd.DataFrame,
) -> None:
    def md_table(df: pd.DataFrame, cols: List[str]) -> str:
        d = df[cols].copy()
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in d.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}" if pd.notna(v) else "")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    # Pass / fail heuristics
    b_all = universe[(universe["universe"] == "ALL") & (universe["model"] == "B")]
    b_csi = universe[(universe["universe"] == "CSI1000") & (universe["model"] == "B")]
    c_cost50 = cost[(cost["model"] == "C") & (cost["cost_bp"] == 50)]
    b_cost50 = cost[(cost["model"] == "B") & (cost["cost_bp"] == 50)]
    b_test = period[(period["period"] == "test") & (period["model"] == "B")]
    c_test = period[(period["period"] == "test") & (period["model"] == "C")]

    lines = [
        "# Composite Production Stress v1 (Milestone 2.1)",
        "",
        f"**Window:** {meta['start']} → {meta['end']} ({meta['n_days']}d)",
        f"**Book:** size+industry CS-z · top_frac={TOP_FRAC} · baseline cost 15bp",
        "",
        "## Alpha roles (locked)",
        "",
        "| Role | Factor |",
        "|------|--------|",
        "| Primary alpha source | TGD20 |",
        "| Independent alpha source | D1 |",
        "| Combination enhancer | FlowDensity20 |",
        "",
        "Net Sharpe stacking ≠ three cores.",
        "",
        "## A. Universe robustness",
        "",
        md_table(
            universe,
            ["universe", "model", "rank_icir", "gross_sharpe", "net_sharpe", "daily_turnover", "n_days"],
        ),
        "",
        "## B. Cost sensitivity (ALL · rolling_ic_60)",
        "",
        md_table(
            cost,
            ["model", "cost_bp", "net_sharpe", "gross_sharpe", "daily_turnover", "mdd_net"],
        ),
        "",
        "## C. Weight robustness (ALL · 15bp)",
        "",
        md_table(
            weight,
            ["scheme", "model", "rank_icir", "net_sharpe", "gross_sharpe", "daily_turnover"],
        ),
        "",
        "## D. Calendar OOS (ALL · rolling_ic_60 · 15bp)",
        "",
        f"Note: discovery starts at data `{meta['start']}` (cfg.START_DAY), not 2018.",
        "",
        md_table(
            period,
            ["period", "model", "rank_icir", "net_sharpe", "gross_sharpe", "daily_turnover", "n_days"],
        ),
        "",
        "## Stress verdict",
        "",
    ]

    def _ns(df):
        return float(df["net_sharpe"].iloc[0]) if len(df) else np.nan

    lines.append(
        f"- ALL B Net Sharpe={_ns(b_all):.2f}; CSI1000 B Net={_ns(b_csi):.2f} "
        f"({'pass: not mega-cap-only' if _ns(b_csi) > 0.5 else 'warn: weak on CSI1000'})."
    )
    lines.append(
        f"- At 50bp RT: B Net={_ns(b_cost50):.2f}, C Net={_ns(c_cost50):.2f} "
        f"({'survives' if min(_ns(b_cost50), _ns(c_cost50)) > 0 else 'breaks under high cost'})."
    )
    lines.append(
        f"- Test period: B Net={_ns(b_test):.2f}, C Net={_ns(c_test):.2f} "
        f"({'OOS positive' if _ns(b_test) > 0 else 'OOS weak — do not promote'})."
    )
    lines += [
        "",
        "## Next",
        "",
        "- If stress passes: freeze B as production baseline candidate; C optional overlay.",
        "- If fails on CSI1000 or Test: diagnose size/regime before Fundamental layer.",
        "- Do **not** expand D4/D5 into composite until stress is clean.",
        "- Next research layer: Fundamental / Value / Risk (information expansion).",
        "",
    ]
    (out / "stress_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-frac", type=float, default=TOP_FRAC)
    args = parser.parse_args()
    top_frac = args.top_frac

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    log("=== Milestone 2.1 Composite Production Stress ===")

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

    log("Build panels (full window) ...")
    tgd, _ = build_tgd20_wide_from_eod_l2(
        start, end, open_=enriched.open, close=enriched.close, use_cache=True, window=20
    )
    panels_raw = {
        "TGD20": tgd.loc[start:end],
        "D1": build_eod_engine_factor("low_vol_liquidity_quality_60d", pv).loc[start:end],
        "FlowDensity20": build_net_active_flow_mktcap(l2, float_mkt, window=20).loc[start:end],
    }
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")
    panels_raw, ret = align_all(panels_raw, ret_full)
    ind = industry.reindex_like(ret)
    mkt = float_mkt.reindex_like(ret)
    panels = {k: si_neut(v, ind, mkt) for k, v in panels_raw.items()}
    log(f"Aligned: {ret.index[0].date()} → {ret.index[-1].date()} ({len(ret)}d)")

    log("Load universe masks ...")
    uni_masks: Dict[str, Optional[pd.DataFrame]] = {"ALL": None}
    for uni, code in cfg.UNIVERSE_LIST.items():
        if code is None:
            continue
        uni_masks[uni] = get_universe_mask(session, start, end, code)

    # --- Baseline signals (rolling IC 60) ---
    log("Build baseline signals (rolling_ic_60) ...")
    base_sigs = build_model_signals(panels, ret, "rolling_ic_60")

    # A. Universe
    log("A. Universe stress ...")
    uni_rows = []
    for uni, mask in uni_masks.items():
        for model, sig in base_sigs.items():
            sig_u = apply_universe_mask(sig, mask)
            # re-zscore within universe for fair CS ranks
            sig_u = cs_zscore(sig_u)
            m = evaluate_signal(sig_u, ret, cost_rt=DEFAULT_ROUND_TRIP_COST, top_frac=top_frac)
            uni_rows.append({"universe": uni, "model": model, "scheme": "rolling_ic_60", **m})
    universe_df = pd.DataFrame(uni_rows)
    universe_df.to_csv(OUT / "universe_stress.csv", index=False)
    log(universe_df.pivot(index="universe", columns="model", values="net_sharpe").round(2).to_string())

    # B. Cost
    log("B. Cost stress ...")
    cost_rows = []
    for cost in COST_GRID:
        for model, sig in base_sigs.items():
            m = evaluate_signal(sig, ret, cost_rt=cost, top_frac=top_frac)
            cost_rows.append({"model": model, "scheme": "rolling_ic_60", "universe": "ALL", **m})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT / "cost_stress.csv", index=False)

    # C. Weight schemes
    log("C. Weight stress ...")
    weight_rows = []
    for scheme in ["static_50_50", "rolling_ic_60", "rolling_ic_120", "vol_adj_ic_60"]:
        log(f"  scheme={scheme}")
        sigs = build_model_signals(panels, ret, scheme)
        for model, sig in sigs.items():
            m = evaluate_signal(sig, ret, cost_rt=DEFAULT_ROUND_TRIP_COST, top_frac=top_frac)
            weight_rows.append({"scheme": scheme, "model": model, "universe": "ALL", **m})
    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(OUT / "weight_stress.csv", index=False)

    # D. Calendar periods
    log("D. Period / OOS stress ...")
    period_rows = []
    for pname, pstart, pend in PERIODS:
        ret_p = slice_period(ret, pstart, pend)
        if len(ret_p) < 40:
            log(f"  skip {pname}: only {len(ret_p)}d")
            continue
        for model, sig in base_sigs.items():
            sig_p = slice_period(sig, pstart, pend)
            m = evaluate_signal(sig_p, ret_p, cost_rt=DEFAULT_ROUND_TRIP_COST, top_frac=top_frac)
            period_rows.append(
                {
                    "period": pname,
                    "period_start": str(ret_p.index[0].date()),
                    "period_end": str(ret_p.index[-1].date()),
                    "model": model,
                    "scheme": "rolling_ic_60",
                    "universe": "ALL",
                    **m,
                }
            )
    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(OUT / "period_stress.csv", index=False)
    log(period_df.pivot(index="period", columns="model", values="net_sharpe").round(2).to_string())

    plot_cost_curve(cost_df, OUT / "charts" / "cost_sensitivity.png")
    plot_universe_bars(universe_df, OUT / "charts" / "universe_net_sharpe.png")

    meta = {
        "start": str(ret.index[0].date()),
        "end": str(ret.index[-1].date()),
        "n_days": int(len(ret)),
        "top_frac": top_frac,
        "note": "Discovery calendar starts at cfg.START_DAY (not 2018 if data begins later).",
    }
    write_report(OUT, meta, universe_df, cost_df, weight_df, period_df)

    verdict = {
        "schema_version": "composite_production_stress_v1",
        "meta": meta,
        "alpha_roles": {
            "primary_source": "TGD20",
            "independent_source": "D1",
            "combination_enhancer": "FlowDensity20",
        },
        "artifacts": [
            "universe_stress.csv",
            "cost_stress.csv",
            "weight_stress.csv",
            "period_stress.csv",
            "stress_report.md",
        ],
    }
    (OUT / "stress_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    log(f"Wrote {OUT / 'stress_report.md'}")
    log("=== Production Stress 2.1 complete ===")


if __name__ == "__main__":
    main()
