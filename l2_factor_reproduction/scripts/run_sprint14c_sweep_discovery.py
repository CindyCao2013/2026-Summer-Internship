#!/usr/bin/env python
"""Sprint 14C — Sweep / Book Penetration Fast Discovery.

Preflight reads Sprint 14B QA/overlap. Freezes factor_contracts.csv BEFORE
any backtest. Discovery sample only: 2023-01-01 .. 2024-12-31.

NO Full Validation / parameter grids / PRE-POST inspection / next family.

Usage:
  /opt/conda/anaconda3/bin/python -m l2_factor_reproduction.scripts.run_sprint14c_sweep_discovery
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import calAnnuRet, calSharpe  # noqa: E402
from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import backtest_factor  # noqa: E402
from l2_factor_reproduction.python.evaluation_protocol_v2 import (  # noqa: E402
    ANNUALIZATION_DAYS,
    FEE_RATE_L1,
    ensure_effective_group_to,
    l1_to_oneway,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    DISCOVERY_END,
    DISCOVERY_START,
    compute_fast_metrics,
    ensure_effective_group_pnl,
    gate_label,
    load_fast_context,
    save_fast_plots,
)

OUT = Path(RESULT_ROOT) / "sprint14_sweep_penetration" / "fast_discovery"
PRIM_DS = Path(RESULT_ROOT) / "primitives" / "sweep_penetration_daily" / "dataset"
LIQ_DS = Path(RESULT_ROOT) / "primitives" / "liquidity_impact_daily" / "dataset"
OB_DS = Path(RESULT_ROOT) / "primitives" / "order_book_daily" / "dataset"
OS_PARQUET = (
    Path(RESULT_ROOT)
    / "primitives"
    / "order_size_distribution_daily"
    / "order_size_distribution_daily_2019-01-01_2026-07-31.parquet"
)
BUILD14B = Path(RESULT_ROOT) / "sprint14_sweep_penetration" / "full_history_build"

CANDIDATES = [
    "sweep_2plus_share",
    "sweep_notional_share",
    "mean_estimated_levels_penetrated",
    "buy_sweep_share",
    "sell_sweep_share",
    "sweep_directional_asymmetry",
]
QA_DIAG = [
    "ambiguous_event_share",
    "median_alignment_lag_ms",
    "usable_event_count",
]
IC_HORIZONS = (1, 3, 5)
NEAR_ALIAS = 0.90
EVENT_CONTRACT = (
    "reference_book(t)=latest VALID SSL2 STRICTLY BEFORE trade; "
    "BUY->ASK; SELL->BID; estimated_levels_penetrated conservative"
)
POSTPROCESS = "RAW→tradability_mask→signal.shift(1)→RankIC/deciles (Fast Discovery primary)"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(x: float, d: int = 3) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# PART A — Preflight (14B artifacts only)
# ---------------------------------------------------------------------------


def preflight() -> Dict[str, Any]:
    qa_md = (BUILD14B / "primitive_QA.md").read_text(encoding="utf-8")
    overlap = pd.read_csv(BUILD14B / "overlap_diagnostics.csv")
    verdict = (BUILD14B / "verdict.txt").read_text(encoding="utf-8").strip()
    manifest = json.loads((BUILD14B / "manifest.json").read_text(encoding="utf-8"))

    pass_line = "PASS = True" in qa_md
    coverage_ok = pass_line and "rows:" in qa_md
    side_ok = "Side contract enforced" in qa_md or "BUY→ASK" in qa_md
    ambig_ok = "ambiguous_event_share" in qa_md  # documented

    # at least one sweep field not near-alias to size/depth reps
    size_depth_rights = {
        "amount_to_depth",
        "large_trade_impact",
        "log_daily_amount",
        "mean_trade_amount_usable",
    }
    non_alias = []
    for _, r in overlap.iterrows():
        if r["left"] in CANDIDATES and r["right"] in size_depth_rights:
            if abs(float(r["mean_daily_xs_spearman"])) < NEAR_ALIAS and not bool(
                r["near_alias_risk"]
            ):
                non_alias.append(
                    f"{r['left']} vs {r['right']} rho={float(r['mean_daily_xs_spearman']):.3f}"
                )
    # stronger: a candidate whose max |rho| to amount_to_depth AND large_trade_impact < 0.90
    by_left: Dict[str, Dict[str, float]] = {}
    for _, r in overlap.iterrows():
        by_left.setdefault(r["left"], {})[r["right"]] = float(
            r["mean_daily_xs_spearman"]
        )
    distinct = []
    for left, m in by_left.items():
        if left not in CANDIDATES and left != "sweep_directional_asymmetry":
            # still check asymmetry even if already in candidates
            pass
        if left not in set(CANDIDATES) | {"sweep_directional_asymmetry"}:
            continue
        atd = abs(m.get("amount_to_depth", 0.0))
        lti = abs(m.get("large_trade_impact", 0.0))
        if atd < NEAR_ALIAS and lti < NEAR_ALIAS:
            distinct.append(left)

    # Explicit known distinct from 14B: sweep_directional_asymmetry
    if "sweep_directional_asymmetry" not in distinct:
        # re-check from overlap file
        sub = overlap.loc[overlap["left"] == "sweep_directional_asymmetry"]
        if len(sub):
            max_abs = sub["mean_daily_xs_spearman"].abs().max()
            if max_abs < NEAR_ALIAS:
                distinct.append("sweep_directional_asymmetry")

    ready = (
        coverage_ok
        and side_ok
        and ambig_ok
        and len(distinct) >= 1
        and "SWEEP_DAILY_PRIMITIVE_READY" in verdict
    )
    result = {
        "passed": ready,
        "coverage_ok": coverage_ok,
        "side_contract_ok": side_ok,
        "ambiguity_documented": ambig_ok,
        "non_alias_primitives": distinct,
        "14b_verdict": verdict,
        "row_count": manifest.get("row_count")
        or json.loads(
            (
                Path(RESULT_ROOT)
                / "primitives"
                / "sweep_penetration_daily"
                / "manifest.json"
            ).read_text()
        ).get("row_count"),
    }
    (OUT / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# PART B — Freeze contracts BEFORE metrics
# ---------------------------------------------------------------------------


def freeze_contracts() -> pd.DataFrame:
    defs = {
        "sweep_2plus_share": (
            "count(usable & estimated_levels_penetrated>=2) / usable_event_count",
            "Higher multi-level penetration frequency signals aggressive liquidity consumption / information urgency",
            "unsigned intensity; sign decided by backtest display only",
        ),
        "sweep_notional_share": (
            "sum(trade_amount | usable & levels>=2) / sum(trade_amount | usable)",
            "Higher notional share in multi-level events signals stronger book pressure",
            "unsigned intensity; sign decided by backtest display only",
        ),
        "mean_estimated_levels_penetrated": (
            "mean(estimated_levels_penetrated | usable)",
            "Higher average levels penetrated = stronger realized book penetration intensity",
            "unsigned intensity; sign decided by backtest display only",
        ),
        "buy_sweep_share": (
            "count(BUY usable & levels>=2)/count(BUY usable)",
            "Aggressive BUY-side penetration intensity",
            "unsigned buy-side intensity",
        ),
        "sell_sweep_share": (
            "count(SELL usable & levels>=2)/count(SELL usable)",
            "Aggressive SELL-side penetration intensity",
            "unsigned sell-side intensity",
        ),
        "sweep_directional_asymmetry": (
            "buy_sweep_share - sell_sweep_share",
            "Buy-minus-sell penetration asymmetry; directional imbalance beyond unsigned intensity",
            "signed asymmetry (buy - sell)",
        ),
    }
    rows = []
    for fid in CANDIDATES:
        formula, hypo, direction = defs[fid]
        rows.append(
            {
                "factor_id": fid,
                "family": "sweep_penetration",
                "architecture_class": "EVENT_DRIVEN_L2",
                "formula": formula,
                "formula_hash": _sha(formula),
                "source_primitives": "sweep_penetration_daily",
                "economic_hypothesis": hypo,
                "expected_direction": direction,
                "event_contract": EVENT_CONTRACT,
                "postprocess": POSTPROCESS,
                "signal_shift": 1,
                "available_time": "after_close_T",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "factor_contracts.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_sweep(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    files = sorted(PRIM_DS.glob("quarter=*/sweep_penetration_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no sweep partitions under {PRIM_DS}")
    frames = []
    for path in files:
        df = pd.read_parquet(path)
        df["TradeDate"] = pd.to_datetime(df["TradeDate"])
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.loc[panel["TradeDate"].between(start, end)].copy()
    return panel.sort_values(["symbol", "TradeDate"], kind="stable").reset_index(
        drop=True
    )


def load_liq_cols(
    cols: List[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    files = sorted(LIQ_DS.glob("quarter=*/liquidity_impact_daily_*.parquet"))
    use = ["symbol", "TradeDate"] + cols
    frames = [pd.read_parquet(p, columns=use) for p in files]
    panel = pd.concat(frames, ignore_index=True)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"])
    panel = panel.loc[panel["TradeDate"].between(start, end)].copy()
    return panel.sort_values(["symbol", "TradeDate"], kind="stable").reset_index(
        drop=True
    )


def load_obi(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    files = sorted(OB_DS.glob("year=*/order_book_daily_*.parquet"))
    if not files:
        files = sorted(
            p
            for p in OB_DS.glob("**/order_book_daily_*.parquet")
            if "validation" not in str(p) and "smoke" not in str(p)
        )
    if not files:
        raise FileNotFoundError(f"no order_book partitions under {OB_DS}")
    use = ["symbol", "TradeDate", "obi_5_mean"]
    frames = [pd.read_parquet(p, columns=use) for p in files]
    panel = pd.concat(frames, ignore_index=True)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"])
    panel = panel.loc[panel["TradeDate"].between(start, end)].copy()
    return panel.sort_values(["symbol", "TradeDate"], kind="stable").reset_index(
        drop=True
    )


def load_large_order_ratio(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Build large_order_ratio_20w from order_size distribution parquet."""
    from l2_factor_reproduction.python.order_size_factors import (
        build_order_size_feature_frame,
    )

    raw = pd.read_parquet(OS_PARQUET)
    raw["TradeDate"] = pd.to_datetime(raw["TradeDate"])
    raw = raw.loc[raw["TradeDate"].between(start, end)].copy()
    feat = build_order_size_feature_frame(raw)
    keep = ["symbol", "TradeDate", "large_order_ratio_20w"]
    return feat[keep].sort_values(["symbol", "TradeDate"], kind="stable").reset_index(
        drop=True
    )


def series_to_narrow(
    symbols: pd.Series, dates: pd.Series, values: pd.Series, factor_id: str
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "symbol": symbols.astype(str).to_numpy(),
            "tradetime": pd.to_datetime(dates) + pd.Timedelta(hours=9, minutes=30),
            "factorname": factor_id,
            "value": values.astype(float).to_numpy(),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    out = out.loc[
        out["tradetime"].between(
            DISCOVERY_START, DISCOVERY_END + pd.Timedelta(hours=23)
        )
    ]
    return out.reset_index(drop=True)


def narrow_to_wide(narrow: pd.DataFrame) -> pd.DataFrame:
    wide = narrow.pivot_table(
        index=pd.to_datetime(narrow["tradetime"]).dt.normalize(),
        columns="symbol",
        values="value",
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def economic_diagnostics(
    group_pnl: pd.DataFrame, group_to: pd.DataFrame
) -> Dict[str, Any]:
    pnl = ensure_effective_group_pnl(group_pnl)
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl.index)
    cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
    g1, g10 = cols[0], cols[-1]
    hl = pnl["H-L"].astype(float)
    hl_l1 = to["H-L"].astype(float).reindex(hl.index).fillna(0.0)
    g10_l1 = to[g10].astype(float).reindex(hl.index).fillna(0.0)
    avg_hl_l1 = float(hl_l1.mean())
    fee_annu = avg_hl_l1 * FEE_RATE_L1 * ANNUALIZATION_DAYS
    hl_net = hl - hl_l1 * FEE_RATE_L1
    g10_gross = pnl[g10].astype(float)
    g10_net = g10_gross - g10_l1 * FEE_RATE_L1
    long_c = g10_gross
    short_c = -pnl[g1].astype(float)
    long_a = float(calAnnuRet(long_c))
    short_a = float(calAnnuRet(short_c))
    if abs(short_a) > abs(long_a) * 1.25:
        dominant = "SHORT"
    elif abs(long_a) > abs(short_a) * 1.25:
        dominant = "LONG"
    else:
        dominant = "BALANCED"
    return {
        "daily_hl_oneway_turnover": l1_to_oneway(avg_hl_l1),
        "fee_annualized_at_7p5bps": fee_annu,
        "approx_net_hl_annual": float(calAnnuRet(hl_net)),
        "approx_net_hl_sharpe": float(calSharpe(hl_net)),
        "G10_gross_excess_annual": float(calAnnuRet(g10_gross)),
        "G10_net_excess_annual": float(calAnnuRet(g10_net)),
        "daily_G10_oneway_turnover": l1_to_oneway(float(g10_l1.mean())),
        "long_contribution": long_a,
        "short_contribution": short_a,
        "dominant_leg": dominant,
    }


def daily_rank_ic(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if int(m.sum()) < 30:
        return float("nan")
    return float(a[m].corr(b[m], method="spearman"))


def persistence_diagnostics(
    wide: pd.DataFrame, ret: pd.DataFrame
) -> Dict[str, float]:
    aligned = wide.reindex(index=ret.index).sort_index()
    dates = aligned.index
    out: Dict[str, float] = {}
    for h in IC_HORIZONS:
        ics = []
        for i in range(len(dates) - h):
            ic = daily_rank_ic(aligned.loc[dates[i]], ret.loc[dates[i + h]])
            if np.isfinite(ic):
                ics.append(ic)
        out[f"IC_t{h}"] = float(np.mean(ics)) if ics else float("nan")
    out["IC_retention_t3"] = (
        out["IC_t3"] / out["IC_t1"]
        if np.isfinite(out["IC_t1"]) and abs(out["IC_t1"]) > 1e-12
        else float("nan")
    )
    out["IC_retention_t5"] = (
        out["IC_t5"] / out["IC_t1"]
        if np.isfinite(out["IC_t1"]) and abs(out["IC_t1"]) > 1e-12
        else float("nan")
    )
    rhos, g10_ret, g1_ret = [], [], []
    for i in range(len(dates) - 1):
        a = aligned.loc[dates[i]]
        b = aligned.loc[dates[i + 1]]
        rho = daily_rank_ic(a, b)
        if np.isfinite(rho):
            rhos.append(rho)
        m = a.notna()
        if int(m.sum()) < 50:
            continue
        ranks = a[m].rank(pct=True)
        top = ranks >= 0.9
        bot = ranks <= 0.1
        ranks2 = b.reindex(ranks.index).rank(pct=True)
        if int(top.sum()) > 0 and ranks2.notna().any():
            g10_ret.append(float((ranks2[top] >= 0.9).mean()))
        if int(bot.sum()) > 0 and ranks2.notna().any():
            g1_ret.append(float((ranks2[bot] <= 0.1).mean()))
    out["rank_persistence_t1"] = float(np.mean(rhos)) if rhos else float("nan")
    out["G10_retention_t1"] = float(np.mean(g10_ret)) if g10_ret else float("nan")
    out["G1_retention_t1"] = float(np.mean(g1_ret)) if g1_ret else float("nan")
    return out


def mean_daily_xs_spearman(
    left: pd.DataFrame, right: pd.DataFrame, col_l: str, col_r: str
) -> float:
    m = left[["symbol", "TradeDate", col_l]].merge(
        right[["symbol", "TradeDate", col_r]],
        on=["symbol", "TradeDate"],
        how="inner",
    )
    ics = []
    for _, g in m.groupby("TradeDate"):
        a = g[col_l].astype(float)
        b = g[col_r].astype(float)
        mask = a.notna() & b.notna()
        if int(mask.sum()) < 30:
            continue
        rho = a[mask].corr(b[mask], method="spearman")
        if np.isfinite(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def run_discovery(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contracts = pd.read_csv(OUT / "factor_contracts.csv")
    mask, ret = load_fast_context("discovery")
    discovery_rows = []
    econ_rows = []
    persist_rows = []
    factors_dir = OUT / "factors"
    plots_dir = OUT / "plots"
    factors_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for _, row in contracts.iterrows():
        fid = row["factor_id"]
        if fid not in panel.columns:
            raise KeyError(fid)
        narrow = series_to_narrow(
            panel["symbol"], panel["TradeDate"], panel[fid], fid
        )
        narrow_path = factors_dir / fid
        narrow_path.mkdir(parents=True, exist_ok=True)
        narrow.to_parquet(narrow_path / "factor_narrow.parquet", index=False)
        print(f"[bt] {fid} rows={len(narrow)}", flush=True)
        t0 = time.time()
        group_pnl, group_to, _ic, summary = backtest_factor(
            narrow,
            start_day=DISCOVERY_START,
            end_day=DISCOVERY_END,
            mask=mask,
            ret_matrix=ret,
        )
        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        econ = economic_diagnostics(group_pnl, group_to)
        gate = gate_label(metrics)

        # plots only for STRONG / RESEARCH
        if gate in ("strong_candidate", "research_candidate"):
            factor_plot_dir = plots_dir / fid
            save_fast_plots(factor_plot_dir, fid, group_pnl, metrics)
            # rename cumulative_hl.png -> cumulative_deciles_hl.png if present
            src = factor_plot_dir / "cumulative_hl.png"
            dst = factor_plot_dir / "cumulative_deciles_hl.png"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            # also place copies at plots/ root with factor prefix
            for name in ("cumulative_deciles_hl.png", "decile_bar.png"):
                p = factor_plot_dir / name
                if p.exists():
                    shutil.copy2(p, plots_dir / f"{fid}_{name}")

        # persistence for STRONG/RESEARCH only (still compute for all for summary completeness
        # but task says only STRONG+RESEARCH — store NaN for FAIL)
        pers: Dict[str, float] = {f"IC_t{h}": float("nan") for h in IC_HORIZONS}
        pers.update(
            {
                "IC_retention_t3": float("nan"),
                "IC_retention_t5": float("nan"),
                "rank_persistence_t1": float("nan"),
                "G10_retention_t1": float("nan"),
                "G1_retention_t1": float("nan"),
            }
        )
        if gate in ("strong_candidate", "research_candidate"):
            wide = narrow_to_wide(narrow)
            m = mask.reindex(index=wide.index, columns=wide.columns)
            m = m.fillna(False).astype(bool)
            wide_m = wide.where(m)
            pers = persistence_diagnostics(wide_m, ret)

        gate_map = {
            "strong_candidate": "STRONG",
            "research_candidate": "RESEARCH",
            "none": "FAIL",
        }
        discovery_rows.append(
            {
                "factor_id": fid,
                "architecture_class": "EVENT_DRIVEN_L2",
                "formula_hash": row["formula_hash"],
                "expected_direction": row["expected_direction"],
                "gate": gate_map[gate],
                "gate_raw": gate,
                "rank_ic": metrics["rank_ic_mean_raw"],
                "icir": metrics["icir_raw"],
                "gross_hl_annual": metrics["hl_annu_ret"],
                "gross_hl_sharpe": metrics["hl_sharpe"],
                "gross_hl_mdd": metrics["hl_mdd"],
                "decile_mono": metrics["decile_mono_spearman"],
                "adjacent_violations": metrics["adjacent_violations"],
                "positive_hl_month_fraction": metrics["positive_hl_month_fraction"],
                "factor_direction": metrics["factor_direction"],
                **{k: econ[k] for k in econ},
                **pers,
                "elapsed_sec": time.time() - t0,
            }
        )
        econ_rows.append({"factor_id": fid, "gate": gate_map[gate], **econ})
        persist_rows.append({"factor_id": fid, "gate": gate_map[gate], **pers})
        print(
            f"  gate={gate_map[gate]} IC={metrics['rank_ic_mean_raw']:.4f} "
            f"grossS={metrics['hl_sharpe']:.2f} "
            f"netS={econ['approx_net_hl_sharpe']:.2f} "
            f"TO={econ['daily_hl_oneway_turnover']:.3f}",
            flush=True,
        )

    discovery = pd.DataFrame(discovery_rows)
    econ_df = pd.DataFrame(econ_rows)
    persist_df = pd.DataFrame(persist_rows)
    discovery.to_csv(OUT / "discovery_summary.csv", index=False)
    econ_df.to_csv(OUT / "economic_diagnostics.csv", index=False)
    persist_df.to_csv(OUT / "persistence_summary.csv", index=False)
    return discovery, econ_df, persist_df


def run_redundancy(panel: pd.DataFrame) -> pd.DataFrame:
    print("[H] redundancy vs representatives", flush=True)
    start, end = DISCOVERY_START, DISCOVERY_END
    liq = load_liq_cols(
        ["large_trade_impact", "amount_to_depth", "depth_recovery_5m"], start, end
    )
    # liquidity_resilience_proxy_5d
    liq = liq.sort_values(["symbol", "TradeDate"], kind="stable")
    liq["liquidity_resilience_proxy_5d"] = (
        liq.groupby("symbol", sort=False)["depth_recovery_5m"]
        .transform(lambda s: s.rolling(5, min_periods=5).mean())
    )
    ob = load_obi(start, end)
    try:
        osz = load_large_order_ratio(start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"  warn order_size load failed: {exc}", flush=True)
        osz = None

    reps = {
        "large_trade_impact": liq,
        "amount_to_depth": liq,
        "liquidity_resilience_proxy_5d": liq,
        "obi_5_mean": ob,
    }
    if osz is not None:
        reps["large_order_ratio_20w"] = osz

    rows = []
    for fid in CANDIDATES:
        for rname, rdf in reps.items():
            rho = mean_daily_xs_spearman(panel, rdf, fid, rname)
            rows.append(
                {
                    "left": fid,
                    "right": rname,
                    "mean_daily_xs_spearman": rho,
                    "near_alias_risk": bool(np.isfinite(rho) and abs(rho) >= NEAR_ALIAS),
                    "window": f"{start.date()}..{end.date()}",
                }
            )
            print(f"  {fid} vs {rname}: rho={rho:.4f}", flush=True)
    # within-family
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1 :]:
            rho = mean_daily_xs_spearman(panel, panel, a, b)
            rows.append(
                {
                    "left": a,
                    "right": b,
                    "mean_daily_xs_spearman": rho,
                    "near_alias_risk": bool(np.isfinite(rho) and abs(rho) >= NEAR_ALIAS),
                    "window": f"{start.date()}..{end.date()}",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "candidate_correlation.csv", index=False)
    return out


def run_data_quality_exposure(panel: pd.DataFrame) -> pd.DataFrame:
    print("[I] data-quality exposure (diagnostic only)", flush=True)
    rows = []
    for fid in CANDIDATES:
        for q in QA_DIAG:
            rho = mean_daily_xs_spearman(panel, panel, fid, q)
            rows.append(
                {
                    "factor_id": fid,
                    "qa_field": q,
                    "mean_daily_xs_spearman": rho,
                    "concern": bool(np.isfinite(rho) and abs(rho) >= 0.70),
                    "note": "QA diagnostic only — not an alpha input",
                }
            )
            print(f"  {fid} vs {q}: rho={rho:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "data_quality_exposure.csv", index=False)
    return out


def decide(
    discovery: pd.DataFrame, corr: pd.DataFrame, dq: pd.DataFrame
) -> str:
    alias_map = {}
    for fid in CANDIDATES:
        sub = corr.loc[
            (corr["left"] == fid)
            & (corr["right"].isin(["amount_to_depth", "large_trade_impact"]))
        ]
        alias_map[fid] = bool(sub["near_alias_risk"].any()) if len(sub) else False

    dq_concern = {}
    for fid in CANDIDATES:
        sub = dq.loc[dq["factor_id"] == fid]
        dq_concern[fid] = bool(sub["concern"].any()) if len(sub) else False

    ready = []
    research = []
    for _, r in discovery.iterrows():
        fid = r["factor_id"]
        gate_ok = r["gate"] in ("STRONG", "RESEARCH")
        net_ok = np.isfinite(r["approx_net_hl_sharpe"]) and r["approx_net_hl_sharpe"] > 0
        to_ok = (
            np.isfinite(r["daily_hl_oneway_turnover"])
            and r["daily_hl_oneway_turnover"] < 1.5
        )
        persist_ok = (
            np.isfinite(r.get("IC_retention_t3", np.nan))
            and r["IC_retention_t3"] > 0.3
            and np.isfinite(r.get("IC_retention_t5", np.nan))
            # t+5 still has some info (same sign as t1 or retention > 0.2)
            and (
                (r["IC_t1"] * r["IC_t5"] > 0)
                or (abs(r.get("IC_retention_t5", 0)) > 0.2)
            )
        )
        non_alias = not alias_map.get(fid, True)
        no_dq = not dq_concern.get(fid, False)

        if gate_ok and net_ok and non_alias:
            research.append(fid)
        # READY requires STRONG Fast Gate + economics + persistence + non-alias + no DQ
        if (
            r["gate"] == "STRONG"
            and net_ok
            and to_ok
            and persist_ok
            and non_alias
            and no_dq
        ):
            ready.append(fid)

    if ready:
        verdict = "A. SWEEP_READY_FOR_SINGLE_FACTOR_FV"
    elif research or any(discovery["gate"].isin(["STRONG", "RESEARCH"])):
        # measurable/distinct info but not FV-ready
        # if ALL viable are aliases and FAIL economics → CLOSE
        viable = discovery.loc[discovery["gate"].isin(["STRONG", "RESEARCH"])]
        if len(viable) == 0:
            verdict = "C. SWEEP_FAMILY_CLOSE"
        elif all(alias_map.get(f, True) for f in viable["factor_id"]) and not any(
            np.isfinite(x) and x > 0 for x in viable["approx_net_hl_sharpe"]
        ):
            verdict = "C. SWEEP_FAMILY_CLOSE"
        elif all(alias_map.get(f, True) for f in viable["factor_id"]) and all(
            (not np.isfinite(x)) or x <= 0 for x in viable["approx_net_hl_sharpe"]
        ):
            verdict = "C. SWEEP_FAMILY_CLOSE"
        else:
            # distinct research OR research with weak gate but measurable
            if research or len(viable) > 0:
                # if only aliases with positive net but no distinct — still RESEARCH if gate hits
                # unless everything collapses to alias AND no distinct asymmetry signal
                distinct_viable = [
                    f for f in viable["factor_id"] if not alias_map.get(f, True)
                ]
                if distinct_viable or research:
                    verdict = "B. SWEEP_RESEARCH_ONLY"
                elif len(viable) > 0:
                    # gate hits but all near-alias to amount_to_depth
                    verdict = "C. SWEEP_FAMILY_CLOSE"
                else:
                    verdict = "C. SWEEP_FAMILY_CLOSE"
            else:
                verdict = "C. SWEEP_FAMILY_CLOSE"
    else:
        verdict = "C. SWEEP_FAMILY_CLOSE"

    (OUT / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    (OUT / "decision_aux.json").write_text(
        json.dumps(
            {
                "ready": ready,
                "research": research,
                "alias_map": alias_map,
                "dq_concern": dq_concern,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return verdict


def write_report(
    pre: Dict[str, Any],
    discovery: pd.DataFrame,
    corr: pd.DataFrame,
    dq: pd.DataFrame,
    verdict: str,
) -> None:
    d = discovery.copy()
    best_ic = d.loc[d["rank_ic"].abs().idxmax()]
    best_gross = d.loc[d["gross_hl_sharpe"].idxmax()]
    best_net = d.loc[d["approx_net_hl_sharpe"].idxmax()]
    buy = d.loc[d["factor_id"] == "buy_sweep_share"].iloc[0]
    sell = d.loc[d["factor_id"] == "sell_sweep_share"].iloc[0]
    asym = d.loc[d["factor_id"] == "sweep_directional_asymmetry"].iloc[0]
    unsigned = d.loc[
        d["factor_id"].isin(
            [
                "sweep_2plus_share",
                "sweep_notional_share",
                "mean_estimated_levels_penetrated",
            ]
        )
    ]
    best_unsigned = unsigned.loc[unsigned["gross_hl_sharpe"].idxmax()]

    alias_rows = corr.loc[
        corr["right"].isin(["amount_to_depth", "large_trade_impact"])
    ]

    # Sprint 12 OBI turnover reference (from known failures — high TO)
    sprint12_note = (
        "Sprint 12 OBI-style failures typically had very high daily one-way turnover "
        "(often >>0.5–1.0) with fees wiping gross Sharpe; compare Sweep TO below."
    )

    ready_ids = json.loads((OUT / "decision_aux.json").read_text())["ready"]
    research_ids = json.loads((OUT / "decision_aux.json").read_text())["research"]

    lines = [
        "# Sprint 14C — Sweep / Book Penetration Fast Discovery",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "## Preflight (14B)",
        "",
        f"- passed: `{pre['passed']}`",
        f"- coverage_ok: `{pre['coverage_ok']}`",
        f"- side_contract_ok: `{pre['side_contract_ok']}`",
        f"- ambiguity_documented: `{pre['ambiguity_documented']}`",
        f"- non_alias_primitives: `{pre['non_alias_primitives']}`",
        f"- 14B verdict: `{pre['14b_verdict']}`",
        "",
        "## Frozen contracts",
        "",
        "See `factor_contracts.csv`. Architecture: `EVENT_DRIVEN_L2`. "
        "Postprocess: RAW Fast Discovery path. signal_shift=1. Sample: 2023-01-01..2024-12-31.",
        "",
        "## Discovery summary",
        "",
        "```",
        d[
            [
                "factor_id",
                "gate",
                "rank_ic",
                "icir",
                "gross_hl_sharpe",
                "approx_net_hl_sharpe",
                "daily_hl_oneway_turnover",
                "decile_mono",
                "adjacent_violations",
                "dominant_leg",
                "IC_retention_t3",
                "IC_retention_t5",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Final Questions",
        "",
        f"1. Strongest RankIC: `{best_ic['factor_id']}` "
        f"(IC={best_ic['rank_ic']:.4f})",
        f"2. Strongest gross H-L Sharpe: `{best_gross['factor_id']}` "
        f"(Sharpe={best_gross['gross_hl_sharpe']:.3f})",
        f"3. Best net @7.5bps: `{best_net['factor_id']}` "
        f"(netS={best_net['approx_net_hl_sharpe']:.3f}, "
        f"netAnn={best_net['approx_net_hl_annual']:.3f})",
        f"4. Buy vs sell: buy gate=`{buy['gate']}` IC={buy['rank_ic']:.4f} "
        f"grossS={buy['gross_hl_sharpe']:.3f}; "
        f"sell gate=`{sell['gate']}` IC={sell['rank_ic']:.4f} "
        f"grossS={sell['gross_hl_sharpe']:.3f} "
        f"→ {'sell' if abs(sell['rank_ic']) > abs(buy['rank_ic']) else 'buy'}-side "
        f"has stronger |IC|",
        f"5. Asymmetry vs best unsigned: asym grossS={asym['gross_hl_sharpe']:.3f} "
        f"gate={asym['gate']}; best unsigned `{best_unsigned['factor_id']}` "
        f"grossS={best_unsigned['gross_hl_sharpe']:.3f} → "
        f"{'asymmetry better' if asym['gross_hl_sharpe'] > best_unsigned['gross_hl_sharpe'] else 'unsigned better / asymmetry not superior'}",
        f"6. Dominant legs: "
        + "; ".join(f"{r.factor_id}={r.dominant_leg}" for r in d.itertuples()),
        f"7. Turnover: max daily_hl_oneway_TO="
        f"{d['daily_hl_oneway_turnover'].max():.3f}; mean="
        f"{d['daily_hl_oneway_turnover'].mean():.3f}. {sprint12_note}",
        f"8. Persistence (STRONG/RESEARCH only): see persistence_summary.csv; "
        f"asym IC_t3={_fmt(asym['IC_t3'],4)} ret_t3={_fmt(asym['IC_retention_t3'])} "
        f"IC_t5={_fmt(asym['IC_t5'],4)} ret_t5={_fmt(asym['IC_retention_t5'])}",
        "9. Alias check (amount_to_depth / large_trade_impact):",
        "",
        "```",
        alias_rows.to_string(index=False),
        "```",
        "",
        "10. Data-quality exposure:",
        "",
        "```",
        dq.to_string(index=False),
        "```",
        "",
        f"11. FV candidate(s): `{ready_ids if ready_ids else 'NONE'}` "
        f"(research list: `{research_ids}`)",
        f"12. Final verdict: `{verdict}`",
        "",
        "## Hard stops",
        "",
        "- NO Full Validation auto-start",
        "- NO parameter optimization / new Sweep variants",
        "- NO PRE/POST inspection for selection",
        "- STOP after this Sprint",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(pre: Dict[str, Any], verdict: str) -> None:
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    manifest = {
        "sprint": "14C",
        "title": "Sweep / Book Penetration Fast Discovery",
        "verdict": verdict,
        "discovery_window": {
            "start": str(DISCOVERY_START.date()),
            "end": str(DISCOVERY_END.date()),
        },
        "candidates": CANDIDATES,
        "architecture_class": "EVENT_DRIVEN_L2",
        "postprocess": POSTPROCESS,
        "signal_shift": 1,
        "preflight": pre,
        "outputs": files,
        "plots_dir": str(OUT / "plots"),
        "fast_gate_unchanged": True,
        "no_full_validation": True,
        "no_parameter_optimization": True,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    print("[A] preflight 14B", flush=True)
    pre = preflight()
    print(json.dumps(pre, indent=2), flush=True)
    if not pre["passed"]:
        msg = "SWEEP_PRIMITIVE_NOT_READY_FOR_DISCOVERY"
        (OUT / "verdict.txt").write_text(msg + "\n", encoding="utf-8")
        (OUT / "report.md").write_text(
            f"# Sprint 14C STOP\n\n**{msg}**\n\nPreflight failed.\n",
            encoding="utf-8",
        )
        print(msg, flush=True)
        return 1

    print("[B] freeze factor_contracts.csv BEFORE metrics", flush=True)
    freeze_contracts()

    print("[load] sweep panel discovery window", flush=True)
    panel = load_sweep(DISCOVERY_START, DISCOVERY_END)
    print(
        f"  rows={len(panel)} symbols={panel['symbol'].nunique()} "
        f"dates={panel['TradeDate'].nunique()}",
        flush=True,
    )
    missing = [c for c in CANDIDATES if c not in panel.columns]
    if missing:
        raise KeyError(f"missing candidate fields: {missing}")

    print("[D-F] Fast Discovery + economics", flush=True)
    discovery, _econ, _pers = run_discovery(panel)

    print("[H] redundancy", flush=True)
    corr = run_redundancy(panel)

    print("[I] data-quality exposure", flush=True)
    dq = run_data_quality_exposure(panel)

    print("[K] decision", flush=True)
    verdict = decide(discovery, corr, dq)
    write_report(pre, discovery, corr, dq, verdict)
    write_manifest(pre, verdict)
    print(f"[DONE] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
