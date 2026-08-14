#!/usr/bin/env python
"""Sprint 12A — ORDER_BOOK / LOB_PRESSURE Baseline Discovery.

Reuse existing order_book_daily primitives + frozen factor_narrow caches.
Discovery window 2023-01-01~2024-12-31. No Full Validation. No new formulas.
Protocol v2.0 thresholds untouched.

Usage:
    python -m l2_factor_reproduction.scripts.run_sprint12_order_book
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
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
from l2_factor_reproduction.python.order_book_factors import (  # noqa: E402
    ORDER_BOOK_FACTOR_SPECS,
)

OUT_ROOT = Path(RESULT_ROOT) / "sprint12_order_book"
FAMILY_ROOT = Path(RESULT_ROOT) / "candidate_pool_v1" / "order_book_family"
FACTORS_DIR = FAMILY_ROOT / "factors"
PRIM_ROOT = Path(RESULT_ROOT) / "primitives"
OB_PRIM = PRIM_ROOT / "order_book_daily"
DDB_PRIM = PRIM_ROOT / "ddb_reference_snapshot"
LIQ_PRIM = PRIM_ROOT / "liquidity_impact_daily"

NEAR_ALIAS_RHO = 0.90

# Mechanism-driven baseline (~16). Exact frozen formulas only; avoid near-alias variants.
BASELINE: List[Dict[str, str]] = [
    # 1. STATIC BOOK PRESSURE
    {"factor": "obi_l1_mean", "mechanism_bucket": "STATIC_BOOK_PRESSURE"},
    {"factor": "obi_l5_mean", "mechanism_bucket": "STATIC_BOOK_PRESSURE"},
    {"factor": "near_far_imbalance", "mechanism_bucket": "STATIC_BOOK_PRESSURE"},
    {"factor": "depth_slope_asymmetry", "mechanism_bucket": "STATIC_BOOK_PRESSURE"},
    {"factor": "depth_concentration_asymmetry", "mechanism_bucket": "STATIC_BOOK_PRESSURE"},
    # 2. PRESSURE PERSISTENCE
    {"factor": "obi_sign_persistence", "mechanism_bucket": "PRESSURE_PERSISTENCE"},
    {"factor": "opening_obi_l5", "mechanism_bucket": "PRESSURE_PERSISTENCE"},
    {"factor": "closing_obi_l5", "mechanism_bucket": "PRESSURE_PERSISTENCE"},
    # 3. PRESSURE CHANGE / SHOCK
    {"factor": "obi_intraday_slope", "mechanism_bucket": "PRESSURE_CHANGE_SHOCK"},
    {"factor": "opening_closing_obi_change", "mechanism_bucket": "PRESSURE_CHANGE_SHOCK"},
    {"factor": "obi_shock_20d", "mechanism_bucket": "PRESSURE_CHANGE_SHOCK"},
    {"factor": "microprice_shock_20d", "mechanism_bucket": "PRESSURE_CHANGE_SHOCK"},
    {"factor": "depth_shock_20d", "mechanism_bucket": "PRESSURE_CHANGE_SHOCK"},
    {"factor": "spread_shock_20d", "mechanism_bucket": "PRESSURE_CHANGE_SHOCK"},
    # 4. PRICE–BOOK DISAGREEMENT
    {"factor": "microprice_deviation_mean", "mechanism_bucket": "PRICE_BOOK_DISAGREEMENT"},
    {"factor": "book_vwap_gap", "mechanism_bucket": "PRICE_BOOK_DISAGREEMENT"},
    {"factor": "relative_spread_mean", "mechanism_bucket": "PRICE_BOOK_DISAGREEMENT"},
    {"factor": "total_depth_volatility", "mechanism_bucket": "PRICE_BOOK_DISAGREEMENT"},
]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2%}"


def _fmt_num(x: float, d: int = 3) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# PART A helper — also ensure Sprint 11 ledger updated from runner
# ---------------------------------------------------------------------------


def update_sprint11_ledger() -> None:
    s11 = Path(RESULT_ROOT) / "fast_discovery" / "price_formation_v1"
    closed = s11 / "Sprint_11_CLOSED.md"
    assert closed.exists(), f"missing {closed}"
    man_path = s11 / "manifest.json"
    m = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    m.update(
        {
            "status": "CLOSED",
            "sprint": "Sprint 11 — Price Formation / Intraday Path Family v1",
            "closed_doc": "Sprint_11_CLOSED.md",
            "strong_fv_outcomes": {
                "tail_return_share": "FROZEN_C_NOT_CONFIRMED",
                "intraday_max_drawdown": "FROZEN_C_NOT_CONFIRMED",
            },
            "realized_kurtosis": "NEAR_ALIAS / NO_FULL_VALIDATION",
            "close_auction_return": "ANOMALOUS_ARTIFACT_WATCHLIST / NO_FULL_VALIDATION",
            "n_strong_entered_fv": 2,
            "n_confirmed_research_factors": 0,
            "parameter_optimization": "FORBIDDEN",
            "next_sprint": "Sprint 12A — ORDER_BOOK / LOB_PRESSURE Baseline Discovery",
        }
    )
    man_path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# PART C — primitive inventory
# ---------------------------------------------------------------------------


def _manifest_coverage(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_primitive_inventory() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    ob = _manifest_coverage(OB_PRIM / "manifest.json")
    cov = ob.get("date_coverage", {})
    # Column-level inventory from schema / known contract
    schema_path = OB_PRIM / "schema_contract.json"
    cols: List[str] = []
    if schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        raw_cols = schema.get("columns") or schema.get("required_columns") or []
        if raw_cols and isinstance(raw_cols[0], dict):
            cols = [str(c.get("name")) for c in raw_cols if c.get("name")]
        else:
            cols = list(raw_cols)
    if not cols:
        # fallback: read one parquet
        files = sorted((OB_PRIM / "dataset").glob("**/*.parquet"))
        if files:
            cols = list(pd.read_parquet(files[0]).columns)

    l2_native_cols = [
        c
        for c in cols
        if c
        not in {
            "symbol",
            "TradeDate",
            "source_exchange",
            "valid_snapshot_count",
            "valid_minute_count",
            "expected_minute_count",
            "coverage_ratio",
            "close_auction_valid",
        }
    ]
    for col in l2_native_cols:
        rows.append(
            {
                "primitive": col,
                "source_table_or_cache": str(OB_PRIM / "dataset"),
                "available": True,
                "coverage_start": cov.get("actual_min", cov.get("requested_start", "")),
                "coverage_end": cov.get("actual_max", cov.get("requested_end", "")),
                "frequency": "symbol-day (from ~3s snapshot → minute-last → daily)",
                "definition": f"order_book_daily.{col} (formula_version={ob.get('formula_version','')})",
                "is_l2_native": True,
                "notes": f"schema={ob.get('schema_version','')}; eligible_rows={ob.get('eligible_row_count','')}",
            }
        )

    # DDB reference (also L2-native but separate family — inventory only)
    ddb_man = DDB_PRIM / "dataset_manifest.json"
    if not ddb_man.exists():
        ddb_man = DDB_PRIM / "manifest.json"
    ddb = _manifest_coverage(ddb_man) if ddb_man.exists() else {}
    ddb_cov = ddb.get("date_coverage", ddb.get("coverage", {}))
    for name in (
        "time_weighted_order_slope_mean",
        "wavg_soir_mean",
        "tra_price_weighted_net_buy_quote_volume_ratio_mean",
        "level10_diff_buy_mean",
        "level10_infer_price_trend_mean",
    ):
        rows.append(
            {
                "primitive": name,
                "source_table_or_cache": str(DDB_PRIM / "dataset"),
                "available": bool(ddb),
                "coverage_start": ddb_cov.get("actual_min", ddb_cov.get("start", "")),
                "coverage_end": ddb_cov.get("actual_max", ddb_cov.get("end", "")),
                "frequency": "symbol-day (DDB official snapshot formulas)",
                "definition": f"ddb_reference_snapshot.{name}",
                "is_l2_native": True,
                "notes": "Inventory only for Sprint 12A; baseline pool uses order_book_daily frozen formulas",
            }
        )

    # Explicitly mark missing queue primitives
    for missing in (
        "queue_imbalance",
        "bid_order_count_depth",
        "ask_order_count_depth",
        "queue_position",
    ):
        rows.append(
            {
                "primitive": missing,
                "source_table_or_cache": "n/a",
                "available": False,
                "coverage_start": "",
                "coverage_end": "",
                "frequency": "n/a",
                "definition": "not implemented (SZSE lacks ten-level BidNums/AskNums)",
                "is_l2_native": True,
                "notes": "MISSING — do not proxy with OHLCV",
            }
        )

    # liquidity_impact hybrid note
    liq = _manifest_coverage(LIQ_PRIM / "manifest.json")
    liq_cov = liq.get("date_coverage", {})
    rows.append(
        {
            "primitive": "liquidity_impact_daily (hybrid trade×book)",
            "source_table_or_cache": str(LIQ_PRIM / "dataset"),
            "available": bool(liq),
            "coverage_start": liq_cov.get("actual_min", ""),
            "coverage_end": liq_cov.get("actual_max", ""),
            "frequency": "symbol-day",
            "definition": "trade x minute-last book impact — not pure LOB family",
            "is_l2_native": False,
            "notes": "Excluded from Sprint 12A ORDER_BOOK baseline pool",
        }
    )
    return pd.DataFrame(rows)


def audit_pass(inventory: pd.DataFrame) -> Tuple[bool, str]:
    ob = inventory.loc[
        inventory["source_table_or_cache"].astype(str).str.contains("order_book_daily")
        & inventory["available"].astype(bool)
    ]
    if ob.empty:
        return False, "No available order_book_daily L2 primitives"
    starts = [s for s in ob["coverage_start"].astype(str) if s]
    ends = [s for s in ob["coverage_end"].astype(str) if s]
    if not starts or min(starts) > "2019-01-02":
        return False, f"order_book coverage start insufficient: {starts[:3]}"
    if not ends or max(ends) < "2026-07-01":
        return False, f"order_book coverage end insufficient: {ends[:3]}"
    # Need core pressure columns
    need = {"obi_5_mean", "weighted_obi_mean", "microprice_deviation_mean", "obi_5_sign_persistence"}
    have = set(ob["primitive"].astype(str))
    missing = sorted(need - have)
    if missing:
        return False, f"Missing core LOB columns: {missing}"
    return True, f"PASS n_l2_native_available={int(ob['available'].sum())}"


# ---------------------------------------------------------------------------
# Candidate registry + discovery
# ---------------------------------------------------------------------------


def build_candidate_registry() -> pd.DataFrame:
    rows = []
    for item in BASELINE:
        name = item["factor"]
        spec = ORDER_BOOK_FACTOR_SPECS[name]
        formula = spec.formula
        rows.append(
            {
                "factor_id": name,
                "formula": formula,
                "formula_hash": _sha256_text(f"{name}|{formula}|order_book_v1"),
                "source_primitive": f"order_book_daily / {formula}",
                "is_l2_native": True,
                "mechanism_bucket": item["mechanism_bucket"],
                "category": spec.category,
                "mechanism": spec.mechanism,
                "lookback_days": spec.lookback_days,
                "signed": spec.signed,
                "expected_redundancy": spec.expected_redundancy or "",
            }
        )
    return pd.DataFrame(rows)


def load_narrow_discovery(factor: str) -> pd.DataFrame:
    path = FACTORS_DIR / factor / "factor_narrow.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    mask = df["tradetime"].between(
        DISCOVERY_START, DISCOVERY_END + pd.Timedelta(hours=23)
    )
    out = df.loc[mask, ["symbol", "tradetime", "factorname", "value"]].copy()
    out["factorname"] = factor
    return out.reset_index(drop=True)


def economic_diagnostics(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
) -> Dict[str, Any]:
    pnl = ensure_effective_group_pnl(group_pnl)
    to = ensure_effective_group_to(group_to, group_pnl).reindex(pnl.index)
    cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
    g1, g10 = cols[0], cols[-1]
    hl = pnl["H-L"].astype(float)
    hl_l1 = to["H-L"].astype(float).reindex(hl.index).fillna(0.0)
    g10_l1 = to[g10].astype(float).reindex(hl.index).fillna(0.0)

    avg_hl_l1 = float(hl_l1.mean())
    avg_hl_ow = l1_to_oneway(avg_hl_l1)
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
    avg_g10_ow = l1_to_oneway(float(g10_l1.mean()))
    g10_fee = float(g10_l1.mean()) * FEE_RATE_L1 * ANNUALIZATION_DAYS
    return {
        "daily_hl_l1_traded_notional": avg_hl_l1,
        "daily_hl_oneway_turnover": avg_hl_ow,
        "annualized_hl_oneway_turnover": avg_hl_ow * ANNUALIZATION_DAYS,
        "fee_annualized_at_7p5bps": fee_annu,
        "approx_net_hl_annual": float(calAnnuRet(hl_net)),
        "approx_net_hl_sharpe": float(calSharpe(hl_net)),
        "G10_daily_oneway_turnover": avg_g10_ow,
        "G10_fee_annualized": g10_fee,
        "G10_net_excess_annual": float(calAnnuRet(g10_net)),
        "G10_gross_excess_annual": float(calAnnuRet(g10_gross)),
        "long_contribution": long_a,
        "short_contribution": short_a,
        "dominant_leg": dominant,  # type: ignore[dict-item]
        "short_leg_share_abs": (
            abs(short_a) / (abs(long_a) + abs(short_a))
            if (abs(long_a) + abs(short_a)) > 0
            else float("nan")
        ),
        "cost_sharpe_retention": (
            float(calSharpe(hl_net)) / float(calSharpe(hl))
            if abs(float(calSharpe(hl))) > 1e-12
            else float("nan")
        ),
        "cost_return_retention": (
            float(calAnnuRet(hl_net)) / float(calAnnuRet(hl))
            if abs(float(calAnnuRet(hl))) > 1e-12
            else float("nan")
        ),
    }


def run_discovery(registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask, ret = load_fast_context("discovery")
    summary_rows = []
    cost_rows = []
    leg_rows = []
    fig_cum = OUT_ROOT / "cumulative_deciles_hl"
    fig_bar = OUT_ROOT / "decile_bar"
    fig_cum.mkdir(parents=True, exist_ok=True)
    fig_bar.mkdir(parents=True, exist_ok=True)

    for _, reg in registry.iterrows():
        name = str(reg["factor_id"])
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        narrow = load_narrow_discovery(name)
        if narrow.empty:
            print("  EMPTY — skip", flush=True)
            continue
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
        # plots: reuse save_fast_plots then copy naming into folders
        tmp = OUT_ROOT / "_tmp_figs" / name
        save_fast_plots(tmp, name, group_pnl, metrics)
        cum_src = tmp / "cumulative_hl.png"
        bar_src = tmp / "decile_bar.png"
        if cum_src.exists():
            (fig_cum / f"{name}.png").write_bytes(cum_src.read_bytes())
        if bar_src.exists():
            (fig_bar / f"{name}.png").write_bytes(bar_src.read_bytes())

        row = {
            "factor_id": name,
            "formula": reg["formula"],
            "formula_hash": reg["formula_hash"],
            "source_primitive": reg["source_primitive"],
            "is_l2_native": True,
            "mechanism_bucket": reg["mechanism_bucket"],
            "gate": gate,
            "rank_ic": metrics["rank_ic_mean_raw"],
            "icir": metrics["icir_raw"],
            "gross_hl_annual": metrics["hl_annu_ret"],
            "gross_hl_sharpe": metrics["hl_sharpe"],
            "gross_hl_mdd": metrics["hl_mdd"],
            "decile_mono": metrics["decile_mono_spearman"],
            "adjacent_violations": metrics["adjacent_violations"],
            "positive_hl_month_fraction": metrics["positive_hl_month_fraction"],
            "G10_gross_excess_annual": econ["G10_gross_excess_annual"],
            "factor_direction": metrics["factor_direction"],
            "n_days": metrics["n_days"],
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
        }
        cost = {"factor_id": name, **{k: v for k, v in econ.items() if k not in (
            "long_contribution", "short_contribution", "dominant_leg", "short_leg_share_abs",
            "G10_gross_excess_annual",
        )}}
        leg = {
            "factor_id": name,
            "long_contribution": econ["long_contribution"],
            "short_contribution": econ["short_contribution"],
            "dominant_leg": econ["dominant_leg"],
            "short_leg_share_abs": econ["short_leg_share_abs"],
            "G10_gross_excess_annual": econ["G10_gross_excess_annual"],
            "G10_net_excess_annual": econ["G10_net_excess_annual"],
            "gross_hl_annual": metrics["hl_annu_ret"],
        }
        # merge econ into summary for ranking
        row.update({
            "daily_hl_oneway_turnover": econ["daily_hl_oneway_turnover"],
            "fee_annualized_at_7p5bps": econ["fee_annualized_at_7p5bps"],
            "approx_net_hl_annual": econ["approx_net_hl_annual"],
            "approx_net_hl_sharpe": econ["approx_net_hl_sharpe"],
            "G10_net_excess_annual": econ["G10_net_excess_annual"],
            "dominant_leg": econ["dominant_leg"],
            "cost_sharpe_retention": econ["cost_sharpe_retention"],
            "short_leg_share_abs": econ["short_leg_share_abs"],
        })
        summary_rows.append(row)
        cost_rows.append(cost)
        leg_rows.append(leg)
        print(
            f"  gate={gate} Sharpe={metrics['hl_sharpe']:.2f} mono={metrics['decile_mono_spearman']:.3f} "
            f"viol={metrics['adjacent_violations']} netS≈{econ['approx_net_hl_sharpe']:.2f} "
            f"TO_ow={econ['daily_hl_oneway_turnover']:.3f} leg={econ['dominant_leg']}",
            flush=True,
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(cost_rows), pd.DataFrame(leg_rows)


# ---------------------------------------------------------------------------
# Redundancy
# ---------------------------------------------------------------------------


def load_wide_discovery(factor: str) -> pd.DataFrame:
    df = load_narrow_discovery(factor)
    df["TradeDate"] = df["tradetime"].dt.normalize()
    wide = df.pivot_table(
        index="TradeDate", columns="symbol", values="value", aggfunc="last"
    )
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def mean_daily_spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    idx = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    if len(idx) == 0 or len(cols) == 0:
        return float("nan")
    rhos = a.loc[idx, cols].corrwith(b.loc[idx, cols], axis=1, method="spearman")
    return float(rhos.mean())


def build_correlation_and_clusters(
    factors: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wides = {}
    for f in factors:
        print(f"  [corr] load {f}", flush=True)
        wides[f] = load_wide_discovery(f)
    pairs = []
    for i, left in enumerate(factors):
        for right in factors[i + 1 :]:
            rho = mean_daily_spearman(wides[left], wides[right])
            pairs.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "mean_daily_spearman": rho,
                    "abs_mean_daily_spearman": abs(rho) if np.isfinite(rho) else np.nan,
                }
            )
    corr = pd.DataFrame(pairs)
    # union-find clusters for |rho|>=0.90
    parent = {f: f for f in factors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in corr.iterrows():
        if np.isfinite(r["abs_mean_daily_spearman"]) and r["abs_mean_daily_spearman"] >= NEAR_ALIAS_RHO:
            union(str(r["factor_left"]), str(r["factor_right"]))

    clusters: Dict[str, List[str]] = {}
    for f in factors:
        clusters.setdefault(find(f), []).append(f)

    # representative = max |gross sharpe| from discovery_summary later; placeholder by name
    cluster_rows = []
    for cid, members in clusters.items():
        members = sorted(members)
        cluster_rows.append(
            {
                "cluster_id": cid,
                "members": "|".join(members),
                "cluster_size": len(members),
                "representative": members[0],  # filled later
            }
        )
    return corr, pd.DataFrame(cluster_rows)


def assign_cluster_representatives(
    clusters: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    s = summary.set_index("factor_id")
    rows = []
    for _, r in clusters.iterrows():
        members = str(r["members"]).split("|")
        scored = []
        for m in members:
            if m not in s.index:
                continue
            row = s.loc[m]
            scored.append(
                (
                    m,
                    float(row["gross_hl_sharpe"]),
                    float(row.get("approx_net_hl_sharpe", np.nan)),
                    float(row.get("decile_mono", np.nan)),
                )
            )
        if not scored:
            rep = members[0]
        else:
            # prefer gate quality then net then gross
            def key(t):
                m = t[0]
                gate = str(s.loc[m, "gate"]) if m in s.index else "none"
                gate_rank = 0 if gate == "strong_candidate" else (1 if gate == "research_candidate" else 2)
                return (gate_rank, -t[2] if np.isfinite(t[2]) else 0.0, -t[1], -t[3] if np.isfinite(t[3]) else 0.0)

            scored.sort(key=key)
            rep = scored[0][0]
        rows.append(
            {
                "cluster_id": r["cluster_id"],
                "members": r["members"],
                "cluster_size": r["cluster_size"],
                "representative": rep,
                "is_singleton": int(r["cluster_size"]) == 1,
            }
        )
    # expand per-factor view
    out = []
    for r in rows:
        for m in str(r["members"]).split("|"):
            out.append(
                {
                    "factor_id": m,
                    "cluster_id": r["cluster_id"],
                    "cluster_members": r["members"],
                    "cluster_size": r["cluster_size"],
                    "representative": r["representative"],
                    "is_representative": m == r["representative"],
                }
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Ranking / selection
# ---------------------------------------------------------------------------


def select_next_fv(summary: pd.DataFrame, clusters: pd.DataFrame) -> Dict[str, Any]:
    reps = set(clusters.loc[clusters["is_representative"], "factor_id"])
    pool = summary.loc[
        summary["factor_id"].isin(reps)
        & summary["gate"].isin(["strong_candidate", "research_candidate"])
    ].copy()
    if pool.empty:
        # fall back: any strong regardless
        pool = summary.loc[summary["gate"] == "strong_candidate"].copy()
    if pool.empty:
        return {
            "status": "SPRINT12A_NO_STRONG_CANDIDATE",
            "next_candidate": None,
            "proceed_to_12b": False,
            "reason": "No Fast Gate strong/research survivors among independent reps",
        }

    pool = pool.assign(
        short_penalty=pool["short_leg_share_abs"].fillna(1.0),
        net_s=pool["approx_net_hl_sharpe"].fillna(-9.0),
        g10_net=pool["G10_net_excess_annual"].fillna(-9.0),
        to_ow=pool["daily_hl_oneway_turnover"].fillna(9.0),
        gate_rank=pool["gate"].map(
            {"strong_candidate": 0, "research_candidate": 1}
        ).fillna(2),
        long_bonus=(pool["dominant_leg"] == "LONG").astype(int),
    )
    pool = pool.sort_values(
        by=[
            "gate_rank",
            "net_s",
            "g10_net",
            "gross_hl_sharpe",
            "decile_mono",
            "to_ow",
            "short_penalty",
        ],
        ascending=[True, False, False, False, False, True, True],
    )
    best = pool.iloc[0]
    # proceed to 12B only if strong AND approx net not catastrophic
    proceed = (
        str(best["gate"]) == "strong_candidate"
        and float(best["approx_net_hl_sharpe"]) >= 0.5
        and float(best["G10_net_excess_annual"]) > -0.05
    )
    return {
        "status": "HAS_CANDIDATE" if proceed or str(best["gate"]) == "strong_candidate" else "RESEARCH_ONLY",
        "next_candidate": str(best["factor_id"]),
        "proceed_to_12b": bool(proceed),
        "gate": str(best["gate"]),
        "gross_hl_sharpe": float(best["gross_hl_sharpe"]),
        "approx_net_hl_sharpe": float(best["approx_net_hl_sharpe"]),
        "G10_net_excess_annual": float(best["G10_net_excess_annual"]),
        "dominant_leg": str(best["dominant_leg"]),
        "daily_hl_oneway_turnover": float(best["daily_hl_oneway_turnover"]),
        "reason": (
            "Fast Gate strong + non-catastrophic net/G10 diagnostics"
            if proceed
            else "Top ranked Fast Gate survivor; economic diagnostics weak — human decision for 12B"
        ),
    }


def render_report(
    inventory: pd.DataFrame,
    registry: pd.DataFrame,
    summary: pd.DataFrame,
    clusters: pd.DataFrame,
    selection: Dict[str, Any],
    audit_msg: str,
) -> str:
    n_prim = int(
        inventory.loc[
            inventory["is_l2_native"].astype(bool) & inventory["available"].astype(bool)
        ].shape[0]
    )
    n_tested = len(summary)
    n_strong = int((summary["gate"] == "strong_candidate").sum())
    n_research = int((summary["gate"] == "research_candidate").sum())
    n_rep = int(clusters["is_representative"].sum()) if not clusters.empty else 0

    gross_only = summary.loc[
        (summary["gross_hl_sharpe"] >= 2.0)
        & (summary["approx_net_hl_sharpe"] < 0.5)
    ]
    short_dom = summary.loc[summary["dominant_leg"] == "SHORT"]
    best_g10 = summary.sort_values("G10_net_excess_annual", ascending=False).iloc[0] if not summary.empty else None
    best_net = summary.sort_values("approx_net_hl_sharpe", ascending=False).iloc[0] if not summary.empty else None
    aliases = clusters.loc[~clusters["is_representative"]] if not clusters.empty else pd.DataFrame()

    lines = [
        "# Sprint 12A — ORDER_BOOK / LOB_PRESSURE Baseline Discovery",
        "",
        "Baseline discovery only. No Full Validation. Protocol v2.0 untouched.",
        f"Discovery window: `{DISCOVERY_START.date()}` ~ `{DISCOVERY_END.date()}`",
        f"Primitive audit: **{audit_msg}**",
        "",
        f"- Genuine L2 order-book primitives available (inventory): **{n_prim}**",
        f"- Baseline candidates tested: **{n_tested}**",
        f"- Independent cluster representatives: **{n_rep}**",
        f"- Fast Gate STRONG: **{n_strong}**; RESEARCH: **{n_research}**",
        f"- Selection: `{selection.get('status')}` → next=`{selection.get('next_candidate')}`",
        f"- Proceed Sprint 12B? **{selection.get('proceed_to_12b')}** — {selection.get('reason')}",
        "",
        "## Discovery summary",
        "",
        summary.sort_values("gross_hl_sharpe", ascending=False).to_string(index=False),
        "",
        "## FINAL QUESTIONS",
        "",
        f"1. Genuine L2 order-book primitives available? **{n_prim}** "
        f"(plus marked MISSING queue_* primitives).",
        f"2. Baseline candidates tested? **{n_tested}**.",
        f"3. Pass existing Fast Gate (STRONG)? **{n_strong}** "
        f"(RESEARCH={n_research}).",
        f"4. Strong gross but weak economics (grossS≥2 & approx_netS<0.5)? **"
        + (", ".join(gross_only["factor_id"].astype(str)) if len(gross_only) else "none")
        + "**.",
        f"5. Primarily short-leg? **"
        + (", ".join(short_dom["factor_id"].astype(str)) if len(short_dom) else "none")
        + "**.",
        f"6. Strongest G10 net excess? **"
        + (
            f"{best_g10['factor_id']} ({_fmt_pct(float(best_g10['G10_net_excess_annual']))})"
            if best_g10 is not None
            else "n/a"
        )
        + "**.",
        f"7. Best approx net H-L Sharpe? **"
        + (
            f"{best_net['factor_id']} ({_fmt_num(float(best_net['approx_net_hl_sharpe']), 2)})"
            if best_net is not None
            else "n/a"
        )
        + "**.",
        f"8. Aliases / non-representatives? **"
        + (
            ", ".join(aliases["factor_id"].astype(str))
            if len(aliases)
            else "none beyond singletons"
        )
        + "**.",
        f"9. ONE representative for next Full Validation? **"
        + (
            f"{selection.get('next_candidate')} "
            f"(Fast Gate STRONG; ECONOMICALLY WEAK — do not auto-FV; "
            f"approx_netS={_fmt_num(float(selection.get('approx_net_hl_sharpe', float('nan'))), 2)}, "
            f"G10_net={_fmt_pct(float(selection.get('G10_net_excess_annual', float('nan'))))}, "
            f"leg={selection.get('dominant_leg')})"
            if selection.get("next_candidate")
            else "SPRINT12A_NO_STRONG_CANDIDATE"
        )
        + "**.",
        f"10. Enough evidence for Sprint 12B FV? **"
        + ("YES" if selection.get("proceed_to_12b") else "NO")
        + f"** — {selection.get('reason')}.",
        "",
        "## STOP",
        "",
        "Do not auto Full Validate. Do not optimize the winner. Do not start another family.",
        "",
    ]
    return "\n".join(lines)


def main() -> Dict[str, Any]:
    t0 = time.perf_counter()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[A] close Sprint 11 ledger", flush=True)
    update_sprint11_ledger()

    print("[C] primitive inventory audit", flush=True)
    inventory = build_primitive_inventory()
    inventory.to_csv(OUT_ROOT / "primitive_inventory.csv", index=False)
    ok, audit_msg = audit_pass(inventory)
    print(f"  {audit_msg}", flush=True)
    if not ok:
        report = (
            "# Sprint 12A STOP — LOB primitives insufficient\n\n"
            f"{audit_msg}\n\nDo NOT replace with OHLCV proxies.\n"
        )
        (OUT_ROOT / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "sprint": "Sprint 12A",
            "status": "STOPPED_MISSING_PRIMITIVES",
            "audit": audit_msg,
        }
        (OUT_ROOT / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print("STOP — missing primitives", flush=True)
        return manifest

    print("[D] candidate registry (frozen existing formulas only)", flush=True)
    registry = build_candidate_registry()
    registry.to_csv(OUT_ROOT / "candidate_registry.csv", index=False)
    print(f"  n={len(registry)}", flush=True)

    print("[E/F/G] discovery + economic diagnostics", flush=True)
    summary, cost_df, leg_df = run_discovery(registry)
    summary.to_csv(OUT_ROOT / "discovery_summary.csv", index=False)
    cost_df.to_csv(OUT_ROOT / "cost_diagnostics.csv", index=False)
    leg_df.to_csv(OUT_ROOT / "leg_diagnostics.csv", index=False)

    print("[H] correlation / clusters", flush=True)
    factors = summary["factor_id"].astype(str).tolist()
    corr, cluster_seed = build_correlation_and_clusters(factors)
    corr.to_csv(OUT_ROOT / "candidate_correlation.csv", index=False)
    clusters = assign_cluster_representatives(cluster_seed, summary)
    clusters.to_csv(OUT_ROOT / "candidate_clusters.csv", index=False)

    print("[I] ranking / selection", flush=True)
    selection = select_next_fv(summary, clusters)
    print(f"  {selection}", flush=True)

    report = render_report(inventory, registry, summary, clusters, selection, audit_msg)
    (OUT_ROOT / "report.md").write_text(report, encoding="utf-8")

    # cleanup tmp figs
    tmp = OUT_ROOT / "_tmp_figs"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    manifest = {
        "sprint": "Sprint 12A — ORDER_BOOK / LOB_PRESSURE Baseline Discovery",
        "status": "COMPLETE",
        "discovery_window": ["2023-01-01", "2024-12-31"],
        "primitive_audit": audit_msg,
        "n_baseline_candidates": int(len(registry)),
        "n_tested": int(len(summary)),
        "n_strong_gate": int((summary["gate"] == "strong_candidate").sum()),
        "n_research_gate": int((summary["gate"] == "research_candidate").sum()),
        "selection": selection,
        "protocol_ref": "evaluation_protocol_v2.0 (untouched)",
        "full_validation_auto": False,
        "new_formulas": False,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "outputs": sorted(p.name for p in OUT_ROOT.rglob("*") if p.is_file()),
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\n[done] -> {OUT_ROOT} ({manifest['elapsed_seconds']}s)", flush=True)
    return manifest


if __name__ == "__main__":
    main()
