"""Sprint 11 — Price Formation / Intraday Path Family v1.

FIRST inventory, THEN discovery on frozen Fast Lane (2023-2024).
No new formulas / primitives / parameter search / Protocol edits.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from Factor_Dev_Lib import calAnnuRet, calSharpe
from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.evaluation_protocol_v2 import l1_to_oneway
from l2_factor_reproduction.python.fast_discovery import (
    DISCOVERY_END,
    DISCOVERY_START,
    compute_fast_metrics,
    ensure_effective_group_pnl,
    gate_label,
    load_fast_context,
    save_fast_plots,
)
from l2_factor_reproduction.python.price_formation_factors import (
    PRICE_FORMATION_FACTOR_SPECS,
)

FAMILY_ROOT = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "price_formation_family"
)
FACTORS_DIR = FAMILY_ROOT / "factors"
OUT_ROOT = Path(RESULT_ROOT) / "fast_discovery" / "price_formation_v1"

NEAR_ALIAS_ABS = 0.90

# Mechanism taxonomy (Sprint 11)
MECHANISM_BUCKET: Dict[str, str] = {
    "overnight_gap": "A_price_discovery_timing",
    "open_to_30m_return": "A_price_discovery_timing",
    "morning_return": "A_price_discovery_timing",
    "afternoon_return": "A_price_discovery_timing",
    "closing_30m_return": "E_early_vs_late_session",
    "lunch_gap_return": "E_early_vs_late_session",
    "close_auction_return": "F_auction_close_effects",
    "vwap_close_deviation": "D_vwap_close_price_formation",
    "close_location_value": "D_vwap_close_price_formation",
    "path_efficiency": "B_intraday_path_efficiency",
    "intraday_return_sign_persistence": "B_intraday_path_efficiency",
    "minute_return_autocorr1": "B_intraday_path_efficiency",
    "variance_ratio_5m": "B_intraday_path_efficiency",
    "realized_volatility": "C_intraday_reversal_excursion",
    "downside_semivariance_share": "C_intraday_reversal_excursion",
    "realized_skewness": "C_intraday_reversal_excursion",
    "realized_kurtosis": "C_intraday_reversal_excursion",
    "jump_share": "C_intraday_reversal_excursion",
    "max_abs_minute_return": "C_intraday_reversal_excursion",
    "tail_return_share": "C_intraday_reversal_excursion",
    "intraday_max_drawdown": "C_intraday_reversal_excursion",
    "intraday_max_drawup": "C_intraday_reversal_excursion",
    "opening_amount_share": "E_early_vs_late_session",
    "closing_amount_share": "E_early_vs_late_session",
    "morning_afternoon_amount_imbalance": "E_early_vs_late_session",
    "volume_concentration_hhi": "G_other",
    "amount_time_center": "E_early_vs_late_session",
    "volume_return_corr": "G_other",
    "volume_abs_return_corr": "G_other",
    "intraday_amihud": "G_other",
    "return_per_amount": "ORDINARY_OHLCV_LOW_PRIORITY",
    "range_per_amount": "ORDINARY_OHLCV_LOW_PRIORITY",
}

ORDINARY = {
    "overnight_gap",
    "open_to_30m_return",
    "morning_return",
    "afternoon_return",
    "realized_volatility",
    "intraday_max_drawdown",
    "intraday_max_drawup",
    "return_per_amount",
    "range_per_amount",
    "close_location_value",  # classic CLV from OHLC
}


def _load_summary() -> pd.DataFrame:
    return pd.read_csv(FAMILY_ROOT / "candidate_summary.csv")


def _load_registry() -> pd.DataFrame:
    return pd.read_csv(FAMILY_ROOT / "factor_registry.csv")


def build_inventory() -> pd.DataFrame:
    reg = _load_registry()
    summary = _load_summary().set_index("factor")
    rows = []
    for _, r in reg.iterrows():
        name = str(r["name"])
        narrow = FACTORS_DIR / name / "factor_narrow.parquet"
        summary_json = FACTORS_DIR / name / "summary.json"
        computed = narrow.exists() and summary_json.exists()
        s = summary.loc[name] if name in summary.index else None
        rows.append(
            {
                "factor": name,
                "exact_formula": r["formula"],
                "mechanism": r["mechanism"],
                "category": r["category"],
                "source_field": "price_formation_daily." + name,
                "available": True,
                "already_computed": computed,
                "already_fast_tested": False,  # none on frozen 2023-2024 Fast Lane
                "test_protocol_version": "candidate_pool_v1_full_sample",
                "sample_start": "2019-01-01" if s is not None else "",
                "sample_end": "2026-07-31" if s is not None else "",
                "existing_hl_sharpe": float(s["hl_sharpe"]) if s is not None else np.nan,
                "existing_mono": float(s["decile_mono_spearman"])
                if s is not None
                else np.nan,
                "existing_violations": np.nan,  # not in full-sample schema
                "existing_g10_excess": float(s["g10_excess_annu_ret"])
                if s is not None
                else np.nan,
                "existing_g10_excess_sharpe": float(s["g10_excess_sharpe"])
                if s is not None
                else np.nan,
                "needs_rerun_on_frozen_fast_lane": True,
                "notes": (
                    "full-sample baseline only; not validated on Protocol v2.0 / "
                    "Fast Discovery 2023-2024 window"
                ),
            }
        )
    return pd.DataFrame(rows)


def _mean_daily_spearman(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> float:
    """left/right: TradeDate x symbol wide panels aligned."""
    common_idx = left.index.intersection(right.index)
    common_cols = left.columns.intersection(right.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        return float("nan")
    a = left.loc[common_idx, common_cols]
    b = right.loc[common_idx, common_cols]
    rhos = a.corrwith(b, axis=1, method="spearman")
    return float(rhos.mean())


def load_factor_wide_discovery(factor: str) -> pd.DataFrame:
    path = FACTORS_DIR / factor / "factor_narrow.parquet"
    df = pd.read_parquet(path)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    df["TradeDate"] = df["tradetime"].dt.normalize()
    df = df.loc[df["TradeDate"].between(DISCOVERY_START, DISCOVERY_END)]
    wide = df.pivot_table(
        index="TradeDate", columns="symbol", values="value", aggfunc="last"
    )
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def build_redundancy(inventory: pd.DataFrame) -> pd.DataFrame:
    """Exact/sign/algebraic + discovery-window |Spearman|>=0.90 near-aliases."""
    pairs_path = FAMILY_ROOT / "high_corr_pairs.csv"
    full_pairs = pd.read_csv(pairs_path)
    # Candidate near-alias pairs from full sample |rho|>=0.85 — verify on discovery
    cand_pairs = full_pairs.loc[
        full_pairs["abs_mean_daily_spearman"] >= 0.85,
        ["factor_left", "factor_right", "mean_daily_spearman", "abs_mean_daily_spearman"],
    ].copy()

    rows = []
    # Always record exact known sign/algebraic from full high-corr
    wide_cache: Dict[str, pd.DataFrame] = {}

    def get_wide(name: str) -> pd.DataFrame:
        if name not in wide_cache:
            print(f"  [corr] load wide {name}", flush=True)
            wide_cache[name] = load_factor_wide_discovery(name)
        return wide_cache[name]

    for _, p in cand_pairs.iterrows():
        a, b = str(p["factor_left"]), str(p["factor_right"])
        disc_rho = _mean_daily_spearman(get_wide(a), get_wide(b))
        abs_disc = abs(disc_rho) if np.isfinite(disc_rho) else float("nan")
        alias_type = (
            "near_alias_discovery"
            if abs_disc >= NEAR_ALIAS_ABS
            else "high_corr_full_not_discovery_alias"
        )
        if abs_disc >= NEAR_ALIAS_ABS and disc_rho < 0:
            alias_type = "sign_or_near_alias_discovery"
        rows.append(
            {
                "factor_a": a,
                "factor_b": b,
                "full_sample_spearman": float(p["mean_daily_spearman"]),
                "discovery_spearman": disc_rho,
                "alias_type": alias_type,
                "is_near_alias": bool(abs_disc >= NEAR_ALIAS_ABS),
            }
        )

    # Singleton notes for all factors
    red = pd.DataFrame(rows)
    # Build clusters from near-aliases
    parent: Dict[str, str] = {f: f for f in inventory["factor"]}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in red.loc[red["is_near_alias"]].iterrows():
        union(str(r["factor_a"]), str(r["factor_b"]))

    summary = _load_summary().set_index("factor")
    cluster_map: Dict[str, List[str]] = {}
    for f in inventory["factor"]:
        root = find(str(f))
        cluster_map.setdefault(root, []).append(str(f))

    out_rows = []
    for root, members in cluster_map.items():
        members = sorted(members)
        # representative = highest |existing hl_sharpe|
        scores = {
            m: abs(float(summary.loc[m, "hl_sharpe"]))
            if m in summary.index
            else 0.0
            for m in members
        }
        rep = max(members, key=lambda m: scores[m])
        for m in members:
            pair_note = ""
            if len(members) > 1:
                sub = red.loc[
                    (
                        ((red.factor_a == m) & (red.factor_b.isin(members)))
                        | ((red.factor_b == m) & (red.factor_a.isin(members)))
                    )
                    & red.is_near_alias
                ]
                if not sub.empty:
                    pair_note = "; ".join(
                        f"{r.factor_a}/{r.factor_b}:{r.discovery_spearman:.3f}"
                        for r in sub.itertuples()
                    )
            out_rows.append(
                {
                    "factor": m,
                    "cluster_id": root,
                    "cluster_members": "|".join(members),
                    "cluster_size": len(members),
                    "is_representative": m == rep,
                    "representative": rep,
                    "alias_detail": pair_note,
                    "reason": (
                        "singleton"
                        if len(members) == 1
                        else f"discovery_|Spearman|>={NEAR_ALIAS_ABS} cluster; rep=max|full_hl_sharpe|"
                    ),
                }
            )
    # Attach pairwise table as separate export; return cluster table
    red.to_csv(OUT_ROOT / "within_family_redundancy_pairs.csv", index=False)
    return pd.DataFrame(out_rows)


def build_taxonomy(inventory: pd.DataFrame, redundancy: pd.DataFrame) -> pd.DataFrame:
    rep = redundancy.set_index("factor")
    rows = []
    for _, r in inventory.iterrows():
        name = str(r["factor"])
        bucket = MECHANISM_BUCKET.get(name, "G_other")
        ordinary = name in ORDINARY or bucket == "ORDINARY_OHLCV_LOW_PRIORITY"
        rows.append(
            {
                "factor": name,
                "mechanism_bucket": bucket
                if not ordinary
                else "ORDINARY_OHLCV_LOW_PRIORITY",
                "ORDINARY_OHLCV_LOW_PRIORITY": ordinary,
                "is_representative": bool(rep.loc[name, "is_representative"])
                if name in rep.index
                else True,
                "representative": rep.loc[name, "representative"]
                if name in rep.index
                else name,
                "registry_mechanism": r["mechanism"],
            }
        )
    return pd.DataFrame(rows)


def contribution_stats(group_pnl: pd.DataFrame) -> Dict[str, float]:
    pnl = ensure_effective_group_pnl(group_pnl)
    cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
    g1, g10 = cols[0], cols[-1]
    long_c = pnl[g10]
    short_c = -pnl[g1]
    return {
        "long_contribution_annual": float(calAnnuRet(long_c)),
        "short_contribution_annual": float(calAnnuRet(short_c)),
        "g10_gross_excess_annual": float(calAnnuRet(pnl[g10])),
        "g1_gross_excess_annual": float(calAnnuRet(pnl[g1])),
        "short_leg_share_abs": (
            abs(float(calAnnuRet(short_c)))
            / (
                abs(float(calAnnuRet(long_c)))
                + abs(float(calAnnuRet(short_c)))
            )
            if (
                abs(float(calAnnuRet(long_c)))
                + abs(float(calAnnuRet(short_c)))
            )
            > 0
            else float("nan")
        ),
    }


def load_narrow_discovery(factor: str) -> pd.DataFrame:
    path = FACTORS_DIR / factor / "factor_narrow.parquet"
    df = pd.read_parquet(path)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    mask = df["tradetime"].between(
        DISCOVERY_START, DISCOVERY_END + pd.Timedelta(hours=23)
    )
    out = df.loc[mask, ["symbol", "tradetime", "factorname", "value"]].copy()
    return out.reset_index(drop=True)


def run_fast_discovery(
    representatives: Sequence[str],
    taxonomy: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask, ret = load_fast_context("discovery")
    tax = taxonomy.set_index("factor")
    summary_rows = []
    contrib_rows = []
    for name in representatives:
        print(f"\n=== {name} ===", flush=True)
        t1 = time.perf_counter()
        narrow = load_narrow_discovery(name)
        if narrow.empty:
            print("  EMPTY narrow — skip", flush=True)
            continue
        group_pnl, group_to, _ic, summary = backtest_factor(
            narrow,
            start_day=DISCOVERY_START,
            end_day=DISCOVERY_END,
            mask=mask,
            ret_matrix=ret,
        )
        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        contrib = contribution_stats(group_pnl)
        pnl = ensure_effective_group_pnl(group_pnl)
        to = group_to.copy()
        to.index = pd.to_datetime(to.index)
        to.columns = [str(c) for c in to.columns]
        cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
        g10 = cols[-1]
        l1_hl = float(to["H-L"].reindex(pnl.index).mean())
        l1_g10 = float(to[g10].reindex(pnl.index).mean())
        gate = gate_label(metrics)
        ordinary = bool(tax.loc[name, "ORDINARY_OHLCV_LOW_PRIORITY"]) if name in tax.index else False
        save_fast_plots(OUT_ROOT / "figures" / name, name, group_pnl, metrics)
        row = {
            "factor": name,
            "mechanism_bucket": tax.loc[name, "mechanism_bucket"]
            if name in tax.index
            else "",
            "ORDINARY_OHLCV_LOW_PRIORITY": ordinary,
            "gate": gate,
            "rank_ic_mean_raw": metrics["rank_ic_mean_raw"],
            "icir_raw": metrics["icir_raw"],
            "hl_annu_ret": metrics["hl_annu_ret"],
            "hl_sharpe": metrics["hl_sharpe"],
            "decile_mono_spearman": metrics["decile_mono_spearman"],
            "adjacent_violations": metrics["adjacent_violations"],
            "positive_hl_month_fraction": metrics["positive_hl_month_fraction"],
            "cum_hl_time_spearman": metrics["cum_hl_time_spearman"],
            "g10_gross_excess_annual": contrib["g10_gross_excess_annual"],
            "g10_excess_sharpe": metrics["g10_excess_sharpe"],
            "g1_gross_excess_annual": contrib["g1_gross_excess_annual"],
            "long_contribution_annual": contrib["long_contribution_annual"],
            "short_contribution_annual": contrib["short_contribution_annual"],
            "short_leg_share_abs": contrib["short_leg_share_abs"],
            "avg_g10_oneway_turnover": l1_to_oneway(l1_g10),
            "avg_hl_oneway_turnover": l1_to_oneway(l1_hl),
            "factor_direction": metrics["factor_direction"],
            "n_days": metrics["n_days"],
            "elapsed_seconds": round(time.perf_counter() - t1, 2),
        }
        summary_rows.append(row)
        contrib_rows.append({"factor": name, **contrib, "hl_sharpe": metrics["hl_sharpe"]})
        print(
            f"  gate={gate} Sharpe={metrics['hl_sharpe']:.2f} "
            f"mono={metrics['decile_mono_spearman']:.3f} "
            f"viol={metrics['adjacent_violations']} "
            f"G10={contrib['g10_gross_excess_annual']:.2%} "
            f"ordinary={ordinary}",
            flush=True,
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrib_rows)


def select_next(summary: pd.DataFrame) -> Dict[str, object]:
    strong = summary.loc[summary["gate"] == "strong_candidate"].copy()
    if strong.empty:
        return {
            "status": "SPRINT11_NO_STRONG_EXISTING_CANDIDATE",
            "factor": None,
            "next_md": "",
        }
    # Prefer non-ordinary, then G10 excess, lower short share, lower TO
    strong = strong.assign(
        ordinary_rank=strong["ORDINARY_OHLCV_LOW_PRIORITY"].astype(int)
    )
    strong = strong.sort_values(
        by=[
            "ordinary_rank",
            "g10_gross_excess_annual",
            "short_leg_share_abs",
            "avg_hl_oneway_turnover",
        ],
        ascending=[True, False, True, True],
    )
    best = strong.iloc[0]
    fid = str(best["factor"])
    md = "\n".join(
        [
            "# Next Full Validation Candidate — Sprint 11",
            "",
            f"**factor:** `{fid}`",
            "",
            "Awaiting human confirmation. Do **not** auto Full Validate.",
            "Do **not** invent price-formation v2 formulas yet.",
            "",
            f"- gate=`{best['gate']}`",
            f"- mechanism_bucket=`{best['mechanism_bucket']}`",
            f"- ORDINARY_OHLCV=`{best['ORDINARY_OHLCV_LOW_PRIORITY']}`",
            f"- H-L Sharpe=`{best['hl_sharpe']:.3f}` mono=`{best['decile_mono_spearman']:.3f}` "
            f"viol=`{int(best['adjacent_violations'])}`",
            f"- G10 excess annual=`{best['g10_gross_excess_annual']:.2%}`",
            f"- short_leg_share=`{best['short_leg_share_abs']:.3f}`",
            f"- H-L one-way TO=`{best['avg_hl_oneway_turnover']:.3f}`",
            "",
            "## Selection order applied",
            "1. STRONG gate",
            "2. Prefer non-ORDINARY_OHLCV",
            "3. Higher G10 excess",
            "4. Lower short-leg dependence",
            "5. Lower one-way turnover",
            "",
        ]
    ) + "\n"
    return {"status": "HAS_STRONG", "factor": fid, "next_md": md}


def render_report(
    inventory: pd.DataFrame,
    redundancy: pd.DataFrame,
    taxonomy: pd.DataFrame,
    summary: pd.DataFrame,
    selection: Dict[str, object],
) -> str:
    n_total = len(inventory)
    n_rep = int(redundancy["is_representative"].sum())
    n_tested = len(summary)
    strong = summary.loc[summary["gate"] == "strong_candidate"]
    best_g10 = (
        summary.sort_values("g10_gross_excess_annual", ascending=False).iloc[0]
        if not summary.empty
        else None
    )
    # Previously strong on full sample?
    prev_strongish = inventory.loc[inventory["existing_hl_sharpe"] >= 3.0]
    # Any of those that become strong on discovery?
    recovered = []
    if not strong.empty:
        recovered = [
            f
            for f in strong["factor"]
            if f in set(prev_strongish["factor"])
        ]

    # Q6: leg mix for selected next candidate (fallback: best STRONG / top H-L)
    if selection.get("status") == "HAS_STRONG" and selection.get("factor") in set(summary["factor"]):
        top = summary.loc[summary["factor"] == selection["factor"]].iloc[0]
        leg_scope = f"selected `{selection['factor']}`"
    elif not strong.empty:
        top = strong.sort_values("g10_gross_excess_annual", ascending=False).iloc[0]
        leg_scope = f"STRONG `{top['factor']}`"
    else:
        top = summary.sort_values("hl_sharpe", ascending=False).iloc[0]
        leg_scope = f"top H-L `{top['factor']}`"
    short_share = float(top["short_leg_share_abs"])
    long_a = float(top["long_contribution_annual"])
    short_a = float(top["short_contribution_annual"])
    if abs(short_a) > abs(long_a) * 1.25:
        leg = f"{leg_scope}: primarily short (short={short_a:.2%}, long={long_a:.2%})"
    elif abs(long_a) > abs(short_a) * 1.25:
        leg = f"{leg_scope}: primarily long (long={long_a:.2%}, short={short_a:.2%})"
    else:
        leg = (
            f"{leg_scope}: mixed (long={long_a:.2%}, short={short_a:.2%}, "
            f"short_share={short_share:.2f})"
        )

    auction = summary.loc[summary["factor"] == "close_auction_return"]
    endpoint_note = "n/a"
    if not auction.empty:
        a = auction.iloc[0]
        endpoint_note = (
            f"YES — close_auction_return discovery Sharpe={a['hl_sharpe']:.2f}, "
            f"mono={a['decile_mono_spearman']:.3f}, viol={int(a['adjacent_violations'])} "
            "(fails STRONG on violations); treat as auction/endpoint artifact risk."
        )

    q3 = (
        "YES — none had frozen Fast Lane tests before; full-sample strongish names "
        "re-checked on 2023-2024: "
        + (", ".join(recovered) if recovered else "none retained STRONG")
        + f" (full-sample |hl_sharpe|>=3 count={len(prev_strongish)})"
    )

    best_g10_strong = (
        strong.sort_values("g10_gross_excess_annual", ascending=False).iloc[0]
        if not strong.empty
        else None
    )
    q5 = "n/a"
    if best_g10 is not None:
        q5 = (
            f"overall `{best_g10['factor']}` ({best_g10['g10_gross_excess_annual']:.2%}, "
            f"gate={best_g10['gate']})"
        )
        if best_g10_strong is not None:
            q5 += (
                f"; among STRONG `{best_g10_strong['factor']}` "
                f"({best_g10_strong['g10_gross_excess_annual']:.2%})"
            )

    lines = [
        "# Sprint 11 — Price Formation / Intraday Path Family v1",
        "",
        "Inventory-first. No new formulas. Protocol v2.0 untouched.",
        f"Discovery window: `{DISCOVERY_START.date()}` ~ `{DISCOVERY_END.date()}`",
        "",
        f"- Registry candidates: **{n_total}**",
        f"- Independent representatives tested: **{n_rep}** (tested={n_tested})",
        f"- Selection: **{selection['status']}**",
        "",
        "## Candidate summary (Fast Lane)",
        "",
        summary.to_string(index=False),
        "",
        "## Report questions",
        "",
        f"1. How many price_formation candidates exist? **{n_total}**",
        f"2. How many independent representatives? **{n_rep}**",
        f"3. Previously-tested strong overlooked by old protocol? **{q3}**",
        f"4. STRONG under Protocol v2.0 Fast Gate? **"
        f"{'YES: ' + ', '.join(strong['factor'].astype(str)) if not strong.empty else 'NO'}**",
        f"5. Best G10 excess? **{q5}**",
        f"6. Alpha long vs short? **{leg}**",
        f"7. Endpoint / auction artifact? **{endpoint_note}**",
        f"8. Next Full Validation? **"
        f"{selection['factor'] if selection['status']=='HAS_STRONG' else selection['status']}**",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_sprint11() -> Dict[str, object]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("[A] existing candidate inventory", flush=True)
    inventory = build_inventory()
    inventory.to_csv(OUT_ROOT / "existing_candidate_inventory.csv", index=False)
    print(f"  n={len(inventory)} already_fast_tested=0", flush=True)

    print("[B] within-family redundancy (discovery |rho|>=0.90)", flush=True)
    redundancy = build_redundancy(inventory)
    redundancy.to_csv(OUT_ROOT / "within_family_redundancy.csv", index=False)
    n_rep = int(redundancy["is_representative"].sum())
    print(f"  representatives={n_rep}", flush=True)

    print("[C] mechanism taxonomy", flush=True)
    taxonomy = build_taxonomy(inventory, redundancy)
    taxonomy.to_csv(OUT_ROOT / "mechanism_taxonomy.csv", index=False)

    reps = redundancy.loc[redundancy["is_representative"], "factor"].tolist()
    print(f"[D] Fast Discovery on {len(reps)} representatives", flush=True)
    summary, contrib = run_fast_discovery(reps, taxonomy)
    summary.to_csv(OUT_ROOT / "candidate_summary.csv", index=False)
    contrib.to_csv(OUT_ROOT / "contribution_decomposition.csv", index=False)

    selection = select_next(summary)
    if selection["status"] == "HAS_STRONG":
        (OUT_ROOT / "next_full_validation_candidate.md").write_text(
            selection["next_md"], encoding="utf-8"
        )
    else:
        (OUT_ROOT / "SPRINT11_NO_STRONG_EXISTING_CANDIDATE").write_text(
            "SPRINT11_NO_STRONG_EXISTING_CANDIDATE\n", encoding="utf-8"
        )

    report = render_report(inventory, redundancy, taxonomy, summary, selection)
    (OUT_ROOT / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "sprint": "Sprint 11 — Price Formation / Intraday Path Family v1",
        "status": "AWAITING_DECISION"
        if selection["status"] == "HAS_STRONG"
        else "FROZEN_NO_STRONG_EXISTING",
        "n_registry": int(len(inventory)),
        "n_representatives_tested": int(len(summary)),
        "selection": selection["status"],
        "next_candidate": selection.get("factor"),
        "discovery_window": [str(DISCOVERY_START.date()), str(DISCOVERY_END.date())],
        "new_formulas": False,
        "protocol_ref": "evaluation_protocol_v2.0 (untouched)",
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\n[done] {selection['status']} next={selection.get('factor')} "
        f"-> {OUT_ROOT} ({manifest['elapsed_seconds']}s)",
        flush=True,
    )
    return manifest
