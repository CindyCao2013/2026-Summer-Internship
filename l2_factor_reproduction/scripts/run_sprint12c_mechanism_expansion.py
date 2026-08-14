#!/usr/bin/env python
"""Sprint 12C — Price–Book Dislocation Focused Mechanism Expansion.

Hypotheses MUST already be frozen in mechanism_hypotheses.csv before metrics.
No Full Validation / new primitives / parameter grids / Sprint 13.

Usage:
    python -m l2_factor_reproduction.scripts.run_sprint12c_mechanism_expansion
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

NEAR_ALIAS = 0.90

OUT = Path(RESULT_ROOT) / "sprint12_order_book" / "mechanism_expansion"
OB_DS = Path(RESULT_ROOT) / "primitives" / "order_book_daily" / "dataset"
FACTORS_DIR = Path(RESULT_ROOT) / "candidate_pool_v1" / "order_book_family" / "factors"
MCAP_PATH = Path(RESULT_ROOT) / "primitives" / "mcap_wide_2019-01-01_2026-07-31.parquet"
PF_DS = Path(RESULT_ROOT) / "primitives" / "price_formation_daily" / "dataset"
S12A_SUMMARY = Path(RESULT_ROOT) / "sprint12_order_book" / "discovery_summary.csv"

REF_FACTORS = [
    "book_vwap_gap",
    "total_depth_volatility",
    "obi_sign_persistence",
    "closing_obi_l5",
]

EPS = 1e-12
IC_HORIZONS = (1, 3, 5)
PERS_HORIZONS = (1,)
N_GROUPS = 10


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(x: float, d: int = 3) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2%}"


# ---------------------------------------------------------------------------
# Load primitives + build NEW factors (exact frozen formulas)
# ---------------------------------------------------------------------------


def load_order_book_discovery() -> pd.DataFrame:
    cols = [
        "symbol",
        "TradeDate",
        "book_vwap_gap_mean",
        "relative_spread_mean",
        "relative_spread_std",
        "log_total_depth_mean",
        "log_total_depth_std",
        "microprice_deviation_mean",
        "microprice_deviation_std",
        "obi_5_mean",
        "relative_spread_p90",
        "log_total_depth_p10",
    ]
    files: List[Path] = []
    for y in (2023, 2024):
        files.extend(sorted((OB_DS / f"year={y}").glob("*.parquet")))
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df = df.loc[
        df["TradeDate"].between(DISCOVERY_START, DISCOVERY_END)
    ].copy()
    return df.reset_index(drop=True)


def build_new_factor_values(ob: pd.DataFrame) -> Dict[str, pd.Series]:
    """Exact formulas from frozen mechanism_hypotheses.csv — do not edit post-hoc."""
    gap = ob["book_vwap_gap_mean"].astype(float)
    spr = ob["relative_spread_mean"].astype(float)
    spr_std = ob["relative_spread_std"].astype(float)
    depth = ob["log_total_depth_mean"].astype(float)
    depth_std = ob["log_total_depth_std"].astype(float)
    mp = ob["microprice_deviation_mean"].astype(float)
    obi5 = ob["obi_5_mean"].astype(float)
    spr_p90 = ob["relative_spread_p90"].astype(float)
    depth_p10 = ob["log_total_depth_p10"].astype(float)

    sign_mp = np.sign(mp.to_numpy())
    sign_obi = np.sign(obi5.to_numpy())
    agree = (sign_mp == sign_obi) & (sign_mp != 0) & (sign_obi != 0)
    disagree = (sign_mp != sign_obi) & (sign_mp != 0) & (sign_obi != 0)

    out = {
        "gap_over_spread_std": gap / (spr_std + EPS),
        "gap_inv_spread_cv": gap * spr / (spr_std + EPS),
        "gap_over_depth_std": gap / (depth_std + EPS),
        "gap_over_spread": gap / (spr + EPS),
        "gap_quality": gap * depth / (spr + EPS),
        "confirmed_microprice": pd.Series(
            np.where(agree, mp.to_numpy(), 0.0), index=ob.index
        ),
        "contradicted_microprice": pd.Series(
            np.where(disagree, mp.to_numpy(), 0.0), index=ob.index
        ),
        "vacuum_state": spr_p90 / (depth_p10 + EPS),
        "gap_in_vacuum": gap * spr_p90 / (depth_p10 + EPS),
    }
    return out


def series_to_narrow(
    symbols: pd.Series,
    dates: pd.Series,
    values: pd.Series,
    factor_id: str,
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
    return out.reset_index(drop=True)


def load_ref_narrow(factor: str) -> pd.DataFrame:
    path = FACTORS_DIR / factor / "factor_narrow.parquet"
    df = pd.read_parquet(path)
    df["tradetime"] = pd.to_datetime(df["tradetime"])
    mask = df["tradetime"].between(
        DISCOVERY_START, DISCOVERY_END + pd.Timedelta(hours=23)
    )
    out = df.loc[mask, ["symbol", "tradetime", "factorname", "value"]].copy()
    out["factorname"] = factor
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


# ---------------------------------------------------------------------------
# Economics (same as Sprint 12)
# ---------------------------------------------------------------------------


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
    return {
        "daily_hl_l1_traded_notional": avg_hl_l1,
        "daily_hl_oneway_turnover": avg_hl_ow,
        "fee_annualized_at_7p5bps": fee_annu,
        "approx_net_hl_annual": float(calAnnuRet(hl_net)),
        "approx_net_hl_sharpe": float(calSharpe(hl_net)),
        "G10_daily_oneway_turnover": avg_g10_ow,
        "G10_net_excess_annual": float(calAnnuRet(g10_net)),
        "G10_gross_excess_annual": float(calAnnuRet(g10_gross)),
        "long_contribution": long_a,
        "short_contribution": short_a,
        "dominant_leg": dominant,
        "short_leg_share_abs": (
            abs(short_a) / (abs(long_a) + abs(short_a))
            if (abs(long_a) + abs(short_a)) > 0
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def daily_rank_ic(a: pd.Series, b: pd.Series) -> float:
    x = a.replace([np.inf, -np.inf], np.nan)
    y = b.replace([np.inf, -np.inf], np.nan)
    valid = x.notna() & y.notna()
    if int(valid.sum()) < 30:
        return float("nan")
    return float(x[valid].corr(y[valid], method="spearman"))


def assign_deciles(row: pd.Series, n: int = N_GROUPS) -> pd.Series:
    s = row.dropna()
    if len(s) < n * 5:
        return pd.Series(index=row.index, dtype=float)
    ranks = s.rank(method="first")
    bins = np.ceil(ranks / len(s) * n).clip(1, n).astype(int)
    out = pd.Series(index=row.index, dtype=float)
    out.loc[bins.index] = bins.astype(float)
    return out


def compute_persistence(
    factors_wide: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
    directions: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    dates = ret.index
    for fid, wide in factors_wide.items():
        aligned = wide.reindex(index=dates, columns=ret.columns)
        ic_map: Dict[int, float] = {}
        for h in IC_HORIZONS:
            ics = []
            for i in range(len(dates) - h):
                ic = daily_rank_ic(aligned.loc[dates[i]], ret.loc[dates[i + h]])
                if np.isfinite(ic):
                    ics.append(ic)
            mean_ic = float(np.nanmean(ics)) if ics else float("nan")
            ic_map[h] = mean_ic

        # rank persistence t1
        rhos = []
        for i in range(len(aligned.index) - 1):
            rho = daily_rank_ic(aligned.iloc[i], aligned.iloc[i + 1])
            if np.isfinite(rho):
                rhos.append(rho)
        rp1 = float(np.nanmean(rhos)) if rhos else float("nan")

        # G10/G1 retention t1 (effective direction)
        direction = int(directions.get(fid, 1))
        eff = aligned * float(direction)
        g10_ret, g1_ret = [], []
        for i in range(len(eff.index) - 1):
            d0 = assign_deciles(eff.iloc[i])
            d1 = assign_deciles(eff.iloc[i + 1])
            common = d0.dropna().index.intersection(d1.dropna().index)
            if len(common) < 50:
                continue
            a0, a1 = d0.loc[common], d1.loc[common]
            g10_mask = a0 == N_GROUPS
            g1_mask = a0 == 1
            if g10_mask.any():
                g10_ret.append(float((a1.loc[g10_mask] == N_GROUPS).mean()))
            if g1_mask.any():
                g1_ret.append(float((a1.loc[g1_mask] == 1).mean()))

        ic1 = ic_map.get(1, float("nan"))
        rows.append(
            {
                "factor_id": fid,
                "IC_t1": ic_map.get(1, float("nan")),
                "IC_t3": ic_map.get(3, float("nan")),
                "IC_t5": ic_map.get(5, float("nan")),
                "IC_retention_t3": (
                    ic_map[3] / ic1
                    if abs(ic1) > 1e-12 and np.isfinite(ic_map.get(3, np.nan))
                    else float("nan")
                ),
                "IC_retention_t5": (
                    ic_map[5] / ic1
                    if abs(ic1) > 1e-12 and np.isfinite(ic_map.get(5, np.nan))
                    else float("nan")
                ),
                "rank_persistence_t1": rp1,
                "G10_retention_t1": float(np.mean(g10_ret)) if g10_ret else float("nan"),
                "G1_retention_t1": float(np.mean(g1_ret)) if g1_ret else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exposures
# ---------------------------------------------------------------------------


def daily_spearman_exposure(signal: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    idx = signal.index.intersection(panel.index)
    cols = signal.columns.intersection(panel.columns)
    return signal.loc[idx, cols].corrwith(panel.loc[idx, cols], axis=1, method="spearman")


def load_price_formation_controls() -> Dict[str, pd.DataFrame]:
    cols = ["symbol", "TradeDate", "close_price", "realized_variance", "daily_amount"]
    files: List[Path] = []
    for y in (2023, 2024):
        for base in (PF_DS, PF_DS.parent):
            ydir = base / f"year={y}"
            if ydir.exists():
                files.extend(sorted(ydir.glob("*.parquet")))
    files = sorted(set(files))
    if not files:
        return {}
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=cols))
        except Exception:
            continue
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    df = df.loc[df["TradeDate"].between(DISCOVERY_START, DISCOVERY_END)]
    out = {}
    for col, name in [
        ("close_price", "log_price"),
        ("realized_variance", "volatility"),
        ("daily_amount", "log_amount"),
    ]:
        wide = df.pivot_table(
            index="TradeDate", columns="symbol", values=col, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        if name == "log_price":
            wide = np.log(wide.where(wide > 0))
        elif name == "log_amount":
            wide = np.log1p(wide.where(wide >= 0))
        out[name] = wide.sort_index()
    return out


def try_load_turnover() -> Optional[pd.DataFrame]:
    try:
        from l2_factor_reproduction.python.mid_trade_amount_research_data import (
            load_turnover_wide,
        )

        turn = load_turnover_wide(DISCOVERY_START, DISCOVERY_END)
        if turn is None or turn.empty:
            return None
        turn.index = pd.to_datetime(turn.index)
        return turn
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] turnover unavailable: {exc}", flush=True)
        return None


def try_industry_r2(signal: pd.DataFrame) -> Optional[float]:
    """Mean cross-sectional R^2 of factor on CITICS industry dummies."""
    try:
        from Factor_Dev_Lib import get_preheat_ind_data_citics

        ind = get_preheat_ind_data_citics(
            DISCOVERY_START.to_pydatetime(), DISCOVERY_END.to_pydatetime()
        ).set_index("TradingDay")
        ind.index = pd.to_datetime(ind.index)
        ind = ind.reindex(index=signal.index, columns=signal.columns)
        r2s = []
        for dt in signal.index:
            y = signal.loc[dt].dropna()
            g = ind.loc[dt].reindex(y.index)
            valid = y.notna() & g.notna()
            if int(valid.sum()) < 50:
                continue
            yy = y[valid].astype(float)
            gg = g[valid].astype(str)
            # group means
            means = yy.groupby(gg).transform("mean")
            ss_tot = float(((yy - yy.mean()) ** 2).sum())
            if ss_tot <= 0:
                continue
            ss_res = float(((yy - means) ** 2).sum())
            r2s.append(1.0 - ss_res / ss_tot)
        return float(np.mean(r2s)) if r2s else None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] industry R2 unavailable: {exc}", flush=True)
        return None


def compute_exposures(
    factors_wide: Dict[str, pd.DataFrame],
    target_ids: List[str],
    ob_controls: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    mcap = pd.read_parquet(MCAP_PATH)
    mcap.index = pd.to_datetime(mcap.index)
    log_cap = np.log(mcap.where(mcap > 0))
    turn = try_load_turnover()
    pf = load_price_formation_controls()

    panels: Dict[str, pd.DataFrame] = {
        "log_FloatMktCap": log_cap,
        **ob_controls,
        **pf,
    }
    if turn is not None:
        panels["Turnover"] = turn

    for fid in target_ids:
        if fid not in factors_wide:
            continue
        sig = factors_wide[fid]
        for feat, panel in panels.items():
            rho = daily_spearman_exposure(sig, panel)
            rows.append(
                {
                    "factor_id": fid,
                    "feature": feat,
                    "mean_spearman": float(rho.mean()) if len(rho) else float("nan"),
                    "abs_mean_spearman": float(rho.abs().mean()) if len(rho) else float("nan"),
                    "median_spearman": float(rho.median()) if len(rho) else float("nan"),
                    "n_days": int(rho.notna().sum()),
                }
            )
        ind_r2 = try_industry_r2(sig)
        if ind_r2 is not None:
            rows.append(
                {
                    "factor_id": fid,
                    "feature": "industry_dummy_R2",
                    "mean_spearman": ind_r2,
                    "abs_mean_spearman": ind_r2,
                    "median_spearman": ind_r2,
                    "n_days": int(sig.shape[0]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlations / clusters
# ---------------------------------------------------------------------------


def mean_daily_spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
    idx = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    if len(idx) == 0 or len(cols) == 0:
        return float("nan")
    rhos = a.loc[idx, cols].corrwith(b.loc[idx, cols], axis=1, method="spearman")
    return float(rhos.mean()) if rhos.notna().any() else float("nan")


def build_correlations(
    factors_wide: Dict[str, pd.DataFrame], ids: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # pairwise
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if a not in factors_wide or b not in factors_wide:
                continue
            rho = mean_daily_spearman(factors_wide[a], factors_wide[b])
            pairs.append(
                {
                    "factor_a": a,
                    "factor_b": b,
                    "mean_daily_spearman": rho,
                    "near_alias": bool(np.isfinite(rho) and abs(rho) >= NEAR_ALIAS),
                }
            )
    corr_df = pd.DataFrame(pairs)

    # simple greedy clusters
    parent = {x: x for x in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in corr_df.iterrows():
        if bool(r["near_alias"]):
            union(str(r["factor_a"]), str(r["factor_b"]))

    # representatives: prefer book_vwap_gap, else first by name
    clusters = {}
    for fid in ids:
        clusters.setdefault(find(fid), []).append(fid)

    rows = []
    for cid, (root, members) in enumerate(sorted(clusters.items(), key=lambda kv: kv[0])):
        rep = "book_vwap_gap" if "book_vwap_gap" in members else sorted(members)[0]
        for m in sorted(members):
            rows.append(
                {
                    "cluster_id": f"C{cid}",
                    "factor_id": m,
                    "is_representative": m == rep,
                    "representative": rep,
                    "cluster_size": len(members),
                }
            )
    return corr_df, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cumulative_deciles_hl").mkdir(parents=True, exist_ok=True)
    (OUT / "decile_bar").mkdir(parents=True, exist_ok=True)
    (OUT / "factor_cache").mkdir(parents=True, exist_ok=True)

    hyp_path = OUT / "mechanism_hypotheses.csv"
    assert hyp_path.exists(), "mechanism_hypotheses.csv must be frozen BEFORE backtest"
    frozen_stamp = OUT / "HYPOTHESES_FROZEN.txt"
    assert frozen_stamp.exists(), "missing HYPOTHESES_FROZEN.txt"
    hyp = pd.read_csv(hyp_path)
    print(f"[0] Loaded frozen hypotheses n={len(hyp)}", flush=True)

    # Verify hashes unchanged
    for _, r in hyp.iterrows():
        expect = _sha(f"{r['factor_id']}|{r['exact_formula']}|sprint12c_v1")
        assert str(r["formula_hash"]) == expect, (
            f"HASH DRIFT {r['factor_id']}: {r['formula_hash']} != {expect}"
        )

    print("[1] Load order_book_daily discovery window", flush=True)
    ob = load_order_book_discovery()
    new_vals = build_new_factor_values(ob)

    # Build narrow caches for new factors
    narrows: Dict[str, pd.DataFrame] = {}
    for fid in hyp["factor_id"].tolist():
        assert fid in new_vals, fid
        narrow = series_to_narrow(ob["symbol"], ob["TradeDate"], new_vals[fid], fid)
        cache = OUT / "factor_cache" / f"{fid}.parquet"
        narrow.to_parquet(cache, index=False)
        narrows[fid] = narrow
        print(f"  built {fid}: n={len(narrow)}", flush=True)

    # References from existing caches
    for fid in REF_FACTORS:
        narrows[fid] = load_ref_narrow(fid)
        print(f"  ref {fid}: n={len(narrows[fid])}", flush=True)

    # Candidate registry
    reg_rows = []
    for _, r in hyp.iterrows():
        reg_rows.append(
            {
                "factor_id": r["factor_id"],
                "is_reference": False,
                "mechanism_group": r["mechanism_group"],
                "exact_formula": r["exact_formula"],
                "formula_hash": r["formula_hash"],
                "source_primitives": r["source_primitives"],
                "expected_raw_direction": r["expected_raw_direction"],
                "why_l2_native": r["why_l2_native"],
                "why_not_alias_of_existing_factor": r["why_not_alias_of_existing_factor"],
            }
        )
    for fid in REF_FACTORS:
        spec = ORDER_BOOK_FACTOR_SPECS[fid]
        formula = spec.formula
        reg_rows.append(
            {
                "factor_id": fid,
                "is_reference": True,
                "mechanism_group": "REFERENCE",
                "exact_formula": formula,
                "formula_hash": _sha(f"{fid}|{formula}|order_book_v1"),
                "source_primitives": f"order_book_daily / {formula}",
                "expected_raw_direction": "",
                "why_l2_native": "Frozen Sprint12 reference — unchanged",
                "why_not_alias_of_existing_factor": "reference",
            }
        )
    registry = pd.DataFrame(reg_rows)
    registry.to_csv(OUT / "candidate_registry.csv", index=False)

    print("[2] Load fast context + run discovery", flush=True)
    mask, ret = load_fast_context("discovery")

    summary_rows = []
    econ_rows = []
    leg_rows = []
    factors_wide: Dict[str, pd.DataFrame] = {}
    directions: Dict[str, int] = {}

    all_ids = hyp["factor_id"].tolist() + REF_FACTORS
    for fid in all_ids:
        print(f"\n=== {fid} ===", flush=True)
        t0 = time.perf_counter()
        narrow = narrows[fid]
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
        directions[fid] = int(metrics["factor_direction"])

        # plots for all (needed for viable reps later)
        tmp = OUT / "_tmp_figs" / fid
        save_fast_plots(tmp, fid, group_pnl, metrics)
        cum_src = tmp / "cumulative_hl.png"
        bar_src = tmp / "decile_bar.png"
        if cum_src.exists():
            (OUT / "cumulative_deciles_hl" / f"{fid}.png").write_bytes(cum_src.read_bytes())
        if bar_src.exists():
            (OUT / "decile_bar" / f"{fid}.png").write_bytes(bar_src.read_bytes())

        wide = narrow_to_wide(narrow)
        m = mask.reindex(index=wide.index, columns=wide.columns)
        m = m.fillna(False).astype(bool)
        wide = wide.where(m)
        factors_wide[fid] = wide

        reg = registry.loc[registry["factor_id"] == fid].iloc[0]
        summary_rows.append(
            {
                "factor_id": fid,
                "is_reference": bool(reg["is_reference"]),
                "mechanism_group": reg["mechanism_group"],
                "exact_formula": reg["exact_formula"],
                "formula_hash": reg["formula_hash"],
                "source_primitives": reg["source_primitives"],
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
                "daily_hl_oneway_turnover": econ["daily_hl_oneway_turnover"],
                "fee_annualized_at_7p5bps": econ["fee_annualized_at_7p5bps"],
                "approx_net_hl_annual": econ["approx_net_hl_annual"],
                "approx_net_hl_sharpe": econ["approx_net_hl_sharpe"],
                "G10_net_excess_annual": econ["G10_net_excess_annual"],
                "daily_G10_oneway_turnover": econ["G10_daily_oneway_turnover"],
                "long_contribution": econ["long_contribution"],
                "short_contribution": econ["short_contribution"],
                "dominant_leg": econ["dominant_leg"],
            }
        )
        econ_rows.append({"factor_id": fid, **econ})
        leg_rows.append(
            {
                "factor_id": fid,
                "long_contribution": econ["long_contribution"],
                "short_contribution": econ["short_contribution"],
                "dominant_leg": econ["dominant_leg"],
                "short_leg_share_abs": econ["short_leg_share_abs"],
                "G10_gross_excess_annual": econ["G10_gross_excess_annual"],
                "G10_net_excess_annual": econ["G10_net_excess_annual"],
            }
        )
        print(
            f"  gate={gate} sharpe={metrics['hl_sharpe']:.3f} "
            f"netS={econ['approx_net_hl_sharpe']:.3f} TO={econ['daily_hl_oneway_turnover']:.3f}",
            flush=True,
        )

    discovery = pd.DataFrame(summary_rows)
    discovery.to_csv(OUT / "discovery_summary.csv", index=False)
    pd.DataFrame(econ_rows).to_csv(OUT / "economic_diagnostics.csv", index=False)
    pd.DataFrame(leg_rows).to_csv(OUT / "leg_diagnostics.csv", index=False)

    print("[3] Persistence diagnostics", flush=True)
    # Align ret to discovery
    pers = compute_persistence(factors_wide, ret, directions)
    pers.to_csv(OUT / "persistence_summary.csv", index=False)

    print("[4] Correlations / clusters", flush=True)
    corr_df, cluster_df = build_correlations(factors_wide, all_ids)
    corr_df.to_csv(OUT / "candidate_correlation.csv", index=False)
    cluster_df.to_csv(OUT / "candidate_clusters.csv", index=False)

    print("[5] Exposure diagnostics (STRONG/RESEARCH + book_vwap_gap)", flush=True)
    # OB controls from same discovery frame
    ob_ctrl = {}
    for col, name in [
        ("relative_spread_mean", "spread"),
        ("log_total_depth_mean", "log_depth"),
        ("book_vwap_gap_mean", "book_vwap_gap_level"),
    ]:
        wide = ob.pivot_table(
            index="TradeDate", columns="symbol", values=col, aggfunc="last"
        )
        wide.index = pd.to_datetime(wide.index)
        ob_ctrl[name] = wide.sort_index()

    expose_ids = ["book_vwap_gap"] + [
        r["factor_id"]
        for r in summary_rows
        if (not r["is_reference"]) and r["gate"] in ("strong_candidate", "research_candidate")
    ]
    # Always include top net economic new candidates even if gate none (diagnostic)
    new_disc = discovery.loc[~discovery["is_reference"]].copy()
    if not new_disc.empty:
        top_net = new_disc.sort_values("approx_net_hl_sharpe", ascending=False).head(3)[
            "factor_id"
        ].tolist()
        for t in top_net:
            if t not in expose_ids:
                expose_ids.append(t)
    expose_ids = list(dict.fromkeys(expose_ids))
    exposure = compute_exposures(factors_wide, expose_ids, ob_ctrl)
    exposure.to_csv(OUT / "exposure_diagnostics.csv", index=False)

    print("[6] Mechanism comparison + report", flush=True)
    write_mechanism_comparison(discovery, pers, cluster_df)
    write_report(discovery, pers, exposure, corr_df, cluster_df, hyp)
    write_manifest(discovery, hyp)
    print("[DONE]", OUT, flush=True)


def write_mechanism_comparison(
    discovery: pd.DataFrame, pers: pd.DataFrame, clusters: pd.DataFrame
) -> None:
    d = discovery.merge(pers, on="factor_id", how="left")
    d = d.merge(
        clusters[["factor_id", "cluster_id", "is_representative", "representative"]],
        on="factor_id",
        how="left",
    )
    cols = [
        "factor_id",
        "is_reference",
        "mechanism_group",
        "gate",
        "gross_hl_sharpe",
        "approx_net_hl_sharpe",
        "approx_net_hl_annual",
        "G10_net_excess_annual",
        "daily_hl_oneway_turnover",
        "decile_mono",
        "adjacent_violations",
        "IC_t1",
        "IC_t3",
        "IC_t5",
        "IC_retention_t3",
        "IC_retention_t5",
        "rank_persistence_t1",
        "G10_retention_t1",
        "dominant_leg",
        "long_contribution",
        "short_contribution",
        "cluster_id",
        "is_representative",
        "representative",
    ]
    d[cols].to_csv(OUT / "mechanism_comparison.csv", index=False)


def _best_new(discovery: pd.DataFrame, col: str, ascending: bool = False) -> str:
    sub = discovery.loc[~discovery["is_reference"]].copy()
    if sub.empty:
        return "none"
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    if sub.empty:
        return "none"
    return str(sub.sort_values(col, ascending=ascending).iloc[0]["factor_id"])


def write_report(
    discovery: pd.DataFrame,
    pers: pd.DataFrame,
    exposure: pd.DataFrame,
    corr: pd.DataFrame,
    clusters: pd.DataFrame,
    hyp: pd.DataFrame,
) -> None:
    d = discovery.merge(pers, on="factor_id", how="left")
    new = d.loc[~d["is_reference"]].copy()
    refs = d.loc[d["is_reference"]].copy()

    def _mech_answer(group: str) -> str:
        sub = new.loc[new["mechanism_group"] == group]
        if sub.empty:
            return "no candidates"
        lines = []
        for _, r in sub.sort_values("approx_net_hl_sharpe", ascending=False).iterrows():
            lines.append(
                f"- `{r['factor_id']}`: gate={r['gate']}, grossS={_fmt(r['gross_hl_sharpe'])}, "
                f"netS={_fmt(r['approx_net_hl_sharpe'])}, TO={_fmt(r['daily_hl_oneway_turnover'])}, "
                f"IC_ret_t3={_fmt(r['IC_retention_t3'])}, rank_pers={_fmt(r['rank_persistence_t1'])}"
            )
        return "\n".join(lines)

    bvg = refs.loc[refs["factor_id"] == "book_vwap_gap"]
    bvg_net = float(bvg["approx_net_hl_sharpe"].iloc[0]) if len(bvg) else float("nan")
    bvg_to = float(bvg["daily_hl_oneway_turnover"].iloc[0]) if len(bvg) else float("nan")
    bvg_s = float(bvg["gross_hl_sharpe"].iloc[0]) if len(bvg) else float("nan")

    # Stability improve?
    stable = new.loc[new["mechanism_group"] == "INTRADAY_STABLE_DISLOCATION"]
    stable_better = []
    for _, r in stable.iterrows():
        better_net = np.isfinite(r["approx_net_hl_sharpe"]) and r["approx_net_hl_sharpe"] > bvg_net
        better_gate = r["gate"] in ("strong_candidate", "research_candidate")
        stable_better.append((r["factor_id"], better_net, better_gate, r["approx_net_hl_sharpe"], r["gate"]))

    # FV policy
    fv_candidates = []
    for _, r in new.iterrows():
        if r["gate"] not in ("strong_candidate", "research_candidate"):
            continue
        if not (np.isfinite(r["approx_net_hl_sharpe"]) and r["approx_net_hl_sharpe"] > 0):
            continue
        if not (np.isfinite(r["IC_retention_t3"]) and r["IC_retention_t3"] > 0.5):
            continue
        # healthier than high-churn 12A failures: TO << 1.0 preferred
        if not (np.isfinite(r["daily_hl_oneway_turnover"]) and r["daily_hl_oneway_turnover"] < 1.0):
            continue
        # not near-alias of book_vwap_gap
        alias = corr.loc[
            (
                ((corr["factor_a"] == r["factor_id"]) & (corr["factor_b"] == "book_vwap_gap"))
                | ((corr["factor_b"] == r["factor_id"]) & (corr["factor_a"] == "book_vwap_gap"))
            )
            & (corr["near_alias"])
        ]
        if len(alias):
            continue
        fv_candidates.append(r["factor_id"])

    # Also check STRONG with full policy
    strong_new = new.loc[new["gate"] == "strong_candidate"]
    ready = False
    for _, r in strong_new.iterrows():
        if (
            np.isfinite(r["approx_net_hl_sharpe"])
            and r["approx_net_hl_sharpe"] > 0
            and np.isfinite(r["IC_retention_t3"])
            and r["IC_retention_t3"] > 0.5
            and r["daily_hl_oneway_turnover"] < 1.0
        ):
            # redundancy check
            alias = corr.loc[
                (
                    ((corr["factor_a"] == r["factor_id"]) & (corr["factor_b"] == "book_vwap_gap"))
                    | ((corr["factor_b"] == r["factor_id"]) & (corr["factor_a"] == "book_vwap_gap"))
                )
                & (corr["near_alias"])
            ]
            if len(alias) == 0:
                ready = True
                fv_candidates.append(r["factor_id"])

    fv_candidates = list(dict.fromkeys(fv_candidates))
    # PART Q: requires EXISTING Fast Gate level — interpret as STRONG for FV recommend
    # Task: "passes the EXISTING required Fast Gate level" + economic...
    # Historically FV only after STRONG. Research alone insufficient.
    strong_fv = [x for x in fv_candidates if x in set(strong_new["factor_id"])]
    verdict = (
        "ORDER_BOOK_READY_FOR_SINGLE_FACTOR_FV"
        if strong_fv
        else "ORDER_BOOK_SNAPSHOT_FAMILY_CLOSE"
    )

    # Exposure notes
    exp_notes = []
    if not exposure.empty:
        for fid in expose_focus(discovery, exposure):
            sub = exposure.loc[exposure["factor_id"] == fid]
            if sub.empty:
                continue
            top = sub.sort_values("abs_mean_spearman", ascending=False).head(4)
            bits = ", ".join(
                f"{r.feature}={_fmt(r.abs_mean_spearman)}" for _, r in top.iterrows()
            )
            exp_notes.append(f"- `{fid}`: top |ρ| → {bits}")

    # Answers
    q1 = answer_stability(stable_better, bvg_net, bvg_s)
    q2 = answer_confidence(new, bvg_net)
    q3, q4 = answer_confirmation(new)
    q5 = answer_vacuum(new, bvg_net, corr)
    q6 = _best_new(discovery, "gross_hl_sharpe")
    q7 = _best_new(discovery, "approx_net_hl_sharpe")
    q8a = _best_new(discovery, "daily_hl_oneway_turnover", ascending=True)
    q8b = _best_new(discovery.merge(pers, on="factor_id"), "rank_persistence_t1")
    q9 = _best_new(discovery, "G10_net_excess_annual")

    report = f"""# Sprint 12C — Price–Book Dislocation Focused Mechanism Expansion

**Status:** COMPLETE  
**Hypotheses frozen before backtest:** YES (`mechanism_hypotheses.csv`, `HYPOTHESES_FROZEN.txt`)  
**New candidates:** {len(hyp)} (≤10)  
**References (unchanged):** {', '.join(REF_FACTORS)}  
**Discovery window:** {DISCOVERY_START.date()} ~ {DISCOVERY_END.date()}  
**Constraints:** NO Full Validation · NO new primitives · NO parameter grids · NO Sprint 13

## Discovery summary (new + references)

{d[['factor_id','is_reference','mechanism_group','gate','gross_hl_sharpe','approx_net_hl_sharpe','daily_hl_oneway_turnover','G10_net_excess_annual','decile_mono','adjacent_violations','IC_retention_t3','rank_persistence_t1','dominant_leg']].to_string(index=False)}

## Mechanism-level conclusions

### INTRADAY_STABLE_DISLOCATION
Does SAME-DAY stability improve raw book_vwap_gap?
{q1}

{_mech_answer('INTRADAY_STABLE_DISLOCATION')}

### DISLOCATION_CONFIDENCE
{q2}

{_mech_answer('DISLOCATION_CONFIDENCE')}

### PRICE_PRESSURE_CONFIRMATION
Confirmation: {q3}
Contradiction: {q4}

{_mech_answer('PRICE_PRESSURE_CONFIRMATION')}

### LIQUIDITY_VACUUM
{q5}

{_mech_answer('LIQUIDITY_VACUUM')}

## Exposure diagnostics (selected)

{chr(10).join(exp_notes) if exp_notes else 'n/a'}

## Redundancy

Near-alias pairs (|ρ|≥{NEAR_ALIAS}):
{corr.loc[corr['near_alias']].to_string(index=False) if corr['near_alias'].any() else '(none)'}

## FINAL QUESTIONS

1. Same-day stability improve raw book_vwap_gap? **{q1.split(chr(10))[0]}**
2. Which confidence definition improves? **{q2.split(chr(10))[0]}**
3. Pressure confirmation strengthen or dilute? **{q3}**
4. Is contradiction informative? **{q4}**
5. Joint vacuum incremental alpha? **{q5.split(chr(10))[0]}**
6. Statistically strongest NEW: **{q6}**
7. Strongest after 7.5bps NEW: **{q7}**
8. Lowest natural TO / highest rank persistence NEW: **TO={q8a} / pers={q8b}**
9. Best G10 net NEW: **{q9}**
10. Disguised size/liquidity/spread? See exposure section; confidence/vacuum formulas with depth/spread terms are the primary suspects.
11. Deserve Single-Factor Full Validation? **{'YES: ' + ', '.join(strong_fv) if strong_fv else 'NO'}**
12. Final verdict: **{verdict}**

{"If CLOSE: next recommended genuinely new L2 family is **DIRECTIONAL_REFILL_ASYMMETRY** (do not implement here)." if verdict.endswith("CLOSE") else ""}

## STOP

No Full Validation. No new primitive implementation. No parameter optimization. No Sprint 13.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def expose_focus(discovery: pd.DataFrame, exposure: pd.DataFrame) -> List[str]:
    return sorted(exposure["factor_id"].unique().tolist())


def answer_stability(stable_better, bvg_net, bvg_s) -> str:
    if not stable_better:
        return "No stable candidates."
    improved = [x for x in stable_better if x[1]]
    gated = [x for x in stable_better if x[2]]
    if not improved and not gated:
        best = max(stable_better, key=lambda x: (x[3] if np.isfinite(x[3]) else -999))
        return (
            f"NO — none beat book_vwap_gap netS={_fmt(bvg_net)} (grossS={_fmt(bvg_s)}). "
            f"Best stable `{best[0]}` netS={_fmt(best[3])} gate={best[4]}."
        )
    if improved:
        names = ", ".join(f"`{x[0]}`(netS={_fmt(x[3])})" for x in improved)
        return f"PARTIAL/YES on net economics vs book_vwap_gap: {names}. Gate clears: {[x[0] for x in gated] or 'none'}."
    return f"NO net improvement; gate clears only: {[x[0] for x in gated]}."


def answer_confidence(new: pd.DataFrame, bvg_net: float) -> str:
    sub = new.loc[new["mechanism_group"] == "DISLOCATION_CONFIDENCE"]
    if sub.empty:
        return "No confidence candidates."
    best = sub.sort_values("approx_net_hl_sharpe", ascending=False).iloc[0]
    beat = best["approx_net_hl_sharpe"] > bvg_net
    return (
        f"`{best['factor_id']}` best netS={_fmt(best['approx_net_hl_sharpe'])} "
        f"gate={best['gate']} (vs book_vwap_gap netS={_fmt(bvg_net)}; "
        f"{'improves' if beat else 'does not improve'} net). "
        "Check exposures for liquidity/spread disguise."
    )


def answer_confirmation(new: pd.DataFrame) -> Tuple[str, str]:
    conf = new.loc[new["factor_id"] == "confirmed_microprice"]
    contra = new.loc[new["factor_id"] == "contradicted_microprice"]
    if conf.empty:
        return "n/a", "n/a"
    c = conf.iloc[0]
    q3 = (
        f"confirmed_microprice gate={c['gate']} grossS={_fmt(c['gross_hl_sharpe'])} "
        f"netS={_fmt(c['approx_net_hl_sharpe'])} — "
        f"{'strengthens' if c['gate'] != 'none' and c['approx_net_hl_sharpe'] > 0 else 'does not strengthen'} "
        "vs needing a clean STRONG dislocation transform."
    )
    if contra.empty:
        return q3, "n/a"
    k = contra.iloc[0]
    q4 = (
        f"contradicted_microprice gate={k['gate']} grossS={_fmt(k['gross_hl_sharpe'])} "
        f"netS={_fmt(k['approx_net_hl_sharpe'])} — "
        f"{'YES informative' if abs(k['gross_hl_sharpe']) >= 1.5 else 'weak / not clearly informative'}."
    )
    return q3, q4


def answer_vacuum(new: pd.DataFrame, bvg_net: float, corr: pd.DataFrame) -> str:
    sub = new.loc[new["mechanism_group"] == "LIQUIDITY_VACUUM"]
    if sub.empty:
        return "No vacuum candidates."
    best = sub.sort_values("approx_net_hl_sharpe", ascending=False).iloc[0]
    alias = corr.loc[
        (
            ((corr.factor_a == best.factor_id) & (corr.factor_b == "book_vwap_gap"))
            | ((corr.factor_b == best.factor_id) & (corr.factor_a == "book_vwap_gap"))
        )
    ]
    rho = float(alias["mean_daily_spearman"].iloc[0]) if len(alias) else float("nan")
    return (
        f"`{best['factor_id']}` best vacuum netS={_fmt(best['approx_net_hl_sharpe'])} "
        f"gate={best['gate']} corr_vs_gap={_fmt(rho)}; "
        f"{'incremental' if best['approx_net_hl_sharpe'] > bvg_net and abs(rho) < 0.9 else 'no clear incremental alpha vs book_vwap_gap'}."
    )


def write_manifest(discovery: pd.DataFrame, hyp: pd.DataFrame) -> None:
    new = discovery.loc[~discovery["is_reference"]]
    strong = new.loc[new["gate"] == "strong_candidate", "factor_id"].tolist()
    research = new.loc[new["gate"] == "research_candidate", "factor_id"].tolist()
    # recompute verdict simply
    verdict = (
        "ORDER_BOOK_READY_FOR_SINGLE_FACTOR_FV"
        if strong
        and any(
            float(discovery.loc[discovery.factor_id == s, "approx_net_hl_sharpe"].iloc[0]) > 0
            for s in strong
        )
        else "ORDER_BOOK_SNAPSHOT_FAMILY_CLOSE"
    )
    # stricter: need positive net among strong
    ok = []
    for s in strong:
        row = discovery.loc[discovery.factor_id == s].iloc[0]
        if float(row["approx_net_hl_sharpe"]) > 0 and float(row["daily_hl_oneway_turnover"]) < 1.0:
            ok.append(s)
    verdict = (
        "ORDER_BOOK_READY_FOR_SINGLE_FACTOR_FV"
        if ok
        else "ORDER_BOOK_SNAPSHOT_FAMILY_CLOSE"
    )
    man = {
        "task": "Sprint 12C — Price–Book Dislocation Focused Mechanism Expansion",
        "hypotheses_frozen_before_backtest": True,
        "n_new_candidates": int(len(hyp)),
        "new_factor_ids": hyp["factor_id"].tolist(),
        "references": REF_FACTORS,
        "discovery_window": {
            "start": str(DISCOVERY_START.date()),
            "end": str(DISCOVERY_END.date()),
        },
        "gates": {
            "strong_new": strong,
            "research_new": research,
        },
        "verdict": verdict,
        "fv_recommended": ok,
        "next_family_if_close": "DIRECTIONAL_REFILL_ASYMMETRY",
        "constraints": [
            "NO_FULL_VALIDATION",
            "NO_NEW_PRIMITIVES",
            "NO_PARAMETER_OPTIMIZATION",
            "NO_SPRINT_13",
            "NO_CROSS_DAY_SMOOTHING",
        ],
        "outputs": [
            "mechanism_hypotheses.csv",
            "candidate_registry.csv",
            "discovery_summary.csv",
            "economic_diagnostics.csv",
            "persistence_summary.csv",
            "exposure_diagnostics.csv",
            "leg_diagnostics.csv",
            "candidate_correlation.csv",
            "candidate_clusters.csv",
            "mechanism_comparison.csv",
            "report.md",
            "manifest.json",
            "cumulative_deciles_hl/",
            "decile_bar/",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
