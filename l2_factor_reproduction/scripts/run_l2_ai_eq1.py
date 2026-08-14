"""Phase EQ-1: executable secondary qualification of 127 frozen features.

Does not train LightGBM/XGBoost, does not generate TC-2 descendants,
does not add cut candidates. Window 2023-01-01 → 2024-12-31.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "10")

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.feature_selection.labels import (  # noqa: E402
    build_labels_wide_panel,
    load_daily_excess_and_bench,
)
from l2_factor_reproduction.feature_selection.panel_io import (  # noqa: E402
    partitions_overlapping,
)
from l2_factor_reproduction.l2_ai_stock_selection.degradation import (  # noqa: E402
    daily_rank_ic_series,
)
from l2_factor_reproduction.l2_ai_stock_selection.entry_investability import (  # noqa: E402
    build_entry_tradable,
)
from l2_factor_reproduction.l2_ai_stock_selection.executable_labels import (  # noqa: E402
    load_production_labels,
)
from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (  # noqa: E402
    AUDIT_WINDOW_END,
    AUDIT_WINDOW_START,
    HORIZONS,
    PRIMARY_EXECUTION_CONTRACT,
)
from l2_factor_reproduction.l2_ai_stock_selection.inventory import (  # noqa: E402
    load_factor_inventory,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import (  # noqa: E402
    residual_mutual_information,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (  # noqa: E402
    EXECUTION,
    FACTOR_QUALIFICATION,
    REPORTS,
    ensure_layout,
)
from l2_factor_reproduction.l2_ai_stock_selection.qualification import (  # noqa: E402
    MI_MAX_SAMPLES,
    PERIODS,
    POSITIVE_CONTROL_PARENTS,
    build_tc2_parent_pool,
    classify_factor,
    daily_hl_and_deciles,
    family_qualification_summary,
    gates_dict,
    hl_from_daily,
    ic_from_daily,
    monotonicity_from_deciles,
    one_period_dominated,
    slice_dates,
)
from l2_factor_reproduction.python.fast_discovery import context_paths  # noqa: E402


FS1_ALIGNED = (
    PROJ_ROOT
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs1_feature_panel_full"
    / "aligned_raw"
)
VWAP_CACHE = EXECUTION / "cache" / "adj_vwap.parquet"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if pd.isna(obj):
        return None
    raise TypeError(type(obj))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _eligible() -> pd.DataFrame:
    inv = load_factor_inventory()
    return inv.loc[inv["eligible_for_fs"].astype(str).str.lower().isin(("true", "1"))].copy()


def _load_tradability(dates: pd.DatetimeIndex):
    mask_path = context_paths("full")["universe_mask"]
    universe = pd.read_parquet(mask_path)
    universe.index = pd.to_datetime(universe.index).normalize()
    universe = universe.reindex(index=dates)
    if not VWAP_CACHE.exists():
        raise FileNotFoundError("missing adj VWAP cache: {}".format(VWAP_CACHE))
    vwap = pd.read_parquet(VWAP_CACHE)
    vwap.index = pd.to_datetime(vwap.index).normalize()
    maps = build_entry_tradable(
        dates=dates,
        universe_mask_t=universe,
        adj_vwap=vwap,
        trade_status_t1=(universe == 1).astype(float),
        not_limit_t1=(universe == 1).astype(float),
    )
    return maps["signal_tradable_T"], maps["entry_tradable_T1"]


def _load_or_build_legacy(dates: pd.DatetimeIndex) -> Dict[int, pd.DataFrame]:
    cache = FACTOR_QUALIFICATION / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    out: Dict[int, pd.DataFrame] = {}
    missing = []
    for h in HORIZONS:
        p = cache / "legacy_c2c_forward_return_{}d.parquet".format(h)
        if p.exists():
            y = pd.read_parquet(p)
            y.index = pd.to_datetime(y.index).normalize()
            out[int(h)] = y
        else:
            missing.append(h)
    if not missing:
        return out
    _log("building legacy C2C labels for horizons {}".format(missing))
    excess, bench, all_dates = load_daily_excess_and_bench("full")
    leg_dates = all_dates.intersection(dates).sort_values()
    built = build_labels_wide_panel(excess, bench, leg_dates, horizons=list(HORIZONS))
    for h in HORIZONS:
        y = built[h]
        y.to_parquet(cache / "legacy_c2c_forward_return_{}d.parquet".format(h))
        out[int(h)] = y
    return out


def _period_ic(ic: pd.Series, dates: pd.DatetimeIndex) -> Dict[str, float]:
    out = {}
    ic = ic.copy()
    ic.index = pd.to_datetime(ic.index).normalize()
    for name, start, end in PERIODS:
        idx = slice_dates(pd.DatetimeIndex(ic.index), start, end)
        sub = ic.reindex(idx).dropna()
        out[name] = float(sub.mean()) if len(sub) else float("nan")
        out[name + "_n"] = int(len(sub))
    return out


def run_metrics(
    names: Sequence[str],
    y_ex: Dict[int, pd.DataFrame],
    y_leg: Dict[int, pd.DataFrame],
    mask_ex: pd.DataFrame,
    mask_leg: pd.DataFrame,
    audit_dates: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = partitions_overlapping(FS1_ALIGNED, AUDIT_WINDOW_START, AUDIT_WINDOW_END)
    ic_ex: Dict[Tuple[str, int], List[pd.Series]] = {(n, h): [] for n in names for h in HORIZONS}
    ic_leg: Dict[Tuple[str, int], List[pd.Series]] = {(n, h): [] for n in names for h in HORIZONS}
    hl_ex: Dict[Tuple[str, int], List[pd.Series]] = {(n, h): [] for n in names for h in HORIZONS}
    hl_leg: Dict[Tuple[str, int], List[pd.Series]] = {(n, h): [] for n in names for h in HORIZONS}
    dec_ex: Dict[Tuple[str, int], List[pd.DataFrame]] = {(n, h): [] for n in names for h in HORIZONS}
    dec_leg: Dict[Tuple[str, int], List[pd.DataFrame]] = {(n, h): [] for n in names for h in HORIZONS}
    cov_ex = {(n, h): [0, 0] for n in names for h in HORIZONS}
    cov_leg = {(n, h): [0, 0] for n in names for h in HORIZONS}
    mi_ex: Dict[Tuple[str, int], List[float]] = {(n, h): [] for n in names for h in HORIZONS}

    for part_i, part in enumerate(parts, start=1):
        _log("  EQ-1 partition {}/{} {}".format(part_i, len(parts), part.name))
        keep = ["TradeDate", "Symbol"] + list(names)
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(str(part)).schema.names)
            cols = [c for c in keep if c in available]
            raw = pd.read_parquet(part, columns=cols)
        except Exception:
            raw = pd.read_parquet(part)
            raw = raw[[c for c in keep if c in raw.columns]]
        raw["TradeDate"] = pd.to_datetime(raw["TradeDate"]).dt.normalize()
        m = (raw["TradeDate"] >= AUDIT_WINDOW_START) & (raw["TradeDate"] <= AUDIT_WINDOW_END)
        raw = raw.loc[m]
        if raw.empty:
            continue
        q_dates = pd.DatetimeIndex(raw["TradeDate"].unique()).sort_values()
        q_dates = q_dates.intersection(audit_dates)
        for name in names:
            if name not in raw.columns:
                continue
            f = raw.pivot_table(index="TradeDate", columns="Symbol", values=name, aggfunc="last")
            f.index = pd.to_datetime(f.index).normalize()
            f = f.reindex(index=q_dates)
            me = mask_ex.reindex_like(f)
            ml = mask_leg.reindex_like(f)
            fe = f.where(me == 1)
            fl = f.where(ml == 1)
            arr_e = fe.to_numpy(dtype=float)
            arr_l = fl.to_numpy(dtype=float)
            nfin_e, ntot_e = int(np.isfinite(arr_e).sum()), int(arr_e.size)
            nfin_l, ntot_l = int(np.isfinite(arr_l).sum()), int(arr_l.size)
            for h in HORIZONS:
                ye = y_ex[h].reindex_like(f)
                yl = y_leg[h].reindex_like(f)
                ic_ex[(name, h)].append(daily_rank_ic_series(fe, ye.where(me == 1)))
                ic_leg[(name, h)].append(daily_rank_ic_series(fl, yl.where(ml == 1)))
                hle, dece = daily_hl_and_deciles(fe, ye.where(me == 1))
                hll, decl = daily_hl_and_deciles(fl, yl.where(ml == 1))
                hl_ex[(name, h)].append(hle)
                hl_leg[(name, h)].append(hll)
                dec_ex[(name, h)].append(dece)
                dec_leg[(name, h)].append(decl)
                cov_ex[(name, h)][0] += nfin_e
                cov_ex[(name, h)][1] += ntot_e
                cov_leg[(name, h)][0] += nfin_l
                cov_leg[(name, h)][1] += ntot_l
                mi = residual_mutual_information(fe, ye.where(me == 1), max_samples=MI_MAX_SAMPLES)
                if np.isfinite(mi):
                    mi_ex[(name, h)].append(mi)
        del raw

    horizon_rows = []
    temporal_rows = []
    for name in names:
        for h in HORIZONS:
            ice = pd.concat(ic_ex[(name, h)], axis=0) if ic_ex[(name, h)] else pd.Series(dtype=float)
            icl = pd.concat(ic_leg[(name, h)], axis=0) if ic_leg[(name, h)] else pd.Series(dtype=float)
            hle = pd.concat(hl_ex[(name, h)], axis=0) if hl_ex[(name, h)] else pd.Series(dtype=float)
            hll = pd.concat(hl_leg[(name, h)], axis=0) if hl_leg[(name, h)] else pd.Series(dtype=float)
            dece = pd.concat(dec_ex[(name, h)], axis=0) if dec_ex[(name, h)] else pd.DataFrame()
            decl = pd.concat(dec_leg[(name, h)], axis=0) if dec_leg[(name, h)] else pd.DataFrame()
            ce, cl = cov_ex[(name, h)], cov_leg[(name, h)]
            ex = ic_from_daily(ice, ce[0] / ce[1] if ce[1] else float("nan"))
            leg = ic_from_daily(icl, cl[0] / cl[1] if cl[1] else float("nan"))
            ex.update(hl_from_daily(hle))
            leg.update(hl_from_daily(hll))
            ex["monotonicity"] = monotonicity_from_deciles(dece)
            leg["monotonicity"] = monotonicity_from_deciles(decl)
            mi_vals = mi_ex[(name, h)]
            ex["mutual_information"] = float(np.median(mi_vals)) if mi_vals else float("nan")
            per = _period_ic(ice, audit_dates)
            per_hl = {}
            hle.index = pd.to_datetime(hle.index).normalize()
            for pname, start, end in PERIODS:
                idx = slice_dates(pd.DatetimeIndex(hle.index), start, end)
                per_hl[pname] = hl_from_daily(hle.reindex(idx))
            dominated = one_period_dominated(per)
            rec = {
                "factor": name,
                "horizon": int(h),
                "rank_ic_mean": ex["rank_ic_mean"],
                "icir": ex["icir"],
                "positive_ic_fraction": ex["positive_ic_fraction"],
                "sign_consistency": ex["sign_consistency"],
                "hl_annu_ret": ex["hl_annu_ret"],
                "hl_sharpe": ex["hl_sharpe"],
                "monotonicity": ex["monotonicity"],
                "coverage": ex["coverage"],
                "n_ic_days": ex["n_ic_days"],
                "mutual_information": ex["mutual_information"],
                "legacy_rank_ic": leg["rank_ic_mean"],
                "exec_rank_ic": ex["rank_ic_mean"],
                "ic_delta": (
                    ex["rank_ic_mean"] - leg["rank_ic_mean"]
                    if np.isfinite(ex["rank_ic_mean"]) and np.isfinite(leg["rank_ic_mean"])
                    else float("nan")
                ),
                "sign_preserved": bool(
                    np.isfinite(leg["rank_ic_mean"])
                    and np.isfinite(ex["rank_ic_mean"])
                    and abs(leg["rank_ic_mean"]) >= 0.008
                    and np.sign(leg["rank_ic_mean"]) == np.sign(ex["rank_ic_mean"])
                ),
                "legacy_hl_sharpe": leg["hl_sharpe"],
                "exec_hl_sharpe": ex["hl_sharpe"],
                "sharpe_delta": (
                    ex["hl_sharpe"] - leg["hl_sharpe"]
                    if np.isfinite(ex["hl_sharpe"]) and np.isfinite(leg["hl_sharpe"])
                    else float("nan")
                ),
                "legacy_monotonicity": leg["monotonicity"],
                "exec_monotonicity": ex["monotonicity"],
                "one_period_dominated": dominated,
                "legacy": {
                    "rank_ic_mean": leg["rank_ic_mean"],
                    "hl_sharpe": leg["hl_sharpe"],
                    "monotonicity": leg["monotonicity"],
                },
            }
            for pname, _, _ in PERIODS:
                rec["ic_{}".format(pname)] = per[pname]
                rec["hl_sharpe_{}".format(pname)] = per_hl[pname]["hl_sharpe"]
            horizon_rows.append(rec)
            temporal_rows.append(
                {
                    "factor": name,
                    "horizon": int(h),
                    **{"rank_ic_{}".format(p[0]): per[p[0]] for p in PERIODS},
                    **{"n_{}".format(p[0]): per[p[0] + "_n"] for p in PERIODS},
                    **{"hl_sharpe_{}".format(p[0]): per_hl[p[0]]["hl_sharpe"] for p in PERIODS},
                    "sign_consistency": ex["sign_consistency"],
                    "one_period_dominated": dominated,
                }
            )
    return pd.DataFrame(horizon_rows), pd.DataFrame(temporal_rows)


def _write_report(
    *,
    class_tab: pd.DataFrame,
    fam: pd.DataFrame,
    pool: pd.DataFrame,
    counts: dict,
    runtime_s: float,
    verdict: str,
    blockers: List[str],
) -> None:
    core = class_tab.loc[class_tab["classification_primary"] == "CORE_ALPHA"]
    aux = class_tab.loc[class_tab["classification_primary"] == "AUXILIARY_ALPHA"]
    nl = class_tab.loc[class_tab["classification_primary"] == "NONLINEAR_REVIEW"]
    decay = class_tab.loc[class_tab["classification_primary"] == "DECAY_TIMING_SENSITIVE"]
    lines = [
        "# 11 — Executable Factor Qualification (EQ-1)",
        "",
        "Window: **2023-01-01 → 2024-12-31**. Contract: **{}**.".format(PRIMARY_EXECUTION_CONTRACT),
        "127 FS-eligible frozen formulas. No LightGBM/XGBoost. No TC-2 descendants.",
        "",
        "Gates were frozen before factor names were classified. See `factor_qualification/gates.json`.",
        "",
        "## Counts",
        "",
        "| class | n |",
        "|---|---:|",
        "| CORE_ALPHA | {} |".format(counts.get("CORE_ALPHA", 0)),
        "| AUXILIARY_ALPHA | {} |".format(counts.get("AUXILIARY_ALPHA", 0)),
        "| NONLINEAR_REVIEW | {} |".format(counts.get("NONLINEAR_REVIEW", 0)),
        "| DECAY_TIMING_SENSITIVE | {} |".format(counts.get("DECAY_TIMING_SENSITIVE", 0)),
        "| DROP | {} |".format(counts.get("DROP", 0)),
        "",
        "Provisional ML pool (not a target): CORE={}, CORE+AUX={}, CORE+AUX+NL={}.".format(
            counts.get("CORE_ALPHA", 0),
            counts.get("CORE_ALPHA", 0) + counts.get("AUXILIARY_ALPHA", 0),
            counts.get("CORE_ALPHA", 0)
            + counts.get("AUXILIARY_ALPHA", 0)
            + counts.get("NONLINEAR_REVIEW", 0),
        ),
        "",
        "## CORE_ALPHA",
        "",
        core[["factor_name", "family", "best_horizon", "best_rank_ic", "best_hl_sharpe", "best_monotonicity", "secondary_flags"]].to_csv(index=False)
        if len(core)
        else "(none)",
        "",
        "## AUXILIARY_ALPHA",
        "",
        aux[["factor_name", "family", "best_horizon", "best_rank_ic", "best_hl_sharpe", "best_monotonicity", "secondary_flags"]].to_csv(index=False)
        if len(aux)
        else "(none)",
        "",
        "## NONLINEAR_REVIEW",
        "",
        nl[["factor_name", "family", "best_horizon", "best_rank_ic", "best_mi", "secondary_flags"]].to_csv(index=False)
        if len(nl)
        else "(none)",
        "",
        "## DECAY_TIMING_SENSITIVE (primary)",
        "",
        decay[["factor_name", "family", "best_horizon", "best_rank_ic", "secondary_flags"]].to_csv(index=False)
        if len(decay)
        else "(none)",
        "",
        "## Family summary",
        "",
        fam.to_csv(index=False) if len(fam) else "(empty)",
        "",
        "## TC-2 parent pool (proposed, not generated)",
        "",
        pool.to_csv(index=False) if len(pool) else "(empty)",
        "",
        "Positive-control names (pre-registered): {}.".format(", ".join(POSITIVE_CONTROL_PARENTS)),
        "",
        "## Runtime / blockers",
        "",
        "{:.1f} seconds. Blockers: {}.".format(runtime_s, blockers or "none"),
        "",
        "Verdict: **{}**".format(verdict),
        "",
        "Do not start TC-2 until this classification is inspected.",
        "",
    ]
    (REPORTS / "11_executable_factor_qualification.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.time()
    ensure_layout()
    FACTOR_QUALIFICATION.mkdir(parents=True, exist_ok=True)
    gates = gates_dict()
    _write_json(FACTOR_QUALIFICATION / "gates.json", gates)
    _log("gates frozen to factor_qualification/gates.json")

    blockers: List[str] = []
    verdict = "READY_FOR_TARGETED_TC2"
    class_tab = pd.DataFrame()
    fam = pd.DataFrame()
    pool = pd.DataFrame()
    counts: Dict[str, int] = {}

    try:
        try:
            y_ex = load_production_labels()
        except Exception as exc:
            blockers.append("BLOCKED_BY_EXECUTABLE_LABELS")
            raise RuntimeError("executable V2V labels missing: {}".format(exc))

        inv = _eligible()
        names = inv["factor_name"].astype(str).tolist()
        families = dict(zip(inv["factor_name"].astype(str), inv["factor_family"].astype(str)))
        clusters = dict(
            zip(
                inv["factor_name"].astype(str),
                inv["redundancy_cluster_080"].fillna("").astype(str),
            )
        )
        if len(names) != 127:
            blockers.append("n_eligible={}".format(len(names)))

        dates = pd.DatetimeIndex(y_ex[1].index).normalize().unique().sort_values()
        audit_dates = dates[(dates >= AUDIT_WINDOW_START) & (dates <= AUDIT_WINDOW_END)]
        if len(audit_dates) < 200:
            blockers.append("BLOCKED_BY_FACTOR_COVERAGE")
            raise RuntimeError("too few audit dates: {}".format(len(audit_dates)))

        _log("loading tradability")
        signal_t, entry_t1 = _load_tradability(dates)
        _log("loading legacy C2C diagnostic labels")
        y_leg = _load_or_build_legacy(dates)
        y_ex_a = {h: y_ex[h].reindex(index=audit_dates) for h in HORIZONS}
        y_leg_a = {h: y_leg[h].reindex(index=audit_dates) for h in HORIZONS}
        mask_ex = entry_t1.reindex(index=audit_dates)
        mask_leg = signal_t.reindex(index=audit_dates)

        _log("scoring 127 factors x 5 horizons")
        horizon_tab, temporal = run_metrics(
            names, y_ex_a, y_leg_a, mask_ex, mask_leg, audit_dates
        )
        horizon_tab["family"] = horizon_tab["factor"].map(families)
        temporal["family"] = temporal["factor"].map(families)

        # classify
        class_rows = []
        grouped = {n: [] for n in names}
        for rec in horizon_tab.to_dict("records"):
            grouped[rec["factor"]].append(rec)
        prelim = []
        for name in names:
            recs = grouped[name]
            for r in recs:
                r["legacy"] = {
                    "rank_ic_mean": r.get("legacy_rank_ic", np.nan),
                    "hl_sharpe": r.get("legacy_hl_sharpe", np.nan),
                }
            prelim.append(
                classify_factor(name, families.get(name, ""), recs, cluster=clusters.get(name, ""))
            )
        prelim_tab = pd.DataFrame(prelim)
        cluster_counts = (
            prelim_tab.loc[
                prelim_tab["classification_primary"].isin(("CORE_ALPHA", "AUXILIARY_ALPHA"))
            ]
            .groupby("redundancy_cluster_080")
            .size()
            .to_dict()
        )
        class_rows = []
        for name in names:
            recs = grouped[name]
            cl = clusters.get(name, "")
            n_cl = int(cluster_counts.get(cl, 1)) if cl else 1
            class_rows.append(
                classify_factor(
                    name,
                    families.get(name, ""),
                    recs,
                    cluster=cl,
                    cluster_core_aux_count=n_cl,
                )
            )
        class_tab = pd.DataFrame(class_rows)
        counts = class_tab["classification_primary"].value_counts().to_dict()
        mean_cov = float(horizon_tab["coverage"].mean()) if len(horizon_tab) else 0.0
        if mean_cov < 0.30:
            blockers.append("BLOCKED_BY_FACTOR_COVERAGE")
            verdict = "BLOCKED_BY_FACTOR_COVERAGE"

        fam = family_qualification_summary(class_tab, horizon_tab)
        pool = build_tc2_parent_pool(class_tab)

        # split outputs
        metric_cols = [
            "factor",
            "family",
            "horizon",
            "rank_ic_mean",
            "icir",
            "positive_ic_fraction",
            "sign_consistency",
            "hl_annu_ret",
            "hl_sharpe",
            "monotonicity",
            "coverage",
            "n_ic_days",
            "mutual_information",
            "one_period_dominated",
        ]
        deg_cols = [
            "factor",
            "family",
            "horizon",
            "legacy_rank_ic",
            "exec_rank_ic",
            "ic_delta",
            "sign_preserved",
            "legacy_hl_sharpe",
            "exec_hl_sharpe",
            "sharpe_delta",
            "legacy_monotonicity",
            "exec_monotonicity",
        ]
        horizon_tab[metric_cols].to_csv(
            FACTOR_QUALIFICATION / "executable_factor_horizon_metrics.csv", index=False
        )
        best = class_tab.copy()
        best.to_csv(FACTOR_QUALIFICATION / "executable_factor_metrics.csv", index=False)
        horizon_tab[deg_cols].to_csv(
            FACTOR_QUALIFICATION / "legacy_vs_v2v_degradation.csv", index=False
        )
        class_tab.to_csv(FACTOR_QUALIFICATION / "factor_classification.csv", index=False)
        fam.to_csv(FACTOR_QUALIFICATION / "family_qualification_summary.csv", index=False)
        temporal.to_csv(FACTOR_QUALIFICATION / "temporal_stability.csv", index=False)
        pool.to_csv(FACTOR_QUALIFICATION / "tc2_parent_pool.csv", index=False)

        n_core = int(counts.get("CORE_ALPHA", 0))
        if n_core == 0 and int(counts.get("AUXILIARY_ALPHA", 0)) == 0:
            verdict = "QUALIFICATION_NEEDS_REVIEW"
        elif blockers:
            pass
        else:
            verdict = "READY_FOR_TARGETED_TC2"

    except Exception as exc:
        _log("RUNNER_ERROR {}".format(exc))
        traceback.print_exc()
        if not blockers:
            msg = str(exc).upper()
            if "LABEL" in msg:
                verdict = "BLOCKED_BY_EXECUTABLE_LABELS"
            elif "COVER" in msg:
                verdict = "BLOCKED_BY_FACTOR_COVERAGE"
            else:
                verdict = "QUALIFICATION_NEEDS_REVIEW"
            blockers.append(verdict)

    runtime_s = time.time() - t0
    payload = {
        "verdict": verdict,
        "blockers": blockers,
        "counts": counts,
        "n_tc2_parents": int(len(pool)) if len(pool) else 0,
        "runtime_seconds": runtime_s,
        "execution_contract": PRIMARY_EXECUTION_CONTRACT,
    }
    _write_json(FACTOR_QUALIFICATION / "eq1_status.json", payload)
    try:
        _write_report(
            class_tab=class_tab if len(class_tab) else pd.DataFrame(),
            fam=fam,
            pool=pool,
            counts=counts,
            runtime_s=runtime_s,
            verdict=verdict,
            blockers=blockers,
        )
    except Exception:
        traceback.print_exc()
    _log("VERDICT {}".format(verdict))
    _log("COUNTS {}".format(counts))
    _log("RUNTIME_S {:.1f}".format(runtime_s))
    return 0 if verdict in ("READY_FOR_TARGETED_TC2", "READY_WITH_MINOR_FIXES") else 2


if __name__ == "__main__":
    sys.exit(main())
