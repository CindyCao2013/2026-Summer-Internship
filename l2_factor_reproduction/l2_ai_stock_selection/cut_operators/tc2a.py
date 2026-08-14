"""TC-2A targeted nonlinear rescue and timing localization.

Does not train trees. Does not prune CORE/AUX. Does not expand to all
25 TC-2 parents. Production eval is EXEC_V2V_TPLUS1_V1 only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.feature_selection.panel_io import partitions_overlapping
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.apply import (
    apply_tc2a_recipes,
    availability_for_recipe,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    PRODUCTION_EXECUTION_CONTRACT,
    RESCUE_CORE_GATES,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.loaders import (
    build_tc2a_panel,
    load_ch_ssl2_month,
    load_ch_tick_signed_month,
    load_ddb_minutes_month,
    month_windows,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.registry import (
    assert_candidate_pool_unchanged,
    snapshot_candidate_pool,
)
from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.tc2a_config import (
    COMMON_CLOSE_END,
    COMMON_CLOSE_START,
    MATERIAL_IMPROVEMENT,
    OVERFLEXIBLE,
    TC2A_EXECUTION_CONTRACT,
    TC2A_MAX_DESCENDANTS_PER_PARENT,
    TC2A_NEGATIVE_CONTROL,
    TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION,
    TC2A_NEGATIVE_CONTROL_REASON,
    TC2A_PARENTS,
    TC2A_POSITIVE_CONTROL,
    TC2A_RECIPES,
    TC2A_WINDOW_END,
    TC2A_WINDOW_START,
    assert_tc2a_budget,
    parent_by_name,
)
from l2_factor_reproduction.l2_ai_stock_selection.degradation import daily_rank_ic_series
from l2_factor_reproduction.l2_ai_stock_selection.entry_investability import (
    build_entry_tradable,
)
from l2_factor_reproduction.l2_ai_stock_selection.executable_labels import (
    load_production_labels,
)
from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (
    AUDIT_WINDOW_END,
    AUDIT_WINDOW_START,
    HORIZONS,
    PRIMARY_EXECUTION_CONTRACT,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import (
    residual_mutual_information,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (
    EXECUTION,
    FACTOR_QUALIFICATION,
    REPORTS,
    TC2A_OUTPUT,
    ensure_layout,
)
from l2_factor_reproduction.l2_ai_stock_selection.qualification import (
    AUX_MIN_COVERAGE,
    AUX_MIN_EVIDENCE,
    AUX_MIN_SIGN_CONSISTENCY,
    CORE_MIN_COVERAGE,
    CORE_MIN_IC_DAYS,
    CORE_MIN_SIGN_CONSISTENCY,
    MI_MAX_SAMPLES,
    PERIODS,
    aux_evidence_count,
    daily_hl_and_deciles,
    hl_from_daily,
    ic_from_daily,
    monotonicity_from_deciles,
    one_period_dominated,
    slice_dates,
)
from l2_factor_reproduction.python.fast_discovery import context_paths

FS1_ALIGNED = (
    Path(__file__).resolve().parents[3]
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs1_feature_panel_full"
    / "aligned_raw"
)
VWAP_CACHE = EXECUTION / "cache" / "adj_vwap.parquet"
CORE_CORR_FACTORS = (
    "signed_amount_impact",
    "relative_spread_mean",
    "avg_buy_trade_size",
)


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


def write_frozen_contract(out_dir: Path) -> pd.DataFrame:
    """Must run before descendant generation."""
    assert_tc2a_budget()
    if not TC2A_NEGATIVE_CONTROL_FROZEN_BEFORE_INSPECTION:
        raise RuntimeError("negative control was not frozen")
    rows = []
    for spec in TC2A_PARENTS:
        rec = dict(spec)
        rec["negative_control"] = str(spec["parent_factor"]) == TC2A_NEGATIVE_CONTROL
        rec["positive_control"] = str(spec["parent_factor"]) == TC2A_POSITIVE_CONTROL
        rec["n_recipes"] = int(
            sum(1 for r in TC2A_RECIPES if r["parent_factor"] == spec["parent_factor"])
        )
        rec["max_descendants"] = TC2A_MAX_DESCENDANTS_PER_PARENT
        rec["execution_contract"] = TC2A_EXECUTION_CONTRACT
        rec["window_start"] = TC2A_WINDOW_START
        rec["window_end"] = TC2A_WINDOW_END
        rows.append(rec)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "tc2a_parent_contract.csv", index=False)
    rec_rows = []
    for rec in TC2A_RECIPES:
        row = dict(rec)
        parent = parent_by_name()[str(rec["parent_factor"])]
        row["parent_type"] = parent["parent_type"]
        row["underlying_primitive"] = parent["underlying_primitive"]
        row["parent_transform"] = parent["parent_transform"]
        row["cut_level"] = parent["cut_level"]
        row["research_question"] = parent["research_question"]
        row["execution_contract"] = TC2A_EXECUTION_CONTRACT
        rec_rows.append(row)
    recipes = pd.DataFrame(rec_rows)
    recipes.to_csv(out_dir / "tc2a_recipes.csv", index=False)
    freeze = {
        "negative_control": TC2A_NEGATIVE_CONTROL,
        "negative_control_reason": TC2A_NEGATIVE_CONTROL_REASON,
        "frozen_before_inspection": True,
        "positive_control": TC2A_POSITIVE_CONTROL,
        "n_parents": len(TC2A_PARENTS),
        "n_recipes": len(TC2A_RECIPES),
        "window": [TC2A_WINDOW_START, TC2A_WINDOW_END],
        "execution_contract": TC2A_EXECUTION_CONTRACT,
        "common_close": {
            "start": COMMON_CLOSE_START,
            "end": COMMON_CLOSE_END,
            "note": "robustness window, not a tuned parameter",
        },
        "do_not_expand_to_remaining_tc2_parents": True,
        "do_not_train_trees": True,
        "do_not_prune_core_aux": True,
    }
    (out_dir / "tc2a_freeze.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def generate_month(
    start,
    end,
    *,
    cache_dir: Path,
    force: bool = False,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    month_wide = cache_dir / "wide_{}.parquet".format(pd.Timestamp(start).strftime("%Y%m"))
    month_meta = cache_dir / "wide_{}.meta.json".format(pd.Timestamp(start).strftime("%Y%m"))
    if month_wide.exists() and month_meta.exists() and not force:
        wide = pd.read_parquet(month_wide)
        meta = json.loads(month_meta.read_text())
        return wide, meta.get("metas", []), meta
    timings = {}
    ddb, sec = load_ddb_minutes_month(start, end, cache_dir=cache_dir, force=force)
    timings["ddb_sec"] = sec
    ssl2, sec = load_ch_ssl2_month(start, end, cache_dir=cache_dir, force=force)
    timings["ssl2_sec"] = sec
    tick, tick_meta = load_ch_tick_signed_month(start, end, cache_dir=cache_dir, force=force)
    timings["tick_sec"] = float(tick_meta.get("load_sec") or 0.0)
    panel = build_tc2a_panel(ddb, ssl2, tick, tick_meta)
    activity = panel["large_order_activity"].to_numpy(dtype=bool) if "large_order_activity" in panel.columns else None
    p_no = float(np.mean(~activity)) if activity is not None and activity.size else float("nan")
    del ddb, ssl2, tick
    wide, metas = apply_tc2a_recipes(panel, TC2A_RECIPES)
    del panel
    extra = {
        "timings": timings,
        "tick_meta": tick_meta,
        "p_no_large_order_activity": p_no,
        "n_rows": int(len(wide)),
        "metas": [
            {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                for k, v in m.items()
            }
            for m in metas
        ],
    }
    wide.to_parquet(month_wide, index=False)
    month_meta.write_text(json.dumps(extra, indent=2, default=_json_default) + "\n")
    return wide, metas, extra


def generate_all_months(
    *,
    out_dir: Path,
    force: bool = False,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    metas: List[dict] = []
    p_no = []
    timings_all = []
    for start, end in month_windows(TC2A_WINDOW_START, TC2A_WINDOW_END):
        _log("TC-2A generate {} → {}".format(start.date(), end.date()))
        wide, month_metas, extra = generate_month(
            start, end, cache_dir=cache_dir, force=force
        )
        if metas == []:
            metas = month_metas
        if len(wide):
            frames.append(wide)
        if np.isfinite(extra.get("p_no_large_order_activity", np.nan)):
            p_no.append(float(extra["p_no_large_order_activity"]))
        timings_all.append(extra.get("timings") or {})
        _log(
            "  rows={} ddb={:.0f}s ssl2={:.0f}s tick={:.0f}s".format(
                len(wide),
                float((extra.get("timings") or {}).get("ddb_sec") or 0),
                float((extra.get("timings") or {}).get("ssl2_sec") or 0),
                float((extra.get("timings") or {}).get("tick_sec") or 0),
            )
        )
    if not frames:
        raise RuntimeError("TC-2A generated empty descendant panel")
    wide = pd.concat(frames, ignore_index=True)
    wide["TradeDate"] = pd.to_datetime(wide["TradeDate"]).dt.normalize()
    summary = {
        "p_no_large_order_activity": float(np.mean(p_no)) if p_no else float("nan"),
        "n_stock_days": int(len(wide)),
        "n_months": len(frames),
        "n_candidates": int(len([c for c in wide.columns if c not in ("TradeDate", "symbol")])),
    }
    return wide, metas, summary


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
    return maps["signal_tradable_T"]


def _period_ic(ic: pd.Series) -> Dict[str, float]:
    out = {}
    ic = ic.copy()
    ic.index = pd.to_datetime(ic.index).normalize()
    for name, start, end in PERIODS:
        idx = slice_dates(pd.DatetimeIndex(ic.index), start, end)
        sub = ic.reindex(idx).dropna()
        out[name] = float(sub.mean()) if len(sub) else float("nan")
        out[name + "_n"] = int(len(sub))
    return out


def _to_wide(long_df: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = long_df.loc[:, ["TradeDate", "symbol", col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    f = sub.pivot_table(index="TradeDate", columns="symbol", values=col, aggfunc="last")
    f.index = pd.to_datetime(f.index).normalize()
    return f


def _load_aligned_factor(name: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    parts = partitions_overlapping(FS1_ALIGNED, AUDIT_WINDOW_START, AUDIT_WINDOW_END)
    frames = []
    for part in parts:
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(str(part)).schema.names)
            cols = [c for c in ("TradeDate", "Symbol", name) if c in available]
            if name not in cols:
                continue
            raw = pd.read_parquet(part, columns=cols)
        except Exception:
            raw = pd.read_parquet(part)
            if name not in raw.columns:
                continue
            raw = raw[["TradeDate", "Symbol", name]]
        raw["TradeDate"] = pd.to_datetime(raw["TradeDate"]).dt.normalize()
        raw = raw.loc[(raw["TradeDate"] >= dates.min()) & (raw["TradeDate"] <= dates.max())]
        if raw.empty:
            continue
        f = raw.pivot_table(index="TradeDate", columns="Symbol", values=name, aggfunc="last")
        f.index = pd.to_datetime(f.index).normalize()
        frames.append(f)
        del raw
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _panel_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    common = a.index.intersection(b.index)
    cols = a.columns.intersection(b.columns)
    if len(common) == 0 or len(cols) == 0:
        return float("nan")
    xa = a.loc[common, cols].to_numpy(dtype=float).ravel()
    xb = b.loc[common, cols].to_numpy(dtype=float).ravel()
    ok = np.isfinite(xa) & np.isfinite(xb)
    if int(ok.sum()) < 200:
        return float("nan")
    return float(np.corrcoef(xa[ok], xb[ok])[0, 1])


def evaluate_factor_panel(
    factor: pd.DataFrame,
    y: pd.DataFrame,
    mask: pd.DataFrame,
    y_leg: Optional[pd.DataFrame] = None,
) -> dict:
    f = factor.copy()
    f.index = pd.to_datetime(f.index).normalize()
    yy = y.reindex_like(f)
    m = mask.reindex_like(f)
    fe = f.where(m == 1)
    ye = yy.where(m == 1)
    ic = daily_rank_ic_series(fe, ye)
    hl, dec = daily_hl_and_deciles(fe, ye)
    arr = fe.to_numpy(dtype=float)
    cov = float(np.isfinite(arr).mean()) if arr.size else float("nan")
    out = ic_from_daily(ic, cov)
    out.update(hl_from_daily(hl))
    out["monotonicity"] = monotonicity_from_deciles(dec)
    out["mutual_information"] = residual_mutual_information(
        fe, ye, max_samples=MI_MAX_SAMPLES
    )
    per = _period_ic(ic)
    out["one_period_dominated"] = one_period_dominated(per)
    hl.index = pd.to_datetime(hl.index).normalize()
    for pname, start, end in PERIODS:
        idx = slice_dates(pd.DatetimeIndex(hl.index), start, end)
        out["hl_sharpe_{}".format(pname)] = hl_from_daily(hl.reindex(idx))["hl_sharpe"]
        out["ic_{}".format(pname)] = per[pname]
    if y_leg is not None:
        yl = y_leg.reindex_like(f).where(m == 1)
        icl = daily_rank_ic_series(fe, yl)
        out["legacy_rank_ic"] = float(icl.mean()) if len(icl.dropna()) else float("nan")
        hll, _ = daily_hl_and_deciles(fe, yl)
        out["legacy_hl_sharpe"] = hl_from_daily(hll)["hl_sharpe"]
    else:
        out["legacy_rank_ic"] = float("nan")
        out["legacy_hl_sharpe"] = float("nan")
    return out


def _finite_abs(x) -> float:
    v = float(x) if x is not None and np.isfinite(x) else float("nan")
    return abs(v) if np.isfinite(v) else float("nan")


def _materially_improved(child: Mapping, parent: Mapping) -> bool:
    d_ic = _finite_abs(child.get("rank_ic_mean")) - _finite_abs(parent.get("rank_ic_mean"))
    d_hl = _finite_abs(child.get("hl_sharpe")) - _finite_abs(parent.get("hl_sharpe"))
    d_mo = _finite_abs(child.get("monotonicity")) - _finite_abs(parent.get("monotonicity"))
    d_mi = float(child.get("mutual_information", np.nan)) - float(
        parent.get("mutual_information", np.nan)
    )
    if np.isfinite(d_ic) and d_ic >= MATERIAL_IMPROVEMENT["delta_abs_ic"]:
        return True
    if np.isfinite(d_hl) and d_hl >= MATERIAL_IMPROVEMENT["delta_hl_sharpe"]:
        return True
    if np.isfinite(d_mo) and d_mo >= MATERIAL_IMPROVEMENT["delta_monotonicity"]:
        return True
    if np.isfinite(d_mi) and d_mi >= 0.005:
        return True
    return False


def classify_descendant(
    *,
    child: Mapping,
    parent: Mapping,
    parent_type_q: str,
    corr_parent: float,
    corr_core: float,
) -> str:
    ic_ok = _finite_abs(child.get("rank_ic_mean")) >= RESCUE_CORE_GATES["abs_rank_ic"]
    hl_ok = _finite_abs(child.get("hl_sharpe")) >= RESCUE_CORE_GATES["hl_sharpe"]
    mo_ok = _finite_abs(child.get("monotonicity")) >= RESCUE_CORE_GATES["monotonicity"]
    cov_ok = float(child.get("coverage", np.nan)) >= CORE_MIN_COVERAGE
    sc_ok = float(child.get("sign_consistency", np.nan)) >= CORE_MIN_SIGN_CONSISTENCY
    n_ok = int(child.get("n_ic_days", 0) or 0) >= CORE_MIN_IC_DAYS
    stable = cov_ok and sc_ok and n_ok and (not bool(child.get("one_period_dominated", False)))
    improved = _materially_improved(child, parent)
    redundant_parent = (
        np.isfinite(corr_parent) and abs(corr_parent) >= MATERIAL_IMPROVEMENT["parent_child_redundant_corr"]
    )
    redundant_core = (
        np.isfinite(corr_core) and abs(corr_core) >= MATERIAL_IMPROVEMENT["core_redundant_corr"]
    )
    if parent_type_q == "POSITIVE_CONTROL":
        if ic_ok and hl_ok and mo_ok and stable:
            return "POSITIVE_CONTROL_CONFIRM"
        return "POSITIVE_CONTROL_WEAK_CUT"
    if parent_type_q == "NEGATIVE_CONTROL":
        return "NEGATIVE_CONTROL_NOT_PROMOTED"
    if (ic_ok and hl_ok and mo_ok and stable) and (redundant_core or redundant_parent):
        return "REDUNDANT_RESCUE"
    if ic_ok and hl_ok and mo_ok and stable and improved:
        return "RESCUED_CORE"
    n_aux, _ = aux_evidence_count(
        {
            "rank_ic_mean": child.get("rank_ic_mean"),
            "hl_sharpe": child.get("hl_sharpe"),
            "monotonicity": child.get("monotonicity"),
        }
    )
    aux_ok = (
        n_aux >= AUX_MIN_EVIDENCE
        and float(child.get("coverage", np.nan)) >= AUX_MIN_COVERAGE
        and float(child.get("sign_consistency", np.nan)) >= AUX_MIN_SIGN_CONSISTENCY
    )
    if aux_ok and improved and not redundant_parent:
        return "RESCUED_AUX"
    mi = float(child.get("mutual_information", np.nan))
    pmi = float(parent.get("mutual_information", np.nan))
    mi_up = np.isfinite(mi) and (
        mi >= 0.01 or (np.isfinite(pmi) and mi > pmi + 0.005)
    )
    if mi_up and (not ic_ok):
        return "NONLINEAR_ONLY"
    return "FAILED_RESCUE"


def classify_timing(parent_name: str, child_rows: pd.DataFrame) -> str:
    if child_rows.empty:
        return "NO_CLEAR_TIMING_STRUCTURE"
    abs_ic = child_rows["rank_ic_mean"].abs()
    abs_leg = child_rows["legacy_rank_ic"].abs() if "legacy_rank_ic" in child_rows.columns else abs_ic * np.nan
    best = float(abs_ic.max()) if abs_ic.notna().any() else float("nan")
    second = float(abs_ic.nlargest(2).iloc[-1]) if int(abs_ic.notna().sum()) >= 2 else 0.0
    localized = np.isfinite(best) and (best - second >= 0.005 or best >= 1.5 * max(second, 1e-6))
    v2v_usable = np.isfinite(best) and best >= 0.008
    c2c_best = float(abs_leg.max()) if abs_leg.notna().any() else float("nan")
    c2c_clear = np.isfinite(c2c_best) and c2c_best >= 0.008
    if localized and v2v_usable:
        return "TIMING_LOCALIZED_EXECUTABLE"
    if (localized or c2c_clear) and (not v2v_usable):
        return "TIMING_LOCALIZED_TOO_FAST"
    return "NO_CLEAR_TIMING_STRUCTURE"


def _parent_eq1_row(horizon_tab: pd.DataFrame, name: str, horizon: int) -> dict:
    sub = horizon_tab.loc[
        (horizon_tab["factor"] == name) & (horizon_tab["horizon"].astype(int) == int(horizon))
    ]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def evaluate_tc2a(
    wide: pd.DataFrame,
    metas: Sequence[Mapping],
    *,
    out_dir: Path,
    p_no_large: float,
) -> Dict[str, pd.DataFrame]:
    y_ex = load_production_labels()
    cache = FACTOR_QUALIFICATION / "cache"
    y_leg: Dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        p = cache / "legacy_c2c_forward_return_{}d.parquet".format(h)
        if p.exists():
            yy = pd.read_parquet(p)
            yy.index = pd.to_datetime(yy.index).normalize()
            y_leg[int(h)] = yy
    dates = pd.DatetimeIndex(pd.to_datetime(wide["TradeDate"]).unique()).sort_values()
    dates = dates[(dates >= AUDIT_WINDOW_START) & (dates <= AUDIT_WINDOW_END)]
    mask = _load_tradability(dates)
    horizon_tab = pd.read_csv(FACTOR_QUALIFICATION / "executable_factor_horizon_metrics.csv")
    parents = parent_by_name()
    names = [c for c in wide.columns if c not in ("TradeDate", "symbol")]
    recipe_map = {str(r["candidate_name"]): r for r in TC2A_RECIPES}
    parent_panels: Dict[str, pd.DataFrame] = {}
    core_panels: Dict[str, pd.DataFrame] = {}
    need_aligned = sorted(set(list(parents) + list(CORE_CORR_FACTORS)))
    for fname in need_aligned:
        _log("  load aligned {}".format(fname))
        panel = _load_aligned_factor(fname, dates)
        if fname in parents:
            parent_panels[fname] = panel
        if fname in CORE_CORR_FACTORS:
            core_panels[fname] = panel

    metric_rows = []
    compare_rows = []
    for i, name in enumerate(names, start=1):
        rec = recipe_map.get(name, {})
        parent_name = str(rec.get("parent_factor") or "")
        parent_spec = parents.get(parent_name, {})
        h = int(parent_spec.get("best_horizon") or 1)
        _log("  eval {}/{} {} h={}".format(i, len(names), name, h))
        f = _to_wide(wide, name)
        f = f.reindex(index=dates)
        ye = y_ex[h]
        yl = y_leg.get(h)
        stats = evaluate_factor_panel(f, ye, mask, y_leg=yl)
        parent_row = _parent_eq1_row(horizon_tab, parent_name, h)
        corr_p = _panel_corr(f, parent_panels[parent_name]) if parent_name in parent_panels else float("nan")
        corr_core = float("nan")
        for cname, cp in core_panels.items():
            val = _panel_corr(f, cp)
            if np.isfinite(val) and (not np.isfinite(corr_core) or abs(val) > abs(corr_core)):
                corr_core = val
        resid_ic = float("nan")
        resid_mi = float("nan")
        if TC2A_POSITIVE_CONTROL in core_panels and not core_panels[TC2A_POSITIVE_CONTROL].empty:
            ctrl = core_panels[TC2A_POSITIVE_CONTROL].reindex_like(f)
            # cheap residual: y residualized on one CORE factor, then RankIC
            ye_h = ye.reindex_like(f)
            m = mask.reindex_like(f)
            y_res = ye_h.where(m == 1).copy()
            c_al = ctrl.where(m == 1)
            common = y_res.index.intersection(c_al.index)
            cols = y_res.columns.intersection(c_al.columns)
            if len(common) and len(cols):
                yr = y_res.loc[common, cols]
                cr = c_al.loc[common, cols]
                resid = yr.copy()
                for dt in yr.index:
                    yv = yr.loc[dt].to_numpy(dtype=float)
                    xv = cr.loc[dt].to_numpy(dtype=float)
                    ok = np.isfinite(yv) & np.isfinite(xv)
                    if int(ok.sum()) < 30:
                        continue
                    x1 = np.column_stack([np.ones(int(ok.sum())), xv[ok]])
                    beta, _, _, _ = np.linalg.lstsq(x1, yv[ok], rcond=None)
                    e = np.full(yv.shape, np.nan)
                    e[ok] = yv[ok] - x1 @ beta
                    resid.loc[dt] = e
                resid_ic = float(daily_rank_ic_series(f.reindex_like(resid), resid).mean())
                resid_mi = residual_mutual_information(
                    f.reindex_like(resid), resid, max_samples=MI_MAX_SAMPLES
                )
        child_pack = {
            "rank_ic_mean": stats["rank_ic_mean"],
            "hl_sharpe": stats["hl_sharpe"],
            "monotonicity": stats["monotonicity"],
            "mutual_information": stats["mutual_information"],
            "coverage": stats["coverage"],
            "sign_consistency": stats["sign_consistency"],
            "n_ic_days": stats["n_ic_days"],
            "one_period_dominated": stats["one_period_dominated"],
        }
        parent_pack = {
            "rank_ic_mean": parent_row.get("rank_ic_mean"),
            "hl_sharpe": parent_row.get("hl_sharpe"),
            "monotonicity": parent_row.get("monotonicity"),
            "mutual_information": parent_row.get("mutual_information"),
        }
        status = classify_descendant(
            child=child_pack,
            parent=parent_pack,
            parent_type_q=str(parent_spec.get("research_question") or ""),
            corr_parent=corr_p,
            corr_core=corr_core,
        )
        d_ic = _finite_abs(stats["rank_ic_mean"]) - _finite_abs(parent_row.get("rank_ic_mean"))
        d_hl = _finite_abs(stats["hl_sharpe"]) - _finite_abs(parent_row.get("hl_sharpe"))
        d_mo = _finite_abs(stats["monotonicity"]) - _finite_abs(parent_row.get("monotonicity"))
        d_mi = float(stats["mutual_information"]) - float(parent_row.get("mutual_information", np.nan))
        metric_rows.append(
            {
                "candidate_name": name,
                "parent_factor": parent_name,
                "parent_type": parent_spec.get("parent_type"),
                "research_question": parent_spec.get("research_question"),
                "horizon": h,
                "execution_contract": PRODUCTION_EXECUTION_CONTRACT,
                "cut_type": rec.get("cut_type"),
                "cut_name": rec.get("cut_name"),
                "reason": rec.get("reason"),
                **stats,
                "residual_rank_ic": resid_ic,
                "residual_mi": resid_mi,
                "corr_parent": corr_p,
                "corr_core": corr_core,
                "delta_abs_IC": d_ic,
                "delta_HL_Sharpe": d_hl,
                "delta_monotonicity": d_mo,
                "delta_MI": d_mi,
                "rescue_status": status,
                "effective_close_start": rec.get("cut_name") in ("close", "close_minus_open") and "14:30:00" or "",
                "p_no_large_order_activity": p_no_large if parent_name == "large_order_pressure" else float("nan"),
            }
        )
        compare_rows.append(
            {
                "parent_factor": parent_name,
                "child": name,
                "parent_rank_ic": parent_row.get("rank_ic_mean"),
                "child_rank_ic": stats["rank_ic_mean"],
                "delta_abs_rank_ic": d_ic,
                "parent_hl_sharpe": parent_row.get("hl_sharpe"),
                "child_hl_sharpe": stats["hl_sharpe"],
                "delta_hl_sharpe": d_hl,
                "parent_monotonicity": parent_row.get("monotonicity"),
                "child_monotonicity": stats["monotonicity"],
                "delta_monotonicity": d_mo,
                "parent_mi": parent_row.get("mutual_information"),
                "child_mi": stats["mutual_information"],
                "delta_mi": d_mi,
                "correlation_parent_child": corr_p,
                "correlation_to_existing_core": corr_core,
                "child_residual_ic": resid_ic,
                "rescue_status": status,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    compare = pd.DataFrame(compare_rows)
    timing_rows = []
    for spec in TC2A_PARENTS:
        if spec["research_question"] != "TIMING_LOCALIZATION":
            continue
        sub = metrics.loc[metrics["parent_factor"] == spec["parent_factor"]]
        timing_rows.append(
            {
                "parent_factor": spec["parent_factor"],
                "parent_type": spec["parent_type"],
                "n_descendants": int(len(sub)),
                "best_child": (sub.loc[sub["rank_ic_mean"].abs().idxmax(), "candidate_name"] if len(sub) else ""),
                "best_abs_ic": float(sub["rank_ic_mean"].abs().max()) if len(sub) else float("nan"),
                "best_hl_sharpe": (
                    float(sub.loc[sub["rank_ic_mean"].abs().idxmax(), "hl_sharpe"]) if len(sub) else float("nan")
                ),
                "timing_status": classify_timing(str(spec["parent_factor"]), sub),
            }
        )
    timing = pd.DataFrame(timing_rows)
    nc = metrics.loc[metrics["parent_factor"] == TC2A_NEGATIVE_CONTROL].copy()
    nc_core = int((nc["rescue_status"] == "RESCUED_CORE").sum())  # should be 0 by construction
    # Count how many *would* pass CORE/AUX gates ignoring the promotion block.
    would_core = 0
    would_aux = 0
    for _, row in nc.iterrows():
        ic_ok = _finite_abs(row["rank_ic_mean"]) >= RESCUE_CORE_GATES["abs_rank_ic"]
        hl_ok = _finite_abs(row["hl_sharpe"]) >= RESCUE_CORE_GATES["hl_sharpe"]
        mo_ok = _finite_abs(row["monotonicity"]) >= RESCUE_CORE_GATES["monotonicity"]
        n_aux, _ = aux_evidence_count(row)
        if ic_ok and hl_ok and mo_ok:
            would_core += 1
        elif n_aux >= AUX_MIN_EVIDENCE:
            would_aux += 1
    overflex = (
        would_core >= OVERFLEXIBLE["neg_control_core_descendants"]
        or would_aux >= OVERFLEXIBLE["neg_control_aux_descendants"]
    )
    neg = pd.DataFrame(
        [
            {
                "negative_control": TC2A_NEGATIVE_CONTROL,
                "frozen_before_inspection": True,
                "n_descendants": int(len(nc)),
                "negative_control_best_IC": float(nc["rank_ic_mean"].abs().max()) if len(nc) else float("nan"),
                "negative_control_best_Sharpe": float(nc["hl_sharpe"].abs().max()) if len(nc) else float("nan"),
                "negative_control_best_mono": float(nc["monotonicity"].abs().max()) if len(nc) else float("nan"),
                "n_would_pass_core_gates": would_core,
                "n_would_pass_aux_gates": would_aux,
                "promoted": 0,
                "CUT_SEARCH_SPACE_OVERFLEXIBLE": bool(overflex),
            }
        ]
    )
    return {
        "metrics": metrics,
        "compare": compare,
        "timing": timing,
        "negative": neg,
    }


def _verdict(metrics: pd.DataFrame, timing: pd.DataFrame, neg: pd.DataFrame) -> str:
    if bool(neg.iloc[0]["CUT_SEARCH_SPACE_OVERFLEXIBLE"]):
        return "CUT_SEARCH_SPACE_OVERFLEXIBLE"
    if metrics.empty or metrics["rank_ic_mean"].notna().sum() == 0:
        return "DATA_OR_TIMING_BLOCKER"
    n_core = int((metrics["rescue_status"] == "RESCUED_CORE").sum())
    n_aux = int((metrics["rescue_status"] == "RESCUED_AUX").sum())
    n_nl = int((metrics["rescue_status"] == "NONLINEAR_ONLY").sum())
    n_te = int((timing["timing_status"] == "TIMING_LOCALIZED_EXECUTABLE").sum()) if len(timing) else 0
    n_tf = int((timing["timing_status"] == "TIMING_LOCALIZED_TOO_FAST").sum()) if len(timing) else 0
    if n_core >= 1 and n_aux + n_core >= 3 and n_te >= 1:
        return "TC2A_RESCUE_SIGNAL_CONFIRMED"
    if n_core + n_aux + n_nl + n_te + n_tf >= 2:
        return "TC2A_USEFUL_BUT_LIMITED"
    if n_core + n_aux + n_nl + n_te == 0:
        return "NO_RESCUE_SIGNAL"
    return "TC2A_USEFUL_BUT_LIMITED"


def write_report(
    *,
    out_dir: Path,
    metrics: pd.DataFrame,
    compare: pd.DataFrame,
    timing: pd.DataFrame,
    neg: pd.DataFrame,
    registry: pd.DataFrame,
    summary: dict,
    verdict: str,
) -> None:
    n_core = int((metrics["rescue_status"] == "RESCUED_CORE").sum())
    n_aux = int((metrics["rescue_status"] == "RESCUED_AUX").sum())
    n_nl = int((metrics["rescue_status"] == "NONLINEAR_ONLY").sum())
    n_fail = int((metrics["rescue_status"] == "FAILED_RESCUE").sum())
    n_red = int((metrics["rescue_status"] == "REDUNDANT_RESCUE").sum())
    n_te = int((timing["timing_status"] == "TIMING_LOCALIZED_EXECUTABLE").sum()) if len(timing) else 0
    n_tf = int((timing["timing_status"] == "TIMING_LOCALIZED_TOO_FAST").sum()) if len(timing) else 0
    n_nc = int((timing["timing_status"] == "NO_CLEAR_TIMING_STRUCTURE").sum()) if len(timing) else 0
    lines = [
        "# 12 — TC-2A Nonlinear Rescue & Timing Localization",
        "",
        "**Window:** {} → {}".format(TC2A_WINDOW_START, TC2A_WINDOW_END),
        "**Contract:** `{}` (legacy C2C is diagnostic only)".format(PRIMARY_EXECUTION_CONTRACT),
        "**Parents:** 12 pre-registered. **Recipes:** {}. **Max descendants/parent:** {}.".format(
            len(TC2A_RECIPES), TC2A_MAX_DESCENDANTS_PER_PARENT
        ),
        "**Negative control (frozen before generation):** `{}`.".format(TC2A_NEGATIVE_CONTROL),
        "**Not done:** LightGBM/XGBoost, CORE/AUX prune, remaining TC-2 parents.",
        "",
        "Do not treat COMMON_CLOSE as a tuned parameter. DDB omits observed 14:57–14:59; those minutes are not zero-filled.",
        "",
        "## Inspection counts",
        "",
        "| item | n |",
        "|---|---:|",
        "| RESCUED_CORE | {} |".format(n_core),
        "| RESCUED_AUX | {} |".format(n_aux),
        "| NONLINEAR_ONLY | {} |".format(n_nl),
        "| REDUNDANT_RESCUE | {} |".format(n_red),
        "| FAILED_RESCUE | {} |".format(n_fail),
        "| TIMING_LOCALIZED_EXECUTABLE | {} |".format(n_te),
        "| TIMING_LOCALIZED_TOO_FAST | {} |".format(n_tf),
        "| NO_CLEAR_TIMING_STRUCTURE | {} |".format(n_nc),
        "",
        "## Negative control",
        "",
        neg.to_csv(index=False) if len(neg) else "(empty)",
        "",
        "## Timing localization",
        "",
        timing.to_csv(index=False) if len(timing) else "(none)",
        "",
        "## Rescued / control descendants",
        "",
    ]
    keep = metrics.loc[
        metrics["rescue_status"].isin(
            [
                "RESCUED_CORE",
                "RESCUED_AUX",
                "NONLINEAR_ONLY",
                "REDUNDANT_RESCUE",
                "POSITIVE_CONTROL_CONFIRM",
            ]
        )
    ]
    cols = [
        "candidate_name",
        "parent_factor",
        "rescue_status",
        "rank_ic_mean",
        "hl_sharpe",
        "monotonicity",
        "mutual_information",
        "delta_abs_IC",
        "delta_HL_Sharpe",
        "corr_parent",
        "corr_core",
        "ic_y2023",
        "ic_y2024",
    ]
    cols = [c for c in cols if c in keep.columns]
    lines.append(keep[cols].to_csv(index=False) if len(keep) else "(none met rescue/control confirm)")
    lines.extend(
        [
            "",
            "## Large-order zero activity",
            "",
            "P(no large-order activity) = {}. Zero denominators were left missing, not filled with 0.".format(
                summary.get("p_no_large_order_activity")
            ),
            "",
            "## Sample",
            "",
            "- stock-days: {}".format(summary.get("n_stock_days")),
            "- candidates: {}".format(summary.get("n_candidates")),
            "- months: {}".format(summary.get("n_months")),
            "",
            "## Verdict",
            "",
            "**{}**".format(verdict),
            "",
            "Stop rule: do not proceed automatically to the remaining TC-2 parents (TC-2B).",
            "",
        ]
    )
    (REPORTS / "12_tc2a_nonlinear_timing_rescue.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_tc2a(*, out_dir: Optional[Path] = None, force: bool = False) -> dict:
    ensure_layout()
    out_dir = Path(out_dir or TC2A_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_before = snapshot_candidate_pool()
    t0 = time.time()
    _log("TC-2A freeze negative_control={}".format(TC2A_NEGATIVE_CONTROL))
    write_frozen_contract(out_dir)
    wide, metas, summary = generate_all_months(out_dir=out_dir, force=force)
    cand_path = out_dir / "tc2a_candidates.parquet"
    wide.to_parquet(cand_path, index=False)
    _log("generated {} stock-days, {} candidates".format(len(wide), summary["n_candidates"]))
    tables = evaluate_tc2a(
        wide, metas, out_dir=out_dir, p_no_large=float(summary.get("p_no_large_order_activity") or np.nan)
    )
    metrics = tables["metrics"]
    compare = tables["compare"]
    timing = tables["timing"]
    neg = tables["negative"]
    registry_rows = []
    for rec in TC2A_RECIPES:
        row = dict(rec)
        parent = parent_by_name()[str(rec["parent_factor"])]
        row.update(availability_for_recipe(rec))
        row["parent_type"] = parent["parent_type"]
        row["underlying_primitive"] = parent["underlying_primitive"]
        row["parent_transform"] = parent["parent_transform"]
        row["cut_level"] = parent["cut_level"]
        row["execution_contract"] = PRODUCTION_EXECUTION_CONTRACT
        hit = metrics.loc[metrics["candidate_name"] == rec["candidate_name"]]
        if len(hit):
            row["rescue_status"] = hit.iloc[0]["rescue_status"]
            row["rank_ic_mean"] = hit.iloc[0]["rank_ic_mean"]
            row["hl_sharpe"] = hit.iloc[0]["hl_sharpe"]
            row["monotonicity"] = hit.iloc[0]["monotonicity"]
        registry_rows.append(row)
    registry = pd.DataFrame(registry_rows)
    metrics.to_csv(out_dir / "tc2a_candidate_metrics.csv", index=False)
    compare.to_csv(out_dir / "tc2a_parent_child_comparison.csv", index=False)
    timing.to_csv(out_dir / "tc2a_timing_localization.csv", index=False)
    neg.to_csv(out_dir / "tc2a_negative_control.csv", index=False)
    registry.to_csv(out_dir / "tc2a_registry.csv", index=False)
    verdict = _verdict(metrics, timing, neg)
    write_report(
        out_dir=out_dir,
        metrics=metrics,
        compare=compare,
        timing=timing,
        neg=neg,
        registry=registry,
        summary=summary,
        verdict=verdict,
    )
    (out_dir / "tc2a_summary.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "summary": summary,
                "runtime_sec": time.time() - t0,
                "n_rescued_core": int((metrics["rescue_status"] == "RESCUED_CORE").sum()),
                "n_rescued_aux": int((metrics["rescue_status"] == "RESCUED_AUX").sum()),
                "n_nonlinear_only": int((metrics["rescue_status"] == "NONLINEAR_ONLY").sum()),
            },
            indent=2,
            default=_json_default,
        )
        + "\n"
    )
    assert_candidate_pool_unchanged(pool_before)
    _log("TC-2A verdict={} runtime={:.0f}s".format(verdict, time.time() - t0))
    return {"verdict": verdict, "out_dir": str(out_dir), "summary": summary}
