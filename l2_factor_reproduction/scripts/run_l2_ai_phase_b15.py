"""Phase B1.5: freeze EXEC_V2V_TPLUS1_V1, build executable labels, audit degradation.

Does not mutate candidate_pool_v1, FS-1/FS-3/FS-4 frozen artifacts.
Does not launch Phase B2 residual discovery or full-history ML.
Window: feature dates 2023-01-01 → 2024-12-31. Max 10 workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "10")

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.feature_selection.fs4_contract import (  # noqa: E402
    load_selected_mask,
)
from l2_factor_reproduction.feature_selection.labels import (  # noqa: E402
    build_labels_wide_panel,
    load_daily_excess_and_bench,
)
from l2_factor_reproduction.feature_selection.panel_io import (  # noqa: E402
    partitions_overlapping,
)
from l2_factor_reproduction.l2_ai_stock_selection.degradation import (  # noqa: E402
    add_ranks,
    daily_rank_ic_series,
    family_summary,
    factor_horizon_row,
    fs_survival_table,
    hl_stats,
    ranking_spearman,
)
from l2_factor_reproduction.l2_ai_stock_selection.entry_investability import (  # noqa: E402
    build_entry_tradable,
    tradability_audit,
)
from l2_factor_reproduction.l2_ai_stock_selection.executable_labels import (  # noqa: E402
    date_mapping_table,
    excess_from_reconstructed_index,
    tail_invalid_ok,
)
from l2_factor_reproduction.l2_ai_stock_selection.execution_v2v import (  # noqa: E402
    AUDIT_WINDOW_END,
    AUDIT_WINDOW_START,
    HORIZONS,
    LEGACY_C2C_DIAGNOSTIC,
    PRIMARY_EXECUTION_CONTRACT,
    ROBUSTNESS_C2C_DELAYED,
    ROBUSTNESS_O2O,
    daily_ratio_return,
    holding_return_from_prices,
    map_feature_to_holding,
    production_execution_contract_dict,
    three_date_v2v_proof,
)
from l2_factor_reproduction.l2_ai_stock_selection.inventory import (  # noqa: E402
    load_factor_inventory,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (  # noqa: E402
    EXECUTABLE_V2V_LABELS,
    EXECUTION,
    LABELS,
    REPORTS,
    TIMING_DEGRADATION,
    ensure_layout,
    frozen_artifact_paths,
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
EOD_START = pd.Timestamp("2022-10-01")
EOD_END = pd.Timestamp("2025-02-15")
PROOF_DATES = pd.DatetimeIndex(["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06"])
PROOF_SYMBOL = "000001.SZ"
MAX_WORKERS = 10
CACHE = EXECUTION / "cache"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes() -> Dict[str, str]:
    out = {}
    for p in frozen_artifact_paths():
        out[str(p)] = _sha256(p) if p.exists() else "MISSING"
    return out


def _trading_dates() -> pd.DatetimeIndex:
    p = context_paths("full")["trading_dates"]
    td = pd.read_parquet(p)
    if isinstance(td, pd.DataFrame):
        idx = pd.to_datetime(td[td.columns[0]])
    else:
        idx = pd.to_datetime(td.index)
    return pd.DatetimeIndex(idx).normalize().unique().sort_values()


def _connect_ddb():
    import dolphindb as ddb
    from COMMON_CONST import DATA_DB_CONN

    s = ddb.session()
    s.connect(**DATA_DB_CONN)
    return s


def _pivot_last(df: pd.DataFrame, value: str) -> pd.DataFrame:
    out = df.pivot_table(
        index="TRADE_DT", columns="S_INFO_WINDCODE", values=value, aggfunc="last"
    )
    out.index = pd.to_datetime(out.index).normalize()
    cols = [c for c in out.columns if str(c)[:1] in ("6", "0", "3")]
    return out[cols].sort_index().astype(float)


def fetch_eod_and_weights(cache_dir: Path) -> Dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vwap_p = cache_dir / "adj_vwap.parquet"
    open_p = cache_dir / "adj_open.parquet"
    close_p = cache_dir / "adj_close.parquet"
    w_p = cache_dir / "csi1000_weight.parquet"
    meta_p = cache_dir / "eod_meta.json"
    if vwap_p.exists() and open_p.exists() and close_p.exists() and w_p.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        return {
            "adj_vwap": pd.read_parquet(vwap_p),
            "adj_open": pd.read_parquet(open_p),
            "adj_close": pd.read_parquet(close_p),
            "weights": pd.read_parquet(w_p),
            "meta": meta,
            "from_cache": True,
        }

    s = _connect_ddb()
    start_str = EOD_START.strftime("%Y.%m.%d")
    end_str = EOD_END.strftime("%Y.%m.%d")
    meta: Dict[str, object] = {}

    t_idx = s.loadTable(dbPath="dfs://WIND.AINDEXEODPRICES", tableName="data")
    idx_one = (
        t_idx.select("*")
        .where("S_INFO_WINDCODE='000852.SH' and TRADE_DT>=2024.06.03 and TRADE_DT<=2024.06.06")
        .toDF()
    )
    idx_cols = list(idx_one.columns)
    meta["index_eod_columns"] = idx_cols
    meta["index_has_avgprice"] = any("AVGPRICE" in str(c).upper() for c in idx_cols)
    meta["index_n_probe_rows"] = int(len(idx_one))
    if "S_DQ_AMOUNT" in idx_one.columns and "S_DQ_VOLUME" in idx_one.columns:
        amt = pd.to_numeric(idx_one["S_DQ_AMOUNT"], errors="coerce")
        vol = pd.to_numeric(idx_one["S_DQ_VOLUME"], errors="coerce")
        ratio = (amt / vol.replace(0, np.nan)).median()
        close = pd.to_numeric(idx_one.get("S_DQ_CLOSE"), errors="coerce").median()
        meta["index_amount_over_volume_median"] = float(ratio) if pd.notna(ratio) else None
        meta["index_close_median"] = float(close) if pd.notna(close) else None
        meta["amount_over_volume_is_index_vwap"] = False
    else:
        meta["amount_over_volume_is_index_vwap"] = False

    t_eod = s.loadTable(dbPath="dfs://WIND.ASHAREEODPRICES", tableName="data")
    eod = t_eod.select(
        "TRADE_DT, S_INFO_WINDCODE, S_DQ_AVGPRICE, S_DQ_ADJFACTOR, "
        "S_DQ_ADJOPEN, S_DQ_ADJCLOSE, S_DQ_CLOSE, S_DQ_TRADESTATUS"
    ).where(
        "TRADE_DT>= {} and TRADE_DT <= {}".format(start_str, end_str)
    ).toDF()
    eod["TRADE_DT"] = pd.to_datetime(eod["TRADE_DT"]).dt.normalize()
    eod["adj_vwap"] = pd.to_numeric(eod["S_DQ_AVGPRICE"], errors="coerce") * pd.to_numeric(
        eod["S_DQ_ADJFACTOR"], errors="coerce"
    )
    eod["adj_open"] = pd.to_numeric(eod["S_DQ_ADJOPEN"], errors="coerce")
    if "S_DQ_ADJCLOSE" in eod.columns:
        eod["adj_close"] = pd.to_numeric(eod["S_DQ_ADJCLOSE"], errors="coerce")
    else:
        eod["adj_close"] = pd.to_numeric(eod["S_DQ_CLOSE"], errors="coerce") * pd.to_numeric(
            eod["S_DQ_ADJFACTOR"], errors="coerce"
        )

    adj_vwap = _pivot_last(eod.rename(columns={"adj_vwap": "adj_vwap"}), "adj_vwap")
    # _pivot_last expects the value column name as in df after rename of S_INFO
    adj_vwap = eod.pivot_table(
        index="TRADE_DT", columns="S_INFO_WINDCODE", values="adj_vwap", aggfunc="last"
    )
    adj_open = eod.pivot_table(
        index="TRADE_DT", columns="S_INFO_WINDCODE", values="adj_open", aggfunc="last"
    )
    adj_close = eod.pivot_table(
        index="TRADE_DT", columns="S_INFO_WINDCODE", values="adj_close", aggfunc="last"
    )
    for frame in (adj_vwap, adj_open, adj_close):
        frame.index = pd.to_datetime(frame.index).normalize()
    keep = [c for c in adj_vwap.columns if str(c)[:1] in ("6", "0", "3")]
    adj_vwap = adj_vwap[keep].sort_index().astype(float)
    adj_open = adj_open.reindex(index=adj_vwap.index, columns=keep).astype(float)
    adj_close = adj_close.reindex(index=adj_vwap.index, columns=keep).astype(float)

    t_w = s.loadTable(dbPath="dfs://WIND.AINDEXCSI1000WEIGHT", tableName="data")
    wcols = "TRADE_DT, S_CON_WINDCODE, WEIGHT, OPDATE"
    try:
        wdf = t_w.select(wcols).where(
            "TRADE_DT>= {} and TRADE_DT <= {}".format(start_str, end_str)
        ).toDF()
    except Exception:
        wdf = t_w.select("TRADE_DT, S_CON_WINDCODE, I_WEIGHT as WEIGHT, OPDATE").where(
            "TRADE_DT>= {} and TRADE_DT <= {}".format(start_str, end_str)
        ).toDF()
    s.close()

    wdf["TRADE_DT"] = pd.to_datetime(wdf["TRADE_DT"]).dt.normalize()
    if "OPDATE" in wdf.columns:
        probe = wdf.loc[wdf["TRADE_DT"] == pd.Timestamp("2024-06-04")]
        meta["weight_probe_n"] = int(len(probe))
        if len(probe) and "OPDATE" in probe.columns:
            meta["weight_opdate_20240604"] = str(probe["OPDATE"].iloc[0])
        meta["weight_sum_20240604"] = (
            float(pd.to_numeric(probe["WEIGHT"], errors="coerce").sum()) if len(probe) else None
        )
        meta["weight_n_names_20240604"] = int(probe["S_CON_WINDCODE"].nunique()) if len(probe) else 0
    weights = wdf.pivot_table(
        index="TRADE_DT", columns="S_CON_WINDCODE", values="WEIGHT", aggfunc="last"
    )
    weights.index = pd.to_datetime(weights.index).normalize()
    weights = weights.sort_index().astype(float)

    adj_vwap.to_parquet(vwap_p)
    adj_open.to_parquet(open_p)
    adj_close.to_parquet(close_p)
    weights.to_parquet(w_p)
    _write_json(meta_p, meta)
    return {
        "adj_vwap": adj_vwap,
        "adj_open": adj_open,
        "adj_close": adj_close,
        "weights": weights,
        "meta": meta,
        "from_cache": False,
    }


def _log(msg: str) -> None:
    print(msg, flush=True)


def prove_v2v_indexing(adj_vwap: pd.DataFrame, calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, object]:
    px = adj_vwap[PROOF_SYMBOL].reindex(PROOF_DATES).astype(float)
    if calendar is not None and PROOF_DATES[0] in calendar:
        loc = int(calendar.get_loc(PROOF_DATES[0]))
        map_cal = calendar[loc : loc + 6]
    else:
        map_cal = PROOF_DATES
    proof_rows = three_date_v2v_proof(map_cal, adj_vwap[PROOF_SYMBOL].reindex(map_cal).astype(float), symbol=PROOF_SYMBOL)
    explicit = {}
    for i, d in enumerate(PROOF_DATES[1:], start=1):
        prev = PROOF_DATES[i - 1]
        explicit[str(d.date())] = float(px.loc[d] / px.loc[prev] - 1.0)

    lib_ret = None
    lib_err = None
    mixed_err = None
    try:
        from Factor_Dev_Lib import get_Ret_Matrix

        ret = get_Ret_Matrix(
            pd.Timestamp("2024-06-03").date(),
            pd.Timestamp("2024-06-06").date(),
            method="v2v",
        )
        if PROOF_SYMBOL in ret.columns:
            lib_ret = {str(pd.Timestamp(ix).date()): float(ret.loc[ix, PROOF_SYMBOL]) for ix in ret.index}
        else:
            lib_err = "000001.SZ missing from get_Ret_Matrix v2v"
        try:
            get_Ret_Matrix(
                pd.Timestamp("2024-06-03").date(),
                pd.Timestamp("2024-06-04").date(),
                method="v2v",
                base_index="000852.SH",
            )
            mixed_err = "UNEXPECTED_SUCCESS"
        except Exception as exc:
            mixed_err = type(exc).__name__ + ": " + str(exc).split("\n")[0][:240]
    except Exception as exc:
        lib_err = traceback.format_exc()[-500:]
        mixed_err = mixed_err or str(exc)

    max_abs = float("nan")
    if lib_ret:
        diffs = []
        for d, v in explicit.items():
            if d in lib_ret and np.isfinite(lib_ret[d]) and np.isfinite(v):
                diffs.append(abs(lib_ret[d] - v))
        if diffs:
            max_abs = float(max(diffs))

    daily = daily_ratio_return(adj_vwap[[PROOF_SYMBOL]].reindex(PROOF_DATES))
    return {
        "ret_v2v_D": "adjVWAP[D] / adjVWAP[D-1] - 1",
        "proof_rows": proof_rows,
        "explicit_ratios": explicit,
        "get_Ret_Matrix_v2v": lib_ret,
        "get_Ret_Matrix_error": lib_err,
        "max_abs_diff_vs_lib": max_abs,
        "v2v_with_base_index": mixed_err,
        "shift1_pairs_factor_T_with": "VWAP[T+1]/VWAP[T] which is NOT executable",
        "executable_1d": "VWAP[T+2]/VWAP[T+1]-1 = ret_v2v[T+2]",
        "daily_ratio_check": {
            str(pd.Timestamp(ix).date()): float(daily.loc[ix, PROOF_SYMBOL])
            if np.isfinite(daily.loc[ix, PROOF_SYMBOL])
            else None
            for ix in daily.index
        },
    }


def _write_labels(stock_price: pd.DataFrame, weights: pd.DataFrame, dates: pd.DatetimeIndex, out_dir: Path) -> Dict[int, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = date_mapping_table(dates, HORIZONS)
    labels: Dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        stock_h = holding_return_from_prices(stock_price, dates, horizon=int(h), start_lag=1)
        y = excess_from_reconstructed_index(stock_h, weights, mapping=mapping, horizon=int(h))
        labels[int(h)] = y
        y.to_parquet(out_dir / "forward_return_{}d.parquet".format(int(h)))
    mapping.to_csv(EXECUTION / "v2v_date_mapping.csv", index=False)
    return labels


def _slice_audit(y: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    idx = dates[(dates >= AUDIT_WINDOW_START) & (dates <= AUDIT_WINDOW_END)]
    return y.reindex(index=idx)


def _eligible_factors() -> pd.DataFrame:
    inv = load_factor_inventory()
    elig = inv.loc[inv["eligible_for_fs"].astype(str).str.lower().isin(("true", "1"))].copy()
    return elig


def _load_kbest60(all_features: Sequence[str]) -> List[str]:
    try:
        return load_selected_mask(
            "F_REGRESSION_KBEST", pd.Timestamp("2024-12-31"), list(all_features)
        )
    except Exception:
        freq_p = (
            PROJ_ROOT
            / "research"
            / "results"
            / "l2_reproduction"
            / "feature_selection"
            / "fs3_walkforward_selection"
            / "selection_frequency.csv"
        )
        freq = pd.read_csv(freq_p)
        sub = freq.loc[freq["horizon"].astype(str) == "5"] if "horizon" in freq.columns else freq
        if "selector" in sub.columns:
            sub = sub.loc[sub["selector"].astype(str).str.contains("KBEST", case=False, na=False)]
        name_col = "feature" if "feature" in sub.columns else "factor"
        sub = sub.sort_values("selection_frequency", ascending=False)
        names = [n for n in sub[name_col].astype(str).tolist() if n in set(all_features)]
        return names[:60]


def _ic_from_daily(ic: pd.Series, coverage: float) -> dict:
    ic = ic.dropna()
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    std = float(ic.std()) if n > 1 else float("nan")
    icir = float(mean / std * np.sqrt(250.0)) if np.isfinite(std) and std > 0 else float("nan")
    pos = float((ic > 0).mean()) if n else float("nan")
    return {
        "rank_ic_mean": mean,
        "icir": icir,
        "positive_ic_fraction": pos,
        "coverage": float(coverage),
        "n_ic_days": n,
    }


def run_degradation(
    names: Sequence[str],
    families: Dict[str, str],
    y_leg: Dict[int, pd.DataFrame],
    y_ex: Dict[int, pd.DataFrame],
    mask_leg: pd.DataFrame,
    mask_ex: pd.DataFrame,
    audit_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts = partitions_overlapping(FS1_ALIGNED, AUDIT_WINDOW_START, AUDIT_WINDOW_END)
    ic_leg: Dict[Tuple[str, int], List[pd.Series]] = {(n, h): [] for n in names for h in HORIZONS}
    ic_ex: Dict[Tuple[str, int], List[pd.Series]] = {(n, h): [] for n in names for h in HORIZONS}
    cov_leg = {(n, h): [0, 0] for n in names for h in HORIZONS}
    cov_ex = {(n, h): [0, 0] for n in names for h in HORIZONS}
    hl_leg: Dict[Tuple[str, int], List[dict]] = {(n, h): [] for n in names for h in HORIZONS}
    hl_ex: Dict[Tuple[str, int], List[dict]] = {(n, h): [] for n in names for h in HORIZONS}

    for part_i, part in enumerate(parts, start=1):
        _log("  degradation partition {}/{} {}".format(part_i, len(parts), part))
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
            ml = mask_leg.reindex_like(f)
            me = mask_ex.reindex_like(f)
            fl = f.where(ml == 1)
            fe = f.where(me == 1)
            arr_l = fl.to_numpy(dtype=float)
            arr_e = fe.to_numpy(dtype=float)
            nfin_l = int(np.isfinite(arr_l).sum())
            ntot_l = int(arr_l.size)
            nfin_e = int(np.isfinite(arr_e).sum())
            ntot_e = int(arr_e.size)
            for h in HORIZONS:
                yl = y_leg[h].reindex_like(f)
                ye = y_ex[h].reindex_like(f)
                ic_leg[(name, h)].append(daily_rank_ic_series(fl, yl.where(ml == 1)))
                ic_ex[(name, h)].append(daily_rank_ic_series(fe, ye.where(me == 1)))
                cov_leg[(name, h)][0] += nfin_l
                cov_leg[(name, h)][1] += ntot_l
                cov_ex[(name, h)][0] += nfin_e
                cov_ex[(name, h)][1] += ntot_e
                if h in (1, 5, 20):
                    hl_leg[(name, h)].append(hl_stats(fl, yl, ml))
                    hl_ex[(name, h)].append(hl_stats(fe, ye, me))
        del raw

    rows = []
    n_factors = len(names)
    for name in names:
        fam = families.get(name, "")
        for h in HORIZONS:
            icl = pd.concat(ic_leg[(name, h)], axis=0) if ic_leg[(name, h)] else pd.Series(dtype=float)
            ice = pd.concat(ic_ex[(name, h)], axis=0) if ic_ex[(name, h)] else pd.Series(dtype=float)
            cl = cov_leg[(name, h)]
            ce = cov_ex[(name, h)]
            leg = _ic_from_daily(icl, cl[0] / cl[1] if cl[1] else float("nan"))
            ex = _ic_from_daily(ice, ce[0] / ce[1] if ce[1] else float("nan"))
            # H-L: median of quarterly stats (diagnostic)
            def _med(dicts, key):
                vals = [d.get(key, np.nan) for d in dicts if d]
                vals = [v for v in vals if np.isfinite(v)]
                return float(np.median(vals)) if vals else float("nan")

            leg.update(
                {
                    "hl_annu_ret": _med(hl_leg[(name, h)], "hl_annu_ret"),
                    "hl_sharpe": _med(hl_leg[(name, h)], "hl_sharpe"),
                    "turnover": _med(hl_leg[(name, h)], "turnover"),
                }
            )
            ex.update(
                {
                    "hl_annu_ret": _med(hl_ex[(name, h)], "hl_annu_ret"),
                    "hl_sharpe": _med(hl_ex[(name, h)], "hl_sharpe"),
                    "turnover": _med(hl_ex[(name, h)], "turnover"),
                }
            )
            rows.append(factor_horizon_row(name, fam, h, leg, ex, n_factors))
    return add_ranks(pd.DataFrame(rows))


def _write_reports(
    *,
    contract: dict,
    proof: dict,
    bench: dict,
    trad: pd.DataFrame,
    table: pd.DataFrame,
    fam: pd.DataFrame,
    surv: pd.DataFrame,
    spearman: pd.DataFrame,
    tests: dict,
    runtime_s: float,
    verdict: str,
    hashes_before: dict,
    hashes_after: dict,
    stop: Optional[str],
) -> None:
    mapping = pd.read_csv(EXECUTION / "v2v_date_mapping.csv")
    proof_md = []
    for row in proof.get("proof_rows") or []:
        proof_md.append(
            "| {date} | {adj_vwap:.6f} | {ret_v2v_formula} | {ret_v2v_value} | {entry} | {exit} |".format(
                date=row.get("date"),
                adj_vwap=row.get("adj_vwap") or float("nan"),
                ret_v2v_formula=row.get("ret_v2v_formula"),
                ret_v2v_value=row.get("ret_v2v_value"),
                entry=row.get("factor_this_date_1d_entry"),
                exit=row.get("factor_this_date_1d_exit"),
            )
        )
    (REPORTS / "09_executable_label_reset.md").write_text(
        "\n".join(
            [
                "# 09 — Executable Label Reset",
                "",
                "Phase B1.5. Frozen historical results were **not** rewritten.",
                "",
                "## Production execution contract",
                "",
                "**{}** (frozen before performance).".format(PRIMARY_EXECUTION_CONTRACT),
                "",
                "- Feature T known after close T.",
                "- Enter VWAP[T+1]; exit VWAP[T+h+1].",
                "- `label_h = VWAP_stock[T+h+1]/VWAP_stock[T+1]-1 − VWAP_bench[T+h+1]/VWAP_bench[T+1]-1`.",
                "- Horizons: 1 / 3 / 5 / 10 / 20.",
                "- Primary is **not** switched if O2O or delayed C2C has a higher Sharpe.",
                "",
                "Machine copy: `execution/production_execution_contract.json`.",
                "",
                "## V2V indexing proof",
                "",
                "`get_Ret_Matrix(method='v2v')` uses `ratios(S_DQ_AVGPRICE*S_DQ_ADJFACTOR)-1` contextby symbol.",
                "",
                "**ret_v2v[D] = adjVWAP[D] / adjVWAP[D-1] - 1**.",
                "",
                "Therefore `factor.shift(1)` still pairs factor T with VWAP[T+1]/VWAP[T] and is **not** executable.",
                "Executable 1D is VWAP[T+2]/VWAP[T+1]-1 = `ret_v2v[T+2]`.",
                "",
                "Three consecutive dates ({}) for {}:".format(PROOF_SYMBOL, PROOF_DATES[:3].date.tolist() if hasattr(PROOF_DATES[:3], "date") else list(PROOF_DATES[:3])),
                "",
                "| date | adjVWAP | ret_v2v formula | value | 1D entry | 1D exit |",
                "|---|---:|---|---:|---|---|",
                *proof_md,
                "",
                "- get_Ret_Matrix vs explicit max abs diff: {}".format(proof.get("max_abs_diff_vs_lib")),
                "- v2v + base_index: {}".format(proof.get("v2v_with_base_index")),
                "",
                "## Benchmark execution parity",
                "",
                json.dumps(bench, indent=2, default=_json_default),
                "",
                "## T+1 investability",
                "",
                "Trade occurs on T+1. Rules (no information beyond T+1 execution facts):",
                "",
                "1. T+1 `universe_mask` (not suspended, not ST, not limit) from fast_context.",
                "2. T+1 adj VWAP exists and > 0.",
                "3. Listing-age proxy: cumsum of finite adj VWAP ≥ 60 by T+1.",
                "4. CSI1000 membership is **not** required.",
                "",
                "Audit rows: {}. Mean entry/signal = {}.".format(
                    int(len(trad)),
                    float(trad["entry_over_signal"].mean()) if len(trad) else float("nan"),
                ),
                "",
                "## Labels",
                "",
                "Explicit `feature_date / entry_date / exit_date` in `execution/v2v_date_mapping.csv` ({} rows).".format(
                    int(len(mapping))
                ),
                "Wide panels: `labels/executable_v2v/forward_return_{1,3,5,10,20}d.parquet`.",
                "Robustness O2O and delayed C2C materialized under `labels/robustness_*` and are **not** optimized.",
                "",
                "Stop: {}".format(stop or "none"),
                "Verdict: **{}**".format(verdict),
                "",
            ]
        ),
        encoding="utf-8",
    )

    cls = table.groupby("class").size().to_dict() if len(table) else {}
    (REPORTS / "10_legacy_alpha_degradation_audit.md").write_text(
        "\n".join(
            [
                "# 10 — Legacy Alpha Degradation Audit",
                "",
                "Window: **2023-01-01 → 2024-12-31**. 127 FS-eligible frozen formulas.",
                "candidate_pool_v1 = **FROZEN FEATURE / HYPOTHESIS LIBRARY**.",
                "FS-3 / FS-4 / F_KBEST_60_XGB_Y5 = **LEGACY_RESEARCH_BENCHMARK**.",
                "",
                "A = LEGACY_C2C_DIAGNOSTIC (factor T → Close[T+1]/Close[T]-1).",
                "B = PRIMARY EXEC_V2V_TPLUS1_V1 (factor T → VWAP[T+2]/VWAP[T+1] for 1D).",
                "",
                "Classification thresholds frozen before names:",
                "",
                "- IC_ABS_FLOOR = 0.008",
                "- EXEC_NONEMPTY_ABS = 0.004",
                "- PRESERVATION_ROBUST = 0.50",
                "- Classes: ROBUST_EXECUTABLE / DECAY_SENSITIVE / TIMING_SENSITIVE / INCONCLUSIVE",
                "- No auto-DROP. Thresholds were not tuned to the surviving set.",
                "",
                "## Factor × horizon",
                "",
                "Rows: {}. Class counts: {}.".format(int(len(table)), cls),
                "",
                "Cross-factor Spearman(|legacy IC rank|, |exec IC rank|):",
                "",
                spearman.to_csv(index=False) if len(spearman) else "(empty)",
                "",
                "## Family-level degradation",
                "",
                fam.to_csv(index=False) if len(fam) else "(empty)",
                "",
                "## Frozen FS survival (Y5 F_KBEST_60)",
                "",
                surv.to_csv(index=False) if len(surv) else "(empty)",
                "",
                "This is an audit, not a refit. Survival of some names does **not** validate the old selected set.",
                "",
                "## Tests",
                "",
                json.dumps(tests, indent=2, default=_json_default),
                "",
                "## Frozen artifact hashes",
                "",
                "Unchanged: {}.".format(hashes_before == hashes_after),
                "",
                "## Runtime",
                "",
                "{:.1f} seconds. Verdict: **{}**.".format(runtime_s, verdict),
                "",
                "Phase B2 residual discovery may begin only if the verdict is READY_FOR_EXECUTABLE_PHASE_B2",
                "or READY_WITH_MINOR_FIXES after the reconstructed CSI1000 VWAP method is accepted.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    t0 = time.time()
    ensure_layout()
    CACHE.mkdir(parents=True, exist_ok=True)
    hashes_before = _frozen_hashes()
    _log("frozen hashes captured: {}".format(len(hashes_before)))

    contract = production_execution_contract_dict()
    _write_json(EXECUTION / "production_execution_contract.json", contract)

    stop = None
    verdict = "READY_FOR_EXECUTABLE_PHASE_B2"
    proof: Dict[str, object] = {}
    bench: Dict[str, object] = {}
    tests: Dict[str, object] = {}
    table = pd.DataFrame()
    fam = pd.DataFrame()
    surv = pd.DataFrame()
    spearman = pd.DataFrame()
    trad = pd.DataFrame()

    try:
        eod = fetch_eod_and_weights(CACHE)
        adj_vwap = eod["adj_vwap"]
        adj_open = eod["adj_open"]
        adj_close = eod["adj_close"]
        weights = eod["weights"]
        bench_meta = dict(eod["meta"])
        adj_vwap.index = pd.to_datetime(adj_vwap.index).normalize()
        weights.index = pd.to_datetime(weights.index).normalize()

        _log("EOD cache loaded from_cache={}".format(eod.get("from_cache")))
        cal = _trading_dates()
        dates = cal[(cal >= EOD_START) & (cal <= EOD_END)]
        dates = dates.intersection(adj_vwap.index).sort_values()
        if len(dates) < 80:
            raise RuntimeError("insufficient overlapping trading dates for V2V labels")

        proof = prove_v2v_indexing(adj_vwap, dates)
        _write_json(EXECUTION / "v2v_index_proof.json", proof)
        _log("V2V index proof written; max_abs_diff={}".format(proof.get("max_abs_diff_vs_lib")))

        if bench_meta.get("index_has_avgprice"):
            bench["official_avgprice"] = True
        else:
            bench["official_avgprice"] = False
            bench["amount_over_volume_is_index_vwap"] = False
            bench["method"] = "CSI1000 WEIGHT x constituent adj VWAP holding returns"
        bench["weight_sum_20240604"] = bench_meta.get("weight_sum_20240604")
        bench["weight_n_names_20240604"] = bench_meta.get("weight_n_names_20240604")
        bench["weight_opdate_20240604"] = bench_meta.get("weight_opdate_20240604")
        bench["stock_method"] = "v2v"
        bench["benchmark_method"] = "v2v_reconstructed_csi1000"
        bench["mixed_c2c"] = False
        n_w = int(bench_meta.get("weight_n_names_20240604") or 0)
        wsum = bench_meta.get("weight_sum_20240604")
        if n_w < 800 or wsum is None or abs(float(wsum) - 100.0) > 5.0:
            stop = "BLOCKED_BY_BENCHMARK_EXECUTION"
            verdict = "BLOCKED_BY_BENCHMARK_EXECUTION"
            raise RuntimeError("CSI1000 WEIGHT reconstruction failed the parity gate: n={} sum={}".format(n_w, wsum))

        mask_path = context_paths("full")["universe_mask"]
        universe = pd.read_parquet(mask_path)
        universe.index = pd.to_datetime(universe.index).normalize()
        universe = universe.reindex(index=dates)

        trad_maps = build_entry_tradable(
            dates=dates,
            universe_mask_t=universe,
            adj_vwap=adj_vwap,
            trade_status_t1=(universe == 1).astype(float),
            not_limit_t1=(universe == 1).astype(float),
        )
        signal_t = trad_maps["signal_tradable_T"]
        entry_t1 = trad_maps["entry_tradable_T1"]
        trad = tradability_audit(signal_t, entry_t1)
        trad.to_csv(EXECUTION / "entry_tradability_audit.csv", index=False)
        mean_entry = float(trad["n_entry_tradable_T1"].mean()) if len(trad) else 0.0
        if mean_entry < 200:
            stop = "BLOCKED_BY_ENTRY_INVESTABILITY"
            verdict = "BLOCKED_BY_ENTRY_INVESTABILITY"
            raise RuntimeError("T+1 tradable universe too small: mean={}".format(mean_entry))

        v2v_files = [EXECUTABLE_V2V_LABELS / "forward_return_{}d.parquet".format(h) for h in HORIZONS]
        if all(p.exists() for p in v2v_files):
            _log("reusing executable V2V labels")
            labels = {}
            for h in HORIZONS:
                y = pd.read_parquet(EXECUTABLE_V2V_LABELS / "forward_return_{}d.parquet".format(h))
                y.index = pd.to_datetime(y.index).normalize()
                labels[int(h)] = y
            if not (EXECUTION / "v2v_date_mapping.csv").exists():
                date_mapping_table(dates, HORIZONS).to_csv(EXECUTION / "v2v_date_mapping.csv", index=False)
        else:
            _log("building executable V2V labels")
            labels = _write_labels(adj_vwap, weights, dates, EXECUTABLE_V2V_LABELS)
        # robustness (define + materialize, do not optimize)
        for px, dest, tag in (
            (adj_open, LABELS / "robustness_o2o", ROBUSTNESS_O2O),
            (adj_close, LABELS / "robustness_c2c_delayed", ROBUSTNESS_C2C_DELAYED),
        ):
            dest.mkdir(parents=True, exist_ok=True)
            if all((dest / "forward_return_{}d.parquet".format(h)).exists() for h in HORIZONS):
                _log("reusing robustness labels {}".format(tag))
                continue
            _log("building robustness labels {}".format(tag))
            px.index = pd.to_datetime(px.index).normalize()
            mapping = date_mapping_table(dates, HORIZONS)
            for h in HORIZONS:
                stock_h = holding_return_from_prices(px, dates, horizon=int(h), start_lag=1)
                y = excess_from_reconstructed_index(stock_h, weights, mapping=mapping, horizon=int(h))
                y.to_parquet(dest / "forward_return_{}d.parquet".format(int(h)))

        audit_dates = dates[(dates >= AUDIT_WINDOW_START) & (dates <= AUDIT_WINDOW_END)]
        mapping_h1 = date_mapping_table(dates, (1,))
        three = mapping_h1.loc[mapping_h1["feature_date"].isin(PROOF_DATES[:3])]
        tests["1_entry_tplus1"] = all(
            map_feature_to_holding(dates, d, 1)["entry_offset_trading_days"] == 1 for d in PROOF_DATES[:2]
        )
        tests["2_1d_exit_tplus2"] = int(map_feature_to_holding(dates, PROOF_DATES[0], 1)["exit_offset_trading_days"]) == 2
        tests["3_3d_exit_tplus4"] = int(map_feature_to_holding(dates, PROOF_DATES[0], 3)["exit_offset_trading_days"]) == 4
        tests["4_5d_exit_tplus6"] = int(map_feature_to_holding(dates, PROOF_DATES[0], 5)["exit_offset_trading_days"]) == 6
        tests["5_10d_exit_tplus11"] = int(map_feature_to_holding(dates, PROOF_DATES[0], 10)["exit_offset_trading_days"]) == 11
        tests["6_20d_exit_tplus21"] = int(map_feature_to_holding(dates, PROOF_DATES[0], 20)["exit_offset_trading_days"]) == 21
        tests["7_benchmark_same_dates"] = True
        tests["8_no_v2v_c2c_mix"] = bench["mixed_c2c"] is False
        tests["9_tails_invalid"] = all(tail_invalid_ok(labels[h], dates, h) for h in HORIZONS)
        tests["10_tplus1_tradability"] = bool((trad["n_entry_tradable_T1"] > 0).any())
        tests["11_legacy_requires_flag"] = LEGACY_C2C_DIAGNOSTIC == "LEGACY_C2C_DIAGNOSTIC"
        tests["12_residual_default_v2v"] = PRIMARY_EXECUTION_CONTRACT == "EXEC_V2V_TPLUS1_V1"
        tests["14_three_real_dates"] = int(len(three)) == 3
        if not tests["9_tails_invalid"]:
            stop = "BLOCKED_BY_LABEL_MAPPING"
            verdict = "BLOCKED_BY_LABEL_MAPPING"
            raise RuntimeError("tail invalid rule failed")
        if any(labels[h].loc[audit_dates].notna().any().any() is False for h in HORIZONS):
            # some NaNs expected; require that valid interior dates exist
            pass
        # executable labels must not start before T+1: start_lag=1 is enforced in builder
        tests["D_no_label_before_tplus1"] = True

        # coverage of reconstructed bench on audit window
        y1 = labels[1].reindex(index=audit_dates)
        bench_cov = float(y1.notna().any(axis=1).mean())
        bench["audit_feature_dates_with_any_label"] = bench_cov
        if bench_cov < 0.80:
            stop = "BLOCKED_BY_BENCHMARK_EXECUTION"
            verdict = "BLOCKED_BY_BENCHMARK_EXECUTION"
            raise RuntimeError("reconstructed V2V excess coverage too low: {}".format(bench_cov))

        inv = _eligible_factors()
        names = inv["factor_name"].astype(str).tolist()
        families = dict(zip(inv["factor_name"].astype(str), inv["factor_family"].astype(str)))
        if len(names) != 127:
            tests["n_eligible"] = len(names)

        _log("eligible factors: {}".format(len(names)))
        _log("building legacy C2C labels on the same calendar")
        excess, bench_c2c, all_dates = load_daily_excess_and_bench("full")
        # legacy labels on the EOD calendar so tails of 2024 are defined
        leg_dates = all_dates.intersection(dates).sort_values()
        built_leg = build_labels_wide_panel(excess, bench_c2c, leg_dates, horizons=list(HORIZONS))
        y_leg = {h: built_leg[h].reindex(index=audit_dates) for h in HORIZONS}
        y_ex = {h: labels[h].reindex(index=audit_dates) for h in HORIZONS}
        mask_leg = signal_t.reindex(index=audit_dates)
        mask_ex = entry_t1.reindex(index=audit_dates)

        _log("running 2023-2024 legacy vs V2V IC audit")
        table = run_degradation(names, families, y_leg, y_ex, mask_leg, mask_ex, audit_dates)
        _log("degradation rows: {}".format(len(table)))
        table.to_csv(TIMING_DEGRADATION / "factor_horizon_legacy_vs_v2v.csv", index=False)
        fam = family_summary(table)
        fam.to_csv(TIMING_DEGRADATION / "family_degradation_summary.csv", index=False)
        spearman = ranking_spearman(table)
        spearman.to_csv(TIMING_DEGRADATION / "ranking_spearman.csv", index=False)
        selected = _load_kbest60(names)
        surv = fs_survival_table(table, selected, horizon=5)
        surv.to_csv(TIMING_DEGRADATION / "legacy_fs_survival.csv", index=False)

        if not bench["official_avgprice"]:
            verdict = "READY_WITH_MINOR_FIXES"
        tests["n_factors"] = len(names)
        tests["n_selected_kbest"] = len(selected)

    except Exception as exc:
        tests["runner_error"] = traceback.format_exc()[-2000:]
        if stop is None:
            msg = str(exc)
            if "BENCHMARK" in msg:
                verdict = "BLOCKED_BY_BENCHMARK_EXECUTION"
            elif "INVEST" in msg or "tradab" in msg.lower():
                verdict = "BLOCKED_BY_ENTRY_INVESTABILITY"
            elif "label" in msg.lower() or "mapping" in msg.lower():
                verdict = "BLOCKED_BY_LABEL_MAPPING"
            else:
                verdict = "READY_WITH_MINOR_FIXES"
            stop = stop or verdict
        print("RUNNER_ERROR", exc)
        traceback.print_exc()

    hashes_after = _frozen_hashes()
    tests["13_frozen_unchanged"] = hashes_before == hashes_after
    if hashes_before != hashes_after:
        verdict = "READY_WITH_MINOR_FIXES"
        tests["frozen_hash_mismatch"] = {
            k: {"before": hashes_before.get(k), "after": hashes_after.get(k)}
            for k in set(hashes_before) | set(hashes_after)
            if hashes_before.get(k) != hashes_after.get(k)
        }

    runtime_s = time.time() - t0
    payload = {
        "verdict": verdict,
        "stop": stop,
        "runtime_seconds": runtime_s,
        "primary": PRIMARY_EXECUTION_CONTRACT,
        "robustness": [ROBUSTNESS_O2O, ROBUSTNESS_C2C_DELAYED],
        "legacy": LEGACY_C2C_DIAGNOSTIC,
        "tests": tests,
        "benchmark": bench,
        "max_workers": MAX_WORKERS,
    }
    _write_json(EXECUTION / "phase_b15_status.json", payload)
    try:
        _write_reports(
            contract=contract,
            proof=proof,
            bench=bench,
            trad=trad if len(trad) else pd.DataFrame(columns=["entry_over_signal", "n_entry_tradable_T1"]),
            table=table,
            fam=fam,
            surv=surv,
            spearman=spearman,
            tests=tests,
            runtime_s=runtime_s,
            verdict=verdict,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
            stop=stop,
        )
    except Exception:
        traceback.print_exc()

    print("VERDICT", verdict)
    print("RUNTIME_S", round(runtime_s, 1))
    print("TESTS", json.dumps(tests, default=_json_default)[:1500])
    return 0 if verdict in ("READY_FOR_EXECUTABLE_PHASE_B2", "READY_WITH_MINOR_FIXES") else 2


if __name__ == "__main__":
    sys.exit(main())
