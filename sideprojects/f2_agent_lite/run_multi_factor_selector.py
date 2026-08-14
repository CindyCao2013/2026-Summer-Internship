#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全市场透明多因子选股（线性 / 可解释，无 Transformer）.

流程：成分 PIT → 批量因子 → MAD 去极值 → 截面 Z → 市值+行业回归中性化
      → 等权 / 滚动 ICIR 加权 → RotationBacktester

可选：接入 FS-1 L2 面板（已 Huatai 市值+行业中性化），先按滚动 ICIR 筛选再合并。

用法::

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_multi_factor_selector \\
        --index 000300.SH --weights icir

    /opt/conda/anaconda3/bin/python -m sideprojects.f2_agent_lite.run_multi_factor_selector \\
        --index 000300.SH --weights icir --with-fs1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sideprojects.f2_agent_lite.backtest.rotation_backtester import RotationBacktester  # noqa: E402
from sideprojects.f2_agent_lite.config import Config  # noqa: E402
from sideprojects.f2_agent_lite.data import db_connector as db  # noqa: E402

# ---------------------------------------------------------------------------
# Factor registry (extend here)
# ---------------------------------------------------------------------------
CORE_FACTOR_NAMES = ["mom_20d", "mom_60d", "bp_ttm", "ep_ttm", "roe_ttm", "north_ratio"]
FACTOR_NAMES = list(CORE_FACTOR_NAMES)  # mutated at runtime when --with-fs1

FS1_DEFAULT_ROOT = (
    _REPO_ROOT
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs1_feature_panel"
)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if pd.isna(obj):
        return None
    return obj


def mad_winsorize_cs(wide: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """Cross-sectional MAD winsorization (per row / day)."""
    med = wide.median(axis=1)
    mad = (wide.sub(med, axis=0)).abs().median(axis=1)
    # MAD=0 → no clip
    scale = (1.4826 * mad).replace(0.0, np.nan)
    lo = med.sub(n_mad * scale, axis=0)
    hi = med.add(n_mad * scale, axis=0)
    out = wide.clip(lower=lo, upper=hi, axis=0)
    # rows with zero MAD stay unchanged
    const = mad.isna() | (mad <= 0)
    if const.any():
        out.loc[const] = wide.loc[const]
    return out


def cs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0)


def neutralize_size_industry(
    factor_wide: pd.DataFrame,
    log_mktcap: pd.DataFrame,
    ind_map: Dict[str, str],
) -> pd.DataFrame:
    """Daily OLS residual of factor ~ log_mktcap + industry dummies."""
    if factor_wide.empty:
        return factor_wide
    out = pd.DataFrame(index=factor_wide.index, columns=factor_wide.columns, dtype=float)
    symbols = list(factor_wide.columns)
    ind = pd.Series({s: ind_map.get(str(s), "OTHER") for s in symbols})

    for d in factor_wide.index:
        y = factor_wide.loc[d]
        x_cap = log_mktcap.loc[d] if d in log_mktcap.index else pd.Series(index=symbols, dtype=float)
        df = pd.DataFrame({"y": y, "cap": x_cap, "ind": ind}).dropna()
        if len(df) < 10:
            out.loc[d] = y
            continue
        # industry dummies (drop first)
        dummies = pd.get_dummies(df["ind"], prefix="ind", drop_first=True, dtype=float)
        X = pd.concat([df[["cap"]].astype(float), dummies], axis=1)
        X = sm_add_const(X)
        try:
            beta = np.linalg.lstsq(X.to_numpy(dtype=float), df["y"].to_numpy(dtype=float), rcond=None)[0]
            resid = df["y"].to_numpy(dtype=float) - X.to_numpy(dtype=float) @ beta
            out.loc[d, df.index] = resid
        except Exception:
            out.loc[d] = y
    return out


def sm_add_const(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.insert(0, "const", 1.0)
    return X


def rank_ic(factor_row: pd.Series, ret_row: pd.Series) -> float:
    a = factor_row.dropna()
    b = ret_row.reindex(a.index).dropna()
    common = a.index.intersection(b.index)
    if len(common) < 8:
        return np.nan
    return float(a.loc[common].rank().corr(b.loc[common].rank()))


def _daily_spearman_ic(factor_wide: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    """Per-day Rank-IC (approx Spearman via argsort ranks) for one factor panel."""
    idx = factor_wide.index.intersection(fwd_ret.index)
    cols = factor_wide.columns.intersection(fwd_ret.columns)
    if len(idx) == 0 or len(cols) == 0:
        return pd.Series(dtype=float)
    F = factor_wide.loc[idx, cols].to_numpy(dtype=float)
    R = fwd_ret.loc[idx, cols].to_numpy(dtype=float)
    out = np.full(len(idx), np.nan, dtype=float)
    for i in range(len(idx)):
        fi = F[i]
        ri = R[i]
        m = np.isfinite(fi) & np.isfinite(ri)
        if int(m.sum()) < 8:
            continue
        a = fi[m]
        b = ri[m]
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        out[i] = float((ra * rb).sum() / denom) if denom > 0 else np.nan
    return pd.Series(out, index=idx)


def calc_icir_weights(
    factor_panels: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling ICIR weights aligned to signal dates (causal: use IC up to t-1).

    Returns
    -------
    weights : DataFrame date x factor
    icir_hist : DataFrame date x factor (ICIR used for that day)
    """
    names = list(factor_panels.keys())
    ic_cols = {name: _daily_spearman_ic(factor_panels[name], fwd_ret) for name in names}
    ic_mat = pd.DataFrame(ic_cols).sort_index()
    dates = list(ic_mat.index)

    # ICIR_t uses IC history ending at t-1 (no look-ahead)
    icir = pd.DataFrame(index=dates, columns=names, dtype=float)
    for name in names:
        s = ic_mat[name].astype(float)
        mu = s.shift(1).rolling(window, min_periods=min_periods).mean()
        sd = s.shift(1).rolling(window, min_periods=min_periods).std(ddof=0)
        icir[name] = mu / sd.replace(0.0, np.nan)

    # Softmax-style positive weights: clip negative ICIR to 0, renormalize
    w = icir.clip(lower=0.0)
    row_sum = w.sum(axis=1).replace(0.0, np.nan)
    equal = 1.0 / max(len(names), 1)
    w = w.div(row_sum, axis=0)
    w = w.fillna(equal)
    w = w.div(w.sum(axis=1), axis=0)
    return w, icir


def build_exec_panel(ohlcv: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in ohlcv.groupby("symbol", sort=False):
        daily = g.set_index("date").sort_index()
        trad = db.compute_tradability_from_ohlcv(g)
        if not trad.empty:
            daily = daily.join(trad.set_index("date")[["tradable"]], how="left")
        else:
            daily["tradable"] = np.nan
        daily["next_open"] = daily["open"].shift(-1)
        daily["next_close"] = daily["close"].shift(-1)
        daily["next_date"] = daily.index.to_series().shift(-1)
        daily["next_tradable"] = daily["tradable"].shift(-1)
        part = daily.reset_index()
        part["symbol"] = sym
        frames.append(
            part[
                [
                    "date",
                    "symbol",
                    "close",
                    "next_open",
                    "next_close",
                    "next_date",
                    "next_tradable",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["next_date"] = pd.to_datetime(out["next_date"])
    return out


def load_raw_factor_panels(
    symbols: List[str],
    start: str,
    end: str,
    preheat_days: int = 120,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return raw factor wides + log_mktcap + fwd_ret + exec_df."""
    preheat = (pd.Timestamp(start) - pd.Timedelta(days=preheat_days)).strftime("%Y-%m-%d")
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    print("[mf] bulk OHLCV ...", flush=True)
    ohlcv = db.get_ohlcv_bulk(symbols, preheat, end_plus)
    print("[mf] bulk valuation ...", flush=True)
    val = db.get_valuation_bulk(symbols, preheat, end_plus)
    print("[mf] bulk ROE ...", flush=True)
    roe = db.get_roe_bulk(symbols, preheat, end)
    print("[mf] bulk north ...", flush=True)
    north = db.get_northbound_bulk(symbols, preheat, end_plus)

    close_w = ohlcv.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    # log returns over 20/60 days
    mom20 = np.log(close_w / close_w.shift(20))
    mom60 = np.log(close_w / close_w.shift(60))
    fwd_ret = close_w.pct_change(1, fill_method=None).shift(-1)  # return from t to t+1

    panels: Dict[str, pd.DataFrame] = {
        "mom_20d": mom20,
        "mom_60d": mom60,
    }
    if not val.empty:
        panels["bp_ttm"] = val.pivot_table(index="date", columns="symbol", values="bp", aggfunc="last")
        panels["ep_ttm"] = val.pivot_table(index="date", columns="symbol", values="ep_ttm", aggfunc="last")
        log_mkt = val.pivot_table(index="date", columns="symbol", values="log_mktcap", aggfunc="last")
    else:
        log_mkt = pd.DataFrame(index=close_w.index, columns=close_w.columns, dtype=float)

    if not roe.empty:
        panels["roe_ttm"] = roe.pivot_table(index="date", columns="symbol", values="roe", aggfunc="last")
    if not north.empty:
        panels["north_ratio"] = north.pivot_table(
            index="date", columns="symbol", values="north_share_ratio", aggfunc="last"
        )

    exec_df = build_exec_panel(ohlcv)
    return panels, log_mkt, fwd_ret, exec_df


def process_factors(
    raw_panels: Dict[str, pd.DataFrame],
    log_mktcap: pd.DataFrame,
    ind_map: Dict[str, str],
    member_mask: pd.DataFrame,
    do_neutral: bool,
) -> Dict[str, pd.DataFrame]:
    """MAD → Z → optional size/industry neutralize; apply membership mask."""
    out = {}
    for name, wide in raw_panels.items():
        print("[mf] process {}".format(name), flush=True)
        w = wide.sort_index()
        # only members enter CS stats
        mem = member_mask.reindex(index=w.index, columns=w.columns)
        w = w.where(mem.notna())
        w = mad_winsorize_cs(w, n_mad=5.0)
        w = cs_zscore(w)
        if do_neutral:
            cap = log_mktcap.reindex(index=w.index, columns=w.columns)
            w = neutralize_size_industry(w, cap, ind_map)
            # re-z after residualization for scale comparability
            w = cs_zscore(w)
        w = w.where(mem.reindex_like(w).notna())
        out[name] = w
    return out


def composite_scores(
    panels: Dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    names = [n for n in (factor_names or FACTOR_NAMES) if n in panels]
    dates = sorted(set().union(*[panels[n].index for n in names]))
    cols = sorted(set().union(*[panels[n].columns for n in names]))
    stack = []
    wstack = []
    for n in names:
        f = panels[n].reindex(index=dates, columns=cols)
        ww = weights[n].reindex(dates).astype(float) if n in weights.columns else pd.Series(1.0 / len(names), index=dates)
        stack.append(f.to_numpy(dtype=float))
        wstack.append(ww.to_numpy(dtype=float))
    arr = np.stack(stack, axis=0)  # F x T x N
    warr = np.stack(wstack, axis=0)  # F x T
    # weighted nanmean
    num = np.nansum(arr * warr[:, :, None], axis=0)
    den = np.nansum(np.where(np.isfinite(arr), warr[:, :, None], 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = num / np.where(den > 0, den, np.nan)
    return pd.DataFrame(score, index=dates, columns=cols)


def _eligible_fs1_factors(panel_root: Path) -> List[str]:
    inv_path = panel_root / "feature_inventory.csv"
    if not inv_path.exists():
        return []
    inv = pd.read_csv(inv_path)
    if "eligible_for_fs" in inv.columns:
        return inv.loc[inv["eligible_for_fs"].astype(bool), "factor"].astype(str).tolist()
    return inv["factor"].astype(str).tolist()


def load_fs1_processed_long(
    panel_root: Path,
    start: str,
    end: str,
    symbols: Optional[List[str]] = None,
    factors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load FS-1 processed_ind_cap_z_v1 partitions; optional symbol/factor filter."""
    import pyarrow.parquet as pq

    root = Path(panel_root) / "processed_ind_cap_z_v1"
    if not root.exists():
        raise FileNotFoundError("FS-1 processed panel missing: {}".format(root))
    factor_cols = factors if factors is not None else _eligible_fs1_factors(Path(panel_root))
    if not factor_cols:
        sample = next(root.rglob("part.parquet"))
        factor_cols = [
            c
            for c in pq.ParquetFile(sample).schema.names
            if c not in ("TradeDate", "Symbol", "year", "quarter")
        ]
    want = ["TradeDate", "Symbol"] + list(factor_cols)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    load_start = start_ts - pd.Timedelta(days=120)
    sym_set = set(str(s) for s in symbols) if symbols is not None else None
    frames = []
    for part in sorted(root.rglob("part.parquet")):
        names = set(pq.ParquetFile(part).schema.names)
        use_cols = [c for c in want if c in names]
        df = pd.read_parquet(part, columns=use_cols)
        miss = [c for c in want if c not in df.columns]
        for c in miss:
            df[c] = np.nan
        df = df[want]
        df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
        df["Symbol"] = df["Symbol"].astype(str)
        df = df[(df["TradeDate"] >= load_start) & (df["TradeDate"] <= end_ts)]
        if sym_set is not None:
            df = df[df["Symbol"].isin(sym_set)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=want)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["TradeDate", "Symbol"], keep="last")
    return out


def fs1_long_to_panels(
    long_df: pd.DataFrame,
    factors: List[str],
    member_mask: pd.DataFrame,
    re_zscore: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Pivot FS-1 long → wide panels; mask to index members; optional CS re-Z."""
    panels: Dict[str, pd.DataFrame] = {}
    if long_df.empty:
        return panels
    for name in factors:
        if name not in long_df.columns:
            continue
        wide = long_df.pivot_table(index="TradeDate", columns="Symbol", values=name, aggfunc="last").sort_index()
        mem = member_mask.reindex(index=wide.index, columns=wide.columns)
        wide = wide.where(mem.notna())
        if re_zscore:
            wide = cs_zscore(wide)
            wide = wide.where(mem.reindex_like(wide).notna())
        panels[name] = wide
    return panels


def screen_fs1_icir(
    panels: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    start: str,
    end: str,
    icir_min: float = 0.1,
    window: int = 60,
    min_periods: int = 20,
) -> Tuple[List[str], pd.Series]:
    """Keep L2 factors whose mean rolling ICIR over [start,end] exceeds icir_min."""
    if not panels:
        return [], pd.Series(dtype=float)
    print("[mf] FS-1 ICIR screen on {} factors ...".format(len(panels)), flush=True)
    _, icir_hist = calc_icir_weights(panels, fwd_ret, window=window, min_periods=min_periods)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    icir_test = icir_hist.loc[(icir_hist.index >= start_ts) & (icir_hist.index <= end_ts)]
    icir_mean = icir_test.mean(numeric_only=True).sort_values(ascending=False)
    selected = [n for n, v in icir_mean.items() if pd.notna(v) and float(v) > icir_min]
    print(
        "[mf] FS-1 screen: {}/{} with ICIR>{:.2f}".format(len(selected), len(icir_mean), icir_min),
        flush=True,
    )
    if not icir_mean.empty:
        top = icir_mean.head(10)
        print("[mf] FS-1 top ICIR:\n{}".format(top.round(3).to_string()), flush=True)
    return selected, icir_mean


def diagnose_factors(
    panels: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    start: str,
    end: str,
    icir_window: int = 60,
    min_periods: int = 20,
    source_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Per-factor Rank-IC / rolling-ICIR diagnostics over [start, end]."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    rows = []
    ic_store: Dict[str, pd.Series] = {}
    for name, pan in panels.items():
        print("[diagnose] IC {}".format(name), flush=True)
        ic = _daily_spearman_ic(pan, fwd_ret)
        ic = ic.loc[(ic.index >= start_ts) & (ic.index <= end_ts)]
        ic_store[name] = ic
        mu = ic.rolling(icir_window, min_periods=min_periods).mean()
        sd = ic.rolling(icir_window, min_periods=min_periods).std(ddof=0)
        icir = mu / sd.replace(0.0, np.nan)
        wealth = (1.0 + ic.fillna(0.0) * 0.01).cumprod()
        peak = wealth.cummax()
        dd = (wealth / peak - 1.0).min() if len(wealth) else np.nan
        n_obs = int(ic.notna().sum())
        rows.append(
            {
                "factor": name,
                "source": (source_map or {}).get(name, "core" if name in CORE_FACTOR_NAMES else "l2"),
                "n_ic_days": n_obs,
                "date_min": str(ic.dropna().index.min().date()) if n_obs else None,
                "date_max": str(ic.dropna().index.max().date()) if n_obs else None,
                "ic_mean": float(ic.mean()) if n_obs else np.nan,
                "ic_std": float(ic.std(ddof=0)) if n_obs else np.nan,
                "ic_ir": float(ic.mean() / ic.std(ddof=0)) if n_obs and ic.std(ddof=0) > 0 else np.nan,
                "ic_pos_ratio": float((ic > 0).mean()) if n_obs else np.nan,
                "icir_mean": float(icir.mean()) if icir.notna().any() else np.nan,
                "icir_std": float(icir.std(ddof=0)) if icir.notna().any() else np.nan,
                "icir_pos_ratio": float((icir > 0).mean()) if icir.notna().any() else np.nan,
                "ic_path_max_dd": float(dd) if pd.notna(dd) else np.nan,
            }
        )
    stats = pd.DataFrame(rows).sort_values("icir_mean", ascending=False)
    return stats, ic_store


def cluster_dedup_factors(
    panels: Dict[str, pd.DataFrame],
    candidates: List[str],
    icir_rank: pd.Series,
    corr_thresh: float = 0.7,
    sample_days: int = 120,
) -> Tuple[List[str], pd.DataFrame]:
    """Greedy keep: sort by ICIR desc, drop later factors with |corr|>thresh vs kept."""
    if not candidates:
        return [], pd.DataFrame()
    # sample cross-sectional stacked correlation via date-wise concat of z-scores
    dates = sorted(set().union(*[panels[n].index for n in candidates if n in panels]))
    if not dates:
        return list(candidates), pd.DataFrame()
    # take last sample_days overlapping dates for speed
    dates = dates[-sample_days:]
    mats = []
    for n in candidates:
        if n not in panels:
            continue
        w = panels[n].reindex(index=dates).astype(float)
        # flatten to series indexed by (date, symbol) — correlate across names on pooled rows
        mats.append(w.stack(future_stack=True).rename(n))
    if not mats:
        return list(candidates), pd.DataFrame()
    wide = pd.concat(mats, axis=1)
    corr = wide.corr(method="pearson")
    order = sorted(candidates, key=lambda x: float(icir_rank.get(x, -np.inf)), reverse=True)
    kept: List[str] = []
    dropped = []
    for f in order:
        if f not in corr.columns:
            kept.append(f)
            continue
        conflict = False
        for k in kept:
            if k in corr.columns and abs(float(corr.loc[f, k])) > corr_thresh:
                conflict = True
                dropped.append({"factor": f, "dropped_vs": k, "corr": float(corr.loc[f, k])})
                break
        if not conflict:
            kept.append(f)
    return kept, corr


def run_diagnose(args: argparse.Namespace) -> Dict:
    """Full-history factor ICIR diagnosis + optional corr dedup + slim backtest."""
    cfg = Config()
    start, end = args.start, args.end
    index_code = args.index
    do_neutral = not args.no_neutral
    with_fs1 = bool(getattr(args, "with_fs1", False))
    # diagnose: load FS-1 when panel exists unless --no-fs1-diagnose
    load_fs1 = with_fs1 or (not bool(getattr(args, "no_fs1_diagnose", False)))
    fs1_root = Path(getattr(args, "fs1_root", None) or FS1_DEFAULT_ROOT)
    icir_window = int(getattr(args, "icir_window", 60))
    icir_min = float(getattr(args, "diagnose_icir_min", 0.05))
    corr_thresh = float(getattr(args, "diagnose_corr_max", 0.7))

    print("\n" + "=" * 70)
    print("[因子诊断] {}  {}→{}  neutral={}".format(index_code, start, end, do_neutral))
    print("=" * 70)

    preheat = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    mask = db.get_index_member_mask(index_code, preheat, end_plus)
    if mask.empty:
        return {"error": "empty member mask"}
    mask.index = pd.to_datetime(mask.index).normalize()
    symbols = sorted(mask.columns.astype(str).tolist())
    print("[diagnose] union={} mean_daily≈{:.0f}".format(len(symbols), float(mask.notna().sum(axis=1).mean())), flush=True)

    ind_map: Dict[str, str] = {}
    if do_neutral:
        ind_map = db.get_citics_l1_industry_map(symbols, asof=end)

    raw, log_mkt, fwd_ret, exec_df = load_raw_factor_panels(symbols, start, end)
    panels = process_factors(raw, log_mkt, ind_map, mask, do_neutral=do_neutral)
    source_map = {n: "core" for n in panels}

    fs1_note = "not_loaded"
    if load_fs1 and fs1_root.exists() and (fs1_root / "processed_ind_cap_z_v1").exists():
        print("[diagnose] load all eligible FS-1 factors (coverage may be partial) ...", flush=True)
        elig = _eligible_fs1_factors(fs1_root)
        long_df = load_fs1_processed_long(fs1_root, start, end, symbols=symbols, factors=elig)
        fs1_panels = fs1_long_to_panels(long_df, elig, mask, re_zscore=True)
        for n, p in fs1_panels.items():
            panels[n] = p
            source_map[n] = "l2_fs1"
        fs1_note = "loaded_n={}_rows={}".format(len(fs1_panels), len(long_df))
        del long_df, fs1_panels
    else:
        fs1_note = "fs1_skipped_or_missing"

    stats, ic_store = diagnose_factors(
        panels, fwd_ret, start, end, icir_window=icir_window, source_map=source_map
    )
    result_dir = Path(cfg.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tag = "{}_diagnose_{}_{}".format(
        index_code.replace(".", ""),
        pd.Timestamp(start).strftime("%Y%m%d"),
        pd.Timestamp(end).strftime("%Y%m%d"),
    )
    out_csv = result_dir / "factor_icir_full_history.csv"

    # 1.2 screen — require meaningful date coverage so short-window L2 doesn't dominate
    n_calendar = max(1, int((pd.Timestamp(end) - pd.Timestamp(start)).days * 252 / 365))
    stats["coverage_ratio"] = stats["n_ic_days"] / float(n_calendar)
    stats = stats.sort_values("icir_mean", ascending=False)
    stats.to_csv(out_csv, index=False)
    stats.to_csv(result_dir / "factor_icir_full_history_{}.csv".format(tag), index=False)

    print("\n[诊断] top ICIR:\n{}".format(stats.head(15).to_string(index=False)), flush=True)
    print("[诊断] wrote {}".format(out_csv), flush=True)

    full_cov = stats["coverage_ratio"] >= 0.5
    cand = stats.loc[(stats["icir_mean"] > icir_min) & full_cov, "factor"].tolist()
    partial = stats.loc[(stats["icir_mean"] > icir_min) & ~full_cov, "factor"].tolist()
    print(
        "[diagnose] ICIR>{:.2f} & coverage≥50%: {}/{} (partial-coverage excluded: {})".format(
            icir_min, len(cand), len(stats), len(partial)
        ),
        flush=True,
    )
    if partial:
        print("[diagnose] partial (mostly L2 2023-24 only): {}".format(partial[:15]), flush=True)

    # 1.3 corr cluster
    icir_rank = stats.set_index("factor")["icir_mean"]
    kept, corr = cluster_dedup_factors(panels, cand, icir_rank, corr_thresh=corr_thresh)
    print(
        "[diagnose] after corr dedup |ρ|>{:.2f}: {} kept".format(corr_thresh, len(kept)),
        flush=True,
    )
    print("[diagnose] slim set: {}".format(kept), flush=True)
    if not corr.empty:
        corr.to_csv(result_dir / "factor_corr_candidates_{}.csv".format(tag))

    slim_path = result_dir / "factor_slim_set_{}.json".format(tag)
    slim_payload = {
        "icir_min": icir_min,
        "corr_thresh": corr_thresh,
        "candidates_icir_gt_min": cand,
        "slim_after_dedup": kept,
        "fs1": fs1_note,
        "n_factors_diagnosed": len(stats),
    }
    with open(slim_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(slim_payload), f, indent=2, ensure_ascii=False)
    with open(result_dir / "factor_slim_set.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(slim_payload), f, indent=2, ensure_ascii=False)

    # 1.4 slim backtest with ICIR weights
    slim_metrics = {}
    if kept and not getattr(args, "diagnose_skip_backtest", False):
        print("[diagnose] slim ICIR backtest n={} ...".format(len(kept)), flush=True)
        use = {n: panels[n] for n in kept if n in panels}
        weights, icir_hist = calc_icir_weights(use, fwd_ret, window=icir_window, min_periods=20)
        score = composite_scores(use, weights, factor_names=list(use.keys()))
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        score = score.loc[(score.index >= start_ts) & (score.index <= end_ts)]
        score_long = score.stack(future_stack=True).rename("score").reset_index()
        score_long.columns = ["date", "symbol", "score"]
        score_long["date"] = pd.to_datetime(score_long["date"]).dt.normalize()
        score_long = score_long.dropna(subset=["score"])
        merged = score_long.merge(exec_df, on=["date", "symbol"], how="inner")
        merged = merged.rename(
            columns={"close": "close_px", "next_open": "open_px", "next_close": "next_close_px"}
        )
        merged["tradable_exec"] = merged["next_tradable"].notna()
        merged = merged.dropna(subset=["score", "open_px", "next_close_px", "next_date"])
        merged = merged[merged["tradable_exec"]].copy()
        bt = RotationBacktester(
            initial_cash=cfg.initial_cash,
            cost_rate=cfg.cost_rate,
            top_k=None,
            bottom_k=None,
            top_frac=float(args.top_frac),
            bottom_frac=float(args.bottom_frac),
            long_gross=cfg.rotation_long_gross,
            short_gross=cfg.rotation_short_gross,
            rebalance_every=int(args.rebalance_every),
            use_vol_scaling=False,
        )
        result = bt.run(merged)
        slim_metrics = result.metrics or {}
        avg_to = float(result.equity["turnover"].mean()) if not result.equity.empty else 0.0
        print(
            "[diagnose slim] sharpe={:.3f} ann={:.2f}% mdd={:.2f}% to={:.2f}%".format(
                float(slim_metrics.get("sharpe", 0.0)),
                100 * float(slim_metrics.get("annualized_return", 0.0)),
                100 * float(slim_metrics.get("max_drawdown", 0.0)),
                100 * avg_to,
            ),
            flush=True,
        )
        if not result.equity.empty:
            result.equity.to_csv(result_dir / "multi_factor_slim_{}_equity.csv".format(tag))
        slim_payload["slim_backtest"] = {
            "strategy": slim_metrics,
            "avg_daily_turnover": avg_to,
            "factors": list(use.keys()),
        }
        with open(slim_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(slim_payload), f, indent=2, ensure_ascii=False)
        with open(result_dir / "factor_slim_set.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(slim_payload), f, indent=2, ensure_ascii=False)
        with open(result_dir / "multi_factor_slim_{}_summary.json".format(tag), "w", encoding="utf-8") as f:
            json.dump(_json_safe(slim_payload), f, indent=2, ensure_ascii=False)

    summary = {
        "scheme": "factor_diagnose",
        "index_code": index_code,
        "start": start,
        "end": end,
        "fs1": fs1_note,
        "n_diagnosed": len(stats),
        "n_icir_gt_min": len(cand),
        "n_slim": len(kept),
        "slim_factors": kept,
        "icir_min": icir_min,
        "corr_thresh": corr_thresh,
        "top_icir": stats.head(20).to_dict(orient="records"),
        "slim_backtest": slim_payload.get("slim_backtest"),
        "outputs": {
            "factor_icir_full_history": str(out_csv),
            "slim_set": str(slim_path),
        },
    }
    with open(result_dir / "factor_diagnose_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    print("[结果] {}".format(result_dir / "factor_diagnose_summary.json"), flush=True)
    return summary


def run(args: argparse.Namespace) -> Dict:
    global FACTOR_NAMES
    cfg = Config()
    start = args.start
    end = args.end
    index_code = args.index
    do_neutral = not args.no_neutral
    with_fs1 = bool(getattr(args, "with_fs1", False))
    fs1_root = Path(getattr(args, "fs1_root", None) or FS1_DEFAULT_ROOT)
    fs1_icir_min = float(getattr(args, "fs1_icir_min", 0.1))

    print("\n" + "=" * 70)
    print(
        "[多因子选股] {}  weights={}  neutral={}  fs1={}".format(
            index_code, args.weights, do_neutral, with_fs1
        )
    )
    print("=" * 70)

    preheat = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    print("[mf] index mask ...", flush=True)
    mask = db.get_index_member_mask(index_code, preheat, end_plus)
    if mask.empty:
        return {"error": "empty member mask"}
    mask.index = pd.to_datetime(mask.index).normalize()
    symbols = sorted(mask.columns.astype(str).tolist())
    print(
        "[mf] union={} mean_daily≈{:.0f}".format(len(symbols), float(mask.notna().sum(axis=1).mean())),
        flush=True,
    )

    ind_map: Dict[str, str] = {}
    if do_neutral:
        print("[mf] Citics L1 map ...", flush=True)
        ind_map = db.get_citics_l1_industry_map(symbols, asof=end)
        print("[mf] industry covered {}/{}".format(len(ind_map), len(symbols)), flush=True)

    raw, log_mkt, fwd_ret, exec_df = load_raw_factor_panels(symbols, start, end)
    panels = process_factors(raw, log_mkt, ind_map, mask, do_neutral=do_neutral)

    fs1_selected: List[str] = []
    fs1_icir_all: Dict[str, float] = {}
    if with_fs1:
        print("[mf] load FS-1 processed panel from {}".format(fs1_root), flush=True)
        elig = _eligible_fs1_factors(fs1_root)
        long_df = load_fs1_processed_long(fs1_root, start, end, symbols=symbols, factors=elig)
        print(
            "[mf] FS-1 long rows={} factors={} symbols={}".format(
                len(long_df), len(elig), long_df["Symbol"].nunique() if not long_df.empty else 0
            ),
            flush=True,
        )
        fs1_panels_all = fs1_long_to_panels(long_df, elig, mask, re_zscore=True)
        fs1_selected, fs1_icir_series = screen_fs1_icir(
            fs1_panels_all,
            fwd_ret,
            start=start,
            end=end,
            icir_min=fs1_icir_min,
            window=int(getattr(args, "icir_window", 60)),
            min_periods=20,
        )
        fs1_icir_all = {
            str(k): (None if pd.isna(v) else float(v)) for k, v in fs1_icir_series.items()
        }
        # free memory of non-selected panels
        for name in fs1_selected:
            panels[name] = fs1_panels_all[name]
        del fs1_panels_all, long_df

    names = [n for n in CORE_FACTOR_NAMES if n in panels] + [
        n for n in fs1_selected if n in panels and n not in CORE_FACTOR_NAMES
    ]
    FACTOR_NAMES = list(names)
    if not names:
        return {"error": "no factors available"}

    icir_window = int(getattr(args, "icir_window", 60))
    if args.weights == "equal":
        all_dates = sorted(set().union(*[panels[n].index for n in names]))
        weights = pd.DataFrame(1.0 / len(names), index=all_dates, columns=names)
        icir_hist = pd.DataFrame(np.nan, index=all_dates, columns=names)
    else:
        print(
            "[mf] rolling ICIR weights (window={}, n_factors={}) ...".format(icir_window, len(names)),
            flush=True,
        )
        # only score with active factor set
        use_panels = {n: panels[n] for n in names}
        weights, icir_hist = calc_icir_weights(
            use_panels, fwd_ret, window=icir_window, min_periods=20
        )

    score = composite_scores(panels, weights, factor_names=names)
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    score = score.loc[(score.index >= start_ts) & (score.index <= end_ts)]

    score_long = score.stack(future_stack=True).rename("score").reset_index()
    score_long.columns = ["date", "symbol", "score"]
    score_long["date"] = pd.to_datetime(score_long["date"]).dt.normalize()
    score_long = score_long.dropna(subset=["score"])

    merged = score_long.merge(exec_df, on=["date", "symbol"], how="inner")
    merged = merged.rename(
        columns={
            "close": "close_px",
            "next_open": "open_px",
            "next_close": "next_close_px",
        }
    )
    merged["tradable_exec"] = merged["next_tradable"].notna()
    merged = merged.dropna(subset=["score", "open_px", "next_close_px", "next_date"])
    merged = merged[merged["tradable_exec"]].copy()
    print(
        "[mf] signal panel rows={} days={} names={}".format(
            len(merged), merged["date"].nunique(), merged["symbol"].nunique()
        ),
        flush=True,
    )

    bt = RotationBacktester(
        initial_cash=cfg.initial_cash,
        cost_rate=cfg.cost_rate,
        top_k=None,
        bottom_k=None,
        top_frac=float(args.top_frac),
        bottom_frac=float(args.bottom_frac),
        long_gross=cfg.rotation_long_gross,
        short_gross=cfg.rotation_short_gross,
        rebalance_every=int(args.rebalance_every),
        use_vol_scaling=False,
    )
    result = bt.run(merged)
    metrics = result.metrics or {}
    bh = result.equal_weight_bh_metrics or {}
    avg_to = float(result.equity["turnover"].mean()) if not result.equity.empty else 0.0
    sr = float(metrics.get("sharpe", 0.0))
    ann = float(metrics.get("annualized_return", 0.0))

    # factor ICIR stats (mean over test window)
    icir_test = icir_hist.loc[(icir_hist.index >= start_ts) & (icir_hist.index <= end_ts)]
    icir_mean = icir_test.mean(numeric_only=True)
    n_pos = int((icir_mean > 0.1).sum()) if not icir_mean.empty else 0

    print("\n[多因子回测结果]", flush=True)
    print("  年化: {:.2f}%".format(100 * ann))
    print("  夏普: {:.3f}".format(sr))
    print("  回撤: {:.2f}%".format(100 * float(metrics.get("max_drawdown", 0.0))))
    print("  换手: {:.2f}%".format(100 * avg_to))
    print("  ICIR>0.1 因子数: {}/{}".format(n_pos, len(names)))
    if fs1_selected:
        print("  并入 FS-1 L2: {}".format(fs1_selected), flush=True)
    if not icir_mean.empty:
        print("  因子ICIR均值:\n{}".format(icir_mean.round(3).to_string()), flush=True)

    result_dir = Path(cfg.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tag = index_code.replace(".", "") + "_" + args.weights + ("_neut" if do_neutral else "_raw")
    if with_fs1:
        tag = tag + "_fs1"
    tag = tag + "_{}_{}".format(
        pd.Timestamp(start).strftime("%Y%m%d"),
        pd.Timestamp(end).strftime("%Y%m%d"),
    )
    if not result.equity.empty:
        result.equity.to_csv(result_dir / "multi_factor_equity.csv")
        result.equity.to_csv(result_dir / "multi_factor_{}_equity.csv".format(tag))
    merged.to_csv(result_dir / "multi_factor_{}_signals.csv".format(tag), index=False)

    expo = pd.DataFrame({"icir_mean": icir_mean})
    if args.weights == "icir":
        expo["weight_mean"] = weights.loc[(weights.index >= start_ts) & (weights.index <= end_ts)].mean()
    else:
        expo["weight_mean"] = 1.0 / len(names)
    expo.to_csv(result_dir / "factor_exposure_stats.csv")
    expo.to_csv(result_dir / "factor_exposure_stats_{}.csv".format(tag))
    if fs1_icir_all:
        pd.Series(fs1_icir_all, name="icir_mean").sort_values(ascending=False).to_csv(
            result_dir / "fs1_icir_screen_{}.csv".format(tag)
        )

    summary = {
        "scheme": "multi_factor_linear_fs1" if with_fs1 else "multi_factor_linear",
        "index_code": index_code,
        "weights": args.weights,
        "industry_size_neutral": do_neutral,
        "factors": names,
        "core_factors": [n for n in CORE_FACTOR_NAMES if n in names],
        "fs1_factors_selected": fs1_selected,
        "fs1_icir_min": fs1_icir_min if with_fs1 else None,
        "config": {
            "start": start,
            "end": end,
            "top_frac": args.top_frac,
            "bottom_frac": args.bottom_frac,
            "rebalance_every": args.rebalance_every,
            "mad_n": 5,
            "icir_window": icir_window,
            "with_fs1": with_fs1,
            "fs1_root": str(fs1_root) if with_fs1 else None,
        },
        "strategy": metrics,
        "equal_weight_bh": bh,
        "avg_daily_turnover": avg_to,
        "factor_icir_mean": {k: (None if pd.isna(v) else float(v)) for k, v in icir_mean.items()},
        "n_factors_icir_gt_0_1": n_pos,
        "selection_stats": result.selection_stats,
        "comparison": {
            "baseline_6f_icir_sharpe": 2.436,
            "layer1_no_indneut_sharpe": 1.010,
            "layer1_indneut_sharpe": 0.707,
            "delta_vs_6f_baseline": sr - 2.436,
            "delta_vs_layer1_no_indneut": sr - 1.010,
        },
    }
    safe = _json_safe(summary)
    with open(result_dir / "multi_factor_summary.json", "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(result_dir / "multi_factor_{}_summary.json".format(tag), "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, allow_nan=False)
    print("\n[结果] {}".format(result_dir / "multi_factor_summary.json"), flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transparent multi-factor CSI selector")
    p.add_argument("--index", default="000300.SH")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--top-frac", type=float, default=0.2)
    p.add_argument("--bottom-frac", type=float, default=0.2)
    p.add_argument("--weights", choices=["equal", "icir"], default="icir")
    p.add_argument("--no-neutral", action="store_true", help="Skip size+industry neutralization")
    p.add_argument("--rebalance-every", type=int, default=7)
    p.add_argument("--icir-window", type=int, default=60)
    p.add_argument("--with-fs1", action="store_true", help="Merge ICIR-screened FS-1 L2 factors")
    p.add_argument("--fs1-root", default=str(FS1_DEFAULT_ROOT))
    p.add_argument("--fs1-icir-min", type=float, default=0.1, help="Min mean rolling ICIR to keep L2 factor")
    p.add_argument("--diagnose", action="store_true", help="Per-factor ICIR diagnosis + corr dedup + slim BT")
    p.add_argument("--no-fs1-diagnose", action="store_true", help="Skip FS-1 load in diagnose mode")
    p.add_argument("--diagnose-icir-min", type=float, default=0.05)
    p.add_argument("--diagnose-corr-max", type=float, default=0.7)
    p.add_argument("--diagnose-skip-backtest", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = run_diagnose(args) if args.diagnose else run(args)
    if "error" in out:
        print("[阻断] {}".format(out), flush=True)
        sys.exit(2)
