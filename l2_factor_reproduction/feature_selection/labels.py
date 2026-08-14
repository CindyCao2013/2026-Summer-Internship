"""FS-3 PIT-safe future excess-return labels (standalone Y contract).

Reuses fast_context ret_matrix + benchmark_return:
  daily excess = stock_c2c - benchmark_c2c vs 000852.SH
  multi-day: compound stock and bench separately, then subtract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT, UNIVERSE
from l2_factor_reproduction.python.fast_discovery import context_paths

LABEL_CONTRACT_VERSION = "fs3_labels_v1"
CANONICAL_HORIZONS: Tuple[int, ...] = (1, 5, 20)
HORIZON_COL = {1: "y_1d", 5: "y_5d", 20: "y_20d"}


def label_contract_dict() -> Dict[str, object]:
    return {
        "contract_version": LABEL_CONTRACT_VERSION,
        "horizons_trading_days": list(CANONICAL_HORIZONS),
        "benchmark_id": UNIVERSE,
        "daily_return_source": "fast_context/{window}/ret_matrix.parquet",
        "daily_method": "c2c benchmark-relative excess (stock - index)",
        "multi_day_method": (
            "stock_cum = prod(1+r_stock)-1; "
            "bench_cum = prod(1+r_bench)-1; "
            "excess = stock_cum - bench_cum"
        ),
        "feature_date_T_maps_to": "returns on trading days T+1 .. T+h inclusive",
        "no_truncation": True,
    }


def load_daily_excess_and_bench(
    window: str = "full",
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Load daily excess ret matrix, benchmark series, and trading calendar."""
    paths = context_paths(window)
    ret = pd.read_parquet(paths["ret_matrix"])
    ret.index = pd.to_datetime(ret.index).normalize()
    bench = pd.read_parquet(paths["benchmark_return"])
    bench.index = pd.to_datetime(bench.index).normalize()
    if "benchmark_ret" in bench.columns:
        b = bench["benchmark_ret"]
    else:
        b = bench.iloc[:, 0]
    b = b.reindex(ret.index)
    dates = pd.DatetimeIndex(sorted(ret.index.unique()))
    return ret, b, dates


def recover_stock_returns(
    excess: pd.DataFrame,
    bench: pd.Series,
) -> pd.DataFrame:
    """stock_daily = excess + bench (same day)."""
    return excess.add(bench, axis=0)


def next_trading_dates(
    dates: pd.DatetimeIndex,
    feature_date: pd.Timestamp,
    horizon: int,
) -> Optional[pd.DatetimeIndex]:
    """Return exactly ``horizon`` trading dates strictly after feature_date."""
    feature_date = pd.Timestamp(feature_date).normalize()
    # positions strictly after feature_date
    pos = dates.searchsorted(feature_date, side="right")
    end = pos + horizon
    if end > len(dates):
        return None
    return dates[pos:end]


def compound_excess_path(
    stock_rets: pd.Series,
    bench_rets: pd.Series,
) -> float:
    """Compound stock and bench over the same dates; return excess."""
    s = stock_rets.astype(float)
    b = bench_rets.astype(float)
    if s.isna().any() or b.isna().any():
        return float("nan")
    stock_cum = float(np.prod(1.0 + s.to_numpy()) - 1.0)
    bench_cum = float(np.prod(1.0 + b.to_numpy()) - 1.0)
    return stock_cum - bench_cum


def build_labels_for_symbols(
    excess: pd.DataFrame,
    bench: pd.Series,
    dates: pd.DatetimeIndex,
    symbols: Sequence[str],
    horizons: Sequence[int] = CANONICAL_HORIZONS,
) -> pd.DataFrame:
    """Build long label table for selected symbols (memory-friendly)."""
    stock = recover_stock_returns(excess, bench)
    cols = [c for c in symbols if c in stock.columns]
    stock = stock.reindex(columns=cols)
    excess = excess.reindex(columns=cols)

    rows: List[Dict[str, object]] = []
    date_list = list(dates)
    for i, feat_dt in enumerate(date_list):
        row_base = {"TradeDate": feat_dt, "benchmark_id": UNIVERSE}
        # precompute horizons windows
        windows: Dict[int, Optional[pd.DatetimeIndex]] = {}
        for h in horizons:
            if i + h >= len(date_list):
                windows[h] = None
            else:
                # trading days strictly after feat_dt: date_list[i+1 : i+1+h]
                windows[h] = pd.DatetimeIndex(date_list[i + 1 : i + 1 + h])

        # vectorized per-horizon across symbols
        for h in horizons:
            w = windows[h]
            col_y = HORIZON_COL[h]
            start_c = f"label_start_{h}d"
            end_c = f"label_end_{h}d"
            valid_c = f"label_valid_{h}d"
            if w is None or len(w) < h:
                for sym in cols:
                    rows.append(
                        {
                            **row_base,
                            "Symbol": sym,
                            col_y: np.nan,
                            start_c: pd.NaT,
                            end_c: pd.NaT,
                            valid_c: False,
                            "_h": h,
                        }
                    )
                continue
            # slice
            s_block = stock.loc[w, cols]
            b_block = bench.reindex(w)
            # invalid if any NA in path for that symbol
            for j, sym in enumerate(cols):
                s_path = s_block.iloc[:, j]
                yv = compound_excess_path(s_path, b_block)
                ok = np.isfinite(yv)
                rows.append(
                    {
                        **row_base,
                        "Symbol": sym,
                        col_y: yv if ok else np.nan,
                        start_c: w[0],
                        end_c: w[-1],
                        valid_c: bool(ok),
                        "_h": h,
                    }
                )

    # The loop above creates one row per (date, symbol, horizon) — need pivot to wide
    # Rebuild more efficiently:
    return _pivot_label_rows(rows, horizons)


def _pivot_label_rows(
    rows: List[Dict[str, object]],
    horizons: Sequence[int],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # merge horizons into one row per TradeDate×Symbol
    keys = ["TradeDate", "Symbol", "benchmark_id"]
    out = df[keys].drop_duplicates().copy()
    for h in horizons:
        sub = df.loc[df["_h"] == h, keys + [HORIZON_COL[h], f"label_start_{h}d", f"label_end_{h}d", f"label_valid_{h}d"]]
        out = out.merge(sub, on=keys, how="left")
    return out


def build_labels_wide_panel(
    excess: pd.DataFrame,
    bench: pd.Series,
    dates: pd.DatetimeIndex,
    horizons: Sequence[int] = CANONICAL_HORIZONS,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[int, pd.DataFrame]:
    """Return {horizon: DataFrame date×symbol of labels} plus meta frames.

    Efficient numpy path for full universe.
    """
    stock = recover_stock_returns(excess, bench)
    if symbols is not None:
        cols = [c for c in symbols if c in stock.columns]
        stock = stock.reindex(columns=cols)
    else:
        cols = list(stock.columns)

    stock = stock.reindex(index=dates)
    bench_a = bench.reindex(index=dates)
    n_dates, n_sym = stock.shape
    stock_v = stock.to_numpy(dtype=float)
    bench_v = bench_a.to_numpy(dtype=float)

    out: Dict[int, pd.DataFrame] = {}
    meta_start: Dict[int, pd.Series] = {}
    meta_end: Dict[int, pd.Series] = {}
    meta_valid: Dict[int, pd.DataFrame] = {}

    for h in horizons:
        lab = np.full((n_dates, n_sym), np.nan, dtype=float)
        valid = np.zeros((n_dates, n_sym), dtype=bool)
        starts = pd.Series(pd.NaT, index=dates)
        ends = pd.Series(pd.NaT, index=dates)
        for i in range(n_dates):
            j0 = i + 1
            j1 = i + 1 + h
            if j1 > n_dates:
                break
            # compound over [j0, j1)
            s_path = stock_v[j0:j1, :]  # h × n_sym
            b_path = bench_v[j0:j1]  # h
            # invalid where any stock NA or any bench NA
            bad_s = ~np.isfinite(s_path).all(axis=0)
            bad_b = not np.isfinite(b_path).all()
            if bad_b:
                continue
            stock_cum = np.prod(1.0 + s_path, axis=0) - 1.0
            bench_cum = float(np.prod(1.0 + b_path) - 1.0)
            y = stock_cum - bench_cum
            y[bad_s] = np.nan
            lab[i, :] = y
            valid[i, :] = np.isfinite(y)
            starts.iloc[i] = dates[j0]
            ends.iloc[i] = dates[j1 - 1]
        out[h] = pd.DataFrame(lab, index=dates, columns=cols)
        meta_start[h] = starts
        meta_end[h] = ends
        meta_valid[h] = pd.DataFrame(valid, index=dates, columns=cols)

    # stash meta on dict under special keys for caller
    out["_meta_start"] = meta_start  # type: ignore
    out["_meta_end"] = meta_end  # type: ignore
    out["_meta_valid"] = meta_valid  # type: ignore
    return out


def write_label_partitions(
    label_map: Dict,
    out_dir: Path,
    horizons: Sequence[int] = CANONICAL_HORIZONS,
) -> None:
    """Write wide label matrices + start/end/valid metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_start = label_map["_meta_start"]
    meta_end = label_map["_meta_end"]
    meta_valid = label_map["_meta_valid"]
    for h in horizons:
        hdir = out_dir / f"horizon={h}"
        hdir.mkdir(parents=True, exist_ok=True)
        label_map[h].to_parquet(hdir / "y_wide.parquet")
        meta_valid[h].to_parquet(hdir / "label_valid.parquet")
        pd.DataFrame(
            {
                "TradeDate": label_map[h].index,
                f"label_start_{h}d": meta_start[h].to_numpy(),
                f"label_end_{h}d": meta_end[h].to_numpy(),
            }
        ).to_parquet(hdir / "label_bounds.parquet", index=False)

    cov_rows = []
    for h in horizons:
        y = label_map[h]
        v = meta_valid[h]
        cov_rows.append(
            {
                "horizon": h,
                "n_dates": int(y.shape[0]),
                "n_symbols": int(y.shape[1]),
                "n_valid": int(v.to_numpy().sum()),
                "valid_ratio": float(v.to_numpy().mean()),
                "date_min": str(y.index.min().date()),
                "date_max": str(y.index.max().date()),
            }
        )
    pd.DataFrame(cov_rows).to_csv(out_dir / "label_coverage.csv", index=False)
    (out_dir / "label_contract.json").write_text(
        json.dumps(label_contract_dict(), indent=2), encoding="utf-8"
    )


def label_contract_hash() -> str:
    payload = json.dumps(label_contract_dict(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def audit_label_parity_sample(
    excess: pd.DataFrame,
    bench: pd.Series,
    dates: pd.DatetimeIndex,
    label_y1: pd.DataFrame,
    *,
    n_dates: int = 5,
    n_symbols: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare Y1 to excess.loc[next_day] (same as T+1 daily excess)."""
    rng = np.random.default_rng(seed)
    # usable feature dates: need at least 1 future day
    usable = dates[:-1]
    pick_dates = sorted(rng.choice(usable, size=min(n_dates, len(usable)), replace=False))
    rows = []
    for dt in pick_dates:
        dt = pd.Timestamp(dt).normalize()
        nxt = dates[dates.searchsorted(dt, side="right")]
        # random symbols with finite excess on nxt
        finite_syms = excess.columns[np.isfinite(excess.loc[nxt].to_numpy())]
        if len(finite_syms) == 0:
            continue
        pick_syms = list(
            rng.choice(finite_syms, size=min(n_symbols, len(finite_syms)), replace=False)
        )
        for sym in pick_syms:
            ref = float(excess.loc[nxt, sym])
            lab = float(label_y1.loc[dt, sym]) if sym in label_y1.columns else np.nan
            # For h=1, compound(stock)-compound(bench) == stock-bench daily
            abs_diff = abs(ref - lab) if np.isfinite(ref) and np.isfinite(lab) else np.nan
            rows.append(
                {
                    "TradeDate": str(dt.date()),
                    "Symbol": sym,
                    "horizon": 1,
                    "reference_return": ref,
                    "fs3_label": lab,
                    "abs_diff": abs_diff,
                    "pass": bool(np.isfinite(abs_diff) and abs_diff < 1e-12),
                }
            )
    return pd.DataFrame(rows)


def audit_label_boundaries(
    dates: pd.DatetimeIndex,
    meta_end: Dict[int, pd.Series],
    meta_valid: Dict[int, pd.DataFrame],
    horizons: Sequence[int] = CANONICAL_HORIZONS,
) -> pd.DataFrame:
    """Check end-of-sample truncation and horizon lengths."""
    rows = []
    for h in horizons:
        ends = meta_end[h]
        valid = meta_valid[h]
        # last h feature dates must be invalid (no full future window)
        last_dates = dates[-h:]
        for dt in last_dates:
            vrate = float(valid.loc[dt].mean()) if dt in valid.index else np.nan
            rows.append(
                {
                    "check": "end_of_sample_no_truncation",
                    "horizon": h,
                    "TradeDate": str(pd.Timestamp(dt).date()),
                    "valid_ratio": vrate,
                    "pass": bool(np.isfinite(vrate) and vrate == 0.0),
                }
            )
        # for a mid date, end - start should span h trading days
        mid = dates[len(dates) // 2]
        if pd.notna(ends.loc[mid]):
            # count trading days in (feature, end]
            feat_pos = dates.get_loc(mid)
            end_pos = dates.get_loc(pd.Timestamp(ends.loc[mid]))
            n_fwd = int(end_pos - feat_pos)
            rows.append(
                {
                    "check": "exact_horizon_trading_days",
                    "horizon": h,
                    "TradeDate": str(pd.Timestamp(mid).date()),
                    "valid_ratio": float(valid.loc[mid].mean()),
                    "pass": n_fwd == h,
                    "n_fwd_trading_days": n_fwd,
                }
            )
    return pd.DataFrame(rows)
