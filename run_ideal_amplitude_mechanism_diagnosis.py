#!/usr/bin/env python
"""III-A closure — IdealAmplitude mechanism diagnosis (no formula retune).

Diagnose mono≈0.11 vs high Sharpe:
  - payoff curve (decile / ventile mean returns)
  - tail contribution (extreme buckets share of H-L PnL)
  - U-shape / non-monotonic pattern tests

Outputs:
  research/reports/ideal_amplitude_v1/mechanism_diagnosis/
  docs/milestone_3_0_iiia_closure.md (written by companion / this script summary)

Usage:
  OMP_NUM_THREADS=1 python run_ideal_amplitude_mechanism_diagnosis.py --eval-days 252
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

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
from alpha_investability import daily_hl_pnl_and_turnover, series_performance
from factor_attribution import align_signal, cs_zscore
from factor_cutting.ideal_amplitude import compute_ideal_amplitude
from factor_data_loaders import load_eod_enriched_tables

OUT = Path("research/reports/ideal_amplitude_v1/mechanism_diagnosis")
CACHE = Path("research/cache/ideal_amplitude_panels")
SIGNAL_SHIFT = 1
TOP_FRAC = 0.10


def log(msg: str) -> None:
    print(msg, flush=True)


def bucket_means(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    n_groups: int,
    signal_shift: int = 1,
) -> pd.Series:
    sig = align_signal(signal, signal_shift)
    r = ret.reindex_like(sig)
    buckets = {i: [] for i in range(1, n_groups + 1)}
    for dt_ in sig.index:
        row_s = sig.loc[dt_]
        row_r = r.loc[dt_]
        m = row_s.notna() & row_r.notna()
        if m.sum() < n_groups * 5:
            continue
        ranks = row_s[m].rank(pct=True)
        rets = row_r[m]
        for g in range(1, n_groups + 1):
            lo = (g - 1) / n_groups
            hi = g / n_groups
            sel = (ranks >= lo) & (ranks < hi) if g < n_groups else ranks >= lo
            if sel.any():
                buckets[g].append(float(rets[sel].mean()))
    return pd.Series({g: np.mean(v) if v else np.nan for g, v in buckets.items()})


def mono_stats(payoff: pd.Series, *, prefer_decreasing: bool) -> dict:
    vals = payoff.dropna().values.astype(float)
    if len(vals) < 3:
        return {"mono_frac": np.nan, "spearman_vs_rank": np.nan, "u_shape_score": np.nan}
    diffs = np.diff(vals)
    if prefer_decreasing:
        mono_frac = float(np.mean(diffs < 0))
    else:
        mono_frac = float(np.mean(diffs > 0))
    ranks = np.arange(1, len(vals) + 1, dtype=float)
    spearman = float(pd.Series(vals).corr(pd.Series(ranks), method="spearman"))
    # U-shape: ends higher than middle (or lower for inverted U)
    mid = vals[len(vals) // 2]
    ends = 0.5 * (vals[0] + vals[-1])
    u_shape_score = float(ends - mid)  # >0 → U (ends above mid)
    return {
        "mono_frac": mono_frac,
        "spearman_vs_bucket": spearman,
        "u_shape_score": u_shape_score,
        "mid_minus_mean_ends": float(mid - ends),
    }


def tail_contribution(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    top_frac: float = 0.10,
    extreme_frac: float = 0.05,
) -> dict:
    """Share of H-L gross PnL attributable to extreme 5% vs next 5% within book."""
    sig = align_signal(signal, SIGNAL_SHIFT)
    r = ret.reindex_like(sig)
    ranks = sig.rank(axis=1, pct=True)
    # long = high ranks (after we may flip later); use raw signal book then flip by IC
    long_all = ranks >= (1 - top_frac)
    short_all = ranks <= top_frac
    long_ext = ranks >= (1 - extreme_frac)
    short_ext = ranks <= extreme_frac
    long_inner = long_all & ~long_ext
    short_inner = short_all & ~short_ext

    def _pnl(mask):
        w = mask.div(mask.sum(axis=1).replace(0, np.nan), axis=0)
        return w.mul(r).sum(axis=1)

    long_pnl = _pnl(long_all)
    short_pnl = _pnl(short_all)
    hl = long_pnl - short_pnl
    ext = _pnl(long_ext) - _pnl(short_ext)
    inn = _pnl(long_inner) - _pnl(short_inner)
    # direction so mean HL > 0
    direction = 1.0 if hl.mean() >= 0 else -1.0
    hl_a, ext_a, inn_a = direction * hl, direction * ext, direction * inn
    total = float(hl_a.sum())
    return {
        "direction": int(direction),
        "hl_mean": float(hl_a.mean()),
        "extreme_5pct_mean": float(ext_a.mean()),
        "inner_5to10pct_mean": float(inn_a.mean()),
        "extreme_share_of_cum_pnl": float(ext_a.sum() / total) if abs(total) > 1e-12 else np.nan,
        "inner_share_of_cum_pnl": float(inn_a.sum() / total) if abs(total) > 1e-12 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-days", type=int, default=252)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    log("=== IdealAmplitude mechanism diagnosis (no retune) ===")

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)

    close_full = enriched.close.loc[start:end]
    high_full = enriched.high.loc[start:end]
    low_full = enriched.low.loc[start:end]
    open_full = enriched.open.loc[start:end]
    ret_full = Factor_Dev_Lib.get_Ret_Matrix(start, end, method="c2c")

    warm = 40
    idx = close_full.index
    n = min(args.eval_days, len(idx))
    eval_index = idx[-n:]
    slice_start = idx[max(0, len(idx) - n - warm)]
    close = close_full.loc[slice_start:end]
    high = high_full.loc[slice_start:end]
    low = low_full.loc[slice_start:end]
    open_ = open_full.loc[slice_start:end]

    cache_tag = f"{close.index[0].date()}_{close.index[-1].date()}"
    cache_path = CACHE / f"ideal_amplitude_{cache_tag}.pkl"
    if cache_path.exists():
        log(f"Load cache {cache_path}")
        fac = pd.read_pickle(cache_path)
    else:
        log("Compute IdealAmplitude ...")
        fac = compute_ideal_amplitude(high, low, close, open_=open_, return_legs=False)
        CACHE.mkdir(parents=True, exist_ok=True)
        fac.to_pickle(cache_path)

    fac = fac.reindex(index=eval_index)
    signal = cs_zscore(fac)
    ret = ret_full.reindex(index=signal.index, columns=signal.columns)

    ic = daily_rank_ic_series(signal, ret, signal_shift=SIGNAL_SHIFT)
    prefer_decreasing = ic.mean() < 0  # paper negative IC → short high V
    direction = -1 if prefer_decreasing else 1
    sig_book = signal * direction

    decile = bucket_means(signal, ret, 10, SIGNAL_SHIFT)
    ventile = bucket_means(signal, ret, 20, SIGNAL_SHIFT)
    # signed book payoff (long high signal after direction flip)
    decile_book = bucket_means(sig_book, ret, 10, SIGNAL_SHIFT)
    ventile_book = bucket_means(sig_book, ret, 20, SIGNAL_SHIFT)

    mono_raw = mono_stats(decile, prefer_decreasing=prefer_decreasing)
    mono_book = mono_stats(decile_book, prefer_decreasing=False)
    mono_v = mono_stats(ventile_book, prefer_decreasing=False)

    tail = tail_contribution(sig_book, ret, top_frac=TOP_FRAC, extreme_frac=0.05)
    gross, to = daily_hl_pnl_and_turnover(
        sig_book, ret, top_frac=TOP_FRAC, bottom_frac=TOP_FRAC, signal_shift=SIGNAL_SHIFT
    )
    perf = series_performance((gross if gross.mean() >= 0 else -gross).dropna())

    # Classify pattern
    u = mono_book["u_shape_score"]
    mono_f = mono_book["mono_frac"]
    if mono_f >= 0.7:
        pattern = "mostly_monotonic"
    elif abs(u) > abs(np.nanmean(decile_book.values)) * 0.5 and u > 0:
        pattern = "u_shaped_suspect"
    elif abs(u) > abs(np.nanmean(decile_book.values)) * 0.5 and u < 0:
        pattern = "inverted_u_suspect"
    elif tail["extreme_share_of_cum_pnl"] is not None and tail["extreme_share_of_cum_pnl"] > 0.7:
        pattern = "tail_dominated"
    else:
        pattern = "non_monotonic_mixed"

    diagnosis = {
        "window": f"last_{n}d",
        "rank_ic": float(ic.mean()),
        "rank_icir": float(icir_from_daily(ic)),
        "gross_sharpe_signed": perf["sharpe"],
        "paper_prefer_decreasing_raw": prefer_decreasing,
        "mono_raw_decile": mono_raw,
        "mono_signed_decile": mono_book,
        "mono_signed_ventile": mono_v,
        "tail": tail,
        "pattern_label": pattern,
        "decile_raw_mean_ret": {str(k): float(v) if pd.notna(v) else None for k, v in decile.items()},
        "decile_signed_mean_ret": {
            str(k): float(v) if pd.notna(v) else None for k, v in decile_book.items()
        },
        "interpretation": {
            "high_sharpe_low_mono": (
                "H-L uses only extremes; middle deciles can be disordered "
                "without destroying top-vs-bottom edge."
            ),
            "do_not": "Do not retune formula; do not promote status without payoff fix.",
        },
    }
    (OUT / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"bucket": decile.index, "raw_mean_ret": decile.values}).to_csv(
        OUT / "payoff_decile_raw.csv", index=False
    )
    pd.DataFrame({"bucket": decile_book.index, "signed_mean_ret": decile_book.values}).to_csv(
        OUT / "payoff_decile_signed.csv", index=False
    )
    pd.DataFrame({"bucket": ventile_book.index, "signed_mean_ret": ventile_book.values}).to_csv(
        OUT / "payoff_ventile_signed.csv", index=False
    )
    pd.DataFrame([tail]).to_csv(OUT / "tail_contribution.csv", index=False)

    # Charts
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(decile_book.index.astype(str), decile_book.values)
    ax.set_title(f"IdealAmplitude signed decile payoff ({pattern})")
    ax.axhline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "payoff_decile.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ventile_book.index, ventile_book.values, marker="o", ms=3)
    ax.set_title("IdealAmplitude signed ventile payoff")
    ax.axhline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "payoff_ventile.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["extreme 5%", "inner 5–10%", "H-L 10%"],
        [tail["extreme_5pct_mean"], tail["inner_5to10pct_mean"], tail["hl_mean"]],
    )
    ax.set_title("Tail vs inner contribution (mean daily)")
    ax.axhline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "tail_vs_inner.png", dpi=140)
    plt.close(fig)

    log(f"pattern={pattern}")
    log(f"mono_signed_decile={mono_book['mono_frac']:.2f} spearman={mono_book['spearman_vs_bucket']:.2f}")
    log(
        f"tail extreme share={tail['extreme_share_of_cum_pnl']:.2f} "
        f"u_shape={mono_book['u_shape_score']:.5f}"
    )
    log(f"Wrote {OUT}")
    log("=== diagnosis complete (status unchanged) ===")


if __name__ == "__main__":
    main()
