#!/usr/bin/env python
"""D5 upside_fragility_20d — 2015–2016 A-share bull/crash window plots.

Same figure set as Library appendix:
  - cumulative_long_short.png (decile + H-L)
  - quantile_return.png
  - rank_ic_timeseries.png

Data note:
  DolphinDB ``WIND.ASHAREEODPRICES`` only starts 2018-01-02.
  This runner pulls OHLCV + c2c returns from Wind Oracle for the bull window.

Usage:
  OMP_NUM_THREADS=1 python run_d5_bull_2015_2016_plots.py
"""

from __future__ import annotations

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
from alpha_d4_expansion_stack import daily_rank_ic_series, decile_group_means
from alpha_research_report import monotonicity_score, save_cumulative_decile_figure
from factor_attribution import align_signal, hl_sharpe_from_composite
from factor_data_loaders import load_eod_wide_tables_from_wind_oracle
from factor_formulas import build_factor_cache
from factor_formulas_eod_engine import build_eod_engine_factor

# 2015–2016 covers the late bull peak (H1'15) and post-crash regime (H2'15–2016)
START = dt.datetime(2015, 1, 1)
END = dt.datetime(2016, 12, 31)
# D5 window=20d; short preheat (DDB default 400d is unnecessary and expensive on Oracle)
PREHEAT_DAYS = 60
FACTOR = "upside_fragility_20d"
OUT = Path("research/alpha_library_v1/figures/D5_upside_fragility/bull_2015_2016")
METRICS_OUT = Path("research/results/d5_bull_2015_2016")


def log(msg: str) -> None:
    print(msg, flush=True)


def build_group_cum_pnl(signal: pd.DataFrame, ret: pd.DataFrame, n: int = 10):
    """Silent groupTest → (cum_pnl, group_pnl, group_to)."""
    sig = align_signal(signal, 1)
    r = ret.reindex_like(sig)
    old_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        _, group_pnl_df, group_to_df = Factor_Dev_Lib.groupTest(sig, r, n=n, fee=0, info="silent")
    finally:
        plt.show = old_show
    return group_pnl_df.cumsum(), group_pnl_df, group_to_df


def save_ic_figure(ic_daily: pd.Series, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ic_daily.plot(ax=ax, alpha=0.35, linewidth=0.8, label="daily IC")
    ic_daily.rolling(20).mean().plot(ax=ax, color="red", label="20d MA")
    ic_daily.rolling(60, min_periods=20).mean().plot(ax=ax, color="darkblue", label="60d MA")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{name} — rank IC (2015–2016)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_quantile_bar(group_means: pd.Series, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    group_means.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"{name} — decile mean daily return (2015–2016)")
    ax.set_xlabel("Decile")
    ax.set_ylabel("Mean daily return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def window_metrics(signal: pd.DataFrame, ret: pd.DataFrame, label: str) -> dict:
    sharpe, ann_ret, direction = hl_sharpe_from_composite(signal, ret)
    ic_daily = daily_rank_ic_series(signal, ret)
    ic_mean = float(ic_daily.mean())
    icir = float(ic_mean / ic_daily.std() * np.sqrt(250)) if ic_daily.std() > 0 else np.nan
    gmeans = decile_group_means(signal, ret)
    mono = monotonicity_score(gmeans)
    pos_ratio = float((ic_daily > 0).mean()) if len(ic_daily.dropna()) else np.nan
    return {
        "window": label,
        "n_days": int(len(ret)),
        "start": str(ret.index[0].date()) if len(ret) else None,
        "end": str(ret.index[-1].date()) if len(ret) else None,
        "rank_ic": ic_mean,
        "icir": icir,
        "ic_positive_ratio": pos_ratio,
        "gross_hl_sharpe": sharpe,
        "hl_annu_ret": ann_ret,
        "direction": direction,
        "monotonicity": mono,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.mkdir(parents=True, exist_ok=True)

    preheat = START - dt.timedelta(days=PREHEAT_DAYS)
    log(f"=== D5 {FACTOR} | bull/crash window {START.date()} -> {END.date()} ===")
    log(
        f"Source: Wind Oracle ASHAREEODPRICES "
        f"(DDB EOD starts 2018; preheat {preheat.date()}, {PREHEAT_DAYS}d)"
    )

    eod, ret_full = load_eod_wide_tables_from_wind_oracle(preheat, END, keep_cache=False)
    if eod.close.empty:
        raise RuntimeError("Wind Oracle returned empty EOD for 2015–2016 window")

    pv_cache = build_factor_cache(
        df_close=eod.close,
        df_open=eod.open,
        df_high=eod.high,
        df_low=eod.low,
        df_volume=eod.volume,
        df_amount=eod.amount,
        df_turnover=eod.turnover,
    )

    wide = build_eod_engine_factor(FACTOR, pv_cache).loc[START:END]
    ret = ret_full.loc[START:END].reindex(columns=wide.columns)
    panel = wide.reindex(index=ret.index, columns=ret.columns)
    log(f"Panel: {panel.index[0].date()} -> {panel.index[-1].date()} ({len(panel)}d)")

    name = f"{FACTOR} | 2015-2016"
    from factor_runner import compute_group_stats, format_group_stats_title

    cum, group_pnl_df, group_to_df = build_group_cum_pnl(panel, ret)
    sig = align_signal(panel, 1)
    r = ret.reindex_like(sig)
    stats = compute_group_stats(sig, r, group_pnl_df, group_to_df)
    metrics_full = window_metrics(panel, ret, "2015_2016_full")
    title = format_group_stats_title(stats) + f", mono={metrics_full['monotonicity']:.2f}"
    save_cumulative_decile_figure(
        cum,
        factor_name=name,
        stats_title=title,
        out_path=OUT / "cumulative_long_short.png",
    )
    gmeans = decile_group_means(panel, ret)
    save_quantile_bar(gmeans, name, OUT / "quantile_return.png")
    ic_daily = daily_rank_ic_series(panel, ret)
    save_ic_figure(ic_daily, name, OUT / "rank_ic_timeseries.png")
    ic_daily.to_csv(OUT / "rank_ic_daily.csv", header=["rank_ic"])

    # Sub-windows: bull peak vs post-crash
    sub_rows = [metrics_full]
    for label, s, e in [
        ("2015H1_bull_peak", "2015-01-01", "2015-06-12"),
        ("2015H2_2016_post_crash", "2015-06-15", "2016-12-31"),
    ]:
        ret_s = ret.loc[s:e]
        if len(ret_s) < 40:
            continue
        sig_s = panel.reindex(index=ret_s.index, columns=ret_s.columns)
        m = window_metrics(sig_s, ret_s, label)
        sub_rows.append(m)
        log(
            f"  [{label}] IC={m['rank_ic']:.4f} ICIR={m['icir']:.2f} "
            f"gross Sharpe={m['gross_hl_sharpe']:.2f} mono={m['monotonicity']:.2f}"
        )

    summary = {
        "factor": FACTOR,
        "period": "2015-01-01 -> 2016-12-31",
        "data_source": "Wind Oracle ASHAREEODPRICES (DDB EOD unavailable before 2018-01-02)",
        "note": (
            "A-share 2015 bull peak + 2015–2016 post-crash regime. "
            "Same plot set as Library D5 appendix for regime stress test."
        ),
        "full_sample": metrics_full,
        "sub_windows": sub_rows[1:],
        "figures": {
            "cumulative": str(OUT / "cumulative_long_short.png"),
            "quantile": str(OUT / "quantile_return.png"),
            "ic_ts": str(OUT / "rank_ic_timeseries.png"),
        },
    }
    (METRICS_OUT / "d5_bull_2015_2016_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    pd.DataFrame(sub_rows).to_csv(METRICS_OUT / "d5_bull_2015_2016_windows.csv", index=False)

    log(
        f"\nFull 2015–2016: IC={metrics_full['rank_ic']:.4f} ICIR={metrics_full['icir']:.2f} "
        f"gross Sharpe={metrics_full['gross_hl_sharpe']:.2f} mono={metrics_full['monotonicity']:.2f}"
    )
    log(f"Figures -> {OUT}")
    log(f"Metrics -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
