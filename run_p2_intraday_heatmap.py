#!/usr/bin/env python
"""P2 intraday HML heatmaps via the original minute-bar backtest template.

Stamps daily `net_active_flow_mktcap_20d` onto standard bartimes
(09:59 … 14:29), runs get_cs_group_performance on ZZ1000, and writes
annualized-return / Sharpe heatmaps (same as Intraday_Factor_Test_Process).

Usage:
  OMP_NUM_THREADS=1 python run_p2_intraday_heatmap.py
  OMP_NUM_THREADS=1 python run_p2_intraday_heatmap.py --sample-days 504
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import factor_config as cfg
import intraday_lib
from COMMON_CONST import DATA_DB_CONN
from data_preheat import preheat_ret_matrix
from factor_attribution import cs_zscore
from factor_data_loaders import connect_ddb, load_eod_enriched_tables
from factor_formulas_l2_flow_p2 import build_net_active_flow_mktcap
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel
from l2_data_loaders import build_l2_daily_cache

OUT = Path("research/reports/l2_flow_density_v1/heatmaps")
FACTOR = "net_active_flow_mktcap_20d"
INDEX = "000852.SH"
SHARE = "PREHEAT_RET_MATRIX_ZZ1000"
BARTIMES = [
    (9, 59),
    (10, 29),
    (10, 59),
    (11, 29),
    (13, 29),
    (13, 59),
    (14, 29),
]
RET_COLS = [
    "Ret_15",
    "Ret_30",
    "Ret_60",
    "Ret_90",
    "Ret_120",
    "Ret_180",
    "Ret_EOD",
    "Ret_NDay",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_preheat(start: dt.datetime, end: dt.datetime) -> None:
    s = connect_ddb()
    s.run(intraday_lib.ddb_functions)
    try:
        ok = bool(s.run(f'defined("{SHARE}", SHARED)'))
    except Exception:
        ok = False
    if ok:
        # sanity: non-empty
        try:
            n = int(s.run(f"size({SHARE})"))
            log(f"{SHARE}: OK (size={n})")
            if n > 0:
                s.close()
                return
        except Exception as exc:
            log(f"{SHARE} defined but unreadable ({exc}); rebuilding…")
    else:
        log(f"{SHARE} missing — preheating ZZ1000 ret matrix (slow)…")
    s.close()
    preheat_ret_matrix(start, end, INDEX)
    log(f"{SHARE}: preheated")


def panel_to_narrow(panel: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    """Wide Date×Symbol → narrow tradetime/symbol/factorname/value at all bartimes."""
    wide = panel.dropna(how="all")
    stacked = wide.stack()
    stacked = stacked.reset_index()
    stacked.columns = ["Date", "symbol", "value"]
    stacked["Date"] = pd.to_datetime(stacked["Date"])
    stacked = stacked.dropna(subset=["value"])

    pieces = []
    for h, m in BARTIMES:
        part = stacked.copy()
        part["tradetime"] = part["Date"] + pd.Timedelta(hours=h, minutes=m)
        part["factorname"] = factor_name
        pieces.append(part[["tradetime", "symbol", "factorname", "value"]])
    out = pd.concat(pieces, ignore_index=True)
    # DDB-friendly types
    out["symbol"] = out["symbol"].astype(str)
    out["factorname"] = out["factorname"].astype(str)
    out["value"] = out["value"].astype(float)
    return out


def run_heatmap(session, narrow: pd.DataFrame, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    upload = "P2_net_active_flow"
    session.upload({upload: narrow})
    session.run(
        f"""
        signal = {upload}
        index_code = "{INDEX}"
        signal = select *, date(tradetime) as Date from signal
        signal = filter_in_index(signal, index_code)
        group_data_ret, summary = get_cs_group_performance(signal, {SHARE}, group_num=5)
        """
    )
    signal_data = session.run("signal")
    intraday_lib.get_signal_count(signal_data, save_path=str(save_path))

    group_data_ret = session.run("group_data_ret")
    group_data_ret = intraday_lib.subtract_market_return(group_data_ret)
    # Persist the normalized excess-group panel consumed by the analyzer.
    group_data_ret.to_parquet(save_path / "group_data_ret.parquet", index=False)

    performance_results = intraday_lib.analyze_group_performance_by_bartime(
        group_data_ret,
        ret_columns=RET_COLS,
        save_plots=True,
        show_plots=False,
        save_path=str(save_path),
    )
    intraday_lib.create_group_heatmap(
        performance_results,
        group_name="group_HML",
        key_name="annualized_return",
        save_plot=True,
        show_plot=False,
        save_path=str(save_path),
    )
    intraday_lib.create_group_heatmap(
        performance_results,
        group_name="group_HML",
        key_name="sharpe",
        save_plot=True,
        show_plot=False,
        high_contrast=True,
        save_path=str(save_path),
    )
    intraday_lib.save_performance_summary(
        performance_results,
        filename=str(save_path / "group_performance_summary.csv"),
    )
    log(f"Heatmaps -> {save_path}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-days",
        type=int,
        default=504,
        help="Use last N trading days of the P2 panel (default 504 discovery-style)",
    )
    parser.add_argument(
        "--skip-preheat-check",
        action="store_true",
        help="Assume PREHEAT_RET_MATRIX_ZZ1000 already shared",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    start, end = cfg.START_DAY, cfg.END_DAY
    preheat = start - dt.timedelta(days=cfg.PREHEAT_CALENDAR_DAYS)
    log(f"=== P2 intraday heatmap | factor={FACTOR} | last {args.sample_days}d ===")

    if not args.skip_preheat_check:
        ensure_preheat(start, end)

    enriched, session = load_eod_enriched_tables(preheat, end)
    session.run(intraday_lib.ddb_functions)
    industry = load_citics_industry_panel(start, end)
    l2_cache = build_l2_daily_cache(preheat, end, session=session, close=enriched.close)

    float_mkt = enriched.float_mktcap.loc[start:end]
    ind = industry.reindex(index=float_mkt.index, columns=float_mkt.columns)
    raw = build_net_active_flow_mktcap(l2_cache, float_mkt, window=20).loc[start:end]
    neut = cs_zscore(neutralize_size_industry(raw, ind, float_mkt))

    # last N calendar-aligned rows present in panel
    panel = neut.dropna(how="all")
    if args.sample_days and len(panel) > args.sample_days:
        panel = panel.iloc[-args.sample_days :]
    log(f"Panel: {panel.index[0].date()} -> {panel.index[-1].date()} ({len(panel)}d)")

    # shift 1d: T signal for T+1 (avoid same-bar look-ahead vs minute rets)
    panel = panel.shift(1).iloc[1:]
    narrow = panel_to_narrow(panel, FACTOR)
    log(f"Narrow rows: {len(narrow):,} ({len(BARTIMES)} bartimes)")

    save_path = OUT / FACTOR
    run_heatmap(session, narrow, save_path)

    # short README
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# P2 Intraday Heatmaps",
                "",
                f"**Factor:** `{FACTOR}` (daily, size+industry neut, shift-1)",
                f"**Universe:** CSI1000 (`{INDEX}`)",
                f"**Sample:** last {args.sample_days}d stamped onto bartimes {BARTIMES}",
                "",
                "Pipeline: `Intraday_Factor_Test_Process` / `intraday_lib.create_group_heatmap`",
                "",
                f"## Outputs (`{FACTOR}/`)",
                "- `group_HML_annualized_return_heatmap.png`",
                "- `group_HML_sharpe_heatmap.png`",
                "- `group_performance_summary.csv`",
                "- `group_data_ret.parquet`",
                "",
                "Note: same daily signal is stamped on all bartimes — heatmap variation",
                "is mainly across **return horizons**, not true intraday signal decay.",
                "True open/close cumulative-flow variants belong in P2 deepen.",
                "",
            ]
        )
    )
    log("Done.")


if __name__ == "__main__":
    main()
