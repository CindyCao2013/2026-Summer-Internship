#!/usr/bin/env python
"""P1 — Intraday timing heatmaps for daily HF→EOD factors.

Usage examples::

  OMP_NUM_THREADS=1 python run_intraday_heatmap_diagnostics.py
  OMP_NUM_THREADS=1 python run_intraday_heatmap_diagnostics.py \\
      --factors TGD20,APM_ActiveV2 --universe CSI1000 \\
      --start 2021-01-01 --end 2024-06-30
  OMP_NUM_THREADS=1 python run_intraday_heatmap_diagnostics.py --use-ddb

Offline mode (default) loads factor panels from research/cache and builds
Rank-IC / HML heatmaps with a mock-or-cached return matrix helper.
``--use-ddb`` additionally runs get_cs_group_performance heatmaps.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd

import factor_config as cfg
from intraday_heatmap_lib import (
    DEFAULT_BARTIMES,
    DEFAULT_HORIZONS,
    UNIVERSE_INDEX,
    load_factor_panel_from_cache,
    run_ddb_group_heatmap,
    run_factor_heatmap_offline,
    stamp_panel_to_narrow,
)


OUT_ROOT = Path("research/results/intraday_heatmap")


def log(msg: str) -> None:
    print(msg, flush=True)


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d")


def _resolve_factors(arg: Optional[str]) -> List[str]:
    if arg:
        return [x.strip() for x in arg.split(",") if x.strip()]
    return list(
        getattr(
            cfg,
            "INTRADAY_HEATMAP_FACTORS",
            [
                "TGD20",
                "SmartMoneyActiveV2",
                "APM_ActiveV2",
                "IdealAmplitude_ActiveV2",
                "IdealReversal_ActiveV2",
            ],
        )
    )


def _load_ret_long_offline(
    start: dt.datetime,
    end: dt.datetime,
    bartimes: List[str],
    horizons: List[str],
    panel_symbols: List[str],
) -> pd.DataFrame:
    """Build a lightweight synthetic ret matrix aligned to panel symbols.

    Used when ``--ret-parquet`` is not provided and ``--use-ddb`` is off.
    Values are random-normal (reproducible) so the pipeline can be smoke-tested;
    for research conclusions pass a real ret parquet or ``--use-ddb``.
    """
    rng = np_random = __import__("numpy").random.default_rng(42)
    dates = pd.bdate_range(start, end)
    # subsample symbols for speed if huge
    syms = panel_symbols
    if len(syms) > 800:
        syms = list(syms[:800])
    rows = []
    for d in dates:
        for lab in bartimes:
            h, m = map(int, lab.split(":"))
            for sym in syms:
                rec = {
                    "symbol": sym,
                    "Date": pd.Timestamp(d),
                    "Bartime": dt.time(h, m),
                    "BartimeLabel": f"{h:02d}:{m:02d}",
                }
                for hz in horizons:
                    rec[hz] = float(rng.normal(0, 0.01))
                rows.append(rec)
    log(
        f"[offline-ret] synthetic ret_long rows={len(rows):,} "
        f"(set --ret-parquet or --use-ddb for real returns)"
    )
    return pd.DataFrame(rows)


def _load_ret_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Symbol" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"Symbol": "symbol"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday timing heatmap diagnostics")
    parser.add_argument("--factors", type=str, default=None, help="Comma-separated names")
    parser.add_argument(
        "--universe",
        type=str,
        default="ALL",
        choices=list(UNIVERSE_INDEX.keys()),
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default=str(OUT_ROOT),
        help="Output root (default research/results/intraday_heatmap)",
    )
    parser.add_argument(
        "--ret-parquet",
        type=str,
        default=None,
        help="Optional long ret matrix parquet (Symbol/Date/Bartime/Ret_*)",
    )
    parser.add_argument(
        "--use-ddb",
        action="store_true",
        help="Also run DDB get_cs_group_performance heatmaps (needs preheat)",
    )
    parser.add_argument(
        "--no-shift",
        action="store_true",
        help="Do not shift panel by 1 day (default shifts to avoid look-ahead)",
    )
    parser.add_argument(
        "--skip-limit-filter",
        action="store_true",
        help="Skip EOD not-limit mask",
    )
    args = parser.parse_args()

    start = _parse_date(args.start) if args.start else cfg.START_DAY
    end = _parse_date(args.end) if args.end else cfg.END_DAY
    factors = _resolve_factors(args.factors)
    bartimes = list(getattr(cfg, "INTRADAY_HEATMAP_BARTIMES", DEFAULT_BARTIMES))
    horizons = list(getattr(cfg, "INTRADAY_HEATMAP_HORIZONS", DEFAULT_HORIZONS))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    log("=== 分钟择时热力图诊断报告 ===")
    log(f"factors={factors}")
    log(f"universe={args.universe} | {start.date()} → {end.date()}")
    log(f"bartimes={bartimes}")
    log(f"horizons={horizons}")

    not_limit = None
    if not args.skip_limit_filter:
        try:
            import Factor_Dev_Lib as fdl

            not_limit = fdl.get_EOD_Not_Limit(start, end)
            log("Applied get_EOD_Not_Limit mask")
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: not_limit unavailable ({exc})")

    summary_blocks: List[str] = [
        "=== 分钟择时热力图诊断报告 ===",
        f"宇宙: {args.universe}, 区间: {start.date()} ~ {end.date()}",
        f"Bartimes: {', '.join(bartimes)}",
        f"Horizons: {', '.join(horizons)}",
        "",
    ]

    # Shared ret matrix (loaded lazily after first panel symbols known)
    ret_long: Optional[pd.DataFrame] = None
    session = None
    share = "PREHEAT_RET_MATRIX_ZZ1000"
    index_code = UNIVERSE_INDEX.get(args.universe) or "000852.SH"

    if args.use_ddb:
        from factor_data_loaders import connect_ddb
        import intraday_lib

        session = connect_ddb()
        session.run(intraday_lib.ddb_functions)
        try:
            ok = bool(session.run(f'defined("{share}", SHARED)'))
        except Exception:
            ok = False
        if not ok:
            log(f"WARNING: {share} missing — DDB heatmaps may fail; run data_preheat.py")

    for fname in factors:
        log(f"\n----- {fname} -----")
        try:
            panel = load_factor_panel_from_cache(fname, start, end)
        except FileNotFoundError as exc:
            log(f"[SKIP] {fname}: {exc}")
            summary_blocks.append(f"因子: {fname}\n  SKIP: {exc}\n")
            continue

        if ret_long is None:
            if args.ret_parquet:
                ret_long = _load_ret_parquet(Path(args.ret_parquet))
            elif args.use_ddb and session is not None:
                # Pull a slice of shared matrix into pandas once
                try:
                    ret_long = session.run(
                        f"""
                        select * from {share}
                        where Date >= {start.strftime('%Y.%m.%d')}
                          and Date <= {end.strftime('%Y.%m.%d')}
                        """
                    )
                    ret_long = pd.DataFrame(ret_long)
                    log(f"Loaded {share} rows={len(ret_long):,}")
                except Exception as exc:  # noqa: BLE001
                    log(f"WARNING: cannot pull {share} ({exc}); using synthetic ret")
                    ret_long = _load_ret_long_offline(
                        start, end, bartimes, horizons, list(panel.columns.astype(str))
                    )
            else:
                ret_long = _load_ret_long_offline(
                    start, end, bartimes, horizons, list(panel.columns.astype(str))
                )

        paths = run_factor_heatmap_offline(
            fname,
            panel,
            ret_long,
            out_dir=out_root,
            universe=args.universe,
            start=start,
            end=end,
            bartimes=bartimes,
            horizons=horizons,
            not_limit=not_limit,
            shift_days=0 if args.no_shift else 1,
        )
        summary_blocks.append((paths["summary"]).read_text(encoding="utf-8").rstrip())
        summary_blocks.append("")

        if args.use_ddb and session is not None:
            narrow = stamp_panel_to_narrow(
                panel.loc[start:end],
                fname,
                bartimes=bartimes,
                shift_days=0 if args.no_shift else 1,
            )
            ddb_dir = out_root / fname / "ddb"
            try:
                run_ddb_group_heatmap(
                    session,
                    narrow,
                    factor_name=fname,
                    index_code=index_code if args.universe != "ALL" else "000852.SH",
                    share_name=share,
                    out_dir=ddb_dir,
                    ret_columns=horizons,
                )
                log(f"DDB heatmaps -> {ddb_dir}/")
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING: DDB heatmap failed for {fname}: {exc}")

    summary_path = out_root / "summary.txt"
    summary_path.write_text("\n".join(summary_blocks) + "\n", encoding="utf-8")
    log(f"\nSummary -> {summary_path}")
    if session is not None:
        session.close()
    log("Done.")


if __name__ == "__main__":
    main()
