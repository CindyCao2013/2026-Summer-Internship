#!/usr/bin/env python3
"""Sprint 3.1 production-path trace + Python vs DDB backtest parity.

Does NOT modify Intraday_Factor_Test_Process.py / intraday_lib.py.

Usage::

    # 1) Trace production path + signal parity (May 2024 sample)
    python factors/intraday/close_vwap_deviation/run_backtest_parity.py --trace --signal

    # 2) Full backtest parity (uses factor_config date window; needs DDB + PREHEAT)
    RUN_DDB_TESTS=1 python factors/intraday/close_vwap_deviation/run_backtest_parity.py --backtest

    # 3) All checks
    RUN_DDB_TESTS=1 python factors/intraday/close_vwap_deviation/run_backtest_parity.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import unittest.mock
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTOR = "close_vwap_deviation"
PARITY_ROOT = ROOT / "result" / "intraday" / "_parity_close_vwap_deviation"
SIGNAL_SAMPLE_START = "2024-05-01"
SIGNAL_SAMPLE_END = "2024-05-31"


def trace_production_path() -> None:
    """Intraday path → build_intraday_narrow_table → compute → ddb_version."""
    import factor_config as cfg
    from intraday_formulas import build_intraday_narrow_table
    from minute_bar_store import get_default_store

    fake = pd.DataFrame(
        {
            "bartime": [pd.Timestamp("2024-05-01 09:59:00")],
            "symbol": ["600000.SH"],
            "factorname": [FACTOR],
            "value": [0.01],
        }
    )
    store = get_default_store(start_date=SIGNAL_SAMPLE_START)
    with unittest.mock.patch(
        "factors.intraday.close_vwap_deviation.compute.ddb_version",
        return_value=fake,
    ) as mock_ddb:
        with unittest.mock.patch.object(cfg, "INTRADAY_CLOSE_VWAP_USE_DDB", True):
            out = build_intraday_narrow_table(
                FACTOR,
                SIGNAL_SAMPLE_START,
                SIGNAL_SAMPLE_END,
                store=store,
            )
    mock_ddb.assert_called_once()
    assert list(out.columns) == ["tradetime", "symbol", "factorname", "value"]
    print("TRACE OK: build_intraday_narrow_table → compute_close_vwap_deviation → ddb_version")


def _build_signals(use_ddb: bool, start: str, end: str) -> pd.DataFrame:
    import factor_config as cfg
    from intraday_formulas import build_intraday_narrow_table
    from minute_bar_store import get_default_store

    hist = getattr(cfg, "INTRADAY_ALPHA_STORE_START", cfg.MINUTE_BAR_HISTORY_START)
    store = get_default_store(start_date=hist)
    with unittest.mock.patch.object(cfg, "INTRADAY_CLOSE_VWAP_USE_DDB", use_ddb):
        return build_intraday_narrow_table(FACTOR, start, end, store=store)


def compare_signals(py: pd.DataFrame, db: pd.DataFrame) -> Dict[str, Any]:
    from factors.intraday.close_vwap_deviation.test_compare import assert_consistency

    py_n = py.rename(columns={"tradetime": "bartime"})
    db_n = db.rename(columns={"tradetime": "bartime"})
    metrics = assert_consistency(py_n, db_n)
    bt_py = sorted(pd.to_datetime(py["tradetime"]).dt.strftime("%H:%M").unique())
    bt_db = sorted(pd.to_datetime(db["tradetime"]).dt.strftime("%H:%M").unique())
    uni_py = py.groupby("tradetime")["symbol"].nunique()
    uni_db = db.groupby("tradetime")["symbol"].nunique()
    return {
        **metrics,
        "signal_rows_python": len(py),
        "signal_rows_ddb": len(db),
        "bartimes_python": bt_py,
        "bartimes_ddb": bt_db,
        "universe_mean_python": float(uni_py.mean()) if len(uni_py) else 0.0,
        "universe_mean_ddb": float(uni_db.mean()) if len(uni_db) else 0.0,
        "universe_by_bartime_max_diff": float((uni_py - uni_db).abs().max())
        if len(uni_py) and len(uni_db) and uni_py.index.equals(uni_db.index)
        else None,
    }


def run_signal_parity(start: str, end: str) -> Dict[str, Any]:
    print(f"SIGNAL parity {start} → {end} ...", flush=True)
    py = _build_signals(False, start, end)
    db = _build_signals(True, start, end)
    metrics = compare_signals(py, db)
    print("SIGNAL PASS", json.dumps(metrics, indent=2, default=str), flush=True)
    return metrics


def _extract_hl_sharpe(summary_csv: Path, ret_window: str = "Ret_15") -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    hl = df[(df["Return_Window"] == ret_window) & (df["Group"] == "group_HML")].copy()
    hl["Bartime"] = pd.to_datetime(hl["Bartime"]).dt.strftime("%H:%M")
    return hl.set_index("Bartime")[["Sharpe_Ratio", "Annualized_Return"]]


def _extract_ic_stats_from_process_output(stdout: str) -> Dict[str, float]:
    """Best-effort parse from Intraday_Factor_Test_Process console (IC_Mean lines)."""
    # Full IC table is printed by DDB; parity script also reads from saved plots metadata.
    return {}


def _run_backtest_once(label: str, use_ddb: bool, *, eval_start: Optional[str], eval_end: Optional[str]) -> Path:
    out_dir = PARITY_ROOT / label
    PARITY_ROOT.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["INTRADAY_CLOSE_VWAP_USE_DDB"] = "true" if use_ddb else "false"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    # Launcher: patch factor list + copy results after run
    eval_patch = ""
    if eval_start and eval_end:
        eval_patch = f"""
cfg.INTRADAY_MINUTE_EVAL_START = __import__('datetime').datetime.fromisoformat("{eval_start}")
cfg.INTRADAY_MINUTE_EVAL_END = __import__('datetime').datetime.fromisoformat("{eval_end}")
"""

    launcher = f"""
import shutil
from pathlib import Path
import factor_config as cfg

cfg.INTRADAY_CUSTOM_FACTOR_LIST = ["{FACTOR}"]
cfg.INTRADAY_CLOSE_VWAP_USE_DDB = {use_ddb}
{eval_patch}

src = Path("result/intraday/{FACTOR}")
dst = Path("{out_dir}")
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)

# Run main process (module-level execution)
with open("Intraday_Factor_Test_Process.py", encoding="utf-8") as f:
    code = compile(f.read(), "Intraday_Factor_Test_Process.py", "exec")
    exec(code, {{"__name__": "__main__", "__file__": "Intraday_Factor_Test_Process.py"}})

if src.exists():
    shutil.copytree(src, dst)
"""
    print(f"BACKTEST {label} (DDB={use_ddb}) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", launcher],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    log_path = out_dir / "run.log"
    log_path.write_text(proc.stdout + "\n---stderr---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Backtest {label} failed (see {log_path}):\n{proc.stderr[-2000:]}")
    print(f"BACKTEST {label} done → {out_dir}", flush=True)
    return out_dir


def compare_backtest_results(py_dir: Path, db_dir: Path) -> Dict[str, Any]:
    py_csv = py_dir / "group_performance_summary.csv"
    db_csv = db_dir / "group_performance_summary.csv"
    if not py_csv.exists() or not db_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {py_csv} / {db_csv}")

    py_hl = _extract_hl_sharpe(py_csv)
    db_hl = _extract_hl_sharpe(db_csv)
    joined = py_hl.join(db_hl, lsuffix="_py", rsuffix="_ddb", how="inner")
    joined["sharpe_abs_diff"] = (joined["Sharpe_Ratio_py"] - joined["Sharpe_Ratio_ddb"]).abs()
    joined["sharpe_rel_err"] = joined["sharpe_abs_diff"] / joined["Sharpe_Ratio_py"].abs().replace(0, np.nan)

    # Parse signal count from run logs if present
    def _signal_rows(log_path: Path) -> Optional[int]:
        if not log_path.exists():
            return None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if f"[BUILD] {FACTOR} rows=" in line:
                return int(line.split("rows=")[1].replace(",", "").split()[0])
        return None

    report = {
        "hl_sharpe_max_abs_diff": float(joined["sharpe_abs_diff"].max()),
        "hl_sharpe_max_rel_err": float(joined["sharpe_rel_err"].max()),
        "hl_sharpe_by_bartime": joined.reset_index().to_dict(orient="records"),
        "signal_rows_python": _signal_rows(py_dir / "run.log"),
        "signal_rows_ddb": _signal_rows(db_dir / "run.log"),
    }

    # Thresholds aligned with Sprint 2/3.1 gates
    assert report["hl_sharpe_max_rel_err"] <= 0.01, report
    if report["signal_rows_python"] and report["signal_rows_ddb"]:
        row_diff = abs(report["signal_rows_python"] - report["signal_rows_ddb"])
        row_rel = row_diff / max(report["signal_rows_python"], 1)
        report["signal_row_rel_diff"] = row_rel
        if row_rel > 0.001:
            print(f"WARNING: signal row count diff {row_diff} ({row_rel:.4%})", flush=True)

    print("BACKTEST PARITY PASS", json.dumps(report, indent=2, default=str), flush=True)
    return report


def run_backtest_parity(eval_start: Optional[str] = None, eval_end: Optional[str] = None) -> Dict[str, Any]:
    if os.environ.get("RUN_DDB_TESTS") != "1":
        print("Set RUN_DDB_TESTS=1 for live DDB backtest parity.", flush=True)
        return {}
    py_dir = _run_backtest_once("python", use_ddb=False, eval_start=eval_start, eval_end=eval_end)
    db_dir = _run_backtest_once("ddb", use_ddb=True, eval_start=eval_start, eval_end=eval_end)
    return compare_backtest_results(py_dir, db_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint 3.1 close_vwap_deviation parity")
    parser.add_argument("--trace", action="store_true", help="Verify production dispatch path")
    parser.add_argument("--signal", action="store_true", help="Compare narrow signals (sample month)")
    parser.add_argument("--backtest", action="store_true", help="Full backtest Python vs DDB")
    parser.add_argument("--all", action="store_true", help="Run trace + signal + backtest")
    parser.add_argument("--backtest-start", default=None, help="Override INTRADAY_MINUTE_EVAL_START (YYYY-MM-DD)")
    parser.add_argument("--backtest-end", default=None, help="Override INTRADAY_MINUTE_EVAL_END (YYYY-MM-DD)")
    parser.add_argument("--start", default=SIGNAL_SAMPLE_START)
    parser.add_argument("--end", default=SIGNAL_SAMPLE_END)
    args = parser.parse_args()

    if not any([args.trace, args.signal, args.backtest, args.all]):
        parser.print_help()
        return 0

    if args.trace or args.all:
        trace_production_path()

    if args.signal or args.all:
        if os.environ.get("RUN_DDB_TESTS") != "1":
            print("Set RUN_DDB_TESTS=1 for live DDB signal parity.", flush=True)
            return 1
        run_signal_parity(args.start, args.end)

    if args.backtest or args.all:
        run_backtest_parity(eval_start=args.backtest_start, eval_end=args.backtest_end)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
