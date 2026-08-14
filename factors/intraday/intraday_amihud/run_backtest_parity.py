#!/usr/bin/env python3
"""Run live signal and unchanged-engine backtest parity for intraday_amihud."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import unittest.mock
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTOR = "intraday_amihud"
# discovery_v1 derives this key mechanically from the complete factor name.
DISCOVERY_FLAG = "INTRADAY_AMIHUD_USE_DDB"
PARITY_ROOT = ROOT / "result" / "intraday" / "_parity_intraday_amihud"
SAMPLE_START = "2024-05-01"
SAMPLE_END = "2024-05-31"


def trace_production_path() -> None:
    """Prove the registered production dispatcher reaches discovery-v1 DDB."""
    from intraday_formulas import build_intraday_narrow_table

    fake = pd.DataFrame(
        {
            "bartime": [pd.Timestamp("2024-05-06 09:59:00")],
            "symbol": ["600000.SH"],
            "factorname": [FACTOR],
            "value": [1.0],
        }
    )
    with unittest.mock.patch(
        "factors.intraday.discovery_v1.ddb_version", return_value=fake
    ) as mock_ddb:
        with unittest.mock.patch.dict(os.environ, {DISCOVERY_FLAG: "true"}):
            out = build_intraday_narrow_table(
                FACTOR, SAMPLE_START, SAMPLE_END, store=None
            )
    mock_ddb.assert_called_once()
    assert list(out.columns) == ["tradetime", "symbol", "factorname", "value"]
    print(
        "TRACE OK: build_intraday_narrow_table -> discovery_v1.ddb_version",
        flush=True,
    )


def _build_signal(use_ddb: bool, start: str, end: str) -> pd.DataFrame:
    from intraday_formulas import build_intraday_narrow_table
    from minute_bar_store import get_default_store

    store = get_default_store(start_date=start)
    with unittest.mock.patch.dict(
        os.environ, {DISCOVERY_FLAG: "true" if use_ddb else "false"}
    ):
        return build_intraday_narrow_table(FACTOR, start, end, store=store)


def run_signal_parity(start: str, end: str) -> Dict[str, Any]:
    from factors.intraday.intraday_amihud.test_compare import assert_consistency

    print(f"SIGNAL parity {start} -> {end} ...", flush=True)
    py = _build_signal(False, start, end)
    db = _build_signal(True, start, end)
    metrics = assert_consistency(
        py.rename(columns={"tradetime": "bartime"}),
        db.rename(columns={"tradetime": "bartime"}),
    )
    print("SIGNAL PASS", json.dumps(metrics, indent=2, default=str), flush=True)
    return metrics


def _run_backtest_once(
    label: str,
    use_ddb: bool,
    *,
    eval_start: Optional[str],
    eval_end: Optional[str],
) -> Path:
    out_dir = PARITY_ROOT / label
    PARITY_ROOT.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    eval_patch = ""
    if eval_start and eval_end:
        eval_patch = f"""
cfg.INTRADAY_MINUTE_EVAL_START = dt.datetime.fromisoformat("{eval_start}")
cfg.INTRADAY_MINUTE_EVAL_END = dt.datetime.fromisoformat("{eval_end}")
"""

    launcher = f"""
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd
import factor_config as cfg
import intraday_lib

cfg.INTRADAY_CUSTOM_FACTOR_LIST = ["{FACTOR}"]
{eval_patch}

src = Path("result/intraday/{FACTOR}")
dst = Path(r"{out_dir}")
if src.exists():
    shutil.rmtree(src)

namespace = {{"__name__": "__main__", "__file__": "Intraday_Factor_Test_Process.py"}}
with open("Intraday_Factor_Test_Process.py", encoding="utf-8") as f:
    exec(compile(f.read(), "Intraday_Factor_Test_Process.py", "exec"), namespace)

if not src.exists():
    raise RuntimeError("Backtest produced no result directory")
if dst.exists():
    shutil.rmtree(dst)
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(src, dst)

s = namespace["s"]
pd.DataFrame(s.run("summary['ic_mean']")).to_csv(dst / "ic_mean.csv", index=False)
signal = pd.DataFrame(s.run("signal"))
signal["tradetime"] = pd.to_datetime(signal["tradetime"])
stats = {{
    "signal_count": int(len(signal)),
    "tradetime_count": int(signal["tradetime"].nunique()),
    "universe_count": int(signal["symbol"].nunique()),
    "universe_mean_per_signal": float(
        signal.groupby("tradetime")["symbol"].nunique().mean()
    ),
    "turnover_b_hl": float(intraday_lib.intraday_turnover_b_hl()),
}}
(dst / "evaluation_stats.json").write_text(
    json.dumps(stats, indent=2), encoding="utf-8"
)
"""

    env = os.environ.copy()
    env[DISCOVERY_FLAG] = "true" if use_ddb else "false"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"BACKTEST {label} (DDB={use_ddb}) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", launcher],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_path.write_text(
        proc.stdout + "\n---stderr---\n" + proc.stderr, encoding="utf-8"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Backtest {label} failed; see {log_path}:\n{proc.stderr[-2000:]}"
        )
    print(f"BACKTEST {label} done -> {out_dir}", flush=True)
    return out_dir


def _normalize_bartime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str)).dt.strftime("%H:%M:%S")


def _compare_ic(py_dir: Path, db_dir: Path) -> Dict[str, float]:
    py = pd.read_csv(py_dir / "ic_mean.csv")
    db = pd.read_csv(db_dir / "ic_mean.csv")
    ret_key = "RetType" if "RetType" in py.columns else "valueType"
    py["Bartime"] = _normalize_bartime(py["Bartime"])
    db["Bartime"] = _normalize_bartime(db["Bartime"])
    joined = py.merge(
        db,
        on=["Bartime", ret_key],
        suffixes=("_python", "_ddb"),
        how="outer",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise AssertionError("IC key mismatch between Python and DolphinDB")
    return {
        "ic_mean_max_abs_diff": float(
            (joined["IC_Mean_python"] - joined["IC_Mean_ddb"]).abs().max()
        ),
        "icir_max_abs_diff": float(
            (joined["IC_IR_python"] - joined["IC_IR_ddb"]).abs().max()
        ),
    }


def _compare_performance(py_dir: Path, db_dir: Path) -> Dict[str, float]:
    py = pd.read_csv(py_dir / "group_performance_summary.csv")
    db = pd.read_csv(db_dir / "group_performance_summary.csv")
    keys = ["Return_Window", "Bartime", "Group"]
    py["Bartime"] = _normalize_bartime(py["Bartime"])
    db["Bartime"] = _normalize_bartime(db["Bartime"])
    joined = py.merge(db, on=keys, suffixes=("_python", "_ddb"), how="outer")
    if len(joined) != len(py) or len(joined) != len(db):
        raise AssertionError("Performance summary key mismatch")
    diff = (joined["Sharpe_Ratio_python"] - joined["Sharpe_Ratio_ddb"]).abs()
    return {
        "excess_decile_sharpe_max_abs_diff": float(
            diff[joined["Group"].str.match(r"group_\d+")].max()
        ),
        "hl_sharpe_max_abs_diff": float(
            diff[joined["Group"] == "group_HML"].max()
        ),
    }


def compare_backtests(py_dir: Path, db_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    report.update(_compare_ic(py_dir, db_dir))
    report.update(_compare_performance(py_dir, db_dir))
    py_stats = json.loads((py_dir / "evaluation_stats.json").read_text())
    db_stats = json.loads((db_dir / "evaluation_stats.json").read_text())
    report["python_evaluation"] = py_stats
    report["ddb_evaluation"] = db_stats
    report["signal_count_diff"] = abs(
        py_stats["signal_count"] - db_stats["signal_count"]
    )
    report["universe_count_diff"] = abs(
        py_stats["universe_count"] - db_stats["universe_count"]
    )
    report["turnover_abs_diff"] = abs(
        py_stats["turnover_b_hl"] - db_stats["turnover_b_hl"]
    )

    numeric_gates = (
        "ic_mean_max_abs_diff",
        "icir_max_abs_diff",
        "excess_decile_sharpe_max_abs_diff",
        "hl_sharpe_max_abs_diff",
        "turnover_abs_diff",
    )
    for key in numeric_gates:
        if not np.isfinite(report[key]) or report[key] > 1e-10:
            raise AssertionError({key: report[key]})
    if report["signal_count_diff"] or report["universe_count_diff"]:
        raise AssertionError(report)
    print("BACKTEST PARITY PASS", json.dumps(report, indent=2), flush=True)
    return report


def run_backtest_parity(
    eval_start: Optional[str], eval_end: Optional[str]
) -> Dict[str, Any]:
    py_dir = _run_backtest_once(
        "python", False, eval_start=eval_start, eval_end=eval_end
    )
    db_dir = _run_backtest_once(
        "ddb", True, eval_start=eval_start, eval_end=eval_end
    )
    return compare_backtests(py_dir, db_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--signal", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--end", default=SAMPLE_END)
    parser.add_argument("--backtest-start", default=SAMPLE_START)
    parser.add_argument("--backtest-end", default=SAMPLE_END)
    args = parser.parse_args()

    if not any((args.trace, args.signal, args.backtest, args.all)):
        parser.print_help()
        return 0
    if args.trace or args.all:
        trace_production_path()
    if args.signal or args.backtest or args.all:
        if os.environ.get("RUN_DDB_TESTS") != "1":
            raise RuntimeError("Set RUN_DDB_TESTS=1 for live DDB validation")
    if args.signal or args.all:
        run_signal_parity(args.start, args.end)
    if args.backtest or args.all:
        run_backtest_parity(args.backtest_start, args.backtest_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
