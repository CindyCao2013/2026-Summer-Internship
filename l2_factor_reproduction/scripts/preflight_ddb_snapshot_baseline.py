#!/usr/bin/env python
"""Pre-flight gate for the Sprint 6A long-sample baseline run.

Frozen sequence (all must PASS before expand_ddb_snapshot_family.py
--force-backtest is allowed):

  1. py_compile on every pipeline module in the DDB snapshot chain
  2. golden unit tests (tests/test_ddb_snapshot_golden.py)
  3. factor coverage audit (factor_coverage.csv vs frozen expectations)
  4. narrow hash manifest verification (factor_narrow.parquet sha256
     must match narrow_manifest.json)

Usage:
    python preflight_ddb_snapshot_baseline.py
Exit code 0 = all gates passed; 1 = at least one gate failed.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    COVERAGE_THRESHOLD,
    FACTOR_NAMES,
    FORMULA_VERSION,
    SCHEMA_VERSION,
)

POOL_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "ddb_reference_snapshot_family"
)
SCRIPTS = [
    "l2_factor_reproduction/python/ch_ddb_snapshot.py",
    "l2_factor_reproduction/python/ddb_snapshot_factors.py",
    "l2_factor_reproduction/scripts/build_ddb_snapshot_primitive.py",
    "l2_factor_reproduction/scripts/build_ddb_snapshot_narrow.py",
    "l2_factor_reproduction/scripts/expand_ddb_snapshot_family.py",
    "l2_factor_reproduction/scripts/validate_ddb_snapshot_window.py",
]
GOLDEN_TEST = "l2_factor_reproduction/tests/golden_ddb_reference.py"
EXPECTED_DATE_MIN = pd.Timestamp("2019-01-01")
EXPECTED_DATE_MAX = pd.Timestamp("2026-07-31")
MIN_DATES = 1500  # long sample must span ~7.5y of trading days


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_compile() -> bool:
    ok = True
    for rel in SCRIPTS:
        path = PROJ_ROOT / rel
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"  [compile OK] {rel}")
        except py_compile.PyCompileError as exc:
            print(f"  [compile FAIL] {rel}: {exc}")
            ok = False
    return ok


def gate_unit_tests() -> bool:
    result = subprocess.run(
        [sys.executable, str(PROJ_ROOT / GOLDEN_TEST)],
        capture_output=True,
        text=True,
        cwd=str(PROJ_ROOT),
    )
    tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
    for line in tail:
        print(f"  [golden] {line}")
    passed = result.returncode == 0
    print(f"  [golden {'PASS' if passed else 'FAIL'}]")
    return passed


def _expected_trade_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Frozen trading calendar from the WIND EOD store (same source the
    backtest mask/returns are built on)."""
    import dolphindb as ddb

    from COMMON_CONST import DATA_DB_CONN

    session = ddb.session()
    session.connect(**DATA_DB_CONN)
    dates = session.run(
        "select distinct TRADE_DT as dt from "
        "loadTable('dfs://WIND.ASHAREEODPRICES', 'data') "
        f"where TRADE_DT >= {start:%Y.%m.%d}, TRADE_DT <= {end:%Y.%m.%d} "
        "order by dt"
    )
    return pd.DatetimeIndex(pd.to_datetime(dates["dt"]))


def gate_coverage() -> bool:
    """Requested start/end are calendar bounds, NOT required trading days.
    Validation is against the frozen trading calendar: expected vs observed
    vs missing vs unexpected dates."""
    path = POOL_DIR / "factor_coverage.csv"
    if not path.exists():
        print(f"  [coverage FAIL] {path} missing")
        return False
    coverage = pd.read_csv(path)
    ok = True
    missing = set(FACTOR_NAMES) - set(coverage["factor"])
    if missing:
        print(f"  [coverage FAIL] factors missing: {sorted(missing)}")
        ok = False

    requested_start = pd.Timestamp(EXPECTED_DATE_MIN)
    requested_end = pd.Timestamp(EXPECTED_DATE_MAX)
    expected = _expected_trade_dates(requested_start, requested_end)
    print(
        f"  [calendar] requested {requested_start.date()}..{requested_end.date()}"
        f" -> {len(expected)} expected trading days "
        f"({expected.min().date()}..{expected.max().date()})"
    )

    observed_by_factor = {}
    for name in FACTOR_NAMES:
        narrow = POOL_DIR / "factors" / name / "factor_narrow.parquet"
        if not narrow.exists():
            continue
        dates = pd.read_parquet(narrow, columns=["tradetime"])["tradetime"]
        observed_by_factor[name] = (
            pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique()
        )

    audit_rows = []
    for row in coverage.itertuples():
        problems = []
        if row.n_factor_rows <= 0:
            problems.append("empty narrow")
        if row.n_dates < MIN_DATES:
            problems.append(f"n_dates {row.n_dates} < {MIN_DATES}")
        observed = observed_by_factor.get(row.factor)
        if observed is not None:
            missing_dates = expected.difference(observed)
            unexpected = observed.difference(expected)
            if len(unexpected):
                problems.append(f"{len(unexpected)} dates outside calendar")
            # missing dates are tolerated only if they are genuine market-wide
            # data holes; any missing date is reported explicitly
            if len(missing_dates):
                problems.append(
                    f"{len(missing_dates)} expected trading dates missing "
                    f"(first: {missing_dates.min().date()})"
                )
            audit_rows.append(
                {
                    "factor": row.factor,
                    "requested_start": str(requested_start.date()),
                    "requested_end": str(requested_end.date()),
                    "expected_trade_dates": len(expected),
                    "observed_trade_dates": len(observed),
                    "missing_trade_dates": len(missing_dates),
                    "missing_first": str(missing_dates.min().date())
                    if len(missing_dates) else "",
                    "missing_last": str(missing_dates.max().date())
                    if len(missing_dates) else "",
                    "unexpected_dates": len(unexpected),
                }
            )
        status = "FAIL " + "; ".join(problems) if problems else "OK"
        if problems:
            ok = False
        print(f"  [coverage {status}] {row.factor}: rows={row.n_factor_rows:,}")
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(
            POOL_DIR / "trade_date_coverage_audit.csv", index=False
        )
    return ok


def gate_hashes() -> bool:
    manifest_path = POOL_DIR / "narrow_manifest.json"
    if not manifest_path.exists():
        print(f"  [hash FAIL] {manifest_path} missing")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("narrow_sha256")
    if not recorded:
        print("  [hash FAIL] narrow_manifest.json has no narrow_sha256 block")
        return False
    if manifest.get("formula_version") != FORMULA_VERSION:
        print(
            "  [hash FAIL] formula_version drift: manifest="
            f"{manifest.get('formula_version')} code={FORMULA_VERSION}"
        )
        return False
    if manifest.get("primitive_schema_version") != SCHEMA_VERSION:
        print(
            "  [hash FAIL] schema_version drift: manifest="
            f"{manifest.get('primitive_schema_version')} code={SCHEMA_VERSION}"
        )
        return False
    ok = True
    for name in FACTOR_NAMES:
        path = POOL_DIR / "factors" / name / "factor_narrow.parquet"
        if not path.exists():
            print(f"  [hash FAIL] {name}: narrow missing")
            ok = False
            continue
        actual = _sha256(path)
        match = recorded.get(name) == actual
        if not match:
            ok = False
        print(
            f"  [hash {'OK' if match else 'FAIL'}] {name} "
            f"sha256={actual[:12]}..."
        )
    return ok


def main() -> int:
    print("[gate 1/4] py_compile")
    g1 = gate_compile()
    print("[gate 2/4] golden unit tests")
    g2 = gate_unit_tests()
    print("[gate 3/4] factor coverage audit")
    g3 = gate_coverage()
    print("[gate 4/4] narrow hash manifest")
    g4 = gate_hashes()
    passed = all([g1, g2, g3, g4])
    print(
        f"[preflight {'PASS' if passed else 'FAIL'}] "
        f"compile={g1} golden={g2} coverage={g3} hash={g4}"
    )
    if not passed:
        print(
            "BLOCKED: expand_ddb_snapshot_family.py --force-backtest must not "
            "run until all gates pass"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
