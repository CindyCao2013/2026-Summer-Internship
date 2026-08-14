#!/usr/bin/env python
"""Run the full AlphaNet chain. Default is the synthetic smoke path."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS = Path(__file__).resolve().parent


def _run(script: str, extra: list) -> int:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="smoke")
    p.add_argument("--live", action="store_true", help="use DolphinDB instead of synthetic smoke")
    args = p.parse_args()
    if not args.live or args.variant in ("smoke", "ci"):
        return _run("run_smoke.py", [])
    for script, extra in (
        ("run_prepare_data.py", []),
        ("run_train.py", ["--variant", args.variant]),
        ("run_evaluate.py", ["--variant", args.variant]),
        ("run_enhance.py", ["--variant", args.variant]),
        ("run_shap.py", ["--variant", args.variant]),
    ):
        code = _run(script, extra)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
