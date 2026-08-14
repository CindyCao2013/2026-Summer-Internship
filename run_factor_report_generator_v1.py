#!/usr/bin/env python3
"""Materialize Factor Report Template v1 packs (TGD20 + FlowDensity20)."""

from __future__ import annotations

import argparse
import json
import sys

from factor_report_generator import (
    assemble_flow_density20,
    assemble_tgd20,
    validate_pack,
    write_factors_index,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Factor Report Generator v1")
    p.add_argument(
        "--factor",
        choices=("TGD20", "FlowDensity20", "all"),
        default="all",
        help="Which factor pack to assemble",
    )
    args = p.parse_args(argv)

    built = []
    if args.factor in ("TGD20", "all"):
        out = assemble_tgd20()
        built.append(("TGD20", out, validate_pack("TGD20")))
    if args.factor in ("FlowDensity20", "all"):
        out = assemble_flow_density20()
        built.append(("FlowDensity20", out, validate_pack("FlowDensity20")))

    write_factors_index()

    for name, out, v in built:
        print(f"[ok] {name} -> {out}")
        print(json.dumps(v, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
