#!/usr/bin/env python
"""Re-apply strict triage to existing attribution CSV (no DDB reload)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import factor_config as cfg
from factor_attribution import (
    attribution_conclusion_loose,
    passes_strict_incremental,
    strict_triage_conclusion,
)

OUT = cfg.RESEARCH_DIR


def reapply(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        s = row
        ic_raw = row.get("ic_raw", float("nan"))
        ic_stack = row.get("ic_after_ohlcv_stack", float("nan"))
        df.at[i, "strict_pass"] = passes_strict_incremental(ic_raw, ic_stack)
        df.at[i, "conclusion_loose"] = attribution_conclusion_loose(s)
        df.at[i, "conclusion"] = strict_triage_conclusion(s)
    df.to_csv(path, index=False)
    return df


def main():
    targets = [
        OUT / "cn_broker_attribution.csv",
        OUT / "fundamental_attribution.csv",
    ]
    for p in targets:
        if not p.exists():
            print(f"SKIP {p} (not found)")
            continue
        df = reapply(p)
        n = int(df["strict_pass"].sum()) if "strict_pass" in df.columns else 0
        print(f"Updated {p} — strict_pass={n}/{len(df)}")
        print(df[["factor_name", "ic_raw", "ic_after_ohlcv_stack", "strict_pass", "conclusion"]].to_string(index=False))


if __name__ == "__main__":
    main()
