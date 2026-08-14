#!/usr/bin/env python
"""Fast L2 Discovery Lane runner。

示例：

    # 一个 family 的全部冻结因子，discovery 冻结窗口
    python -m l2_factor_reproduction.scripts.fast_discovery \
        --family liquidity_impact --factors all --window discovery

    # 多个 family 指定因子
    python -m l2_factor_reproduction.scripts.fast_discovery \
        --family price_formation --factors close_auction_return,xxx

输出（fast_discovery/<window>/）：

    fast_summary.csv    每因子一行：十项指标 + gate 标记
    fast_profile.csv    分阶段耗时
    figures/<factor>/cumulative_hl.png
    figures/<factor>/decile_bar.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    FAMILY_ADAPTERS,
    FAST_DISCOVERY_DIR,
    WINDOWS,
    load_fast_context,
    run_fast_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(FAMILY_ADAPTERS),
        help="daily primitive family（本 runner 只允许已落盘 primitive）",
    )
    parser.add_argument(
        "--factors",
        default="all",
        help="逗号分隔或 all（默认）",
    )
    parser.add_argument(
        "--window",
        default="discovery",
        choices=list(WINDOWS),
        help="discovery=2023-2024 冻结窗（默认）；full=2019-2026",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    adapter = FAMILY_ADAPTERS[args.family]
    if args.factors.strip().lower() in {"", "all"}:
        names = list(adapter.factor_names)
    else:
        names = [
            item.strip() for item in args.factors.split(",") if item.strip()
        ]

    out_root = (
        Path(args.output_root)
        if args.output_root
        else FAST_DISCOVERY_DIR / args.window
    )
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    context = load_fast_context(args.window)
    summary, profile = run_fast_batch(
        args.family,
        names,
        window=args.window,
        output_root=out_root,
        context=context,
    )
    wall = time.perf_counter() - t0

    summary_path = out_root / "fast_summary.csv"
    profile_path = out_root / "fast_profile.csv"
    if summary_path.exists():
        summary = pd.concat(
            [pd.read_csv(summary_path), summary], ignore_index=True
        ).drop_duplicates(
            subset=["factor", "family", "window"], keep="last"
        )
    if profile_path.exists():
        profile = pd.concat(
            [pd.read_csv(profile_path), profile], ignore_index=True
        ).drop_duplicates(
            subset=["factor", "family", "window"], keep="last"
        )
    summary.to_csv(summary_path, index=False)
    profile.to_csv(profile_path, index=False)
    print(
        f"[done] {len(names)} factors in {wall:.1f}s "
        f"({wall / max(len(names), 1):.1f}s/factor) -> {out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
