#!/usr/bin/env python
"""Full Fast Discovery for the 3 post-novelty LR survivors.

Reuses the frozen Fast Discovery engine (T+1, excess c2c vs 000852.SH,
backtest_factor, Fast Gate labels). Does not mutate candidate_pool_v1,
FAMILY_ADAPTERS, BDL thresholds, or FS/ML.

Window: Fast Discovery ``discovery`` = 2023-01-01 → 2024-12-31, every
trading date (not the Lite every-5th calendar). Long-sample 2019–2026
is not started here.

Usage:
  /opt/conda/anaconda3/bin/python -m l2_factor_reproduction.scripts.run_lr_fast_discovery
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from l2_factor_reproduction.discovery_lite.candidate_matrix import (  # noqa: E402
    load_trading_calendar,
)
from l2_factor_reproduction.liquidity_resilience.contracts import (  # noqa: E402
    FAMILY_NAME,
    LR_RESULT_ROOT,
)
from l2_factor_reproduction.liquidity_resilience.full_novelty import (  # noqa: E402
    POST_NOVELTY_SURVIVORS,
)
from l2_factor_reproduction.liquidity_resilience.materialize import (  # noqa: E402
    daily_to_narrow,
    materialize_trading_dates,
)
from l2_factor_reproduction.python.backtest import backtest_factor  # noqa: E402
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    DISCOVERY_END,
    DISCOVERY_START,
    compute_fast_metrics,
    gate_label,
    load_fast_context,
    save_fast_plots,
)
from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402

OUT_DIR = LR_RESULT_ROOT / "fast_discovery" / "discovery"
MAT_DIR = LR_RESULT_ROOT / "fast_discovery" / "discovery_materialization"


def _write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# LR Full Fast Discovery — 3 post-novelty survivors",
        "",
        "Status remains `DISCOVERY_CANDIDATE`. Not frozen into `candidate_pool_v1`.",
        "FS/ML was not rerun. Fast Gate labels are tags, not KEEP/DROP.",
        "",
        "## Window",
        "",
        f"- Fast Discovery `discovery`: `{DISCOVERY_START.date()}` → `{DISCOVERY_END.date()}`",
        "- Target: excess c2c vs `000852.SH`, `prepare_factor_signal(signal_shift=1)`",
        "- Long-sample 2019–2026: **not run**",
        "",
        "## Incremental-information question",
        "",
        "Does the factor contain information not already represented by Order Book state,",
        "Trade Flow, Liquidity Impact, Price Formation, or existing resilience-like proxies?",
        "Stage B full-universe novelty said all three are `PASS_INDEPENDENT` vs 138 existing",
        "factors. This run asks whether that independence survives the full 2023–2024",
        "Fast Discovery engine (RankIC, ICIR, H-L Sharpe, decile, turnover).",
        "",
        "## Results",
        "",
        summary.to_string(index=False),
        "",
        "## Next (not started)",
        "",
        "- long-sample / robustness 2019–2026",
        "- family-level redundancy vs frozen pool on the full discovery calendar",
        "- research factor freeze / ML feature universe",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args()

    names = list(POST_NOVELTY_SURVIVORS)
    cal = load_trading_calendar("discovery")
    dates = cal[(cal >= DISCOVERY_START) & (cal <= DISCOVERY_END)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[lr-fd] Full Fast Discovery on {len(names)} post-novelty survivors "
        f"n_dates={len(dates)} window={DISCOVERY_START.date()}→{DISCOVERY_END.date()}",
        flush=True,
    )
    print("[lr-fd] not mutating candidate_pool_v1 / FAMILY_ADAPTERS / FS / ML", flush=True)

    if args.skip_materialize:
        panel_path = MAT_DIR / "panel.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(panel_path)
        panel = pd.read_parquet(panel_path)
        print(f"[lr-fd] reused {panel_path} rows={len(panel)}", flush=True)
    else:
        client = connect_hf_client()
        panel = materialize_trading_dates(
            dates, client=client, out_dir=MAT_DIR, keep_names=names
        )

    t0 = time.perf_counter()
    mask, ret = load_fast_context("discovery")
    context_s = time.perf_counter() - t0
    summary_rows = []
    profile_rows = []
    for name in names:
        path = MAT_DIR / "factors" / name / "factor_narrow.parquet"
        if not path.exists():
            if name not in panel.columns:
                raise FileNotFoundError(path)
            dest = path.parent
            dest.mkdir(parents=True, exist_ok=True)
            daily_to_narrow(panel, name).to_parquet(path, index=False)
        narrow = pd.read_parquet(path)
        t_bt = time.perf_counter()
        group_pnl, group_to, _rank_ic, summary = backtest_factor(
            narrow,
            start_day=DISCOVERY_START,
            end_day=DISCOVERY_END,
            mask=mask,
            ret_matrix=ret,
        )
        bt_s = time.perf_counter() - t_bt
        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        metrics["gate"] = gate_label(metrics)
        t_plot = time.perf_counter()
        save_fast_plots(OUT_DIR / "figures" / name, name, group_pnl, metrics)
        plot_s = time.perf_counter() - t_plot
        summary_rows.append(
            {
                "factor": name,
                "family": FAMILY_NAME,
                "window": "discovery",
                "discovery_status": "DISCOVERY_CANDIDATE",
                **metrics,
            }
        )
        profile_rows.append(
            {
                "factor": name,
                "family": FAMILY_NAME,
                "window": "discovery",
                "context_load_seconds": round(context_s, 3),
                "backtest_seconds": round(bt_s, 3),
                "plot_seconds": round(plot_s, 3),
            }
        )
        print(
            f"[lr-fd] {name}: sharpe={metrics['hl_sharpe']:.2f} "
            f"rank_ic={metrics['rank_ic_mean_raw']:.4f} "
            f"icir={metrics['icir_raw']:.2f} "
            f"mono={metrics['decile_mono_spearman']:.2f} "
            f"to={metrics['avg_hl_turnover']:.2f} "
            f"gate={metrics['gate']}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    profile = pd.DataFrame(profile_rows)
    summary.to_csv(OUT_DIR / "fast_summary.csv", index=False)
    profile.to_csv(OUT_DIR / "fast_profile.csv", index=False)
    _write_report(summary, OUT_DIR)
    pointer = {
        "window": "discovery",
        "start": str(DISCOVERY_START.date()),
        "end": str(DISCOVERY_END.date()),
        "survivors": names,
        "output": str(OUT_DIR),
        "materialization": str(MAT_DIR),
        "discovery_status": "DISCOVERY_CANDIDATE",
        "candidate_pool_v1_mutated": False,
        "fs_ml_rerun": False,
        "long_sample_2019_2026": False,
    }
    (OUT_DIR / "run_pointer.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[lr-fd] wrote {OUT_DIR}. STOP. Do not freeze into the pool or rerun FS/ML.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
