#!/usr/bin/env python3
"""L2 Feature Factory v1 — POC discovery screen (classic IC / H-L / mono).

Defaults: 200 CSI1000 names, 2024H1 train screen. Residual is last and only
for survivors. No entropy / ML.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from research.l2_alpha.export_l2_intraday_panel import (  # noqa: E402
    _csi1000_symbols,
    _limit_symbols_balanced,
)
from research.l2_alpha.feature_factory.export import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_PANEL,
    export_day,
)
from research.l2_alpha.feature_factory.registry import (  # noqa: E402
    L2_FF_ALL_COLUMNS,
)
from research.l2_alpha.l2_factor_panel import to_evaluation_signal  # noqa: E402
from research.l2_alpha.l2_factor_registry import (  # noqa: E402
    DEFAULT_BARTIMES,
    DEFAULT_HORIZONS,
)
from research.run_l2_alpha_validation_v21 import (  # noqa: E402
    GATE_HL_SHARPE_TRAIN,
    GATE_ICIR,
    GATE_MONO,
    _decile_gate_pass,
    _evaluate_rich,
    _hl_flat,
    _connect,
)

DEFAULT_OUTPUT = PROJECT / "research/results/l2_feature_factory_v1"


def _ensure_panel(
    *,
    start: str,
    end: str,
    panel_dir: Path,
    bartimes: List[str],
    symbols: List[str],
    rebuild: bool,
) -> pd.DataFrame:
    days = list(pd.bdate_range(start, end))
    missing = [
        d
        for d in days
        if rebuild or not (panel_dir / f"{d.strftime('%Y%m%d')}.parquet").exists()
    ]
    if missing:
        client = connect_hf_client()
        try:
            for i, d in enumerate(missing, 1):
                day_s = d.strftime("%Y-%m-%d")
                path = export_day(
                    day_s,
                    symbols=symbols,
                    output_dir=panel_dir,
                    bartimes=bartimes,
                    client=client,
                )
                print(
                    f"[ff] panel {i}/{len(missing)} {day_s} → {path}",
                    flush=True,
                )
        finally:
            client.close()
    frames = []
    for d in days:
        path = panel_dir / f"{d.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(out["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates
    return out


def _screen_decisions(train_hl: pd.DataFrame, deciles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in train_hl.iterrows():
        factor = row["factor"]
        bt = str(row["bartime"])
        hz = str(row["return_window"])
        spearman = float(row.get("decile_mono_spearman", float("nan")))
        sub = deciles[
            (deciles["factor"] == factor)
            & (deciles["bartime"].astype(str) == bt)
            & (deciles["horizon"].astype(str) == hz)
        ]
        signed = (
            sub.set_index("group")["signed_mean_excess_return"]
            if not sub.empty
            else None
        )
        mono_ok = _decile_gate_pass(signed, spearman=spearman)
        stand_ok = (
            abs(float(row["annualized_icir"])) > GATE_ICIR
            and float(row["hl_sharpe"]) > GATE_HL_SHARPE_TRAIN
            and mono_ok
        )
        rows.append(
            {
                "factor": factor,
                "bartime": bt,
                "horizon": hz,
                "direction": int(row["direction"]),
                "rank_ic": float(row["rank_ic"]),
                "icir": float(row["annualized_icir"]),
                "hl_sharpe": float(row["hl_sharpe"]),
                "mono_spearman": spearman,
                "mono_pass": mono_ok,
                "standalone_pass": stand_ok,
                "decision": "KEEP" if stand_ok else "DROP",
            }
        )
    return pd.DataFrame(rows)


def _write_report(path: Path, decisions: pd.DataFrame, meta: dict) -> None:
    keep = decisions[decisions["decision"] == "KEEP"] if not decisions.empty else decisions
    lines = [
        "# L2 Feature Factory v1 — POC Discovery Report",
        "",
        "## Scope",
        "",
        "```json",
        json.dumps(meta, indent=2),
        "```",
        "",
        "## Gates (train screen)",
        "",
        f"- |ICIR| > {GATE_ICIR}",
        f"- H-L Sharpe > {GATE_HL_SHARPE_TRAIN}",
        f"- Decile mono (G1→G10, Spearman ≥ {GATE_MONO})",
        "",
        "## KEEP candidates",
        "",
    ]
    if keep is None or keep.empty:
        lines.append("_None cleared the train screen._")
    else:
        try:
            lines.append(keep.to_markdown(index=False))
        except Exception:  # noqa: BLE001
            lines.append("```\n" + keep.to_string(index=False) + "\n```")
    lines += [
        "",
        "## Full decisions",
        "",
    ]
    if decisions.empty:
        lines.append("_Empty._")
    else:
        try:
            lines.append(decisions.to_markdown(index=False))
        except Exception:  # noqa: BLE001
            lines.append("```\n" + decisions.to_string(index=False) + "\n```")
    lines += [
        "",
        "## Next",
        "",
        "OOS + residual only for KEEP factors (Phase 3). Do not expand ML/entropy yet.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bartimes", default=",".join(DEFAULT_BARTIMES))
    parser.add_argument("--horizons", default=",".join(DEFAULT_HORIZONS))
    parser.add_argument("--limit-symbols", type=int, default=200)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full CSI1000 (sets --limit-symbols 0). Follow-on to POC, not default.",
    )
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument(
        "--max-factors",
        type=int,
        default=0,
        help="Optional cap for debug (0 = all 20)",
    )
    args = parser.parse_args(argv)

    bartimes = [b.strip() for b in args.bartimes.split(",") if b.strip()]
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    args.panel_dir.mkdir(parents=True, exist_ok=True)

    if args.full:
        args.limit_symbols = 0
    symbols = _csi1000_symbols(args.start, args.end)
    if args.limit_symbols > 0:
        symbols = _limit_symbols_balanced(symbols, args.limit_symbols)
    print(
        f"[ff] discovery {args.start}→{args.end} symbols={len(symbols)} "
        f"bartimes={bartimes} horizons={horizons}",
        flush=True,
    )

    panel = _ensure_panel(
        start=args.start,
        end=args.end,
        panel_dir=args.panel_dir,
        bartimes=bartimes,
        symbols=symbols,
        rebuild=args.rebuild_panel,
    )
    if panel.empty:
        print("[ff] empty panel", flush=True)
        return 2
    factors = list(L2_FF_ALL_COLUMNS)
    if args.max_factors > 0:
        factors = factors[: args.max_factors]
    present = sorted(panel["factor"].unique())
    print(
        f"[ff] panel rows={len(panel)} factors_present={len(present)}",
        flush=True,
    )

    session = _connect()
    all_metrics = []
    all_deciles = []
    for factor_name in factors:
        signal = to_evaluation_signal(panel, factor_name)
        signal = signal[
            signal["tradetime"].dt.strftime("%H:%M").isin(bartimes)
        ]
        if signal.empty:
            print(f"[ff] skip empty {factor_name}", flush=True)
            continue
        print(
            f"[ff] evaluate {factor_name} rows={len(signal)}",
            flush=True,
        )
        metrics, _, deciles = _evaluate_rich(
            session,
            signal,
            factor_name=factor_name,
            period_name="train_2024H1",
            horizons=horizons,
            frozen_direction=None,
        )
        if not metrics.empty:
            all_metrics.append(metrics)
        if not deciles.empty:
            all_deciles.append(deciles)

    if not all_metrics:
        print("[ff] no metrics", flush=True)
        return 3

    metrics_long = pd.concat(all_metrics, ignore_index=True)
    metrics_long.to_csv(args.output / "l2_ff_metrics_long.csv", index=False)
    train_hl = _hl_flat(metrics_long)
    train_hl.to_csv(args.output / "l2_ff_metrics.csv", index=False)

    deciles = (
        pd.concat(all_deciles, ignore_index=True)
        if all_deciles
        else pd.DataFrame()
    )
    if not deciles.empty:
        deciles.to_csv(args.output / "l2_ff_decile_returns.csv", index=False)

    # Best tuple per factor by |ICIR|
    best_rows = []
    for factor, sub in train_hl.groupby("factor"):
        scored = sub.copy()
        scored["abs_icir"] = scored["annualized_icir"].abs()
        best_rows.append(
            scored.sort_values(["abs_icir", "hl_sharpe"], ascending=False).iloc[0]
        )
    best = pd.DataFrame(best_rows)
    decisions = _screen_decisions(best, deciles)
    decisions.to_csv(args.output / "l2_ff_decisions.csv", index=False)

    meta = {
        "phase": "2.2_feature_factory_v1",
        "start": args.start,
        "end": args.end,
        "n_symbols": len(symbols),
        "bartimes": bartimes,
        "horizons": horizons,
        "factors": factors,
        "gates": {
            "icir": GATE_ICIR,
            "hl_sharpe_train": GATE_HL_SHARPE_TRAIN,
            "mono": GATE_MONO,
        },
        "n_keep": int((decisions["decision"] == "KEEP").sum())
        if not decisions.empty
        else 0,
    }
    (args.output / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(args.output / "l2_ff_report.md", decisions, meta)
    print(decisions.to_string(index=False), flush=True)
    print(f"[ff] done → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
