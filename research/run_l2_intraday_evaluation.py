#!/usr/bin/env python3
"""Sprint 4.4 Phase 2 — evaluate SSL2 L2 factors via existing DDB evaluation.

Does not modify research/run_intraday_evaluation_v2.py. Reuses its exact
group/market aggregation and core/evaluation/intraday_metrics helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from core.evaluation.intraday_metrics import (  # noqa: E402
    build_group_excess_panel,
    build_hl_panel,
    summarize_cross_sectional_metrics,
    summarize_ic_series,
)
from research.l2_alpha.l2_factor_panel import to_evaluation_signal  # noqa: E402
from research.l2_alpha.l2_factor_registry import (  # noqa: E402
    DEFAULT_BARTIMES,
    DEFAULT_HORIZONS,
    EXISTING_BASELINE_FACTORS,
    L2_PHASE2_FACTORS,
)
from research.l2_alpha.export_l2_intraday_panel import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_PANEL_DIR,
    export_day,
    _csi1000_symbols,
    _limit_symbols_balanced,
)
from research.l2_alpha.clickhouse_ssl2 import connect_hf_client  # noqa: E402
from research.run_intraday_alpha_discovery_v1 import (  # noqa: E402
    _build_factor,
    _rank_zscore,
)
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    _bt_string,
    _connect,
    _evaluate,
)
from research.run_intraday_evaluation_v2 import (  # noqa: E402
    _aggregate_exact_group_and_market,
)

DEFAULT_RESULTS = PROJECT / "research/results/l2_alpha_discovery_v1"


def _load_panel(panel_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    for day in pd.bdate_range(start, end):
        path = panel_dir / f"{day.strftime('%Y%m%d')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _ensure_panel(
    *,
    start: str,
    end: str,
    panel_dir: Path,
    bartimes: List[str],
    limit_symbols: int,
    rebuild: bool,
) -> pd.DataFrame:
    days = list(pd.bdate_range(start, end))
    missing = [
        d
        for d in days
        if rebuild or not (panel_dir / f"{d.strftime('%Y%m%d')}.parquet").exists()
    ]
    if missing:
        symbols = _csi1000_symbols(start, end)
        if limit_symbols > 0:
            symbols = _limit_symbols_balanced(symbols, limit_symbols)
        client = connect_hf_client()
        try:
            for d in missing:
                day_s = d.strftime("%Y-%m-%d")
                path = export_day(
                    day_s,
                    symbols=symbols,
                    output_dir=panel_dir,
                    bartimes=bartimes,
                    client=client,
                )
                print(f"[l2_eval] exported {path}", flush=True)
        finally:
            client.close()
    return _load_panel(panel_dir, start, end)


def _ic_lookup(ic_ts: pd.DataFrame, bartime: str, horizon: str, direction: int) -> dict:
    frame = ic_ts.copy()
    frame["bartime_key"] = _bt_string(frame["Bartime"])
    ret_key = "RetType" if "RetType" in frame.columns else "valueType"
    selected = frame[
        (frame["bartime_key"] == bartime)
        & (frame[ret_key].astype(str) == horizon)
    ]
    return summarize_ic_series(selected["Rank_IC"], direction=direction)


def _decile_monotonicity(group_means: pd.Series) -> float:
    """Spearman of decile index vs mean excess return after direction."""
    if len(group_means) < 3:
        return float("nan")
    x = np.arange(1, len(group_means) + 1, dtype=float)
    y = group_means.to_numpy(dtype=float)
    if np.nanstd(y) == 0:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def evaluate_l2_factor(
    session,
    signal: pd.DataFrame,
    *,
    factor_name: str,
    horizons: List[str],
) -> pd.DataFrame:
    if signal.empty:
        return pd.DataFrame()
    filtered, _, _, ic_ts = _evaluate(
        session,
        f"l2_{factor_name}",
        signal,
        apply_limit_filter=True,
    )
    groups, market = _aggregate_exact_group_and_market(
        session,
        f"l2_{factor_name}",
        filtered,
        horizons,
    )
    panel = build_group_excess_panel(groups, market)
    rows = []
    for (bartime, horizon), combo in panel.groupby(
        ["Bartime", "return_window"], sort=True
    ):
        provisional = _ic_lookup(ic_ts, str(bartime), str(horizon), 1)
        direction = 1 if provisional["rank_ic"] >= 0 else -1
        ic = _ic_lookup(ic_ts, str(bartime), str(horizon), direction)
        hl = build_hl_panel(combo, direction=direction)
        metrics = summarize_cross_sectional_metrics(
            combo, hl, factor_name=factor_name
        )
        metrics["direction"] = direction
        metrics["rank_ic"] = ic["rank_ic"]
        metrics["annualized_icir"] = ic["annualized_icir"]
        metrics["ic_win_rate"] = ic["ic_win_rate"]

        # Monotonicity on direction-adjusted group excess means G1..G10
        gmeans = (
            combo.groupby("group")["group_return_excess"]
            .mean()
            .reindex([f"G{i}" for i in range(1, 11)])
        )
        signed_means = direction * gmeans
        mono = _decile_monotonicity(signed_means.dropna())
        metrics["decile_mono_spearman"] = mono
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _daily_ic_series(
    session,
    signal: pd.DataFrame,
    *,
    factor_name: str,
    bartime: str,
    horizon: str,
) -> pd.Series:
    """Extract daily Rank_IC series for one bartime/horizon."""
    _, _, _, ic_ts = _evaluate(
        session, f"ic_{factor_name}", signal, apply_limit_filter=True
    )
    frame = ic_ts.copy()
    frame["bartime_key"] = _bt_string(frame["Bartime"])
    ret_key = "RetType" if "RetType" in frame.columns else "valueType"
    selected = frame[
        (frame["bartime_key"] == bartime)
        & (frame[ret_key].astype(str) == horizon)
    ].copy()
    if selected.empty:
        return pd.Series(dtype=float)
    selected["Date"] = pd.to_datetime(selected["Date"])
    return selected.set_index("Date")["Rank_IC"].astype(float).sort_index()


def _residualize_same_slot(
    target: pd.DataFrame,
    controls: Dict[str, pd.DataFrame],
    bartime: str,
) -> pd.DataFrame:
    wide = target[["tradetime", "symbol", "value"]].rename(
        columns={"value": "target"}
    )
    names = []
    for name, control in controls.items():
        mask = control["tradetime"].dt.strftime("%H:%M") == bartime
        if not mask.any():
            continue
        names.append(name)
        wide = wide.merge(
            control.loc[mask, ["tradetime", "symbol", "value"]].rename(
                columns={"value": name}
            ),
            on=["tradetime", "symbol"],
            how="inner",
        )
    if not names:
        return pd.DataFrame(columns=["tradetime", "symbol", "factorname", "value"])

    def _daily(group: pd.DataFrame) -> pd.DataFrame:
        cols = ["target", *names]
        ranked = group[cols].apply(_rank_zscore).dropna()
        if len(ranked) < max(50, len(names) + 5):
            return pd.DataFrame(columns=["tradetime", "symbol", "value"])
        y = ranked["target"].to_numpy(float)
        x = ranked[names].to_numpy(float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        out = group.loc[ranked.index, ["tradetime", "symbol"]].copy()
        out["value"] = y - design @ beta
        return out

    resid = (
        wide.groupby("tradetime", group_keys=False).apply(_daily).reset_index(drop=True)
    )
    resid["factorname"] = "target_resid"
    return resid[["tradetime", "symbol", "factorname", "value"]]


def run_independence(
    session,
    panel: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
    anchor_bartime: str = "14:29",
    anchor_horizon: str = "Ret_30",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IC correlation + residual IC vs RV/CVWAP/Amihud at one anchor slot."""
    # Build baseline signals (full period) then filter slot.
    baselines = {}
    for name in EXISTING_BASELINE_FACTORS:
        sig = _build_factor(name, period_start, period_end)
        sig = sig[sig["tradetime"].dt.strftime("%H:%M") == anchor_bartime].copy()
        baselines[name] = sig

    ic_map = {}
    for name, sig in baselines.items():
        ic_map[name] = _daily_ic_series(
            session,
            sig,
            factor_name=name,
            bartime=anchor_bartime,
            horizon=anchor_horizon,
        )

    residual_rows = []
    for factor_name in L2_PHASE2_FACTORS:
        signal = to_evaluation_signal(panel, factor_name)
        signal = signal[
            signal["tradetime"].dt.strftime("%H:%M") == anchor_bartime
        ].copy()
        if signal.empty:
            continue
        ic_map[factor_name] = _daily_ic_series(
            session,
            signal,
            factor_name=factor_name,
            bartime=anchor_bartime,
            horizon=anchor_horizon,
        )
        resid_signal = _residualize_same_slot(signal, baselines, anchor_bartime)
        if resid_signal.empty:
            continue
        resid_ic = _daily_ic_series(
            session,
            resid_signal,
            factor_name=f"{factor_name}_resid",
            bartime=anchor_bartime,
            horizon=anchor_horizon,
        )
        raw = summarize_ic_series(ic_map[factor_name], direction=1)
        direction = 1 if raw["rank_ic"] >= 0 else -1
        resid_summary = summarize_ic_series(resid_ic, direction=direction)
        residual_rows.append(
            {
                "factor": factor_name,
                "bartime": anchor_bartime,
                "horizon": anchor_horizon,
                "raw_rank_ic": raw["rank_ic"],
                "raw_icir": direction
                * summarize_ic_series(ic_map[factor_name], direction=direction)[
                    "annualized_icir"
                ],
                "residual_rank_ic": resid_summary["rank_ic"],
                "residual_signed_icir": resid_summary["annualized_icir"],
                "residual_ic_win_rate": resid_summary["ic_win_rate"],
                "n_dates": resid_summary["n_dates"],
            }
        )

    ic_df = pd.DataFrame(ic_map)
    corr = ic_df.corr(method="spearman")
    return corr, pd.DataFrame(residual_rows)


def _hl_summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    hl = metrics[metrics["metric_scope"] == "cross_sectional_hl"].copy()
    if hl.empty:
        return hl
    keep = [
        "factor",
        "bartime",
        "return_window",
        "direction",
        "rank_ic",
        "annualized_icir",
        "ic_win_rate",
        "hl_sharpe",
        "hl_annualized_return",
        "decile_mono_spearman",
        "hl_market_beta",
    ]
    cols = [c for c in keep if c in hl.columns]
    return hl[cols].sort_values(
        ["factor", "bartime", "return_window"]
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-06-03")
    parser.add_argument("--end", default="2024-06-05")
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--bartimes", default=",".join(DEFAULT_BARTIMES)
    )
    parser.add_argument(
        "--horizons", default=",".join(DEFAULT_HORIZONS)
    )
    parser.add_argument("--limit-symbols", type=int, default=0)
    parser.add_argument("--rebuild-panel", action="store_true")
    parser.add_argument(
        "--skip-independence",
        action="store_true",
        help="Skip residual/correlation vs RV/CVWAP/Amihud",
    )
    args = parser.parse_args(argv)

    bartimes = [b.strip() for b in args.bartimes.split(",") if b.strip()]
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    print(
        f"[l2_eval] panel {args.start}→{args.end} bartimes={bartimes} "
        f"horizons={horizons}",
        flush=True,
    )
    panel = _ensure_panel(
        start=args.start,
        end=args.end,
        panel_dir=args.panel_dir,
        bartimes=bartimes,
        limit_symbols=args.limit_symbols,
        rebuild=args.rebuild_panel,
    )
    if panel.empty:
        print("[l2_eval] empty panel — abort", flush=True)
        return 2
    print(
        f"[l2_eval] panel rows={len(panel)} factors={sorted(panel['factor'].unique())}",
        flush=True,
    )

    session = _connect()
    all_metrics = []
    for factor_name in L2_PHASE2_FACTORS:
        signal = to_evaluation_signal(panel, factor_name)
        signal = signal[
            signal["tradetime"].dt.strftime("%H:%M").isin(bartimes)
        ]
        print(
            f"[l2_eval] evaluate {factor_name} rows={len(signal)}",
            flush=True,
        )
        if signal.empty:
            continue
        metrics = evaluate_l2_factor(
            session, signal, factor_name=factor_name, horizons=horizons
        )
        if not metrics.empty:
            all_metrics.append(metrics)

    if not all_metrics:
        print("[l2_eval] no metrics produced", flush=True)
        return 3

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.output / "l2_metrics_long.csv", index=False)
    summary = _hl_summary_table(metrics)
    summary.to_csv(args.output / "l2_alpha_discovery_v1.csv", index=False)
    print(f"[l2_eval] wrote {args.output / 'l2_alpha_discovery_v1.csv'}", flush=True)

    if not args.skip_independence:
        print("[l2_eval] independence analysis …", flush=True)
        corr, residual = run_independence(
            session,
            panel,
            period_start=args.start,
            period_end=args.end,
        )
        corr.to_csv(args.output / "l2_ic_correlation.csv")
        residual.to_csv(args.output / "l2_residual_ic.csv", index=False)
        print(
            f"[l2_eval] wrote correlation/residual tables "
            f"residual_rows={len(residual)}",
            flush=True,
        )

    meta = {
        "start": args.start,
        "end": args.end,
        "bartimes": bartimes,
        "horizons": horizons,
        "factors": list(L2_PHASE2_FACTORS),
        "note": (
            "Ret_5 unavailable in PREHEAT_RET_MATRIX; using Ret_15/30/60. "
            "Bartimes must be PREHEAT :29/:59 slots (e.g. 09:59, 14:29)."
        ),
    }
    (args.output / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
