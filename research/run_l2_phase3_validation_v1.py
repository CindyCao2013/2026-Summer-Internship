#!/usr/bin/env python3
"""Phase 3 — frozen L2 Feature Factory OOS + residual + cost validation.

No re-screening. No new features. Inputs: Phase 2.3 research candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from research.intraday_portfolio_simulator_v1 import (  # noqa: E402
    _fetch_extreme_constituents,
    _performance_summary,
    _simulate_ledger,
)
from research.l2_alpha.l2_factor_panel import to_evaluation_signal  # noqa: E402
from research.l2_alpha.l2_factor_registry import EXISTING_BASELINE_FACTORS  # noqa: E402
from research.run_intraday_alpha_discovery_v1 import _build_factor  # noqa: E402
from research.run_intraday_alpha_library_v1 import _connect, _evaluate  # noqa: E402
from research.run_l2_alpha_validation_v21 import (  # noqa: E402
    GATE_HL_SHARPE_OOS,
    GATE_HL_SHARPE_TRAIN,
    GATE_ICIR,
    GATE_MONO,
    GATE_RESID_ICIR,
    PERIODS,
    _decile_gate_pass,
    _evaluate_rich,
    _filter_period,
    _hl_flat,
    _ic_series_table,
    _residual_bundle,
)

DEFAULT_PANEL = PROJECT / "research/results/l2_feature_factory_v1/panel"
DEFAULT_OUTPUT = PROJECT / "research/results/l2_phase3_validation"
DEFAULT_CANDIDATES = (
    PROJECT / "research/results/l2_feature_factory_v1/l2_phase3_candidates.csv"
)

# Freeze = train-best tuples (not forced 14:29 for all).
FROZEN_CANDIDATES: List[dict] = [
    {
        "factor": "woi_mean10",
        "bartime": "14:29",
        "horizon": "Ret_30",
        "direction": -1,
        "tier": 1,
        "role": "production_candidate",
    },
    {
        "factor": "depth_imb_mean10",
        "bartime": "14:29",
        "horizon": "Ret_30",
        "direction": -1,
        "tier": 2,
        "role": "research_candidate",
    },
    {
        "factor": "woi_delta30",
        "bartime": "13:59",
        "horizon": "Ret_15",
        "direction": 1,
        "tier": 2,
        "role": "research_candidate",
    },
    {
        "factor": "woi_std20",
        "bartime": "10:59",
        "horizon": "Ret_30",
        "direction": -1,
        "tier": 2,
        "role": "research_candidate",
    },
]

COST_LADDER_BPS = (10.0, 15.0, 20.0)
FULL_START = "2024-01-01"
FULL_END = "2025-08-18"


def _slot_signal(panel: pd.DataFrame, factor: str, bartime: str) -> pd.DataFrame:
    signal = to_evaluation_signal(panel, factor)
    if signal.empty:
        return signal
    return signal[signal["tradetime"].dt.strftime("%H:%M") == bartime].copy()


def _signed_means_from_deciles(
    deciles: pd.DataFrame,
    *,
    factor: str,
    period: str,
    bartime: str,
    horizon: str,
) -> Optional[pd.Series]:
    if deciles.empty:
        return None
    sub = deciles[
        (deciles["factor"] == factor)
        & (deciles["period"] == period)
        & (deciles["bartime"].astype(str) == bartime)
        & (deciles["horizon"].astype(str) == horizon)
    ]
    if sub.empty:
        return None
    return sub.set_index("group")["signed_mean_excess_return"]


def _simulate_cost_ladder(
    session,
    signal: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    bartime: str,
    horizon: str,
    direction: int,
    cost_bps_list: Tuple[float, ...],
) -> List[dict]:
    slot = signal[signal["tradetime"].dt.strftime("%H:%M") == bartime].copy()
    if slot.empty:
        return []
    filtered, _, _, _ = _evaluate(
        session,
        f"p3sim_{factor_name}_{period_name}",
        slot,
        apply_limit_filter=True,
    )
    constituents = _fetch_extreme_constituents(
        session, f"p3_{factor_name}_{period_name}", filtered, horizon
    )
    rows = []
    for bps in cost_bps_list:
        ledger = _simulate_ledger(
            constituents,
            factor_name=factor_name,
            period_name=period_name,
            horizon=horizon,
            direction=direction,
            one_way_cost_bps=float(bps),
        )
        summary = _performance_summary(
            ledger,
            freeze_sha256="l2_phase3",
            simulation_sha256=f"l2_phase3_cost_{bps}",
            max_gross_parity_diff=np.nan,
        )
        summary["period"] = period_name
        summary["bartime"] = bartime
        summary["horizon"] = horizon
        summary["direction"] = direction
        summary["factor"] = factor_name
        summary["one_way_cost_bps"] = float(bps)
        rows.append(summary)
    return rows


def _survival_decision(
    *,
    cand: dict,
    train: Optional[pd.Series],
    validation: Optional[pd.Series],
    oos: Optional[pd.Series],
    resid_train: Optional[dict],
    resid_oos: Optional[dict],
    train_means: Optional[pd.Series],
    oos_means: Optional[pd.Series],
    cost_oos: Optional[pd.DataFrame],
) -> dict:
    """KEEP / WATCH / DROP under Phase-3 production gates."""
    if train is None:
        return {
            "factor": cand["factor"],
            "tier": cand["tier"],
            "role": cand["role"],
            "bartime": cand["bartime"],
            "horizon": cand["horizon"],
            "direction": cand["direction"],
            "decision": "DROP",
            "reason": "missing_train_metrics",
        }

    train_spearman = float(train.get("decile_mono_spearman", np.nan))
    train_mono = _decile_gate_pass(train_means, spearman=train_spearman)
    train_ok = (
        abs(float(train["annualized_icir"])) > GATE_ICIR
        and float(train["hl_sharpe"]) > GATE_HL_SHARPE_TRAIN
        and train_mono
    )

    oos_spearman = (
        float(oos.get("decile_mono_spearman", np.nan)) if oos is not None else np.nan
    )
    oos_mono = (
        _decile_gate_pass(oos_means, spearman=oos_spearman)
        if oos is not None
        else False
    )
    oos_ok = (
        oos is not None
        and float(oos["hl_sharpe"]) > GATE_HL_SHARPE_OOS
        and oos_mono
    )

    resid_icir = (
        float(resid_train["residual_icir"]) if resid_train is not None else np.nan
    )
    indep_ok = abs(resid_icir) > GATE_RESID_ICIR if np.isfinite(resid_icir) else False

    # Cost: diagnostic; positive net Sharpe at 10bp is soft evidence.
    cost_ok_10 = False
    be_bps = np.nan
    if cost_oos is not None and not cost_oos.empty:
        row10 = cost_oos[cost_oos["one_way_cost_bps"] == 10.0]
        if not row10.empty:
            cost_ok_10 = float(row10.iloc[0]["net_ls_sharpe"]) > 0
            be_bps = float(row10.iloc[0].get("break_even_one_way_cost_bps", np.nan))

    if train_ok and oos_ok and indep_ok:
        decision = "KEEP"
        reason = "train+oos+residual_pass"
    elif train_ok and indep_ok and not oos_ok:
        decision = "WATCH"
        reason = "train_and_residual_ok_oos_fail"
    elif (train_ok or oos_ok) and not indep_ok:
        decision = "WATCH"
        reason = "signal_but_not_independent"
    elif oos_ok or (validation is not None and float(validation["hl_sharpe"]) > 1.0):
        decision = "WATCH"
        reason = "partial_oos_evidence"
    else:
        decision = "DROP"
        reason = "failed_core_gates"

    def _m(row: Optional[pd.Series], key: str) -> float:
        if row is None:
            return np.nan
        return float(row.get(key, np.nan))

    return {
        "factor": cand["factor"],
        "tier": cand["tier"],
        "role": cand["role"],
        "bartime": cand["bartime"],
        "horizon": cand["horizon"],
        "direction": int(cand["direction"]),
        "train_rank_ic": _m(train, "rank_ic"),
        "train_icir": _m(train, "annualized_icir"),
        "train_hl_sharpe": _m(train, "hl_sharpe"),
        "train_mono": train_spearman,
        "train_mono_pass": train_mono,
        "train_pass": train_ok,
        "val_hl_sharpe": _m(validation, "hl_sharpe"),
        "val_icir": _m(validation, "annualized_icir"),
        "oos_rank_ic": _m(oos, "rank_ic"),
        "oos_icir": _m(oos, "annualized_icir"),
        "oos_hl_sharpe": _m(oos, "hl_sharpe"),
        "oos_mono": oos_spearman,
        "oos_mono_pass": oos_mono,
        "oos_pass": oos_ok,
        "residual_train_icir": resid_icir,
        "residual_train_hl": (
            float(resid_train["residual_hl_sharpe"]) if resid_train else np.nan
        ),
        "residual_oos_icir": (
            float(resid_oos["residual_icir"]) if resid_oos else np.nan
        ),
        "residual_oos_hl": (
            float(resid_oos["residual_hl_sharpe"]) if resid_oos else np.nan
        ),
        "independence_pass": indep_ok,
        "cost_net_sharpe_10bp": (
            float(cost_oos.loc[cost_oos["one_way_cost_bps"] == 10.0, "net_ls_sharpe"].iloc[0])
            if cost_oos is not None
            and not cost_oos.empty
            and (cost_oos["one_way_cost_bps"] == 10.0).any()
            else np.nan
        ),
        "break_even_one_way_bps": be_bps,
        "cost_ok_10bp_net_positive": cost_ok_10,
        "decision": decision,
        "reason": reason,
    }


def _write_report(
    path: Path,
    *,
    survival: pd.DataFrame,
    oos_metrics: pd.DataFrame,
    residual: pd.DataFrame,
    cost: pd.DataFrame,
    meta: dict,
) -> None:
    lines = [
        "# Phase 3 — L2 Feature Factory Validation",
        "",
        "## Research question",
        "",
        "Is WOI imbalance reversal (and near-miss research candidates) a",
        "2025-still-working, cost-feasible, residual-independent alpha?",
        "",
        "## Freeze (no re-optimization)",
        "",
        "```json",
        json.dumps(FROZEN_CANDIDATES, indent=2),
        "```",
        "",
        "## Gates",
        "",
        f"- Train: |ICIR|>{GATE_ICIR}, HL>{GATE_HL_SHARPE_TRAIN}, mono gate",
        f"- OOS (2025): HL>{GATE_HL_SHARPE_OOS}, mono gate",
        f"- Residual vs RV/CVWAP/Amihud: |ICIR|>{GATE_RESID_ICIR}",
        "- Cost ladder 10/15/20bp: diagnostic (net Sharpe / break-even)",
        "",
        "## Factor survival",
        "",
    ]
    show = survival[
        [
            c
            for c in [
                "factor",
                "tier",
                "decision",
                "reason",
                "train_hl_sharpe",
                "oos_hl_sharpe",
                "residual_train_icir",
                "cost_net_sharpe_10bp",
                "break_even_one_way_bps",
            ]
            if c in survival.columns
        ]
    ]
    try:
        lines.append(show.to_markdown(index=False))
    except Exception:  # noqa: BLE001
        lines.append("```\n" + show.to_string(index=False) + "\n```")

    lines += ["", "## OOS metrics (all periods)", ""]
    try:
        lines.append(
            oos_metrics[
                [
                    "factor",
                    "period",
                    "bartime",
                    "return_window",
                    "direction",
                    "rank_ic",
                    "annualized_icir",
                    "hl_sharpe",
                    "decile_mono_spearman",
                ]
            ].to_markdown(index=False)
        )
    except Exception:  # noqa: BLE001
        lines.append("```\n" + oos_metrics.to_string(index=False) + "\n```")

    lines += ["", "## Residual", ""]
    if residual.empty:
        lines.append("_empty_")
    else:
        try:
            lines.append(residual.to_markdown(index=False))
        except Exception:  # noqa: BLE001
            lines.append("```\n" + residual.to_string(index=False) + "\n```")

    lines += ["", "## Cost ladder (test_2025)", ""]
    if cost.empty:
        lines.append("_empty_")
    else:
        cost_show = cost[
            cost["period"] == "test_2025_available"
        ] if "period" in cost.columns else cost
        cols = [
            c
            for c in [
                "factor",
                "one_way_cost_bps",
                "gross_ls_sharpe",
                "net_ls_sharpe",
                "break_even_one_way_cost_bps",
                "avg_daily_turnover",
            ]
            if c in cost_show.columns
        ]
        try:
            lines.append(cost_show[cols].to_markdown(index=False))
        except Exception:  # noqa: BLE001
            lines.append("```\n" + cost_show[cols].to_string(index=False) + "\n```")

    lines += [
        "",
        "## Meta",
        "",
        "```json",
        json.dumps(meta, indent=2),
        "```",
        "",
        "## Non-goals",
        "",
        "- No new features / entropy / ML",
        "- No re-selection of bartime/horizon/direction",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=FULL_START)
    parser.add_argument("--end", default=FULL_END)
    parser.add_argument("--skip-execution", action="store_true")
    parser.add_argument("--skip-independence", action="store_true")
    parser.add_argument(
        "--cost-bps",
        default=",".join(str(x) for x in COST_LADDER_BPS),
        help="Comma-separated one-way cost bps ladder",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    cost_bps = tuple(
        float(x.strip()) for x in args.cost_bps.split(",") if x.strip()
    )

    print(f"[p3] load panel {args.start}→{args.end}", flush=True)
    # Soft load: allow missing holiday files if empty parquets absent.
    frames = []
    missing = []
    for day in pd.bdate_range(args.start, args.end):
        path = args.panel_dir / f"{day.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            missing.append(day.strftime("%Y-%m-%d"))
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frames.append(frame)
    if missing:
        # Require coverage in each period for frozen bartimes evaluation.
        print(
            f"[p3] WARNING missing {len(missing)} panel days "
            f"(first={missing[0]} last={missing[-1]})",
            flush=True,
        )
        # Fail hard if validation/test mostly missing.
        for pname, period in PERIODS.items():
            need = [
                d
                for d in missing
                if period["start"] <= d <= period["end"]
            ]
            if pname != "train_2024H1" and len(need) > 5:
                raise FileNotFoundError(
                    f"Period {pname} missing {len(need)} panel days; "
                    "export Feature Factory OOS panel first "
                    "(see research/docs/l2_phase3_validation_v1.md)."
                )
    if not frames:
        raise FileNotFoundError(f"No panel rows under {args.panel_dir}")
    panel = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(panel["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    panel["date"] = dates
    print(
        f"[p3] panel rows={len(panel)} dates={panel['date'].nunique()} "
        f"symbols={panel['symbol'].nunique()}",
        flush=True,
    )

    session = _connect()
    all_metrics: List[pd.DataFrame] = []
    all_deciles: List[pd.DataFrame] = []
    all_ic: List[pd.DataFrame] = []
    residual_rows: List[dict] = []
    cost_rows: List[dict] = []

    # ---- Evaluate frozen tuples on train / val / test ----
    for period_name, period in PERIODS.items():
        p_panel = _filter_period(panel, period["start"], period["end"])
        for cand in FROZEN_CANDIDATES:
            factor = cand["factor"]
            bt = cand["bartime"]
            hz = cand["horizon"]
            direction = int(cand["direction"])
            signal = _slot_signal(p_panel, factor, bt)
            print(
                f"[p3] {period_name} {factor} {bt}/{hz} dir={direction} "
                f"rows={len(signal)}",
                flush=True,
            )
            if signal.empty:
                continue
            metrics, ic_ts, deciles = _evaluate_rich(
                session,
                signal,
                factor_name=factor,
                period_name=period_name,
                horizons=[hz],
                frozen_direction=direction,
            )
            if not metrics.empty:
                all_metrics.append(metrics)
            if not deciles.empty:
                all_deciles.append(deciles)
            tab = _ic_series_table(
                ic_ts,
                factor=factor,
                period=period_name,
                bartime=bt,
                horizon=hz,
                direction=direction,
            )
            if not tab.empty:
                all_ic.append(tab)

            if not args.skip_execution and period_name in (
                "train_2024H1",
                "validation_2024H2",
                "test_2025_available",
            ):
                cost_rows.extend(
                    _simulate_cost_ladder(
                        session,
                        signal,
                        factor_name=factor,
                        period_name=period_name,
                        bartime=bt,
                        horizon=hz,
                        direction=direction,
                        cost_bps_list=cost_bps,
                    )
                )

    # ---- Residual independence ----
    if not args.skip_independence:
        for period_name, period in PERIODS.items():
            p_panel = _filter_period(panel, period["start"], period["end"])
            needed_bt = sorted({c["bartime"] for c in FROZEN_CANDIDATES})
            baseline_cache: Dict[str, Dict[str, pd.DataFrame]] = {
                bt: {} for bt in needed_bt
            }
            for name in EXISTING_BASELINE_FACTORS:
                print(f"[p3] baseline {name} {period_name}", flush=True)
                full = _build_factor(name, period["start"], period["end"])
                for bt in needed_bt:
                    baseline_cache[bt][name] = full[
                        full["tradetime"].dt.strftime("%H:%M") == bt
                    ].copy()
            for cand in FROZEN_CANDIDATES:
                bt = cand["bartime"]
                print(
                    f"[p3] residual {period_name} {cand['factor']} @{bt}",
                    flush=True,
                )
                residual_rows.append(
                    _residual_bundle(
                        session,
                        p_panel,
                        factor_name=cand["factor"],
                        period_name=period_name,
                        bartime=bt,
                        horizon=cand["horizon"],
                        direction=int(cand["direction"]),
                        baselines=baseline_cache[bt],
                    )
                )

    metrics_long = (
        pd.concat(all_metrics, ignore_index=True)
        if all_metrics
        else pd.DataFrame()
    )
    oos_metrics = _hl_flat(metrics_long) if not metrics_long.empty else pd.DataFrame()
    oos_metrics.to_csv(args.output / "oos_metrics.csv", index=False)
    metrics_long.to_csv(args.output / "oos_metrics_long.csv", index=False)

    if all_deciles:
        pd.concat(all_deciles, ignore_index=True).to_csv(
            args.output / "oos_decile_returns.csv", index=False
        )
    if all_ic:
        pd.concat(all_ic, ignore_index=True).to_csv(
            args.output / "oos_ic_series.csv", index=False
        )

    residual_df = pd.DataFrame(residual_rows)
    if not residual_df.empty:
        residual_df.to_csv(args.output / "residual_metrics.csv", index=False)

    cost_df = pd.DataFrame(cost_rows)
    if not cost_df.empty:
        cost_df.to_csv(args.output / "cost_analysis.csv", index=False)

    decile_all = (
        pd.concat(all_deciles, ignore_index=True) if all_deciles else pd.DataFrame()
    )

    survival_rows = []
    for cand in FROZEN_CANDIDATES:
        factor = cand["factor"]
        bt = cand["bartime"]
        hz = cand["horizon"]

        def _row(period: str) -> Optional[pd.Series]:
            if oos_metrics.empty:
                return None
            sub = oos_metrics[
                (oos_metrics["factor"] == factor)
                & (oos_metrics["period"] == period)
                & (oos_metrics["bartime"].astype(str) == bt)
                & (oos_metrics["return_window"].astype(str) == hz)
            ]
            return sub.iloc[0] if not sub.empty else None

        def _resid(period: str) -> Optional[dict]:
            if residual_df.empty:
                return None
            sub = residual_df[
                (residual_df["factor"] == factor)
                & (residual_df["period"] == period)
            ]
            return sub.iloc[0].to_dict() if not sub.empty else None

        cost_oos = None
        if not cost_df.empty:
            cost_oos = cost_df[
                (cost_df["factor"] == factor)
                & (cost_df["period"] == "test_2025_available")
            ]

        survival_rows.append(
            _survival_decision(
                cand=cand,
                train=_row("train_2024H1"),
                validation=_row("validation_2024H2"),
                oos=_row("test_2025_available"),
                resid_train=_resid("train_2024H1"),
                resid_oos=_resid("test_2025_available"),
                train_means=_signed_means_from_deciles(
                    decile_all,
                    factor=factor,
                    period="train_2024H1",
                    bartime=bt,
                    horizon=hz,
                ),
                oos_means=_signed_means_from_deciles(
                    decile_all,
                    factor=factor,
                    period="test_2025_available",
                    bartime=bt,
                    horizon=hz,
                ),
                cost_oos=cost_oos,
            )
        )

    survival = pd.DataFrame(survival_rows)
    survival.to_csv(args.output / "factor_survival.csv", index=False)

    meta = {
        "phase": "3_validation",
        "start": args.start,
        "end": args.end,
        "periods": PERIODS,
        "frozen": FROZEN_CANDIDATES,
        "cost_bps": list(cost_bps),
        "gates": {
            "icir": GATE_ICIR,
            "hl_sharpe_train": GATE_HL_SHARPE_TRAIN,
            "hl_sharpe_oos": GATE_HL_SHARPE_OOS,
            "mono": GATE_MONO,
            "residual_icir": GATE_RESID_ICIR,
        },
        "n_panel_dates": int(panel["date"].nunique()),
        "n_symbols": int(panel["symbol"].nunique()),
        "missing_panel_days": len(missing),
    }
    (args.output / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        args.output / "l2_phase3_report.md",
        survival=survival,
        oos_metrics=oos_metrics,
        residual=residual_df,
        cost=cost_df,
        meta=meta,
    )
    print(survival.to_string(index=False), flush=True)
    print(f"[p3] done → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
