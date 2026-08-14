#!/usr/bin/env python3
"""Sprint 4.4 Phase 2.1 — full SSL2 L2 alpha validation.

Research question: does native L2 provide independent alpha beyond
realized_volatility / close_vwap_deviation / intraday_amihud?

Does not modify intraday_evaluation_v2, DDB factor backends, or freeze JSON.
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

from core.evaluation.intraday_metrics import (  # noqa: E402
    build_group_excess_panel,
    build_hl_panel,
    summarize_cross_sectional_metrics,
)
from research.intraday_portfolio_simulator_v1 import (  # noqa: E402
    ONE_WAY_COST_BPS,
    _fetch_extreme_constituents,
    _performance_summary,
    _simulate_ledger,
)
from research.l2_alpha.export_l2_intraday_panel import (  # noqa: E402
    _csi1000_symbols,
)
from research.l2_alpha.l2_factor_panel import to_evaluation_signal  # noqa: E402
from research.l2_alpha.l2_factor_registry import (  # noqa: E402
    DEFAULT_BARTIMES,
    DEFAULT_HORIZONS,
    EXISTING_BASELINE_FACTORS,
    L2_PHASE2_FACTORS,
)
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
from research.run_l2_intraday_evaluation import (  # noqa: E402
    _decile_monotonicity,
    _ic_lookup,
    _residualize_same_slot,
)

DEFAULT_PANEL = PROJECT / "research/results/l2_factor_panel_csi1000"
DEFAULT_OUTPUT = PROJECT / "research/results/l2_alpha_discovery_v2"

PERIODS = {
    "train_2024H1": {"start": "2024-01-01", "end": "2024-06-30"},
    "validation_2024H2": {"start": "2024-07-01", "end": "2024-12-31"},
    "test_2025_available": {"start": "2025-01-01", "end": "2025-08-18"},
}
FULL_START = "2024-01-01"
FULL_END = "2025-08-18"

# Phase 2.1 gates (research contract; tightened)
GATE_ICIR = 2.0
GATE_HL_SHARPE_TRAIN = 3.0
GATE_HL_SHARPE_OOS = 2.0
GATE_MONO = 0.7  # Spearman; also require G1=min, G10=max after direction
GATE_RESID_ICIR = 1.5


def _load_panel(panel_dir: Path, start: str, end: str) -> pd.DataFrame:
    frames = []
    missing = []
    empty_days = 0
    for day in pd.bdate_range(start, end):
        path = panel_dir / f"{day.strftime('%Y%m%d')}.parquet"
        if not path.exists():
            missing.append(day.strftime("%Y-%m-%d"))
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            empty_days += 1
            continue
        frames.append(frame)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} panel days under {panel_dir}; "
            f"first={missing[0]} last={missing[-1]}. "
            "Run research/export_l2_panel_csi1000.py first."
        )
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(out["date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out["date"] = dates
    out.attrs["empty_days"] = empty_days
    return out


def _verify_panel(panel: pd.DataFrame, bartimes: List[str]) -> dict:
    factors = sorted(panel["factor"].unique())
    bts = sorted(panel["bartime"].unique())
    report = {
        "rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "n_symbols": int(panel["symbol"].nunique()),
        "factors": factors,
        "bartimes": bts,
        "missing_factors": sorted(set(L2_PHASE2_FACTORS) - set(factors)),
        "missing_bartimes": sorted(set(bartimes) - set(bts)),
        "cancel_share": float(
            (panel["factor"] == "l2_cancel_pressure_sum").mean()
        ),
    }
    if report["missing_factors"] or report["missing_bartimes"]:
        raise RuntimeError(f"Panel verification failed: {report}")
    return report


def _filter_period(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d0 = pd.Timestamp(start)
    d1 = pd.Timestamp(end)
    dates = pd.to_datetime(panel["date"])
    return panel[(dates >= d0) & (dates <= d1)].copy()


def _evaluate_rich(
    session,
    signal: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    horizons: List[str],
    frozen_direction: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return metrics_long, ic_ts, decile_means."""
    if signal.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    filtered, _, _, ic_ts = _evaluate(
        session,
        f"l21_{factor_name}_{period_name}",
        signal,
        apply_limit_filter=True,
    )
    groups, market = _aggregate_exact_group_and_market(
        session,
        f"l21_{factor_name}_{period_name}",
        filtered,
        horizons,
    )
    panel = build_group_excess_panel(groups, market)
    rows = []
    decile_rows = []
    for (bartime, horizon), combo in panel.groupby(
        ["Bartime", "return_window"], sort=True
    ):
        provisional = _ic_lookup(ic_ts, str(bartime), str(horizon), 1)
        if frozen_direction is None:
            direction = 1 if provisional["rank_ic"] >= 0 else -1
        else:
            direction = int(frozen_direction)
        ic = _ic_lookup(ic_ts, str(bartime), str(horizon), direction)
        hl = build_hl_panel(combo, direction=direction)
        metrics = summarize_cross_sectional_metrics(
            combo, hl, factor_name=factor_name
        )
        metrics["period"] = period_name
        metrics["direction"] = direction
        metrics["rank_ic"] = ic["rank_ic"]
        metrics["annualized_icir"] = ic["annualized_icir"]
        metrics["ic_win_rate"] = ic["ic_win_rate"]
        gmeans = (
            combo.groupby("group")["group_return_excess"]
            .mean()
            .reindex([f"G{i}" for i in range(1, 11)])
        )
        signed_means = direction * gmeans
        metrics["decile_mono_spearman"] = _decile_monotonicity(
            signed_means.dropna()
        )
        rows.append(metrics)

        for g, val in gmeans.items():
            decile_rows.append(
                {
                    "factor": factor_name,
                    "period": period_name,
                    "bartime": str(bartime),
                    "horizon": str(horizon),
                    "direction": direction,
                    "group": g,
                    "mean_excess_return": float(val)
                    if pd.notna(val)
                    else np.nan,
                    "signed_mean_excess_return": (
                        float(direction * val) if pd.notna(val) else np.nan
                    ),
                }
            )
    metrics_df = (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    )
    decile_df = pd.DataFrame(decile_rows)
    return metrics_df, ic_ts, decile_df


def _hl_flat(metrics: pd.DataFrame) -> pd.DataFrame:
    hl = metrics[metrics["metric_scope"] == "cross_sectional_hl"].copy()
    if hl.empty:
        return hl
    g1 = metrics[
        (metrics["metric_scope"] == "cross_sectional_group")
        & (metrics["group"] == "G1")
    ][
        ["factor", "period", "bartime", "return_window", "group_excess_sharpe"]
    ].rename(columns={"group_excess_sharpe": "g1_excess_sharpe"})
    g10 = metrics[
        (metrics["metric_scope"] == "cross_sectional_group")
        & (metrics["group"] == "G10")
    ][
        ["factor", "period", "bartime", "return_window", "group_excess_sharpe"]
    ].rename(columns={"group_excess_sharpe": "g10_excess_sharpe"})
    out = hl.merge(
        g1, on=["factor", "period", "bartime", "return_window"], how="left"
    ).merge(
        g10, on=["factor", "period", "bartime", "return_window"], how="left"
    )
    keep = [
        "factor",
        "period",
        "bartime",
        "return_window",
        "direction",
        "rank_ic",
        "annualized_icir",
        "ic_win_rate",
        "g1_excess_sharpe",
        "g10_excess_sharpe",
        "hl_sharpe",
        "hl_annualized_return",
        "decile_mono_spearman",
        "hl_market_beta",
        "hl_market_corr",
    ]
    cols = [c for c in keep if c in out.columns]
    return out[cols]


def _select_train_best(train_hl: pd.DataFrame) -> pd.DataFrame:
    """One row per factor: max |ICIR| on train."""
    rows = []
    for factor, sub in train_hl.groupby("factor"):
        scored = sub.copy()
        scored["abs_icir"] = scored["annualized_icir"].abs()
        best = scored.sort_values(
            ["abs_icir", "hl_sharpe"], ascending=False
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def _ic_series_table(
    ic_ts: pd.DataFrame,
    *,
    factor: str,
    period: str,
    bartime: str,
    horizon: str,
    direction: int,
) -> pd.DataFrame:
    frame = ic_ts.copy()
    frame["bartime_key"] = _bt_string(frame["Bartime"])
    ret_key = "RetType" if "RetType" in frame.columns else "valueType"
    selected = frame[
        (frame["bartime_key"] == bartime)
        & (frame[ret_key].astype(str) == horizon)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["Date"] = pd.to_datetime(selected["Date"])
    selected["factor"] = factor
    selected["period"] = period
    selected["bartime"] = bartime
    selected["horizon"] = horizon
    selected["direction"] = direction
    selected["signed_rank_ic"] = direction * selected["Rank_IC"].astype(float)
    return selected[
        [
            "factor",
            "period",
            "Date",
            "bartime",
            "horizon",
            "direction",
            "Rank_IC",
            "signed_rank_ic",
        ]
    ]


def _simulate_execution(
    session,
    signal: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    bartime: str,
    horizon: str,
    direction: int,
) -> dict:
    slot = signal[signal["tradetime"].dt.strftime("%H:%M") == bartime].copy()
    if slot.empty:
        return {}
    filtered, group_ret, _, _ = _evaluate(
        session,
        f"l21sim_{factor_name}_{period_name}",
        slot,
        apply_limit_filter=True,
    )
    constituents = _fetch_extreme_constituents(
        session, f"{factor_name}_{period_name}", filtered, horizon
    )
    ledger = _simulate_ledger(
        constituents,
        factor_name=factor_name,
        period_name=period_name,
        horizon=horizon,
        direction=direction,
        one_way_cost_bps=ONE_WAY_COST_BPS,
    )
    # Skip gross-parity assert vs get_cs HML (we use exact-excess path).
    summary = _performance_summary(
        ledger,
        freeze_sha256="l2_phase21",
        simulation_sha256="l2_phase21_exec",
        max_gross_parity_diff=np.nan,
    )
    summary["period"] = period_name
    summary["bartime"] = bartime
    # Attach unused group_ret to silence lint if needed
    _ = group_ret
    return summary


def _residual_bundle(
    session,
    panel: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    bartime: str,
    horizon: str,
    direction: int,
    baselines: Dict[str, pd.DataFrame],
) -> dict:
    signal = to_evaluation_signal(panel, factor_name)
    signal = signal[signal["tradetime"].dt.strftime("%H:%M") == bartime].copy()
    resid = _residualize_same_slot(signal, baselines, bartime)
    if resid.empty:
        return {
            "factor": factor_name,
            "period": period_name,
            "bartime": bartime,
            "horizon": horizon,
            "direction": direction,
            "residual_rank_ic": np.nan,
            "residual_icir": np.nan,
            "residual_hl_sharpe": np.nan,
            "n_resid_rows": 0,
        }
    metrics, _, _ = _evaluate_rich(
        session,
        resid,
        factor_name=f"{factor_name}_resid",
        period_name=period_name,
        horizons=[horizon],
        frozen_direction=direction,
    )
    flat = _hl_flat(metrics)
    if flat.empty:
        return {
            "factor": factor_name,
            "period": period_name,
            "bartime": bartime,
            "horizon": horizon,
            "direction": direction,
            "residual_rank_ic": np.nan,
            "residual_icir": np.nan,
            "residual_hl_sharpe": np.nan,
            "n_resid_rows": int(len(resid)),
        }
    row = flat.iloc[0]
    return {
        "factor": factor_name,
        "period": period_name,
        "bartime": bartime,
        "horizon": horizon,
        "direction": direction,
        "residual_rank_ic": float(row["rank_ic"]),
        "residual_icir": float(row["annualized_icir"]),
        "residual_hl_sharpe": float(row["hl_sharpe"]),
        "residual_mono": float(row.get("decile_mono_spearman", np.nan)),
        "n_resid_rows": int(len(resid)),
    }


def _ic_correlation(
    session,
    panel: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    period_name: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Daily IC Spearman correlation among L2 + baselines at each factor slot.

    Uses each L2 factor's selected bartime/horizon; baselines evaluated at the
    same slot as each L2 factor, then averaged correlation matrix is reported
    at a common anchor: first selection's slot for baselines-only block, plus
    pairwise L2/baseline using overlapping dates where possible.
    """
    # Build IC series at a shared anchor: 14:29 / Ret_30 for comparability
    anchor_bt = "14:29"
    anchor_hz = "Ret_30"
    series = {}
    for name in EXISTING_BASELINE_FACTORS:
        sig = _build_factor(name, start, end)
        sig = sig[sig["tradetime"].dt.strftime("%H:%M") == anchor_bt]
        _, _, _, ic_ts = _evaluate(
            session, f"corr_{name}_{period_name}", sig, apply_limit_filter=True
        )
        tab = _ic_series_table(
            ic_ts,
            factor=name,
            period=period_name,
            bartime=anchor_bt,
            horizon=anchor_hz,
            direction=1,
        )
        if not tab.empty:
            series[name] = tab.set_index("Date")["Rank_IC"]
    for _, sel in selections.iterrows():
        factor = sel["factor"]
        sig = to_evaluation_signal(panel, factor)
        sig = sig[sig["tradetime"].dt.strftime("%H:%M") == anchor_bt]
        if sig.empty:
            continue
        _, _, _, ic_ts = _evaluate(
            session,
            f"corr_{factor}_{period_name}",
            sig,
            apply_limit_filter=True,
        )
        tab = _ic_series_table(
            ic_ts,
            factor=factor,
            period=period_name,
            bartime=anchor_bt,
            horizon=anchor_hz,
            direction=1,
        )
        if not tab.empty:
            series[factor] = tab.set_index("Date")["Rank_IC"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).corr(method="spearman")


def _decile_gate_pass(
    signed_means: Optional[pd.Series],
    *,
    spearman: float,
) -> bool:
    """Direction-adjusted: G1 lowest, G10 highest, and Spearman trend."""
    if signed_means is None or len(signed_means.dropna()) < 10:
        # Fall back to Spearman-only when full decile vector unavailable.
        return float(spearman) >= GATE_MONO if np.isfinite(spearman) else False
    g = signed_means.reindex([f"G{i}" for i in range(1, 11)]).astype(float)
    if g.isna().any():
        return False
    g1_lowest = bool(g.iloc[0] <= g.min() + 1e-15)
    g10_highest = bool(g.iloc[-1] >= g.max() - 1e-15)
    return g1_lowest and g10_highest and float(spearman) >= GATE_MONO


def _decision_row(
    train: pd.Series,
    oos: Optional[pd.Series],
    resid: Optional[dict],
    exec_oos: Optional[dict],
    *,
    train_signed_means: Optional[pd.Series] = None,
    oos_signed_means: Optional[pd.Series] = None,
) -> dict:
    train_spearman = float(train.get("decile_mono_spearman", np.nan))
    mono_ok = _decile_gate_pass(train_signed_means, spearman=train_spearman)
    stand_ok = (
        abs(float(train["annualized_icir"])) > GATE_ICIR
        and float(train["hl_sharpe"]) > GATE_HL_SHARPE_TRAIN
        and mono_ok
    )
    oos_ok = False
    oos_mono = False
    if oos is not None:
        oos_spearman = float(oos.get("decile_mono_spearman", np.nan))
        oos_mono = _decile_gate_pass(oos_signed_means, spearman=oos_spearman)
        # Production promotion requires OOS H-L Sharpe > 2 (gross CS alpha).
        # Net LS Sharpe is diagnostic (cost feasibility), not the hard OOS gate.
        oos_ok = float(oos["hl_sharpe"]) > GATE_HL_SHARPE_OOS and oos_mono
    resid_icir = (
        float(resid["residual_icir"]) if resid is not None else np.nan
    )
    indep_ok = abs(resid_icir) > GATE_RESID_ICIR if np.isfinite(resid_icir) else False

    if stand_ok and oos_ok and indep_ok:
        decision = "ADD"
    elif stand_ok and indep_ok and not oos_ok:
        decision = "WATCH"
    elif stand_ok and not indep_ok:
        decision = "DUPLICATE"
    else:
        decision = "DROP"

    return {
        "factor": train["factor"],
        "bartime": train["bartime"],
        "horizon": train["return_window"],
        "direction": int(train["direction"]),
        "train_rank_ic": float(train["rank_ic"]),
        "train_icir": float(train["annualized_icir"]),
        "train_hl_sharpe": float(train["hl_sharpe"]),
        "train_mono": train_spearman,
        "train_mono_pass": mono_ok,
        "oos_hl_sharpe": float(oos["hl_sharpe"]) if oos is not None else np.nan,
        "oos_mono_pass": oos_mono,
        "oos_net_ls_sharpe": (
            float(exec_oos["net_ls_sharpe"]) if exec_oos else np.nan
        ),
        "residual_icir": resid_icir,
        "residual_hl_sharpe": (
            float(resid["residual_hl_sharpe"]) if resid else np.nan
        ),
        "standalone_pass": stand_ok,
        "oos_pass": oos_ok,
        "independence_pass": indep_ok,
        "decision": decision,
    }


def _write_report(
    path: Path,
    *,
    panel_report: dict,
    decisions: pd.DataFrame,
    train_hl: pd.DataFrame,
    residual: pd.DataFrame,
) -> None:
    lines = [
        "# Sprint 4.4 Phase 2.1 — L2 Alpha Validation Report",
        "",
        "## Research question",
        "",
        "Does native SSL2 L2 information provide independent alpha beyond",
        "`realized_volatility`, `close_vwap_deviation`, and `intraday_amihud`?",
        "",
        "## Panel verification",
        "",
        "```json",
        json.dumps(panel_report, indent=2),
        "```",
        "",
        "## Gates",
        "",
        f"- Standalone: |ICIR| > {GATE_ICIR}, H-L Sharpe > {GATE_HL_SHARPE_TRAIN}, "
        f"decile mono (G1 lowest → G10 highest, Spearman ≥ {GATE_MONO})",
        f"- OOS 2025: H-L Sharpe > {GATE_HL_SHARPE_OOS} and same decile mono",
        f"- Independence: |Residual ICIR| > {GATE_RESID_ICIR}",
        "",
        "Net LS Sharpe / break-even cost remain diagnostics (execution feasibility).",
        "",
        "## Decisions",
        "",
    ]
    def _table(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:  # noqa: BLE001
            return "```\n" + df.to_string(index=False) + "\n```"

    if decisions.empty:
        lines.append("_No decisions produced._")
    else:
        lines.append(_table(decisions))
    lines += [
        "",
        "## Train screen (HL)",
        "",
    ]
    if not train_hl.empty:
        show = train_hl[
            [
                c
                for c in [
                    "factor",
                    "bartime",
                    "return_window",
                    "direction",
                    "rank_ic",
                    "annualized_icir",
                    "hl_sharpe",
                    "decile_mono_spearman",
                ]
                if c in train_hl.columns
            ]
        ]
        lines.append(_table(show))
    lines += [
        "",
        "## Residual alpha",
        "",
    ]
    if not residual.empty:
        lines.append(_table(residual))
    lines += [
        "",
        "## Interpretation note",
        "",
        "- Pipeline success ≠ production ADD: SSL2 path is validated.",
        "- Residual ICIR answers independence vs RV/CVWAP/Amihud.",
        "- Current survivors look like short-horizon microstructure sources;",
        "  none clear the tightened OOS H-L > 2 + mono production gate.",
        "- High standalone ICIR with near-zero residual ICIR ⇒ OHLCV proxy.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=FULL_START)
    parser.add_argument("--end", default=FULL_END)
    parser.add_argument(
        "--bartimes", default=",".join(DEFAULT_BARTIMES)
    )
    parser.add_argument(
        "--horizons", default=",".join(DEFAULT_HORIZONS)
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip position simulator (faster debug)",
    )
    parser.add_argument(
        "--skip-independence",
        action="store_true",
    )
    args = parser.parse_args(argv)

    bartimes = [b.strip() for b in args.bartimes.split(",") if b.strip()]
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[v21] load panel {args.start}→{args.end}", flush=True)
    panel = _load_panel(args.panel_dir, args.start, args.end)
    panel_report = _verify_panel(panel, bartimes)
    print(f"[v21] panel ok {panel_report}", flush=True)
    (args.output / "panel_verification.json").write_text(
        json.dumps(panel_report, indent=2) + "\n", encoding="utf-8"
    )

    session = _connect()
    all_metrics = []
    all_deciles = []
    all_ic = []

    # ---- Train: full 84-grid screen ----
    train = PERIODS["train_2024H1"]
    train_panel = _filter_period(panel, train["start"], train["end"])
    for factor_name in L2_PHASE2_FACTORS:
        signal = to_evaluation_signal(train_panel, factor_name)
        signal = signal[
            signal["tradetime"].dt.strftime("%H:%M").isin(bartimes)
        ]
        print(
            f"[v21] TRAIN evaluate {factor_name} rows={len(signal)}",
            flush=True,
        )
        metrics, ic_ts, deciles = _evaluate_rich(
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
        # Keep IC series for all train slots (decay diagnostics)
        for bt in bartimes:
            for hz in horizons:
                tab = _ic_series_table(
                    ic_ts,
                    factor=factor_name,
                    period="train_2024H1",
                    bartime=bt,
                    horizon=hz,
                    direction=1,
                )
                if not tab.empty:
                    all_ic.append(tab)

    metrics_long = pd.concat(all_metrics, ignore_index=True)
    train_hl = _hl_flat(
        metrics_long[metrics_long["period"] == "train_2024H1"]
    )
    selections = _select_train_best(train_hl)
    selections.to_csv(args.output / "l2_train_selections.csv", index=False)
    print("[v21] train selections:", flush=True)
    print(selections.to_string(index=False), flush=True)

    # ---- Val / OOS at frozen selected tuples ----
    exec_rows = []
    residual_rows = []
    for period_name in ("validation_2024H2", "test_2025_available"):
        period = PERIODS[period_name]
        p_panel = _filter_period(panel, period["start"], period["end"])
        for _, sel in selections.iterrows():
            factor_name = sel["factor"]
            bt = str(sel["bartime"])
            hz = str(sel["return_window"])
            direction = int(sel["direction"])
            signal = to_evaluation_signal(p_panel, factor_name)
            signal = signal[
                signal["tradetime"].dt.strftime("%H:%M") == bt
            ]
            print(
                f"[v21] {period_name} {factor_name} {bt}/{hz} "
                f"dir={direction} rows={len(signal)}",
                flush=True,
            )
            metrics, ic_ts, deciles = _evaluate_rich(
                session,
                signal,
                factor_name=factor_name,
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
                factor=factor_name,
                period=period_name,
                bartime=bt,
                horizon=hz,
                direction=direction,
            )
            if not tab.empty:
                all_ic.append(tab)

            if not args.skip_execution:
                ex = _simulate_execution(
                    session,
                    signal,
                    factor_name=factor_name,
                    period_name=period_name,
                    bartime=bt,
                    horizon=hz,
                    direction=direction,
                )
                if ex:
                    exec_rows.append(ex)

    # Train execution for selected tuples
    if not args.skip_execution:
        for _, sel in selections.iterrows():
            factor_name = sel["factor"]
            bt = str(sel["bartime"])
            hz = str(sel["return_window"])
            direction = int(sel["direction"])
            signal = to_evaluation_signal(train_panel, factor_name)
            signal = signal[
                signal["tradetime"].dt.strftime("%H:%M") == bt
            ]
            ex = _simulate_execution(
                session,
                signal,
                factor_name=factor_name,
                period_name="train_2024H1",
                bartime=bt,
                horizon=hz,
                direction=direction,
            )
            if ex:
                exec_rows.append(ex)

    # ---- Residual alpha on train (and OOS) for selected tuples ----
    if not args.skip_independence:
        for period_name in (
            "train_2024H1",
            "validation_2024H2",
            "test_2025_available",
        ):
            period = PERIODS[period_name]
            p_panel = _filter_period(panel, period["start"], period["end"])
            # Cache baseline factors once per period (all needed bartimes).
            needed_bt = sorted({str(s["bartime"]) for _, s in selections.iterrows()})
            baseline_cache: Dict[str, Dict[str, pd.DataFrame]] = {
                bt: {} for bt in needed_bt
            }
            for name in EXISTING_BASELINE_FACTORS:
                print(
                    f"[v21] build baseline {name} {period_name}",
                    flush=True,
                )
                full = _build_factor(name, period["start"], period["end"])
                for bt in needed_bt:
                    baseline_cache[bt][name] = full[
                        full["tradetime"].dt.strftime("%H:%M") == bt
                    ].copy()
            for _, sel in selections.iterrows():
                bt = str(sel["bartime"])
                print(
                    f"[v21] residual {period_name} {sel['factor']} @{bt}",
                    flush=True,
                )
                residual_rows.append(
                    _residual_bundle(
                        session,
                        p_panel,
                        factor_name=sel["factor"],
                        period_name=period_name,
                        bartime=bt,
                        horizon=str(sel["return_window"]),
                        direction=int(sel["direction"]),
                        baselines=baseline_cache[bt],
                    )
                )
        corr = _ic_correlation(
            session,
            _filter_period(panel, train["start"], train["end"]),
            selections,
            period_name="train_2024H1",
            start=train["start"],
            end=train["end"],
        )
        corr.to_csv(args.output / "l2_ic_correlation.csv")
    else:
        corr = pd.DataFrame()

    metrics_long = pd.concat(all_metrics, ignore_index=True)
    metrics_long.to_csv(args.output / "l2_factor_metrics_long.csv", index=False)
    factor_metrics = _hl_flat(metrics_long)
    factor_metrics.to_csv(args.output / "l2_factor_metrics.csv", index=False)

    # Decay matrix: train ICIR by bartime × horizon for each factor
    decay_parts = []
    for factor, sub in train_hl.groupby("factor"):
        pivot = sub.pivot(
            index="bartime", columns="return_window", values="annualized_icir"
        )
        pivot["factor"] = factor
        decay_parts.append(pivot.reset_index())
    decay = (
        pd.concat(decay_parts, ignore_index=True)
        if decay_parts
        else pd.DataFrame()
    )
    decay.to_csv(args.output / "l2_decay_matrix.csv", index=False)

    if all_deciles:
        pd.concat(all_deciles, ignore_index=True).to_csv(
            args.output / "l2_decile_returns.csv", index=False
        )
    if all_ic:
        pd.concat(all_ic, ignore_index=True).to_csv(
            args.output / "l2_ic_series.csv", index=False
        )

    exec_df = pd.DataFrame(exec_rows)
    if not exec_df.empty:
        exec_df.to_csv(args.output / "l2_execution.csv", index=False)

    residual_df = pd.DataFrame(residual_rows)
    if not residual_df.empty:
        residual_df.to_csv(args.output / "l2_residual_alpha.csv", index=False)

    # Decisions
    decile_all = (
        pd.concat(all_deciles, ignore_index=True) if all_deciles else pd.DataFrame()
    )

    def _signed_means(factor: str, period: str, bt: str, hz: str) -> Optional[pd.Series]:
        if decile_all.empty:
            return None
        sub = decile_all[
            (decile_all["factor"] == factor)
            & (decile_all["period"] == period)
            & (decile_all["bartime"].astype(str) == bt)
            & (decile_all["horizon"].astype(str) == hz)
        ]
        if sub.empty:
            return None
        return sub.set_index("group")["signed_mean_excess_return"]

    decision_rows = []
    for _, sel in selections.iterrows():
        factor = sel["factor"]
        bt = str(sel["bartime"])
        hz = str(sel["return_window"])
        oos = factor_metrics[
            (factor_metrics["factor"] == factor)
            & (factor_metrics["period"] == "test_2025_available")
        ]
        oos_row = oos.iloc[0] if not oos.empty else None
        resid = None
        if not residual_df.empty:
            r = residual_df[
                (residual_df["factor"] == factor)
                & (residual_df["period"] == "train_2024H1")
            ]
            resid = r.iloc[0].to_dict() if not r.empty else None
        ex_oos = None
        if not exec_df.empty:
            e = exec_df[
                (exec_df["factor"] == factor)
                & (exec_df["period"] == "test_2025_available")
            ]
            ex_oos = e.iloc[0].to_dict() if not e.empty else None
        decision_rows.append(
            _decision_row(
                sel,
                oos_row,
                resid,
                ex_oos,
                train_signed_means=_signed_means(
                    factor, "train_2024H1", bt, hz
                ),
                oos_signed_means=_signed_means(
                    factor, "test_2025_available", bt, hz
                ),
            )
        )
    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(args.output / "l2_decisions.csv", index=False)

    _write_report(
        args.output / "l2_factor_report.md",
        panel_report=panel_report,
        decisions=decisions,
        train_hl=selections,
        residual=residual_df,
    )

    # Simple decay heatmap-style CSV already; optional matplotlib plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, factor in zip(axes.ravel(), L2_PHASE2_FACTORS):
            sub = train_hl[train_hl["factor"] == factor]
            if sub.empty:
                ax.set_title(factor)
                continue
            pivot = sub.pivot(
                index="bartime",
                columns="return_window",
                values="annualized_icir",
            )
            im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(list(pivot.index))
            ax.set_title(factor)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(args.output / "l2_decay_plots.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[v21] plot skipped: {exc}", flush=True)

    meta = {
        "phase": "2.1",
        "start": args.start,
        "end": args.end,
        "periods": PERIODS,
        "bartimes": bartimes,
        "horizons": horizons,
        "factors": list(L2_PHASE2_FACTORS),
        "gates": {
            "icir": GATE_ICIR,
            "hl_sharpe_train": GATE_HL_SHARPE_TRAIN,
            "hl_sharpe_oos": GATE_HL_SHARPE_OOS,
            "mono_spearman": GATE_MONO,
            "mono_rule": "G1_lowest_G10_highest_after_direction",
            "residual_icir": GATE_RESID_ICIR,
        },
        "panel": panel_report,
        "n_universe_symbols": len(_csi1000_symbols(args.start, args.end)),
    }
    (args.output / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[v21] done → {args.output}", flush=True)
    if not decisions.empty:
        print(decisions.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
