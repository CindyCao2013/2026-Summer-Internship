#!/usr/bin/env python3
"""Unified multi-horizon evaluation for Intraday Alpha Library v2 candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factors.intraday.discovery_v1 import ddb_version as discovery_ddb  # noqa: E402
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    FACTORS as PRODUCTION_COMPUTERS,
    _as_tradetime,
    _connect,
    _evaluate,
    _performance_rows,
    _slot_correlations,
)

PRODUCTION_FACTORS = list(PRODUCTION_COMPUTERS)
DISCOVERY_FACTORS = [
    "bartime_ofi",
    "ofi_persistence",
    "active_buy_shock",
    "average_active_trade_size",
    "intraday_amihud",
    "realized_volatility",
    "minute_skew",
]
FACTOR_FAMILY = {
    "close_vwap_deviation": "price",
    "volume_front_loading": "temporal",
    "volume_back_loading": "temporal",
    "late_session_strength": "flow",
    "active_buy_sell_imbalance": "flow",
    "bartime_ofi": "flow",
    "ofi_persistence": "flow",
    "active_buy_shock": "flow",
    "average_active_trade_size": "trade_behavior",
    "intraday_amihud": "liquidity",
    "realized_volatility": "volatility",
    "minute_skew": "distribution",
}
EVALUATION_WINDOWS = ["Ret_15", "Ret_30", "Ret_60", "Ret_120", "Ret_EOD", "Ret_NDay"]


def _build_factor(factor_name: str, start: str, end: str) -> pd.DataFrame:
    if factor_name in PRODUCTION_COMPUTERS:
        narrow = PRODUCTION_COMPUTERS[factor_name](start, end)
    else:
        narrow = discovery_ddb(factor_name, start, end)
    return _as_tradetime(narrow, factor_name)


def _rank_zscore(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="average", pct=True)
    std = ranked.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (ranked - ranked.mean()) / std


def _residualize_against_production(
    target_name: str,
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rank-OLS target on same-slot production factors, cross-section by date."""
    target = signals[target_name].copy()
    target["slot"] = target["tradetime"].dt.strftime("%H:%M")
    parts = []
    for slot, target_slot in target.groupby("slot", sort=True):
        wide = target_slot[["tradetime", "symbol", "value"]].rename(
            columns={"value": target_name}
        )
        controls = []
        for control_name in PRODUCTION_FACTORS:
            control = signals[control_name]
            mask = control["tradetime"].dt.strftime("%H:%M") == slot
            if not mask.any():
                continue
            controls.append(control_name)
            wide = wide.merge(
                control.loc[mask, ["tradetime", "symbol", "value"]].rename(
                    columns={"value": control_name}
                ),
                on=["tradetime", "symbol"],
                how="inner",
            )
        if not controls or wide.empty:
            continue

        def _daily_residual(group: pd.DataFrame) -> pd.DataFrame:
            columns = [target_name, *controls]
            ranked = group[columns].apply(_rank_zscore).dropna()
            if len(ranked) < max(50, len(controls) + 5):
                return pd.DataFrame(columns=["tradetime", "symbol", "value"])
            y = ranked[target_name].to_numpy(dtype=float)
            x = ranked[controls].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            out = group.loc[ranked.index, ["tradetime", "symbol"]].copy()
            out["value"] = y - design @ beta
            return out

        residual = (
            wide.groupby("tradetime", group_keys=False)
            .apply(_daily_residual)
            .reset_index(drop=True)
        )
        parts.append(residual)
    if not parts:
        return pd.DataFrame(
            columns=["tradetime", "symbol", "factorname", "value"]
        )
    result = pd.concat(parts, ignore_index=True)
    result["factorname"] = f"{target_name}_resid_prod"
    return result[["tradetime", "symbol", "factorname", "value"]]


def _assign_role(
    *,
    backend: str,
    abs_icir: float,
    hl_sharpe: float,
    residual_abs_icir: float,
    cost_gate: bool,
) -> str:
    if backend == "production_ddb":
        if abs_icir >= 2.0 and hl_sharpe >= 2.0:
            return "Base"
        if abs_icir >= 2.0 and hl_sharpe >= 1.0:
            return "Satellite"
        if abs_icir >= 1.0 or hl_sharpe >= 1.0:
            return "Enhancer"
        return "Drop"
    if (
        cost_gate
        and
        residual_abs_icir >= 1.5
        and abs_icir >= 2.0
        and hl_sharpe >= 1.0
    ):
        return "Base"
    if residual_abs_icir >= 3.0 and abs_icir >= 2.0 and hl_sharpe >= 3.0:
        return "Satellite"
    if residual_abs_icir >= 0.75 and abs_icir >= 1.25:
        return "Enhancer"
    return "Drop"


def _candidate_summary(performance: pd.DataFrame) -> pd.DataFrame:
    raw = performance[
        performance["factor"].isin(PRODUCTION_FACTORS + DISCOVERY_FACTORS)
        & performance["return_window"].isin(EVALUATION_WINDOWS)
        & (performance["n_dates"] >= 80)
    ].copy()
    rows = []
    for factor_name in PRODUCTION_FACTORS + DISCOVERY_FACTORS:
        factor_rows = raw[raw["factor"] == factor_name].copy()
        if factor_rows.empty:
            continue
        factor_rows["abs_icir"] = factor_rows["icir_annualized"].abs()
        best = factor_rows.sort_values(
            ["abs_icir", "hl_sharpe_directional"],
            ascending=False,
        ).iloc[0]
        residual_name = f"{factor_name}_resid_prod"
        residual = performance[
            (performance["factor"] == residual_name)
            & (performance["bartime"] == best["bartime"])
            & (performance["return_window"] == best["return_window"])
        ]
        residual_ic = np.nan
        residual_icir = np.nan
        residual_retention = np.nan
        if len(residual):
            residual_ic = float(residual.iloc[0]["ic_mean"])
            residual_icir = float(residual.iloc[0]["icir_annualized"])
            if best["icir_annualized"] != 0:
                residual_retention = abs(residual_icir) / abs(
                    float(best["icir_annualized"])
                )
        backend = (
            "production_ddb"
            if factor_name in PRODUCTION_FACTORS
            else "candidate_ddb"
        )
        role = _assign_role(
            backend=backend,
            abs_icir=abs(float(best["icir_annualized"])),
            hl_sharpe=float(best["hl_sharpe_directional"]),
            residual_abs_icir=(
                abs(residual_icir) if np.isfinite(residual_icir) else 0.0
            ),
            cost_gate=bool(best["hl_sharpe_after_7p5bps"] > 0),
        )
        rows.append(
            {
                "factor": factor_name,
                "family": FACTOR_FAMILY[factor_name],
                "backend": backend,
                "best_bartime": best["bartime"],
                "best_horizon": best["return_window"],
                "IC": float(best["ic_mean"]),
                "ICIR": float(best["icir_annualized"]),
                "IC_win_rate": float(best["ic_directional_win_rate"]),
                "Sharpe": float(best["hl_sharpe_directional"]),
                "turnover": float(best["turnover_b_hl"]),
                "cost_sharpe": float(best["hl_sharpe_after_7p5bps"]),
                "residual_IC": residual_ic,
                "residual_ICIR": residual_icir,
                "residual_retention": residual_retention,
                "role": role,
                "cost_gate": bool(best["hl_sharpe_after_7p5bps"] > 0),
                "n_dates": int(best["n_dates"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument(
        "--output",
        default="research/results/intraday_alpha_discovery_v1",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild candidate roles from an existing performance_all.csv.",
    )
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        performance = pd.read_csv(output / "performance_all.csv")
        candidates = _candidate_summary(performance)
        candidates.to_csv(
            output / "intraday_alpha_library_v2_candidates.csv", index=False
        )
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["role_rule"] = (
            "Exploratory role: Base requires positive 7.5bps cost Sharpe for "
            "discovery factors; Satellite requires strong standalone and "
            "production-control residual evidence; Enhancer is gross-only or "
            "narrower evidence. Best slot/horizon is in-sample diagnostic."
        )
        summary["candidates"] = candidates.where(
            pd.notna(candidates), None
        ).to_dict(orient="records")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Candidate roles rebuilt → {output}", flush=True)
        return 0
    session = _connect()
    signals: Dict[str, pd.DataFrame] = {}
    performance_rows = []

    for factor_name in PRODUCTION_FACTORS + DISCOVERY_FACTORS:
        print(f"[BUILD] {factor_name}", flush=True)
        signal = _build_factor(factor_name, args.start, args.end)
        filtered, group_ret, ic_mean, ic_ts = _evaluate(
            session,
            factor_name,
            signal,
            apply_limit_filter=True,
        )
        signals[factor_name] = filtered
        performance_rows.extend(
            _performance_rows(factor_name, group_ret, ic_mean, ic_ts)
        )
        print(
            f"[DONE] {factor_name}: raw={len(signal):,} "
            f"evaluated={len(filtered):,}",
            flush=True,
        )

    correlations = _slot_correlations(signals)
    for factor_name in DISCOVERY_FACTORS:
        print(f"[RESIDUAL] {factor_name}", flush=True)
        residual = _residualize_against_production(factor_name, signals)
        if residual.empty:
            continue
        _, group_ret, ic_mean, ic_ts = _evaluate(
            session,
            f"{factor_name}_resid_prod",
            residual,
            apply_limit_filter=False,
        )
        performance_rows.extend(
            _performance_rows(
                f"{factor_name}_resid_prod",
                group_ret,
                ic_mean,
                ic_ts,
            )
        )

    performance = pd.DataFrame(performance_rows)
    candidates = _candidate_summary(performance)
    decay = performance[
        performance["factor"].isin(PRODUCTION_FACTORS + DISCOVERY_FACTORS)
        & performance["return_window"].isin(EVALUATION_WINDOWS)
    ].copy()

    candidates.to_csv(
        output / "intraday_alpha_library_v2_candidates.csv", index=False
    )
    decay.to_csv(output / "decay_by_slot.csv", index=False)
    performance.to_csv(output / "performance_all.csv", index=False)
    correlations.to_csv(output / "spearman_by_slot.csv", index=False)
    pd.DataFrame(
        [
            {
                "factor": "large_active_buy_ratio",
                "reason": (
                    "Bar-level average-ticket proxy; no tick trade-size or "
                    "OrderID source. Excluded from primary role assignment."
                ),
                "status": "proxy_only",
            }
        ]
    ).to_csv(output / "excluded_proxies.csv", index=False)
    summary = {
        "start": args.start,
        "end": args.end,
        "universe": "000852.SH",
        "horizons": EVALUATION_WINDOWS,
        "residual_method": (
            "Daily cross-sectional rank-z OLS against production factors "
            "available at the same bartime"
        ),
        "role_rule": (
            "Exploratory role: Base requires positive 7.5bps cost Sharpe for "
            "discovery factors; Satellite requires strong standalone and "
            "production-control residual evidence; Enhancer is gross-only or "
            "narrower evidence. Best slot/horizon is in-sample diagnostic."
        ),
        "candidates": candidates.where(pd.notna(candidates), None).to_dict(
            orient="records"
        ),
        "signal_rows": {name: int(len(frame)) for name, frame in signals.items()},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Intraday Alpha Discovery v1 → {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
