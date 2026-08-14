#!/usr/bin/env python3
"""Evaluate frozen intraday alpha specifications without OOS reselection."""

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

import Factor_Dev_Lib as fdl  # noqa: E402
import intraday_lib  # noqa: E402
from research.freeze_intraday_alpha_v1 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_FREEZE,
    verify_spec,
)
from research.run_intraday_alpha_discovery_v1 import (  # noqa: E402
    DISCOVERY_FACTORS,
    PRODUCTION_FACTORS,
    _build_factor,
    _rank_zscore,
)
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    _bt_string,
    _connect,
    _evaluate,
)

DEFAULT_RESULT = ROOT / "research/results/intraday_alpha_oos_v1"


def _build_factor_chunked(
    factor_name: str,
    start: str,
    end: str,
    chunk_months: int,
) -> pd.DataFrame:
    """Build bounded date chunks to avoid large server-side intermediates."""
    cursor = pd.Timestamp(start)
    final = pd.Timestamp(end)
    parts = []
    while cursor <= final:
        chunk_end = min(
            cursor + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1),
            final,
        )
        print(
            f"  [CHUNK] {factor_name}: {cursor.date()} → {chunk_end.date()}",
            flush=True,
        )
        parts.append(
            _build_factor(
                factor_name,
                cursor.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )
        cursor = chunk_end + pd.Timedelta(days=1)
    result = pd.concat(parts, ignore_index=True)
    return result.drop_duplicates(
        subset=["tradetime", "symbol", "factorname"],
        keep="last",
    )


def _filter_slots(signal: pd.DataFrame, slots: set[str]) -> pd.DataFrame:
    bartime = pd.to_datetime(signal["tradetime"]).dt.strftime("%H:%M")
    result = signal.loc[bartime.isin(slots)].copy()
    if result.empty:
        raise ValueError(f"No signal rows for frozen slots {sorted(slots)}")
    return result


def _fixed_metric(
    factor_name: str,
    period_name: str,
    spec: dict,
    group_ret: pd.DataFrame,
    ic_mean: pd.DataFrame,
    ic_ts: pd.DataFrame,
) -> dict:
    """Read one frozen tuple and apply its train-period direction."""
    bartime = str(spec["bartime"])
    horizon = str(spec["horizon"])
    direction = int(spec["direction"])
    if direction not in (-1, 1):
        raise ValueError(f"{factor_name}: invalid frozen direction {direction}")

    excess = intraday_lib.subtract_market_return(group_ret).copy()
    excess["bartime_key"] = _bt_string(excess["Bartime"])
    ic_mean = ic_mean.copy()
    ic_ts = ic_ts.copy()
    ic_mean["bartime_key"] = _bt_string(ic_mean["Bartime"])
    ic_ts["bartime_key"] = _bt_string(ic_ts["Bartime"])
    ret_key_mean = "RetType" if "RetType" in ic_mean.columns else "valueType"
    ret_key_ts = "RetType" if "RetType" in ic_ts.columns else "valueType"

    hl = excess[
        (excess["bartime_key"] == bartime)
        & (excess["group"] == "group_HML")
    ].sort_values("Date")
    if hl.empty or horizon not in hl.columns:
        raise ValueError(
            f"{factor_name}: missing frozen tuple {bartime}/{horizon}"
        )
    hl_ret = pd.to_numeric(hl[horizon], errors="coerce").dropna()
    signed_hl_ret = direction * hl_ret
    turnover = float(intraday_lib.intraday_turnover_b_hl())
    fee_bps = float(getattr(fdl, "IMPLIED_ANNU_FEE_BPS", 7.5))
    net_ret = signed_hl_ret - turnover * fee_bps / 1e4

    im = ic_mean[
        (ic_mean[ret_key_mean].astype(str) == horizon)
        & (ic_mean["bartime_key"] == bartime)
    ]
    its = ic_ts[
        (ic_ts[ret_key_ts].astype(str) == horizon)
        & (ic_ts["bartime_key"] == bartime)
    ]
    if len(im) != 1:
        raise ValueError(
            f"{factor_name}: expected one IC row for {bartime}/{horizon}, "
            f"found {len(im)}"
        )
    raw_ic = float(im.iloc[0]["IC_Mean"])
    raw_icir = float(im.iloc[0]["IC_IR"]) * np.sqrt(250)
    ic_series = pd.to_numeric(its["Rank_IC"], errors="coerce").dropna()
    return {
        "factor": factor_name,
        "family": spec["family"],
        "backend": spec["backend"],
        "research_role": spec["research_role"],
        "period": period_name,
        "bartime": bartime,
        "horizon": horizon,
        "direction": direction,
        "portfolio_rule": spec["portfolio_rule"],
        "train_raw_ic": float(spec["train_raw_ic"]),
        "train_raw_icir": float(spec["train_raw_icir"]),
        "train_fixed_direction_sharpe": float(
            spec["train_fixed_direction_sharpe"]
        ),
        "oos_raw_ic": raw_ic,
        "oos_signed_ic": direction * raw_ic,
        "oos_raw_icir": raw_icir,
        "oos_signed_icir": direction * raw_icir,
        "oos_fixed_direction_ic_win_rate": float(
            (direction * ic_series > 0).mean()
        ),
        "oos_raw_hl_sharpe": float(fdl.calSharpe(hl_ret)),
        "oos_fixed_direction_hl_sharpe": float(
            fdl.calSharpe(signed_hl_ret)
        ),
        "legacy_turnover_model": turnover,
        "legacy_model_cost_sharpe_7p5bps": float(fdl.calSharpe(net_ret)),
        "n_dates": int(hl["Date"].nunique()),
    }


def _ic_decay(
    factor_name: str,
    period_name: str,
    spec: dict,
    ic_mean: pd.DataFrame,
    ic_ts: pd.DataFrame,
) -> list[dict]:
    """Diagnostic decay at the frozen slot; never used for OOS selection."""
    bartime = str(spec["bartime"])
    direction = int(spec["direction"])
    im = ic_mean.copy()
    its = ic_ts.copy()
    im["bartime_key"] = _bt_string(im["Bartime"])
    its["bartime_key"] = _bt_string(its["Bartime"])
    ret_key_mean = "RetType" if "RetType" in im.columns else "valueType"
    ret_key_ts = "RetType" if "RetType" in its.columns else "valueType"
    rows = []
    for horizon in sorted(im[ret_key_mean].astype(str).unique()):
        mean_row = im[
            (im[ret_key_mean].astype(str) == horizon)
            & (im["bartime_key"] == bartime)
        ]
        series = its[
            (its[ret_key_ts].astype(str) == horizon)
            & (its["bartime_key"] == bartime)
        ]
        if len(mean_row) != 1:
            continue
        raw_ic = float(mean_row.iloc[0]["IC_Mean"])
        raw_icir = float(mean_row.iloc[0]["IC_IR"]) * np.sqrt(250)
        rank_ic = pd.to_numeric(series["Rank_IC"], errors="coerce").dropna()
        rows.append(
            {
                "factor": factor_name,
                "period": period_name,
                "frozen_bartime": bartime,
                "return_window": horizon,
                "is_frozen_horizon": horizon == spec["horizon"],
                "direction": direction,
                "raw_ic": raw_ic,
                "signed_ic": direction * raw_ic,
                "raw_icir_annualized": raw_icir,
                "signed_icir_annualized": direction * raw_icir,
                "fixed_direction_ic_win_rate": float(
                    (direction * rank_ic > 0).mean()
                ),
            }
        )
    return rows


def _residualize_frozen(
    target_name: str,
    target: pd.DataFrame,
    production_signals: Dict[str, pd.DataFrame],
    controls: list[str],
    bartime: str,
) -> pd.DataFrame:
    """Apply the exact frozen same-slot control set; no dynamic selection."""
    wide = target[["tradetime", "symbol", "value"]].rename(
        columns={"value": target_name}
    )
    for control_name in controls:
        if control_name not in production_signals:
            raise ValueError(
                f"{target_name}: frozen control {control_name} was not built"
            )
        control = production_signals[control_name]
        mask = control["tradetime"].dt.strftime("%H:%M") == bartime
        if not mask.any():
            raise ValueError(
                f"{target_name}: frozen control {control_name} has no "
                f"{bartime} rows"
            )
        wide = wide.merge(
            control.loc[mask, ["tradetime", "symbol", "value"]].rename(
                columns={"value": control_name}
            ),
            on=["tradetime", "symbol"],
            how="inner",
        )

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

    result = (
        wide.groupby("tradetime", group_keys=False)
        .apply(_daily_residual)
        .reset_index(drop=True)
    )
    result["factorname"] = f"{target_name}_resid_frozen"
    return result[["tradetime", "symbol", "factorname", "value"]]


def _period_verdict(row: pd.Series) -> str:
    if row["oos_signed_icir"] <= 0 or row["oos_signed_ic"] <= 0:
        return "drop"
    if (
        row["oos_signed_icir"] >= 1.0
        and row["oos_fixed_direction_hl_sharpe"] >= 1.0
    ):
        return "retain"
    return "watch"


def _residual_period_verdict(row: pd.Series) -> str:
    signed_ic = row.get("oos_residual_signed_ic", np.nan)
    signed_icir = row.get("oos_residual_signed_icir", np.nan)
    signed_sharpe = row.get(
        "oos_residual_fixed_direction_hl_sharpe", np.nan
    )
    if not np.isfinite(signed_icir):
        return "not_applicable"
    if signed_icir <= 0 or signed_ic <= 0:
        return "drop"
    if signed_icir >= 1.0 and signed_sharpe >= 1.0:
        return "retain"
    return "watch"


def _aggregate_verdict(verdicts: list[str]) -> str:
    if verdicts and all(item == "retain" for item in verdicts):
        return "retain"
    if "drop" in verdicts:
        return "drop"
    return "watch"


def _wide_summary(metrics: pd.DataFrame, period_order: list[str]) -> pd.DataFrame:
    rows = []
    for factor_name, factor_rows in metrics.groupby("factor", sort=False):
        first = factor_rows.iloc[0]
        row = {
            "factor": factor_name,
            "family": first["family"],
            "bartime": first["bartime"],
            "horizon": first["horizon"],
            "direction": int(first["direction"]),
            "portfolio_rule": first["portfolio_rule"],
            "train_IC": first["train_raw_ic"],
            "train_ICIR": first["train_raw_icir"],
            "train_sharpe": first["train_fixed_direction_sharpe"],
            "legacy_turnover_model": first["legacy_turnover_model"],
        }
        raw_verdicts = []
        residual_verdicts = []
        for period_name in period_order:
            period_row = factor_rows[factor_rows["period"] == period_name]
            if period_row.empty:
                continue
            value = period_row.iloc[0]
            prefix = f"OOS_{period_name}"
            row[f"{prefix}_IC"] = value["oos_raw_ic"]
            row[f"{prefix}_ICIR"] = value["oos_raw_icir"]
            row[f"{prefix}_signed_ICIR"] = value["oos_signed_icir"]
            row[f"{prefix}_sharpe"] = value[
                "oos_fixed_direction_hl_sharpe"
            ]
            row[f"{prefix}_legacy_model_cost_sharpe"] = value[
                "legacy_model_cost_sharpe_7p5bps"
            ]
            row[f"{prefix}_residual_ICIR"] = value.get(
                "oos_residual_raw_icir", np.nan
            )
            row[f"{prefix}_residual_signed_ICIR"] = value.get(
                "oos_residual_signed_icir", np.nan
            )
            row[f"{prefix}_n_dates"] = int(value["n_dates"])
            raw_verdicts.append(_period_verdict(value))
            residual_verdict = _residual_period_verdict(value)
            if residual_verdict != "not_applicable":
                residual_verdicts.append(residual_verdict)
        row["raw_conclusion"] = _aggregate_verdict(raw_verdicts)
        row["residual_conclusion"] = (
            _aggregate_verdict(residual_verdicts)
            if residual_verdicts
            else "not_applicable"
        )
        if "drop" in (row["raw_conclusion"], row["residual_conclusion"]):
            row["conclusion"] = "drop"
        elif row["raw_conclusion"] == "retain" and row[
            "residual_conclusion"
        ] in ("retain", "not_applicable"):
            row["conclusion"] = "retain"
        else:
            row["conclusion"] = "watch"
        row["portfolio_cost_status"] = (
            "superseded_by_intraday_portfolio_simulator_v1"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _required_production_slots(freeze: dict) -> dict[str, set[str]]:
    required = {
        factor_name: {str(freeze["factors"][factor_name]["bartime"])}
        for factor_name in PRODUCTION_FACTORS
    }
    for factor_name in DISCOVERY_FACTORS:
        spec = freeze["factors"][factor_name]
        for control_name in spec["residual_controls"]:
            required[control_name].add(str(spec["bartime"]))
    return required


def _run_period(
    session,
    freeze: dict,
    period_name: str,
    start: str,
    end: str,
    chunk_months: int,
) -> tuple[list[dict], list[dict]]:
    print(f"[PERIOD] {period_name}: {start} → {end}", flush=True)
    required_slots = _required_production_slots(freeze)
    production_signals: Dict[str, pd.DataFrame] = {}
    metrics = []
    decay_rows = []

    for factor_name in PRODUCTION_FACTORS:
        spec = freeze["factors"][factor_name]
        print(
            f"[BUILD] {factor_name} slots={sorted(required_slots[factor_name])}",
            flush=True,
        )
        signal = _filter_slots(
            _build_factor_chunked(
                factor_name,
                start,
                end,
                chunk_months,
            ),
            required_slots[factor_name],
        )
        filtered, group_ret, ic_mean, ic_ts = _evaluate(
            session,
            f"oos_{period_name}_{factor_name}",
            signal,
            apply_limit_filter=True,
        )
        filtered["factorname"] = factor_name
        production_signals[factor_name] = filtered
        metrics.append(
            _fixed_metric(
                factor_name,
                period_name,
                spec,
                group_ret,
                ic_mean,
                ic_ts,
            )
        )
        decay_rows.extend(
            _ic_decay(
                factor_name,
                period_name,
                spec,
                ic_mean,
                ic_ts,
            )
        )

    for factor_name in DISCOVERY_FACTORS:
        spec = freeze["factors"][factor_name]
        bartime = str(spec["bartime"])
        print(
            f"[BUILD] {factor_name} frozen={bartime}/{spec['horizon']} "
            f"direction={spec['direction']}",
            flush=True,
        )
        signal = _filter_slots(
            _build_factor_chunked(
                factor_name,
                start,
                end,
                chunk_months,
            ),
            {bartime},
        )
        filtered, group_ret, ic_mean, ic_ts = _evaluate(
            session,
            f"oos_{period_name}_{factor_name}",
            signal,
            apply_limit_filter=True,
        )
        filtered["factorname"] = factor_name
        metric = _fixed_metric(
            factor_name,
            period_name,
            spec,
            group_ret,
            ic_mean,
            ic_ts,
        )
        decay_rows.extend(
            _ic_decay(
                factor_name,
                period_name,
                spec,
                ic_mean,
                ic_ts,
            )
        )
        residual = _residualize_frozen(
            factor_name,
            filtered,
            production_signals,
            list(spec["residual_controls"]),
            bartime,
        )
        _, resid_group, resid_mean, resid_ts = _evaluate(
            session,
            f"oos_{period_name}_{factor_name}_resid_frozen",
            residual,
            apply_limit_filter=False,
        )
        residual_spec = dict(spec)
        residual_spec["direction"] = int(spec["residual_direction"])
        residual_spec["portfolio_rule"] = (
            "long_high_short_low"
            if residual_spec["direction"] == 1
            else "long_low_short_high"
        )
        residual_metric = _fixed_metric(
            f"{factor_name}_resid_frozen",
            period_name,
            residual_spec,
            resid_group,
            resid_mean,
            resid_ts,
        )
        metric.update(
            {
                "residual_controls": "|".join(spec["residual_controls"]),
                "residual_direction": int(spec["residual_direction"]),
                "oos_residual_raw_ic": residual_metric["oos_raw_ic"],
                "oos_residual_signed_ic": residual_metric["oos_signed_ic"],
                "oos_residual_raw_icir": residual_metric["oos_raw_icir"],
                "oos_residual_signed_icir": residual_metric[
                    "oos_signed_icir"
                ],
                "oos_residual_fixed_direction_hl_sharpe": residual_metric[
                    "oos_fixed_direction_hl_sharpe"
                ],
                "legacy_residual_model_cost_sharpe_7p5bps": residual_metric[
                    "legacy_model_cost_sharpe_7p5bps"
                ],
            }
        )
        metrics.append(metric)
    return metrics, decay_rows


def _checkpoint_paths(output: Path, period_name: str) -> tuple[Path, Path, Path]:
    period_dir = output / "periods" / period_name
    return (
        period_dir / "metrics.csv",
        period_dir / "decay.csv",
        period_dir / "metadata.json",
    )


def _write_checkpoint(
    output: Path,
    period_name: str,
    freeze_sha256: str,
    metrics: list[dict],
    decay_rows: list[dict],
) -> None:
    metrics_path, decay_path, metadata_path = _checkpoint_paths(
        output, period_name
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.DataFrame(decay_rows).to_csv(decay_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "period": period_name,
                "freeze_sha256": freeze_sha256,
                "complete": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_checkpoint(
    output: Path,
    period_name: str,
    freeze_sha256: str,
) -> tuple[list[dict], list[dict]] | None:
    metrics_path, decay_path, metadata_path = _checkpoint_paths(
        output, period_name
    )
    if not all(
        path.exists() for path in (metrics_path, decay_path, metadata_path)
    ):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("freeze_sha256") != freeze_sha256
        or not metadata.get("complete")
    ):
        raise ValueError(f"{period_name}: stale or incomplete checkpoint")
    return (
        pd.read_csv(metrics_path).to_dict(orient="records"),
        pd.read_csv(decay_path).to_dict(orient="records"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--period",
        action="append",
        help=(
            "Frozen period name. Repeat to select periods; defaults to all "
            "periods in the freeze spec."
        ),
    )
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=6,
        help="Maximum DDB factor-query span; default 6 months.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete period checkpoints with the same freeze hash.",
    )
    args = parser.parse_args()
    freeze = verify_spec(args.freeze)
    available_periods = freeze["oos_periods"]
    period_order = args.period or list(available_periods)
    unknown = [name for name in period_order if name not in available_periods]
    if unknown:
        raise ValueError(f"Periods are not frozen in the spec: {unknown}")
    if args.chunk_months < 1:
        raise ValueError("--chunk-months must be positive")
    train_end = pd.Timestamp(freeze["train_period"]["end"])
    for name in period_order:
        if pd.Timestamp(available_periods[name]["start"]) <= train_end:
            raise ValueError(f"{name} overlaps the training period")

    args.output.mkdir(parents=True, exist_ok=True)
    session = _connect()
    metrics = []
    decay_rows = []
    for period_name in period_order:
        period = available_periods[period_name]
        checkpoint = (
            _read_checkpoint(
                args.output,
                period_name,
                freeze["spec_sha256"],
            )
            if args.resume
            else None
        )
        if checkpoint is not None:
            print(f"[RESUME] {period_name}", flush=True)
            period_metrics, period_decay = checkpoint
            metrics.extend(period_metrics)
            decay_rows.extend(period_decay)
            continue
        period_metrics, period_decay = _run_period(
            session,
            freeze,
            period_name,
            period["start"],
            period["end"],
            args.chunk_months,
        )
        _write_checkpoint(
            args.output,
            period_name,
            freeze["spec_sha256"],
            period_metrics,
            period_decay,
        )
        metrics.extend(period_metrics)
        decay_rows.extend(period_decay)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame = metrics_frame.rename(
        columns={
            "turnover_b_hl": "legacy_turnover_model",
            "oos_cost_sharpe_7p5bps": (
                "legacy_model_cost_sharpe_7p5bps"
            ),
            "oos_residual_cost_sharpe_7p5bps": (
                "legacy_residual_model_cost_sharpe_7p5bps"
            ),
        }
    )
    metrics_frame["period_verdict"] = metrics_frame.apply(
        _period_verdict, axis=1
    )
    metrics_frame["residual_period_verdict"] = metrics_frame.apply(
        _residual_period_verdict, axis=1
    )
    wide = _wide_summary(metrics_frame, period_order)
    decay_frame = pd.DataFrame(decay_rows)
    metrics_frame.to_csv(args.output / "oos_metrics_long.csv", index=False)
    wide.to_csv(args.output / "intraday_alpha_oos_v1.csv", index=False)
    decay_frame.to_csv(
        args.output / "oos_ic_decay_diagnostic.csv",
        index=False,
    )
    summary = {
        "freeze_id": freeze["freeze_id"],
        "freeze_sha256": freeze["spec_sha256"],
        "locked_dimensions": freeze["locked_dimensions"],
        "periods": {
            name: available_periods[name] for name in period_order
        },
        "direction_policy": (
            "Always multiply raw IC and raw high-minus-low returns by the "
            "2024H1 frozen direction; never infer direction from OOS returns"
        ),
        "decay_policy": (
            "Non-frozen horizons are diagnostics only and cannot change the "
            "frozen horizon or conclusion"
        ),
        "cost_policy": (
            "Fixed turnover-4 fields are legacy display-model diagnostics. "
            "Use research/results/intraday_portfolio_simulator_v1 for "
            "position-level turnover and transaction costs."
        ),
        "results": wide.where(pd.notna(wide), None).to_dict(
            orient="records"
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Frozen OOS evaluation → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
