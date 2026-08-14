#!/usr/bin/env python3
"""Regenerate canonical intraday metrics with exact market-excess groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.intraday_metrics import (  # noqa: E402
    ANNUALIZATION_DAYS,
    build_group_excess_panel,
    build_hl_panel,
    summarize_cross_sectional_metrics,
    summarize_ic_series,
)
from research.freeze_intraday_alpha_v1 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_FREEZE,
    verify_spec,
)
from research.run_intraday_alpha_discovery_v1 import (  # noqa: E402
    DISCOVERY_FACTORS,
    EVALUATION_WINDOWS,
    FACTOR_FAMILY,
    PRODUCTION_FACTORS,
)
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    RET_MATRIX,
    _bt_string,
    _connect,
    _evaluate,
)
from research.run_intraday_alpha_oos_v1 import (  # noqa: E402
    _build_factor_chunked,
    _filter_slots,
)

DEFAULT_OUTPUT = ROOT / "research/results/intraday_evaluation_v2"
ALL_FACTORS = PRODUCTION_FACTORS + DISCOVERY_FACTORS
GROUP_NUM = 10
SPEC_VERSION = "intraday_evaluation_metrics_v2"


def _evaluation_hash(freeze_sha256: str) -> str:
    payload = {
        "version": SPEC_VERSION,
        "freeze_sha256": freeze_sha256,
        "annualization_days": ANNUALIZATION_DAYS,
        "market_return": "exact_filtered_constituent_equal_weight",
        "raw_hl": "G10_minus_G1",
        "beta_policy": "diagnostic_only",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_exact_group_and_market(
    session,
    factor_name: str,
    filtered_signal: pd.DataFrame,
    horizons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for horizon in horizons:
        if not re.fullmatch(r"Ret_[A-Za-z0-9]+", horizon):
            raise ValueError(f"Unsafe return column: {horizon}")
    upload_name = "evalv2_" + re.sub(r"[^A-Za-z0-9_]", "_", factor_name)
    session.upload({upload_name: filtered_signal})
    ret_symbols = "`" + "`".join(horizons)
    session.run(
        f"""
v2Signal = {upload_name}
v2Base = select symbol as Symbol, date(tradetime) as Date,
    second(tradetime) as Bartime, tradetime, value as Signal_Value
    from v2Signal
v2Ranked = select Symbol, Date, Bartime, tradetime, Signal_Value,
    "group_" + string(rank(Signal_Value, groupNum={GROUP_NUM})) as factor_group
    from v2Base
    context by tradetime
v2Joined = ej(v2Ranked, {RET_MATRIX}, `Symbol`Date`Bartime)
v2RetCols = {ret_symbols}
v2Long = v2Joined.unpivot(
    keyColNames=`Symbol`Date`Bartime`factor_group,
    valueColNames=v2RetCols
)
v2Market = select count(value) as n_market_assets,
    avg(value) as market_return
    from v2Long
    where isValid(value)
    group by Date, Bartime, valueType
v2Groups = select count(value) as n_assets,
    avg(value) as group_return_raw
    from v2Long
    where isValid(value)
    group by Date, Bartime, valueType, factor_group
"""
    )
    market = pd.DataFrame(session.run("v2Market")).rename(
        columns={"valueType": "return_window"}
    )
    groups = pd.DataFrame(session.run("v2Groups")).rename(
        columns={
            "valueType": "return_window",
            "factor_group": "group",
        }
    )
    for frame in (market, groups):
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["Bartime"] = _bt_string(frame["Bartime"])
        frame["return_window"] = frame["return_window"].astype(str)
    return groups, market


def _ic_lookup(
    ic_ts: pd.DataFrame,
    *,
    bartime: str,
    return_window: str,
    direction: int,
) -> dict:
    frame = ic_ts.copy()
    frame["bartime_key"] = _bt_string(frame["Bartime"])
    ret_key = "RetType" if "RetType" in frame.columns else "valueType"
    selected = frame[
        (frame["bartime_key"] == bartime)
        & (frame[ret_key].astype(str) == return_window)
    ]
    return summarize_ic_series(
        selected["Rank_IC"],
        direction=direction,
    )


def _summarize_factor_period(
    *,
    factor_name: str,
    period_name: str,
    group_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    ic_ts: pd.DataFrame,
    frozen_direction: int | None,
) -> pd.DataFrame:
    panel = build_group_excess_panel(group_returns, market_returns)
    parts = []
    for (bartime, return_window), combo in panel.groupby(
        ["Bartime", "return_window"],
        sort=True,
    ):
        provisional_ic = _ic_lookup(
            ic_ts,
            bartime=str(bartime),
            return_window=str(return_window),
            direction=1,
        )
        direction = (
            int(frozen_direction)
            if frozen_direction is not None
            else (1 if provisional_ic["rank_ic"] >= 0 else -1)
        )
        ic_metrics = _ic_lookup(
            ic_ts,
            bartime=str(bartime),
            return_window=str(return_window),
            direction=direction,
        )
        hl = build_hl_panel(combo, direction=direction)
        metrics = summarize_cross_sectional_metrics(
            combo,
            hl,
            factor_name=factor_name,
        )
        metrics["period"] = period_name
        metrics["rank_ic"] = ic_metrics["rank_ic"]
        metrics["annualized_icir"] = ic_metrics["annualized_icir"]
        metrics["ic_win_rate"] = ic_metrics["ic_win_rate"]
        parts.append(metrics)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _checkpoint_paths(
    output: Path,
    factor_name: str,
    period_name: str,
) -> tuple[Path, Path]:
    checkpoint = output / "checkpoints" / factor_name / period_name
    return checkpoint / "metrics.csv", checkpoint / "metadata.json"


def _write_checkpoint(
    output: Path,
    factor_name: str,
    period_name: str,
    metrics: pd.DataFrame,
    evaluation_sha256: str,
) -> None:
    metrics_path, metadata_path = _checkpoint_paths(
        output,
        factor_name,
        period_name,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "complete": True,
                "evaluation_sha256": evaluation_sha256,
                "rows": int(len(metrics)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_checkpoint(
    output: Path,
    factor_name: str,
    period_name: str,
    evaluation_sha256: str,
) -> pd.DataFrame | None:
    metrics_path, metadata_path = _checkpoint_paths(
        output,
        factor_name,
        period_name,
    )
    if not metrics_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not metadata.get("complete")
        or metadata.get("evaluation_sha256") != evaluation_sha256
    ):
        raise ValueError(
            f"{factor_name}/{period_name}: stale evaluation checkpoint"
        )
    return pd.read_csv(metrics_path)


def _periods(freeze: dict) -> dict[str, dict]:
    return {
        "train_2024H1": freeze["train_period"],
        "validation_2024H2": freeze["oos_periods"]["validation_2024H2"],
        "test_2025_available": freeze["oos_periods"][
            "test_2025_available"
        ],
    }


def _run_factor_period(
    session,
    freeze: dict,
    *,
    factor_name: str,
    period_name: str,
    period: dict,
    chunk_months: int,
) -> pd.DataFrame:
    spec = freeze["factors"][factor_name]
    is_train = period_name == "train_2024H1"
    horizons = EVALUATION_WINDOWS if is_train else [str(spec["horizon"])]
    print(
        f"[EVAL V2] {factor_name}/{period_name} "
        f"horizons={horizons}",
        flush=True,
    )
    signal = _build_factor_chunked(
        factor_name,
        period["start"],
        period["end"],
        chunk_months,
    )
    if not is_train:
        signal = _filter_slots(signal, {str(spec["bartime"])})
    filtered, _, _, ic_ts = _evaluate(
        session,
        f"evalv2_{factor_name}_{period_name}",
        signal,
        apply_limit_filter=True,
    )
    groups, market = _aggregate_exact_group_and_market(
        session,
        f"{factor_name}_{period_name}",
        filtered,
        horizons,
    )
    return _summarize_factor_period(
        factor_name=factor_name,
        period_name=period_name,
        group_returns=groups,
        market_returns=market,
        ic_ts=ic_ts,
        frozen_direction=(None if is_train else int(spec["direction"])),
    )


def _select_row(
    metrics: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    bartime: str,
    horizon: str,
    group: str,
) -> pd.Series | None:
    selected = metrics[
        (metrics["factor"] == factor_name)
        & (metrics["period"] == period_name)
        & (metrics["bartime"].astype(str) == bartime)
        & (metrics["return_window"].astype(str) == horizon)
        & (metrics["group"] == group)
    ]
    return None if selected.empty else selected.iloc[0]


def _candidate_v3(
    metrics: pd.DataFrame,
    freeze: dict,
    execution: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    for factor_name in ALL_FACTORS:
        spec = freeze["factors"][factor_name]
        bartime = str(spec["bartime"])
        horizon = str(spec["horizon"])
        hl = _select_row(
            metrics,
            factor_name=factor_name,
            period_name="train_2024H1",
            bartime=bartime,
            horizon=horizon,
            group="H-L",
        )
        g1 = _select_row(
            metrics,
            factor_name=factor_name,
            period_name="train_2024H1",
            bartime=bartime,
            horizon=horizon,
            group="G1",
        )
        g10 = _select_row(
            metrics,
            factor_name=factor_name,
            period_name="train_2024H1",
            bartime=bartime,
            horizon=horizon,
            group="G10",
        )
        if hl is None or g1 is None or g10 is None:
            continue
        row = {
            "factor": factor_name,
            "family": FACTOR_FAMILY[factor_name],
            "backend": spec["backend"],
            "research_role": spec["research_role"],
            "bartime": bartime,
            "horizon": horizon,
            "direction": int(spec["direction"]),
            "rank_ic": hl["rank_ic"],
            "annualized_icir": hl["annualized_icir"],
            "ic_win_rate": hl["ic_win_rate"],
            "g1_excess_sharpe": g1["group_excess_sharpe"],
            "g10_excess_sharpe": g10["group_excess_sharpe"],
            "hl_sharpe": hl["hl_sharpe"],
            "hl_annualized_return": hl["hl_annualized_return"],
            "hl_market_beta": hl["hl_market_beta"],
            "hl_market_corr": hl["hl_market_corr"],
            "direction_consistent": hl["direction_consistent"],
        }
        if execution is not None and factor_name in set(execution["factor"]):
            ex = execution[execution["factor"] == factor_name].iloc[0]
            row.update(
                {
                    "test_2025_net_ls_sharpe": ex.get(
                        "test_2025_available_net_ls_sharpe", np.nan
                    ),
                    "test_2025_break_even_cost_bps": ex.get(
                        "test_2025_available_break_even_one_way_cost_bps",
                        np.nan,
                    ),
                    "test_2025_cost_headroom_bps": ex.get(
                        "test_2025_available_cost_headroom_bps", np.nan
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _oos_diagnostics(
    metrics: pd.DataFrame,
    freeze: dict,
) -> pd.DataFrame:
    rows = []
    for factor_name in ALL_FACTORS:
        spec = freeze["factors"][factor_name]
        for period_name in ("validation_2024H2", "test_2025_available"):
            row = _select_row(
                metrics,
                factor_name=factor_name,
                period_name=period_name,
                bartime=str(spec["bartime"]),
                horizon=str(spec["horizon"]),
                group="H-L",
            )
            if row is not None:
                rows.append(row.to_dict())
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--factor", action="append")
    parser.add_argument("--period", action="append")
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    freeze = verify_spec(args.freeze)
    factor_order = args.factor or ALL_FACTORS
    period_map = _periods(freeze)
    period_order = args.period or list(period_map)
    if any(name not in ALL_FACTORS for name in factor_order):
        raise ValueError("Requested factor is outside the frozen v2 library")
    if any(name not in period_map for name in period_order):
        raise ValueError("Requested period is outside the v2 specification")
    if args.chunk_months < 1:
        raise ValueError("Chunk months must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_sha256 = _evaluation_hash(freeze["spec_sha256"])
    session = _connect()
    parts = []
    for factor_name in factor_order:
        for period_name in period_order:
            checkpoint = (
                _read_checkpoint(
                    args.output,
                    factor_name,
                    period_name,
                    evaluation_sha256,
                )
                if args.resume
                else None
            )
            if checkpoint is not None:
                print(f"[RESUME V2] {factor_name}/{period_name}", flush=True)
                parts.append(checkpoint)
                continue
            metrics = _run_factor_period(
                session,
                freeze,
                factor_name=factor_name,
                period_name=period_name,
                period=period_map[period_name],
                chunk_months=args.chunk_months,
            )
            _write_checkpoint(
                args.output,
                factor_name,
                period_name,
                metrics,
                evaluation_sha256,
            )
            parts.append(metrics)

    performance = pd.concat(parts, ignore_index=True)
    is_complete_library = (
        set(factor_order) == set(ALL_FACTORS)
        and set(period_order) == set(period_map)
    )
    if not is_complete_library:
        print(
            "Partial evaluation checkpoints updated; canonical v2 artifacts "
            "were not overwritten.",
            flush=True,
        )
        return 0
    performance.to_csv(args.output / "performance_all_v2.csv", index=False)
    execution_path = (
        ROOT
        / "research/results/intraday_portfolio_simulator_v1"
        / "intraday_portfolio_cost_v2.csv"
    )
    execution = (
        pd.read_csv(execution_path) if execution_path.exists() else None
    )
    candidates = _candidate_v3(performance, freeze, execution)
    candidates.to_csv(
        args.output / "intraday_alpha_library_v3_candidates.csv",
        index=False,
    )
    oos = _oos_diagnostics(performance, freeze)
    oos.to_csv(args.output / "frozen_oos_diagnostics_v2.csv", index=False)
    summary = {
        "spec_version": SPEC_VERSION,
        "evaluation_sha256": evaluation_sha256,
        "freeze_sha256": freeze["spec_sha256"],
        "annualization_days": ANNUALIZATION_DAYS,
        "market_return": "exact_filtered_constituent_equal_weight",
        "raw_hl": "G10_minus_G1",
        "beta_policy": "diagnostic_only",
        "rows": int(len(performance)),
        "candidate_rows": int(len(candidates)),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Unified intraday evaluation v2 → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
