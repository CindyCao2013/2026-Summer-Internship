#!/usr/bin/env python3
"""Position-level cost simulation for frozen intraday alpha tuples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Factor_Dev_Lib as fdl  # noqa: E402
from research.freeze_intraday_alpha_v1 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_FREEZE,
    verify_spec,
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

DEFAULT_OUTPUT = ROOT / "research/results/intraday_portfolio_simulator_v1"
DEFAULT_FACTORS = [
    "close_vwap_deviation",
    "intraday_amihud",
    "realized_volatility",
    "minute_skew",
]
ONE_WAY_COST_BPS = 7.5
LONG_GROSS = 0.5
SHORT_GROSS = 0.5
GROUP_NUM = 10


def _simulation_hash(freeze_sha256: str, cost_bps: float) -> str:
    payload = {
        "freeze_sha256": freeze_sha256,
        "one_way_cost_bps": float(cost_bps),
        "long_gross": LONG_GROSS,
        "short_gross": SHORT_GROSS,
        "group_num": GROUP_NUM,
        "holding_rule": "flat_at_frozen_horizon",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fetch_extreme_constituents(
    session,
    factor_name: str,
    filtered_signal: pd.DataFrame,
    horizon: str,
) -> pd.DataFrame:
    if not re.fullmatch(r"Ret_[A-Za-z0-9]+", horizon):
        raise ValueError(f"Unsafe frozen return column: {horizon}")
    upload_name = "portfolio_" + re.sub(r"[^A-Za-z0-9_]", "_", factor_name)
    session.upload({upload_name: filtered_signal})
    result = session.run(
        f"""
portfolioSignal = {upload_name}
portfolioBase = select symbol as Symbol, date(tradetime) as Date,
    second(tradetime) as Bartime, tradetime, value as Signal_Value
    from portfolioSignal
portfolioRanked = select Symbol, Date, Bartime, tradetime, Signal_Value,
    "group_" + string(rank(Signal_Value, groupNum={GROUP_NUM})) as factor_group
    from portfolioBase
    context by tradetime
portfolioJoined = ej(portfolioRanked, {RET_MATRIX}, `Symbol`Date`Bartime)
portfolioResult = select Symbol, Date, Bartime, tradetime, Signal_Value,
    factor_group, {horizon} as asset_return
    from portfolioJoined
    where factor_group in `group_0`group_{GROUP_NUM - 1}
      and isValid({horizon})
portfolioResult
"""
    )
    out = pd.DataFrame(result)
    if out.empty:
        raise ValueError(f"{factor_name}: no valid extreme-decile returns")
    out["Date"] = pd.to_datetime(out["Date"])
    out["tradetime"] = pd.to_datetime(out["tradetime"])
    out["asset_return"] = pd.to_numeric(
        out["asset_return"], errors="coerce"
    )
    out = out.dropna(subset=["asset_return"])
    if (out["asset_return"] <= -1).any():
        raise ValueError(f"{factor_name}: asset return <= -100%")
    return out


def _build_positions(
    day: pd.DataFrame,
    direction: int,
) -> pd.DataFrame:
    if direction not in (-1, 1):
        raise ValueError(f"Invalid frozen direction: {direction}")
    long_group = "group_9" if direction == 1 else "group_0"
    short_group = "group_0" if direction == 1 else "group_9"
    long_leg = day[day["factor_group"] == long_group].copy()
    short_leg = day[day["factor_group"] == short_group].copy()
    if long_leg.empty or short_leg.empty:
        raise ValueError("Both frozen-direction extreme deciles are required")
    long_leg["side"] = "long"
    long_leg["entry_weight"] = LONG_GROSS / len(long_leg)
    short_leg["side"] = "short"
    short_leg["entry_weight"] = -SHORT_GROSS / len(short_leg)
    positions = pd.concat([long_leg, short_leg], ignore_index=True)
    if not np.isclose(
        positions.loc[positions["side"] == "long", "entry_weight"].sum(),
        LONG_GROSS,
    ):
        raise AssertionError("Long weights do not sum to +50%")
    if not np.isclose(
        positions.loc[positions["side"] == "short", "entry_weight"].sum(),
        -SHORT_GROSS,
    ):
        raise AssertionError("Short weights do not sum to -50%")
    return positions


def _simulate_day(
    positions: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    horizon: str,
    direction: int,
    one_way_cost_bps: float,
) -> dict:
    weights = positions["entry_weight"].to_numpy(dtype=float)
    returns = positions["asset_return"].to_numpy(dtype=float)
    gross_ls_return = float(np.dot(weights, returns))
    entry_traded_notional = float(np.abs(weights).sum())
    exit_values = np.abs(weights) * (1.0 + returns)
    exit_traded_notional = float(exit_values.sum())
    traded_notional_turnover = (
        entry_traded_notional + exit_traded_notional
    )
    half_l1_turnover = 0.5 * traded_notional_turnover
    transaction_cost = traded_notional_turnover * one_way_cost_bps / 1e4
    date = pd.Timestamp(positions["Date"].iloc[0])
    bartime = _bt_string(positions["Bartime"]).iloc[0]
    return {
        "factor": factor_name,
        "period": period_name,
        "Date": date,
        "bartime": bartime,
        "horizon": horizon,
        "direction": direction,
        "long_count": int((positions["side"] == "long").sum()),
        "short_count": int((positions["side"] == "short").sum()),
        "long_gross": LONG_GROSS,
        "short_gross": SHORT_GROSS,
        "entry_traded_notional": entry_traded_notional,
        "exit_traded_notional": exit_traded_notional,
        "traded_notional_turnover": traded_notional_turnover,
        "half_l1_turnover": half_l1_turnover,
        "one_way_cost_bps": float(one_way_cost_bps),
        "transaction_cost": transaction_cost,
        "gross_ls_return": gross_ls_return,
        "net_ls_return": gross_ls_return - transaction_cost,
    }


def _simulate_ledger(
    constituents: pd.DataFrame,
    *,
    factor_name: str,
    period_name: str,
    horizon: str,
    direction: int,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    rows = []
    for _, day in constituents.groupby("tradetime", sort=True):
        positions = _build_positions(day, direction)
        rows.append(
            _simulate_day(
                positions,
                factor_name=factor_name,
                period_name=period_name,
                horizon=horizon,
                direction=direction,
                one_way_cost_bps=one_way_cost_bps,
            )
        )
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def _expected_hl(
    group_ret: pd.DataFrame,
    *,
    bartime: str,
    horizon: str,
    direction: int,
) -> pd.DataFrame:
    frame = group_ret.copy()
    frame["bartime_key"] = _bt_string(frame["Bartime"])
    hl = frame[
        (frame["group"] == "group_HML")
        & (frame["bartime_key"] == bartime)
    ][["Date", horizon]].copy()
    hl["Date"] = pd.to_datetime(hl["Date"])
    hl["expected_gross_return"] = (
        0.5 * direction * pd.to_numeric(hl[horizon], errors="coerce")
    )
    return hl[["Date", "expected_gross_return"]]


def _assert_gross_parity(
    ledger: pd.DataFrame,
    group_ret: pd.DataFrame,
    *,
    bartime: str,
    horizon: str,
    direction: int,
    tolerance: float = 1e-12,
) -> float:
    expected = _expected_hl(
        group_ret,
        bartime=bartime,
        horizon=horizon,
        direction=direction,
    )
    aligned = ledger[["Date", "gross_ls_return"]].merge(
        expected,
        on="Date",
        how="inner",
    )
    if len(aligned) != len(ledger):
        raise AssertionError(
            f"Gross-parity date mismatch: ledger={len(ledger)}, "
            f"aligned={len(aligned)}"
        )
    max_diff = float(
        (
            aligned["gross_ls_return"] - aligned["expected_gross_return"]
        )
        .abs()
        .max()
    )
    if not np.isfinite(max_diff) or max_diff > tolerance:
        raise AssertionError(
            f"Gross-return parity failed: {max_diff} > {tolerance}"
        )
    return max_diff


def _performance_summary(
    ledger: pd.DataFrame,
    *,
    freeze_sha256: str,
    simulation_sha256: str,
    max_gross_parity_diff: float,
) -> dict:
    gross = ledger.set_index("Date")["gross_ls_return"]
    net = ledger.set_index("Date")["net_ls_return"]
    gross_mdd, _ = fdl.calMDD(gross)
    net_mdd, _ = fdl.calMDD(net)
    first = ledger.iloc[0]
    break_even_cost_bps = float(
        gross.mean()
        / ledger["traded_notional_turnover"].mean()
        * 1e4
    )
    return {
        "factor": first["factor"],
        "period": first["period"],
        "bartime": first["bartime"],
        "horizon": first["horizon"],
        "direction": int(first["direction"]),
        "n_dates": int(ledger["Date"].nunique()),
        "avg_long_count": float(ledger["long_count"].mean()),
        "avg_short_count": float(ledger["short_count"].mean()),
        "avg_entry_traded_notional": float(
            ledger["entry_traded_notional"].mean()
        ),
        "avg_exit_traded_notional": float(
            ledger["exit_traded_notional"].mean()
        ),
        "avg_traded_notional_turnover": float(
            ledger["traded_notional_turnover"].mean()
        ),
        "avg_half_l1_turnover": float(ledger["half_l1_turnover"].mean()),
        "avg_transaction_cost": float(ledger["transaction_cost"].mean()),
        "one_way_cost_bps": float(first["one_way_cost_bps"]),
        "break_even_one_way_cost_bps": break_even_cost_bps,
        "cost_headroom_bps": (
            break_even_cost_bps - float(first["one_way_cost_bps"])
        ),
        "gross_ls_annualized_return": float(fdl.calAnnuRet(gross)),
        "net_ls_annualized_return": float(fdl.calAnnuRet(net)),
        "gross_ls_sharpe": float(fdl.calSharpe(gross)),
        "net_ls_sharpe": float(fdl.calSharpe(net)),
        "gross_ls_max_drawdown": float(gross_mdd),
        "net_ls_max_drawdown": float(net_mdd),
        "max_gross_parity_diff": max_gross_parity_diff,
        "freeze_sha256": freeze_sha256,
        "simulation_sha256": simulation_sha256,
    }


def _checkpoint_paths(
    output: Path,
    factor_name: str,
    period_name: str,
) -> tuple[Path, Path]:
    checkpoint = output / "checkpoints" / factor_name / period_name
    return checkpoint / "daily_ledger.csv", checkpoint / "metadata.json"


def _write_checkpoint(
    output: Path,
    factor_name: str,
    period_name: str,
    ledger: pd.DataFrame,
    summary: dict,
) -> None:
    ledger_path, metadata_path = _checkpoint_paths(
        output, factor_name, period_name
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(ledger_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {"complete": True, "summary": summary},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _read_checkpoint(
    output: Path,
    factor_name: str,
    period_name: str,
    simulation_sha256: str,
) -> tuple[pd.DataFrame, dict] | None:
    ledger_path, metadata_path = _checkpoint_paths(
        output, factor_name, period_name
    )
    if not ledger_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = metadata.get("summary", {})
    if (
        not metadata.get("complete")
        or summary.get("simulation_sha256") != simulation_sha256
    ):
        raise ValueError(
            f"{factor_name}/{period_name}: stale simulator checkpoint"
        )
    ledger = pd.read_csv(ledger_path)
    ledger["Date"] = pd.to_datetime(ledger["Date"])
    ledger = ledger.rename(
        columns={
            "gross_return": "gross_ls_return",
            "net_return": "net_ls_return",
        }
    )
    legacy_summary_names = {
        "gross_annualized_return": "gross_ls_annualized_return",
        "net_annualized_return": "net_ls_annualized_return",
        "gross_sharpe": "gross_ls_sharpe",
        "net_sharpe": "net_ls_sharpe",
        "gross_max_drawdown": "gross_ls_max_drawdown",
        "net_max_drawdown": "net_ls_max_drawdown",
    }
    for old_name, new_name in legacy_summary_names.items():
        if old_name in summary and new_name not in summary:
            summary[new_name] = summary.pop(old_name)
    if "break_even_one_way_cost_bps" not in summary:
        break_even_cost_bps = float(
            ledger["gross_ls_return"].mean()
            / ledger["traded_notional_turnover"].mean()
            * 1e4
        )
        summary["break_even_one_way_cost_bps"] = break_even_cost_bps
        summary["cost_headroom_bps"] = (
            break_even_cost_bps - float(summary["one_way_cost_bps"])
        )
    return ledger, summary


def _wide_summary(
    summaries: pd.DataFrame,
    period_order: list[str],
) -> pd.DataFrame:
    rows = []
    for factor_name, group in summaries.groupby("factor", sort=False):
        first = group.iloc[0]
        row = {
            "factor": factor_name,
            "bartime": first["bartime"],
            "horizon": first["horizon"],
            "direction": int(first["direction"]),
            "one_way_cost_bps": first["one_way_cost_bps"],
        }
        for period_name in period_order:
            selected = group[group["period"] == period_name]
            if selected.empty:
                continue
            value = selected.iloc[0]
            prefix = period_name
            for field in (
                "avg_half_l1_turnover",
                "avg_traded_notional_turnover",
                "break_even_one_way_cost_bps",
                "cost_headroom_bps",
                "gross_ls_annualized_return",
                "net_ls_annualized_return",
                "gross_ls_sharpe",
                "net_ls_sharpe",
                "gross_ls_max_drawdown",
                "net_ls_max_drawdown",
                "n_dates",
            ):
                row[f"{prefix}_{field}"] = value[field]
        net_sharpes = group["net_ls_sharpe"].dropna()
        row["net_profitable_all_periods"] = bool(
            len(net_sharpes) == len(period_order)
            and (group["net_ls_annualized_return"] > 0).all()
        )
        row["net_ls_sharpe_positive_all_periods"] = bool(
            len(net_sharpes) == len(period_order)
            and (net_sharpes > 0).all()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _periods(freeze: dict) -> Dict[str, dict]:
    return {
        "train_2024H1": freeze["train_period"],
        "validation_2024H2": freeze["oos_periods"]["validation_2024H2"],
        "test_2025_available": freeze["oos_periods"][
            "test_2025_available"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--factor", action="append")
    parser.add_argument("--period", action="append")
    parser.add_argument("--one-way-cost-bps", type=float, default=ONE_WAY_COST_BPS)
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    freeze = verify_spec(args.freeze)
    factor_order = args.factor or DEFAULT_FACTORS
    unknown_factors = [
        name for name in factor_order if name not in freeze["factors"]
    ]
    if unknown_factors:
        raise ValueError(f"Factors are not frozen: {unknown_factors}")
    available_periods = _periods(freeze)
    period_order = args.period or list(available_periods)
    unknown_periods = [
        name for name in period_order if name not in available_periods
    ]
    if unknown_periods:
        raise ValueError(f"Unknown periods: {unknown_periods}")
    if args.one_way_cost_bps < 0:
        raise ValueError("One-way cost cannot be negative")
    if args.chunk_months < 1:
        raise ValueError("Chunk months must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    simulation_sha256 = _simulation_hash(
        freeze["spec_sha256"],
        args.one_way_cost_bps,
    )
    session = _connect()
    ledgers = []
    summaries = []
    for factor_name in factor_order:
        spec = freeze["factors"][factor_name]
        for period_name in period_order:
            checkpoint = (
                _read_checkpoint(
                    args.output,
                    factor_name,
                    period_name,
                    simulation_sha256,
                )
                if args.resume
                else None
            )
            if checkpoint is not None:
                print(f"[RESUME] {factor_name}/{period_name}", flush=True)
                ledger, summary = checkpoint
                ledgers.append(ledger)
                summaries.append(summary)
                continue
            period = available_periods[period_name]
            print(
                f"[SIMULATE] {factor_name}/{period_name} "
                f"{spec['bartime']}/{spec['horizon']} "
                f"direction={spec['direction']}",
                flush=True,
            )
            signal = _filter_slots(
                _build_factor_chunked(
                    factor_name,
                    period["start"],
                    period["end"],
                    args.chunk_months,
                ),
                {str(spec["bartime"])},
            )
            filtered, group_ret, _, _ = _evaluate(
                session,
                f"cost_{factor_name}_{period_name}",
                signal,
                apply_limit_filter=True,
            )
            constituents = _fetch_extreme_constituents(
                session,
                f"{factor_name}_{period_name}",
                filtered,
                str(spec["horizon"]),
            )
            ledger = _simulate_ledger(
                constituents,
                factor_name=factor_name,
                period_name=period_name,
                horizon=str(spec["horizon"]),
                direction=int(spec["direction"]),
                one_way_cost_bps=args.one_way_cost_bps,
            )
            max_diff = _assert_gross_parity(
                ledger,
                group_ret,
                bartime=str(spec["bartime"]),
                horizon=str(spec["horizon"]),
                direction=int(spec["direction"]),
            )
            summary = _performance_summary(
                ledger,
                freeze_sha256=freeze["spec_sha256"],
                simulation_sha256=simulation_sha256,
                max_gross_parity_diff=max_diff,
            )
            _write_checkpoint(
                args.output,
                factor_name,
                period_name,
                ledger,
                summary,
            )
            ledgers.append(ledger)
            summaries.append(summary)

    ledger_frame = pd.concat(ledgers, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    wide = _wide_summary(summary_frame, period_order)
    ledger_frame.to_csv(
        args.output / "daily_portfolio_ledger_v2.csv",
        index=False,
    )
    summary_frame.to_csv(
        args.output / "factor_period_summary_v2.csv",
        index=False,
    )
    wide.to_csv(
        args.output / "intraday_portfolio_cost_v2.csv",
        index=False,
    )
    run_summary = {
        "freeze_sha256": freeze["spec_sha256"],
        "simulation_sha256": simulation_sha256,
        "portfolio": {
            "long_gross": LONG_GROSS,
            "short_gross": SHORT_GROSS,
            "holding_rule": "flat_at_frozen_horizon",
            "one_way_cost_bps": args.one_way_cost_bps,
            "cost_formula": (
                "traded_notional_turnover * one_way_cost_bps / 10000"
            ),
            "turnover_conventions": {
                "half_l1": "0.5 * sum(abs(all entry and exit trades))",
                "traded_notional": "sum(abs(all entry and exit trades))",
            },
            "execution_warning": (
                "Synthetic market-neutral feasibility test; A-share T+1 and "
                "short-borrow availability are not modeled."
            ),
        },
        "results": wide.where(pd.notna(wide), None).to_dict(orient="records"),
    }
    (args.output / "summary_v2.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Intraday portfolio cost v2 schema → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
