#!/usr/bin/env python3
"""Intraday Alpha Library v1 density audit.

Audits five production DDB-native factors on the unchanged ZZ1000 intraday
evaluation layer. Correlations are computed only within matching execution
slots; ABSI is residualized cross-sectionally against LSS at 09:59.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Factor_Dev_Lib as fdl  # noqa: E402
import intraday_lib  # noqa: E402
from COMMON_CONST import DATA_DB_CONN  # noqa: E402
from factors.intraday.active_buy_sell_imbalance.compute import (  # noqa: E402
    ddb_version as abs_imbalance_ddb,
)
from factors.intraday.close_vwap_deviation.compute import (  # noqa: E402
    ddb_version as close_vwap_ddb,
)
from factors.intraday.late_session_strength.compute import (  # noqa: E402
    ddb_version as late_strength_ddb,
)
from factors.intraday.volume_back_loading.compute import (  # noqa: E402
    ddb_version as volume_back_ddb,
)
from factors.intraday.volume_front_loading.compute import (  # noqa: E402
    ddb_version as volume_front_ddb,
)

FACTORS = {
    "close_vwap_deviation": close_vwap_ddb,
    "volume_front_loading": volume_front_ddb,
    "volume_back_loading": volume_back_ddb,
    "late_session_strength": late_strength_ddb,
    "active_buy_sell_imbalance": abs_imbalance_ddb,
}
INDEX_CODE = "000852.SH"
RET_MATRIX = "PREHEAT_RET_MATRIX_ZZ1000"
PRIMARY_RET = "Ret_15"


def _connect():
    import dolphindb as ddb
    import dolphindb.settings as keys

    session = ddb.session(protocol=keys.PROTOCOL_DDB)
    session.connect(**DATA_DB_CONN)
    session.run(intraday_lib.ddb_functions)
    if not bool(session.run(f'defined("{RET_MATRIX}", SHARED)')):
        raise RuntimeError(f"{RET_MATRIX} is missing; run data_preheat.py first")
    return session


def _as_tradetime(narrow: pd.DataFrame, factor_name: str) -> pd.DataFrame:
    out = narrow.rename(columns={"bartime": "tradetime"}).copy()
    out["tradetime"] = pd.to_datetime(out["tradetime"])
    out["factorname"] = factor_name
    return out[["tradetime", "symbol", "factorname", "value"]].dropna(
        subset=["value"]
    )


def _evaluate(
    session,
    factor_name: str,
    signal: pd.DataFrame,
    *,
    apply_limit_filter: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    upload_name = "audit_" + "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in factor_name
    )
    session.upload({upload_name: signal})
    limit_block = ""
    if apply_limit_filter:
        limit_block = """
signal = select *, second(tradetime) as Bartime from signal
syms = exec distinct symbol from signal
minD = min(exec Date from signal)
maxD = max(exec Date from signal)
limit_t = get_limit_status(minD, maxD, syms)
limit_t = select Symbol as symbol, Date, Bartime, Limit_Status from limit_t
signal = lj(signal, limit_t, `symbol`Date`Bartime)
signal = select * from signal where isNull(Limit_Status) or Limit_Status = 0
signal = select tradetime, symbol, factorname, value, Date from signal
"""
    session.run(
        f"""
signal = {upload_name}
signal = select *, date(tradetime) as Date from signal
signal = filter_in_index(signal, "{INDEX_CODE}")
{limit_block}
group_data_ret, summary = get_cs_group_performance(
    signal, {RET_MATRIX}, group_num=10
)
"""
    )
    filtered = pd.DataFrame(session.run("signal"))[
        ["tradetime", "symbol", "factorname", "value"]
    ]
    filtered["tradetime"] = pd.to_datetime(filtered["tradetime"])
    group_ret = pd.DataFrame(session.run("group_data_ret"))
    ic_mean = pd.DataFrame(session.run("summary['ic_mean']"))
    ic_ts = pd.DataFrame(session.run("summary['ic_ts']"))
    return filtered, group_ret, ic_mean, ic_ts


def _bt_string(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    return parsed.dt.strftime("%H:%M")


def _performance_rows(
    factor_name: str,
    group_ret: pd.DataFrame,
    ic_mean: pd.DataFrame,
    ic_ts: pd.DataFrame,
) -> list[dict]:
    excess = intraday_lib.subtract_market_return(group_ret)
    excess = excess.copy()
    excess["bartime_key"] = _bt_string(excess["Bartime"])
    ic_mean = ic_mean.copy()
    ic_ts = ic_ts.copy()
    ic_mean["bartime_key"] = _bt_string(ic_mean["Bartime"])
    ic_ts["bartime_key"] = _bt_string(ic_ts["Bartime"])
    ret_key_mean = "RetType" if "RetType" in ic_mean.columns else "valueType"
    ret_key_ts = "RetType" if "RetType" in ic_ts.columns else "valueType"

    turnover = float(intraday_lib.intraday_turnover_b_hl())
    fee_bps = float(getattr(fdl, "IMPLIED_ANNU_FEE_BPS", 7.5))
    daily_cost = turnover * fee_bps / 1e4
    rows = []
    for ret_type in sorted(
        set(ic_mean[ret_key_mean].astype(str))
        & {c for c in excess.columns if str(c).startswith("Ret_")}
    ):
        for bartime in sorted(excess["bartime_key"].dropna().unique()):
            groups = excess[
                (excess["bartime_key"] == bartime)
            ].copy()
            low = groups[groups["group"] == "group_0"].sort_values("Date")
            high = groups[groups["group"] == "group_9"].sort_values("Date")
            hl = groups[groups["group"] == "group_HML"].sort_values("Date")
            if hl.empty or ret_type not in hl.columns:
                continue
            hl_ret = pd.to_numeric(hl[ret_type], errors="coerce").dropna()
            direction = 1.0 if hl_ret.mean() >= 0 else -1.0
            net_ret = direction * hl_ret - daily_cost

            im = ic_mean[
                (ic_mean[ret_key_mean].astype(str) == ret_type)
                & (ic_mean["bartime_key"] == bartime)
            ]
            its = ic_ts[
                (ic_ts[ret_key_ts].astype(str) == ret_type)
                & (ic_ts["bartime_key"] == bartime)
            ]
            ic = float(im.iloc[0]["IC_Mean"]) if len(im) else np.nan
            icir = (
                float(im.iloc[0]["IC_IR"]) * np.sqrt(250) if len(im) else np.nan
            )
            ic_series = pd.to_numeric(its.get("Rank_IC"), errors="coerce").dropna()
            ic_direction = 1.0 if ic >= 0 else -1.0
            win_rate = (
                float((ic_direction * ic_series > 0).mean())
                if len(ic_series)
                else np.nan
            )

            def _sharpe(frame: pd.DataFrame) -> float:
                if frame.empty:
                    return np.nan
                return float(
                    fdl.calSharpe(pd.to_numeric(frame[ret_type], errors="coerce"))
                )

            rows.append(
                {
                    "factor": factor_name,
                    "bartime": bartime,
                    "return_window": ret_type,
                    "ic_mean": ic,
                    "icir_annualized": icir,
                    "ic_directional_win_rate": win_rate,
                    "g1_excess_sharpe": _sharpe(low),
                    "g10_excess_sharpe": _sharpe(high),
                    "hl_sharpe_raw": float(fdl.calSharpe(hl_ret)),
                    "hl_sharpe_directional": float(fdl.calSharpe(direction * hl_ret)),
                    "hl_sharpe_after_7p5bps": float(fdl.calSharpe(net_ret)),
                    "turnover_b_hl": turnover,
                    "n_dates": int(hl["Date"].nunique()),
                }
            )
    return rows


def _slot_correlations(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    long_rows = []
    all_slots = sorted(
        {
            t.strftime("%H:%M")
            for frame in signals.values()
            for t in pd.to_datetime(frame["tradetime"]).dt.time.unique()
        }
    )
    for slot in all_slots:
        slot_frames = {}
        for name, frame in signals.items():
            mask = frame["tradetime"].dt.strftime("%H:%M") == slot
            if mask.any():
                slot_frames[name] = frame.loc[
                    mask, ["tradetime", "symbol", "value"]
                ]
        names = sorted(slot_frames)
        for i, left_name in enumerate(names):
            for right_name in names[i:]:
                merged = slot_frames[left_name].merge(
                    slot_frames[right_name],
                    on=["tradetime", "symbol"],
                    suffixes=("_left", "_right"),
                )
                daily = merged.groupby("tradetime").apply(
                    lambda g: g["value_left"].corr(
                        g["value_right"], method="spearman"
                    )
                ).dropna()
                if daily.empty:
                    continue
                long_rows.append(
                    {
                        "bartime": slot,
                        "factor_a": left_name,
                        "factor_b": right_name,
                        "mean_spearman": float(daily.mean()),
                        "median_spearman": float(daily.median()),
                        "n_dates": int(len(daily)),
                        "mean_cross_section_n": float(
                            merged.groupby("tradetime").size().mean()
                        ),
                    }
                )
    return pd.DataFrame(long_rows)


def _residualize_absi_on_lss(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    absi = signals["active_buy_sell_imbalance"]
    lss = signals["late_session_strength"]
    absi = absi[absi["tradetime"].dt.strftime("%H:%M") == "09:59"]
    lss = lss[lss["tradetime"].dt.strftime("%H:%M") == "09:59"]
    merged = absi.merge(
        lss,
        on=["tradetime", "symbol"],
        suffixes=("_absi", "_lss"),
    )

    def _residual(group: pd.DataFrame) -> pd.DataFrame:
        x = group["value_lss"].to_numpy(dtype=float)
        y = group["value_absi"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        out = group[["tradetime", "symbol"]].copy()
        out["value"] = y - design @ beta
        return out

    residual = (
        merged.groupby("tradetime", group_keys=False)
        .apply(_residual)
        .reset_index(drop=True)
    )
    residual["factorname"] = "active_buy_sell_imbalance_resid_lss"
    return residual[["tradetime", "symbol", "factorname", "value"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument(
        "--output",
        default="research/results/intraday_alpha_library_v1",
    )
    parser.add_argument("--no-limit-filter", action="store_true")
    args = parser.parse_args()

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    session = _connect()
    filtered_signals: Dict[str, pd.DataFrame] = {}
    performance = []

    for factor_name, computer in FACTORS.items():
        print(f"[BUILD] {factor_name}", flush=True)
        narrow = _as_tradetime(computer(args.start, args.end), factor_name)
        filtered, group_ret, ic_mean, ic_ts = _evaluate(
            session,
            factor_name,
            narrow,
            apply_limit_filter=not args.no_limit_filter,
        )
        filtered_signals[factor_name] = filtered
        performance.extend(
            _performance_rows(factor_name, group_ret, ic_mean, ic_ts)
        )
        print(
            f"[DONE] {factor_name}: raw={len(narrow):,} "
            f"evaluated={len(filtered):,}",
            flush=True,
        )

    correlations = _slot_correlations(filtered_signals)
    residual = _residualize_absi_on_lss(filtered_signals)
    _, resid_group, resid_ic_mean, resid_ic_ts = _evaluate(
        session,
        "active_buy_sell_imbalance_resid_lss",
        residual,
        apply_limit_filter=False,  # Inputs already share the filtered universe.
    )
    performance.extend(
        _performance_rows(
            "active_buy_sell_imbalance_resid_lss",
            resid_group,
            resid_ic_mean,
            resid_ic_ts,
        )
    )

    perf_df = pd.DataFrame(performance)
    perf_df.to_csv(output / "performance_by_slot.csv", index=False)
    correlations.to_csv(output / "spearman_by_slot.csv", index=False)
    residual.to_parquet(output / "absi_residual_lss_0959.parquet", index=False)

    primary = perf_df[perf_df["return_window"] == PRIMARY_RET].copy()
    summary = {
        "start": args.start,
        "end": args.end,
        "index": INDEX_CODE,
        "primary_return_window": PRIMARY_RET,
        "fee_bps_per_turnover": float(
            getattr(fdl, "IMPLIED_ANNU_FEE_BPS", 7.5)
        ),
        "limit_filter": not args.no_limit_filter,
        "factor_signal_rows": {
            key: int(len(value)) for key, value in filtered_signals.items()
        },
        "primary_performance": primary.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "correlations": correlations.replace({np.nan: None}).to_dict(
            orient="records"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Intraday Alpha Library v1 → {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
