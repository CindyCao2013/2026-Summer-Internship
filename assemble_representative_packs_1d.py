#!/usr/bin/env python3
"""Assemble representative factor packs for Milestone 1D (artifact harvest only)."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
FACTORS = REPO / "research" / "reports" / "factors"


def _implied_fee(to: float, bp: float = 7.5) -> float:
    return float(to) * bp / 1e4 * 250.0


def assemble_flow() -> Path:
    """FlowDensity20 pack already exists; refresh mechanism/diagnostic pointers only."""
    out = FACTORS / "FlowDensity20"
    out.mkdir(parents=True, exist_ok=True)
    for d in ("mechanism", "execution", "diagnostics", "artifacts", "charts"):
        (out / d).mkdir(exist_ok=True)
    src_mech = REPO / "research" / "reports" / "l2_flow_density_v1" / "mechanism"
    for name in (
        "amount_orth_summary.md",
        "amount_orth_verdict.json",
        "mechanism_summary.md",
        "mechanism_amount_neutral.csv",
        "mechanism.csv",
        "mechanism_analysis.csv",
    ):
        s = src_mech / name
        if s.exists():
            shutil.copy2(s, out / "mechanism" / name)
            if name.endswith(".csv") and name in ("mechanism.csv", "mechanism_analysis.csv"):
                shutil.copy2(s, out / name)
    src_exec = REPO / "research" / "reports" / "l2_flow_density_v1" / "execution" / "execution_summary.csv"
    if src_exec.exists() and not (out / "execution_summary.csv").exists():
        shutil.copy2(src_exec, out / "execution_summary.csv")
    return out


def assemble_d1() -> Path:
    src_fig = (
        REPO
        / "research/reports/d1_liquidity_density_v1/confirmation_1455d/low_vol_liquidity_quality_60d/figures"
    )
    out = FACTORS / "D1_LiquidityQuality60d"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)

    rows = [
        dict(
            factor="D1_LiquidityQuality60d",
            period="confirmation_1455d",
            universe="ALL",
            mode="raw",
            rank_ic=0.0573,
            annu_ic=0.0573 * math.sqrt(250),
            icir=6.0135,
            hl_annu_ret=0.4351,
            hl_sharpe=2.2614,
            hl_mdd=-0.1973,
            daily_turnover=0.4822,
            implied_annu_fee=_implied_fee(0.4822),
            net_sharpe=None,
            monotonicity=0.80,
            direction=1,
        ),
        dict(
            factor="D1_LiquidityQuality60d",
            period="confirmation_1455d",
            universe="CSI1000",
            mode="diag_CSI1000",
            rank_ic=0.0534,
            annu_ic=0.0534 * math.sqrt(250),
            icir=5.3675,
            hl_annu_ret=0.3641,
            hl_sharpe=1.8444,
            hl_mdd=None,
            daily_turnover=0.5001,
            implied_annu_fee=_implied_fee(0.5001),
            net_sharpe=None,
            monotonicity=None,
            direction=1,
        ),
        dict(
            factor="D1_LiquidityQuality60d",
            period="confirmation_1455d",
            universe="CSI500",
            mode="diag_CSI500",
            rank_ic=0.0468,
            annu_ic=0.0468 * math.sqrt(250),
            icir=4.3643,
            hl_annu_ret=0.1833,
            hl_sharpe=0.8982,
            hl_mdd=None,
            daily_turnover=0.5127,
            implied_annu_fee=_implied_fee(0.5127),
            net_sharpe=None,
            monotonicity=None,
            direction=1,
        ),
        dict(
            factor="D1_LiquidityQuality60d",
            period="confirmation_1455d",
            universe="CSI300",
            mode="diag_CSI300",
            rank_ic=0.0406,
            annu_ic=0.0406 * math.sqrt(250),
            icir=3.1960,
            hl_annu_ret=0.1486,
            hl_sharpe=0.6249,
            hl_mdd=None,
            daily_turnover=0.5142,
            implied_annu_fee=_implied_fee(0.5142),
            net_sharpe=None,
            monotonicity=None,
            direction=1,
        ),
    ]
    pd.DataFrame(rows).to_csv(out / "factor_summary.csv", index=False)

    mech = pd.DataFrame(
        [
            dict(
                signal="low_vol_liquidity_quality_60d",
                category="canonical",
                rank_ic=0.0573,
                icir=6.0135,
                hl_sharpe=2.2614,
                net_sharpe=None,
                monotonicity=0.80,
                daily_turnover=0.4822,
            ),
        ]
    )
    mech.to_csv(out / "mechanism.csv", index=False)
    mech.to_csv(out / "mechanism_analysis.csv", index=False)

    yearly = pd.DataFrame(
        [
            {
                "period": "confirmation_1455d",
                "kind": "block",
                "n_days": 1455,
                "rank_ic": 0.0573,
                "icir": 6.0135,
                "pos_ic_frac": 0.6534,
            }
        ]
    )
    yearly.to_csv(out / "yearly_stability.csv", index=False)
    yearly.to_csv(out / "stability.csv", index=False)

    mapping = {
        "quantile_return.png": "decile_return.png",
        "cumulative_long_short.png": "cumulative_long_short.png",
        "ic_timeseries.png": "ic_curve.png",
    }
    for src_name, dst_name in mapping.items():
        s = src_fig / src_name
        if s.exists():
            shutil.copy2(s, out / dst_name)
            shutil.copy2(s, out / "figures" / dst_name)

    pd.DataFrame(rows)[
        ["universe", "rank_ic", "icir", "hl_sharpe", "hl_annu_ret", "daily_turnover"]
    ].to_csv(out / "diagnostics_universe_ladder.csv", index=False)

    pd.DataFrame(
        {"horizon_days": [1, 5, 10, 20], "rank_ic": [0.0504, 0.0708, 0.0811, 0.0929]}
    ).to_csv(out / "diagnostics_ic_decay.csv", index=False)

    metrics = {
        "schema_version": "factor_report_v1_harvest",
        "factor": "D1_LiquidityQuality60d",
        "research_score": {
            "RankIC": 0.0573,
            "ICIR": 6.0135,
            "Sharpe": 2.2614,
            "Monotonicity": 0.80,
            "MDD": -0.1973,
        },
        "production_score": {"Turnover_raw": 0.4822},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return out


def assemble_ideal_reversal() -> Path:
    src = REPO / "research" / "reports" / "factor_cutting_v1" / "ideal_reversal"
    out = FACTORS / "IdealReversal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    meta = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    m = meta["mechanism"]
    legs = {x["leg"]: x for x in m["legs"]}
    spread = legs.get("spread", {})

    rank_ic = float(meta["rank_ic"])
    icir = float(meta["icir"])
    row = dict(
        factor="IdealReversal",
        period="cutting_v1_harvest",
        universe="ALL",
        mode="raw",
        rank_ic=rank_ic,
        annu_ic=rank_ic * math.sqrt(250),
        icir=icir,
        hl_annu_ret=float(meta["hl_annu_ret"]),
        hl_sharpe=float(meta["hl_sharpe"]),
        hl_mdd=None,
        daily_turnover=None,
        implied_annu_fee=None,
        net_sharpe=None,
        monotonicity=float(meta["monotonicity"]),
        direction=int(meta["direction"]),
    )
    rows = [row]
    neut_path = src / "robustness" / "neutralization.csv"
    if neut_path.exists():
        ndf = pd.read_csv(neut_path)
        for _, r in ndf.iterrows():
            mode = str(r.get("mode", r.get("neutralization", "unknown"))).lower()
            mode = mode.replace("+", "_").replace(" ", "_")
            if mode not in ("size", "industry", "size_industry", "raw"):
                continue
            ric = float(r.get("rank_ic", r.get("RankIC", rank_ic)))
            rows.append(
                dict(
                    factor="IdealReversal",
                    period="cutting_v1_harvest",
                    universe="ALL",
                    mode=mode,
                    rank_ic=ric,
                    annu_ic=ric * math.sqrt(250),
                    icir=float(r.get("icir", r.get("ICIR", icir))),
                    hl_annu_ret=float(r.get("hl_annu_ret", meta["hl_annu_ret"])),
                    hl_sharpe=float(r.get("hl_sharpe", meta["hl_sharpe"])),
                    hl_mdd=r.get("hl_mdd"),
                    daily_turnover=r.get("daily_turnover"),
                    implied_annu_fee=None,
                    net_sharpe=r.get("net_sharpe"),
                    monotonicity=r.get("monotonicity", meta["monotonicity"]),
                    direction=int(meta["direction"]),
                )
            )
    pd.DataFrame(rows).drop_duplicates(subset=["mode"], keep="first").to_csv(
        out / "factor_summary.csv", index=False
    )

    mech = pd.DataFrame(
        [
            dict(
                signal="M_high",
                category="cutting_leg",
                rank_ic=legs["high"]["rank_ic"],
                icir=legs["high"]["icir"],
                hl_sharpe=None,
                net_sharpe=None,
                monotonicity=None,
                daily_turnover=None,
            ),
            dict(
                signal="M_low",
                category="cutting_leg",
                rank_ic=legs["low"]["rank_ic"],
                icir=legs["low"]["icir"],
                hl_sharpe=None,
                net_sharpe=None,
                monotonicity=None,
                daily_turnover=None,
            ),
            dict(
                signal="M_spread",
                category="cutting_output",
                rank_ic=spread.get("rank_ic", rank_ic),
                icir=spread.get("icir", icir),
                hl_sharpe=meta["hl_sharpe"],
                net_sharpe=None,
                monotonicity=meta["monotonicity"],
                daily_turnover=None,
            ),
            dict(
                signal="Ret20_baseline",
                category="object",
                rank_ic=-0.0437,
                icir=-4.06,
                hl_sharpe=None,
                net_sharpe=None,
                monotonicity=None,
                daily_turnover=None,
            ),
        ]
    )
    mech.to_csv(out / "mechanism.csv", index=False)
    mech.to_csv(out / "mechanism_analysis.csv", index=False)

    yearly = pd.DataFrame(
        [
            {
                "period": "full_sample",
                "kind": "block",
                "n_days": int(meta["n_days"]),
                "rank_ic": rank_ic,
                "icir": icir,
                "pos_ic_frac": float(meta["ic_pos_ratio"]),
            }
        ]
    )
    yearly.to_csv(out / "yearly_stability.csv", index=False)
    yearly.to_csv(out / "stability.csv", index=False)

    chart_map = {
        src / "ic_analysis" / "rank_ic_timeseries.png": "ic_curve.png",
        src / "portfolio" / "decile_return.png": "decile_return.png",
        src / "portfolio" / "long_short_curve.png": "cumulative_long_short.png",
    }
    for s, name in chart_map.items():
        if s.exists():
            shutil.copy2(s, out / name)
            shutil.copy2(s, out / "figures" / name)

    for p in (src / "mechanism").glob("*.png"):
        shutil.copy2(p, out / "figures" / p.name)

    metrics = {
        "schema_version": "factor_report_v1_harvest",
        "factor": "IdealReversal",
        "research_score": {
            "RankIC": rank_ic,
            "ICIR": icir,
            "Sharpe": float(meta["hl_sharpe"]),
            "Monotonicity": float(meta["monotonicity"]),
        },
        "production_score": {},
        "cutting": {
            "knife": meta["knife"],
            "purity": m.get("purity"),
            "separation": m.get("separation"),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    paths = {
        "FlowDensity20": assemble_flow(),
        "D1_LiquidityQuality60d": assemble_d1(),
        "IdealReversal": assemble_ideal_reversal(),
    }
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
