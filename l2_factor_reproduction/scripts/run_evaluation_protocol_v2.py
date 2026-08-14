#!/usr/bin/env python
"""Factor Evaluation Protocol v2.0 — Final Convention Fix + freeze.

Usage:
    python -m l2_factor_reproduction.scripts.run_evaluation_protocol_v2
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from Factor_Dev_Lib import groupTest  # noqa: E402
from l2_factor_reproduction.config.settings import BACKTEST_SILENT, RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    backtest_factor,
    compute_rank_ic,
    narrow_to_wide,
    prepare_factor_signal,
)
from l2_factor_reproduction.python.evaluation_protocol_v2 import (  # noqa: E402
    ACCEPTABLE_FACTOR,
    ANNUALIZATION_DAYS,
    BENCHMARK,
    FEE_BPS_PER_TRADED_NOTIONAL,
    FEE_RATE_L1,
    FREEZE_DATE,
    HIGH_TURNOVER_L1_DAILY,
    HIGH_TURNOVER_ONEWAY_DAILY,
    OUT_ROOT,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
    SAMPLES,
    SIGNAL_SHIFT,
    STRONG_FACTOR,
    STRATEGY_FULL_PROVISIONAL,
    FrozenFactorSpec,
    assign_factor_grade,
    assign_long_only_status,
    check_effective_turnover_parity,
    ensure_effective_group_to,
    factor_layer_metrics,
    label_block,
    load_benchmark_return,
    long_only_metrics,
)
from l2_factor_reproduction.python.fast_discovery import (  # noqa: E402
    FULL_END,
    FULL_START,
    ensure_effective_group_pnl,
    load_fast_context,
)
from l2_factor_reproduction.scripts.validate_effective_spread_proxy import (  # noqa: E402
    build_factor_narrow as build_esp,
)
from l2_factor_reproduction.scripts.validate_liquidity_resilience_proxy_5d import (  # noqa: E402
    build_factor_narrow as build_resilience,
)

SPECS: List[Tuple[FrozenFactorSpec, Callable[..., pd.DataFrame]]] = [
    (
        FrozenFactorSpec(
            factor_id="liquidity_resilience_proxy_5d",
            factor_hash=(
                "e1dc110104e6fdef8af515c3b45da032cdaa89f1117519ca2147cb8173dc95b3"
            ),
            exact_formula=(
                "liquidity_resilience_proxy_5d = rolling_mean_5d(depth_recovery_5m); "
                "depth_recovery_5m = mean((Depth5_{t+5}-Depth5_t)/Depth5_t) over "
                "high-impact minutes; high-impact = |minute_mid_return| >= same-day "
                "90th percentile; rolling window = 5 trading days, min_periods=5; "
                "signal_shift = T+1; NO shock_weight; NO recovery clip [0,1]; "
                "NO weighted mean recovery"
            ),
            source_primitive="liquidity_impact_daily.depth_recovery_5m",
            expected_direction=-1,
        ),
        build_resilience,
    ),
    (
        FrozenFactorSpec(
            factor_id="effective_spread_proxy",
            factor_hash=(
                "1a0577314b7e974272fb1a98007cb4c6a0b5486a62f02601f1db9e673c83bbf2"
            ),
            exact_formula=(
                "effective_spread_proxy = mean(2*sign(signed_amount)*"
                "(trade_vwap-midquote)/midquote); minute approximation over "
                "continuous auction minutes; signed_amount = active_buy_amount - "
                "active_sell_amount; midquote = (bid1+ask1)/2; signal_shift = T+1; "
                "NOT per-trade prevailing-quote effective spread"
            ),
            source_primitive="liquidity_impact_daily.effective_spread_proxy",
            expected_direction=1,
        ),
        build_esp,
    ),
]

EXCLUDE_INVENTORY = {
    "liquidity_resilience_proxy_5d",
    "effective_spread_proxy",
    "effective_spread_persistence_5d",
}
NEAR_MECH = {
    "liquidity_cost_persistence",  # same family as resilience / spread
}


def _fmt_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.2%}"


def _fmt_num(x: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{digits}f}"


def _labels_md(labels: Dict[str, str]) -> str:
    return " | ".join(f"**{k}**={v}" for k, v in labels.items())


def snapshot_old_outputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve pre-v2.0 CSVs for old-vs-final parity."""
    bak = OUT_ROOT / "_pre_v2_0_snapshot"
    bak.mkdir(parents=True, exist_ok=True)
    f_old = OUT_ROOT / "factor_level_summary.csv"
    l_old = OUT_ROOT / "long_only_summary.csv"
    if f_old.exists():
        shutil.copy2(f_old, bak / "factor_level_summary.csv")
    if l_old.exists():
        shutil.copy2(l_old, bak / "long_only_summary.csv")
    factor_old = pd.read_csv(bak / "factor_level_summary.csv") if (bak / "factor_level_summary.csv").exists() else pd.DataFrame()
    long_old = pd.read_csv(bak / "long_only_summary.csv") if (bak / "long_only_summary.csv").exists() else pd.DataFrame()
    return factor_old, long_old


def raw_and_effective_turnover(
    narrow: pd.DataFrame,
    mask: pd.DataFrame,
    ret: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.Series]:
    """Return raw pnl/to, effective pnl/to, direction, rank_ic_raw."""
    signal_raw, ret_aln = prepare_factor_signal(
        narrow_to_wide(narrow),
        start=FULL_START,
        end=FULL_END,
        mask=mask,
        ret_matrix=ret,
    )
    rank_ic_raw = compute_rank_ic(signal_raw, ret_aln)
    info = "silent" if BACKTEST_SILENT else "silent"
    _r, pnl_raw, to_raw = groupTest(signal_raw, ret_aln, n=10, info=info)
    direction = 1 if float(pnl_raw["H-L"].mean()) > 0 else -1

    group_pnl, group_to, _rank_ic_eff, summary = backtest_factor(
        narrow,
        start_day=FULL_START,
        end_day=FULL_END,
        mask=mask,
        ret_matrix=ret,
    )
    assert int(summary["factor_direction"]) == direction
    # Defensive effective transform (no-op if already effective)
    pnl_eff = ensure_effective_group_pnl(group_pnl)
    to_eff = ensure_effective_group_to(group_to, group_pnl)
    return pnl_raw, to_raw, pnl_eff, to_eff, direction, rank_ic_raw


def evaluate_one(
    spec: FrozenFactorSpec,
    builder: Callable[..., pd.DataFrame],
    mask: pd.DataFrame,
    ret: pd.DataFrame,
    benchmark: pd.Series,
) -> Dict[str, Any]:
    print(f"\n=== {spec.factor_id} ===", flush=True)
    narrow = builder(FULL_START, FULL_END)
    pnl_raw, to_raw, group_pnl, group_to, direction, rank_ic_raw = raw_and_effective_turnover(
        narrow, mask, ret
    )
    if direction != spec.expected_direction:
        raise RuntimeError(
            f"{spec.factor_id}: direction={direction} != expected={spec.expected_direction}"
        )

    parity = check_effective_turnover_parity(
        factor_id=spec.factor_id,
        raw_direction=direction,
        to_raw=to_raw,
        to_eff=group_to,
    )
    print(
        f"  turnover parity pass={parity['pass']} "
        f"raw_g1={parity['raw_g1_l1']:.4f} raw_g10={parity['raw_g10_l1']:.4f} "
        f"eff_g1={parity['effective_g1_l1']:.4f} eff_g10={parity['effective_g10_l1']:.4f}",
        flush=True,
    )
    if not parity["pass"]:
        raise RuntimeError(f"PART G FAIL for {spec.factor_id}: {parity}")

    factor_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    by_sample: Dict[str, Dict[str, Any]] = {}
    for sample_name, (s0, s1) in SAMPLES.items():
        fmet = factor_layer_metrics(
            group_pnl, group_to, rank_ic_raw, sample=sample_name, start=s0, end=s1
        )
        lmet = long_only_metrics(
            group_pnl, group_to, benchmark, sample=sample_name, start=s0, end=s1
        )
        labels = label_block(
            factor_id=spec.factor_id,
            factor_hash=spec.factor_hash,
            sample=sample_name,
            return_kind="GROSS+NET_7P5BPS_L1",
        )
        fmet = {**labels, **fmet, "factor_direction": direction}
        lmet = {**labels, **lmet, "factor_direction": direction}
        factor_rows.append(fmet)
        long_rows.append(lmet)
        by_sample[sample_name] = {"factor": fmet, "long": lmet}
        print(
            f"  [{sample_name}] grossS={fmet['gross_hl_sharpe']:.2f} "
            f"netS={fmet['net_hl_sharpe']:.2f} "
            f"hl_ow={fmet['avg_daily_hl_oneway_turnover']:.3f} "
            f"excess={lmet['excess_annual_return']:.2%} IR={lmet['IR']:.2f}",
            flush=True,
        )

    full = by_sample["FULL"]["factor"]
    post = by_sample["POST"]["factor"]
    long_full = by_sample["FULL"]["long"]
    long_post = by_sample["POST"]["long"]
    factor_grade, flags, retention = assign_factor_grade(full, post)
    long_only_status, strat_warns = assign_long_only_status(long_full, long_post)

    return {
        "spec": spec,
        "direction": direction,
        "parity": parity,
        "factor_rows": factor_rows,
        "long_rows": long_rows,
        "by_sample": by_sample,
        "factor_grade": factor_grade,
        "long_only_status": long_only_status,
        "flags": flags,
        "strategy_warnings": strat_warns,
        "retention": retention,
    }


def build_final_parity_csv(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        full = r["by_sample"]["FULL"]["factor"]
        long_full = r["by_sample"]["FULL"]["long"]
        rows.append(
            {
                "factor": r["spec"].factor_id,
                "FACTOR_HASH": r["spec"].factor_hash,
                "factor_grade": r["factor_grade"],
                "long_only_status": r["long_only_status"],
                "gross_hl_annual": full["gross_hl_annual"],
                "gross_hl_sharpe": full["gross_hl_sharpe"],
                "net_hl_annual": full["net_hl_annual"],
                "net_hl_sharpe": full["net_hl_sharpe"],
                "hl_l1_traded_notional_daily": full["avg_daily_hl_l1_traded_notional"],
                "hl_oneway_turnover_daily": full["avg_daily_hl_oneway_turnover"],
                "hl_oneway_turnover_annualized": full["annualized_hl_oneway_turnover"],
                "long_l1_traded_notional_daily": full["avg_daily_long_l1_traded_notional"],
                "long_oneway_turnover_daily": full["avg_daily_long_oneway_turnover"],
                "annual_fee": full["fee_annualized"],
                "long_net_annual": long_full["long_net_annual_return"],
                "net_excess_annual": long_full["excess_annual_return"],
                "IR": long_full["IR"],
                "positive_hl_month_fraction": full["positive_hl_month_fraction"],
                "positive_hl_month_fraction_sum_legacy": full[
                    "positive_hl_month_fraction_sum_legacy"
                ],
                "positive_excess_month_fraction": long_full[
                    "positive_excess_month_fraction"
                ],
                "positive_excess_month_fraction_sum_legacy": long_full[
                    "positive_excess_month_fraction_sum_legacy"
                ],
                "POST_flags": "|".join(r["flags"]),
                "strategy_warnings": "|".join(r["strategy_warnings"]),
                "PROTOCOL_VERSION": PROTOCOL_VERSION,
                "UNIVERSE": BENCHMARK,
                "BENCHMARK": BENCHMARK,
                "COST": f"{FEE_BPS_PER_TRADED_NOTIONAL:g}bps per L1",
                "SIGNAL_SHIFT": SIGNAL_SHIFT,
            }
        )
    return pd.DataFrame(rows)


def build_old_vs_final(
    results: List[Dict[str, Any]],
    factor_old: pd.DataFrame,
    long_old: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for r in results:
        fid = r["spec"].factor_id
        full = r["by_sample"]["FULL"]["factor"]
        long_full = r["by_sample"]["FULL"]["long"]
        fo = factor_old.loc[
            (factor_old["FACTOR_ID"] == fid) & (factor_old["SAMPLE"] == "FULL")
        ]
        lo = long_old.loc[
            (long_old["FACTOR_ID"] == fid) & (long_old["SAMPLE"] == "FULL")
        ]
        if fo.empty or lo.empty:
            raise RuntimeError(f"missing old snapshot for {fid}")
        fo = fo.iloc[0]
        lo = lo.iloc[0]

        def _delta(new: float, old: float) -> float:
            return float(new) - float(old)

        # Expected invariant metrics
        checks = {
            "rank_ic": (full["rank_ic"], fo["rank_ic"], 1e-10, True),
            "icir": (full["icir"], fo["icir"], 1e-8, True),
            "gross_hl_annual": (full["gross_hl_annual"], fo["gross_hl_annual"], 1e-10, True),
            "gross_hl_sharpe": (full["gross_hl_sharpe"], fo["gross_hl_sharpe"], 1e-10, True),
            "decile_mono_gross": (
                full["decile_mono_gross"],
                fo["decile_mono_gross"],
                1e-10,
                True,
            ),
            "adjacent_violations_gross": (
                full["adjacent_violations_gross"],
                fo["adjacent_violations_gross"],
                0,
                True,
            ),
            "net_hl_annual": (full["net_hl_annual"], fo["net_hl_annual"], 1e-10, True),
            "net_hl_sharpe": (full["net_hl_sharpe"], fo["net_hl_sharpe"], 1e-10, True),
            "excess_annual_return": (
                long_full["excess_annual_return"],
                lo["excess_annual_return"],
                1e-10,
                True,
            ),
            "IR": (long_full["IR"], lo["IR"], 1e-10, True),
        }
        for name, (new_v, old_v, atol, must_equal) in checks.items():
            ok = bool(np.isclose(float(new_v), float(old_v), rtol=0, atol=atol))
            if must_equal and not ok:
                raise RuntimeError(
                    f"UNEXPECTED CHANGE {fid}.{name}: old={old_v} new={new_v}"
                )
            rows.append(
                {
                    "factor": fid,
                    "metric": name,
                    "old": float(old_v),
                    "new": float(new_v),
                    "delta": _delta(new_v, old_v),
                    "expected_change": "none",
                    "pass": ok,
                }
            )

        # Allowed: turnover display L1 -> one-way (= old/2)
        old_to = float(fo["avg_daily_hl_turnover"])
        new_ow = float(full["avg_daily_hl_oneway_turnover"])
        new_l1 = float(full["avg_daily_hl_l1_traded_notional"])
        rows.append(
            {
                "factor": fid,
                "metric": "hl_l1_traded_notional_daily",
                "old": old_to,
                "new": new_l1,
                "delta": _delta(new_l1, old_to),
                "expected_change": "none_L1_identity",
                "pass": bool(np.isclose(new_l1, old_to, rtol=0, atol=1e-10)),
            }
        )
        rows.append(
            {
                "factor": fid,
                "metric": "hl_oneway_turnover_daily",
                "old": old_to,
                "new": new_ow,
                "delta": _delta(new_ow, old_to),
                "expected_change": "display_half_of_old_L1",
                "pass": bool(np.isclose(new_ow, 0.5 * old_to, rtol=0, atol=1e-10)),
            }
        )

        # Allowed: month fraction compound vs sum
        rows.append(
            {
                "factor": fid,
                "metric": "positive_hl_month_fraction",
                "old": float(fo["positive_hl_month_fraction"]),
                "new": float(full["positive_hl_month_fraction"]),
                "delta": _delta(
                    full["positive_hl_month_fraction"], fo["positive_hl_month_fraction"]
                ),
                "expected_change": "compound_vs_sum_allowed",
                "pass": True,
            }
        )
        rows.append(
            {
                "factor": fid,
                "metric": "positive_excess_month_fraction",
                "old": float(lo["positive_excess_month_fraction"]),
                "new": float(long_full["positive_excess_month_fraction"]),
                "delta": _delta(
                    long_full["positive_excess_month_fraction"],
                    lo["positive_excess_month_fraction"],
                ),
                "expected_change": "compound_vs_sum_allowed",
                "pass": True,
            }
        )
    return pd.DataFrame(rows)


def render_protocol_md() -> str:
    return "\n".join(
        [
            f"# Factor Evaluation Protocol v{PROTOCOL_VERSION} (FINAL FROZEN)",
            "",
            f"Status: **{PROTOCOL_STATUS}**",
            f"Freeze date: **{FREEZE_DATE}**",
            "",
            "## Turnover convention (FINAL)",
            "",
            "Internal engine field:",
            "",
            "```",
            "l1_traded_notional_g,t = sum_i |w_i,t - w_i,t-1|",
            "```",
            "",
            "Reported conventional turnover (default name **turnover**):",
            "",
            "```",
            "oneway_turnover_g,t = 0.5 * l1_traded_notional_g,t",
            "```",
            "",
            "Never call sum|dw| \"one-way turnover\" without qualification.",
            "",
            "## Cost (economics unchanged)",
            "",
            "```",
            "cost = l1_traded_notional * 7.5bps",
            "     = oneway_turnover * 15bps",
            "```",
            "",
            "Full replacement: L1=2.0, one-way=1.0, cost=15bps.",
            "",
            f"HIGH_TURNOVER_ONEWAY_DAILY = {HIGH_TURNOVER_ONEWAY_DAILY} "
            f"(≡ historical L1 >= {HIGH_TURNOVER_L1_DAILY}).",
            "",
            "## Factor Grade ⊥ Long-only Status",
            "",
            "- `factor_grade`: A / B / C from Factor Layer FULL + POST flags only",
            "- `long_only_status`: PROVISIONAL_PASS / PROVISIONAL_FAIL",
            "- Strategy fail does **not** demote factor_grade",
            "",
            "## Thresholds FROZEN — no refit on these two factors",
            "",
            "STRONG:",
            json.dumps(STRONG_FACTOR, indent=2),
            "",
            "ACCEPTABLE:",
            json.dumps(ACCEPTABLE_FACTOR, indent=2),
            "",
            "Strategy FULL provisional:",
            json.dumps(STRATEGY_FULL_PROVISIONAL, indent=2),
            "",
            f"BENCHMARK / UNIVERSE = `{BENCHMARK}` | SIGNAL_SHIFT = `{SIGNAL_SHIFT}`",
            f"Fee display annualization = {ANNUALIZATION_DAYS}; Sharpe/IR = √252",
            "",
            "## Monthly win rate",
            "",
            "`monthly_return = (1+r).resample('ME').prod() - 1`",
            "",
        ]
    ) + "\n"


def render_cost_audit(results: List[Dict[str, Any]]) -> str:
    lines = [
        f"# Cost Convention Audit — Protocol v{PROTOCOL_VERSION}",
        "",
        "## Definitions",
        "",
        "- L1 traded notional = Σ|Δw| (engine)",
        "- one-way turnover = 0.5 × L1 (report default)",
        "- cost = L1 × 7.5bps = one-way × 15bps",
        "",
        "## Empirical (FULL)",
        "",
    ]
    for r in results:
        full = r["by_sample"]["FULL"]["factor"]
        lines += [
            f"### {r['spec'].factor_id}",
            "",
            f"- hl L1 daily = `{full['avg_daily_hl_l1_traded_notional']:.6f}`",
            f"- hl one-way daily = `{full['avg_daily_hl_oneway_turnover']:.6f}` "
            f"(= L1/2)",
            f"- long L1 / one-way = "
            f"`{full['avg_daily_long_l1_traded_notional']:.6f}` / "
            f"`{full['avg_daily_long_oneway_turnover']:.6f}`",
            f"- annual fee (L1×7.5bps×250) = `{full['fee_annualized']:.4%}`",
            f"- turnover parity pass = `{r['parity']['pass']}`",
            "",
        ]
    lines += [
        "## Q1 answer",
        "",
        "Engine sum|dw| is **L1 traded notional**. "
        "Standard one-way turnover is **half** of that.",
        "",
        "## Q2 answer",
        "",
        "**No** — net PnL must not change from turnover renaming; "
        "cost still uses L1 × 7.5bps.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_card(result: Dict[str, Any]) -> str:
    spec: FrozenFactorSpec = result["spec"]
    full = result["by_sample"]["FULL"]["factor"]
    post = result["by_sample"]["POST"]["factor"]
    long_full = result["by_sample"]["FULL"]["long"]
    long_post = result["by_sample"]["POST"]["long"]
    labels = label_block(
        factor_id=spec.factor_id,
        factor_hash=spec.factor_hash,
        sample="FULL",
        return_kind="GROSS / NET_7P5BPS_L1",
    )
    return "\n".join(
        [
            f"# Evaluation Card v{PROTOCOL_VERSION} — {spec.factor_id}",
            "",
            _labels_md(labels),
            "",
            f"**factor_grade:** `{result['factor_grade']}`",
            f"**long_only_status:** `{result['long_only_status']}`",
            "",
            f"- FACTOR_HASH = `{spec.factor_hash}`",
            f"- raw_direction = `{result['direction']}`",
            f"- turnover_parity_pass = `{result['parity']['pass']}`",
            "",
            "## Factor Layer FULL",
            "",
            f"| Metric | GROSS | NET |",
            f"|---|---:|---:|",
            f"| RankIC / ICIR | {_fmt_num(full['rank_ic'], 4)} / {_fmt_num(full['icir'], 2)} | — |",
            f"| H-L annual | {_fmt_pct(full['gross_hl_annual'])} | {_fmt_pct(full['net_hl_annual'])} |",
            f"| H-L Sharpe | {_fmt_num(full['gross_hl_sharpe'], 2)} | {_fmt_num(full['net_hl_sharpe'], 2)} |",
            f"| mono / viol | {_fmt_num(full['decile_mono_gross'], 3)} / {full['adjacent_violations_gross']} | {_fmt_num(full['decile_mono_net'], 3)} / {full['adjacent_violations_net']} |",
            f"| pos H-L month (compound) | {_fmt_pct(full['positive_hl_month_fraction'])} | legacy_sum={_fmt_pct(full['positive_hl_month_fraction_sum_legacy'])} |",
            "",
            f"- hl L1 daily / one-way daily = "
            f"`{_fmt_num(full['avg_daily_hl_l1_traded_notional'], 4)}` / "
            f"`{_fmt_num(full['avg_daily_hl_oneway_turnover'], 4)}`",
            f"- hl one-way annualized (*250) = "
            f"`{_fmt_num(full['annualized_hl_oneway_turnover'], 1)}`",
            f"- fee annualized = `{_fmt_pct(full['fee_annualized'])}`",
            "",
            "## POST",
            "",
            f"- RankIC={_fmt_num(post['rank_ic'], 4)} Sharpe={_fmt_num(post['gross_hl_sharpe'], 2)} "
            f"mono={_fmt_num(post['decile_mono_gross'], 3)} viol={post['adjacent_violations_gross']}",
            f"- IC_retention={_fmt_num(result['retention']['IC_retention'], 3)} "
            f"Sharpe_retention={_fmt_num(result['retention']['Sharpe_retention'], 3)}",
            "",
            "## Long-only FULL (G10 vs frozen benchmark)",
            "",
            f"- long_net_annual = `{_fmt_pct(long_full['long_net_annual_return'])}`",
            f"- excess_annual = `{_fmt_pct(long_full['excess_annual_return'])}`",
            f"- IR = `{_fmt_num(long_full['IR'], 2)}`",
            f"- long one-way daily = `{_fmt_num(long_full['avg_daily_long_oneway_turnover'], 4)}`",
            f"- pos excess month (compound) = `{_fmt_pct(long_full['positive_excess_month_fraction'])}` "
            f"(legacy_sum={_fmt_pct(long_full['positive_excess_month_fraction_sum_legacy'])})",
            "",
            f"POST excess={_fmt_pct(long_post['excess_annual_return'])} IR={_fmt_num(long_post['IR'], 2)}",
            "",
            f"## Flags",
            "",
            f"- factor POST/risk: `{', '.join(result['flags']) or 'none'}`",
            f"- strategy warnings: `{', '.join(result['strategy_warnings']) or 'none'}`",
            "",
        ]
    ) + "\n"


def build_next_inventory() -> Tuple[pd.DataFrame, str]:
    path = (
        Path(RESULT_ROOT)
        / "fast_discovery"
        / "low_turnover_v1"
        / "candidate_summary.csv"
    )
    s = pd.read_csv(path)
    # Also pull benchmark fast_summary for effective_spread_proxy context if present
    rows = []
    for _, row in s.iterrows():
        fid = str(row["factor"])
        if fid in EXCLUDE_INVENTORY:
            continue
        mech = str(row.get("mechanism", ""))
        if mech in NEAR_MECH and fid != "buy_absorption_5d":
            # exclude same-mechanism near neighbors of spread/resilience
            # keep flow_price_response / order_size / book / cancel
            if "liquidity_cost" in mech:
                continue
        gate = str(row.get("gate", ""))
        # convert discovery avg_hl_turnover (L1) → one-way
        l1 = float(row["avg_hl_turnover"]) if pd.notna(row["avg_hl_turnover"]) else np.nan
        rows.append(
            {
                "factor": fid,
                "mechanism": mech,
                "fast_hl_sharpe": row.get("hl_sharpe"),
                "fast_hl_annual": row.get("hl_annu_ret"),
                "mono": row.get("decile_mono_spearman"),
                "violations": row.get("adjacent_violations"),
                "g10_gross_excess_annual": np.nan,  # not in sprint8 summary
                "g10_excess_sharpe": row.get("g10_excess_sharpe"),
                "g10_oneway_turnover": np.nan,  # need G10 L1; only H-L L1 available
                "hl_oneway_turnover_discovery": (
                    0.5 * l1 if np.isfinite(l1) else np.nan
                ),
                "positive_excess_month_fraction": row.get("positive_month_fraction"),
                "fast_gate": gate,
                "source_primitive": row.get("source_primitive"),
            }
        )
    inv = pd.DataFrame(rows)
    strong = inv.loc[
        (inv["fast_gate"] == "strong_candidate")
        & (inv["fast_hl_sharpe"] >= 3.0)
        & (inv["mono"] >= 0.85)
        & (inv["violations"] <= 1)
    ].copy()
    if strong.empty:
        status = "SPRINT8_STRONG_INVENTORY_EXHAUSTED"
        next_id = status
    else:
        status = "HAS_STRONG"
        # tie-break by g10_excess_sharpe then mechanism diversity
        strong = strong.sort_values(
            ["g10_excess_sharpe", "fast_hl_sharpe"], ascending=False
        )
        next_id = str(strong.iloc[0]["factor"])
    return inv, next_id


def render_finish_answers(results: List[Dict[str, Any]], next_id: str, ovf: pd.DataFrame) -> str:
    lines = [
        f"# Protocol v{PROTOCOL_VERSION} — Final Answers",
        "",
        "## 1. Engine sum|dw| vs standard one-way turnover",
        "",
        "- Engine: **L1 traded notional** = Σ|Δw|",
        "- Report: **one-way turnover** = 0.5 × L1",
        "",
        "## 2. Must net PnL change because of turnover rename?",
        "",
        "**No.** cost = L1 × 7.5bps unchanged.",
        "",
        "## 3. Two factors — L1 vs one-way (FULL daily mean)",
        "",
    ]
    for r in results:
        f = r["by_sample"]["FULL"]["factor"]
        lines.append(
            f"- `{r['spec'].factor_id}`: L1=`{f['avg_daily_hl_l1_traded_notional']:.4f}`, "
            f"one-way=`{f['avg_daily_hl_oneway_turnover']:.4f}`"
        )
    lines += [
        "",
        "## 4. Negative-direction turnover sync flip?",
        "",
    ]
    for r in results:
        p = r["parity"]
        lines.append(
            f"- `{r['spec'].factor_id}` dir={p['raw_direction']}: pass=`{p['pass']}` "
            f"(eff_g10={p['effective_g10_l1']:.4f}, expected={p['expected_effective_g10_l1']:.4f})"
        )
    # Q5
    res = next(x for x in results if x["spec"].factor_id == "liquidity_resilience_proxy_5d")
    ir_row = ovf.loc[
        (ovf.factor == "liquidity_resilience_proxy_5d") & (ovf.metric == "IR")
    ].iloc[0]
    ex_row = ovf.loc[
        (ovf.factor == "liquidity_resilience_proxy_5d")
        & (ovf.metric == "excess_annual_return")
    ].iloc[0]
    lines += [
        "",
        "## 5. Did resilience Long-only excess/IR get corrected?",
        "",
        f"**No correction needed** (parity already correct). "
        f"excess delta={ex_row['delta']:.2e}, IR delta={ir_row['delta']:.2e}.",
        "",
        "## 6. Monthly win-rate compound vs sum",
        "",
    ]
    for r in results:
        f = r["by_sample"]["FULL"]["factor"]
        lf = r["by_sample"]["FULL"]["long"]
        lines.append(
            f"- `{r['spec'].factor_id}` H-L: compound={f['positive_hl_month_fraction']:.4f} "
            f"vs sum={f['positive_hl_month_fraction_sum_legacy']:.4f} "
            f"(Δ={f['positive_hl_month_fraction']-f['positive_hl_month_fraction_sum_legacy']:+.4f}); "
            f"excess: compound={lf['positive_excess_month_fraction']:.4f} "
            f"vs sum={lf['positive_excess_month_fraction_sum_legacy']:.4f}"
        )
    lines += ["", "## 7. Final grades", ""]
    for r in results:
        lines.append(
            f"- `{r['spec'].factor_id}`: factor_grade=`{r['factor_grade']}`, "
            f"long_only_status=`{r['long_only_status']}`, "
            f"flags=`{','.join(r['flags']) or 'none'}`, "
            f"strat=`{','.join(r['strategy_warnings']) or 'none'}`"
        )
    lines += [
        "",
        "## 8. Protocol v2.0 FINAL FROZEN?",
        "",
        f"**YES** — version={PROTOCOL_VERSION}, status={PROTOCOL_STATUS}, date={FREEZE_DATE}.",
        "",
        "## 9. Next independent candidate",
        "",
        f"**{next_id}**",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    t0 = time.perf_counter()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("[0] snapshot pre-v2.0 outputs for old-vs-final", flush=True)
    factor_old, long_old = snapshot_old_outputs()

    print("[1] load fast_context + benchmark", flush=True)
    mask, ret = load_fast_context("full")
    benchmark = load_benchmark_return("full")

    results: List[Dict[str, Any]] = []
    print("[2] evaluate two frozen factors (v2.0)", flush=True)
    for spec, builder in SPECS:
        results.append(evaluate_one(spec, builder, mask, ret, benchmark))

    print("[3] write parity / summaries / freeze docs", flush=True)
    factor_df = pd.DataFrame([row for r in results for row in r["factor_rows"]])
    long_df = pd.DataFrame([row for r in results for row in r["long_rows"]])
    factor_df.to_csv(OUT_ROOT / "factor_level_summary.csv", index=False)
    long_df.to_csv(OUT_ROOT / "long_only_summary.csv", index=False)

    parity_df = pd.DataFrame([r["parity"] for r in results])
    parity_df.to_csv(OUT_ROOT / "effective_direction_turnover_parity.csv", index=False)
    if not bool(parity_df["pass"].all()):
        raise RuntimeError("PART G FAIL — see effective_direction_turnover_parity.csv")

    final_df = build_final_parity_csv(results)
    final_df.to_csv(OUT_ROOT / "protocol_v2_final_parity.csv", index=False)

    ovf = build_old_vs_final(results, factor_old, long_old)
    ovf.to_csv(OUT_ROOT / "protocol_v2_old_vs_final.csv", index=False)
    if not bool(ovf.loc[ovf.expected_change == "none", "pass"].all()):
        raise RuntimeError("Unexpected metric changes — STOP")
    if not bool(
        ovf.loc[ovf.expected_change == "none_L1_identity", "pass"].all()
    ):
        raise RuntimeError("L1 identity break — STOP")
    if not bool(
        ovf.loc[ovf.expected_change == "display_half_of_old_L1", "pass"].all()
    ):
        raise RuntimeError("one-way != L1/2 — STOP")

    (OUT_ROOT / "protocol.md").write_text(render_protocol_md(), encoding="utf-8")
    (OUT_ROOT / "cost_convention_audit.md").write_text(
        render_cost_audit(results), encoding="utf-8"
    )

    for r in results:
        d = OUT_ROOT / r["spec"].factor_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "evaluation_card.md").write_text(render_card(r), encoding="utf-8")
        meta = {
            "PROTOCOL_VERSION": PROTOCOL_VERSION,
            "FACTOR_ID": r["spec"].factor_id,
            "FACTOR_HASH": r["spec"].factor_hash,
            "factor_grade": r["factor_grade"],
            "long_only_status": r["long_only_status"],
            "flags": r["flags"],
            "strategy_warnings": r["strategy_warnings"],
            "retention": r["retention"],
            "parity": r["parity"],
            "BENCHMARK": BENCHMARK,
            "COST": f"{FEE_BPS_PER_TRADED_NOTIONAL:g}bps per L1 traded notional",
        }
        (d / "evaluation_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print("[4] next independent candidate inventory", flush=True)
    inv, next_id = build_next_inventory()
    inv.to_csv(OUT_ROOT / "next_independent_candidate_inventory.csv", index=False)
    (OUT_ROOT / "next_candidate_status.txt").write_text(next_id + "\n", encoding="utf-8")

    (OUT_ROOT / "finish_answers.md").write_text(
        render_finish_answers(results, next_id, ovf), encoding="utf-8"
    )

    freeze = {
        "protocol_version": PROTOCOL_VERSION,
        "status": PROTOCOL_STATUS,
        "freeze_date": FREEZE_DATE,
        "frozen_utc": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "factors_reevaluated": [r["spec"].factor_id for r in results],
        "factor_grades": {r["spec"].factor_id: r["factor_grade"] for r in results},
        "long_only_statuses": {
            r["spec"].factor_id: r["long_only_status"] for r in results
        },
        "turnover_convention": {
            "l1_traded_notional": "sum|dw|",
            "oneway_turnover": "0.5 * l1",
            "cost": "l1 * 7.5bps = oneway * 15bps",
            "HIGH_TURNOVER_ONEWAY_DAILY": HIGH_TURNOVER_ONEWAY_DAILY,
            "HIGH_TURNOVER_L1_DAILY_equiv": HIGH_TURNOVER_L1_DAILY,
        },
        "thresholds_frozen": True,
        "no_further_threshold_edits": True,
        "next_candidate": next_id,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT_ROOT / "freeze_registry.json").write_text(
        json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\n===== Protocol v2.0 FINAL =====")
    for r in results:
        print(
            f"{r['spec'].factor_id}: grade={r['factor_grade']} "
            f"long_only={r['long_only_status']} "
            f"parity={r['parity']['pass']}"
        )
    print(f"next: {next_id}")
    print(f"artifacts -> {OUT_ROOT} ({freeze['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
