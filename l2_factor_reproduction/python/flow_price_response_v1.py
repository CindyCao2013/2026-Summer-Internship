"""Sprint 9 — Flow × Price Response / Absorption Family v1.

Hard boundaries:
- Read only trade_flow_daily + price_formation_daily (+ fast_context).
- Discovery window 2023-01-01 .. 2024-12-31; T+1; each formula once.
- No Protocol/threshold edits, no Raw Tick/SSL2, no new primitives.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Factor_Dev_Lib import calAnnuRet, calMDD, calSharpe
from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.evaluation_protocol_v2 import l1_to_oneway
from l2_factor_reproduction.python.fast_discovery import (
    DISCOVERY_END,
    DISCOVERY_START,
    PRIMITIVE_BUFFER_DAYS,
    compute_fast_metrics,
    ensure_effective_group_pnl,
    gate_label,
    load_fast_context,
    save_fast_plots,
)
from l2_factor_reproduction.python.low_turnover_v1 import (
    load_primitive_panel,
    _to_narrow,
)

EPS = 1e-12
ROLLING_DAYS = 5
OUT_DIR = Path(RESULT_ROOT) / "fast_discovery" / "flow_price_response_v1"
_PRIMITIVES = Path(RESULT_ROOT) / "primitives"

TRADE_FLOW_COLS = ("active_buy_amt", "active_sell_amt", "total_amt")
# No literal intraday_return; use frozen open_to_close_return.
INTRADAY_PROXY = "open_to_close_return"
PRICE_COLS = (INTRADAY_PROXY,)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mechanism: str
    builder: Callable[[pd.DataFrame], pd.Series]
    description: str


def _crank(series: pd.Series, dates: pd.Series) -> pd.Series:
    """Point-in-time cross-sectional pct_rank − 0.5."""
    return series.groupby(dates, sort=False).rank(pct=True, method="average") - 0.5


def _rolling_mean_nd(
    values: pd.Series,
    symbols: pd.Series,
    window: int = ROLLING_DAYS,
) -> pd.Series:
    return values.groupby(symbols, sort=False).transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )


def _rolling_pos_frac(
    values: pd.Series,
    symbols: pd.Series,
    window: int = ROLLING_DAYS,
) -> pd.Series:
    """mean(value > 0) over rolling window, min_periods=window."""

    def _frac(x: np.ndarray) -> float:
        if np.any(~np.isfinite(x)):
            # rolling windows from pandas already require min_periods;
            # still guard NaNs inside.
            finite = np.isfinite(x)
            if finite.sum() < window:
                return np.nan
            return float(np.mean(x[finite] > 0))
        return float(np.mean(x > 0))

    return values.groupby(symbols, sort=False).transform(
        lambda s: s.rolling(window, min_periods=window).apply(_frac, raw=True)
    )


def build_merged_panel(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    flow = load_primitive_panel(
        _PRIMITIVES / "trade_flow_daily",
        list(TRADE_FLOW_COLS),
        start,
        end,
        buffer_days=PRIMITIVE_BUFFER_DAYS,
    )
    # trade_flow may live as a single parquet (not dataset/); loader handles both.
    price = load_primitive_panel(
        _PRIMITIVES / "price_formation_daily" / "dataset",
        list(PRICE_COLS),
        start,
        end,
        buffer_days=PRIMITIVE_BUFFER_DAYS,
    )
    merged = flow.merge(
        price.loc[:, ["symbol", "TradeDate", INTRADAY_PROXY]],
        on=["symbol", "TradeDate"],
        how="inner",
    )
    merged = merged.sort_values(["symbol", "TradeDate"], kind="stable").reset_index(
        drop=True
    )
    buy = pd.to_numeric(merged["active_buy_amt"], errors="coerce")
    sell = pd.to_numeric(merged["active_sell_amt"], errors="coerce")
    denom = buy + sell
    valid = denom > 0
    merged["buy_intensity"] = (buy / denom).where(valid)
    merged["sell_intensity"] = (sell / denom).where(valid)
    merged["net_flow"] = ((buy - sell) / denom).where(valid)
    merged["intraday_return"] = pd.to_numeric(
        merged[INTRADAY_PROXY], errors="coerce"
    )
    return merged


def build_capability() -> pd.DataFrame:
    rows = []
    # Probe schemas via panel load on a thin window
    start, end = DISCOVERY_START, DISCOVERY_END
    try:
        flow = load_primitive_panel(
            _PRIMITIVES / "trade_flow_daily",
            list(TRADE_FLOW_COLS),
            start,
            end,
            buffer_days=5,
        )
        flow_ok = True
        flow_reason = ""
        flow_n = int(len(flow))
        flow_cols_present = list(TRADE_FLOW_COLS)
    except Exception as exc:  # noqa: BLE001
        flow_ok = False
        flow_reason = str(exc)
        flow_n = 0
        flow_cols_present = []

    try:
        price = load_primitive_panel(
            _PRIMITIVES / "price_formation_daily" / "dataset",
            list(PRICE_COLS),
            start,
            end,
            buffer_days=5,
        )
        price_ok = True
        price_reason = ""
        price_n = int(len(price))
        # document that intraday_return is proxied
        has_literal_intraday = False
    except Exception as exc:  # noqa: BLE001
        price_ok = False
        price_reason = str(exc)
        price_n = 0
        has_literal_intraday = False

    rows.append(
        {
            "primitive": "trade_flow_daily",
            "required_fields": ",".join(TRADE_FLOW_COLS),
            "available": flow_ok,
            "n_rows_discovery_probe": flow_n,
            "fields_found": ",".join(flow_cols_present),
            "notes": flow_reason or "active_buy/sell/total_amt present",
        }
    )
    rows.append(
        {
            "primitive": "price_formation_daily",
            "required_fields": "intraday_return OR open_to_close_return",
            "available": price_ok,
            "n_rows_discovery_probe": price_n,
            "fields_found": INTRADAY_PROXY if price_ok else "",
            "notes": (
                "literal intraday_return absent; "
                f"using frozen proxy `{INTRADAY_PROXY}`"
                if price_ok and not has_literal_intraday
                else price_reason
            ),
        }
    )
    rows.append(
        {
            "primitive": "derived_intensity_net_flow",
            "required_fields": "buy_intensity,sell_intensity,net_flow",
            "available": bool(flow_ok and price_ok),
            "n_rows_discovery_probe": flow_n if flow_ok else 0,
            "fields_found": "buy_intensity,sell_intensity,net_flow,intraday_return",
            "notes": (
                "denom=(active_buy_amt+active_sell_amt); denom<=0 → NA; "
                f"intraday_return := {INTRADAY_PROXY}"
            ),
        }
    )
    return pd.DataFrame(rows)


def _daily_sell_absorption(panel: pd.DataFrame) -> pd.Series:
    dates = panel["TradeDate"]
    return _crank(panel["intraday_return"], dates) + _crank(
        panel["sell_intensity"], dates
    )


def build_sell_absorption_5d(panel: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(_daily_sell_absorption(panel), panel["symbol"])


def build_buy_exhaustion_5d(panel: pd.DataFrame) -> pd.Series:
    dates = panel["TradeDate"]
    daily = _crank(panel["buy_intensity"], dates) - _crank(
        panel["intraday_return"], dates
    )
    return _rolling_mean_nd(daily, panel["symbol"])


def build_flow_price_divergence_5d(panel: pd.DataFrame) -> pd.Series:
    dates = panel["TradeDate"]
    daily = _crank(panel["intraday_return"], dates) - _crank(
        panel["net_flow"], dates
    )
    return _rolling_mean_nd(daily, panel["symbol"])


def build_flow_price_residual_5d(panel: pd.DataFrame) -> pd.Series:
    """Daily CS OLS: ret = a + b*net_flow + eps; factor = rolling_mean_5d(eps)."""
    dates = panel["TradeDate"]
    ret = panel["intraday_return"].astype(float)
    flow = panel["net_flow"].astype(float)

    def _resid_one_day(idx: np.ndarray) -> np.ndarray:
        y = ret.iloc[idx].to_numpy(dtype=float)
        x = flow.iloc[idx].to_numpy(dtype=float)
        out = np.full(len(idx), np.nan, dtype=float)
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 3:
            return out
        yy = y[mask]
        xx = x[mask]
        x_des = np.column_stack([np.ones(len(xx)), xx])
        try:
            beta, *_ = np.linalg.lstsq(x_des, yy, rcond=None)
        except np.linalg.LinAlgError:
            return out
        fitted = beta[0] + beta[1] * xx
        out[np.where(mask)[0]] = yy - fitted
        return out

    # groupby TradeDate indices
    residual = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, idx in panel.groupby(dates, sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        residual.iloc[idx_arr] = _resid_one_day(idx_arr)
    return _rolling_mean_nd(residual, panel["symbol"])


def build_sell_absorption_consistency_5d(panel: pd.DataFrame) -> pd.Series:
    sa = _daily_sell_absorption(panel)
    mean_sa = _rolling_mean_nd(sa, panel["symbol"])
    consistency = _rolling_pos_frac(sa, panel["symbol"])
    return mean_sa * consistency


CANDIDATES: List[CandidateSpec] = [
    CandidateSpec(
        "sell_absorption_5d",
        "flow_price_response",
        build_sell_absorption_5d,
        "crank(ret)+crank(sell_intensity); heavy selling with resilient price",
    ),
    CandidateSpec(
        "buy_exhaustion_5d",
        "flow_price_response",
        build_buy_exhaustion_5d,
        "crank(buy_intensity)-crank(ret); heavy buying with weak price response",
    ),
    CandidateSpec(
        "flow_price_divergence_5d",
        "flow_price_response",
        build_flow_price_divergence_5d,
        "crank(ret)-crank(net_flow); price vs implied flow",
    ),
    CandidateSpec(
        "flow_price_residual_5d",
        "flow_price_response",
        build_flow_price_residual_5d,
        "rolling_mean_5d of CS OLS residual ret~net_flow",
    ),
    CandidateSpec(
        "sell_absorption_consistency_5d",
        "flow_price_response",
        build_sell_absorption_consistency_5d,
        "mean_SA_5d * mean(SA>0 over 5d)",
    ),
]


def _group_annu(pnl: pd.DataFrame, col: str) -> float:
    if col not in pnl.columns:
        return float("nan")
    return float(calAnnuRet(pnl[col]))


def contribution_row(
    factor: str,
    group_pnl: pd.DataFrame,
    metrics: Dict[str, float],
) -> Dict[str, float]:
    """H-L ≈ G10_excess + (−G1_excess) on already-excess group_pnl."""
    pnl = ensure_effective_group_pnl(group_pnl)
    cols = [c for c in pnl.columns if c != "H-L"]
    cols = sorted(cols, key=lambda c: int(c))
    g1, g10 = cols[0], cols[-1]
    g10_ex = pnl[g10]
    g1_ex = pnl[g1]
    long_contrib = g10_ex  # excess vs benchmark
    short_contrib = -g1_ex  # benchmark − G1_abs
    hl = pnl["H-L"]
    # identity check on daily means / annual
    long_annu = float(calAnnuRet(long_contrib))
    short_annu = float(calAnnuRet(short_contrib))
    hl_annu = float(calAnnuRet(hl))
    sum_annu = float(calAnnuRet(long_contrib + short_contrib))
    return {
        "factor": factor,
        "g10_excess_annual": long_annu,
        "g1_excess_annual": float(calAnnuRet(g1_ex)),
        "hl_annual": hl_annu,
        "long_contribution_annual": long_annu,
        "short_contribution_annual": short_annu,
        "long_plus_short_annual": sum_annu,
        "identity_gap_annual": hl_annu - sum_annu,
        "g10_excess_sharpe": float(calSharpe(g10_ex)),
        "g1_excess_sharpe": float(calSharpe(g1_ex)),
        "short_leg_share_abs": (
            abs(short_annu) / (abs(long_annu) + abs(short_annu))
            if (abs(long_annu) + abs(short_annu)) > 0
            else float("nan")
        ),
        "hl_sharpe": float(metrics.get("hl_sharpe", calSharpe(hl))),
    }


def enrich_metrics(
    group_pnl: pd.DataFrame,
    group_to: pd.DataFrame,
    metrics: Dict[str, float],
) -> Dict[str, float]:
    pnl = ensure_effective_group_pnl(group_pnl)
    to = group_to.copy()
    to.index = pd.to_datetime(to.index)
    to.columns = [str(c) for c in to.columns]
    cols = sorted([c for c in pnl.columns if c != "H-L"], key=lambda c: int(c))
    g1, g10 = cols[0], cols[-1]
    out = dict(metrics)
    out["g10_gross_excess_annual"] = float(calAnnuRet(pnl[g10]))
    out["g1_gross_excess_annual"] = float(calAnnuRet(pnl[g1]))
    out["g10_excess_sharpe"] = float(calSharpe(pnl[g10]))
    # one-way = 0.5 * L1 (Protocol v2.0 reporting)
    l1_hl = float(to["H-L"].reindex(pnl.index).mean())
    l1_g10 = float(to[g10].reindex(pnl.index).mean())
    out["avg_hl_l1_traded_notional"] = l1_hl
    out["avg_g10_l1_traded_notional"] = l1_g10
    out["avg_hl_oneway_turnover"] = l1_to_oneway(l1_hl)
    out["avg_g10_oneway_turnover"] = l1_to_oneway(l1_g10)
    # keep legacy key as L1 for plot box compatibility; also store oneway in metrics
    out["avg_hl_turnover"] = l1_hl
    return out


def run_sprint9(*, output_root: Optional[Path] = None) -> Tuple[pd.DataFrame, ...]:
    out_root = Path(output_root) if output_root else OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("[0] primitive capability", flush=True)
    capability = build_capability()
    capability.to_csv(out_root / "primitive_capability.csv", index=False)
    if not bool(capability.loc[capability.primitive == "derived_intensity_net_flow", "available"].iloc[0]):
        raise RuntimeError("Sprint 9 primitives unavailable — see primitive_capability.csv")

    print("[1] load discovery fast_context + merged panel", flush=True)
    mask, ret = load_fast_context("discovery")
    panel = build_merged_panel(DISCOVERY_START, DISCOVERY_END)
    # Restrict factor construction dates to discovery (buffer already in load)
    panel = panel.loc[
        panel["TradeDate"].between(DISCOVERY_START, DISCOVERY_END)
    ].reset_index(drop=True)
    print(f"  panel rows={len(panel):,} names/day≈{panel.groupby('TradeDate').size().mean():.0f}", flush=True)

    summary_rows: List[Dict[str, object]] = []
    contrib_rows: List[Dict[str, float]] = []
    profile_rows: List[Dict[str, object]] = []

    for spec in CANDIDATES:
        print(f"\n=== {spec.name} ===", flush=True)
        t1 = time.perf_counter()
        values = spec.builder(panel)
        narrow = _to_narrow(panel["symbol"], panel["TradeDate"], values, spec.name)
        group_pnl, group_to, _rank_ic, summary = backtest_factor(
            narrow,
            start_day=DISCOVERY_START,
            end_day=DISCOVERY_END,
            mask=mask,
            ret_matrix=ret,
        )
        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        metrics = enrich_metrics(group_pnl, group_to, metrics)
        gate = gate_label(metrics)
        elapsed = time.perf_counter() - t1

        fig_dir = out_root / "figures" / spec.name
        save_fast_plots(fig_dir, spec.name, group_pnl, metrics)

        contrib = contribution_row(spec.name, group_pnl, metrics)
        contrib_rows.append(contrib)

        row = {
            "factor": spec.name,
            "mechanism": spec.mechanism,
            "source_primitive": "trade_flow_daily+price_formation_daily",
            "window": "discovery",
            "gate": gate,
            "description": spec.description,
            "rank_ic_mean_raw": metrics["rank_ic_mean_raw"],
            "icir_raw": metrics["icir_raw"],
            "hl_annu_ret": metrics["hl_annu_ret"],
            "hl_sharpe": metrics["hl_sharpe"],
            "hl_mdd": metrics["hl_mdd"],
            "decile_mono_spearman": metrics["decile_mono_spearman"],
            "adjacent_violations": metrics["adjacent_violations"],
            "positive_hl_month_fraction": metrics["positive_hl_month_fraction"],
            "cum_hl_time_spearman": metrics["cum_hl_time_spearman"],
            "g10_gross_excess_annual": metrics["g10_gross_excess_annual"],
            "g10_excess_sharpe": metrics["g10_excess_sharpe"],
            "g1_gross_excess_annual": metrics["g1_gross_excess_annual"],
            "avg_g10_oneway_turnover": metrics["avg_g10_oneway_turnover"],
            "avg_hl_oneway_turnover": metrics["avg_hl_oneway_turnover"],
            "avg_hl_l1_traded_notional": metrics["avg_hl_l1_traded_notional"],
            "factor_direction": metrics["factor_direction"],
            "n_days": metrics["n_days"],
            "n_names_avg": metrics["n_names_avg"],
            "long_contribution_annual": contrib["long_contribution_annual"],
            "short_contribution_annual": contrib["short_contribution_annual"],
            "short_leg_share_abs": contrib["short_leg_share_abs"],
            "identity_gap_annual": contrib["identity_gap_annual"],
            "elapsed_seconds": round(elapsed, 2),
        }
        summary_rows.append(row)
        profile_rows.append(
            {
                "factor": spec.name,
                "elapsed_seconds": round(elapsed, 2),
                "n_factor_rows": int(len(narrow)),
                "gate": gate,
            }
        )
        print(
            f"  gate={gate} Sharpe={metrics['hl_sharpe']:.2f} "
            f"mono={metrics['decile_mono_spearman']:.3f} "
            f"viol={metrics['adjacent_violations']} "
            f"G10_annu={metrics['g10_gross_excess_annual']:.2%} "
            f"short_share={contrib['short_leg_share_abs']:.2f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    contrib_df = pd.DataFrame(contrib_rows)
    profile = pd.DataFrame(profile_rows)
    summary.to_csv(out_root / "candidate_summary.csv", index=False)
    contrib_df.to_csv(out_root / "contribution_decomposition.csv", index=False)
    profile.to_csv(out_root / "fast_profile.csv", index=False)

    selection = select_next(summary)
    report = render_report(summary, contrib_df, capability, selection)
    (out_root / "report.md").write_text(report, encoding="utf-8")

    if selection["status"] == "HAS_STRONG":
        (out_root / "next_full_validation_candidate.md").write_text(
            selection["next_md"], encoding="utf-8"
        )
    else:
        (out_root / "SPRINT9_NO_STRONG_CANDIDATE").write_text(
            "SPRINT9_NO_STRONG_CANDIDATE\n", encoding="utf-8"
        )

    manifest = {
        "sprint": "Sprint 9 — Flow × Price Response / Absorption Family v1",
        "status": "FROZEN" if selection["status"] != "HAS_STRONG" else "AWAITING_FULL_VALIDATION_CONFIRM",
        "discovery_window": [str(DISCOVERY_START.date()), str(DISCOVERY_END.date())],
        "n_candidates": len(CANDIDATES),
        "selection": selection["status"],
        "next_candidate": selection.get("factor"),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "intraday_return_proxy": INTRADAY_PROXY,
        "protocol_ref": "evaluation_protocol_v2.0 (untouched)",
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\n[done] selection={selection['status']} "
        f"next={selection.get('factor')} -> {out_root} "
        f"({manifest['elapsed_seconds']}s)",
        flush=True,
    )
    return summary, contrib_df, profile, capability


def select_next(summary: pd.DataFrame) -> Dict[str, object]:
    strong = summary.loc[summary["gate"] == "strong_candidate"].copy()
    if strong.empty:
        return {"status": "SPRINT9_NO_STRONG_CANDIDATE", "factor": None, "next_md": ""}

    # Ranking: G10 excess, lower short-leg dependence, lower one-way TO
    strong = strong.sort_values(
        by=[
            "g10_gross_excess_annual",
            "short_leg_share_abs",
            "avg_hl_oneway_turnover",
        ],
        ascending=[False, True, True],
    )
    best = strong.iloc[0]
    fid = str(best["factor"])
    md = "\n".join(
        [
            f"# Next Full Validation Candidate — Sprint 9",
            "",
            f"**factor:** `{fid}`",
            "",
            "Awaiting human confirmation. Do **not** auto Full Validate.",
            "",
            "## Discovery metrics",
            "",
            f"- gate = `{best['gate']}`",
            f"- gross H-L Sharpe = `{best['hl_sharpe']:.3f}`",
            f"- mono / violations = `{best['decile_mono_spearman']:.3f}` / `{int(best['adjacent_violations'])}`",
            f"- H-L annual = `{best['hl_annu_ret']:.2%}`",
            f"- G10 excess annual / Sharpe = `{best['g10_gross_excess_annual']:.2%}` / `{best['g10_excess_sharpe']:.2f}`",
            f"- G1 excess annual = `{best['g1_gross_excess_annual']:.2%}`",
            f"- short_leg_share_abs = `{best['short_leg_share_abs']:.3f}`",
            f"- H-L one-way turnover = `{best['avg_hl_oneway_turnover']:.3f}`",
            f"- G10 one-way turnover = `{best['avg_g10_oneway_turnover']:.3f}`",
            f"- factor_direction = `{int(best['factor_direction'])}`",
            "",
            "## Selection rationale",
            "",
            "1. Passed STRONG gate",
            "2. Highest G10 excess among strong",
            "3. Prefer lower short-leg dependence",
            "4. Prefer lower one-way turnover",
            "",
            "## Mechanism",
            "",
            f"{best['description']}",
            "",
        ]
    ) + "\n"
    return {"status": "HAS_STRONG", "factor": fid, "next_md": md}


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        return df.to_string(index=False)


def render_report(
    summary: pd.DataFrame,
    contrib: pd.DataFrame,
    capability: pd.DataFrame,
    selection: Dict[str, object],
) -> str:
    lines = [
        "# Sprint 9 — Flow × Price Response / Absorption Family v1",
        "",
        f"Discovery window: `{DISCOVERY_START.date()}` ~ `{DISCOVERY_END.date()}`",
        f"intraday_return proxy: `{INTRADAY_PROXY}`",
        "Protocol v2.0: untouched. Thresholds: untouched.",
        "",
        "## Primitive capability",
        "",
        _table(capability),
        "",
        "## Candidate summary",
        "",
        _table(
            summary[
                [
                    "factor",
                    "gate",
                    "hl_sharpe",
                    "hl_annu_ret",
                    "decile_mono_spearman",
                    "adjacent_violations",
                    "g10_gross_excess_annual",
                    "g10_excess_sharpe",
                    "g1_gross_excess_annual",
                    "short_leg_share_abs",
                    "avg_hl_oneway_turnover",
                    "avg_g10_oneway_turnover",
                    "factor_direction",
                ]
            ]
        ),
        "",
        "## H-L contribution decomposition",
        "",
        _table(contrib),
        "",
        "## Selection",
        "",
        f"Status: **{selection['status']}**",
        f"Next: `{selection.get('factor')}`",
        "",
        "## Report questions",
        "",
    ]

    strong = summary.loc[summary["gate"] == "strong_candidate"]
    q1 = (
        "YES — " + ", ".join(strong["factor"].astype(str))
        if not strong.empty
        else "NO"
    )

    sell = summary.loc[summary["factor"] == "sell_absorption_5d"]
    if not sell.empty:
        s = sell.iloc[0]
        mono = float(s["decile_mono_spearman"])
        viol = int(s["adjacent_violations"])
        if mono >= 0.85 and viol <= 1:
            q2 = (
                f"Continuous decile structure likely (mono={mono:.3f}, viol={viol}); "
                "not a pure endpoint effect."
            )
        elif mono >= 0.70:
            q2 = (
                f"Partial structure (mono={mono:.3f}, viol={viol}); "
                "some mid-decile noise remains."
            )
        else:
            q2 = (
                f"Weak / endpoint-like (mono={mono:.3f}, viol={viol}); "
                "not a smooth continuous decile."
            )
    else:
        q2 = "sell_absorption_5d unavailable"

    # Alpha source across all / best
    best_hl = summary.sort_values("hl_sharpe", ascending=False).iloc[0]
    short_share = float(best_hl["short_leg_share_abs"])
    long_a = float(best_hl["long_contribution_annual"])
    short_a = float(best_hl["short_contribution_annual"])
    if abs(short_a) > abs(long_a) * 1.25:
        q3 = (
            f"Primarily G1 short leg for top H-L `{best_hl['factor']}` "
            f"(short_annu={short_a:.2%}, long_annu={long_a:.2%}, "
            f"short_share={short_share:.2f})."
        )
    elif abs(long_a) > abs(short_a) * 1.25:
        q3 = (
            f"Primarily G10 long leg for top H-L `{best_hl['factor']}` "
            f"(long_annu={long_a:.2%}, short_annu={short_a:.2%})."
        )
    else:
        q3 = (
            f"Mixed long+short for top H-L `{best_hl['factor']}` "
            f"(long={long_a:.2%}, short={short_a:.2%}, share={short_share:.2f})."
        )

    best_g10 = summary.sort_values("g10_gross_excess_annual", ascending=False).iloc[0]
    q4 = (
        f"`{best_g10['factor']}` "
        f"(G10 excess annual={best_g10['g10_gross_excess_annual']:.2%}, "
        f"Sharpe={best_g10['g10_excess_sharpe']:.2f})."
    )

    both = summary.loc[
        (summary["gate"] == "strong_candidate")
        & (summary["g10_gross_excess_annual"] > 0.05)
    ]
    if not both.empty:
        q5 = "YES — " + ", ".join(
            f"{r.factor} (G10={r.g10_gross_excess_annual:.2%})"
            for r in both.itertuples()
        )
    else:
        near = summary.loc[
            (summary["hl_sharpe"] >= 2.5) & (summary["g10_gross_excess_annual"] > 0)
        ]
        q5 = (
            "NO strong∩meaningful-G10 pair. "
            + (
                "Near: " + ", ".join(near["factor"].astype(str))
                if not near.empty
                else "No near misses with positive G10."
            )
        )

    q6 = (
        str(selection["factor"])
        if selection["status"] == "HAS_STRONG"
        else "SPRINT9_NO_STRONG_CANDIDATE"
    )

    lines += [
        f"1. H-L Sharpe≥3 & mono≥0.85? **{q1}**",
        f"2. sell_absorption continuous vs endpoint? **{q2}**",
        f"3. Alpha from G10 vs G1? **{q3}**",
        f"4. Strongest G10 excess? **{q4}**",
        f"5. Strong H-L + meaningful G10? **{q5}**",
        f"6. Next Full Validation? **{q6}**",
        "",
    ]
    return "\n".join(lines) + "\n"
