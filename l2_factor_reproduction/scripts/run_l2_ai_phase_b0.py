"""Phase B0/B1 runner: timing closure, label parity, tiny residual/ratio smokes.

Does not mutate candidate_pool_v1, FS-1/FS-3/FS-4 frozen artifacts,
or launch 2023-2024 discovery / full-history FS/ML / GRU / AlphaNet.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.feature_selection.labels import (  # noqa: E402
    build_labels_wide_panel,
    load_daily_excess_and_bench,
    recover_stock_returns,
)
from l2_factor_reproduction.feature_selection.panel_io import (  # noqa: E402
    load_processed_panel_slice,
)
from l2_factor_reproduction.l2_ai_stock_selection.contracts import (  # noqa: E402
    PRODUCTION_LABEL_STATUS,
    RESULT_ROOT,
    TIMING_VERDICT,
    data_contract_dict,
)
from l2_factor_reproduction.l2_ai_stock_selection.execution_timing import (  # noqa: E402
    execution_timing_contract_dict,
    three_date_walkthrough,
)
from l2_factor_reproduction.l2_ai_stock_selection.labels_ai_v1 import (  # noqa: E402
    label_status_payload,
    run_fs3_parity,
    tail_truncation_rows,
)
from l2_factor_reproduction.l2_ai_stock_selection.model_contract import (  # noqa: E402
    LGBM_PARAMS,
    assert_lgbm_baseline_consistent,
    model_contract_dict,
)
from l2_factor_reproduction.l2_ai_stock_selection.nonlinear import (  # noqa: E402
    binned_conditional_return,
    nonlinear_should_review,
    rank_ic,
    residual_mutual_information,
)
from l2_factor_reproduction.l2_ai_stock_selection.paths import (  # noqa: E402
    CANDIDATE_DISCOVERY,
    LABELS,
    RATIO_SMOKE,
    REPORTS,
    TIMING,
    ensure_layout,
)
from l2_factor_reproduction.l2_ai_stock_selection.ratio_catalog import (  # noqa: E402
    NEAR_ZERO_DENOM,
    ratio_diagnostics,
    safe_ratio,
)
from l2_factor_reproduction.l2_ai_stock_selection.residual_alpha import (  # noqa: E402
    candidate_incremental_metrics,
    demean_within_groups,
    pooled_train_window_residual,
    residualize_panel_with_diagnostics,
)
from l2_factor_reproduction.l2_ai_stock_selection.score_availability import (  # noqa: E402
    audit_fs4_holdout,
)
from l2_factor_reproduction.python.fast_discovery import context_paths  # noqa: E402


SMOKE_FACTORS = (
    ("net_buy_ratio", "trade_flow"),
    ("net_buy_count_ratio", "trade_flow"),
    ("mid_order_ratio_4w_20w", "order_size"),
    ("large_order_ratio_20w", "order_size"),
    ("obi_l5_mean", "order_book"),
    ("relative_spread_mean", "order_book"),
    ("overnight_gap", "price_formation"),
    ("realized_volatility", "price_formation"),
    ("signed_amount_impact", "liquidity_impact"),
    ("cancel_value_pressure", "cancel_lifecycle"),
)
R1_L2 = (
    "net_buy_ratio",
    "obi_l5_mean",
    "signed_amount_impact",
    "cancel_value_pressure",
)
FS1_ALIGNED = (
    PROJ_ROOT
    / "research"
    / "results"
    / "l2_reproduction"
    / "feature_selection"
    / "fs1_feature_panel_full"
    / "aligned_raw"
)
MCAP_PATH = (
    PROJ_ROOT
    / "research"
    / "results"
    / "l2_reproduction"
    / "primitives"
    / "mcap_wide_2019-01-01_2026-07-31.parquet"
)
INDUSTRY_PATH = (
    PROJ_ROOT
    / "research"
    / "results"
    / "l2_reproduction"
    / "primitives"
    / "citics_industry_wide_full.parquet"
)
RATIO_MONTH = "2024-06"


def _json_default(obj):
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if pd.isna(obj):
        return None
    raise TypeError(type(obj))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _trading_dates() -> pd.DatetimeIndex:
    p = context_paths("full")["trading_dates"]
    td = pd.read_parquet(p)
    if isinstance(td, pd.DataFrame):
        col = td.columns[0]
        idx = pd.to_datetime(td[col])
    else:
        idx = pd.to_datetime(td.index)
    return pd.DatetimeIndex(idx).normalize().unique().sort_values()


def run_timing(dates: pd.DatetimeIndex) -> dict:
    # Three consecutive dates around 2024-06 for the explicit walkthrough.
    start = pd.Timestamp("2024-06-03")
    loc = int(dates.searchsorted(start, side="left"))
    walk_dates = dates[loc : loc + 4]
    contract = execution_timing_contract_dict(walkthrough_dates=walk_dates)
    _write_json(RESULT_ROOT / "execution_timing_contract.json", contract)
    walk = contract["three_date_walkthrough"]
    lines = [
        "# 07 Execution Timing Audit",
        "",
        "Phase B0. Frozen historical backtests are **not** rewritten.",
        "",
        "## Verdict",
        "",
        "**{}**".format(contract["verdict"]),
        "",
        "This is the AI-v1 production-contract verdict. The frozen L2 stack",
        "still uses `signal.shift(1)` × same-index c2c; that mapping is traced",
        "below and is economically inconsistent with post-close factor availability.",
        "",
        "## 1. Intraday cutoff by family",
        "",
        "| Family | Session window | 14:56–15:00 | Close auction | 15:00 tick | Available |",
        "|---|---|---|---|---|---|",
    ]
    for row in contract["primitive_cutoffs"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["family"],
                row["session_window"],
                row["includes_1456_1500"],
                row["includes_close_auction"],
                row["includes_1500_tick"],
                row["technically_available"],
            )
        )
    lines.extend(
        [
            "",
            "There is **no uniform documented pre-close cutoff** (14:30 / 14:55).",
            "`backtest.py` states the factor is a same-session aggregate known only after close T.",
            "",
            "## 2. Do any factors include 14:56–15:00, close auction, or post-close prints?",
            "",
            "Yes.",
            "",
            "- trade_flow / order_size: `09:30:00 <= ExchTime < 15:00:01` includes **15:00:00**.",
            "- order_book daily means exclude `is_close_auction` but include 14:56–14:59;",
            "  `close_auction_*` uses the 15:00 snapshot; `closing_obi_l5` uses 14:30–14:59.",
            "- price_formation: continuous through 14:59 **and** `close_auction_return` at 15:00.",
            "- liquidity_impact: 09:30–11:29 + 13:00–14:59 (includes 14:56–14:59, not 15:00).",
            "- cancel_lifecycle: minutes 570–689 and 780–899 (through 14:59, not 15:00).",
            "",
            "## 3. Timestamp the factor is technically available",
            "",
            "After session close T. Auction-using formulas are available only after the 15:00 print.",
            "",
            "## 4. Execution timestamp assumed by groupTest",
            "",
            "`prepare_factor_signal` shifts the factor once, then reindexes `ret` onto the",
            "shifted signal. `groupTest` multiplies those already-aligned frames and does",
            "**not** shift again. Implied c2c holding for signal row D starts at Close[D-1].",
            "",
            "## 5. What `signal.shift(1)` actually pairs",
            "",
            "Traced indexes, not the English word “shift”:",
            "",
            "1. `get_Ret_Matrix(..., method='c2c')` builds `ret[D] = Close[D]/PreClose[D]-1`",
            "   = `Close[D]/Close[D-1]-1` (`Factor_Dev_Lib.py`).",
            "2. `signal = factor.shift(1)` so `signal[D] = factor[D-1]`.",
            "3. RankIC / group PnL use `signal[D]` with `ret[D]`.",
            "",
            "Therefore **factor T → Close[T+1]/Close[T]-1**, i.e. Close[T] to Close[T+1],",
            "**not** Close[T+1] to Close[T+2].",
            "",
            "That return **starts at Close[T]**. If factor T is known only after Close[T],",
            "the position cannot be established at Close[T].",
            "",
            "## 6. Three consecutive trading dates",
            "",
        ]
    )
    for rec in walk:
        lines.append(
            "- Factor **{factor_date}** (known after close {factor_date}) → "
            "`signal[{signal_index_date}]` × `{return_formula}`. "
            "Holding starts at Close[{holding_start_close}], which has already passed.".format(**rec)
        )
    lines.extend(
        [
            "",
            "## Required AI-v1 correction (not applied to frozen backtests)",
            "",
            "- T+1 open / VWAP into a subsequent return, **or**",
            "- shift the c2c window one extra session: factor T → Close[T+1] to Close[T+2].",
            "",
            "Do **not** silently retain T+1 c2c as the production AI-v1 label.",
            "",
            "## Machine-readable contract",
            "",
            "`research/results/l2_ai_stock_selection_v1/execution_timing_contract.json`",
            "",
        ]
    )
    (REPORTS / "07_execution_timing_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return contract


def run_score_availability(dates: pd.DatetimeIndex) -> pd.DataFrame:
    audit = audit_fs4_holdout(trading_dates=dates)
    out = TIMING / "score_availability_audit.csv"
    audit.to_csv(out, index=False)
    return audit


def run_labels() -> dict:
    excess, bench, dates = load_daily_excess_and_bench("full")
    rebuilt = build_labels_wide_panel(
        excess, bench, dates, horizons=(1, 3, 5, 10, 20)
    )
    parity = run_fs3_parity(rebuilt=rebuilt)
    parity_path = LABELS / "label_parity.csv"
    parity.to_csv(parity_path, index=False)
    parity_ok = bool(parity["pass"].all())
    status = label_status_payload(parity_ok=parity_ok)
    _write_json(LABELS / "label_status.json", status)
    (LABELS / "DIAGNOSTIC_ONLY.txt").write_text(
        "Y3/Y10 are diagnostic-only under verdict {}.\n"
        "They reuse FS-3 c2c economics and must not be treated as the AI-v1 "
        "production label while T+1 c2c is not executable.\n".format(TIMING_VERDICT),
        encoding="utf-8",
    )
    # Always materialize the files the task asked for; mark diagnostic.
    rebuilt[3].to_parquet(LABELS / "forward_return_3d.parquet")
    rebuilt[10].to_parquet(LABELS / "forward_return_10d.parquet")
    tail = tail_truncation_rows(rebuilt, dates, (3, 10))
    tail.to_csv(LABELS / "tail_truncation_audit.csv", index=False)
    return {
        "parity": parity,
        "parity_ok": parity_ok,
        "status": status,
        "tail": tail,
        "n_dates": int(len(dates)),
        "n_symbols": int(rebuilt[3].shape[1]),
    }


def _wide_from_long(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    w = df.pivot_table(index="TradeDate", columns="Symbol", values=value_col, aggfunc="last")
    w.index = pd.to_datetime(w.index).normalize()
    return w.sort_index()


def _load_style_controls(dates: Sequence, symbols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    symbols = list(symbols)
    excess, bench, all_dates = load_daily_excess_and_bench("full")
    stock = recover_stock_returns(excess, bench)
    # Need 20-day lookback before smoke dates.
    loc0 = int(all_dates.searchsorted(dates.min(), side="left"))
    hist = all_dates[max(0, loc0 - 40) : int(all_dates.searchsorted(dates.max(), side="right"))]
    stock_h = stock.reindex(index=hist, columns=symbols)
    mom = stock_h.rolling(20, min_periods=10).sum().reindex(index=dates)
    vol = stock_h.rolling(20, min_periods=10).std().reindex(index=dates)
    controls = {
        "momentum_20d": mom,
        "volatility_20d": vol,
    }
    if MCAP_PATH.exists():
        mcap = pd.read_parquet(MCAP_PATH)
        mcap.index = pd.to_datetime(mcap.index).normalize()
        ln_m = np.log(mcap.reindex(index=dates, columns=symbols).astype(float))
        controls["ln_mktcap"] = ln_m.replace([np.inf, -np.inf], np.nan)
    if INDUSTRY_PATH.exists():
        ind = pd.read_parquet(INDUSTRY_PATH)
        ind.index = pd.to_datetime(ind.index).normalize()
        controls["industry"] = ind.reindex(index=dates, columns=symbols)
    return controls


def _industry_demean_frame(wide: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns, dtype=float)
    industry = industry.reindex(index=wide.index, columns=wide.columns)
    for dt in wide.index:
        out.loc[dt] = demean_within_groups(wide.loc[dt], industry.loc[dt])
    return out


def run_residual_and_nonlinear_smoke(label_y1: pd.DataFrame) -> dict:
    dates_all = _trading_dates()
    start = pd.Timestamp("2024-06-03")
    loc = int(dates_all.searchsorted(start, side="left"))
    smoke_dates = list(dates_all[loc : loc + 5])
    cols = ["TradeDate", "Symbol"] + [n for n, _ in SMOKE_FACTORS]
    panel = load_processed_panel_slice(
        FS1_ALIGNED,
        smoke_dates[0],
        smoke_dates[-1],
        columns=cols[2:],
    )
    if panel.empty:
        raise RuntimeError("aligned_raw slice for 2024-06 smoke is empty")
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    # Deterministic 50 names with coverage on all 5 dates.
    counts = panel.groupby("Symbol")["TradeDate"].nunique()
    eligible = sorted(counts[counts >= len(smoke_dates)].index.tolist())
    symbols = eligible[:50]
    panel = panel[panel["Symbol"].isin(symbols)]
    factors = {}
    families = {}
    for name, fam in SMOKE_FACTORS:
        if name not in panel.columns:
            continue
        factors[name] = _wide_from_long(panel, name).reindex(index=smoke_dates, columns=symbols)
        families[name] = fam
    leak_probe_dates = list(dates_all[loc : loc + 7])
    y = label_y1.reindex(index=leak_probe_dates, columns=symbols)
    styles = _load_style_controls(smoke_dates, symbols)
    industry = styles.get("industry")
    y_d = _industry_demean_frame(y, industry) if industry is not None else y
    r0_controls = {}
    for key in ("ln_mktcap", "momentum_20d", "volatility_20d"):
        if key in styles:
            x = styles[key].reindex(index=smoke_dates, columns=symbols)
            r0_controls[key] = _industry_demean_frame(x, industry) if industry is not None else x
    turnover_status = "UNAVAILABLE"
    # Turnover requires DDB; skip rather than block the smoke.
    r0, r0_diag = residualize_panel_with_diagnostics(
        y_d, r0_controls, train_dates=smoke_dates, min_obs=20
    )
    r1_controls = dict(r0_controls)
    for name in R1_L2:
        if name not in factors:
            continue
        x = factors[name].reindex(index=smoke_dates, columns=symbols)
        cov = float(np.isfinite(x.to_numpy(dtype=float)).mean())
        if cov < 0.80:
            continue
        r1_controls["l2_" + name] = (
            _industry_demean_frame(x, industry) if industry is not None else x
        )
    r1, r1_diag = residualize_panel_with_diagnostics(
        y_d, r1_controls, train_dates=smoke_dates, min_obs=20
    )
    pooled = pooled_train_window_residual(
        y_d, r0_controls, train_dates=smoke_dates, min_obs=20
    )
    oos_idx = [d for d in r0.index if d not in set(smoke_dates)]
    leak_r0 = int(r0.loc[oos_idx].notna().to_numpy().sum()) if oos_idx else 0
    rows = []
    nl_rows = []
    for name, wide in factors.items():
        m0 = candidate_incremental_metrics(wide, y, r0, train_dates=smoke_dates)
        m1 = candidate_incremental_metrics(wide, y, r1, train_dates=smoke_dates)
        mp = candidate_incremental_metrics(wide, y, pooled, train_dates=smoke_dates)
        n_obs = int(np.isfinite(wide.reindex(index=smoke_dates, columns=symbols).to_numpy()).sum())
        rec = {
            "factor": name,
            "family": families.get(name, ""),
            "raw_rank_ic": m0["raw_rank_ic"],
            "residual_rank_ic_R0": m0["residual_rank_ic"],
            "residual_rank_ic_R1": m1["residual_rank_ic"],
            "residual_rank_ic_pooled_R0": mp["residual_rank_ic"],
            "raw_MI": m0["raw_mi"],
            "residual_MI_R0": m0["residual_mi"],
            "residual_MI_R1": m1["residual_mi"],
            "n_obs": n_obs,
            "n_dates": len(smoke_dates),
            "n_symbols": len(symbols),
            "condition_number_R0_median": float(pd.to_numeric(r0_diag.get("condition_number"), errors="coerce").median()) if len(r0_diag) else float("nan"),
            "condition_number_R1_median": float(pd.to_numeric(r1_diag.get("condition_number"), errors="coerce").median()) if len(r1_diag) else float("nan"),
            "residual_std_R0_median": float(pd.to_numeric(r0_diag.get("residual_std"), errors="coerce").median()) if len(r0_diag) else float("nan"),
            "residual_std_R1_median": float(pd.to_numeric(r1_diag.get("residual_std"), errors="coerce").median()) if len(r1_diag) else float("nan"),
            "residual_mean_R0_absmax": float(pd.to_numeric(r0_diag.get("residual_mean"), errors="coerce").abs().max()) if len(r0_diag) else float("nan"),
            "oos_residual_nonzero_count": leak_r0,
            "turnover_20d": turnover_status,
            "note": "mechanics only; 5 dates are not alpha evidence",
        }
        rows.append(rec)
        review = nonlinear_should_review(m0["raw_rank_ic"], m0["residual_mi"])
        bins = binned_conditional_return(wide.reindex(index=smoke_dates, columns=symbols), y)
        nl_rows.append(
            {
                "factor": name,
                "family": families.get(name, ""),
                "RankIC": m0["raw_rank_ic"],
                "MutualInformation": m0["raw_mi"],
                "residual_MI_R0": m0["residual_mi"],
                "binned_mean_y_low": float(bins["mean_y"].iloc[0]) if len(bins) else float("nan"),
                "binned_mean_y_high": float(bins["mean_y"].iloc[-1]) if len(bins) else float("nan"),
                "n_bins": int(len(bins)),
                "flag": "NONLINEAR_REVIEW" if review else "NO_NONLINEAR_FLAG",
                "auto_keep": False,
                "note": "weak RankIC + strong MI → REVIEW, never automatic KEEP; 5 dates ≠ alpha",
            }
        )
    residual_tbl = pd.DataFrame(rows)
    nonlinear_tbl = pd.DataFrame(nl_rows)
    residual_tbl.to_csv(CANDIDATE_DISCOVERY / "residual_alpha_smoke.csv", index=False)
    nonlinear_tbl.to_csv(CANDIDATE_DISCOVERY / "nonlinear_smoke.csv", index=False)
    r0_diag.to_csv(CANDIDATE_DISCOVERY / "residual_r0_diagnostics.csv", index=False)
    r1_diag.to_csv(CANDIDATE_DISCOVERY / "residual_r1_diagnostics.csv", index=False)
    return {
        "residual": residual_tbl,
        "nonlinear": nonlinear_tbl,
        "smoke_dates": [str(pd.Timestamp(d).date()) for d in smoke_dates],
        "n_symbols": len(symbols),
        "oos_residual_nonzero_count": leak_r0,
        "turnover_20d": turnover_status,
        "r0_controls": list(r0_controls),
        "r1_controls": list(r1_controls),
    }


def run_ratio_smoke() -> pd.DataFrame:
    dates_all = _trading_dates()
    month_dates = dates_all[(dates_all >= "2024-06-01") & (dates_all <= "2024-06-30")]
    cols = [
        "signed_amount_impact",
        "relative_spread_mean",
        "realized_volatility",
    ]
    panel = load_processed_panel_slice(FS1_ALIGNED, month_dates.min(), month_dates.max(), columns=cols)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    specs = [
        {
            "name": "impact_over_realized_vol",
            "num": "signed_amount_impact",
            "den": "realized_volatility",
            "family": "liquidity_impact x price_formation",
            "meaning": "Price impact per unit of realized volatility.",
        },
        {
            "name": "spread_over_realized_vol",
            "num": "relative_spread_mean",
            "den": "realized_volatility",
            "family": "order_book x price_formation",
            "meaning": "Quoted spread in volatility units.",
        },
    ]
    rows = []
    for spec in specs:
        num = panel[spec["num"]].to_numpy(dtype=float)
        den = panel[spec["den"]].to_numpy(dtype=float)
        ratio = safe_ratio(num, den, eps=NEAR_ZERO_DENOM)
        tmp = panel[["TradeDate", "Symbol"]].copy()
        tmp["ratio"] = ratio
        wide = tmp.pivot_table(index="TradeDate", columns="Symbol", values="ratio", aggfunc="last")
        cs = wide.notna().mean(axis=1).mean() if wide.size else float("nan")
        rec = ratio_diagnostics(
            spec["name"],
            num,
            den,
            ratio,
            family=spec["family"],
            economic_meaning=spec["meaning"],
            coverage_n_dates=int(wide.shape[0]),
            coverage_n_symbols=int(wide.shape[1]),
            n_cs_with_finite=float(cs),
        )
        rec["numerator"] = spec["num"]
        rec["denominator"] = spec["den"]
        rec["month"] = RATIO_MONTH
        rec["inserted_into_candidate_pool_v1"] = False
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(RATIO_SMOKE / "ratio_diagnostics.csv", index=False)
    return out


def write_smoke_report(payload: dict) -> None:
    parity = payload["labels"]["parity"]
    score = payload["score_audit"]
    residual = payload["smoke"]["residual"]
    nonlinear = payload["smoke"]["nonlinear"]
    ratios = payload["ratios"]
    lines = [
        "# 08 Phase B0/B1 Smoke Report",
        "",
        "Mechanics-only run. Five dates are **not** alpha evidence.",
        "Frozen FS-1 / FS-3 / FS-4 / candidate_pool_v1 were not rewritten.",
        "",
        "## A. Execution timing",
        "",
        "Verdict: **{}**".format(payload["timing"]["verdict"]),
        "",
        "See `reports/07_execution_timing_audit.md`.",
        "",
        "## B. Label parity",
        "",
        parity.to_string(index=False),
        "",
        "Y3/Y10 status: **{}**".format(payload["labels"]["status"]["status"]),
        "",
        "## C. FS-4 score availability (adapter; frozen files untouched)",
        "",
        "Frozen `n_invalid_score_dates` sum = {}.".format(int(score["n_invalid_score_dates"].sum())),
        "After AI-v1 filter `score_date > train_label_end_max`, invalid count = {}.".format(
            int(score["n_invalid_after_ai_v1_filter"].sum())
        ),
        "",
        score[
            [
                "route",
                "refit_anchor",
                "train_end",
                "train_label_end_max",
                "first_allowed_score_date",
                "actual_first_score_date",
                "n_invalid_score_dates",
                "n_invalid_after_ai_v1_filter",
            ]
        ].to_string(index=False),
        "",
        "## D. Residual-alpha smoke",
        "",
        "Canonical residual = **A** (cross-sectional per date). Pooled residual is a separate column.",
        "Dates: {}. Symbols/date: {}. turnover_20d: {}.".format(
            payload["smoke"]["smoke_dates"],
            payload["smoke"]["n_symbols"],
            payload["smoke"]["turnover_20d"],
        ),
        "R0 controls: {}. R1 adds {}.".format(
            payload["smoke"]["r0_controls"],
            [c for c in payload["smoke"]["r1_controls"] if c.startswith("l2_")],
        ),
        "OOS residual nonzero count: {} (must be 0).".format(payload["smoke"]["oos_residual_nonzero_count"]),
        "",
        residual.to_string(index=False),
        "",
        "## E. Nonlinear smoke",
        "",
        "Weak RankIC + strong MI → NONLINEAR_REVIEW, never automatic KEEP.",
        "",
        nonlinear.to_string(index=False),
        "",
        "## F. Ratio smoke (2024-06, not inserted into candidate_pool_v1)",
        "",
        ratios.to_string(index=False),
        "",
        "## G. LightGBM / jury contract",
        "",
        "num_leaves={}, max_depth={}. Jury states: DROP / REVIEW / KEEP.".format(
            LGBM_PARAMS["num_leaves"], LGBM_PARAMS["max_depth"]
        ),
        "Nonlinear override → REVIEW_NONLINEAR. Tree gain alone → never KEEP.",
        "",
        "## Stop conditions / Phase B2",
        "",
        "Phase B2 must not start while execution timing is C2C_TPLUS1_NOT_EXECUTABLE.",
        "",
        "Runtime seconds: {:.1f}".format(payload["runtime_sec"]),
        "",
    ]
    (REPORTS / "08_phase_b_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    t0 = time.perf_counter()
    ensure_layout()
    assert_lgbm_baseline_consistent()
    dates = _trading_dates()
    errors = []
    timing = run_timing(dates)
    try:
        score_audit = run_score_availability(dates)
    except Exception as exc:
        errors.append("score_availability: {}".format(exc))
        traceback.print_exc()
        score_audit = pd.DataFrame()
    try:
        labels = run_labels()
    except Exception as exc:
        errors.append("labels: {}".format(exc))
        traceback.print_exc()
        labels = {"parity": pd.DataFrame(), "parity_ok": False, "status": {}, "tail": pd.DataFrame()}
    smoke = {
        "residual": pd.DataFrame(),
        "nonlinear": pd.DataFrame(),
        "smoke_dates": [],
        "n_symbols": 0,
        "oos_residual_nonzero_count": -1,
        "turnover_20d": "UNAVAILABLE",
        "r0_controls": [],
        "r1_controls": [],
    }
    try:
        y1 = labels.get("parity")
        # Reload Y1 from rebuilt parquet path: reconstruct from FS-3 frozen for smoke.
        from l2_factor_reproduction.l2_ai_stock_selection.labels_ai_v1 import load_frozen_fs3_wide

        y1_wide = load_frozen_fs3_wide(1)
        smoke = run_residual_and_nonlinear_smoke(y1_wide)
    except Exception as exc:
        errors.append("residual_smoke: {}".format(exc))
        traceback.print_exc()
    try:
        ratios = run_ratio_smoke()
    except Exception as exc:
        errors.append("ratio_smoke: {}".format(exc))
        traceback.print_exc()
        ratios = pd.DataFrame()
    _write_json(RESULT_ROOT / "data_contract.json", data_contract_dict())
    _write_json(RESULT_ROOT / "model_contract.json", model_contract_dict())
    runtime = time.perf_counter() - t0
    payload = {
        "timing": timing,
        "score_audit": score_audit,
        "labels": labels,
        "smoke": smoke,
        "ratios": ratios,
        "runtime_sec": runtime,
        "errors": errors,
    }
    write_smoke_report(payload)
    print("TIMING_VERDICT={}".format(timing["verdict"]))
    print("RUNTIME_SEC={:.1f}".format(runtime))
    if errors:
        print("ERRORS:")
        for e in errors:
            print(" -", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
