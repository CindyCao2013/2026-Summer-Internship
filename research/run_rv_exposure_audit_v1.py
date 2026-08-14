#!/usr/bin/env python3
"""Sprint 4.3 — realized_volatility exposure audit runner.

Builds an asset-level 14:29 panel, runs Fama–MacBeth return regressions and a
progressive residual-IC chain against market-structure controls, and writes
Case A / B / mixed evidence under research/results/rv_exposure_audit_v1/.

Does not mutate the freeze file or re-search bartime / horizon / direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.rv_exposure_audit import (  # noqa: E402
    CONTROL_ORDER,
    FROZEN_DIRECTION,
    build_audit_summary,
    exposure_correlations,
    industry_demean_panel,
    progressive_fama_macbeth,
    progressive_residual_ic_chain,
)
from factors.intraday.discovery_v1 import ddb_version as discovery_ddb  # noqa: E402
from industry_neutral import load_citics_industry_panel  # noqa: E402
from research.freeze_intraday_alpha_v1 import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_FREEZE,
    verify_spec,
)
from research.run_intraday_alpha_library_v1 import (  # noqa: E402
    RET_MATRIX,
    _as_tradetime,
    _connect,
)
from research.run_intraday_alpha_oos_v1 import _filter_slots  # noqa: E402

DEFAULT_OUTPUT = ROOT / "research/results/rv_exposure_audit_v1"
FACTOR_NAME = "realized_volatility"
BARTIME = "14:29"
HORIZON = "Ret_30"
DIRECTION = FROZEN_DIRECTION
SPEC_VERSION = "rv_exposure_audit_v1"

PERIODS = {
    "train_2024H1": {"start": "2024-01-01", "end": "2024-06-30"},
    "validation_2024H2": {"start": "2024-07-01", "end": "2024-12-31"},
    "test_2025_available": {"start": "2025-01-01", "end": "2025-08-18"},
}


def _spec_hash(freeze_sha256: str) -> str:
    payload = {
        "version": SPEC_VERSION,
        "freeze_sha256": freeze_sha256,
        "factor": FACTOR_NAME,
        "bartime": BARTIME,
        "horizon": HORIZON,
        "direction": DIRECTION,
        "controls": list(CONTROL_ORDER),
        "y": "market_excess_Ret_30",
        "rank_z_ols": True,
        "hac_lags": 5,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _fetch_signal_return_panel(
    session,
    signal: pd.DataFrame,
) -> pd.DataFrame:
    """Join 14:29 RV to Ret_30 and exact market-excess return."""
    if not re.fullmatch(r"Ret_[A-Za-z0-9]+", HORIZON):
        raise ValueError(f"Unsafe horizon: {HORIZON}")
    upload_name = "rv_audit_signal"
    session.upload({upload_name: signal})
    result = session.run(
        f"""
auditSignal = {upload_name}
auditBase = select symbol as Symbol, date(tradetime) as Date,
    second(tradetime) as Bartime, tradetime, value as rv
    from auditSignal
auditJoined = ej(auditBase, {RET_MATRIX}, `Symbol`Date`Bartime)
auditMkt = select Date, Bartime, avg({HORIZON}) as market_return,
    count({HORIZON}) as n_market_assets
    from auditJoined
    where isValid({HORIZON})
    group by Date, Bartime
auditPanel = select Symbol, Date, Bartime, tradetime, rv,
    {HORIZON} as ret_raw, market_return, n_market_assets,
    {HORIZON} - market_return as ret_excess
    from ej(auditJoined, auditMkt, `Date`Bartime)
    where isValid(rv) and isValid({HORIZON})
auditPanel
"""
    )
    out = pd.DataFrame(result)
    if out.empty:
        raise ValueError("RV audit panel is empty after return join")
    out["Date"] = pd.to_datetime(out["Date"])
    out["tradetime"] = pd.to_datetime(out["tradetime"])
    for col in ("rv", "ret_raw", "ret_excess", "market_return"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.rename(columns={"Symbol": "symbol"})
    out["symbol"] = out["symbol"].astype(str)
    return out.dropna(subset=["rv", "ret_excess"])


def _build_style_controls(
    start: str,
    end: str,
    symbols: list[str],
) -> Dict[str, pd.DataFrame]:
    """Lagged daily style panels + CITICS industry (wide)."""
    import datetime as dt

    from factor_data_loaders import load_eod_enriched_tables

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    preheat = start_ts - dt.timedelta(days=60)
    enriched, _ = load_eod_enriched_tables(preheat.to_pydatetime(), end_ts.to_pydatetime())

    close = enriched.close.reindex(columns=symbols)
    amount = enriched.amount.reindex(columns=symbols)
    float_mkt = enriched.float_mktcap.reindex(columns=symbols)

    # Point-in-time: use t-1 EOD values on date t (known before 14:29).
    size = np.log(float_mkt.replace(0, np.nan)).shift(1)
    liquidity = amount.rolling(20, min_periods=10).mean().shift(1)
    daily_ret = close.pct_change()
    hist_vol = daily_ret.rolling(20, min_periods=10).std().shift(1)
    momentum_20d = close.pct_change(20).shift(1)

    industry = load_citics_industry_panel(preheat, end_ts)
    industry = industry.reindex(columns=symbols)

    return {
        "size": size.loc[start_ts:end_ts],
        "liquidity": liquidity.loc[start_ts:end_ts],
        "hist_vol": hist_vol.loc[start_ts:end_ts],
        "momentum_20d": momentum_20d.loc[start_ts:end_ts],
        "industry": industry.loc[start_ts:end_ts],
    }


def _stamp_wide_on_panel(
    panel: pd.DataFrame,
    wide: pd.DataFrame,
    col_name: str,
) -> pd.Series:
    """Map wide Date×symbol panel onto long audit rows."""
    long = (
        wide.stack(dropna=False)
        .rename(col_name)
        .rename_axis(["Date", "symbol"])
        .reset_index()
    )
    long["Date"] = pd.to_datetime(long["Date"])
    long["symbol"] = long["symbol"].astype(str)
    merged = panel[["Date", "symbol"]].merge(
        long, on=["Date", "symbol"], how="left"
    )
    return merged[col_name]


def _fetch_session_momentum(session, panel: pd.DataFrame) -> pd.Series:
    """Open → 14:29 simple return from the minute bar table."""
    start = pd.Timestamp(panel["Date"].min()).strftime("%Y.%m.%d")
    end = pd.Timestamp(panel["Date"].max()).strftime("%Y.%m.%d")
    symbols = sorted(panel["symbol"].astype(str).unique().tolist())
    # Chunk symbol filters to keep DDB script size bounded.
    parts = []
    chunk_size = 400
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        sym_list = "`" + "`".join(chunk)
        result = session.run(
            f"""
t = loadTable("dfs://QV_Trade_to_MinuteBar", "Stock_one_minute")
bars0 = select Symbol, Date, second(Bartime) as Bartime, Close as close_adj
    from t
    where Date between {start} : {end}
      and Symbol in {sym_list}
      and ((second(Bartime) >= 09:30:00 and second(Bartime) <= 11:30:00)
        or (second(Bartime) >= 13:00:00 and second(Bartime) <= 15:00:00))
openPx = select Symbol, Date, first(close_adj) as open_px
    from bars0
    context by Symbol, Date csort Bartime
px1429 = select Symbol, Date, close_adj as px_1429
    from bars0
    where Bartime == 14:29:00
select Symbol, Date, px_1429 \\ open_px - 1.0 as session_mom
    from ej(openPx, px1429, `Symbol`Date)
    where isValid(open_px) and isValid(px_1429) and open_px > 0
"""
        )
        parts.append(pd.DataFrame(result))
    if not parts:
        return pd.Series(np.nan, index=panel.index, name="session_mom")
    mom = pd.concat(parts, ignore_index=True)
    if mom.empty:
        return pd.Series(np.nan, index=panel.index, name="session_mom")
    mom["Date"] = pd.to_datetime(mom["Date"])
    mom = mom.rename(columns={"Symbol": "symbol"})
    mom["symbol"] = mom["symbol"].astype(str)
    merged = panel[["Date", "symbol"]].merge(
        mom[["Date", "symbol", "session_mom"]],
        on=["Date", "symbol"],
        how="left",
    )
    return pd.to_numeric(merged["session_mom"], errors="coerce")


def build_audit_panel(
    session,
    start: str,
    end: str,
    *,
    include_session_mom: bool = True,
) -> pd.DataFrame:
    narrow = discovery_ddb(FACTOR_NAME, start, end)
    signal = _filter_slots(_as_tradetime(narrow, FACTOR_NAME), {BARTIME})
    if signal.empty:
        raise ValueError(f"No {FACTOR_NAME} rows at {BARTIME}")
    panel = _fetch_signal_return_panel(session, signal)
    symbols = sorted(panel["symbol"].unique().tolist())
    controls = _build_style_controls(start, end, symbols)
    for name in ("size", "liquidity", "hist_vol", "momentum_20d"):
        panel[name] = _stamp_wide_on_panel(panel, controls[name], name)
    panel["industry"] = _stamp_wide_on_panel(
        panel, controls["industry"], "industry"
    ).astype("object")
    if include_session_mom:
        panel["session_mom"] = _fetch_session_momentum(session, panel)
    else:
        panel["session_mom"] = np.nan
    return panel


def _available_controls(panel: pd.DataFrame, min_frac: float = 0.50) -> list[str]:
    """Keep controls that are populated for most of the audit panel."""
    out = []
    for col in CONTROL_ORDER:
        if col not in panel.columns:
            continue
        frac = float(panel[col].notna().mean())
        if frac >= min_frac:
            out.append(col)
    return out


def run_audit_on_panel(panel: pd.DataFrame) -> dict:
    available = _available_controls(panel)
    corr = exposure_correlations(panel, "rv", available)
    fm = progressive_fama_macbeth(
        panel, "ret_excess", "rv", controls=available
    )
    ic_chain = progressive_residual_ic_chain(
        panel,
        "rv",
        "ret_excess",
        controls=available,
        direction=DIRECTION,
    )

    # Industry-adjusted full model (demean y + x within industry).
    demean_cols = ["rv", "ret_excess", *available]
    demeaned = industry_demean_panel(panel, demean_cols)
    fm_ind = progressive_fama_macbeth(
        demeaned, "ret_excess", "rv", controls=available
    )
    fm_ind = fm_ind.copy()
    fm_ind["model"] = "indadj_" + fm_ind["model"].astype(str)

    summary = build_audit_summary(fm, ic_chain, corr, direction=DIRECTION)
    summary["industry_adjusted_rv_tstat"] = float("nan")
    ind_rv = fm_ind[
        (fm_ind["variable"] == "rv")
        & fm_ind["model"].str.startswith("indadj_full")
    ]
    if ind_rv.empty:
        ind_rv = fm_ind[fm_ind["variable"] == "rv"]
    if not ind_rv.empty:
        summary["industry_adjusted_rv_tstat"] = float(
            ind_rv.iloc[-1]["tstat_nw"]
        )
        summary["industry_adjusted_rv_mean_coef"] = float(
            ind_rv.iloc[-1]["mean_coef"]
        )

    return {
        "exposure_correlations": corr,
        "fama_macbeth": pd.concat([fm, fm_ind], ignore_index=True),
        "residual_ic_chain": ic_chain,
        "summary": summary,
    }


def _write_outputs(
    out_dir: Path,
    period: str,
    artifacts: dict,
    *,
    freeze_sha256: str,
    audit_sha256: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    period_dir = out_dir / period
    period_dir.mkdir(parents=True, exist_ok=True)
    artifacts["exposure_correlations"].to_csv(
        period_dir / "exposure_correlations.csv", index=False
    )
    artifacts["fama_macbeth"].to_csv(
        period_dir / "fama_macbeth.csv", index=False
    )
    artifacts["residual_ic_chain"].to_csv(
        period_dir / "residual_ic_chain.csv", index=False
    )
    payload = {
        **artifacts["summary"],
        "period": period,
        "freeze_sha256": freeze_sha256,
        "audit_sha256": audit_sha256,
    }
    (period_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--period",
        default="train_2024H1",
        choices=list(PERIODS),
        help="Frozen sample window (default train_2024H1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=DEFAULT_FREEZE,
        help="Path to locked freeze JSON",
    )
    parser.add_argument(
        "--skip-session-mom",
        action="store_true",
        help="Skip open→14:29 session momentum (faster / offline-friendly)",
    )
    args = parser.parse_args(argv)

    freeze = verify_spec(args.freeze)
    freeze_sha = freeze["spec_sha256"]
    factor_spec = freeze["factors"][FACTOR_NAME]
    if (
        factor_spec["bartime"] != BARTIME
        or factor_spec["horizon"] != HORIZON
        or int(factor_spec["direction"]) != DIRECTION
    ):
        raise SystemExit(
            "Freeze tuple drift: expected "
            f"{FACTOR_NAME} {BARTIME}/{HORIZON}/{DIRECTION}"
        )

    audit_sha = _spec_hash(freeze_sha)
    window = PERIODS[args.period]
    print(
        f"[rv_exposure_audit] period={args.period} "
        f"{window['start']}→{window['end']}",
        flush=True,
    )
    session = _connect()
    panel = build_audit_panel(
        session,
        window["start"],
        window["end"],
        include_session_mom=not args.skip_session_mom,
    )
    print(
        f"[rv_exposure_audit] panel rows={len(panel)} "
        f"dates={panel['Date'].nunique()} names={panel['symbol'].nunique()}",
        flush=True,
    )
    artifacts = run_audit_on_panel(panel)
    _write_outputs(
        args.output,
        args.period,
        artifacts,
        freeze_sha256=freeze_sha,
        audit_sha256=audit_sha,
    )
    summary = artifacts["summary"]
    print(
        f"[rv_exposure_audit] verdict={summary['verdict']} "
        f"rv_t={summary['rv_tstat_full']:.2f} "
        f"retention={summary['residual_ic_retention']:.3f} "
        f"drop={summary['dominant_ic_drop_step']}",
        flush=True,
    )
    print(f"[rv_exposure_audit] wrote {args.output / args.period}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
