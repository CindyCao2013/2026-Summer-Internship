"""RV exposure audit math (Sprint 4.3).

Pure-Python Fama–MacBeth regressions and progressive residual-IC chain for the
frozen realized_volatility @ 14:29 / Ret_30 / direction=-1 tuple.

No I/O, no DolphinDB, no freeze mutation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

ANNUALIZATION_DAYS = 250
DEFAULT_HAC_LAGS = 5
MIN_CS_OBS = 50
FROZEN_DIRECTION = -1
CONTROL_ORDER: tuple[str, ...] = (
    "size",
    "liquidity",
    "hist_vol",
    "momentum_20d",
    "session_mom",
)

VERDICT_CASE_A = "case_a_independent_alpha"
VERDICT_CASE_B = "case_b_style_proxy"
VERDICT_MIXED = "mixed_partial_alpha"


def rank_zscore(series: pd.Series) -> pd.Series:
    """Percentile rank then z-score; NaN-safe."""
    ranked = series.rank(method="average", pct=True)
    std = ranked.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (ranked - ranked.mean()) / std


def newey_west_tstat(
    series: pd.Series,
    *,
    lags: int = DEFAULT_HAC_LAGS,
) -> float:
    """t-stat of mean(series) under Newey–West HAC variance."""
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < max(10, lags + 2):
        return float("nan")
    mu = float(x.mean())
    demean = x - mu
    gamma0 = float(np.dot(demean, demean) / n)
    var = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(demean[lag:], demean[:-lag]) / n)
        var += 2.0 * weight * gamma
    if var <= 0 or not np.isfinite(var):
        return float("nan")
    se = np.sqrt(var / n)
    if se <= 0:
        return float("nan")
    return mu / se


def daily_cs_ols(
    frame: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    min_obs: int = MIN_CS_OBS,
    rank_z: bool = True,
) -> Optional[Dict[str, float]]:
    """One cross-section OLS. Returns dict of coefficients + n_obs."""
    cols = [y_col, *x_cols]
    sub = frame[cols].dropna()
    if len(sub) < max(min_obs, len(x_cols) + 5):
        return None
    if rank_z:
        sub = sub.apply(rank_zscore).dropna()
        if len(sub) < max(min_obs, len(x_cols) + 5):
            return None
    y = sub[y_col].to_numpy(dtype=float)
    x = sub[list(x_cols)].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    out = {"intercept": float(beta[0]), "n_obs": float(len(sub))}
    for name, coef in zip(x_cols, beta[1:]):
        out[name] = float(coef)
    return out


def fama_macbeth(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    *,
    date_col: str = "Date",
    min_obs: int = MIN_CS_OBS,
    rank_z: bool = True,
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> pd.DataFrame:
    """Daily CS OLS then time-series mean / NW t-stat for each coefficient."""
    daily_rows: List[dict] = []
    for date, group in panel.groupby(date_col, sort=True):
        coef = daily_cs_ols(
            group,
            y_col,
            x_cols,
            min_obs=min_obs,
            rank_z=rank_z,
        )
        if coef is None:
            continue
        row = {"Date": date, **coef}
        daily_rows.append(row)
    if not daily_rows:
        return pd.DataFrame(
            columns=[
                "model",
                "variable",
                "mean_coef",
                "tstat_nw",
                "n_days",
                "mean_n_obs",
            ]
        )

    daily = pd.DataFrame(daily_rows)
    summary_rows = []
    variables = ["intercept", *x_cols]
    for var in variables:
        series = daily[var]
        summary_rows.append(
            {
                "variable": var,
                "mean_coef": float(series.mean()),
                "tstat_nw": newey_west_tstat(series, lags=hac_lags),
                "n_days": int(series.notna().sum()),
                "mean_n_obs": float(daily["n_obs"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < MIN_CS_OBS:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


def daily_rank_ic(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str,
    *,
    date_col: str = "Date",
) -> pd.Series:
    rows = []
    dates = []
    for date, group in panel.groupby(date_col, sort=True):
        ic = spearman_ic(group[signal_col], group[return_col])
        dates.append(date)
        rows.append(ic)
    return pd.Series(rows, index=pd.to_datetime(dates), name="rank_ic")


def summarize_signed_ic(
    rank_ic: pd.Series,
    *,
    direction: int = FROZEN_DIRECTION,
) -> Dict[str, float]:
    clean = rank_ic.dropna()
    if clean.empty:
        return {
            "raw_ic": float("nan"),
            "signed_ic": float("nan"),
            "raw_icir": float("nan"),
            "signed_icir": float("nan"),
            "ic_win_rate": float("nan"),
            "n_dates": 0.0,
        }
    raw_ic = float(clean.mean())
    raw_std = float(clean.std(ddof=0))
    raw_icir = (
        raw_ic / raw_std * np.sqrt(ANNUALIZATION_DAYS)
        if raw_std > 0
        else float("nan")
    )
    signed = direction * clean
    return {
        "raw_ic": raw_ic,
        "signed_ic": float(signed.mean()),
        "raw_icir": raw_icir,
        "signed_icir": float(
            signed.mean() / signed.std(ddof=0) * np.sqrt(ANNUALIZATION_DAYS)
            if signed.std(ddof=0) > 0
            else float("nan")
        ),
        "ic_win_rate": float((signed > 0).mean()),
        "n_dates": float(len(clean)),
    }


def residualize_signal(
    panel: pd.DataFrame,
    signal_col: str,
    control_cols: Sequence[str],
    *,
    date_col: str = "Date",
    min_obs: int = MIN_CS_OBS,
) -> pd.Series:
    """Daily rank-z OLS residual of signal on controls; index matches panel."""
    resid = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, group in panel.groupby(date_col, sort=False):
        cols = [signal_col, *control_cols]
        ranked = group[cols].apply(rank_zscore).dropna()
        if len(ranked) < max(min_obs, len(control_cols) + 5):
            continue
        y = ranked[signal_col].to_numpy(dtype=float)
        x = ranked[list(control_cols)].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid.loc[ranked.index] = y - design @ beta
    return resid


def exposure_correlations(
    panel: pd.DataFrame,
    signal_col: str,
    control_cols: Sequence[str],
    *,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Mean daily Spearman of signal vs each control."""
    rows = []
    for control in control_cols:
        ics = []
        for _, group in panel.groupby(date_col, sort=True):
            ics.append(spearman_ic(group[signal_col], group[control]))
        series = pd.Series(ics)
        rows.append(
            {
                "control": control,
                "mean_spearman": float(series.mean()),
                "n_dates": int(series.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def progressive_residual_ic_chain(
    panel: pd.DataFrame,
    signal_col: str,
    return_col: str,
    controls: Sequence[str] = CONTROL_ORDER,
    *,
    date_col: str = "Date",
    direction: int = FROZEN_DIRECTION,
) -> pd.DataFrame:
    """Raw then progressive residual IC after adding controls in order."""
    rows = []
    raw_ic_series = daily_rank_ic(
        panel, signal_col, return_col, date_col=date_col
    )
    raw_summary = summarize_signed_ic(raw_ic_series, direction=direction)
    rows.append(
        {
            "step": "raw",
            "controls": "",
            **raw_summary,
            "retention": 1.0,
        }
    )
    active: List[str] = []
    for control in controls:
        if control not in panel.columns:
            continue
        active.append(control)
        work = panel.copy()
        work["_resid"] = residualize_signal(
            work, signal_col, active, date_col=date_col
        )
        ic_series = daily_rank_ic(
            work, "_resid", return_col, date_col=date_col
        )
        summary = summarize_signed_ic(ic_series, direction=direction)
        raw_abs = abs(raw_summary["raw_ic"])
        retention = (
            abs(summary["raw_ic"]) / raw_abs if raw_abs > 1e-12 else float("nan")
        )
        rows.append(
            {
                "step": f"after_{control}",
                "controls": ",".join(active),
                **summary,
                "retention": float(retention),
            }
        )
    return pd.DataFrame(rows)


def dominant_ic_drop_step(chain: pd.DataFrame) -> str:
    """Control step that removed the largest share of |raw IC|."""
    if chain.empty:
        return "insufficient"
    raw = chain[chain["step"] == "raw"]
    if raw.empty:
        return "insufficient"
    prev = abs(float(raw.iloc[0]["raw_ic"]))
    best_step = "none"
    best_drop = 0.0
    for _, row in chain.iterrows():
        if row["step"] == "raw":
            continue
        cur = abs(float(row["raw_ic"]))
        drop = prev - cur
        if np.isfinite(drop) and drop > best_drop:
            best_drop = drop
            best_step = str(row["step"]).replace("after_", "")
        prev = cur
    return best_step


def progressive_fama_macbeth(
    panel: pd.DataFrame,
    y_col: str,
    signal_col: str,
    controls: Sequence[str] = CONTROL_ORDER,
    *,
    date_col: str = "Date",
    hac_lags: int = DEFAULT_HAC_LAGS,
) -> pd.DataFrame:
    """Univariate then progressive FM models ending in the full control set."""
    model_specs: List[tuple[str, List[str]]] = [
        ("univariate", [signal_col]),
    ]
    active: List[str] = []
    for control in controls:
        if control not in panel.columns:
            continue
        active.append(control)
        model_specs.append((f"plus_{control}", [signal_col, *active]))
    if active:
        model_specs.append((f"full_{'+'.join(active)}", [signal_col, *active]))

    # Deduplicate while preserving order (full may equal last plus_*).
    seen = set()
    unique_specs = []
    for name, cols in model_specs:
        key = tuple(cols)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append((name, cols))

    frames = []
    for name, cols in unique_specs:
        summary = fama_macbeth(
            panel,
            y_col,
            cols,
            date_col=date_col,
            hac_lags=hac_lags,
        )
        if summary.empty:
            continue
        summary = summary.copy()
        summary.insert(0, "model", name)
        frames.append(summary)
    if not frames:
        return pd.DataFrame(
            columns=[
                "model",
                "variable",
                "mean_coef",
                "tstat_nw",
                "n_days",
                "mean_n_obs",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def classify_verdict(
    *,
    rv_tstat_full: float,
    rv_mean_coef_full: float,
    residual_retention: float,
    direction: int = FROZEN_DIRECTION,
) -> str:
    """Map FM + residual-IC evidence to Case A / B / mixed.

    Frozen direction equals sign(train raw IC). FM regresses raw excess return
    on RV, so E[coef_RV] should share that sign (negative when direction=-1).
    """
    sign_ok = np.isfinite(rv_mean_coef_full) and (
        np.sign(rv_mean_coef_full) == np.sign(direction)
    )
    strong_t = np.isfinite(rv_tstat_full) and abs(rv_tstat_full) >= 2.0
    if (
        sign_ok
        and strong_t
        and np.isfinite(residual_retention)
        and residual_retention >= 0.50
    ):
        return VERDICT_CASE_A
    if (not strong_t) or (
        np.isfinite(residual_retention) and residual_retention < 0.30
    ):
        return VERDICT_CASE_B
    return VERDICT_MIXED


def build_audit_summary(
    fm_table: pd.DataFrame,
    ic_chain: pd.DataFrame,
    corr_table: pd.DataFrame,
    *,
    signal_col: str = "rv",
    direction: int = FROZEN_DIRECTION,
) -> dict:
    """Compact machine-readable summary for results/summary.json."""
    full_rows = fm_table[fm_table["variable"] == signal_col]
    if full_rows.empty:
        rv_t = float("nan")
        rv_coef = float("nan")
        full_model = ""
    else:
        # Prefer the longest model (last matching row).
        last = full_rows.iloc[-1]
        rv_t = float(last["tstat_nw"])
        rv_coef = float(last["mean_coef"])
        full_model = str(last["model"])

    resid_rows = ic_chain[ic_chain["step"] != "raw"]
    retention = (
        float(resid_rows.iloc[-1]["retention"])
        if not resid_rows.empty
        else float("nan")
    )
    raw_row = ic_chain[ic_chain["step"] == "raw"]
    raw_signed_icir = (
        float(raw_row.iloc[0]["signed_icir"]) if not raw_row.empty else float("nan")
    )
    resid_signed_icir = (
        float(resid_rows.iloc[-1]["signed_icir"])
        if not resid_rows.empty
        else float("nan")
    )
    verdict = classify_verdict(
        rv_tstat_full=rv_t,
        rv_mean_coef_full=rv_coef,
        residual_retention=retention,
        direction=direction,
    )
    return {
        "factor": "realized_volatility",
        "bartime": "14:29",
        "horizon": "Ret_30",
        "direction": direction,
        "full_model": full_model,
        "rv_mean_coef_full": rv_coef,
        "rv_tstat_full": rv_t,
        "raw_signed_icir": raw_signed_icir,
        "residual_signed_icir": resid_signed_icir,
        "residual_ic_retention": retention,
        "dominant_ic_drop_step": dominant_ic_drop_step(ic_chain),
        "verdict": verdict,
        "exposure_correlations": corr_table.to_dict(orient="records"),
    }


def industry_demean_panel(
    panel: pd.DataFrame,
    cols: Iterable[str],
    *,
    date_col: str = "Date",
    industry_col: str = "industry",
    min_industry_names: int = 3,
) -> pd.DataFrame:
    """Within-date, within-industry demean for listed numeric columns."""
    out = panel.copy()
    for col in cols:
        demeaned = pd.Series(np.nan, index=panel.index, dtype=float)
        for _, day in panel.groupby(date_col, sort=False):
            if industry_col not in day.columns:
                continue
            for _, bucket in day.groupby(industry_col, sort=False):
                vals = bucket[col]
                if vals.notna().sum() < min_industry_names:
                    continue
                demeaned.loc[bucket.index] = vals - vals.mean()
        out[col] = demeaned
    return out
