#!/usr/bin/env python3
"""Build the missing buy-side diagnostics for the standalone TGD20 package.

The factor formula is frozen.  This script only evaluates the existing TGD20
panel using point-in-time signal_shift=1.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Factor_Dev_Lib
import factor_config as cfg
import intraday_lib
from alpha_investability import (
    classify_market_regimes,
    long_book_excess_performance,
    series_performance,
)
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from factor_runner import get_universe_mask
from industry_neutral import load_citics_industry_panel


PACKAGE = ROOT / "research_delivery/TGD20_research_package"
DATA_OUT = PACKAGE / "data/analysis"
FIG_OUT = PACKAGE / "figures"
SOURCE_REPORT = ROOT / "research/reports/tgd_v1"
PANEL_CACHE = (
    ROOT / "research/cache/tgd_panels/TGD20_20200101_20251231_w20.parquet"
)
START = pd.Timestamp("2022-01-28")
END = pd.Timestamp("2025-12-31")
ANNUAL_DAYS = 250


def ensure_dirs() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)


def compound(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod() - 1.0


def save_frame(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(DATA_OUT / name, index=index)


def rank_ic(signal: pd.DataFrame, forward_ret: pd.DataFrame) -> pd.Series:
    common_i = signal.index.intersection(forward_ret.index)
    common_c = signal.columns.intersection(forward_ret.columns)
    s = signal.loc[common_i, common_c]
    r = forward_ret.loc[common_i, common_c]
    return s.rank(axis=1, pct=True).corrwith(r.rank(axis=1, pct=True), axis=1)


def perf_row(ret: pd.Series) -> Dict[str, float]:
    p = series_performance(ret.dropna())
    return {
        "n_days": p["n_days"],
        "annual_return": p["annu_ret"],
        "sharpe": p["sharpe"],
        "max_drawdown": p["max_drawdown"],
        "calmar": p["calmar"],
        "positive_day_ratio": float((ret.dropna() > 0).mean()),
    }


def cs_spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    a, b = a.align(b, join="inner", axis=0)
    a, b = a.align(b, join="inner", axis=1)
    return a.rank(axis=1, pct=True).corrwith(b.rank(axis=1, pct=True), axis=1)


def cs_residualize(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional OLS residual of y on x (with intercept)."""
    y, x = y.align(x, join="inner", axis=0)
    y, x = y.align(x, join="inner", axis=1)
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for dt in y.index:
        frame = pd.DataFrame({"y": y.loc[dt], "x": x.loc[dt]}).dropna()
        if len(frame) < 50:
            continue
        xv = frame["x"].to_numpy()
        yv = frame["y"].to_numpy()
        design = np.column_stack([np.ones(len(frame)), xv])
        coef, _, _, _ = np.linalg.lstsq(design, yv, rcond=None)
        resid = yv - design @ coef
        out.loc[dt, frame.index] = resid
    return out


def summarize_ic_series(daily_ic: pd.Series, label: str) -> Dict[str, float]:
    s = daily_ic.dropna()
    mean = float(s.mean()) if len(s) else np.nan
    std = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return {
        "label": label,
        "n_days": int(len(s)),
        "mean_rank_ic": mean,
        "ic_std": std,
        "icir_annualized": mean / std * math.sqrt(ANNUAL_DAYS)
        if std and std > 0
        else np.nan,
        "positive_ic_ratio": float((s > 0).mean()) if len(s) else np.nan,
        "t_stat": mean / std * math.sqrt(len(s)) if std and std > 0 else np.nan,
    }


def build_rolling_ic(daily_ic: pd.Series) -> pd.DataFrame:
    s = daily_ic.dropna()
    out = pd.DataFrame(index=s.index)
    out["rank_ic"] = s
    for window in (60, 120, 250):
        roll = s.rolling(window, min_periods=max(40, window // 2))
        out[f"mean_{window}d"] = roll.mean()
        out[f"std_{window}d"] = roll.std(ddof=1)
        out[f"icir_{window}d"] = out[f"mean_{window}d"] / out[f"std_{window}d"] * math.sqrt(
            ANNUAL_DAYS
        )
        out[f"pos_ratio_{window}d"] = s.rolling(
            window, min_periods=max(40, window // 2)
        ).apply(lambda x: float((x > 0).mean()), raw=False)
    return out


def build_residual_attribution(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    amount: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Residualize size+industry TGD on close return / vol / liquidity proxies."""
    base = signal.loc[START:END]
    fwd = ret.loc[START:END]
    close_ret = ret.loc[START:END]
    vol20 = ret.rolling(20, min_periods=15).std().loc[START:END]
    liq = np.log(amount.where(amount > 0).rolling(20, min_periods=15).mean()).loc[
        START:END
    ]
    overnight = ret.shift(1).loc[START:END]

    controls = {
        "baseline_SI": None,
        "perp_same_day_return": close_ret,
        "perp_vol20": vol20,
        "perp_log_adv20": liq,
        "perp_prev_day_return": overnight,
    }
    rows = []
    residual_ics = {}
    base_ic = np.nan
    for label, control in controls.items():
        if control is None:
            residual = base
        else:
            residual = cs_residualize(base, control)
        daily = rank_ic(residual.shift(1), fwd).dropna()
        residual_ics[label] = daily
        row = summarize_ic_series(daily, label)
        if label == "baseline_SI":
            base_ic = row["mean_rank_ic"]
            row["retention_vs_baseline"] = 1.0
        else:
            row["retention_vs_baseline"] = (
                row["mean_rank_ic"] / base_ic
                if base_ic and not np.isnan(base_ic)
                else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    series = pd.DataFrame(residual_ics)
    return summary, series


def build_neutralization_waterfall() -> pd.DataFrame:
    path = (
        PACKAGE
        / "artifacts/research/reports/factors/TGD20/factor_summary.csv"
    )
    if not path.exists():
        path = SOURCE_REPORT / "replication/factor_summary.csv"
    summary = pd.read_csv(path)
    keep = summary[summary["mode"].isin(["raw", "size", "industry", "size_industry"])].copy()
    order = {"raw": 0, "size": 1, "industry": 2, "size_industry": 3}
    keep["step"] = keep["mode"].map(order)
    keep = keep.sort_values("step")
    return keep[
        [
            "mode",
            "rank_ic",
            "icir",
            "hl_sharpe",
            "hl_mdd",
            "daily_turnover",
            "net_sharpe",
            "monotonicity",
        ]
    ].reset_index(drop=True)


def industry_matched_long_return(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    industry: pd.DataFrame,
    top_frac: float = 0.10,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Top-decile long with each industry's weight matched to valid-universe EW."""
    sig = signal.shift(1)
    r = ret.reindex_like(sig)
    ind = industry.reindex_like(sig)
    long_ret = pd.Series(index=sig.index, dtype=float)
    universe_ret = pd.Series(index=sig.index, dtype=float)
    selected_n = pd.Series(index=sig.index, dtype=float)

    for dt in sig.index:
        frame = pd.DataFrame(
            {"signal": sig.loc[dt], "ret": r.loc[dt], "industry": ind.loc[dt]}
        ).dropna()
        if len(frame) < 100:
            continue
        frame["rank"] = frame.groupby("industry")["signal"].rank(
            pct=True, method="first"
        )
        selected = frame[frame["rank"] > 1.0 - top_frac].copy()
        if selected.empty:
            continue
        universe_industry_weight = (
            frame.groupby("industry").size().astype(float) / float(len(frame))
        )
        selected_count = selected.groupby("industry").size().astype(float)
        selected["weight"] = selected["industry"].map(
            universe_industry_weight / selected_count
        )
        weight_sum = selected["weight"].sum()
        if weight_sum <= 0:
            continue
        selected["weight"] /= weight_sum
        long_ret.loc[dt] = float((selected["weight"] * selected["ret"]).sum())
        universe_ret.loc[dt] = float(frame["ret"].mean())
        selected_n.loc[dt] = float(len(selected))
    return long_ret, universe_ret, selected_n


def build_yearly_tables(
    raw_excess: pd.Series,
    si_excess: pd.Series,
    industry_excess: pd.Series,
    hml: pd.Series,
) -> pd.DataFrame:
    rows = []
    series = {
        "raw_long_minus_universe": raw_excess,
        "size_industry_long_minus_universe": si_excess,
        "industry_matched_long_minus_universe": industry_excess,
        "raw_high_minus_low": hml,
    }
    for label, values in series.items():
        for year, part in values.dropna().groupby(values.dropna().index.year):
            row = {"portfolio": label, "year": int(year)}
            row.update(perf_row(part))
            rows.append(row)
    return pd.DataFrame(rows)


def build_ic_decay(signal: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = signal.loc[START:END]
    for horizon in (1, 2, 3, 5, 10, 20):
        fwd = (
            (1.0 + daily_ret)
            .rolling(horizon, min_periods=horizon)
            .apply(np.prod, raw=True)
            .shift(-horizon)
            - 1.0
        )
        daily = rank_ic(base, fwd.reindex_like(base)).dropna()
        mean = float(daily.mean())
        std = float(daily.std(ddof=1))
        rows.append(
            {
                "horizon_days": horizon,
                "mean_rank_ic": mean,
                "icir_annualized": mean / std * math.sqrt(ANNUAL_DAYS)
                if std > 0
                else np.nan,
                "t_stat": mean / std * math.sqrt(len(daily)) if std > 0 else np.nan,
                "n_days": len(daily),
                "retention_vs_1d": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and out.loc[0, "mean_rank_ic"] != 0:
        out["retention_vs_1d"] = out["mean_rank_ic"] / out.loc[0, "mean_rank_ic"]
    return out


def build_style_exposures(
    raw: pd.DataFrame,
    si: pd.DataFrame,
    ret: pd.DataFrame,
    amount: pd.DataFrame,
    float_mktcap: pd.DataFrame,
) -> pd.DataFrame:
    market = ret.mean(axis=1)
    mean_ret = ret.rolling(60, min_periods=40).mean()
    mean_market = market.rolling(60, min_periods=40).mean()
    cov = (
        ret.mul(market, axis=0).rolling(60, min_periods=40).mean()
        - mean_ret.mul(mean_market, axis=0)
    )
    beta = cov.div(market.rolling(60, min_periods=40).var(), axis=0)
    styles = {
        "Size_log_float_mktcap": np.log(float_mktcap.where(float_mktcap > 0)),
        "Momentum_20d": (1.0 + ret).rolling(20, min_periods=15).apply(
            np.prod, raw=True
        )
        - 1.0,
        "Volatility_20d": ret.rolling(20, min_periods=15).std(),
        "Liquidity_log_ADV20": np.log(
            amount.where(amount > 0).rolling(20, min_periods=15).mean()
        ),
        "Beta_60d": beta,
    }
    rows = []
    for signal_name, signal in {"raw": raw, "size_industry": si}.items():
        for style_name, style in styles.items():
            daily = cs_spearman(signal.loc[START:END], style.loc[START:END]).dropna()
            std = float(daily.std(ddof=1))
            rows.append(
                {
                    "signal": signal_name,
                    "style": style_name,
                    "mean_cs_spearman": float(daily.mean()),
                    "t_stat": float(daily.mean() / std * math.sqrt(len(daily)))
                    if std > 0
                    else np.nan,
                    "n_days": len(daily),
                }
            )
    return pd.DataFrame(rows)


def build_regime_table(
    long_ret: pd.Series, universe_ret: pd.Series, excess_ret: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    regimes = pd.DataFrame(index=excess_ret.index)
    regimes["direction"] = classify_market_regimes(
        universe_ret, window=60, bull_thresh=0.08, bear_thresh=-0.08
    )
    vol = universe_ret.rolling(20, min_periods=15).std() * math.sqrt(ANNUAL_DAYS)
    regimes["volatility"] = np.where(
        vol >= vol.loc[START:END].median(), "high_vol", "low_vol"
    )
    rows = []
    for dimension in ("direction", "volatility"):
        for regime, idx in regimes.groupby(dimension).groups.items():
            row = {"dimension": dimension, "regime": regime}
            row.update(perf_row(excess_ret.loc[idx]))
            row["long_annual_return"] = perf_row(long_ret.loc[idx])["annual_return"]
            row["universe_annual_return"] = perf_row(universe_ret.loc[idx])[
                "annual_return"
            ]
            rows.append(row)
    return pd.DataFrame(rows), regimes


def build_subuniverse_table(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    session,
) -> pd.DataFrame:
    rows = []
    for name in ("CSI300", "CSI500", "CSI1000"):
        mask = get_universe_mask(session, START, END, cfg.UNIVERSE_LIST[name])
        masked_signal = signal.mul(mask.reindex_like(signal))
        result = long_book_excess_performance(
            masked_signal.loc[START:END],
            ret.loc[START:END],
            top_frac=0.10,
            signal_shift=1,
            direction=1,
        )
        rows.append(
            {
                "universe": name,
                "excess_sharpe": result["excess_sharpe"],
                "excess_annual_return": result["excess_annu_ret"],
                "excess_max_drawdown": result["excess_max_drawdown"],
                "selected_count_mean": result["selected_count_mean"],
                "universe_count_mean": result["universe_count_mean"],
                "n_days": result["n_days"],
            }
        )
    return pd.DataFrame(rows)


def build_capacity(
    signal: pd.DataFrame, amount: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sig = signal.shift(1)
    ranks = sig.rank(axis=1, pct=True, method="first")
    selected = ranks > 0.90
    adv20_cny = amount.rolling(20, min_periods=15).mean().reindex_like(sig) * 1000.0
    selected_adv = adv20_cny.where(selected).sum(axis=1, min_count=1)
    weights = selected.div(selected.sum(axis=1), axis=0).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    rows = []
    for participation in (0.01, 0.03, 0.05, 0.10):
        daily_trade_capacity = selected_adv * participation
        aum_capacity = daily_trade_capacity / turnover.replace(0, np.nan)
        rows.append(
            {
                "adv_participation": participation,
                "median_daily_trade_capacity_cny": float(
                    daily_trade_capacity.median()
                ),
                "median_aum_capacity_cny": float(aum_capacity.median()),
                "p25_aum_capacity_cny": float(aum_capacity.quantile(0.25)),
                "median_selected_names": float(selected.sum(axis=1).median()),
                "median_one_way_turnover": float((turnover / 2.0).median()),
            }
        )
    daily = pd.DataFrame(
        {
            "selected_adv20_cny": selected_adv,
            "two_way_turnover": turnover,
            "selected_names": selected.sum(axis=1),
        }
    )
    return pd.DataFrame(rows), daily


def beta_stripped_metrics(
    long_ret: pd.Series, market_ret: pd.Series
) -> pd.DataFrame:
    df = pd.concat(
        [long_ret.rename("long"), market_ret.rename("market")], axis=1
    ).dropna()
    x = np.column_stack([np.ones(len(df)), df["market"].to_numpy()])
    alpha, beta = np.linalg.lstsq(x, df["long"].to_numpy(), rcond=None)[0]
    residual = pd.Series(
        df["long"].to_numpy() - beta * df["market"].to_numpy(),
        index=df.index,
        name="beta_stripped_return",
    )
    residual.to_csv(DATA_OUT / "beta_stripped_daily.csv")
    p = perf_row(residual)
    p.update(
        {
            "daily_alpha": float(alpha),
            "annualized_alpha_linear": float(alpha * ANNUAL_DAYS),
            "market_beta": float(beta),
        }
    )
    return pd.DataFrame([p])


def draw_plots(
    ic_decay: pd.DataFrame,
    style: pd.DataFrame,
    yearly: pd.DataFrame,
    regime: pd.DataFrame,
    subuniverse: pd.DataFrame,
    capacity: pd.DataFrame,
    daily: pd.DataFrame,
    ic_daily: pd.Series,
    execution: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    rolling_ic: pd.DataFrame,
    residual_attr: pd.DataFrame,
    neut_waterfall: pd.DataFrame,
) -> None:
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ic_decay["horizon_days"], ic_decay["mean_rank_ic"], marker="o")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Forward horizon (trading days)", ylabel="Mean RankIC")
    ax.set_title("TGD20 IC decay")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "ic_decay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(rolling_ic.index, rolling_ic["rank_ic"], color="0.75", lw=0.7, label="Daily RankIC")
    ax.plot(
        rolling_ic.index,
        rolling_ic["mean_60d"],
        color="steelblue",
        lw=1.6,
        label="60d mean",
    )
    ax.plot(
        rolling_ic.index,
        rolling_ic["mean_250d"],
        color="darkorange",
        lw=1.8,
        label="250d mean",
    )
    if rolling_ic["mean_60d"].notna().any():
        ax.fill_between(
            rolling_ic.index,
            rolling_ic["mean_60d"] - rolling_ic["std_60d"],
            rolling_ic["mean_60d"] + rolling_ic["std_60d"],
            color="steelblue",
            alpha=0.15,
            label="60d ±1 std",
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="RankIC", title="Rolling RankIC (size+industry TGD20)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "rolling_ic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = neut_waterfall["mode"].tolist()
    ax.plot(order, neut_waterfall["rank_ic"] * 100, "o-", color="steelblue", label="RankIC (%)")
    ax2 = ax.twinx()
    ax2.plot(order, neut_waterfall["icir"], "s--", color="darkorange", label="ICIR")
    ax.set_ylabel("RankIC (%)")
    ax2.set_ylabel("ICIR")
    ax.set_title("Neutralization waterfall: RankIC / ICIR")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "neutralization_waterfall.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = residual_attr["label"].tolist()
    ax.bar(labels, residual_attr["mean_rank_ic"] * 100, color="teal")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean RankIC (%)")
    ax.set_title("Residual attribution: RankIC after stripping known effects")
    ax.tick_params(axis="x", rotation=20)
    for i, row in residual_attr.iterrows():
        ax.text(
            i,
            row["mean_rank_ic"] * 100,
            f"{row['retention_vs_baseline']:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIG_OUT / "residual_attribution.png", dpi=180)
    plt.close(fig)

    pivot = style.pivot(index="style", columns="signal", values="mean_cs_spearman")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Barra-style exposure proxy (mean CS Spearman)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "style_exposure_matrix.png", dpi=180)
    plt.close(fig)

    yp = yearly.pivot(index="year", columns="portfolio", values="sharpe")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    yp.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Yearly portfolio Sharpe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "yearly_portfolio_sharpe.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    reg = regime.pivot(index="regime", columns="dimension", values="sharpe")
    reg.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Excess Sharpe")
    ax.set_title("Market-regime performance")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "regime_performance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(subuniverse["universe"], subuniverse["excess_sharpe"], color="steelblue")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Exact universe-EW excess Sharpe")
    ax.set_title("Performance by stock universe")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "subuniverse_performance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        capacity["adv_participation"] * 100,
        capacity["median_aum_capacity_cny"] / 1e8,
        marker="o",
    )
    ax.set(xlabel="ADV participation (%)", ylabel="Estimated AUM capacity (CNY 100m)")
    ax.set_title("Capacity sensitivity")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "capacity_by_adv.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    daily["signal_dispersion"].plot(ax=axes[0], color="slateblue")
    axes[0].set_ylabel("CS std")
    axes[0].set_title("Crowding monitor")
    daily["g10_turnover"].rolling(20).mean().plot(ax=axes[1], color="darkorange")
    axes[1].set_ylabel("20d TO")
    daily["rolling_excess_sharpe_60d"].plot(ax=axes[2], color="darkgreen")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_ylabel("60d Sharpe")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "crowding_monitor.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ic_daily.rolling(60).mean().plot(ax=axes[0], color="navy")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("60d mean RankIC")
    compound(daily["size_industry_excess_return"]).plot(
        ax=axes[1], color="darkgreen"
    )
    axes[1].set_ylabel("Cumulative excess")
    axes[0].set_title("Out-of-sample tracking dashboard")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "oos_tracking.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label in (
        ("raw_excess_return", "Raw signal long - universe EW"),
        ("size_industry_excess_return", "Size+industry signal long - universe EW"),
        (
            "industry_matched_excess_return",
            "Industry-matched long - universe EW",
        ),
    ):
        compound(daily[column]).plot(ax=ax, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative excess return")
    ax.set_title("Long-only implementation comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "long_only_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        cost_sensitivity["round_trip_cost_bp"],
        cost_sensitivity["net_excess_sharpe"],
        marker="o",
        color="firebrick",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Transaction cost (bp)", ylabel="Net excess Sharpe")
    ax.set_title("Long-only transaction-cost sensitivity")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "cost_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rebalance = execution[execution["stage"] == "E1"].copy()
    axes[0].plot(
        rebalance["daily_turnover"],
        rebalance["net_sharpe"],
        "o-",
        color="teal",
    )
    for _, row in rebalance.iterrows():
        axes[0].annotate(
            str(row["label"]).split("|")[-1],
            (row["daily_turnover"], row["net_sharpe"]),
            fontsize=7,
        )
    axes[0].set(xlabel="Daily turnover", ylabel="Net Sharpe @15bp")
    axes[0].set_title("Rebalance frontier")
    buffer = execution[execution["stage"] == "E2_buffer"].copy()
    axes[1].scatter(buffer["daily_turnover"], buffer["net_sharpe"], color="purple")
    for _, row in buffer.iterrows():
        axes[1].annotate(
            str(row["label"]).split("|")[-1],
            (row["daily_turnover"], row["net_sharpe"]),
            fontsize=7,
        )
    axes[1].set(xlabel="Daily turnover", ylabel="Net Sharpe @15bp")
    axes[1].set_title("Buffer frontier")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "execution_frontier.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    print("Loading frozen TGD20 panel", flush=True)
    raw = pd.read_parquet(PANEL_CACHE).sort_index()
    raw.index = pd.to_datetime(raw.index)

    print("Loading EOD, market-cap and industry panels", flush=True)
    eod, session = load_eod_enriched_tables(pd.Timestamp("2019-01-01"), END)
    try:
        session.run(intraday_lib.ddb_functions)
        close = eod.close.reindex(index=raw.index, columns=raw.columns)
        ret = Factor_Dev_Lib.get_Ret_Matrix(
            raw.index.min().to_pydatetime(),
            raw.index.max().to_pydatetime(),
            method="c2c",
        ).reindex(index=raw.index, columns=raw.columns)
        amount = eod.amount.reindex_like(raw)
        float_mktcap = eod.float_mktcap.reindex_like(raw)
        industry = load_citics_industry_panel(raw.index.min(), raw.index.max()).reindex_like(
            raw
        )
        si = cs_zscore(neutralize_size_industry(raw, industry, float_mktcap))

        raw_result = long_book_excess_performance(
            cs_zscore(raw.loc[START:END]),
            ret.loc[START:END],
            top_frac=0.10,
            signal_shift=1,
            direction=1,
        )
        si_result = long_book_excess_performance(
            si.loc[START:END],
            ret.loc[START:END],
            top_frac=0.10,
            signal_shift=1,
            direction=1,
        )
        ind_long, ind_universe, selected_n = industry_matched_long_return(
            si.loc[START:END],
            ret.loc[START:END],
            industry.loc[START:END],
        )
        ind_excess = ind_long - ind_universe

        group_cum = pd.read_csv(
            SOURCE_REPORT / "portfolio/group_cum_pnl.csv",
            index_col=0,
            parse_dates=True,
        )
        hml = group_cum["H-L"].diff()
        if len(hml):
            hml.iloc[0] = group_cum["H-L"].iloc[0]

        yearly = build_yearly_tables(
            raw_result["_excess_ret"],
            si_result["_excess_ret"],
            ind_excess,
            hml,
        )
        ic_decay = build_ic_decay(si, ret)
        style = build_style_exposures(raw, si, ret, amount, float_mktcap)
        regime, regime_labels = build_regime_table(
            si_result["_long_ret"],
            si_result["_universe_ew_ret"],
            si_result["_excess_ret"],
        )
        subuniverse = build_subuniverse_table(si, ret, session)
        capacity, capacity_daily = build_capacity(
            si.loc[START:END], amount.loc[START:END]
        )
        beta_metrics = beta_stripped_metrics(
            si_result["_long_ret"], si_result["_universe_ew_ret"]
        )

        sig_shift = si.loc[START:END].shift(1)
        ranks = sig_shift.rank(axis=1, pct=True, method="first")
        selected = ranks > 0.90
        weights = selected.div(selected.sum(axis=1), axis=0).fillna(0.0)
        g10_turnover = weights.diff().abs().sum(axis=1)
        daily = pd.DataFrame(
            {
                "raw_long_return": raw_result["_long_ret"],
                "raw_universe_return": raw_result["_universe_ew_ret"],
                "raw_excess_return": raw_result["_excess_ret"],
                "size_industry_long_return": si_result["_long_ret"],
                "size_industry_universe_return": si_result["_universe_ew_ret"],
                "size_industry_excess_return": si_result["_excess_ret"],
                "industry_matched_long_return": ind_long,
                "industry_matched_universe_return": ind_universe,
                "industry_matched_excess_return": ind_excess,
                "industry_matched_selected_names": selected_n,
                "signal_dispersion": sig_shift.std(axis=1),
                "signal_p90_minus_p10": sig_shift.quantile(0.90, axis=1)
                - sig_shift.quantile(0.10, axis=1),
                "g10_turnover": g10_turnover,
            }
        )
        rolling_mean = daily["size_industry_excess_return"].rolling(60).mean()
        rolling_std = daily["size_industry_excess_return"].rolling(60).std()
        daily["rolling_excess_sharpe_60d"] = (
            rolling_mean / rolling_std * math.sqrt(ANNUAL_DAYS)
        )
        cost_rows = []
        for cost_bp in (0, 5, 10, 15, 20, 30, 50):
            net_excess = (
                daily["size_industry_excess_return"]
                - daily["g10_turnover"] * cost_bp / 10000.0
            )
            row = {
                "round_trip_cost_bp": cost_bp,
                "mean_daily_turnover": float(daily["g10_turnover"].mean()),
            }
            row.update(
                {
                    f"net_excess_{key}": value
                    for key, value in perf_row(net_excess).items()
                }
            )
            cost_rows.append(row)
        cost_sensitivity = pd.DataFrame(cost_rows)

        ic_daily = rank_ic(
            si.loc[START:END].shift(1), ret.loc[START:END]
        ).rename("rank_ic")
        ic_summary = pd.DataFrame(
            [
                summarize_ic_series(ic_daily, "size_industry"),
                summarize_ic_series(
                    rank_ic(cs_zscore(raw.loc[START:END]).shift(1), ret.loc[START:END]),
                    "raw",
                ),
            ]
        )
        rolling_ic = build_rolling_ic(ic_daily)
        residual_attr, residual_ic_series = build_residual_attribution(si, ret, amount)
        neut_waterfall = build_neutralization_waterfall()
        execution = pd.read_csv(SOURCE_REPORT / "execution/all_experiments.csv")

        save_frame(yearly, "yearly_portfolio_performance.csv", index=False)
        save_frame(ic_decay, "ic_decay.csv", index=False)
        save_frame(style, "style_exposure_matrix.csv", index=False)
        save_frame(regime, "regime_performance.csv", index=False)
        save_frame(regime_labels, "regime_labels.csv")
        save_frame(subuniverse, "subuniverse_performance.csv", index=False)
        save_frame(capacity, "capacity_by_adv.csv", index=False)
        save_frame(capacity_daily, "capacity_daily.csv")
        save_frame(beta_metrics, "beta_stripped_metrics.csv", index=False)
        save_frame(daily, "daily_buyside_series.csv")
        save_frame(cost_sensitivity, "long_only_cost_sensitivity.csv", index=False)
        save_frame(ic_summary, "ic_summary.csv", index=False)
        save_frame(rolling_ic, "rolling_ic.csv")
        save_frame(residual_attr, "residual_attribution.csv", index=False)
        save_frame(residual_ic_series, "residual_attribution_daily_ic.csv")
        save_frame(neut_waterfall, "neutralization_waterfall.csv", index=False)
        ic_daily.to_csv(DATA_OUT / "rank_ic_size_industry.csv")

        summary = {
            "sample": {"start": str(START.date()), "end": str(END.date())},
            "headline": {
                "raw_exact_universe_excess_sharpe": raw_result["excess_sharpe"],
                "size_industry_exact_universe_excess_sharpe": si_result[
                    "excess_sharpe"
                ],
                "industry_matched_exact_universe_excess_sharpe": perf_row(
                    ind_excess
                )["sharpe"],
                "beta_stripped_sharpe": beta_metrics.iloc[0]["sharpe"],
                "market_beta": beta_metrics.iloc[0]["market_beta"],
                "size_industry_rank_ic": float(
                    ic_summary.loc[
                        ic_summary["label"] == "size_industry", "mean_rank_ic"
                    ].iloc[0]
                ),
                "size_industry_icir": float(
                    ic_summary.loc[
                        ic_summary["label"] == "size_industry", "icir_annualized"
                    ].iloc[0]
                ),
                "size_industry_positive_ic_ratio": float(
                    ic_summary.loc[
                        ic_summary["label"] == "size_industry", "positive_ic_ratio"
                    ].iloc[0]
                ),
            },
            "method_notes": {
                "signal_shift": 1,
                "annual_days": ANNUAL_DAYS,
                "industry_matched": (
                    "top decile selected within each CITICS industry; industry total "
                    "weights match the valid-universe equal-weight benchmark"
                ),
                "capacity": (
                    "ADV20 participation divided by realized equal-weight long-book "
                    "turnover; indicative, not an order-book simulation"
                ),
                "residual_attribution": (
                    "Daily CS OLS residual of size+industry TGD20 on each control, "
                    "then RankIC vs next-day C2C return"
                ),
            },
        }
        (DATA_OUT / "buyside_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        draw_plots(
            ic_decay,
            style,
            yearly,
            regime,
            subuniverse,
            capacity,
            daily,
            ic_daily,
            execution,
            cost_sensitivity,
            rolling_ic,
            residual_attr,
            neut_waterfall,
        )
    finally:
        try:
            session.close()
        except Exception:
            pass
    print(f"Wrote diagnostics to {PACKAGE}", flush=True)


if __name__ == "__main__":
    main()
