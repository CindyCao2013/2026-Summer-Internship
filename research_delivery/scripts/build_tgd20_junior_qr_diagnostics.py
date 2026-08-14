#!/usr/bin/env python3
"""Build Junior-QR diagnostics for the frozen TGD20 factor.

Data policy:
- TGD20 and ``tgd_eps`` come from the canonical panel built from DolphinDB
  minute bars.
- Returns, OHLC, amount, market cap, industry and index membership are loaded
  from the project's point-in-time DolphinDB/Wind data layer.
- APM uses the existing DolphinDB-derived CSI1000 scout cache.

This script evaluates robustness only.  It does not change or retune TGD20.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
import intraday_lib
from alpha_investability import long_book_excess_performance, series_performance
from execution_layer import (
    buffer_ls_masks,
    downsample_signal,
    net_pnl_series,
    plain_ls_masks,
    pnl_and_turnover_from_weights,
    weights_from_masks,
)
from factor_attribution import cs_zscore
from factor_data_loaders import load_eod_enriched_tables
from factor_formulas_sue import neutralize_size_industry
from industry_neutral import load_citics_industry_panel, panel_industry_demean
from liquidity_normalization import panel_cross_sectional_residual
from data_preheat import CITICS_IND_CODE_DICT


PACKAGE = ROOT / "research_delivery/TGD20_research_package"
DATA_OUT = PACKAGE / "data/analysis"
FIG_OUT = PACKAGE / "figures"
PANEL_CACHE = ROOT / "research/cache/tgd_panels/TGD20_20200101_20251231_w20.parquet"
LONG_CACHE = (
    ROOT / "research/cache/tgd_panels/TGD20_long_20200101_20251231_w20.parquet"
)
APM_CACHE = (
    ROOT
    / "research/cache/apm_session/signal/"
    "apm_cs_wide_CSI1000scout_20210101_20251231.parquet"
)
START = pd.Timestamp("2022-01-28")
END = pd.Timestamp("2025-12-31")
ANNUAL_DAYS = 250


def save_frame(frame: pd.DataFrame, name: str, *, index: bool = True) -> None:
    frame.to_csv(DATA_OUT / name, index=index)


def compound(ret: pd.Series) -> pd.Series:
    return (1.0 + ret.fillna(0.0)).cumprod() - 1.0


def rank_ic(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    signal, ret = signal.align(ret, join="inner", axis=0)
    signal, ret = signal.align(ret, join="inner", axis=1)
    return signal.rank(axis=1, pct=True).corrwith(
        ret.rank(axis=1, pct=True), axis=1
    )


def ic_summary(signal: pd.DataFrame, ret: pd.DataFrame, label: str) -> Dict[str, float]:
    daily = rank_ic(signal.shift(1), ret).dropna()
    std = float(daily.std(ddof=1))
    return {
        "step": label,
        "rank_ic": float(daily.mean()),
        "icir": float(daily.mean() / std * math.sqrt(ANNUAL_DAYS))
        if std > 0
        else np.nan,
        "ic_positive_ratio": float((daily > 0).mean()),
        "n_days": int(len(daily)),
    }


def cross_sectional_corr(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    a, b = a.align(b, join="inner", axis=0)
    a, b = a.align(b, join="inner", axis=1)
    return a.rank(axis=1, pct=True).corrwith(b.rank(axis=1, pct=True), axis=1)


def build_style_proxies(
    ret: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    amount: pd.DataFrame,
    float_mktcap: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    market = ret.mean(axis=1)
    mean_ret = ret.rolling(60, min_periods=40).mean()
    mean_market = market.rolling(60, min_periods=40).mean()
    covariance = (
        ret.mul(market, axis=0).rolling(60, min_periods=40).mean()
        - mean_ret.mul(mean_market, axis=0)
    )
    beta = covariance.div(market.rolling(60, min_periods=40).var(), axis=0)
    adv20 = amount.where(amount > 0).rolling(20, min_periods=15).mean()
    turnover_proxy = amount.where(amount > 0) / float_mktcap.where(float_mktcap > 0)
    amihud20 = (
        ret.abs().div(amount.where(amount > 0)).rolling(20, min_periods=15).mean()
    )
    overnight = open_px.div(close_px.shift(1)).sub(1.0)
    intraday = close_px.div(open_px.where(open_px > 0)).sub(1.0)
    return {
        "Size": np.log(float_mktcap.where(float_mktcap > 0)),
        "Momentum20": (1.0 + ret).rolling(20, min_periods=15).apply(
            np.prod, raw=True
        )
        - 1.0,
        "Momentum60": (1.0 + ret).rolling(60, min_periods=40).apply(
            np.prod, raw=True
        )
        - 1.0,
        "Volatility20": ret.rolling(20, min_periods=15).std(),
        "LiquidityLogADV20": np.log(adv20),
        "TurnoverProxy20": turnover_proxy.rolling(20, min_periods=15).mean(),
        "Amihud20": amihud20,
        "Beta60": beta,
        "Reversal1D": -ret,
        "OvernightReturn": overnight,
        "IntradayReturn": intraday,
    }


def build_style_correlation(
    signals: Dict[str, pd.DataFrame],
    styles: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for signal_name, signal in signals.items():
        for style_name, style in styles.items():
            daily = cross_sectional_corr(
                signal.loc[START:END], style.loc[START:END]
            ).dropna()
            std = float(daily.std(ddof=1))
            rows.append(
                {
                    "signal": signal_name,
                    "style": style_name,
                    "mean_cs_spearman": float(daily.mean()),
                    "t_stat": float(daily.mean() / std * math.sqrt(len(daily)))
                    if std > 0
                    else np.nan,
                    "n_days": int(len(daily)),
                }
            )
    return pd.DataFrame(rows)


def build_neutralization_ladder(
    raw: pd.DataFrame,
    industry: pd.DataFrame,
    styles: Dict[str, pd.DataFrame],
    ret: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    log_size = styles["Size"].reindex_like(raw)
    raw_z = cs_zscore(raw)
    size = cs_zscore(panel_cross_sectional_residual(raw_z, [log_size]))
    industry_only = cs_zscore(panel_industry_demean(raw_z, industry))
    size_industry = cs_zscore(neutralize_size_industry(raw, industry, np.exp(log_size)))

    signals = {
        "Raw": raw_z,
        "Size": size,
        "Industry": industry_only,
        "Size+Industry": size_industry,
    }
    current = size_industry
    for label, style_name in (
        ("+Momentum20", "Momentum20"),
        ("+Volatility20", "Volatility20"),
        ("+LiquidityADV20", "LiquidityLogADV20"),
        ("+Beta60", "Beta60"),
    ):
        current = cs_zscore(
            panel_cross_sectional_residual(
                current, [styles[style_name].reindex_like(current)]
            )
        )
        signals[label] = current
    summary = pd.DataFrame(
        [ic_summary(signal.loc[START:END], ret.loc[START:END], label) for label, signal in signals.items()]
    )
    return summary, signals


def build_rolling_ic(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    daily = rank_ic(signal.loc[START:END].shift(1), ret.loc[START:END]).rename(
        "rank_ic"
    )
    out = daily.to_frame()
    for window in (60, 120, 250):
        out[f"mean_{window}d"] = daily.rolling(window, min_periods=window // 2).mean()
        out[f"std_{window}d"] = daily.rolling(window, min_periods=window // 2).std()
        out[f"icir_{window}d"] = (
            out[f"mean_{window}d"]
            / out[f"std_{window}d"]
            * math.sqrt(ANNUAL_DAYS)
        )
    out["se_250d"] = out["std_250d"] / np.sqrt(
        daily.rolling(250, min_periods=125).count()
    )
    return out


def build_industry_ic(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    industry: pd.DataFrame,
    min_names: int = 20,
) -> pd.DataFrame:
    sig = signal.loc[START:END].shift(1)
    ret = ret.reindex_like(sig)
    industry = industry.reindex_like(sig)
    values: Dict[str, List[float]] = {}
    counts: Dict[str, List[int]] = {}
    for date in sig.index:
        frame = pd.DataFrame(
            {"signal": sig.loc[date], "ret": ret.loc[date], "industry": industry.loc[date]}
        ).dropna()
        for name, part in frame.groupby("industry"):
            if len(part) < min_names:
                continue
            ic = part["signal"].corr(part["ret"], method="spearman")
            if pd.notna(ic):
                key = str(name)
                values.setdefault(key, []).append(float(ic))
                counts.setdefault(key, []).append(int(len(part)))
    rows = []
    for name, series in values.items():
        data = pd.Series(series, dtype=float)
        std = float(data.std(ddof=1))
        rows.append(
            {
                "industry_code": name,
                "industry": CITICS_IND_CODE_DICT.get(name, name),
                "rank_ic": float(data.mean()),
                "icir": float(data.mean() / std * math.sqrt(ANNUAL_DAYS))
                if std > 0
                else np.nan,
                "ic_positive_ratio": float((data > 0).mean()),
                "n_days": int(len(data)),
                "mean_names": float(np.mean(counts[name])),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic")


def build_industry_active_weight(
    signal: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    sig = signal.loc[START:END].shift(1)
    industry = industry.reindex_like(sig)
    active_rows = []
    for date in sig.index:
        frame = pd.DataFrame(
            {"signal": sig.loc[date], "industry": industry.loc[date]}
        ).dropna()
        if len(frame) < 100:
            continue
        selected = frame["signal"].rank(pct=True, method="first") > 0.90
        universe_weight = frame.groupby("industry").size() / len(frame)
        selected_frame = frame[selected]
        if selected_frame.empty:
            continue
        selected_weight = (
            selected_frame.groupby("industry").size() / len(selected_frame)
        )
        joined = pd.concat(
            [
                universe_weight.rename("universe_weight"),
                selected_weight.rename("g10_weight"),
            ],
            axis=1,
        ).fillna(0.0)
        joined["active_weight"] = joined["g10_weight"] - joined["universe_weight"]
        joined["date"] = date
        joined["industry"] = joined.index.astype(str)
        active_rows.append(joined.reset_index(drop=True))
    daily = pd.concat(active_rows, ignore_index=True)
    return (
        daily.groupby("industry")[["universe_weight", "g10_weight", "active_weight"]]
        .mean()
        .sort_values("active_weight")
        .reset_index()
        .rename(columns={"industry": "industry_code"})
        .assign(
            industry=lambda x: x["industry_code"].map(
                lambda code: CITICS_IND_CODE_DICT.get(code, code)
            )
        )
    )


def build_worst_windows(
    excess_ret: pd.Series,
    *,
    horizons: Tuple[int, ...] = (20, 60, 120),
    top_n: int = 3,
) -> pd.DataFrame:
    """Worst non-overlapping compounded excess-return windows."""
    s = excess_ret.dropna().sort_index()
    rows = []
    for horizon in horizons:
        rolling = (1.0 + s).rolling(horizon, min_periods=horizon).apply(
            np.prod, raw=True
        ) - 1.0
        candidates = rolling.dropna().sort_values()
        selected: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for end, value in candidates.items():
            loc = s.index.get_loc(end)
            start = s.index[loc - horizon + 1]
            if any(not (end < old_start or start > old_end) for old_start, old_end in selected):
                continue
            window = s.loc[start:end]
            selected.append((start, end))
            rows.append(
                {
                    "horizon_days": horizon,
                    "rank_within_horizon": len(selected),
                    "start": start,
                    "end": end,
                    "compounded_excess_return": float(value),
                    "annualized_return_linear": float(window.mean() * ANNUAL_DAYS),
                    "daily_volatility": float(window.std(ddof=1)),
                    "positive_day_ratio": float((window > 0).mean()),
                }
            )
            if len(selected) >= top_n:
                break
    return pd.DataFrame(rows)


def build_drawdown_series(excess_ret: pd.Series) -> pd.DataFrame:
    s = excess_ret.dropna().sort_index()
    nav = (1.0 + s).cumprod()
    hwm = nav.cummax()
    drawdown = nav / hwm - 1.0
    return pd.DataFrame({"nav": nav, "high_water_mark": hwm, "drawdown": drawdown})


def _direct_long_excess(
    aligned_signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    top_frac: float = 0.10,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Long-only exact-universe excess when signal is already aligned to return date."""
    sig = aligned_signal.reindex_like(ret)
    valid = sig.notna() & ret.notna()
    ranks = sig.rank(axis=1, pct=True, method="first")
    selected = valid & (ranks > 1.0 - top_frac)
    weights = selected.div(selected.sum(axis=1).replace(0, np.nan), axis=0)
    long_ret = weights.mul(ret).sum(axis=1, min_count=1)
    universe_ret = ret.where(valid).mean(axis=1)
    excess = long_ret - universe_ret
    turnover = weights.fillna(0).diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].fillna(0).abs().sum()
    return excess, turnover, selected.sum(axis=1), valid.sum(axis=1)


def build_tradability_diagnostics(
    signal: pd.DataFrame,
    ret_c2c: pd.DataFrame,
    ret_o2o: pd.DataFrame,
    close_px: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Formation/execution-date filters plus close-T→open-(T+1) timing diagnostic.

    This is not a blocked-order replay. The execution-date variant conservatively
    excludes names that are ST, suspended or close-limit on the return date.
    """
    start = signal.index.min().to_pydatetime()
    end = signal.index.max().to_pydatetime()
    not_limit = Factor_Dev_Lib.get_EOD_Not_Limit(start, end).reindex_like(signal)
    not_st = Factor_Dev_Lib.get_EOD_Not_ST(start, end).reindex_like(signal)
    trade_status = Factor_Dev_Lib.get_TradeStatus(start, end).reindex_like(signal)
    listed_days = close_px.reindex_like(signal).notna().astype(float).cumsum()
    tradable = (
        not_limit.eq(1)
        & not_st.eq(1)
        & trade_status.eq(1)
        & listed_days.ge(60)
    )

    c2c = ret_c2c.reindex_like(signal)
    o2o = ret_o2o.reindex_like(signal)
    variants = {
        "unfiltered_c2c_lag1": signal.shift(1),
        "formation_filtered_c2c_lag1": signal.where(tradable).shift(1),
        "ex_post_execution_filtered_c2c_lag1": signal.shift(1).where(tradable),
        "next_open_o2o_lag2": signal.where(tradable).shift(2),
    }
    rows = []
    daily = {}
    for label, aligned in variants.items():
        returns = o2o if label == "next_open_o2o_lag2" else c2c
        excess, turnover, selected_n, universe_n = _direct_long_excess(
            aligned.loc[START:END], returns.loc[START:END]
        )
        gross = series_performance(excess)
        net_excess = excess - turnover * 0.0015
        net = series_performance(net_excess)
        ic = rank_ic(aligned.loc[START:END], returns.loc[START:END]).dropna()
        ic_std = float(ic.std(ddof=1))
        rows.append(
            {
                "variant": label,
                "rank_ic": float(ic.mean()),
                "icir": float(ic.mean() / ic_std * math.sqrt(ANNUAL_DAYS))
                if ic_std > 0
                else np.nan,
                "gross_excess_sharpe": gross["sharpe"],
                "gross_excess_annual_return": gross["annu_ret"],
                "gross_excess_mdd": gross["max_drawdown"],
                "net_excess_sharpe_15bp": net["sharpe"],
                "net_excess_annual_return_15bp": net["annu_ret"],
                "daily_turnover": float(turnover.mean()),
                "mean_selected_names": float(selected_n.mean()),
                "mean_universe_names": float(universe_n.mean()),
                "tradability_coverage": float(
                    aligned.notna().sum(axis=1).div(
                        signal.shift(1 if "lag1" in label else 2)
                        .notna()
                        .sum(axis=1)
                        .replace(0, np.nan)
                    ).mean()
                ),
                "is_blocked_order_replay": False,
            }
        )
        daily[f"{label}_gross_excess"] = excess
        daily[f"{label}_net_excess_15bp"] = net_excess
        daily[f"{label}_turnover"] = turnover
    return pd.DataFrame(rows), pd.DataFrame(daily)


def build_ma_sensitivity(
    ret: pd.DataFrame,
    industry: pd.DataFrame,
    float_mktcap: pd.DataFrame,
) -> pd.DataFrame:
    long = pd.read_parquet(LONG_CACHE, columns=["date", "symbol", "tgd_eps"])
    long["date"] = pd.to_datetime(long["date"])
    eps = long.pivot(index="date", columns="symbol", values="tgd_eps").sort_index()
    eps = eps.reindex(index=ret.index, columns=ret.columns)
    rows = []
    for window in (10, 20, 30, 60):
        signal = eps.rolling(window, min_periods=max(5, window // 2)).mean()
        si = cs_zscore(neutralize_size_industry(signal, industry, float_mktcap))
        row = ic_summary(si.loc[START:END], ret.loc[START:END], f"MA{window}")
        row["is_frozen_window"] = window == 20
        rows.append(row)
    return pd.DataFrame(rows)


def build_apm_comparison(
    tgd_si: pd.DataFrame,
    ret: pd.DataFrame,
) -> pd.DataFrame:
    if not APM_CACHE.exists():
        return pd.DataFrame()
    apm = pd.read_parquet(APM_CACHE).sort_index()
    apm.index = pd.to_datetime(apm.index)
    common_i = tgd_si.index.intersection(apm.index).intersection(ret.index)
    common_c = tgd_si.columns.intersection(apm.columns).intersection(ret.columns)
    tgd = tgd_si.loc[common_i, common_c]
    apm = apm.loc[common_i, common_c]
    future_ret = ret.loc[common_i, common_c]
    signal_corr = cross_sectional_corr(tgd, apm).dropna()
    tgd_resid = cs_zscore(panel_cross_sectional_residual(tgd, [apm]))
    apm_resid = cs_zscore(panel_cross_sectional_residual(apm, [tgd]))
    rows = []
    for label, signal in (
        ("TGD20", tgd),
        ("TGD20_perp_APM_SessionResidual", tgd_resid),
        ("APM_SessionResidual", apm),
        ("APM_SessionResidual_perp_TGD20", apm_resid),
    ):
        row = ic_summary(signal, future_ret, label)
        row["mean_signal_cs_corr"] = (
            float(signal_corr.mean()) if "perp" not in label else np.nan
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    raw_tgd = out.loc[out["step"] == "TGD20", "icir"].iloc[0]
    raw_apm = out.loc[out["step"] == "APM_SessionResidual", "icir"].iloc[0]
    out["icir_retention"] = np.where(
        out["step"] == "TGD20_perp_APM_SessionResidual",
        out["icir"] / raw_tgd,
        np.where(
            out["step"] == "APM_SessionResidual_perp_TGD20",
            out["icir"] / raw_apm if raw_apm != 0 else np.nan,
            1.0,
        ),
    )
    return out


def execution_curve(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
    *,
    freq: int = 1,
    entry: Optional[float] = None,
    exit_: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series]:
    sampled = downsample_signal(signal, freq) if freq > 1 else signal
    if entry is not None and exit_ is not None:
        long_mask, short_mask = buffer_ls_masks(
            sampled, entry_frac=entry, exit_frac=exit_
        )
    else:
        long_mask, short_mask = plain_ls_masks(sampled, top_frac=0.10)
    w_long, w_short = weights_from_masks(
        sampled, long_mask, short_mask, method="ew"
    )
    gross, turnover = pnl_and_turnover_from_weights(
        w_long, w_short, ret, signal_shift=1
    )
    return net_pnl_series(gross, turnover, 0.0015), turnover


def build_execution_curves(
    signal: pd.DataFrame,
    ret: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    curves = {}
    rows = []
    schemes = [
        ("Daily", 1, None, None),
        ("Every5D", 5, None, None),
        ("Every10D", 10, None, None),
        ("Every20D", 20, None, None),
        ("Buffer5_15", 1, 0.05, 0.15),
    ]
    for label, freq, entry, exit_ in schemes:
        net, turnover = execution_curve(
            signal.loc[START:END],
            ret.loc[START:END],
            freq=freq,
            entry=entry,
            exit_=exit_,
        )
        curves[label] = net
        perf = series_performance(net.dropna())
        rows.append(
            {
                "scheme": label,
                "net_sharpe_15bp": perf["sharpe"],
                "net_annual_return_15bp": perf["annu_ret"],
                "net_max_drawdown_15bp": perf["max_drawdown"],
                "daily_turnover": float(turnover.mean()),
            }
        )
    return pd.DataFrame(curves), pd.DataFrame(rows)


def draw_plots(
    rolling_ic: pd.DataFrame,
    style_corr: pd.DataFrame,
    neutral_ladder: pd.DataFrame,
    industry_ic: pd.DataFrame,
    industry_weight: pd.DataFrame,
    ma_sensitivity: pd.DataFrame,
    apm: pd.DataFrame,
    execution_daily: pd.DataFrame,
    worst_windows: pd.DataFrame,
    drawdown: pd.DataFrame,
    tradability: pd.DataFrame,
) -> None:
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(rolling_ic.index, rolling_ic["mean_60d"], label="60D mean RankIC", lw=1.2)
    ax.plot(rolling_ic.index, rolling_ic["mean_250d"], label="250D mean RankIC", lw=2)
    lower = rolling_ic["mean_250d"] - rolling_ic["se_250d"]
    upper = rolling_ic["mean_250d"] + rolling_ic["se_250d"]
    ax.fill_between(rolling_ic.index, lower, upper, alpha=0.18, label="250D ±1 SE")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="RankIC", title="TGD20 rolling RankIC stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_OUT / "rolling_rank_ic.png", dpi=180)
    plt.close(fig)

    pivot = style_corr.pivot(
        index="style", columns="signal", values="mean_cs_spearman"
    )
    limit = max(0.45, float(np.nanmax(np.abs(pivot.values))))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("TGD20 style correlation (mean daily CS Spearman)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "expanded_style_correlation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(neutral_ladder["step"], neutral_ladder["rank_ic"], color="steelblue")
    axes[1].bar(neutral_ladder["step"], neutral_ladder["icir"], color="darkorange")
    axes[0].set_ylabel("Mean RankIC")
    axes[1].set_ylabel("Annualized ICIR")
    for ax in axes:
        ax.tick_params(axis="x", rotation=55)
        ax.axhline(0, color="black", lw=0.8)
    axes[0].set_title("Sequential neutralization: RankIC")
    axes[1].set_title("Sequential neutralization: ICIR")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "neutralization_waterfall.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = industry_ic.sort_values("rank_ic")
    colors = np.where(plot_data["rank_ic"] >= 0, "steelblue", "firebrick")
    ax.barh(plot_data["industry"].astype(str), plot_data["rank_ic"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Mean RankIC", title="TGD20 RankIC by CITICS industry")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "industry_rank_ic.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_data = industry_weight.sort_values("active_weight")
    colors = np.where(plot_data["active_weight"] >= 0, "steelblue", "firebrick")
    ax.barh(
        plot_data["industry"].astype(str), plot_data["active_weight"], color=colors
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Average G10 weight minus universe weight",
        title="TGD20 G10 active industry weight",
    )
    fig.tight_layout()
    fig.savefig(FIG_OUT / "g10_industry_active_weight.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(ma_sensitivity["step"], ma_sensitivity["rank_ic"], color="steelblue")
    axes[1].bar(ma_sensitivity["step"], ma_sensitivity["icir"], color="darkorange")
    axes[0].set_ylabel("Mean RankIC")
    axes[1].set_ylabel("Annualized ICIR")
    axes[0].set_title("MA-window sensitivity: RankIC")
    axes[1].set_title("MA-window sensitivity: ICIR")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "ma_window_sensitivity.png", dpi=180)
    plt.close(fig)

    if not apm.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        apm_label_map = {
            "TGD20": "TGD20",
            "TGD20_perp_APM_SessionResidual": "TGD20 ⊥\nadapted APM",
            "APM_SessionResidual": "adapted APM",
            "APM_SessionResidual_perp_TGD20": "adapted APM\n⊥ TGD20",
        }
        apm_labels = apm["step"].map(apm_label_map).fillna(apm["step"])
        axes[0].bar(apm_labels, apm["rank_ic"], color="steelblue")
        axes[1].bar(apm_labels, apm["icir"], color="darkorange")
        for ax in axes:
            ax.axhline(0, color="black", lw=0.8)
            ax.tick_params(axis="x", rotation=35)
        axes[0].set_title("TGD20 vs adapted APM: RankIC")
        axes[1].set_title("TGD20 vs adapted APM: ICIR")
        fig.tight_layout()
        fig.savefig(FIG_OUT / "tgd20_apm_independence.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for column in execution_daily:
        compound(execution_daily[column]).plot(ax=ax, label=column)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        ylabel="Cumulative net H-L return",
        title="TGD20 rebalance schemes: net curves at 15bp round-trip cost",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_OUT / "rebalance_net_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(drawdown.index, drawdown["nav"], color="steelblue", label="SI G10 excess NAV")
    axes[0].plot(
        drawdown.index,
        drawdown["high_water_mark"],
        color="0.45",
        lw=0.9,
        label="High-water mark",
    )
    axes[0].set(ylabel="NAV", title="TGD20 long-only excess: NAV and drawdown")
    axes[0].legend(fontsize=8)
    axes[1].fill_between(
        drawdown.index,
        drawdown["drawdown"],
        0,
        color="firebrick",
        alpha=0.45,
    )
    axes[1].set(ylabel="Drawdown", xlabel="Date")
    for _, row in worst_windows[worst_windows["horizon_days"] == 60].iterrows():
        axes[1].axvspan(
            pd.Timestamp(row["start"]),
            pd.Timestamp(row["end"]),
            color="darkorange",
            alpha=0.12,
        )
    fig.tight_layout()
    fig.savefig(FIG_OUT / "worst_periods_drawdown.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    label_map = {
        "unfiltered_c2c_lag1": "Unfiltered\nC2C",
        "formation_filtered_c2c_lag1": "Formation\nfilter",
        "ex_post_execution_filtered_c2c_lag1": "Ex-post\nfilter*",
        "next_open_o2o_lag2": "Next-open\nO2O",
    }
    labels = tradability["variant"].map(label_map).fillna(tradability["variant"])
    axes[0].bar(labels, tradability["gross_excess_sharpe"], color="steelblue")
    axes[0].set(ylabel="Gross excess Sharpe", title="Tradability / timing diagnostics")
    axes[1].bar(labels, tradability["net_excess_sharpe_15bp"], color="darkorange")
    axes[1].set(ylabel="Net excess Sharpe @15bp", title="Cost-adjusted comparison")
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "tradability_execution_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    print("Loading canonical DolphinDB-derived TGD20 panel", flush=True)
    raw = pd.read_parquet(PANEL_CACHE).sort_index()
    raw.index = pd.to_datetime(raw.index)

    print("Loading point-in-time EOD and industry data from DolphinDB", flush=True)
    eod, session = load_eod_enriched_tables(pd.Timestamp("2019-01-01"), END)
    try:
        session.run(intraday_lib.ddb_functions)
        ret = Factor_Dev_Lib.get_Ret_Matrix(
            raw.index.min().to_pydatetime(),
            raw.index.max().to_pydatetime(),
            method="c2c",
        ).reindex(index=raw.index, columns=raw.columns)
        open_px = eod.open.reindex_like(raw)
        close_px = eod.close.reindex_like(raw)
        amount = eod.amount.reindex_like(raw)
        float_mktcap = eod.float_mktcap.reindex_like(raw)
        industry = load_citics_industry_panel(
            raw.index.min(), raw.index.max()
        ).reindex_like(raw)

        si = cs_zscore(neutralize_size_industry(raw, industry, float_mktcap))
        styles = build_style_proxies(ret, open_px, close_px, amount, float_mktcap)
        style_corr = build_style_correlation({"Raw": raw, "Size+Industry": si}, styles)
        neutral_ladder, signals = build_neutralization_ladder(
            raw, industry, styles, ret
        )
        rolling_ic = build_rolling_ic(si, ret)
        industry_ic = build_industry_ic(si, ret, industry)
        industry_weight = build_industry_active_weight(si, industry)
        ma_sensitivity = build_ma_sensitivity(ret, industry, float_mktcap)
        apm = build_apm_comparison(si.loc[START:END], ret.loc[START:END])
        execution_daily, execution_summary = build_execution_curves(si, ret)
        prior_daily = pd.read_csv(
            DATA_OUT / "daily_buyside_series.csv", index_col=0, parse_dates=True
        )
        excess_ret = prior_daily["size_industry_excess_return"].dropna()
        worst_windows = build_worst_windows(excess_ret)
        drawdown = build_drawdown_series(excess_ret)
        ret_o2o = Factor_Dev_Lib.get_Ret_Matrix(
            raw.index.min().to_pydatetime(),
            raw.index.max().to_pydatetime(),
            method="o2o",
        ).reindex_like(raw)
        tradability, tradability_daily = build_tradability_diagnostics(
            si, ret, ret_o2o, close_px
        )

        save_frame(rolling_ic, "rolling_rank_ic.csv")
        save_frame(style_corr, "expanded_style_correlation.csv", index=False)
        save_frame(neutral_ladder, "neutralization_ladder.csv", index=False)
        save_frame(industry_ic, "industry_rank_ic.csv", index=False)
        save_frame(industry_weight, "g10_industry_active_weight.csv", index=False)
        save_frame(ma_sensitivity, "ma_window_sensitivity.csv", index=False)
        if not apm.empty:
            save_frame(apm, "tgd20_apm_comparison.csv", index=False)
        save_frame(execution_daily, "rebalance_net_daily.csv")
        save_frame(execution_summary, "rebalance_net_summary.csv", index=False)
        save_frame(worst_windows, "worst_excess_windows.csv", index=False)
        save_frame(drawdown, "long_only_excess_drawdown.csv")
        save_frame(tradability, "tradability_execution_summary.csv", index=False)
        save_frame(tradability_daily, "tradability_execution_daily.csv")

        metadata = {
            "sample": {"start": str(START.date()), "end": str(END.date())},
            "signal_shift": 1,
            "annual_days": ANNUAL_DAYS,
            "data_provenance": {
                "tgd20": "canonical panel generated from DolphinDB Stock_one_minute",
                "returns_and_styles": "point-in-time DolphinDB/Wind EOD loaders",
                "industry": "CITICS historical industry panel",
                "apm": (
                    "DolphinDB-derived CSI1000 APM_SessionResidual (adapted) cache; "
                    "stock PM bars are exact, index leg uses an EOD session proxy"
                ),
            },
            "research_guardrail": (
                "MA sensitivity is a robustness diagnostic; TGD20 remains frozen at MA20"
            ),
            "execution_guardrail": (
                "Tradability variants are cross-sectional filtering/timing diagnostics, "
                "not a stateful blocked-order replay. next_open_o2o_lag2 uses close-T "
                "signal and open-(T+1) to open-(T+2) return."
            ),
        }
        (DATA_OUT / "junior_qr_diagnostics_meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        draw_plots(
            rolling_ic,
            style_corr,
            neutral_ladder,
            industry_ic,
            industry_weight,
            ma_sensitivity,
            apm,
            execution_daily,
            worst_windows,
            drawdown,
            tradability,
        )
    finally:
        try:
            session.close()
        except Exception:
            pass
    print("Wrote Junior-QR diagnostics from real project data", flush=True)


if __name__ == "__main__":
    main()
