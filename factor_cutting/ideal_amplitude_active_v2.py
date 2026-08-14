# ============================================================
# factor_cutting/ideal_amplitude_active_v2.py
# 理想振幅因子 2.0：实现振幅 / 主动净额波动
# ============================================================
"""IdealAmplitude_ActiveV2 — 振幅质量切割（振幅/主动净额波动）.

Distinct from original IdealAmplitude (price × amplitude). Uses L2 active
net flow volatility to separate institutional accumulation from noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_cutting.engine import CuttingSpec, KnifeSpec, ObjectSpec, OutputSpec

# ---------- 参数 ----------
AMP_EWM_SPAN = 5
AMP_EWM_MIN_PERIODS = 3
MIN_MINUTES_PER_DAY = 30
EPS = 1e-12

FORMULA_VERSION = "ideal_amplitude_active_v2_ewm5_amp_over_netvol"

IDEAL_AMPLITUDE_ACTIVE_V2_SPEC = CuttingSpec(
    name="ideal_amplitude_active_v2",
    paper="振幅因子的隐藏结构 (L2 ActiveV2 upgrade)",
    direction_paper="negative_ic",
    status="implemented_active_v2",
    object=ObjectSpec(
        variable="realized_amplitude",
        additive=False,
        formula="(High-Low)/Open",
    ),
    knife=KnifeSpec(
        variable="active_net_volatility",
        method="quantile_split",
        window=AMP_EWM_SPAN,
        formula="std((active_buy-active_sell)/amount)",
    ),
    output=OutputSpec(
        aggregate="mean",
        op="ratio",
        formula="ewm(amp/(net_vol+eps), span=5)",
    ),
)


def _realized_amplitude(minutes: pd.DataFrame) -> float:
    """日内实现振幅：(High - Low) / Open"""
    h = minutes["high"].max()
    l = minutes["low"].min()
    o = minutes["open"].iloc[0]
    if o <= 0 or pd.isna(o):
        return np.nan
    return float((h - l) / o)


def _active_net_volatility(minutes: pd.DataFrame) -> float:
    """主动买卖净额的分钟波动（相对成交额的比例的标准差）"""
    net = minutes["active_buy_amt"] - minutes["active_sell_amt"]
    amount = minutes["amount"]
    ratio = net / (amount + EPS)
    valid = (amount > 0) & np.isfinite(ratio)
    if int(valid.sum()) < 2:
        return np.nan
    return float(ratio[valid].std())


def compute_daily_amplitude(
    minutes: pd.DataFrame,
    min_minutes: int = MIN_MINUTES_PER_DAY,
) -> pd.DataFrame:
    """计算每日振幅、主动净额波动及原始因子。

    需要列：date, symbol, high, low, open, active_buy_amt, active_sell_amt, amount
    """
    need = {
        "date",
        "symbol",
        "high",
        "low",
        "open",
        "active_buy_amt",
        "active_sell_amt",
        "amount",
    }
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"IdealAmplitude missing columns: {missing}")

    rows = []
    for (sym, d), g in minutes.groupby(["symbol", "date"], sort=False):
        if len(g) < min_minutes:
            rows.append((d, sym, np.nan, np.nan, np.nan))
            continue
        amp = _realized_amplitude(g)
        vol = _active_net_volatility(g)
        raw = (
            amp / (vol + EPS)
            if (np.isfinite(amp) and np.isfinite(vol) and vol > EPS)
            else np.nan
        )
        rows.append((d, sym, amp, vol, raw))
    out = pd.DataFrame(
        rows,
        columns=["date", "symbol", "realized_amp", "active_net_vol", "amp_raw"],
    )
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def ewm_smooth_daily(
    daily: pd.DataFrame,
    span: int = AMP_EWM_SPAN,
    min_periods: int = AMP_EWM_MIN_PERIODS,
) -> pd.DataFrame:
    """对 amp_raw 按股票做 EWM 平滑，生成 amp_smooth。"""
    if "amp_raw" not in daily.columns:
        raise ValueError("daily missing amp_raw")
    out = daily.sort_values(["symbol", "date"]).copy()
    out["amp_smooth"] = out.groupby("symbol")["amp_raw"].transform(
        lambda x: x.ewm(span=span, min_periods=min_periods).mean()
    )
    return out


def apply_amplitude_gate(daily: pd.DataFrame, top_pct: float = 0.20) -> pd.DataFrame:
    """仅保留振幅截面高分位的样本（门控），其他置 NaN。"""
    out = daily.copy()
    for _date, idx in out.groupby("date").groups.items():
        amp = out.loc[idx, "realized_amp"]
        if amp.notna().sum() < 20:
            continue
        threshold = amp.quantile(1 - top_pct)
        out.loc[idx, "amp_raw"] = out.loc[idx, "amp_raw"].where(amp >= threshold)
    return out
