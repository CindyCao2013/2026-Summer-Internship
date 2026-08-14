"""ClickHouse minute-bar factors → daily features for F² Agent Lite.

Sources (cmds):
  - SSE_AL_KLIN_EXG / LOCAL_SSE_AL_KLIN_EXG  (Type='1MIN')
  - SZSE_AL_KLIN_CMD / LOCAL_SZSE_AL_KLIN_CMD (Type='1MIN')

Factors (日频，严格因果滚动):
  1. minute_amplitude  — 分钟理想振幅 × 日内高振幅占比，再 lookback 平滑
  2. price_jump        — 振幅 Z-score 跳跃时点的价峰/价谷合成
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# (table, exchange_suffix, exchtime_is_dt64)
DEFAULT_KLIN_TABLES: Tuple[Tuple[str, str, bool], ...] = (
    ("SSE_AL_KLIN_EXG", ".SH", True),
    ("SZSE_AL_KLIN_CMD", ".SZ", False),
)

# Prefer LOCAL_* for large scans (same schema, MergeTree local).
LOCAL_KLIN_TABLES: Tuple[Tuple[str, str, bool], ...] = (
    ("LOCAL_SSE_AL_KLIN_EXG", ".SH", True),
    ("LOCAL_SZSE_AL_KLIN_CMD", ".SZ", False),
)

CONTINUOUS_SESSION_SQL = """
(
  (toHour(ExchTime) = 9 AND toMinute(ExchTime) >= 30)
  OR (toHour(ExchTime) = 10)
  OR (toHour(ExchTime) = 11 AND toMinute(ExchTime) <= 30)
  OR (toHour(ExchTime) >= 13 AND toHour(ExchTime) < 15)
  OR (toHour(ExchTime) = 15 AND toMinute(ExchTime) = 0)
)
"""


def _ch_literal_start(ts: str, *, dt64: bool) -> str:
    text = str(ts).strip()
    if len(text) == 10:
        text = f"{text} 00:00:00"
    if dt64:
        return f"toDateTime64('{text}', 6, 'Asia/Shanghai')"
    return f"toDateTime('{text}', 'Asia/Shanghai')"


def _bare_and_suffix(symbols: Sequence[str]) -> Tuple[List[str], List[str]]:
    sh, sz = [], []
    for s in symbols:
        s = str(s).strip()
        bare = s.split(".")[0]
        if s.endswith(".SZ") or bare.startswith(("0", "3")):
            sz.append(bare)
        else:
            sh.append(bare)
    return sorted(set(sh)), sorted(set(sz))


def fetch_minute_data_from_clickhouse(
    client,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    *,
    use_local_tables: bool = False,
    tables: Optional[Sequence[Tuple[str, str, bool]]] = None,
) -> pd.DataFrame:
    """Batch-read 1-minute OHLCV for Wind-style symbols from ClickHouse KLIN."""
    if not symbols:
        return pd.DataFrame(
            columns=["symbol", "date", "minute", "open", "high", "low", "close", "volume", "amount"]
        )

    table_specs = list(tables) if tables is not None else (
        list(LOCAL_KLIN_TABLES) if use_local_tables else list(DEFAULT_KLIN_TABLES)
    )
    sh_syms, sz_syms = _bare_and_suffix(symbols)
    # end exclusive: next calendar day so end_date session is included
    end_excl = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frames: List[pd.DataFrame] = []
    for table, suffix, dt64 in table_specs:
        if suffix == ".SH":
            bare = sh_syms
        else:
            bare = sz_syms
        if not bare:
            continue
        in_list = ", ".join(f"'{c}'" for c in bare)
        start_lit = _ch_literal_start(start_date, dt64=dt64)
        end_lit = _ch_literal_start(end_excl, dt64=dt64)
        sql = f"""
        SELECT
            concat(Symbol, '{suffix}') AS symbol,
            toDate(ExchTime) AS date,
            ExchTime AS minute,
            toFloat64(Open) AS open,
            toFloat64(High) AS high,
            toFloat64(Low) AS low,
            toFloat64(Close) AS close,
            toFloat64(Volume) AS volume,
            toFloat64(Amount) AS amount
        FROM cmds.{table}
        WHERE Type = '1MIN'
          AND Symbol IN ({in_list})
          AND ExchTime >= {start_lit}
          AND ExchTime < {end_lit}
          AND {CONTINUOUS_SESSION_SQL}
        ORDER BY Symbol, ExchTime
        """
        result = client.query(sql)
        if not result.result_rows:
            continue
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["symbol", "date", "minute", "open", "high", "low", "close", "volume", "amount"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["minute"] = pd.to_datetime(out["minute"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values(["symbol", "minute"]).reset_index(drop=True)


def _daily_amplitude_raw(df: pd.DataFrame) -> pd.Series:
    """Per (symbol, date): mean(amp) * fraction of minutes in top-20% amp."""
    work = df.copy()
    mid = (work["high"] + work["low"]) / 2.0
    work["amp"] = np.where(mid > 0, (work["high"] - work["low"]) / mid, np.nan)
    work = work.dropna(subset=["amp"])
    if work.empty:
        return pd.Series(dtype=float)

    def _one_day(g: pd.DataFrame) -> float:
        amp = g["amp"]
        if amp.empty:
            return np.nan
        thr = amp.quantile(0.8)
        high_frac = float((amp >= thr).mean()) if np.isfinite(thr) else np.nan
        return float(amp.mean()) * high_frac

    daily = work.groupby(["symbol", "date"], sort=False).apply(_one_day, include_groups=False)
    daily.name = "minute_amplitude_raw"
    return daily


def _smooth_symbol_day_series(raw: pd.Series, lookback: int, name: str) -> pd.Series:
    """Causal rolling mean on a (symbol, date) MultiIndex series."""
    if raw.empty:
        return pd.Series(dtype=float, name=name)
    min_p = max(3, lookback // 2)
    parts = []
    for sym, s in raw.groupby(level=0):
        s = s.droplevel(0).sort_index()
        smoothed = s.rolling(window=lookback, min_periods=min_p).mean()
        smoothed.index = pd.MultiIndex.from_product([[sym], smoothed.index], names=["symbol", "date"])
        parts.append(smoothed)
    out = pd.concat(parts).sort_index()
    out.name = name
    return out


def calc_minute_amplitude_factor(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Minute ideal amplitude + intraday amp cut, then causal lookback mean.

    Kaiyuan 报告该族因子 rankIC 为负，这里输出取负号，使更高值对应更低振幅
    （与模型「高分做多」方向一致）。滚动窗口仅使用截止 t 的历史日。
    """
    smoothed = _smooth_symbol_day_series(_daily_amplitude_raw(df), lookback, "minute_amplitude")
    if smoothed.empty:
        return smoothed
    return (-smoothed).rename("minute_amplitude")


def _daily_price_jump_raw(df: pd.DataFrame) -> pd.Series:
    """Jump minutes = amp > 1σ within day; signed by close vs day mid."""
    work = df.copy()
    mid = (work["high"] + work["low"]) / 2.0
    work["amp"] = np.where(mid > 0, (work["high"] - work["low"]) / mid, np.nan)
    work = work.dropna(subset=["amp", "close"])
    if work.empty:
        return pd.Series(dtype=float)

    def _one_day(g: pd.DataFrame) -> float:
        amp = g["amp"].to_numpy(dtype=float)
        close = g["close"].to_numpy(dtype=float)
        if len(amp) < 10:
            return np.nan
        mu = np.nanmean(amp)
        sd = np.nanstd(amp)
        if not np.isfinite(sd) or sd < 1e-12:
            return 0.0
        z = (amp - mu) / sd
        jump = z > 1.0
        if not jump.any():
            return 0.0
        day_mid = np.nanmedian(close)
        # +1 peak (above mid), -1 valley (below), 0 ridge (~mid)
        signed = np.where(
            close > day_mid * 1.001,
            1.0,
            np.where(close < day_mid * 0.999, -1.0, 0.0),
        )
        return float(np.nanmean(z[jump] * signed[jump]))

    daily = work.groupby(["symbol", "date"], sort=False).apply(_one_day, include_groups=False)
    daily.name = "price_jump_raw"
    return daily


def calc_price_jump_factor(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Price-jump factor (peak/ridge/valley) with causal lookback smoothing."""
    return _smooth_symbol_day_series(_daily_price_jump_raw(df), lookback, "price_jump")


def compute_minute_daily_factors(
    minute_df: pd.DataFrame,
    lookback: int = 10,
) -> pd.DataFrame:
    """Return long frame: date, symbol, minute_amplitude, price_jump."""
    empty = pd.DataFrame(columns=["date", "symbol", "minute_amplitude", "price_jump"])
    if minute_df is None or minute_df.empty:
        return empty

    amp = calc_minute_amplitude_factor(minute_df, lookback=lookback)
    jmp = calc_price_jump_factor(minute_df, lookback=lookback)
    if amp.empty and jmp.empty:
        return empty

    panel = pd.concat([amp, jmp], axis=1).reset_index()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    for c in ["minute_amplitude", "price_jump"]:
        if c not in panel.columns:
            panel[c] = np.nan
        else:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")
    return (
        panel[["date", "symbol", "minute_amplitude", "price_jump"]]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
