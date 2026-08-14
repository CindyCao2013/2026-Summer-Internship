"""Fast Option-B SmartMoney Q from minute_feature frames."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from factor_cutting.smart_money import (
    LOOKBACK_DAYS,
    MIN_MINUTES_DEFAULT,
    TOP_CUMVOL_PCT,
    _q_one_window,
)


def compute_daily_smart_money_q_fast(
    minutes: pd.DataFrame,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    top_cumvol_pct: float = TOP_CUMVOL_PCT,
    min_minutes: int = MIN_MINUTES_DEFAULT,
    dates: Optional[Sequence] = None,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Same semantics as compute_daily_smart_money_q; less pandas in the hot loop."""
    need = {"date", "symbol", "volume", "amount", "close", "smart_score"}
    missing = need - set(minutes.columns)
    if missing:
        raise ValueError(f"minutes missing columns: {missing}")

    df = minutes
    want = None
    if dates is not None:
        want = set(pd.Timestamp(x) for x in pd.to_datetime(list(dates)))

    rows: list[dict] = []
    symbols = df["symbol"].astype(str).unique()
    n_sym = len(symbols)

    # group once
    for si, (sym, g) in enumerate(df.groupby(df["symbol"].astype(str), sort=False)):
        if progress_every and si > 0 and si % progress_every == 0:
            print(f"  Q compute {si}/{n_sym} symbols ...", flush=True)

        g = g.sort_values(["date", "bartime"] if "bartime" in g.columns else ["date"])
        # list of (date, arrays)
        day_pack = []
        for d, sub in g.groupby("date", sort=True):
            d = pd.Timestamp(d)
            day_pack.append(
                (
                    d,
                    sub["smart_score"].to_numpy(dtype=float),
                    sub["volume"].to_numpy(dtype=float),
                    sub["amount"].to_numpy(dtype=float),
                    sub["close"].to_numpy(dtype=float),
                    len(sub),
                )
            )
        if len(day_pack) < lookback_days:
            continue

        for i in range(lookback_days - 1, len(day_pack)):
            t = day_pack[i][0]
            if want is not None and t not in want:
                continue
            win = day_pack[i - lookback_days + 1 : i + 1]
            n_min = sum(p[5] for p in win)
            if n_min < min_minutes:
                rows.append({"date": t, "symbol": sym, "Q": np.nan})
                continue
            sc = np.concatenate([p[1] for p in win])
            vol = np.concatenate([p[2] for p in win])
            am = np.concatenate([p[3] for p in win])
            cl = np.concatenate([p[4] for p in win])
            q = _q_one_window(sc, vol, am, cl, top_cumvol_pct=top_cumvol_pct)
            rows.append({"date": t, "symbol": sym, "Q": q})

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "Q"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)
