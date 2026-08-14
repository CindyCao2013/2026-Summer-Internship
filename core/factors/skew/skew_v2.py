"""SKEW v2: mechanism-purified return-distribution variants.

Priority (do NOT start with minute RSKEW):

P0  Vol-neutral / Vol-adjusted residual skew
P0  TailSKEW = upside-skew − downside-skew
P1  MAX-residual skew
P1  TGD-residual skew
P2  MAD winsorize sensitivity

All outputs are research panels; delivery alpha convention remains
``Alpha = -raw`` unless the panel is already an alpha residual.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from core.factors.skew.skew import alpha_from_skew
from liquidity_normalization import panel_cross_sectional_residual

_DEFAULT_MIN = {
    20: 10,
    60: 40,
    120: 80,
}


def mad_winsorize_cs(panel: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """Cross-sectional MAD winsorize (vectorized via apply)."""

    def _one_row(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 30:
            return s
        med = valid.median()
        mad = (valid - med).abs().median() * 1.4826
        if not np.isfinite(mad) or mad < 1e-12:
            return s
        return s.clip(med - n_mad * mad, med + n_mad * mad)

    return panel.apply(_one_row, axis=1)


def cs_residualize(
    y: pd.DataFrame,
    controls: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Daily cross-sectional residual of y on control panels (+ intercept)."""
    aligned = [c.reindex(index=y.index, columns=y.columns) for c in controls]
    return panel_cross_sectional_residual(y, list(aligned))


def vol_adjusted_skew(
    skew: pd.DataFrame,
    ret_1d: pd.DataFrame,
    *,
    vol_window: int = 20,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """SKEW / VOL — strip volatility scale from skew magnitude."""
    mp = vol_window // 2 if min_periods is None else int(min_periods)
    vol = ret_1d.rolling(vol_window, min_periods=mp).std()
    return skew / vol.replace(0.0, np.nan)


def vol_residual_skew(
    skew: pd.DataFrame,
    ret_1d: pd.DataFrame,
    *,
    vol_window: int = 20,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Cross-sectional residual of SKEW on VOL (P0 residualization)."""
    mp = vol_window // 2 if min_periods is None else int(min_periods)
    vol = ret_1d.rolling(vol_window, min_periods=mp).std()
    return cs_residualize(skew, [vol])


def upside_skew(
    ret_1d: pd.DataFrame,
    window: int = 60,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Upside third-moment ratio using clipped positive returns (dense)."""
    mp = _DEFAULT_MIN.get(window, max(5, window // 2))
    if min_periods is not None:
        mp = int(min_periods)
    up = ret_1d.clip(lower=0.0)
    m3 = up.pow(3).rolling(window, min_periods=mp).mean()
    m2 = up.pow(2).rolling(window, min_periods=mp).mean()
    return m3 / m2.pow(1.5).replace(0.0, np.nan)


def downside_skew(
    ret_1d: pd.DataFrame,
    window: int = 60,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Downside third-moment ratio using absolute negative returns (dense)."""
    mp = _DEFAULT_MIN.get(window, max(5, window // 2))
    if min_periods is not None:
        mp = int(min_periods)
    down = (-ret_1d).clip(lower=0.0)
    m3 = down.pow(3).rolling(window, min_periods=mp).mean()
    m2 = down.pow(2).rolling(window, min_periods=mp).mean()
    return m3 / m2.pow(1.5).replace(0.0, np.nan)


def tail_skew(
    ret_1d: pd.DataFrame,
    window: int = 60,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """TailSKEW = USKEW − DSKEW (lottery asymmetry of up vs down tails)."""
    return upside_skew(ret_1d, window, min_periods=min_periods) - downside_skew(
        ret_1d, window, min_periods=min_periods
    )


def max_return(
    ret_1d: pd.DataFrame,
    window: int = 20,
    *,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    mp = max(5, window // 2) if min_periods is None else int(min_periods)
    return ret_1d.rolling(window, min_periods=mp).max()


def max_residual_skew(
    skew: pd.DataFrame,
    ret_1d: pd.DataFrame,
    *,
    max_window: int = 20,
) -> pd.DataFrame:
    """SKEW ⊥ MAX20 — distribution component after lottery-max removal."""
    return cs_residualize(skew, [max_return(ret_1d, max_window)])


def tgd_residual_skew(
    skew: pd.DataFrame,
    tgd: pd.DataFrame,
) -> pd.DataFrame:
    """SKEW ⊥ TGD20 — skew after removing temporal-concentration overlap."""
    return cs_residualize(skew, [tgd])


def vol_max_residual_skew(
    skew: pd.DataFrame,
    ret_1d: pd.DataFrame,
    *,
    vol_window: int = 20,
    max_window: int = 20,
) -> pd.DataFrame:
    """SKEW ⊥ (VOL, MAX) — strongest purity candidate among daily variants."""
    vol = ret_1d.rolling(vol_window, min_periods=vol_window // 2).std()
    mx = max_return(ret_1d, max_window)
    return cs_residualize(skew, [vol, mx])


def build_skew_v2_panels(
    ret_1d: pd.DataFrame,
    *,
    idio_skew: Optional[pd.DataFrame] = None,
    total_skew: Optional[pd.DataFrame] = None,
    tgd: Optional[pd.DataFrame] = None,
    skew_window: int = 60,
    as_alpha: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Build named v2 panels. Keys are delivery names (Alpha* when as_alpha)."""
    if total_skew is None:
        from core.factors.skew.skew import total_return_skew

        total_skew = total_return_skew(ret_1d, skew_window)
    base = idio_skew if idio_skew is not None else total_skew

    raw: Dict[str, pd.DataFrame] = {
        "TailSKEW60": tail_skew(ret_1d, 60),
        "VolAdj_IdioSKEW60": vol_adjusted_skew(base, ret_1d),
        "VolResid_IdioSKEW60": vol_residual_skew(base, ret_1d),
        "MaxResid_IdioSKEW60": max_residual_skew(base, ret_1d),
        "VolMaxResid_IdioSKEW60": vol_max_residual_skew(base, ret_1d),
    }
    if tgd is not None:
        raw["TGDResid_IdioSKEW60"] = tgd_residual_skew(base, tgd)
        raw["VolMaxTGDResid_IdioSKEW60"] = cs_residualize(
            base,
            [
                ret_1d.rolling(20, min_periods=10).std(),
                max_return(ret_1d, 20),
                tgd,
            ],
        )

    if not as_alpha:
        return raw
    return {f"Alpha{k}": alpha_from_skew(v) for k, v in raw.items()}
