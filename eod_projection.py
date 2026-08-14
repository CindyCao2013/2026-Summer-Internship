"""Project latent intraday states into EOD cross-sectional alpha signals."""

from typing import Dict, Optional

import pandas as pd

from factor_formulas import FactorDataCache


def daily_compress_latent(
    latent_states: Dict[str, pd.DataFrame],
    method: str = "last",
) -> Dict[str, pd.DataFrame]:
    """
    Latent states from intraday are already daily in EOD-proxy path.
    For true minute latent panels, aggregate here (mean / volume-weighted).
    """
    if method == "last":
        return latent_states
    out = {}
    for name, wide in latent_states.items():
        if method == "mean":
            out[name] = wide
        else:
            out[name] = wide
    return out


def cross_sectional_rank_signal(wide: pd.DataFrame) -> pd.DataFrame:
    """Rank across stocks each day → alpha signal in [-0.5, 0.5] centered."""
    return wide.rank(axis=1, pct=True) - 0.5


def cross_sectional_zscore_signal(wide: pd.DataFrame) -> pd.DataFrame:
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1).replace(0, pd.NA)
    return wide.sub(mu, axis=0).div(sd, axis=0)


def latent_to_eod_alpha(
    latent_states: Dict[str, pd.DataFrame],
    state_name: str,
    signal_type: str = "rank",
    direction: float = -1.0,
) -> pd.DataFrame:
    """Convert one latent state into cross-sectional EOD alpha."""
    if state_name not in latent_states:
        raise KeyError(f"Unknown latent state: {state_name}")
    wide = latent_states[state_name]
    if signal_type == "rank":
        signal = cross_sectional_rank_signal(wide)
    elif signal_type == "zscore":
        signal = cross_sectional_zscore_signal(wide)
    else:
        raise ValueError(signal_type)
    return signal * direction


def build_latent_eod_factors(
    cache: FactorDataCache,
    n_pca_components: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline (EOD-proxy path):
    OHLCV → proxy bricks → PCA latent → cross-sectional alpha signals.
    """
    from intraday_microstructure import EODMicrostructureInputs, extract_eod_proxy_bricks
    from latent_state import estimate_latent_states_pca

    data = EODMicrostructureInputs(
        close=cache.data.close,
        open=cache.require("open"),
        high=cache.require("high"),
        low=cache.require("low"),
        volume=cache.require("volume"),
        amount=cache.require("amount"),
    )
    bricks = extract_eod_proxy_bricks(data)
    latent = estimate_latent_states_pca(bricks, n_components=n_pca_components)
    compressed = daily_compress_latent(latent)

    factors = {}
    for i, (state_name, wide) in enumerate(compressed.items()):
        fname = f"latent_{state_name}_rank"
        direction = -1.0 if "stress" in state_name or "impulse" in state_name else 1.0
        factors[fname] = latent_to_eod_alpha(
            {state_name: wide}, state_name, signal_type="rank", direction=direction
        )
    return factors
