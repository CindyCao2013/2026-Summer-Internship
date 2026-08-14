"""Six ratio features added in AlphaNet-v2 / v3 (HTSC 2020-08-24)."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

RATIO_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("close_free_turn", "close", "free_turn"),
    ("open_turn", "open", "turn"),
    ("volume_low", "volume", "low"),
    ("vwap_high", "vwap", "high"),
    ("low_high", "low", "high"),
    ("vwap_close", "vwap", "close"),
)
RATIO_NAMES: Tuple[str, ...] = tuple(spec[0] for spec in RATIO_SPECS)


def safe_div(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    aligned = denom.reindex(index=numer.index, columns=numer.columns)
    out = numer.astype(float) / aligned.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def add_ratio_features(features: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out = dict(features)
    for name, num, den in RATIO_SPECS:
        if num not in features or den not in features:
            raise KeyError("ratio {} needs {} and {}".format(name, num, den))
        out[name] = safe_div(features[num], features[den])
    return out


def ensure_features(
    features: Mapping[str, pd.DataFrame],
    names: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    out = dict(features)
    needed = [n for n in names if n not in out]
    if needed:
        out = add_ratio_features(out)
    missing = [n for n in names if n not in out]
    if missing:
        raise KeyError("missing features: {}".format(missing))
    return out
