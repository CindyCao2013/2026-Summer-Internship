"""EOD alpha engine data foundation: raw fields, derived cache keys, primitive projections."""

from __future__ import annotations

from typing import Dict, List, TypedDict

import pandas as pd

from factor_formulas import FactorDataCache

# --- Layer 0: DDB raw fields (WIND.ASHAREEODPRICES) ---
EOD_RAW_FIELDS = [
    "S_DQ_OPEN",
    "S_DQ_HIGH",
    "S_DQ_LOW",
    "S_DQ_CLOSE",
    "S_DQ_VOLUME",
    "S_DQ_AMOUNT",
    "S_DQ_TURN",  # optional; falls back to amount/float_mktcap
]

# --- Layer 1: enriched (liquidity norm track only) ---
EOD_ENRICHED_FIELDS = [
    "float_mktcap",
    "total_mktcap",
]

# --- Layer 2: FactorDataCache derived keys ---
EOD_CACHE_DERIVED_KEYS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "volatility_20d",
    "volatility_60d",
    "volume_mean_5d",
    "volume_mean_20d",
    "volume_mean_60d",
    "amount_mean_5d",
    "amount_mean_20d",
    "amount_mean_60d",
    "amount_cv_20d",
    "daily_range",
    "high_low_mean_20d",
    "upper_shadow",
    "lower_shadow",
    "rsi_14",
]

# HF primitive projection families (all map to X = {O,H,L,C,V,A})
PRIMITIVE_PROJECTIONS: Dict[str, List[str]] = {
    "price_signal": [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "volatility_20d",
    ],
    "flow_signal": [
        "amount_mean_20d",
        "volume_mean_20d",
        "amount_cv_20d",
    ],
    "microstructure_proxy": [
        "daily_range",
        "upper_shadow",
        "lower_shadow",
        "high_low_mean_20d",
    ],
}

# Standard primitive basis used in completeness test (10 features)
PRIMITIVE_FEATURE_KEYS = [
    k for keys in PRIMITIVE_PROJECTIONS.values() for k in keys
]


class DataLayerMeta(TypedDict):
    layer: str
    fields: List[str]
    notes: str


EOD_DATA_STACK: List[DataLayerMeta] = [
    {
        "layer": "raw_ohlcv",
        "fields": ["open", "high", "low", "close", "volume", "amount"],
        "notes": "WIND.ASHAREEODPRICES daily panel",
    },
    {
        "layer": "derived_cache",
        "fields": EOD_CACHE_DERIVED_KEYS,
        "notes": "Lazy rolling transforms in FactorDataCache",
    },
    {
        "layer": "enriched_size",
        "fields": EOD_ENRICHED_FIELDS,
        "notes": "Optional float/total mktcap for liquidity normalization",
    },
    {
        "layer": "alpha_projection",
        "fields": ["alpha_k = f_k(X_t, ..., X_{t-n})"],
        "notes": "All EOD factors are nonlinear transforms of the same panel",
    },
]


def build_primitive_features(cache: FactorDataCache) -> Dict[str, pd.DataFrame]:
    """Build the 10-feature OHLCV primitive basis for completeness diagnostics."""
    out = {}
    for key in PRIMITIVE_FEATURE_KEYS:
        out[key] = cache.get(key)
    # Amihud is a composite primitive (price × flow)
    ret = cache.get("ret_1d")
    amount = cache.require("amount")
    out["amihud_daily"] = ret.abs() / amount.replace(0, pd.NA)
    return out


def foundation_summary_text() -> str:
    lines = [
        "EOD data foundation: single daily OHLCV + amount panel (optional mktcap).",
        "All alphas are projections f_k(X_{i,t}, ..., X_{i,t-n}) on the same manifold.",
        "",
        "Primitive families:",
    ]
    for family, keys in PRIMITIVE_PROJECTIONS.items():
        lines.append(f"  - {family}: {', '.join(keys)}")
    return "\n".join(lines)
