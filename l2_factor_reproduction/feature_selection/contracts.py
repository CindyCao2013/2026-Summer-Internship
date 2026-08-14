"""FS-1 dataset contracts — preprocessing ID, eligibility, forbidden columns."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from l2_factor_reproduction.config.settings import RESULT_ROOT

# ---------------------------------------------------------------------------
# Preprocessing contract (ML path only; Fast Discovery RAW is unchanged)
# ---------------------------------------------------------------------------

PREPROCESS_CONTRACT_ID = "HUATAI_STYLE_IND_CAP_Z_V1"
PREPROCESS_STEPS: Tuple[str, ...] = (
    "cross_sectional_mad_winsor",
    "missing_value_handling",
    "industry_ln_mktcap_neutralization",
    "cross_sectional_zscore",
)

# Median ± 5 MAD (tanh=False keeps economic scale; not the tanh squash path)
MAD_THRESHOLD = 5
MAD_TANH = False

# Reuse frozen FloatMktCap cache (same as FV IND_CAP diagnostics)
MCAP_PARQUET = (
    Path(RESULT_ROOT) / "primitives" / "mcap_wide_2019-01-01_2026-07-31.parquet"
)

# ---------------------------------------------------------------------------
# Missingness taxonomy (L2 upgrade vs Huatai industry-mean fill-all)
# ---------------------------------------------------------------------------

MISSINGNESS_REASONS: Tuple[str, ...] = (
    "STRUCTURAL",
    "WARMUP",
    "UNIVERSE",
    "DATA_ERROR",
    "ORDINARY",
)

# Only ORDINARY may be industry-mean imputed in v1
IMPUTABLE_REASONS: Tuple[str, ...] = ("ORDINARY",)

# ---------------------------------------------------------------------------
# Eligibility gates for FS panel membership
# ---------------------------------------------------------------------------

# Fraction of spine dates on which the factor has ≥1 non-null value
MIN_DATE_COVERAGE = 0.50
# Mean over covered dates of (n_symbols_with_value / n_spine_symbols_that_day)
MIN_MEAN_SYMBOL_COVERAGE = 0.20
# Absolute floor on observed rows after spine align (guards empty shards)
MIN_ALIGNED_ROWS = 10_000

# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------

FS1_OUT_ROOT = Path(RESULT_ROOT) / "feature_selection" / "fs1_feature_panel"

FORBIDDEN_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "ret_fwd_1d",
    "ret_fwd_5d",
    "ret_fwd_20d",
    "label",
    "target",
    "y",
    "y1",
    "y5",
    "y20",
    "fwd_ret",
    "forward_return",
    "excess_return",
)

# Columns that must never appear even as substrings in feature names (soft check)
FORBIDDEN_NAME_SUBSTRINGS: Tuple[str, ...] = (
    "ret_fwd",
    "fwd_ret",
    "label",
    "target",
)

FS1_VERDICTS: Tuple[str, ...] = (
    "A. FS1_PANEL_READY",
    "B. FS1_PANEL_READY_WITH_EXCLUSIONS",
    "C. FS1_PANEL_NOT_READY",
)

PARITY_FACTORS: Tuple[Tuple[str, str], ...] = (
    ("mid_order_ratio_4w_20w", "order_size"),
    ("net_buy_ratio", "trade_flow"),
    ("obi_l5_mean", "order_book"),
    ("overnight_gap", "price_formation"),
    ("cancel_value_pressure", "cancel_lifecycle"),
)

PANEL_SCHEMA: Dict[str, object] = {
    "sprint": "FS-1",
    "purpose": "Unified L2 ML feature panel (X only; no labels)",
    "preprocess_contract": PREPROCESS_CONTRACT_ID,
    "preprocess_steps": list(PREPROCESS_STEPS),
    "spine": "fast_context universe_mask == 1 → (TradeDate, Symbol)",
    "layers": ["aligned_raw", "processed_ind_cap_z_v1"],
    "forbidden_columns": list(FORBIDDEN_OUTPUT_COLUMNS),
    "fast_discovery_path": "unchanged RAW + T+1",
}
