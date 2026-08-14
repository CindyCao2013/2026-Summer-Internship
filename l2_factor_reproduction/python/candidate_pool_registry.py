"""Central L2 Candidate Pool v1 family registry and frozen summary schema.

All pool-level scripts (index build, artifact audits, reports) must read the
family list from here instead of hardcoding family names or formula counts.
Adding a new family means appending one ``FamilyConfig`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from l2_factor_reproduction.config.settings import RESULT_ROOT


POOL_ROOT = Path(RESULT_ROOT) / "candidate_pool_v1"

CANDIDATE_SUMMARY_SCHEMA_V1: Tuple[str, ...] = (
    "factor",
    "family",
    "category",
    "mechanism",
    "rank_ic_raw",
    "icir_raw",
    "rank_ic_std",
    "positive_ic_fraction",
    "g10_excess_annu_ret",
    "g10_excess_sharpe",
    "hl_annu_ret",
    "hl_sharpe",
    "hl_mdd",
    "avg_hl_turnover",
    "implied_annu_fee",
    "net_annu_after_fee",
    "sign_consistency",
    "decile_mono_spearman",
    "factor_direction",
    "redundancy_cluster_080",
    "n_factor_rows",
    "date_min",
    "date_max",
    "n_symbols",
)

BASELINE_POLICY: Dict[str, object] = {
    "benchmark": "000852.SH",
    "cost_bps": 7.5,
    "signal_shift": 1,
    "sample_start": "2019-01-01",
    "sample_end": "2026-07-31",
    "rank_ic_direction": "raw frozen formula",
    "group_direction": "effective display only",
    "production_direction": "not decided",
}


@dataclass(frozen=True)
class FamilyConfig:
    """One candidate-pool family (or bridge)."""

    name: str
    title: str
    directory: Path
    registry_csv: str = "factor_registry.csv"
    summary_csv: str = "candidate_summary.csv"
    manifest_json: str = "manifest.json"
    primitive_name: Optional[str] = None
    has_categories: bool = True
    is_bridge: bool = False
    # Per-factor result directories outside the family directory
    # (legacy families store results under RESULT_ROOT/<factor>/).
    external_factor_dirs: bool = False
    cross_reference_files: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def primitive_dir(self) -> Optional[Path]:
        if self.primitive_name is None:
            return None
        return Path(RESULT_ROOT) / "primitives" / self.primitive_name

    def factor_result_dir(self, factor: str) -> Path:
        if self.external_factor_dirs:
            return Path(RESULT_ROOT) / factor
        return self.directory / "factors" / factor


FAMILY_REGISTRY: Dict[str, FamilyConfig] = {
    "trade_flow": FamilyConfig(
        name="trade_flow",
        title="Trade Flow",
        directory=POOL_ROOT / "trade_flow_family",
        primitive_name="trade_flow_daily",
        has_categories=False,
        external_factor_dirs=True,
    ),
    "order_size": FamilyConfig(
        name="order_size",
        title="Order Size",
        directory=POOL_ROOT / "order_size_family",
        primitive_name="order_size_distribution_daily",
        has_categories=False,
        external_factor_dirs=True,
    ),
    "order_book": FamilyConfig(
        name="order_book",
        title="Order Book",
        directory=POOL_ROOT / "order_book_family",
        primitive_name="order_book_daily",
        cross_reference_files=(
            "order_book_vs_trade_flow_corr.csv",
            "order_book_vs_order_size_corr.csv",
        ),
    ),
    "price_formation": FamilyConfig(
        name="price_formation",
        title="Price Formation",
        directory=POOL_ROOT / "price_formation_family",
        primitive_name="price_formation_daily",
        cross_reference_files=(
            "price_formation_vs_trade_flow_corr.csv",
            "price_formation_vs_order_size_corr.csv",
            "price_formation_vs_order_book_corr.csv",
        ),
    ),
    "liquidity_impact": FamilyConfig(
        name="liquidity_impact",
        title="Liquidity / Price Impact",
        directory=POOL_ROOT / "liquidity_impact_family",
        primitive_name="liquidity_impact_daily",
        cross_reference_files=(
            "liquidity_impact_vs_trade_flow_corr.csv",
            "liquidity_impact_vs_order_size_corr.csv",
            "liquidity_impact_vs_order_book_corr.csv",
            "liquidity_impact_vs_price_formation_corr.csv",
        ),
    ),
    "ddb_reference_snapshot": FamilyConfig(
        name="ddb_reference_snapshot",
        title="DolphinDB Reference Snapshot (LOB Dynamics)",
        directory=POOL_ROOT / "ddb_reference_snapshot_family",
        primitive_name="ddb_reference_snapshot",
        cross_reference_files=(
            "cross_family_correlation.csv",
            "factor_correlation_spearman.csv",
        ),
    ),
    "cancel_lifecycle": FamilyConfig(
        name="cancel_lifecycle",
        title="Cancellation / Order Lifecycle",
        directory=POOL_ROOT / "cancel_lifecycle_family",
        primitive_name="cancel_lifecycle_daily",
        cross_reference_files=(
            "cross_family_correlation.csv",
            "factor_correlation_spearman.csv",
        ),
    ),
}

BRIDGE_CONFIG = FamilyConfig(
    name="trade_flow_mcap_bridge",
    title="Sprint-2 mcap bridge",
    directory=POOL_ROOT,
    primitive_name=None,
    has_categories=False,
    is_bridge=True,
    external_factor_dirs=True,
)
BRIDGE_FACTOR = "net_buy_amount_mcap"


def active_families() -> List[FamilyConfig]:
    """Families whose family directory exists (built and frozen so far)."""
    return [
        config
        for config in FAMILY_REGISTRY.values()
        if not config.is_bridge and config.directory.is_dir()
    ]


def get_family(name: str) -> FamilyConfig:
    return FAMILY_REGISTRY[name]


def expected_formula_count() -> int:
    """Total frozen formulas expected from per-family registries + bridge."""
    total = 0
    import pandas as pd

    for config in active_families():
        total += int(
            len(pd.read_csv(config.directory / config.registry_csv))
        )
    bridge_summary = Path(RESULT_ROOT) / BRIDGE_FACTOR / "summary.json"
    if bridge_summary.exists():
        total += 1
    return total


MISSING_REASON_NO_CATEGORY = (
    "legacy family registry frozen before category taxonomy existed"
)
