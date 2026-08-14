"""Frozen LR v1 contracts. Thresholds are pre-registered, not IC-tuned."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.python import liquidity_impact_daily as lid

FAMILY_NAME = "liquidity_resilience"
SCHEMA_VERSION = "liquidity_resilience_lr1_v1"

LR_RESULT_ROOT = Path(RESULT_ROOT) / "liquidity_resilience"
LR0_DIR = LR_RESULT_ROOT / "lr0_feasibility"
LR1_MAT_DIR = LR_RESULT_ROOT / "lr1_lite_materialization"
LR1_FACTOR_DIR = LR1_MAT_DIR / "factors"
BDL_LINK_DIR = LR_RESULT_ROOT / "bdl"

# Reuse liquidity-impact L5 depth and continuous-auction grid. Do not invent L10.
DEPTH_LEVELS = 5
EXPECTED_CONTINUOUS_MINUTES = lid.EXPECTED_CONTINUOUS_MINUTES  # 240
CANONICAL_SOURCE = lid.CANONICAL_SOURCE

# Causal shock rules (option 2/3). Not same-day percentile. Not IC-searched.
TRAILING_WINDOW = 20
TRAILING_MIN_OBS = 10
FLOW_SHOCK_MULT = 3.0
DEPTH_DEPLETION_FRAC = 0.20
SPREAD_WIDEN_FRAC = 0.50

DENOM_FLOOR_DEPTH = 1.0
DENOM_FLOOR_SPREAD = 1e-4
DENOM_FLOOR_OBI = 0.05
DENOM_FLOOR_FLOW = 1.0

HORIZONS = (1, 3, 5)

SHOCK_ACTIVE_BUY = "ACTIVE_BUY_SHOCK"
SHOCK_ACTIVE_SELL = "ACTIVE_SELL_SHOCK"
SHOCK_DEPTH = "DEPTH_DEPLETION_SHOCK"
SHOCK_SPREAD = "SPREAD_WIDENING_SHOCK"

PIT_SEMANTICS = (
    "Daily post-market L2 factor for TradeDate T. Shock detection and recovery "
    "paths use only continuous-auction minutes on T with information <= event "
    "minute t. Horizon t+h must be a valid same-session bar; ineligible otherwise. "
    "No T+1 book, no next-day recovery, no same-day future quantile."
)
SESSION_BOUNDARY_RULE = (
    "Reuse liquidity_impact_daily continuous auction: 09:30-11:29 and 13:00-14:59 "
    "(mkey 570-689 AM, 780-899 PM). Recovery indexing is +h valid session bars "
    "(integer mkey + h in the same session), never timestamp+timedelta. "
    "Windows must not cross opening auction, lunch, closing auction, or session end."
)

# BDL registry extras. Event-sparse by design: no-event is NA, not 0.
BDL_SPARSE_EVENT = True
BDL_LOOKBACK_DAYS = 1
BDL_PIT_STATUS = "PASS"


def _cand(**kwargs) -> Dict[str, object]:
    row = {
        "family": FAMILY_NAME,
        "lookback_days": BDL_LOOKBACK_DAYS,
        "signed": True,
        "sparse_event": BDL_SPARSE_EVENT,
        "pit_status": BDL_PIT_STATUS,
        "pit_semantics": PIT_SEMANTICS,
        "session_boundary_rule": SESSION_BOUNDARY_RULE,
        "registry_status": "FROZEN_LR1_CANDIDATE",
        "normalization": "none",
    }
    row.update(kwargs)
    return row


def frozen_candidate_specs() -> List[Dict[str, object]]:
    """24 economically distinct LR v1 formulas. Frozen before any RankIC."""
    specs: List[Dict[str, object]] = []

    # Block A — ask replenishment after aggressive buy shock (6)
    for h in HORIZONS:
        specs.append(
            _cand(
                name=f"ask_depth_recovery_{h}m",
                subfamily="ask_replenishment",
                formula=(
                    f"median over ACTIVE_BUY_SHOCK events of "
                    f"(ask5_{{t+{h}}}-ask5_t0)/(ask5_pre-ask5_t0); "
                    f"exclude if (ask5_pre-ask5_t0)<{DENOM_FLOOR_DEPTH}"
                ),
                mechanism=(
                    "After aggressive buy flow consumes ask liquidity, measure how "
                    f"completely L5 ask depth returns toward the causal pre-event level in {h} valid minutes."
                ),
                shock_type=SHOCK_ACTIVE_BUY,
                shock_side="buy",
                recovery_variable="ask_depth_5",
                recovery_horizon_min=h,
                aggregation="event_median",
                primitive_dependencies="ask_depth_5,active_buy_amount,active_sell_amount",
                positive_value_meaning=(
                    f"stronger ask-side replenishment within {h}m after aggressive buy shock"
                ),
                expected_redundancy="ask_depth_recovery_1m/3m/5m likely high Spearman",
            )
        )
    for h in HORIZONS:
        specs.append(
            _cand(
                name=f"ask_replenishment_efficiency_{h}m",
                subfamily="ask_replenishment",
                formula=(
                    f"shock-size-weighted mean over ACTIVE_BUY_SHOCK of "
                    f"(ask5_{{t+{h}}}-ask5_t0)/active_buy_amount_t0; "
                    f"weight=active_buy_amount_t0; exclude if flow<{DENOM_FLOOR_FLOW}"
                ),
                mechanism=(
                    "Ask depth recovered per CNY of aggressive buy flow "
                    f"over {h} valid minutes (mixed units: depth CNY / flow CNY)."
                ),
                shock_type=SHOCK_ACTIVE_BUY,
                shock_side="buy",
                recovery_variable="ask_depth_5",
                recovery_horizon_min=h,
                aggregation="shock_size_weighted_mean",
                normalization="flow_cnY",
                primitive_dependencies="ask_depth_5,active_buy_amount,active_sell_amount",
                positive_value_meaning=(
                    f"more ask depth replenished per unit buy-shock size within {h}m"
                ),
                expected_redundancy="efficiency vs recovery_fraction related but not a constant multiple",
            )
        )

    # Block B — bid replenishment after aggressive sell shock (6)
    for h in HORIZONS:
        specs.append(
            _cand(
                name=f"bid_depth_recovery_{h}m",
                subfamily="bid_replenishment",
                formula=(
                    f"median over ACTIVE_SELL_SHOCK events of "
                    f"(bid5_{{t+{h}}}-bid5_t0)/(bid5_pre-bid5_t0); "
                    f"exclude if (bid5_pre-bid5_t0)<{DENOM_FLOOR_DEPTH}"
                ),
                mechanism=(
                    "After aggressive sell flow consumes bid liquidity, measure how "
                    f"completely L5 bid depth returns toward the causal pre-event level in {h} valid minutes."
                ),
                shock_type=SHOCK_ACTIVE_SELL,
                shock_side="sell",
                recovery_variable="bid_depth_5",
                recovery_horizon_min=h,
                aggregation="event_median",
                primitive_dependencies="bid_depth_5,active_buy_amount,active_sell_amount",
                positive_value_meaning=(
                    f"stronger bid-side replenishment within {h}m after aggressive sell shock"
                ),
                expected_redundancy="bid_depth_recovery_1m/3m/5m likely high Spearman",
            )
        )
    for h in HORIZONS:
        specs.append(
            _cand(
                name=f"bid_replenishment_efficiency_{h}m",
                subfamily="bid_replenishment",
                formula=(
                    f"shock-size-weighted mean over ACTIVE_SELL_SHOCK of "
                    f"(bid5_{{t+{h}}}-bid5_t0)/active_sell_amount_t0; "
                    f"weight=active_sell_amount_t0; exclude if flow<{DENOM_FLOOR_FLOW}"
                ),
                mechanism=(
                    "Bid depth recovered per CNY of aggressive sell flow "
                    f"over {h} valid minutes (mixed units: depth CNY / flow CNY)."
                ),
                shock_type=SHOCK_ACTIVE_SELL,
                shock_side="sell",
                recovery_variable="bid_depth_5",
                recovery_horizon_min=h,
                aggregation="shock_size_weighted_mean",
                normalization="flow_cnY",
                primitive_dependencies="bid_depth_5,active_buy_amount,active_sell_amount",
                positive_value_meaning=(
                    f"more bid depth replenished per unit sell-shock size within {h}m"
                ),
                expected_redundancy="efficiency vs recovery_fraction related but not a constant multiple",
            )
        )

    # Block C — spread resilience (4). No speed=fraction/h aliases.
    for h in HORIZONS:
        specs.append(
            _cand(
                name=f"spread_recovery_{h}m",
                subfamily="spread_resilience",
                formula=(
                    f"median over SPREAD_WIDENING_SHOCK of "
                    f"(spread_t0-spread_{{t+{h}}})/(spread_t0-spread_pre); "
                    f"exclude if (spread_t0-spread_pre)<{DENOM_FLOOR_SPREAD}"
                ),
                mechanism=(
                    f"How completely relative spread returns toward the pre-shock level in {h} valid minutes "
                    "after a causal spread-widening shock."
                ),
                shock_type=SHOCK_SPREAD,
                shock_side="both",
                recovery_variable="spread",
                recovery_horizon_min=h,
                aggregation="event_median",
                primitive_dependencies="bid1,ask1,spread",
                positive_value_meaning=(
                    f"stronger spread restoration toward pre-shock tightness within {h}m"
                ),
                expected_redundancy="spread_recovery_1m/3m/5m likely high Spearman",
            )
        )
    specs.append(
        _cand(
            name="spread_residual_width_5m",
            subfamily="spread_resilience",
            formula=(
                "median over SPREAD_WIDENING_SHOCK of "
                "(spread_{t+5}-spread_pre)/spread_pre; "
                f"exclude if spread_pre<{DENOM_FLOOR_SPREAD}"
            ),
            mechanism=(
                "Remaining extra relative-spread width vs the causal pre-event baseline "
                "5 valid minutes after a spread-widening shock (level residual, not completeness ratio)."
            ),
            shock_type=SHOCK_SPREAD,
            shock_side="both",
            recovery_variable="spread",
            recovery_horizon_min=5,
            aggregation="event_median",
            primitive_dependencies="bid1,ask1,spread",
            positive_value_meaning=(
                "larger remaining extra-width vs pre after 5m (weaker tightness restoration)"
            ),
            expected_redundancy="related to spread_recovery_5m but different normalization",
        )
    )

    # Block D — OBI restoration / persistence (5)
    for side, shock, h in (
        ("buy", SHOCK_ACTIVE_BUY, 3),
        ("buy", SHOCK_ACTIVE_BUY, 5),
        ("sell", SHOCK_ACTIVE_SELL, 3),
        ("sell", SHOCK_ACTIVE_SELL, 5),
    ):
        specs.append(
            _cand(
                name=f"obi_restoration_{side}_{h}m",
                subfamily="imbalance_restoration",
                formula=(
                    f"median over {shock} of "
                    f"1-abs(obi_{{t+{h}}}-obi_pre)/abs(obi_t0-obi_pre); "
                    f"exclude if abs(obi_t0-obi_pre)<{DENOM_FLOOR_OBI}"
                ),
                mechanism=(
                    f"How completely L5 OBI returns toward its causal pre-shock state "
                    f"within {h} valid minutes after an aggressive {side} shock."
                ),
                shock_type=shock,
                shock_side=side,
                recovery_variable="obi_5",
                recovery_horizon_min=h,
                aggregation="event_median",
                primitive_dependencies="bid_depth_5,ask_depth_5,active_buy_amount,active_sell_amount",
                positive_value_meaning=(
                    f"stronger OBI restoration toward pre-shock equilibrium within {h}m after {side} shock"
                ),
                expected_redundancy="3m vs 5m restoration likely correlated; buy vs sell distinct sides",
            )
        )
    specs.append(
        _cand(
            name="obi_shock_persistence_5m",
            subfamily="imbalance_restoration",
            formula=(
                "median over DEPTH_DEPLETION_SHOCK of "
                "abs(obi_{t+5}-obi_pre)/abs(obi_t0-obi_pre); "
                f"exclude if abs(obi_t0-obi_pre)<{DENOM_FLOOR_OBI}"
            ),
            mechanism=(
                "Persistence of L5 OBI displacement 5 valid minutes after a depth-depletion shock. "
                "Higher means the imbalance has not returned toward the pre-shock state."
            ),
            shock_type=SHOCK_DEPTH,
            shock_side="both",
            recovery_variable="obi_5",
            recovery_horizon_min=5,
            aggregation="event_median",
            primitive_dependencies="bid_depth_5,ask_depth_5",
            positive_value_meaning=(
                "greater OBI displacement persistence 5m after depth depletion (imbalance remains)"
            ),
            expected_redundancy="near complement of an OBI restoration metric on a different shock type",
        )
    )

    # Block E — buy/sell resilience asymmetry (3)
    for h in (3, 5):
        specs.append(
            _cand(
                name=f"depth_resilience_asymmetry_{h}m",
                subfamily="resilience_asymmetry",
                formula=(
                    f"median(bid_depth_recovery after ACTIVE_SELL_SHOCK, {h}m) "
                    f"- median(ask_depth_recovery after ACTIVE_BUY_SHOCK, {h}m); "
                    "NA unless both sides have >=1 eligible event"
                ),
                mechanism=(
                    f"Signed difference between bid replenishment after sell shocks and "
                    f"ask replenishment after buy shocks at {h}m. Positive: bid side recovers more."
                ),
                shock_type="ACTIVE_SELL_SHOCK-ACTIVE_BUY_SHOCK",
                shock_side="asymmetric",
                recovery_variable="bid_depth_5-ask_depth_5",
                recovery_horizon_min=h,
                aggregation="difference_of_event_medians",
                primitive_dependencies="bid_depth_5,ask_depth_5,active_buy_amount,active_sell_amount",
                positive_value_meaning=(
                    f"bid replenishment after sell shocks exceeds ask replenishment after buy shocks at {h}m"
                ),
                expected_redundancy="3m vs 5m asymmetry likely correlated",
            )
        )
    specs.append(
        _cand(
            name="spread_resilience_asymmetry_5m",
            subfamily="resilience_asymmetry",
            formula=(
                "median(spread_recovery_fraction_5m | ACTIVE_SELL_SHOCK) "
                "- median(spread_recovery_fraction_5m | ACTIVE_BUY_SHOCK); "
                "NA unless both sides have >=1 eligible event"
            ),
            mechanism=(
                "Asymmetry of spread restoration after aggressive sell vs buy flow shocks. "
                "Positive: spread recovers more after sell shocks than after buy shocks."
            ),
            shock_type="ACTIVE_SELL_SHOCK-ACTIVE_BUY_SHOCK",
            shock_side="asymmetric",
            recovery_variable="spread",
            recovery_horizon_min=5,
            aggregation="difference_of_event_medians",
            primitive_dependencies="bid1,ask1,spread,active_buy_amount,active_sell_amount",
            positive_value_meaning=(
                "spread restores more after sell shocks than after buy shocks at 5m"
            ),
            expected_redundancy="distinct from depth_resilience_asymmetry_5m",
        )
    )

    names = [str(s["name"]) for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate LR candidate names")
    if not (20 <= len(specs) <= 30):
        raise ValueError(f"LR v1 candidate count {len(specs)} outside 20-30")
    return specs


FROZEN_CANDIDATE_SPECS: Tuple[Dict[str, object], ...] = tuple(frozen_candidate_specs())
FROZEN_CANDIDATE_NAMES: Tuple[str, ...] = tuple(str(s["name"]) for s in FROZEN_CANDIDATE_SPECS)

REGISTRY_COLUMNS = (
    "name",
    "family",
    "subfamily",
    "formula",
    "mechanism",
    "shock_type",
    "shock_side",
    "recovery_variable",
    "recovery_horizon_min",
    "aggregation",
    "normalization",
    "primitive_dependencies",
    "signed",
    "positive_value_meaning",
    "pit_semantics",
    "session_boundary_rule",
    "expected_redundancy",
    "registry_status",
    "lookback_days",
    "sparse_event",
    "pit_status",
)
