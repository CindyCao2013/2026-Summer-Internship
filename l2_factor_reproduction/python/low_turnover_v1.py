"""Sprint 8 — Low-Turnover L2 Discovery v1.

Strict boundary:
- Read only existing daily primitives (no Raw Tick / SSL2 / new extraction).
- Frozen discovery window 2023-01-01 .. 2024-12-31; each formula once.
- Reuses Fast Discovery Lane backtest / metrics / plots; does not alter
  the frozen lane for performance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Factor_Dev_Lib import implied_annu_fee
from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.python.backtest import backtest_factor
from l2_factor_reproduction.python.ch_cancel_lifecycle import (
    build_candidates as build_cancel_candidates,
)
from l2_factor_reproduction.python.fast_discovery import (
    DISCOVERY_END,
    DISCOVERY_START,
    FAST_DISCOVERY_DIR,
    PRIMITIVE_BUFFER_DAYS,
    WINDOWS,
    _drop_nested_partitions,
    _file_window_overlap,
    compute_fast_metrics,
    load_fast_context,
    save_fast_plots,
)
from l2_factor_reproduction.python.order_size_factors import (
    build_order_size_feature_frame,
)

EPS = 1e-12
ROLLING_DAYS = 5
OUT_DIR = Path(RESULT_ROOT) / "fast_discovery" / "low_turnover_v1"
_PRIMITIVES = Path(RESULT_ROOT) / "primitives"

# Best non-obvious-alias order-size level representative (family report):
# small_order_ratio_1w — top IC/Sharpe/mono among R1 level signals.
ORDER_SIZE_REPRESENTATIVE = "small_order_ratio_1w"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mechanism: str
    required_columns: Tuple[str, ...]
    source_primitive: str
    builder: Optional[Callable[..., pd.Series]] = None
    available: bool = True
    reason_if_unavailable: str = ""


def _cs_rank(series: pd.Series, dates: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0, 1] within each TradeDate."""
    return series.groupby(dates, sort=False).rank(pct=True, method="average")


def _rolling_mean_nd(
    values: pd.Series,
    symbols: pd.Series,
    window: int = ROLLING_DAYS,
) -> pd.Series:
    grouped = values.groupby(symbols, sort=False)
    return grouped.transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )


def _to_narrow(symbol: pd.Series, dates: pd.Series, values: pd.Series, name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "symbol": symbol.astype(str).to_numpy(),
            "tradetime": (
                pd.to_datetime(dates) + pd.Timedelta(hours=9, minutes=30)
            ),
            "factorname": name,
            "value": pd.to_numeric(values, errors="coerce").to_numpy(
                dtype=float
            ),
        }
    )
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _partition_files(primitive_dir: Path) -> List[Path]:
    files = sorted(primitive_dir.glob("**/*.parquet"))
    files = [
        path
        for path in files
        if "validation" not in path.name and "smoke" not in path.name
    ]
    return _drop_nested_partitions(files)


def load_primitive_panel(
    primitive_dir: Path,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    buffer_days: int = PRIMITIVE_BUFFER_DAYS,
) -> pd.DataFrame:
    """Load selected columns from a daily primitive directory."""
    files = [
        path
        for path in _partition_files(primitive_dir)
        if _file_window_overlap(path, start, end, buffer_days)
    ]
    if not files:
        raise FileNotFoundError(
            f"no partitions under {primitive_dir} for "
            f"[{start.date()}, {end.date()}]"
        )
    # Always keep keys; some files may lack optional columns.
    key_cols = {"symbol", "TradeDate"}
    wanted = list(dict.fromkeys([*key_cols, *columns]))
    frames: List[pd.DataFrame] = []
    for path in files:
        frame = pd.read_parquet(path)
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        frames.append(frame.loc[:, wanted])
    panel = pd.concat(frames, ignore_index=True)
    panel["symbol"] = panel["symbol"].astype(str)
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    lo = start - pd.Timedelta(days=buffer_days)
    panel = panel.loc[panel["TradeDate"].between(lo, end)].copy()
    panel = panel.sort_values(["symbol", "TradeDate"], kind="stable")
    panel = panel.drop_duplicates(["symbol", "TradeDate"], keep="last")
    for column in columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    return panel.reset_index(drop=True)


def _cancel_discovery_coverage() -> Tuple[bool, str]:
    """Cancel is formal only if discovery-window coverage is complete enough."""
    manifest = (
        _PRIMITIVES
        / "cancel_lifecycle_daily"
        / "manifest_worker_20230101_20241231.json"
    )
    if not manifest.exists():
        return False, "cancel_lifecycle_daily discovery-window manifest missing"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    coverage = payload.get("date_coverage", {})
    actual_max = pd.Timestamp(coverage.get("actual_max", "NaT"))
    # Frozen discovery end is 2024-12-31; incomplete H2 2024 blocks fair run.
    if pd.isna(actual_max) or actual_max < pd.Timestamp("2024-12-01"):
        return (
            False,
            "cancel_lifecycle_daily incomplete for frozen discovery window "
            f"(actual_max={coverage.get('actual_max')}; missing 2024H2); "
            "no Raw re-extraction in this Sprint",
        )
    dataset = _PRIMITIVES / "cancel_lifecycle_daily" / "dataset"
    if not dataset.exists():
        return False, "cancel_lifecycle_daily dataset directory missing"
    return True, ""


def _build_stable_obi(order_book: pd.DataFrame) -> pd.Series:
    daily = order_book["obi_5_mean"] / (order_book["obi_5_std"] + EPS)
    return _rolling_mean_nd(daily, order_book["symbol"])


def _build_persistent_obi(order_book: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(order_book["obi_5_mean"], order_book["symbol"])


def _build_book_depth_quality(order_book: pd.DataFrame) -> pd.Series:
    # Primitive stores log-depth mean/std (canonical total_depth state).
    quality = order_book["log_total_depth_mean"] / (
        order_book["log_total_depth_std"] + EPS
    )
    return _rolling_mean_nd(quality, order_book["symbol"])


def _build_microprice_pressure(order_book: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(
        order_book["microprice_deviation_mean"], order_book["symbol"]
    )


def _merge_flow_price(
    trade_flow: pd.DataFrame, price_formation: pd.DataFrame
) -> pd.DataFrame:
    flow = trade_flow.rename(
        columns={
            "active_buy_amt": "active_buy_amt",
            "active_sell_amt": "active_sell_amt",
            "total_amt": "total_amt",
        }
    )
    price = price_formation.loc[
        :, ["symbol", "TradeDate", "open_to_close_return"]
    ]
    merged = flow.merge(price, on=["symbol", "TradeDate"], how="inner")
    merged = merged.sort_values(["symbol", "TradeDate"], kind="stable")
    total = merged["total_amt"].where(merged["total_amt"].abs() > EPS)
    merged["net_active_flow_ratio"] = (
        merged["active_buy_amt"] - merged["active_sell_amt"]
    ) / total
    merged["active_buy_intensity"] = merged["active_buy_amt"] / total
    merged["active_sell_intensity"] = merged["active_sell_amt"] / total
    return merged.reset_index(drop=True)


def _build_flow_price_divergence(panel: pd.DataFrame) -> pd.Series:
    dates = panel["TradeDate"]
    daily = _cs_rank(panel["open_to_close_return"], dates) - _cs_rank(
        panel["net_active_flow_ratio"], dates
    )
    return _rolling_mean_nd(daily, panel["symbol"])


def _build_sell_absorption(panel: pd.DataFrame) -> pd.Series:
    dates = panel["TradeDate"]
    daily = _cs_rank(panel["open_to_close_return"], dates) + _cs_rank(
        panel["active_sell_intensity"], dates
    )
    return _rolling_mean_nd(daily, panel["symbol"])


def _build_buy_absorption(panel: pd.DataFrame) -> pd.Series:
    dates = panel["TradeDate"]
    # Symmetric: strong buy flow but relatively weak price response.
    daily = _cs_rank(-panel["open_to_close_return"], dates) + _cs_rank(
        panel["active_buy_intensity"], dates
    )
    return _rolling_mean_nd(daily, panel["symbol"])


def _build_flow_price_efficiency(panel: pd.DataFrame) -> pd.Series:
    """Fixed parameter-free rank interaction (centered percentile ranks)."""
    dates = panel["TradeDate"]
    r_ret = _cs_rank(panel["open_to_close_return"], dates) - 0.5
    r_flow = _cs_rank(panel["net_active_flow_ratio"], dates) - 0.5
    daily = r_ret * r_flow
    return _rolling_mean_nd(daily, panel["symbol"])


def _build_effective_spread_persistence(liq: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(liq["effective_spread_proxy"], liq["symbol"])


def _build_spread_quality(order_book: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(
        order_book["relative_spread_mean"], order_book["symbol"]
    )


def _build_price_impact_efficiency(liq: pd.DataFrame) -> pd.Series:
    # Existing daily signed flow×price impact; no new minute access.
    return _rolling_mean_nd(liq["signed_amount_impact"], liq["symbol"])


def _build_liquidity_resilience(liq: pd.DataFrame) -> pd.Series:
    # depth_recovery_5m already aggregated into daily primitive.
    return _rolling_mean_nd(liq["depth_recovery_5m"], liq["symbol"])


def _build_order_size_persistence(order_size_features: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(
        order_size_features[ORDER_SIZE_REPRESENTATIVE],
        order_size_features["symbol"],
    )


def _build_persistent_cancel_pressure(cancel: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(cancel["cancel_value_pressure"], cancel["symbol"])


def _build_persistent_cancel_intensity(cancel: pd.DataFrame) -> pd.Series:
    return _rolling_mean_nd(cancel["cancel_value_intensity"], cancel["symbol"])


def _build_cancel_pressure_consistency(cancel: pd.DataFrame) -> pd.Series:
    """Past-5d same-sign fraction of cancel pressure × |mean pressure|."""
    pressure = cancel["cancel_value_pressure"]
    symbol = cancel["symbol"]

    def _same_sign_frac(series: pd.Series) -> pd.Series:
        def _window(x: np.ndarray) -> float:
            last = x[-1]
            if not np.isfinite(last):
                return np.nan
            return float(np.mean(np.sign(x) == np.sign(last)))

        return series.rolling(
            ROLLING_DAYS, min_periods=ROLLING_DAYS
        ).apply(_window, raw=True)

    same = pressure.groupby(symbol, sort=False).transform(_same_sign_frac)
    mean_abs = pressure.abs().groupby(symbol, sort=False).transform(
        lambda s: s.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean()
    )
    return same * mean_abs


def candidate_registry() -> List[CandidateSpec]:
    cancel_ok, cancel_reason = _cancel_discovery_coverage()
    specs: List[CandidateSpec] = [
        CandidateSpec(
            "stable_obi_5d",
            "persistent_book_state",
            ("obi_5_mean", "obi_5_std"),
            "order_book_daily",
            _build_stable_obi,
        ),
        CandidateSpec(
            "persistent_obi_5d",
            "persistent_book_state",
            ("obi_5_mean",),
            "order_book_daily",
            _build_persistent_obi,
        ),
        CandidateSpec(
            "book_depth_quality_5d",
            "persistent_book_state",
            ("log_total_depth_mean", "log_total_depth_std"),
            "order_book_daily",
            _build_book_depth_quality,
        ),
        CandidateSpec(
            "microprice_pressure_5d",
            "persistent_book_state",
            ("microprice_deviation_mean",),
            "order_book_daily",
            _build_microprice_pressure,
        ),
        CandidateSpec(
            "flow_price_divergence_5d",
            "flow_price_response",
            (
                "active_buy_amt",
                "active_sell_amt",
                "total_amt",
                "open_to_close_return",
            ),
            "trade_flow_daily+price_formation_daily",
            _build_flow_price_divergence,
        ),
        CandidateSpec(
            "sell_absorption_5d",
            "flow_price_response",
            (
                "active_sell_amt",
                "total_amt",
                "open_to_close_return",
            ),
            "trade_flow_daily+price_formation_daily",
            _build_sell_absorption,
        ),
        CandidateSpec(
            "buy_absorption_5d",
            "flow_price_response",
            (
                "active_buy_amt",
                "total_amt",
                "open_to_close_return",
            ),
            "trade_flow_daily+price_formation_daily",
            _build_buy_absorption,
        ),
        CandidateSpec(
            "flow_price_efficiency_5d",
            "flow_price_response",
            (
                "active_buy_amt",
                "active_sell_amt",
                "total_amt",
                "open_to_close_return",
            ),
            "trade_flow_daily+price_formation_daily",
            _build_flow_price_efficiency,
        ),
        CandidateSpec(
            "effective_spread_persistence_5d",
            "liquidity_cost_persistence",
            ("effective_spread_proxy",),
            "liquidity_impact_daily",
            _build_effective_spread_persistence,
        ),
        CandidateSpec(
            "spread_quality_5d",
            "liquidity_cost_persistence",
            ("relative_spread_mean",),
            "order_book_daily",
            _build_spread_quality,
        ),
        CandidateSpec(
            "price_impact_efficiency_5d",
            "liquidity_cost_persistence",
            ("signed_amount_impact",),
            "liquidity_impact_daily",
            _build_price_impact_efficiency,
        ),
        CandidateSpec(
            "liquidity_resilience_proxy_5d",
            "liquidity_cost_persistence",
            ("depth_recovery_5m",),
            "liquidity_impact_daily",
            _build_liquidity_resilience,
        ),
        CandidateSpec(
            "persistent_cancel_pressure_5d",
            "cancellation_persistence",
            ("buy_cancel_value", "sell_cancel_value"),
            "cancel_lifecycle_daily",
            _build_persistent_cancel_pressure if cancel_ok else None,
            available=cancel_ok,
            reason_if_unavailable=cancel_reason,
        ),
        CandidateSpec(
            "persistent_cancel_intensity_5d",
            "cancellation_persistence",
            (
                "buy_cancel_value",
                "sell_cancel_value",
                "total_trade_value",
            ),
            "cancel_lifecycle_daily",
            _build_persistent_cancel_intensity if cancel_ok else None,
            available=cancel_ok,
            reason_if_unavailable=cancel_reason,
        ),
        CandidateSpec(
            "cancel_pressure_consistency_5d",
            "cancellation_persistence",
            ("buy_cancel_value", "sell_cancel_value"),
            "cancel_lifecycle_daily",
            _build_cancel_pressure_consistency if cancel_ok else None,
            available=cancel_ok,
            reason_if_unavailable=cancel_reason,
        ),
        CandidateSpec(
            "order_size_structure_persistence_5d",
            "order_size_persistence",
            (
                "total_amt",
                "cum_amt_10000",
                "cum_amt_40000",
                "cum_amt_50000",
                "cum_amt_200000",
                "cum_amt_1000000",
            ),
            "order_size_distribution_daily",
            _build_order_size_persistence,
        ),
    ]
    return specs


def build_primitive_capability(specs: Optional[Sequence[CandidateSpec]] = None) -> pd.DataFrame:
    """Phase 0 capability map — read-only inventory, no Raw access."""
    specs = list(specs or candidate_registry())
    column_inventory = _inventory_primitive_columns()
    rows = []
    for spec in specs:
        sources = [s.strip() for s in spec.source_primitive.split("+")]
        available_cols: Dict[str, set] = {
            src: column_inventory.get(src, set()) for src in sources
        }
        union_cols = set().union(*available_cols.values()) if available_cols else set()
        # Derived cancel pressure columns need raw cancel fields present.
        required = list(spec.required_columns)
        if spec.source_primitive == "cancel_lifecycle_daily":
            # pressure/intensity are derived; capability checks raw inputs.
            required_check = required
        else:
            required_check = required
        missing = [c for c in required_check if c not in union_cols]
        available = bool(spec.available) and not missing
        reason = spec.reason_if_unavailable
        if missing and not reason:
            reason = f"missing columns: {missing}"
        elif missing and reason:
            reason = f"{reason}; also missing columns: {missing}"
        if not available and not reason:
            reason = "unavailable"
        rows.append(
            {
                "candidate": spec.name,
                "required_columns": "|".join(required),
                "available": available,
                "source_primitive": spec.source_primitive,
                "reason_if_unavailable": "" if available else reason,
            }
        )
    return pd.DataFrame(rows)


def _inventory_primitive_columns() -> Dict[str, set]:
    mapping = {
        "trade_flow_daily": _PRIMITIVES / "trade_flow_daily" / "chunks",
        "order_size_distribution_daily": (
            _PRIMITIVES / "order_size_distribution_daily" / "chunks"
        ),
        "order_book_daily": _PRIMITIVES / "order_book_daily" / "dataset",
        "price_formation_daily": (
            _PRIMITIVES / "price_formation_daily" / "dataset"
        ),
        "liquidity_impact_daily": (
            _PRIMITIVES / "liquidity_impact_daily" / "dataset"
        ),
        "cancel_lifecycle_daily": (
            _PRIMITIVES / "cancel_lifecycle_daily" / "dataset"
        ),
    }
    out: Dict[str, set] = {}
    for name, path in mapping.items():
        if not path.exists():
            out[name] = set()
            continue
        files = _partition_files(path)
        if not files:
            out[name] = set()
            continue
        sample = pd.read_parquet(files[0])
        cols = set(sample.columns)
        # Document derived cancel fields as expressible from raw.
        if name == "cancel_lifecycle_daily" and {
            "buy_cancel_value",
            "sell_cancel_value",
            "total_trade_value",
        }.issubset(cols):
            cols.update(
                {
                    "cancel_value_pressure",
                    "cancel_value_intensity",
                }
            )
        out[name] = cols
    return out


def gate_label_sprint(metrics: Dict[str, float]) -> str:
    if (
        metrics["hl_sharpe"] >= 3.0
        and metrics["decile_mono_spearman"] >= 0.85
        and metrics["adjacent_violations"] <= 1
    ):
        return "strong_candidate"
    if (
        metrics["hl_sharpe"] >= 2.0
        and metrics["decile_mono_spearman"] >= 0.75
        and metrics["adjacent_violations"] <= 2
    ):
        return "research_candidate"
    return "fail"


def _load_all_panels(
    start: pd.Timestamp, end: pd.Timestamp
) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}
    panels["order_book"] = load_primitive_panel(
        _PRIMITIVES / "order_book_daily" / "dataset",
        [
            "obi_5_mean",
            "obi_5_std",
            "log_total_depth_mean",
            "log_total_depth_std",
            "microprice_deviation_mean",
            "relative_spread_mean",
        ],
        start,
        end,
    )
    panels["trade_flow"] = load_primitive_panel(
        _PRIMITIVES / "trade_flow_daily" / "chunks",
        ["active_buy_amt", "active_sell_amt", "total_amt"],
        start,
        end,
    )
    panels["price_formation"] = load_primitive_panel(
        _PRIMITIVES / "price_formation_daily" / "dataset",
        ["open_to_close_return"],
        start,
        end,
    )
    panels["liquidity_impact"] = load_primitive_panel(
        _PRIMITIVES / "liquidity_impact_daily" / "dataset",
        [
            "effective_spread_proxy",
            "signed_amount_impact",
            "depth_recovery_5m",
            "coverage_ratio",
        ],
        start,
        end,
    )
    # Order-size: load raw then build frozen feature frame (representative only).
    order_size_raw = load_primitive_panel(
        _PRIMITIVES / "order_size_distribution_daily" / "chunks",
        [
            "total_amt",
            "trade_cnt",
            "active_buy_amt",
            "active_sell_amt",
            "cum_amt_10000",
            "cum_cnt_10000",
            "buy_cum_amt_10000",
            "sell_cum_amt_10000",
            "cum_amt_40000",
            "cum_cnt_40000",
            "buy_cum_amt_40000",
            "sell_cum_amt_40000",
            "cum_amt_50000",
            "cum_cnt_50000",
            "buy_cum_amt_50000",
            "sell_cum_amt_50000",
            "cum_amt_200000",
            "cum_cnt_200000",
            "buy_cum_amt_200000",
            "sell_cum_amt_200000",
            "cum_amt_1000000",
            "cum_cnt_1000000",
            "buy_cum_amt_1000000",
            "sell_cum_amt_1000000",
        ],
        start,
        end,
    )
    panels["order_size"] = build_order_size_feature_frame(order_size_raw)
    panels["flow_price"] = _merge_flow_price(
        panels["trade_flow"], panels["price_formation"]
    )
    cancel_ok, _ = _cancel_discovery_coverage()
    if cancel_ok:
        cancel_raw = load_primitive_panel(
            _PRIMITIVES / "cancel_lifecycle_daily" / "dataset",
            [
                "buy_cancel_value",
                "sell_cancel_value",
                "buy_cancel_qty",
                "sell_cancel_qty",
                "buy_cancel_event_count",
                "sell_cancel_event_count",
                "total_trade_value",
                "total_trade_qty",
                "total_trade_count",
            ],
            start,
            end,
        )
        panels["cancel"] = build_cancel_candidates(cancel_raw)
    return panels


def _panel_for_spec(
    spec: CandidateSpec, panels: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    mapping = {
        "order_book_daily": "order_book",
        "liquidity_impact_daily": "liquidity_impact",
        "order_size_distribution_daily": "order_size",
        "cancel_lifecycle_daily": "cancel",
        "trade_flow_daily+price_formation_daily": "flow_price",
    }
    key = mapping[spec.source_primitive]
    return panels[key]


def build_candidate_narrow(
    spec: CandidateSpec, panels: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    if not spec.available or spec.builder is None:
        raise RuntimeError(f"{spec.name} is unavailable")
    panel = _panel_for_spec(spec, panels)
    values = spec.builder(panel)
    # Clip to discovery window (builders may use buffer rows).
    mask = panel["TradeDate"].between(DISCOVERY_START, DISCOVERY_END)
    return _to_narrow(
        panel.loc[mask, "symbol"],
        panel.loc[mask, "TradeDate"],
        values.loc[mask],
        spec.name,
    )


def run_low_turnover_v1(
    *,
    output_root: Optional[Path] = None,
    window: str = "discovery",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Phase 0 + available candidates once on the frozen window."""
    if window != "discovery":
        raise ValueError(
            "Sprint 8 only allows the frozen discovery window; "
            f"got {window!r}"
        )
    start, end = WINDOWS[window]
    out_root = Path(output_root) if output_root else OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    figures_root = out_root / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    specs = candidate_registry()
    capability = build_primitive_capability(specs)
    capability.to_csv(out_root / "primitive_capability.csv", index=False)

    t0 = time.perf_counter()
    context = load_fast_context(window)
    context_load_seconds = time.perf_counter() - t0
    mask, ret = context

    t0 = time.perf_counter()
    panels = _load_all_panels(start, end)
    primitive_load_seconds = time.perf_counter() - t0

    summary_rows: List[Dict[str, object]] = []
    profile_rows: List[Dict[str, object]] = []
    available_map = {
        row["candidate"]: bool(row["available"])
        for row in capability.to_dict("records")
    }
    reason_map = {
        row["candidate"]: row["reason_if_unavailable"]
        for row in capability.to_dict("records")
    }

    for spec in specs:
        if not available_map.get(spec.name, False):
            summary_rows.append(
                {
                    "factor": spec.name,
                    "mechanism": spec.mechanism,
                    "source_primitive": spec.source_primitive,
                    "window": window,
                    "gate": "unavailable",
                    "reason_if_unavailable": reason_map.get(spec.name, ""),
                    "rank_ic_mean_raw": np.nan,
                    "icir_raw": np.nan,
                    "hl_annu_ret": np.nan,
                    "hl_sharpe": np.nan,
                    "g10_excess_sharpe": np.nan,
                    "decile_mono_spearman": np.nan,
                    "adjacent_violations": np.nan,
                    "positive_month_fraction": np.nan,
                    "cum_hl_time_spearman": np.nan,
                    "avg_hl_turnover": np.nan,
                    "implied_annu_fee": np.nan,
                    "net_annu_after_fee": np.nan,
                    "factor_direction": np.nan,
                    "n_days": np.nan,
                    "n_names_avg": np.nan,
                }
            )
            profile_rows.append(
                {
                    "family": "low_turnover_v1",
                    "factor": spec.name,
                    "window": window,
                    "primitive_load_seconds": round(primitive_load_seconds, 3),
                    "factor_compute_seconds": 0.0,
                    "context_load_seconds": round(context_load_seconds, 3),
                    "backtest_seconds": 0.0,
                    "plot_seconds": 0.0,
                    "total_seconds": 0.0,
                    "status": "unavailable",
                }
            )
            print(f"[skip] {spec.name}: unavailable", flush=True)
            continue

        t_factor = time.perf_counter()
        narrow = build_candidate_narrow(spec, panels)
        # Keep only discovery dates (narrow already clipped).
        factor_compute_seconds = time.perf_counter() - t_factor

        t_bt = time.perf_counter()
        group_pnl, group_to, rank_ic, summary = backtest_factor(
            narrow,
            start_day=start,
            end_day=end,
            mask=mask,
            ret_matrix=ret,
        )
        backtest_seconds = time.perf_counter() - t_bt

        metrics = compute_fast_metrics(group_pnl, group_to, summary)
        fee = float(summary.get("implied_annu_fee", implied_annu_fee(metrics["avg_hl_turnover"])))
        net = float(metrics["hl_annu_ret"]) - fee
        gate = gate_label_sprint(metrics)

        t_plot = time.perf_counter()
        save_fast_plots(
            figures_root / spec.name, spec.name, group_pnl, metrics
        )
        plot_seconds = time.perf_counter() - t_plot

        summary_rows.append(
            {
                "factor": spec.name,
                "mechanism": spec.mechanism,
                "source_primitive": spec.source_primitive,
                "window": window,
                "gate": gate,
                "reason_if_unavailable": "",
                "rank_ic_mean_raw": metrics["rank_ic_mean_raw"],
                "icir_raw": metrics["icir_raw"],
                "hl_annu_ret": metrics["hl_annu_ret"],
                "hl_sharpe": metrics["hl_sharpe"],
                "g10_excess_sharpe": metrics["g10_excess_sharpe"],
                "decile_mono_spearman": metrics["decile_mono_spearman"],
                "adjacent_violations": metrics["adjacent_violations"],
                "positive_month_fraction": metrics[
                    "positive_hl_month_fraction"
                ],
                "cum_hl_time_spearman": metrics["cum_hl_time_spearman"],
                "avg_hl_turnover": metrics["avg_hl_turnover"],
                "implied_annu_fee": fee,
                "net_annu_after_fee": net,
                "factor_direction": metrics["factor_direction"],
                "n_days": metrics["n_days"],
                "n_names_avg": metrics["n_names_avg"],
            }
        )
        profile_rows.append(
            {
                "family": "low_turnover_v1",
                "factor": spec.name,
                "window": window,
                "primitive_load_seconds": round(primitive_load_seconds, 3),
                "factor_compute_seconds": round(factor_compute_seconds, 3),
                "context_load_seconds": round(context_load_seconds, 3),
                "backtest_seconds": round(backtest_seconds, 3),
                "plot_seconds": round(plot_seconds, 3),
                "total_seconds": round(
                    factor_compute_seconds + backtest_seconds + plot_seconds,
                    3,
                ),
                "status": "ok",
            }
        )
        print(
            f"[fast] {spec.name}: sharpe={metrics['hl_sharpe']:.2f} "
            f"mono={metrics['decile_mono_spearman']:.2f} "
            f"to={metrics['avg_hl_turnover']:.2f} gate={gate} "
            f"(bt={backtest_seconds:.1f}s)",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    profile_df = pd.DataFrame(profile_rows)
    summary_df = _sort_summary(summary_df)
    summary_df.to_csv(out_root / "candidate_summary.csv", index=False)
    profile_df.to_csv(out_root / "fast_profile.csv", index=False)
    report = render_report(summary_df, capability, profile_df)
    (out_root / "report.md").write_text(report, encoding="utf-8")
    return summary_df, profile_df, capability


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    gate_order = {
        "strong_candidate": 0,
        "research_candidate": 1,
        "fail": 2,
        "unavailable": 3,
    }
    out = summary.copy()
    out["_gate_ord"] = out["gate"].map(gate_order).fillna(9)
    out = out.sort_values(
        by=[
            "_gate_ord",
            "hl_sharpe",
            "decile_mono_spearman",
            "positive_month_fraction",
        ],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    return out.drop(columns=["_gate_ord"]).reset_index(drop=True)


def render_report(
    summary: pd.DataFrame,
    capability: pd.DataFrame,
    profile: pd.DataFrame,
) -> str:
    """Markdown report — fast discovery evidence only; no KEEP/DROP."""
    lines: List[str] = []
    lines.append("# Sprint 8 — Low-Turnover L2 Discovery v1")
    lines.append("")
    lines.append(
        "Fast discovery evidence only (frozen window 2023-01-01 ~ 2024-12-31). "
        "No Raw Tick/SSL2, no parameter search, no Full Validation, no KEEP/DROP."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    n_avail = int(capability["available"].sum())
    n_unavail = int((~capability["available"]).sum())
    lines.append(f"- Candidates defined: {len(capability)}")
    lines.append(f"- Available / run once: {n_avail}")
    lines.append(f"- Unavailable (no Raw backfill): {n_unavail}")
    lines.append(f"- Output root: `{OUT_DIR}`")
    lines.append("")

    # Benchmark reference from Fast Lane
    bench_path = (
        Path(RESULT_ROOT)
        / "fast_discovery"
        / "benchmark"
        / "fast_summary.csv"
    )
    lines.append("## Benchmark reference (daily effective_spread_proxy)")
    lines.append("")
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        row = bench.loc[bench["factor"] == "effective_spread_proxy"]
        if len(row):
            r = row.iloc[0]
            lines.append(
                f"- discovery-window `effective_spread_proxy`: "
                f"H-L Sharpe={r['hl_sharpe']:.3f}, "
                f"mono={r['decile_mono_spearman']:.4f}, "
                f"violations={int(r['adjacent_violations'])}, "
                f"turnover={r['avg_hl_turnover']:.3f}, "
                f"pos_month={r['positive_hl_month_fraction']:.3f}"
            )
        else:
            lines.append("- benchmark row not found in fast_summary.csv")
    else:
        lines.append("- benchmark fast_summary.csv not found")
    lines.append("")

    def _section(title: str, gate: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        block = summary.loc[summary["gate"] == gate]
        if block.empty:
            lines.append("_None_")
            lines.append("")
            return
        show_cols = [
            "factor",
            "mechanism",
            "hl_sharpe",
            "decile_mono_spearman",
            "adjacent_violations",
            "positive_month_fraction",
            "cum_hl_time_spearman",
            "avg_hl_turnover",
            "implied_annu_fee",
            "net_annu_after_fee",
            "rank_ic_mean_raw",
            "icir_raw",
            "g10_excess_sharpe",
        ]
        if gate == "unavailable":
            show_cols = [
                "factor",
                "mechanism",
                "source_primitive",
                "reason_if_unavailable",
            ]
        lines.append("```")
        lines.append(
            block[show_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "",
            )
        )
        lines.append("```")
        lines.append("")

    _section("strong_candidate", "strong_candidate")
    _section("research_candidate", "research_candidate")
    _section("fail", "fail")
    _section("unavailable", "unavailable")

    lines.append("## Focus questions")
    lines.append("")
    lines.extend(_answer_focus_questions(summary))
    lines.append("")
    lines.append("## Timing profile (available factors)")
    lines.append("")
    ok = profile.loc[profile.get("status", "ok") == "ok"] if "status" in profile.columns else profile
    if ok.empty:
        lines.append("_No available factors ran._")
    else:
        lines.append(
            f"- Mean backtest seconds/factor: "
            f"{ok['backtest_seconds'].mean():.2f}"
        )
        lines.append(
            f"- Primitive load once: "
            f"{ok['primitive_load_seconds'].iloc[0]:.2f}s"
        )
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append(
        "This report is fast discovery evidence only. "
        "Do not KEEP/DROP. Full Validation (2019-2026) requires "
        "manual confirmation of strong_candidate names."
    )
    lines.append("")
    return "\n".join(lines)


def _answer_focus_questions(summary: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    by_name = summary.set_index("factor")

    def _get(name: str, col: str) -> Optional[float]:
        if name not in by_name.index:
            return None
        val = by_name.loc[name, col]
        return float(val) if pd.notna(val) else None

    # Q1
    esp = _get("effective_spread_persistence_5d", "hl_sharpe")
    esp_to = _get("effective_spread_persistence_5d", "avg_hl_turnover")
    esp_mono = _get("effective_spread_persistence_5d", "decile_mono_spearman")
    esp_gate = (
        by_name.loc["effective_spread_persistence_5d", "gate"]
        if "effective_spread_persistence_5d" in by_name.index
        else "missing"
    )
    bench_path = (
        Path(RESULT_ROOT)
        / "fast_discovery"
        / "benchmark"
        / "fast_summary.csv"
    )
    bench_sharpe = bench_to = bench_mono = None
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        row = bench.loc[bench["factor"] == "effective_spread_proxy"]
        if len(row):
            bench_sharpe = float(row.iloc[0]["hl_sharpe"])
            bench_to = float(row.iloc[0]["avg_hl_turnover"])
            bench_mono = float(row.iloc[0]["decile_mono_spearman"])
    def _fmt(x: Optional[float], nd: int = 3) -> str:
        return "n/a" if x is None else f"{x:.{nd}f}"

    lines.append(
        "1. **effective_spread_proxy → 5D persistence**: "
        f"daily Sharpe≈{_fmt(bench_sharpe)}, TO≈{_fmt(bench_to)}, "
        f"mono≈{_fmt(bench_mono, 4)}; "
        f"persistence_5d Sharpe={_fmt(esp)}, TO={_fmt(esp_to)}, "
        f"mono={_fmt(esp_mono, 4)}, gate={esp_gate}."
    )

    # Q2
    div = _get("flow_price_divergence_5d", "decile_mono_spearman")
    div_s = _get("flow_price_divergence_5d", "hl_sharpe")
    div_v = _get("flow_price_divergence_5d", "adjacent_violations")
    lines.append(
        "2. **Flow-Price Divergence vs Trade Flow alone**: "
        f"flow_price_divergence_5d Sharpe={_fmt(div_s)}, mono={_fmt(div, 4)}, "
        f"violations={_fmt(div_v, 0)} "
        "(compare vs discovery net_buy_ratio mono≈0.94 / Sharpe≈1.68)."
    )

    # Q3
    sell_mono = _get("sell_absorption_5d", "decile_mono_spearman")
    sell_v = _get("sell_absorption_5d", "adjacent_violations")
    sell_g10 = _get("sell_absorption_5d", "g10_excess_sharpe")
    sell_hl = _get("sell_absorption_5d", "hl_sharpe")
    lines.append(
        "3. **Sell Absorption continuity**: "
        f"Sharpe={_fmt(sell_hl)}, mono={_fmt(sell_mono, 4)}, "
        f"violations={_fmt(sell_v, 0)}, "
        f"G10 excess Sharpe={_fmt(sell_g10)} "
        "(inspect decile_bar.png for endpoint jump vs continuous ladder)."
    )

    # Q4
    book = [
        n
        for n in (
            "stable_obi_5d",
            "persistent_obi_5d",
            "book_depth_quality_5d",
            "microprice_pressure_5d",
        )
        if n in by_name.index and by_name.loc[n, "gate"] != "unavailable"
    ]
    cancel = [
        n
        for n in (
            "persistent_cancel_pressure_5d",
            "persistent_cancel_intensity_5d",
            "cancel_pressure_consistency_5d",
        )
        if n in by_name.index
    ]
    book_note = ", ".join(
        f"{n}(TO={_get(n,'avg_hl_turnover'):.2f},S={_get(n,'hl_sharpe'):.2f})"
        for n in book
        if _get(n, "avg_hl_turnover") is not None
    )
    cancel_gates = ", ".join(
        f"{n}={by_name.loc[n, 'gate']}" for n in cancel
    )
    lines.append(
        "4. **Persistent book / cancel vs daily shock**: "
        f"book candidates — {book_note or 'n/a'}; cancel — {cancel_gates}."
    )

    # Q5
    strong = summary.loc[summary["gate"] == "strong_candidate"]
    mechs = sorted(strong["mechanism"].unique()) if len(strong) else []
    lines.append(
        "5. **≥2 distinct-mechanism strong candidates?** "
        f"strong_candidate count={len(strong)}; "
        f"mechanisms={mechs}; "
        f"{'YES' if len(mechs) >= 2 else 'NO'}."
    )
    return lines
