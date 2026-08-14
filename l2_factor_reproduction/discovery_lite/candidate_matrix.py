"""Shared Date × Symbol × candidate matrix.

Dry-run / existing families: load materialized ``factor_narrow.parquet``
(date-filtered, once per factor, no DB). Future families: evaluate many
formulas from a shared primitive panel via an explicit adapter / callable map.
Never ``eval()`` arbitrary formula strings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.discovery_lite.contracts import (
    LITE_END,
    LITE_START,
    REGISTRY_OPTIONAL,
    REGISTRY_REQUIRED,
    lite_trading_dates,
)
from l2_factor_reproduction.python.backtest import narrow_to_wide
from l2_factor_reproduction.python.candidate_pool_registry import (
    BRIDGE_FACTOR,
    FAMILY_REGISTRY,
)
from l2_factor_reproduction.python.fast_discovery import (
    FAMILY_ADAPTERS,
    FamilyAdapter,
    PRIMITIVE_BUFFER_DAYS,
    _drop_nested_partitions,
    _file_window_overlap,
    context_paths,
    load_fast_context,
)
from l2_factor_reproduction.python.order_size_factors import (
    ORDER_SIZE_FACTOR_NAMES,
    build_order_size_feature_frame,
)
from l2_factor_reproduction.python.order_size_factors import (
    feature_to_narrow as order_size_to_narrow,
)

logger = logging.getLogger(__name__)

RESULT_ROOT_P = Path(RESULT_ROOT)

# Explicit safe formula callables for future families (name -> primitive frame to Series).
FORMULA_CALLABLES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def register_formula_callable(
    name: str, fn: Callable[[pd.DataFrame], pd.Series]
) -> None:
    """Register an explicit formula callable. Not an ``eval()`` engine."""
    if not name or not callable(fn):
        raise ValueError("formula callable requires a non-empty name and callable")
    FORMULA_CALLABLES[name] = fn


LR_FAMILY = "liquidity_resilience"
LR_MAT_DIR = RESULT_ROOT_P / "liquidity_resilience" / "lr1_lite_materialization"


def resolve_factor_narrow_path(factor: str, family: str) -> Path:
    """Reuse candidate-pool FamilyConfig paths; do not invent a second registry.

    ``liquidity_resilience`` is a discovery_lite-only family path. It does not
    mutate candidate_pool_v1 / FAMILY_REGISTRY.
    """
    if family == "trade_flow_mcap_bridge" or factor == BRIDGE_FACTOR:
        return RESULT_ROOT_P / BRIDGE_FACTOR / "factor_narrow.parquet"
    if family == LR_FAMILY:
        return LR_MAT_DIR / "factors" / factor / "factor_narrow.parquet"
    if family not in FAMILY_REGISTRY:
        raise KeyError(f"Unknown family {family!r} for factor {factor!r}")
    return FAMILY_REGISTRY[family].factor_result_dir(factor) / "factor_narrow.parquet"


def _order_size_adapter() -> FamilyAdapter:
    """Local wrap of order_size builders. Fast Discovery FAMILY_ADAPTERS is unchanged."""
    return FamilyAdapter(
        name="order_size",
        primitive_dir=RESULT_ROOT_P / "primitives" / "order_size_distribution_daily" / "dataset",
        builder=build_order_size_feature_frame,
        to_narrow=order_size_to_narrow,
        factor_names=tuple(ORDER_SIZE_FACTOR_NAMES),
    )


def family_adapter(family: str) -> Optional[FamilyAdapter]:
    if family in FAMILY_ADAPTERS:
        return FAMILY_ADAPTERS[family]
    if family == "order_size":
        adapter = _order_size_adapter()
        if adapter.primitive_dir.exists():
            return adapter
    return None


def load_candidate_registry(path: Path) -> pd.DataFrame:
    """Load a BDL candidate registry. Missing optional columns are filled."""
    df = pd.read_csv(path)
    if "name" not in df.columns and "factor" in df.columns:
        df = df.rename(columns={"factor": "name"})
    if "name" not in df.columns:
        raise ValueError(f"{path} must contain a name (or factor) column")
    df["name"] = df["name"].astype(str)
    defaults = {
        "family": "",
        "formula": "",
        "mechanism": "",
        "lookback_days": 1,
        "signed": True,
        "positive_value_meaning": "",
        "primitive_dependencies": "",
        "registry_status": "",
        "category": "",
        "expected_redundancy": "",
        "normalization": "",
        "notes": "",
        "replacement_candidate": False,
        "near_alias_exception": False,
        "sparse_event": False,
        "pit_status": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["lookback_days"] = pd.to_numeric(df["lookback_days"], errors="coerce").fillna(1).astype(int)
    for flag in ("signed", "replacement_candidate", "near_alias_exception", "sparse_event"):
        df[flag] = df[flag].map(_as_bool)
    missing_required = [c for c in REGISTRY_REQUIRED if c not in df.columns]
    if missing_required:
        raise ValueError(f"registry missing required columns: {missing_required}")
    _ = REGISTRY_OPTIONAL  # documented optional fields; already defaulted
    return df.reset_index(drop=True)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "0", "false", "no"}:
        return False
    return text in {"1", "true", "yes", "y"}


def load_trading_calendar(window: str = "discovery") -> pd.DatetimeIndex:
    """Canonical sorted trading dates from Fast Discovery fast_context."""
    path = context_paths(window)["trading_dates"]
    frame = pd.read_parquet(path)
    col = "TradeDate" if "TradeDate" in frame.columns else frame.columns[0]
    dates = pd.to_datetime(frame[col]).dt.normalize().sort_values().unique()
    return pd.DatetimeIndex(dates, name="TradeDate")


def load_factor_narrow_slice(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Date-filtered narrow load. Adapted from FS-1 panel helper; FS-1 is not mutated."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59)
    columns = ["symbol", "tradetime", "value"]
    try:
        df = pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("tradetime", ">=", start.to_pydatetime()),
                ("tradetime", "<=", end.to_pydatetime()),
            ],
        )
    except Exception:  # noqa: BLE001 — predicate pushdown unsupported
        df = pd.read_parquet(path, columns=columns)
        tt = pd.to_datetime(df["tradetime"])
        df = df.loc[(tt >= start) & (tt <= end)]
    return df


def narrow_slice_to_wide(narrow: pd.DataFrame) -> pd.DataFrame:
    if narrow is None or narrow.empty:
        return pd.DataFrame()
    wide = narrow_to_wide(narrow)
    return wide.astype(np.float32, copy=False)


def _load_adapter_features(
    adapter: FamilyAdapter,
    start: pd.Timestamp,
    end: pd.Timestamp,
    buffer_days: int = PRIMITIVE_BUFFER_DAYS,
) -> pd.DataFrame:
    """Same primitive-once pattern as fast_discovery.load_family_features."""
    files = [
        path
        for path in adapter.partition_files()
        if _file_window_overlap(path, start, end, buffer_days)
    ]
    files = _drop_nested_partitions(files)
    if not files:
        raise FileNotFoundError(
            f"{adapter.name}: no primitive partitions overlap "
            f"[{start.date()}, {end.date()}]"
        )
    primitive = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    if "TradeDate" not in primitive.columns:
        raise ValueError(f"{adapter.name}: primitive lacks TradeDate")
    primitive["TradeDate"] = pd.to_datetime(primitive["TradeDate"])
    lo = start - pd.Timedelta(days=buffer_days)
    primitive = primitive.loc[primitive["TradeDate"].between(lo, end)]
    features = adapter.builder(primitive)
    features["TradeDate"] = pd.to_datetime(features["TradeDate"])
    return features.loc[features["TradeDate"].between(start, end)].reset_index(drop=True)


@dataclass
class CandidateMatrix:
    registry: pd.DataFrame
    wides: Dict[str, pd.DataFrame]
    trading_dates: pd.DatetimeIndex
    lite_dates: pd.DatetimeIndex
    mask: pd.DataFrame
    ret: pd.DataFrame
    load_meta: Dict[str, object] = field(default_factory=dict)
    availability: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def names(self) -> List[str]:
        return list(self.registry["name"].astype(str))


def load_candidate_matrix(
    registry: pd.DataFrame,
    *,
    window: str = "discovery",
    source: str = "auto",
    context: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    start: pd.Timestamp = LITE_START,
    end: pd.Timestamp = LITE_END,
    novelty_names: Sequence[str] = (),
    verify_hash: bool = True,
) -> CandidateMatrix:
    """Build a shared candidate matrix. One context load; no per-factor DB scan."""
    if source not in {"auto", "materialized", "primitives"}:
        raise ValueError(f"unknown source {source!r}")
    trading_dates = load_trading_calendar(window)
    lite_dates = lite_trading_dates(trading_dates, start=start, end=end)
    if context is None:
        mask, ret = load_fast_context(window, verify_hash=verify_hash)
    else:
        mask, ret = context
    mask = mask.copy()
    ret = ret.copy()
    mask.index = pd.to_datetime(mask.index).normalize()
    ret.index = pd.to_datetime(ret.index).normalize()

    print(
        f"[bdl] budget before exposure load: n_candidates={len(registry)} "
        f"n_lite_dates={len(lite_dates)} n_trading_dates={len(trading_dates)} "
        f"estimated_factor_rows={len(lite_dates)*int(mask.shape[1])*max(len(registry),1)} "
        f"data_sources=fast_context/{window} + materialized factor_narrow "
        f"(no per-factor DB). steps=load,gate0,gate1,gate2,gate3,report",
        flush=True,
    )

    availability_rows: List[Dict[str, object]] = []
    wides: Dict[str, pd.DataFrame] = {}
    sources_used: List[str] = []
    db_scans = 0

    grouped = list(registry.groupby("family", sort=False))
    primitive_cache: Dict[str, pd.DataFrame] = {}

    for family, block in grouped:
        family = str(family)
        adapter = family_adapter(family)
        want = [str(n) for n in block["name"]]
        if family == LR_FAMILY and source != "primitives":
            panel_path = LR_MAT_DIR / "panel.parquet"
            if panel_path.exists():
                panel = pd.read_parquet(panel_path)
                panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
                sym_col = "Symbol" if "Symbol" in panel.columns else "symbol"
                for name in want:
                    if name not in panel.columns:
                        availability_rows.append(
                            _availability_row(
                                name, family, False, False, "REJECT_MISSING_PRIMITIVE"
                            )
                        )
                        continue
                    wide = panel.pivot_table(
                        index="TradeDate",
                        columns=sym_col,
                        values=name,
                        aggfunc="last",
                    )
                    wide.index = pd.to_datetime(wide.index).normalize()
                    wides[name] = wide.astype(np.float32, copy=False)
                    availability_rows.append(
                        _availability_row(name, family, True, True, "OK")
                    )
                sources_used.append(str(panel_path))
                print(
                    f"[bdl] loaded {family} shared panel {panel_path} "
                    f"({len(want)} candidates, no per-factor DB)",
                    flush=True,
                )
                continue
        materialized_paths = {
            name: resolve_factor_narrow_path(name, family) for name in want
        }
        all_materialized = all(path.exists() for path in materialized_paths.values())
        use_primitives = source == "primitives" or (
            source == "auto" and not all_materialized and adapter is not None
        )
        if source == "materialized":
            use_primitives = False

        if use_primitives:
            if adapter is None:
                for name in want:
                    availability_rows.append(
                        _availability_row(
                            name, family, False, False, "REJECT_MISSING_PRIMITIVE"
                        )
                    )
                continue
            if family not in primitive_cache:
                primitive_cache[family] = _load_adapter_features(adapter, start, end)
                sources_used.append(f"primitive:{adapter.primitive_dir}")
            features = primitive_cache[family]
            for name in want:
                if name in FORMULA_CALLABLES:
                    series = FORMULA_CALLABLES[name](features)
                    wide = series.unstack() if isinstance(series, pd.Series) else series
                elif name in features.columns:
                    wide = features.pivot_table(
                        index="TradeDate",
                        columns="symbol" if "symbol" in features.columns else "Symbol",
                        values=name,
                        aggfunc="last",
                    )
                    wide.index = pd.to_datetime(wide.index).normalize()
                else:
                    availability_rows.append(
                        _availability_row(
                            name, family, False, True, "REJECT_MISSING_PRIMITIVE"
                        )
                    )
                    continue
                wides[name] = wide.astype(np.float32, copy=False)
                availability_rows.append(
                    _availability_row(name, family, True, True, "OK")
                )
            continue

        for name in want:
            path = materialized_paths[name]
            if not path.exists():
                availability_rows.append(
                    _availability_row(
                        name, family, False, adapter is not None, "REJECT_MISSING_PRIMITIVE"
                    )
                )
                continue
            narrow = load_factor_narrow_slice(path, start, end)
            wide = narrow_slice_to_wide(narrow)
            wides[name] = wide
            sources_used.append(str(path))
            availability_rows.append(
                _availability_row(name, family, True, True, "OK")
            )
            print(
                f"[bdl] loaded {family}/{name}: "
                f"{wide.shape[0]} dates × {wide.shape[1]} symbols",
                flush=True,
            )

    # Novelty reference exposures (materialized only; sequential, discarded after copy-in).
    novelty_wides: Dict[str, pd.DataFrame] = {}
    for spec in novelty_names:
        if isinstance(spec, str) and ":" in spec:
            family, name = spec.split(":", 1)
        else:
            name = str(spec)
            family = _family_for_pool_factor(name)
        if not family or name in wides:
            continue
        try:
            path = resolve_factor_narrow_path(name, family)
        except KeyError:
            continue
        if not path.exists():
            continue
        novelty_wides[name] = narrow_slice_to_wide(
            load_factor_narrow_slice(path, start, end)
        )
        sources_used.append(f"novelty:{path}")
        print(f"[bdl] novelty ref {family}/{name}", flush=True)

    load_meta = {
        "window": window,
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "n_candidates": int(len(registry)),
        "n_loaded": int(len(wides)),
        "n_lite_dates": int(len(lite_dates)),
        "n_trading_dates": int(len(trading_dates)),
        "estimated_factor_rows": int(
            len(lite_dates) * int(mask.shape[1]) * max(len(wides), 1)
        ),
        "data_sources_loaded": sorted(set(sources_used)),
        "db_scans": db_scans,
        "n_novelty_reference": int(len(novelty_wides)),
        "source_mode": source,
    }
    matrix = CandidateMatrix(
        registry=registry.reset_index(drop=True),
        wides=wides,
        trading_dates=trading_dates,
        lite_dates=lite_dates,
        mask=mask,
        ret=ret,
        load_meta=load_meta,
        availability=pd.DataFrame(availability_rows),
    )
    matrix.load_meta["novelty_wides"] = novelty_wides
    return matrix


def _availability_row(
    name: str,
    family: str,
    materialized: bool,
    primitive_available: bool,
    status: str,
) -> Dict[str, object]:
    return {
        "name": name,
        "family": family,
        "materialized": materialized,
        "primitive_available": primitive_available,
        "load_status": status,
    }


def _family_for_pool_factor(name: str) -> str:
    from l2_factor_reproduction.python.candidate_pool_registry import POOL_ROOT

    registry_path = POOL_ROOT / "candidate_registry.csv"
    if not registry_path.exists():
        return ""
    frame = pd.read_csv(registry_path, usecols=["name", "family"])
    hit = frame.loc[frame["name"].astype(str) == str(name), "family"]
    if hit.empty:
        return ""
    return str(hit.iloc[0])


def panel_on_dates(
    wides: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    names: Iterable[str],
    *,
    symbols: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Long panel TradeDate × Symbol × candidate columns on a date subset."""
    names = [n for n in names if n in wides]
    if not names:
        return pd.DataFrame(columns=["TradeDate", "Symbol"])
    pieces = []
    for name in names:
        wide = wides[name]
        if wide is None or wide.empty:
            continue
        sub = wide.reindex(index=dates)
        if symbols is not None:
            cols = sub.columns.intersection(symbols)
            sub = sub.loc[:, cols]
        long = sub.stack(future_stack=True).rename(name)
        long.index.names = ["TradeDate", "Symbol"]
        pieces.append(long)
    if not pieces:
        return pd.DataFrame(columns=["TradeDate", "Symbol"])
    panel = pd.concat(pieces, axis=1)
    panel = panel.reset_index()
    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"]).dt.normalize()
    return panel
