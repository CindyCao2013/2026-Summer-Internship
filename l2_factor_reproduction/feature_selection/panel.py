"""FS-1 panel adapter: inventory, PIT spine, align raw factors, audits.

Reuses candidate_pool_registry path resolution and fast_context universe mask.
Does not redefine factors; does not attach labels / forward returns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from l2_factor_reproduction.config.settings import RESULT_ROOT
from l2_factor_reproduction.feature_selection.contracts import (
    FORBIDDEN_NAME_SUBSTRINGS,
    FORBIDDEN_OUTPUT_COLUMNS,
    FS1_OUT_ROOT,
    MIN_ALIGNED_ROWS,
    MIN_DATE_COVERAGE,
    MIN_MEAN_SYMBOL_COVERAGE,
    PANEL_SCHEMA,
    PARITY_FACTORS,
)
from l2_factor_reproduction.python.candidate_pool_registry import (
    BRIDGE_CONFIG,
    BRIDGE_FACTOR,
    FAMILY_REGISTRY,
    POOL_ROOT,
)
from l2_factor_reproduction.python.fast_discovery import load_fast_context

logger = logging.getLogger(__name__)

RESULT_ROOT_P = Path(RESULT_ROOT)


# ---------------------------------------------------------------------------
# Path resolution (reuse FamilyConfig; no parallel registry)
# ---------------------------------------------------------------------------


def resolve_factor_narrow_path(factor: str, family: str) -> Path:
    """Locate frozen ``factor_narrow.parquet`` for a registry row."""
    if family == "trade_flow_mcap_bridge" or factor == BRIDGE_FACTOR:
        return RESULT_ROOT_P / BRIDGE_FACTOR / "factor_narrow.parquet"
    if family not in FAMILY_REGISTRY:
        raise KeyError(f"Unknown family {family!r} for factor {factor!r}")
    cfg = FAMILY_REGISTRY[family]
    return cfg.factor_result_dir(factor) / "factor_narrow.parquet"


def load_candidate_registry() -> pd.DataFrame:
    path = POOL_ROOT / "candidate_registry.csv"
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError("candidate_registry.csv missing 'name'")
    return df.rename(columns={"name": "factor"})


def load_candidate_summary() -> pd.DataFrame:
    path = POOL_ROOT / "candidate_summary.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Gate 1 — feature inventory
# ---------------------------------------------------------------------------


def _probe_narrow_stats(path: Path) -> Dict[str, object]:
    """Read only tradetime/symbol for coverage stats."""
    cols = ["tradetime", "symbol"]
    df = pd.read_parquet(path, columns=cols)
    tt = pd.to_datetime(df["tradetime"])
    dates = tt.dt.normalize()
    return {
        "n_rows": int(len(df)),
        "n_dates": int(dates.nunique()),
        "n_symbols": int(df["symbol"].nunique()),
        "date_min": str(dates.min().date()) if len(dates) else "",
        "date_max": str(dates.max().date()) if len(dates) else "",
    }


def build_feature_inventory(
    *,
    out_path: Optional[Path] = None,
    probe_parquet: bool = True,
) -> pd.DataFrame:
    """Freeze feature universe from candidate_registry + materialization probe."""
    reg = load_candidate_registry()
    summary = load_candidate_summary().set_index("factor")

    rows: List[Dict[str, object]] = []
    for _, r in reg.iterrows():
        factor = str(r["factor"])
        family = str(r["family"])
        path = resolve_factor_narrow_path(factor, family)
        materialized = path.exists()
        s = summary.loc[factor] if factor in summary.index else None

        row: Dict[str, object] = {
            "factor": factor,
            "family": family,
            "category": r.get("category", ""),
            "formula": r.get("formula", ""),
            "lookback_days": r.get("lookback_days", ""),
            "registry_status": r.get("registry_status", ""),
            "mechanism": r.get("mechanism", ""),
            "data_source_path": str(path) if materialized else "",
            "registered": True,
            "materialized": bool(materialized),
        }

        if s is not None:
            row.update(
                {
                    "summary_date_min": s.get("date_min", ""),
                    "summary_date_max": s.get("date_max", ""),
                    "summary_n_factor_rows": s.get("n_factor_rows", ""),
                    "summary_n_symbols": s.get("n_symbols", ""),
                    "summary_n_days": s.get("n_days", ""),
                    "max_abs_corr": s.get("max_abs_corr", ""),
                    "redundancy_cluster_080": s.get(
                        "family_redundancy_cluster_080",
                        s.get("redundancy_cluster_080", ""),
                    ),
                }
            )
        else:
            row.update(
                {
                    "summary_date_min": "",
                    "summary_date_max": "",
                    "summary_n_factor_rows": "",
                    "summary_n_symbols": "",
                    "summary_n_days": "",
                    "max_abs_corr": "",
                    "redundancy_cluster_080": "",
                }
            )

        if materialized and probe_parquet:
            try:
                stats = _probe_narrow_stats(path)
                row.update(stats)
                row["probe_ok"] = True
            except Exception as exc:  # noqa: BLE001
                row.update(
                    {
                        "n_rows": 0,
                        "n_dates": 0,
                        "n_symbols": 0,
                        "date_min": "",
                        "date_max": "",
                        "probe_ok": False,
                        "probe_error": str(exc),
                    }
                )
        else:
            row.update(
                {
                    "n_rows": int(s["n_factor_rows"])
                    if s is not None and pd.notna(s.get("n_factor_rows"))
                    else 0,
                    "n_dates": int(s["n_days"])
                    if s is not None and pd.notna(s.get("n_days"))
                    else 0,
                    "n_symbols": int(s["n_symbols"])
                    if s is not None and pd.notna(s.get("n_symbols"))
                    else 0,
                    "date_min": s.get("date_min", "") if s is not None else "",
                    "date_max": s.get("date_max", "") if s is not None else "",
                    "probe_ok": False if not materialized else True,
                }
            )

        # Preliminary eligibility (coverage vs spine filled later in finalize)
        reasons: List[str] = []
        if not materialized:
            reasons.append("NOT_MATERIALIZED")
        if row.get("probe_ok") is False and materialized:
            reasons.append("PROBE_FAILED")
        if int(row.get("n_rows") or 0) < MIN_ALIGNED_ROWS:
            reasons.append("INSUFFICIENT_ROWS")

        # Forbidden name check
        low = factor.lower()
        if any(tok in low for tok in FORBIDDEN_NAME_SUBSTRINGS):
            reasons.append("FORBIDDEN_NAME")

        row["eligible_for_fs"] = len(reasons) == 0
        row["ineligible_reason"] = "|".join(reasons) if reasons else ""
        row["sufficient_history"] = int(row.get("n_dates") or 0) >= 1000
        rows.append(row)

    inv = pd.DataFrame(rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(out_path, index=False)
    return inv


def finalize_eligibility_with_spine_coverage(
    inventory: pd.DataFrame,
    feature_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Merge post-align coverage metrics into inventory eligibility."""
    inv = inventory.copy()
    cov = feature_coverage.set_index("factor")
    for col in (
        "date_coverage",
        "mean_symbol_coverage",
        "n_aligned_nonnull",
        "coverage_ratio",
    ):
        inv[col] = inv["factor"].map(
            lambda f, c=col: cov.loc[f, c] if f in cov.index else np.nan
        )

    reasons = []
    eligible = []
    for _, r in inv.iterrows():
        rs = [x for x in str(r.get("ineligible_reason") or "").split("|") if x]
        if not bool(r["materialized"]):
            if "NOT_MATERIALIZED" not in rs:
                rs.append("NOT_MATERIALIZED")
        dc = r.get("date_coverage")
        sc = r.get("mean_symbol_coverage")
        if pd.notna(dc) and float(dc) < MIN_DATE_COVERAGE:
            rs.append("LOW_DATE_COVERAGE")
        if pd.notna(sc) and float(sc) < MIN_MEAN_SYMBOL_COVERAGE:
            rs.append("LOW_SYMBOL_COVERAGE")
        nn = r.get("n_aligned_nonnull")
        if pd.notna(nn) and int(nn) < MIN_ALIGNED_ROWS:
            rs.append("LOW_ALIGNED_ROWS")
        # dedupe
        rs = list(dict.fromkeys(rs))
        reasons.append("|".join(rs))
        eligible.append(len(rs) == 0)
    inv["ineligible_reason"] = reasons
    inv["eligible_for_fs"] = eligible
    return inv


# ---------------------------------------------------------------------------
# Gate 2 — canonical (TradeDate, Symbol) spine from PIT universe
# ---------------------------------------------------------------------------


def build_spine_from_fast_context(
    window: str = "full",
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    smoke_n_dates: Optional[int] = None,
) -> pd.DataFrame:
    """Spine = investable names where universe_mask == 1.

    Intentionally NOT derived from any single factor's coverage.
    """
    mask, _ret = load_fast_context(window)
    # drop unused ret immediately — FS-1 must not ship labels
    del _ret

    mask.index = pd.to_datetime(mask.index).normalize()
    if start is not None:
        mask = mask.loc[pd.Timestamp(start) :]
    if end is not None:
        mask = mask.loc[: pd.Timestamp(end)]
    if smoke_n_dates is not None and smoke_n_dates > 0:
        keep = sorted(mask.index.unique())[: int(smoke_n_dates)]
        mask = mask.loc[keep]

    # stack where mask == 1
    m = mask.where(mask == 1)
    long = m.stack(future_stack=True).rename("mask_val").reset_index()
    long.columns = ["TradeDate", "Symbol", "mask_val"]
    # future_stack keeps NaNs; drop non-investable
    long = long.dropna(subset=["mask_val"])
    long["TradeDate"] = pd.to_datetime(long["TradeDate"]).dt.normalize()
    long = long[["TradeDate", "Symbol"]].drop_duplicates()
    long = long.sort_values(["TradeDate", "Symbol"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# Gate 3 — align raw factors onto spine
# ---------------------------------------------------------------------------


def _quarter_key(ts: pd.Timestamp) -> Tuple[int, int]:
    ts = pd.Timestamp(ts)
    return int(ts.year), int((ts.month - 1) // 3 + 1)


def iter_quarters(dates: Sequence[pd.Timestamp]) -> List[Tuple[int, int, pd.Timestamp, pd.Timestamp]]:
    """Return list of (year, quarter, q_start, q_end) covering dates."""
    if len(dates) == 0:
        return []
    dmin = pd.Timestamp(min(dates)).normalize()
    dmax = pd.Timestamp(max(dates)).normalize()
    out: List[Tuple[int, int, pd.Timestamp, pd.Timestamp]] = []
    y, q = _quarter_key(dmin)
    cur = pd.Timestamp(year=y, month=3 * (q - 1) + 1, day=1)
    while cur <= dmax:
        y, q = _quarter_key(cur)
        if q == 4:
            nxt = pd.Timestamp(year=y + 1, month=1, day=1)
        else:
            nxt = pd.Timestamp(year=y, month=3 * q + 1, day=1)
        q_end = nxt - pd.Timedelta(days=1)
        out.append((y, q, cur, q_end))
        cur = nxt
    return out


def load_factor_narrow_slice(
    path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load factor narrow rows in [start, end] (inclusive)."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59)
    try:
        df = pd.read_parquet(
            path,
            columns=["symbol", "tradetime", "value"],
            filters=[
                ("tradetime", ">=", start.to_pydatetime()),
                ("tradetime", "<=", end.to_pydatetime()),
            ],
        )
    except Exception:  # noqa: BLE001 — filters unsupported → full read + slice
        df = pd.read_parquet(path, columns=["symbol", "tradetime", "value"])
        tt = pd.to_datetime(df["tradetime"])
        df = df.loc[(tt >= start) & (tt <= end)]
    if df.empty:
        return df
    df = df.copy()
    df["TradeDate"] = pd.to_datetime(df["tradetime"]).dt.normalize()
    df = df.rename(columns={"symbol": "Symbol", "value": "value"})
    df = (
        df.groupby(["TradeDate", "Symbol"], as_index=False)["value"]
        .last()
    )
    return df


def align_quarter_panel(
    spine_q: pd.DataFrame,
    inventory: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Left-join factor values onto a quarter spine.

    Returns (panel, missingness_long sample counts per factor).
    """
    inv = inventory.set_index("factor")
    use = list(factors) if factors is not None else list(inv.index)
    # only materialized
    use = [f for f in use if f in inv.index and bool(inv.loc[f, "materialized"])]

    panel = spine_q[["TradeDate", "Symbol"]].copy()
    miss_rows: List[Dict[str, object]] = []

    if panel.empty or not use:
        return panel, pd.DataFrame(miss_rows)

    q_start = panel["TradeDate"].min()
    q_end = panel["TradeDate"].max()

    for factor in use:
        path = Path(str(inv.loc[factor, "data_source_path"]))
        lookback = inv.loc[factor, "lookback_days"]
        try:
            lookback_i = int(float(lookback)) if pd.notna(lookback) and lookback != "" else 1
        except (TypeError, ValueError):
            lookback_i = 1
        f_date_min = pd.to_datetime(inv.loc[factor, "date_min"], errors="coerce")

        try:
            slim = load_factor_narrow_slice(path, q_start, q_end)
            load_err = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("DATA_ERROR loading %s: %s", factor, exc)
            slim = pd.DataFrame(columns=["TradeDate", "Symbol", "value"])
            load_err = True

        if slim.empty:
            panel[factor] = np.nan
        else:
            slim = slim.rename(columns={"value": factor})
            panel = panel.merge(slim, on=["TradeDate", "Symbol"], how="left")

        # missingness summary for this factor/quarter
        n_spine = len(panel)
        n_nn = int(panel[factor].notna().sum()) if factor in panel.columns else 0
        n_na = n_spine - n_nn

        # classify NA mass (coarse, quarterly)
        warmup_cut = (
            f_date_min + pd.Timedelta(days=max(lookback_i - 1, 0))
            if pd.notna(f_date_min)
            else None
        )
        if load_err:
            reason_mode = "DATA_ERROR"
        elif n_nn == 0:
            reason_mode = "STRUCTURAL"
        else:
            # among NA rows, estimate warmup share
            na_mask = panel[factor].isna()
            if warmup_cut is not None and na_mask.any():
                n_warm = int((na_mask & (panel["TradeDate"] < warmup_cut)).sum())
            else:
                n_warm = 0
            # STRUCTURAL if date has near-zero coverage (<5% of day spine)
            by_day = panel.groupby("TradeDate")[factor].apply(lambda s: s.notna().mean())
            structural_days = int((by_day < 0.05).sum())
            reason_mode = "ORDINARY"
            if n_warm > 0.5 * n_na and n_na > 0:
                reason_mode = "WARMUP"
            elif structural_days > 0.5 * len(by_day):
                reason_mode = "STRUCTURAL"

        miss_rows.append(
            {
                "factor": factor,
                "year": int(q_start.year),
                "quarter": int(_quarter_key(q_start)[1]),
                "n_spine": n_spine,
                "n_nonnull": n_nn,
                "n_null": n_na,
                "null_frac": float(n_na / n_spine) if n_spine else np.nan,
                "dominant_missingness_reason": reason_mode,
                "impute_eligible": reason_mode == "ORDINARY",
            }
        )

    return panel, pd.DataFrame(miss_rows)


def write_partitioned_panel(
    panel: pd.DataFrame,
    out_root: Path,
    year: int,
    quarter: int,
) -> Path:
    """Write one quarter parquet under year=/quarter= hive layout."""
    dest = out_root / f"year={year}" / f"quarter={quarter}"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "part.parquet"
    # hard ban on forbidden columns
    bad = [c for c in panel.columns if c in FORBIDDEN_OUTPUT_COLUMNS]
    if bad:
        raise RuntimeError(f"Forbidden label columns in panel: {bad}")
    panel.to_parquet(path, index=False)
    return path


def compute_feature_coverage(
    inventory: pd.DataFrame,
    spine: pd.DataFrame,
    aligned_root: Path,
) -> pd.DataFrame:
    """Aggregate coverage from written aligned_raw partitions."""
    inv = inventory.set_index("factor")
    factors = [f for f in inv.index if bool(inv.loc[f, "materialized"])]
    spine_dates = spine.groupby("TradeDate").size().rename("n_spine")

    # accumulate non-null counts from partitions
    nn_by_factor_date: Dict[str, Dict[pd.Timestamp, int]] = {f: {} for f in factors}
    total_nn = {f: 0 for f in factors}

    parts = sorted(aligned_root.glob("year=*/quarter=*/part.parquet"))
    for part in parts:
        df = pd.read_parquet(part)
        cols = [c for c in factors if c in df.columns]
        if not cols:
            continue
        g = df.groupby("TradeDate")[cols].apply(lambda x: x.notna().sum())
        for factor in cols:
            series = g[factor] if isinstance(g, pd.DataFrame) else g
            for dt, val in series.items():
                dt = pd.Timestamp(dt).normalize()
                nn_by_factor_date[factor][dt] = int(val)
                total_nn[factor] += int(val)

    rows = []
    n_spine_rows = len(spine)
    n_spine_dates = int(spine["TradeDate"].nunique())
    for factor in factors:
        by_dt = nn_by_factor_date[factor]
        dates_with = sum(1 for v in by_dt.values() if v > 0)
        date_cov = dates_with / n_spine_dates if n_spine_dates else 0.0
        sym_covs = []
        for dt, n_nn in by_dt.items():
            n_s = int(spine_dates.loc[dt]) if dt in spine_dates.index else 0
            if n_s > 0:
                sym_covs.append(n_nn / n_s)
        mean_sym = float(np.mean(sym_covs)) if sym_covs else 0.0
        rows.append(
            {
                "factor": factor,
                "family": inv.loc[factor, "family"],
                "n_aligned_nonnull": int(total_nn[factor]),
                "coverage_ratio": float(total_nn[factor] / n_spine_rows)
                if n_spine_rows
                else 0.0,
                "date_coverage": float(date_cov),
                "mean_symbol_coverage": mean_sym,
                "n_dates_with_data": int(dates_with),
                "n_spine_dates": n_spine_dates,
            }
        )
    return pd.DataFrame(rows)


def compute_family_coverage(feature_coverage: pd.DataFrame) -> pd.DataFrame:
    if feature_coverage.empty:
        return feature_coverage
    g = feature_coverage.groupby("family", as_index=False).agg(
        features=("factor", "count"),
        median_date_coverage=("date_coverage", "median"),
        median_symbol_coverage=("mean_symbol_coverage", "median"),
        median_coverage_ratio=("coverage_ratio", "median"),
    )
    return g


# ---------------------------------------------------------------------------
# Gate 5 — audits
# ---------------------------------------------------------------------------


def audit_key_integrity(panel_root: Path) -> pd.DataFrame:
    rows = []
    for part in sorted(panel_root.glob("year=*/quarter=*/part.parquet")):
        df = pd.read_parquet(part, columns=["TradeDate", "Symbol"])
        n = len(df)
        n_dup = int(df.duplicated(["TradeDate", "Symbol"]).sum())
        rows.append(
            {
                "part": str(part.relative_to(panel_root)),
                "n_rows": n,
                "n_duplicate_keys": n_dup,
                "pass": n_dup == 0,
            }
        )
    return pd.DataFrame(rows)


def audit_label_contamination(panel_root: Path) -> pd.DataFrame:
    rows = []
    for part in sorted(panel_root.glob("year=*/quarter=*/part.parquet")):
        cols = list(pd.read_parquet(part).columns)
        bad = [c for c in cols if c in FORBIDDEN_OUTPUT_COLUMNS]
        soft = [
            c
            for c in cols
            if any(tok in c.lower() for tok in FORBIDDEN_NAME_SUBSTRINGS)
            and c not in ("TradeDate", "Symbol")
        ]
        rows.append(
            {
                "part": str(part.relative_to(panel_root)),
                "n_columns": len(cols),
                "forbidden_exact": "|".join(bad),
                "forbidden_soft": "|".join(soft),
                "pass": len(bad) == 0,
            }
        )
    return pd.DataFrame(rows)


def audit_source_parity(
    inventory: pd.DataFrame,
    aligned_root: Path,
    factors: Sequence[Tuple[str, str]] = PARITY_FACTORS,
    max_dates: int = 20,
) -> pd.DataFrame:
    """Compare aligned_raw vs source factor_narrow on common keys."""
    inv = inventory.set_index("factor")
    rows = []
    parts = sorted(aligned_root.glob("year=*/quarter=*/part.parquet"))
    if not parts:
        return pd.DataFrame(rows)

    import pyarrow.parquet as pq

    sample_frames = []
    for part in parts[:4]:
        schema_names = set(pq.read_schema(part).names)
        use_cols = ["TradeDate", "Symbol"] + [
            f for f, _ in factors if f in schema_names
        ]
        sample_frames.append(pd.read_parquet(part, columns=use_cols))
    aligned = pd.concat(sample_frames, ignore_index=True)
    # subsample dates
    dates = sorted(aligned["TradeDate"].unique())[:max_dates]
    aligned = aligned[aligned["TradeDate"].isin(dates)]

    for factor, _fam in factors:
        if factor not in inv.index or not bool(inv.loc[factor, "materialized"]):
            rows.append(
                {
                    "factor": factor,
                    "pass": False,
                    "reason": "not_materialized",
                }
            )
            continue
        if factor not in aligned.columns:
            rows.append(
                {
                    "factor": factor,
                    "pass": False,
                    "reason": "missing_in_aligned_sample",
                }
            )
            continue
        path = Path(str(inv.loc[factor, "data_source_path"]))
        src = load_factor_narrow_slice(path, min(dates), max(dates))
        src = src.rename(columns={"value": "src_value"})
        m = aligned[["TradeDate", "Symbol", factor]].merge(
            src, on=["TradeDate", "Symbol"], how="inner"
        )
        m = m.dropna(subset=[factor, "src_value"])
        if m.empty:
            rows.append(
                {
                    "factor": factor,
                    "n_common": 0,
                    "max_abs_diff": np.nan,
                    "rank_corr": np.nan,
                    "pass": False,
                    "reason": "no_common_nonnull",
                }
            )
            continue
        diff = (m[factor].astype(float) - m["src_value"].astype(float)).abs()
        # per-date rank corr then mean
        rho = []
        for _, g in m.groupby("TradeDate"):
            if len(g) < 30:
                continue
            rho.append(g[factor].corr(g["src_value"], method="spearman"))
        max_diff = float(diff.max())
        mean_rho = float(np.nanmean(rho)) if rho else float("nan")
        ok = max_diff < 1e-8 and (np.isnan(mean_rho) or mean_rho > 0.999)
        rows.append(
            {
                "factor": factor,
                "n_common": int(len(m)),
                "max_abs_diff": max_diff,
                "rank_corr": mean_rho,
                "pass": bool(ok),
                "reason": "" if ok else "parity_mismatch",
            }
        )
    return pd.DataFrame(rows)


def write_panel_schema(out_dir: Path, extra: Optional[Dict] = None) -> Path:
    payload = dict(PANEL_SCHEMA)
    if extra:
        payload.update(extra)
    path = out_dir / "panel_schema.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
