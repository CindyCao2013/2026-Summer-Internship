#!/usr/bin/env python
"""Sprint 6A Phase 3 — unified Candidate Pool baseline for the frozen
DolphinDB reference snapshot family (5 formulas, daily_mean candidates).

Mirrors expand_order_book_family.py conventions: T+1 signal, same masks,
benchmark-relative returns, identical summary fields; adds the required
intra-family and vs-Order-Book-family correlation evidence.

不做：参数优化、周频优化、中性化搜索、组合、KEEP/DROP。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.config.settings import RESULT_ROOT  # noqa: E402
from l2_factor_reproduction.python.backtest import (  # noqa: E402
    _save_backtest_outputs,
    backtest_factor,
    load_backtest_context,
)
from l2_factor_reproduction.python.candidate_pool import (  # noqa: E402
    correlation_pairs,
    decile_monotonicity,
    load_rank_ic,
    mean_daily_cross_sectional_spearman,
    redundancy_annotations,
    stability_fields,
    yearly_ic_table,
)
from l2_factor_reproduction.python.candidate_pool_registry import (  # noqa: E402
    BASELINE_POLICY,
)
from l2_factor_reproduction.python.ch_ddb_snapshot import (  # noqa: E402
    CLIP_GUARDS,
    FORMULA_VERSION,
    SCHEMA_VERSION,
    SMALL_DENOMINATOR_EPS,
)
from l2_factor_reproduction.python.ddb_snapshot_factors import (  # noqa: E402
    DDB_SNAPSHOT_FACTOR_NAMES,
    DDB_SNAPSHOT_FACTOR_SPECS,
    registry_frame,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_backtest_meta(start: pd.Timestamp, end: pd.Timestamp,
                            narrow_path: Path) -> Dict[str, object]:
    return {
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "signal_shift": 1,
        "benchmark": BASELINE_POLICY["benchmark"],
        "factor_narrow_sha256": _sha256(narrow_path),
        "formula_version": FORMULA_VERSION,
        "primitive_schema_version": SCHEMA_VERSION,
    }


def _backtest_meta_valid(
    directory: Path, expected: Dict[str, object]
) -> bool:
    meta_path = directory / "backtest_meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mismatches = [
        key for key, value in expected.items() if meta.get(key) != value
    ]
    if mismatches:
        print(
            f"[reuse-reject] {directory.name}: meta mismatch {mismatches}",
            flush=True,
        )
        return False
    return True

DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-07-31")
POOL_DIR = (
    Path(RESULT_ROOT) / "candidate_pool_v1" / "ddb_reference_snapshot_family"
)
FACTOR_ROOT = POOL_DIR / "factors"
POOL_ROOT = Path(RESULT_ROOT) / "candidate_pool_v1"
ORDER_BOOK_POOL = POOL_ROOT / "order_book_family"
PRIMITIVE_DIR = (
    Path(RESULT_ROOT) / "primitives" / "ddb_reference_snapshot"
)

# peer families for cross-family redundancy evidence:
# (label, factor narrow directory root, external flat layout)
PEER_FAMILIES: Tuple[Tuple[str, Path, bool], ...] = (
    ("order_book", ORDER_BOOK_POOL / "factors", False),
    ("trade_flow", Path(RESULT_ROOT), True),
    ("order_size", Path(RESULT_ROOT), True),
    ("price_formation", POOL_ROOT / "price_formation_family" / "factors", False),
)


def _parse_names(value: str) -> List[str]:
    if value.strip().lower() in {"", "all"}:
        return list(DDB_SNAPSHOT_FACTOR_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(DDB_SNAPSHOT_FACTOR_SPECS))
    if unknown:
        raise ValueError(f"Unknown DDB snapshot factors: {unknown}")
    return names


def _write_registry(names: List[str]) -> None:
    registry = registry_frame(names)
    registry.to_csv(POOL_DIR / "factor_registry.csv", index=False)
    (POOL_DIR / "factor_registry.json").write_text(
        json.dumps(
            registry.to_dict("records"), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def _run_or_reuse(
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    mask: pd.DataFrame,
    ret_matrix: pd.DataFrame,
    force: bool,
) -> Tuple[Dict[str, object], pd.Series]:
    output = FACTOR_ROOT / name
    summary_path = output / "summary.json"
    rank_ic_path = output / "rank_ic.csv"
    narrow_path = output / "factor_narrow.parquet"
    expected_meta = _expected_backtest_meta(start, end, narrow_path)
    if (
        not force
        and summary_path.exists()
        and rank_ic_path.exists()
        and _backtest_meta_valid(output, expected_meta)
    ):
        print(f"[backtest] reuse {name} (meta verified)", flush=True)
        return (
            json.loads(summary_path.read_text(encoding="utf-8")),
            load_rank_ic(rank_ic_path),
        )
    if not force:
        print(
            f"[backtest] {name}: no verified meta -> full rerun",
            flush=True,
        )
    narrow = pd.read_parquet(
        narrow_path,
        columns=["symbol", "tradetime", "value"],
    )
    group_pnl, group_turnover, rank_ic, summary = backtest_factor(
        narrow,
        start_day=start,
        end_day=end,
        signal_shift=1,
        mask=mask,
        ret_matrix=ret_matrix,
    )
    summary["net_annu_after_fee"] = (
        float(summary["hl_annu_ret_flipped"])
        - float(summary["implied_annu_fee"])
    )
    _save_backtest_outputs(
        str(output), group_pnl, group_turnover, rank_ic, summary,
        factor_name=name,
    )
    meta = dict(expected_meta)
    meta["created_at"] = datetime.now().isoformat(timespec="seconds")
    (output / "backtest_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del narrow
    gc.collect()
    return summary, rank_ic


def _load_narrow_wide(names: List[str], start, end) -> pd.DataFrame:
    """symbol x TradeDate wide frame of the given family factors."""
    wide = None
    for name in names:
        narrow = pd.read_parquet(
            FACTOR_ROOT / name / "factor_narrow.parquet",
            columns=["symbol", "tradetime", "value"],
        )
        block = narrow.rename(columns={"value": name})
        block["TradeDate"] = pd.to_datetime(block["tradetime"]).dt.normalize()
        block = block[["symbol", "TradeDate", name]]
        wide = (
            block
            if wide is None
            else wide.merge(
                block, on=["symbol", "TradeDate"], how="outer",
                validate="one_to_one",
            )
        )
        del narrow, block
        gc.collect()
    mask = wide["TradeDate"].between(start, end)
    return wide.loc[mask].reset_index(drop=True)


def _daily_spearman_fast(
    frame: pd.DataFrame, left: str, right: str, min_names: int = 100
) -> pd.Series:
    """Vectorized mean daily cross-sectional Spearman between two columns."""
    pair = frame[["TradeDate", left, right]].dropna()
    if pair.empty:
        return pd.Series(dtype=float)
    grouped = pair.groupby("TradeDate", sort=True)
    work = pair[["TradeDate"]].copy()
    work["ra"] = grouped[left].rank()
    work["rb"] = grouped[right].rank()
    work["rarb"] = work["ra"] * work["rb"]
    work["ra2"] = work["ra"] * work["ra"]
    work["rb2"] = work["rb"] * work["rb"]
    agg = work.groupby("TradeDate").agg(
        n=("ra", "size"),
        sa=("ra", "sum"),
        sb=("rb", "sum"),
        saa=("ra2", "sum"),
        sbb=("rb2", "sum"),
        sab=("rarb", "sum"),
    )
    n = agg["n"]
    cov = agg["sab"] - agg["sa"] * agg["sb"] / n
    va = agg["saa"] - agg["sa"] ** 2 / n
    vb = agg["sbb"] - agg["sb"] ** 2 / n
    rho = cov / np.sqrt(va * vb)
    rho[(n < min_names) | (va <= 0) | (vb <= 0)] = np.nan
    return rho.dropna()


def _peer_factor_dirs(
    label: str, root: Path, external: bool
) -> List[Tuple[str, Path]]:
    """(factor_name, narrow parquet path) for one peer family.

    Missing family root / registry must not fail the run: warn and skip.
    """
    if external:
        registry = POOL_ROOT / f"{label}_family" / "factor_registry.csv"
        if not registry.exists():
            warnings.warn(
                f"peer family {label}: registry {registry} missing; skipped"
            )
            return []
        names = pd.read_csv(registry)["name"].tolist()
        return [
            (name, root / name / "factor_narrow.parquet")
            for name in names
            if (root / name / "factor_narrow.parquet").exists()
        ]
    if not root.is_dir():
        warnings.warn(f"peer family {label}: root {root} missing; skipped")
        return []
    return [
        (p.name, p / "factor_narrow.parquet")
        for p in sorted(root.iterdir())
        if (p / "factor_narrow.parquet").exists()
    ]


def _cross_family_correlation(
    wide_new: pd.DataFrame, names: List[str], start, end
) -> pd.DataFrame:
    """Each new factor vs frozen peers in Order Book / Trade Flow /
    Order Size / Price Formation (mean daily cross-sectional Spearman).

    Year-blocked merges: each peer narrow is sliced per calendar year
    before merging, so no full-history wide merge is built per peer.
    The realized peer list is saved to cross_family_peer_selection.csv.
    """
    wide = wide_new.copy()
    wide["year"] = wide["TradeDate"].dt.year
    year_blocks = {
        int(year): block.drop(columns="year")
        for year, block in wide.groupby("year")
    }
    del wide
    rows = []
    peer_selection = []
    for label, root, external in PEER_FAMILIES:
        peers = _peer_factor_dirs(label, root, external)
        print(f"[corr] cross-family vs {len(peers)} {label} factors", flush=True)
        for position, (peer, path) in enumerate(peers, start=1):
            narrow = pd.read_parquet(
                path, columns=["symbol", "tradetime", "value"]
            )
            block = narrow.rename(columns={"value": peer})
            block["TradeDate"] = pd.to_datetime(
                block["tradetime"]
            ).dt.normalize()
            block = block.loc[block["TradeDate"].between(start, end)]
            block["year"] = block["TradeDate"].dt.year
            peer_years = {
                int(year): part.drop(columns="year")
                for year, part in block.groupby("year")
            }
            del narrow, block
            rho_parts: Dict[str, List[pd.Series]] = {
                name: [] for name in names
            }
            for year, wblock in year_blocks.items():
                pblock = peer_years.get(year)
                if pblock is None:
                    continue
                merged = wblock.merge(
                    pblock, on=["symbol", "TradeDate"], how="inner"
                )
                for name in names:
                    rho_parts[name].append(
                        _daily_spearman_fast(merged, name, peer)
                    )
                del merged
            peer_selection.append(
                {"peer_family": label, "peer_factor": peer, "path": str(path)}
            )
            for name in names:
                parts = [s for s in rho_parts[name] if len(s)]
                rho = (
                    pd.concat(parts) if parts else pd.Series(dtype=float)
                )
                rows.append(
                    {
                        "ddb_snapshot_factor": name,
                        "peer_family": label,
                        "peer_factor": peer,
                        "mean_daily_spearman": float(rho.mean())
                        if len(rho)
                        else float("nan"),
                        "abs_mean_daily_spearman": float(rho.abs().mean())
                        if len(rho)
                        else float("nan"),
                        "n_days": int(len(rho)),
                    }
                )
            del peer_years
            gc.collect()
            if position % 16 == 0:
                print(f"  [corr] {label} {position}/{len(peers)}", flush=True)
    pd.DataFrame(peer_selection).to_csv(
        POOL_DIR / "cross_family_peer_selection.csv", index=False
    )
    return pd.DataFrame(rows)


def _criterion_list(frame: pd.DataFrame, expression: pd.Series) -> str:
    names = frame.loc[expression.fillna(False), "factor"].tolist()
    return ", ".join(f"`{name}`" for name in names) if names else "无"


TWOS_NAME = "time_weighted_order_slope"


def _winsorize_daily(narrow: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Cross-sectional clip at q / 1-q per tradetime (audit variant only)."""
    out = narrow.copy()
    bounds = out.groupby("tradetime")["value"].quantile([q, 1 - q]).unstack()
    bounds.columns = ["lo", "hi"]
    out = out.merge(bounds, left_on="tradetime", right_index=True, how="left")
    out["value"] = out["value"].clip(out["lo"], out["hi"])
    return out[["symbol", "tradetime", "value"]]


def _decile_members(
    frame: pd.DataFrame, value_col: str, quantile: float, top: bool
) -> Dict[pd.Timestamp, set]:
    """Per-day symbol sets of the extreme decile by `value_col`."""
    ranks = frame.groupby("tradetime")[value_col].rank(pct=True)
    selected = ranks >= quantile if top else ranks <= 1 - quantile + 1e-12
    picked = frame.loc[selected.to_numpy(), ["tradetime", "symbol"]]
    return {
        day: set(block["symbol"])
        for day, block in picked.groupby("tradetime")
    }


def _member_overlap(
    left: Dict[pd.Timestamp, set], right: Dict[pd.Timestamp, set]
) -> float:
    shares = []
    for day, symbols in left.items():
        other = right.get(day)
        if other and symbols:
            shares.append(len(symbols & other) / len(symbols))
    return float(np.mean(shares)) if shares else float("nan")


def _load_twos_diagnostics(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Primitive snapshot-level diagnostics for the twos audit.

    Hard failure on empty glob, missing columns, or schema mismatch —
    no silent skip, no pd.concat([]).
    """
    paths = sorted(
        (PRIMITIVE_DIR / "dataset").glob("year=*/ddb_snapshot_daily_*.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"no ddb_snapshot_daily chunks under {PRIMITIVE_DIR / 'dataset'}"
        )
    required = {
        "TradeDate",
        "valid_snapshot_count",
        f"{TWOS_NAME}_clipped_count",
        f"{TWOS_NAME}_small_denominator_count",
    }
    frames = []
    for path in paths:
        schema_names = set(pq.read_schema(path).names)
        missing = required - schema_names
        if missing:
            raise ValueError(
                f"primitive chunk {path.name} missing diagnostic columns: "
                f"{sorted(missing)}"
            )
        frames.append(pd.read_parquet(path, columns=sorted(required)))
    diag = pd.concat(frames, ignore_index=True)
    return diag.loc[diag["TradeDate"].between(start, end)].reset_index(
        drop=True
    )


def _twos_stability_audit(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    mask: pd.DataFrame,
    ret_matrix: pd.DataFrame,
) -> None:
    """Numerical-instability audit for timeWeightedOrderSlope.

    raw vs winsorized(1%/99%) vs trimmed: RankIC, H-L Sharpe, G10 Excess
    Sharpe; raw-vs-clipped daily rank correlation; G1/G10 member overlap;
    extreme-tail |value| mass share (overall and per year); clip share by
    year; cross-sectional std ratio. Distinguishes rank-based usability
    from raw-value (linear) usability. Frozen guard: the original
    formula's small-denominator issue must stay visible even if clipped
    preprocessing looks better.
    """
    # Terminology (frozen): the factor narrow is the SQL-guarded output of
    # the official unbounded formula (division-by-zero nulled via
    # nullIf(denominator, 0); small denominators <1e-6 and |x|>1e6 are
    # monitored-only, never modified). It must NOT be called "raw".
    # winsor/trimmed are analysis-only cross-sectional variants.
    narrow = pd.read_parquet(
        FACTOR_ROOT / TWOS_NAME / "factor_narrow.parquet",
        columns=["symbol", "tradetime", "value"],
    )
    narrow["tradetime"] = pd.to_datetime(narrow["tradetime"])

    clipped = _winsorize_daily(narrow)
    variants = {"guarded": narrow, "winsor": clipped}
    rank_ics: Dict[str, pd.Series] = {}
    summaries: Dict[str, Dict[str, object]] = {}
    for label, frame in variants.items():
        _, _, ic, summ = backtest_factor(
            frame, start_day=start, end_day=end, signal_shift=1,
            mask=mask, ret_matrix=ret_matrix,
        )
        direction = int(summ.get("factor_direction", 1))
        rank_ics[label] = ic * direction
        summaries[label] = summ

    pair = narrow.merge(
        clipped.rename(columns={"value": "value_clipped"}),
        on=["symbol", "tradetime"], validate="one_to_one",
    )
    rank_corr = pair.groupby("tradetime").apply(
        lambda g: g["value"].corr(g["value_clipped"], method="spearman")
        if len(g) >= 100 else np.nan
    ).dropna()

    per_day_std = pair.groupby("tradetime").apply(
        lambda g: pd.Series(
            {
                "std_raw": g["value"].std(),
                "std_clipped": g["value_clipped"].std(),
            }
        )
    )
    std_ratio = (per_day_std["std_clipped"] / per_day_std["std_raw"]).mean()

    daily = pair.groupby("tradetime")["value"]
    lo = daily.quantile(0.01)
    hi = daily.quantile(0.99)
    keyed = pair.set_index("tradetime")
    lo_v = keyed.index.to_series().map(lo).to_numpy()
    hi_v = keyed.index.to_series().map(hi).to_numpy()
    keyed["is_extreme"] = (keyed["value"].to_numpy() < lo_v) | (
        keyed["value"].to_numpy() > hi_v
    )
    clip_share_year = (
        keyed.groupby(keyed.index.year)["is_extreme"].mean().rename("clip_share")
    )
    # extreme-tail share of cross-sectional absolute mass
    abs_mass = keyed["value"].abs()
    abs_extreme = abs_mass.loc[keyed["is_extreme"]]
    extreme_mass_share = float(abs_extreme.sum() / abs_mass.sum())
    extreme_mass_year = (
        abs_extreme.groupby(abs_extreme.index.year).sum()
        / abs_mass.groupby(keyed.index.year).sum()
    ).rename("extreme_abs_mass_share")

    trimmed = pair.loc[
        ~keyed["is_extreme"].to_numpy(), ["symbol", "tradetime", "value"]
    ]
    _, _, ic_trim, summary_trim = backtest_factor(
        trimmed, start_day=start, end_day=end, signal_shift=1,
        mask=mask, ret_matrix=ret_matrix,
    )
    rank_ics["trimmed"] = ic_trim * int(summary_trim.get("factor_direction", 1))
    summaries["trimmed"] = summary_trim

    # G1/G10 member overlap between variants
    g10_raw = _decile_members(narrow, "value", 0.9, top=True)
    g1_raw = _decile_members(narrow, "value", 0.9, top=False)
    g10_winsor = _decile_members(clipped, "value", 0.9, top=True)
    g1_winsor = _decile_members(clipped, "value", 0.9, top=False)
    g10_trim = _decile_members(trimmed, "value", 0.9, top=True)
    g1_trim = _decile_members(trimmed, "value", 0.9, top=False)
    overlaps = {
        "g10_overlap_raw_vs_winsor": _member_overlap(g10_raw, g10_winsor),
        "g1_overlap_raw_vs_winsor": _member_overlap(g1_raw, g1_winsor),
        "g10_overlap_raw_vs_trimmed": _member_overlap(g10_raw, g10_trim),
        "g1_overlap_raw_vs_trimmed": _member_overlap(g1_raw, g1_trim),
    }

    diag_all = _load_twos_diagnostics(start, end)
    by_year = diag_all.groupby(diag_all["TradeDate"].dt.year).agg(
        snapshots=("valid_snapshot_count", "sum"),
        clipped=(f"{TWOS_NAME}_clipped_count", "sum"),
        small_den=(f"{TWOS_NAME}_small_denominator_count", "sum"),
    )
    by_year["sql_guard_clip_share"] = by_year["clipped"] / by_year["snapshots"]
    by_year["small_den_share"] = by_year["small_den"] / by_year["snapshots"]
    by_year["winsor_clip_share_cs"] = clip_share_year
    by_year = by_year.join(extreme_mass_year, how="left")
    by_year.to_csv(POOL_DIR / "twos_clip_share_by_year.csv")

    def _metric(label: str, key: str) -> float:
        return float(summaries[label].get(key, float("nan")))

    rank_usable = bool(rank_corr.mean() >= 0.99)
    linear_usable = bool(
        std_ratio >= 0.5 and extreme_mass_share <= 0.20
    )
    audit = pd.DataFrame(
        [
            {
                "output_tier": "guarded_formula_output",
                "guard_spec": (
                    "nullIf(denominator,0) in SQL; small_denominator "
                    f"eps={SMALL_DENOMINATOR_EPS} monitored-only; clip cap "
                    f"|x|>{CLIP_GUARDS[TWOS_NAME]:.0e} monitored-only"
                ),
                "small_denominator_eps": SMALL_DENOMINATOR_EPS,
                "clip_guard_threshold": CLIP_GUARDS[TWOS_NAME],
                "rank_ic_guarded": float(rank_ics["guarded"].mean()),
                "rank_ic_winsorized_1pct": float(rank_ics["winsor"].mean()),
                "rank_ic_trimmed_extreme_1pct": float(rank_ics["trimmed"].mean()),
                "ic_delta_clip_minus_guarded": float(
                    rank_ics["winsor"].mean() - rank_ics["guarded"].mean()
                ),
                "ic_delta_guarded_minus_trimmed": float(
                    rank_ics["guarded"].mean() - rank_ics["trimmed"].mean()
                ),
                "hl_sharpe_guarded": _metric("guarded", "hl_sharpe_flipped"),
                "hl_sharpe_winsor": _metric("winsor", "hl_sharpe_flipped"),
                "hl_sharpe_trimmed": _metric("trimmed", "hl_sharpe_flipped"),
                "g10_excess_sharpe_guarded": _metric("guarded", "g10_excess_sharpe"),
                "g10_excess_sharpe_winsor": _metric("winsor", "g10_excess_sharpe"),
                "g10_excess_sharpe_trimmed": _metric("trimmed", "g10_excess_sharpe"),
                **overlaps,
                "extreme_1pct_abs_value_mass_share": extreme_mass_share,
                "daily_rank_corr_raw_vs_clipped_mean": float(rank_corr.mean()),
                "daily_rank_corr_raw_vs_clipped_q05": float(
                    rank_corr.quantile(0.05)
                ),
                "cs_std_ratio_winsor_over_raw": float(std_ratio),
                "rank_based_usable": rank_usable,
                "raw_value_linear_usable": linear_usable,
                "n_days": int(len(rank_corr)),
            }
        ]
    )
    audit.to_csv(POOL_DIR / "twos_stability_audit.csv", index=False)
    print(
        "[audit] twos guarded IC={:.4f} winsor IC={:.4f} trimmed IC={:.4f} "
        "rank_corr={:.4f} std_ratio={:.3f} extreme_mass={:.2%} "
        "rank_usable={} linear_usable={}".format(
            rank_ics["guarded"].mean(), rank_ics["winsor"].mean(),
            rank_ics["trimmed"].mean(), rank_corr.mean(), std_ratio,
            extreme_mass_share, rank_usable, linear_usable,
        ),
        flush=True,
    )


def _twos_report_block() -> str:
    path = POOL_DIR / "twos_stability_audit.csv"
    if not path.exists():
        return "（twos_stability_audit.csv 缺失，待 baseline 运行生成）"
    audit = pd.read_csv(path).iloc[0]
    year_path = POOL_DIR / "twos_clip_share_by_year.csv"
    year_text = ""
    if year_path.exists():
        by_year = pd.read_csv(year_path)
        year_text = (
            "\n- 分年 clip share（SQL guard / winsor 截面）与 small_den：\n\n"
            "```\n"
            + by_year.to_string(index=False)
            + "\n```"
        )
    return (
        "- 术语分层：official_unbounded_formula（官方无界公式）→ "
        "guarded_formula_output（= factor_narrow；SQL 内仅 nullIf(分母,0)，"
        f"small_denominator eps={audit['small_denominator_eps']:.0e} 与 "
        f"clip 上限 |x|>{audit['clip_guard_threshold']:.0e} 只计数不改值）→ "
        "cross_sectionally_winsorized_output（日截面 1%/99%）→ "
        "trimmed_output（剔除双侧 1% 极端）。**narrow 不是 raw official "
        "formula**；分年触发率见 twos_clip_share_by_year.csv\n"
        f"- RankIC：guarded {audit['rank_ic_guarded']:+.4f} → winsorize "
        f"{audit['rank_ic_winsorized_1pct']:+.4f}（Δ {audit['ic_delta_clip_minus_guarded']:+.4f}）；"
        f"trimmed {audit['rank_ic_trimmed_extreme_1pct']:+.4f}"
        f"（极端值贡献 {audit['ic_delta_guarded_minus_trimmed']:+.4f}）\n"
        f"- H-L Sharpe：guarded {audit['hl_sharpe_guarded']:+.2f} / winsor "
        f"{audit['hl_sharpe_winsor']:+.2f} / trimmed {audit['hl_sharpe_trimmed']:+.2f}\n"
        f"- G10 Excess Sharpe：guarded {audit['g10_excess_sharpe_guarded']:+.2f} / "
        f"winsor {audit['g10_excess_sharpe_winsor']:+.2f} / "
        f"trimmed {audit['g10_excess_sharpe_trimmed']:+.2f}\n"
        f"- G10/G1 成员重合率 guarded↔winsor："
        f"{audit['g10_overlap_raw_vs_winsor']:.1%} / "
        f"{audit['g1_overlap_raw_vs_winsor']:.1%}；"
        f"guarded↔trimmed：{audit['g10_overlap_raw_vs_trimmed']:.1%} / "
        f"{audit['g1_overlap_raw_vs_trimmed']:.1%}\n"
        f"- 双侧 1% 极端值占 |factor| 总质量比例："
        f"{audit['extreme_1pct_abs_value_mass_share']:.1%}"
        f"（分年贡献见 twos_clip_share_by_year.csv）\n"
        f"- guarded vs winsor 日截面 rank 相关：mean "
        f"{audit['daily_rank_corr_raw_vs_clipped_mean']:.4f}"
        f"（q05 {audit['daily_rank_corr_raw_vs_clipped_q05']:.4f}）\n"
        f"- winsorize 前后截面标准差比："
        f"{audit['cs_std_ratio_winsor_over_raw']:.4f}\n"
        f"- 可用性判定：rank-based（排序/分位用法）= "
        f"{'可用' if audit['rank_based_usable'] else '受损'}；"
        f"raw-value 线性用法 = "
        f"{'可用' if audit['raw_value_linear_usable'] else '受损（小分母极端值主导线性暴露，禁止线性zscore/回归直接使用 guarded 值）'}"
        + year_text
    )


def _obi_reference_factors() -> List[str]:
    """Canonical static-OBI reference factors, resolved dynamically from
    the order_book factor registry (no hard-coded existence assumption)."""
    canonical = [
        "obi_l1_mean", "obi_l5_mean", "obi_l10_mean", "weighted_obi_mean",
    ]
    registry = ORDER_BOOK_POOL / "factor_registry.csv"
    if not registry.exists():
        return []
    names = pd.read_csv(registry)["name"].tolist()
    return [name for name in canonical if name in names]


def _write_report(summary: pd.DataFrame) -> None:
    wanted_columns = [
        "factor", "rank_ic_raw", "icir_raw", "g10_excess_sharpe",
        "hl_sharpe", "avg_hl_turnover", "net_annu_after_fee",
        "sign_consistency", "decile_mono_spearman",
        "redundancy_cluster_080",
    ]
    display_columns = [c for c in wanted_columns if c in summary.columns]
    view = summary[display_columns]
    try:
        table = view.to_markdown(index=False)
    except ImportError:
        table = "```\n" + view.to_string(index=False) + "\n```"

    def _over_three(column: str) -> pd.Series:
        if column not in summary.columns:
            return pd.Series(False, index=summary.index)
        return summary[column] > 3

    g10_over_three = _over_three("g10_excess_sharpe")
    hl_over_three = _over_three("hl_sharpe")

    cross_path = POOL_DIR / "cross_family_correlation.csv"
    soir_verdict = "（cross_family_correlation.csv 缺失，待补）"
    dynamics_verdict = ""
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        soir = cross.loc[
            (cross["ddb_snapshot_factor"] == "wavg_soir")
            & (cross["peer_factor"] == "weighted_obi_mean")
        ]
        if not len(soir):
            soir_verdict = (
                "cross_family_correlation.csv 已生成，但 wavg_soir vs "
                "weighted_obi_mean 行缺失（见 cross_family_peer_selection.csv "
                "核对 peer 覆盖）"
            )
        if len(soir):
            rho = float(soir["mean_daily_spearman"].iloc[0])
            soir_verdict = (
                f"`wavg_soir` vs `weighted_obi_mean` 整样本日截面 Spearman "
                f"均值 = {rho:.4f}"
                + (
                    "（≥0.8，登记为 weighted OBI 的近别名，不计独立 alpha）"
                    if abs(rho) >= 0.8
                    else "（<0.8，prev 窗口标准化显著改变信号形态，"
                         "不按别名处理，但仍为冗余对照候选）"
                )
            )
        rows = []
        for probe, label in [
            ("level10_diff_buy", "level10_Diff"),
            ("level10_infer_price_trend", "inferPriceTrend"),
        ]:
            sub = cross.loc[cross["ddb_snapshot_factor"] == probe].copy()
            if not len(sub):
                continue
            sub["abs_rho"] = sub["mean_daily_spearman"].abs()
            top = sub.nlargest(3, "abs_rho")
            top_text = "; ".join(
                f"{r.peer_family}/{r.peer_factor}={r.mean_daily_spearman:+.3f}"
                for r in top.itertuples()
            )
            max_abs = float(sub["abs_rho"].max())
            independent = "是" if max_abs < 0.8 else "否（存在高相关 peer）"
            rows.append(
                f"- {label}：全 peer 最大 |ρ| = {max_abs:.3f} "
                f"（独立新维度：{independent}）；top3 相关：{top_text}"
            )
        dynamics_verdict = "\n".join(rows)

    obi_refs = _obi_reference_factors()
    ob_summary_path = ORDER_BOOK_POOL / "candidate_summary.csv"
    ob_note = ""
    obi_note = "实际参与比较的静态 OBI 名单：" + (
        ", ".join(obi_refs) if obi_refs else "（order_book registry 缺失）"
    )
    if ob_summary_path.exists():
        ob = pd.read_csv(ob_summary_path)
        ob_obi = ob.loc[ob["factor"].isin(obi_refs)]
        probes = {
            probe: summary.loc[summary["factor"] == probe]
            for probe in ["level10_diff_buy", "level10_infer_price_trend"]
        }
        if len(ob_obi) and all(len(frame) for frame in probes.values()):
            l10 = probes["level10_diff_buy"].iloc[0]
            ipt = probes["level10_infer_price_trend"].iloc[0]
            best_obi = ob_obi["rank_ic_raw"].abs().max()
            ob_note = (
                f"- {obi_note}\n"
                f"- level10_diff_buy |RankIC|={abs(l10['rank_ic_raw']):.4f} "
                f"vs 静态 OBI 最优 |RankIC|={best_obi:.4f}\n"
                f"- level10_infer_price_trend |RankIC|="
                f"{abs(ipt['rank_ic_raw']):.4f} vs {best_obi:.4f}\n"
                f"- 二者 G10 Excess Sharpe: {l10['g10_excess_sharpe']:.2f} / "
                f"{ipt['g10_excess_sharpe']:.2f}（OBI 最优 "
                f"{ob_obi['g10_excess_sharpe'].max():.2f}）"
            )
        elif len(ob_obi):
            ob_note = (
                f"- {obi_note}\n"
                "- （--factors 子集运行，level10 探针不在本批，跳过对比）"
            )

    lines = [
        "# L2 Candidate Pool v1 — DolphinDB Reference Snapshot Family",
        "",
        "Sprint 6A：官方 DolphinDB §3.1 五个快照公式在公司 ClickHouse 上的"
        "忠实复现（golden 对照通过，见 primitives/ddb_reference_snapshot/"
        "validation）。未做参数优化、二次中性化、周度优化、组合或 KEEP/DROP。",
        "",
        f"- 冻结公式数：{len(summary)}",
        "- IC/ICIR：冻结原方向；分组收益：统一 effective direction",
        "- 转换口径：3s 状态序列 → 分钟末值 → 240 分钟网格 → 日均",
        "",
        "## Unified baseline",
        "",
        table,
        "",
        "## Required questions",
        "",
        "1. **G10 Excess Sharpe > 3**："
        + _criterion_list(summary, g10_over_three)
        + "。",
        "",
        "2. **H-L Sharpe > 3**："
        + _criterion_list(summary, hl_over_three)
        + "。",
        "",
        "3. **wavgSOIR 是否为 weighted OBI 别名**：" + soir_verdict,
        "",
        "4. **level10_Diff / inferPriceTrend 是否形成独立新维度**（证据："
        "`cross_family_correlation.csv`、`factor_correlation_spearman.csv`）：",
        dynamics_verdict or "（cross-family 相关缺失，待补）",
        "",
        "5. **level10_Diff / inferPriceTrend vs 静态 OBI 有效性**：",
        ob_note or "（order_book_family summary 缺失，待补）",
        "",
        "6. **timeWeightedOrderSlope 数值不稳定审计**（冻结要求：不得因 "
        "clipped 预处理表现更好而忽略原公式小分母问题；小分母/极端值"
        "分位数另见 primitives/ddb_reference_snapshot/"
        "validation_*_twos_audit.csv）：",
        _twos_report_block(),
        "",
        "## Boundaries",
        "",
        "高 Sharpe、低相关或稳定 IC 均只是在冻结 baseline 下的证据。"
        "本报告不作正式 KEEP/DROP、生产晋级或组合结论。",
        "",
    ]
    (POOL_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument("--factors", default="all")
    parser.add_argument("--force-backtest", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-corr", action="store_true")
    args = parser.parse_args()

    names = _parse_names(args.factors)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        _write_report(pd.read_csv(POOL_DIR / "candidate_summary.csv"))
        print(f"[done] refreshed {POOL_DIR / 'report.md'}", flush=True)
        return 0
    _write_registry(names)
    coverage = pd.read_csv(POOL_DIR / "factor_coverage.csv").set_index(
        "factor"
    )
    print("[backtest] loading shared return/mask context once", flush=True)
    mask, ret_matrix = load_backtest_context(start, end)

    rows = []
    yearly_parts = []
    for position, name in enumerate(names, start=1):
        print(f"[backtest {position}/{len(names)}] {name}", flush=True)
        summary, rank_ic_effective = _run_or_reuse(
            name, start, end,
            mask=mask, ret_matrix=ret_matrix, force=args.force_backtest,
        )
        direction = int(summary.get("factor_direction", 1))
        rank_ic_raw = rank_ic_effective * direction
        yearly_raw = yearly_ic_table(rank_ic_raw)
        yearly_effective = yearly_ic_table(rank_ic_effective)
        factor_dir = FACTOR_ROOT / name
        yearly_raw.to_csv(factor_dir / "yearly_ic.csv")
        yearly_effective.to_csv(factor_dir / "yearly_ic_effective.csv")
        yearly_raw.assign(factor=name).reset_index().to_csv(
            factor_dir / "yearly_stability.csv", index=False
        )
        yearly_parts.append(yearly_raw.assign(factor=name).reset_index())

        raw_mean = float(summary.get("rank_ic_mean_raw", rank_ic_raw.mean()))
        raw_std = float(rank_ic_raw.std())
        raw_icir = (
            raw_mean / raw_std * np.sqrt(250) if raw_std > 0 else float("nan")
        )
        coverage_row = coverage.loc[name]
        row = {
            "factor": name,
            "category": DDB_SNAPSHOT_FACTOR_SPECS[name].category,
            "mechanism": DDB_SNAPSHOT_FACTOR_SPECS[name].mechanism,
            "lookback_days": DDB_SNAPSHOT_FACTOR_SPECS[name].lookback_days,
            "n_factor_rows": int(coverage_row["n_factor_rows"]),
            "date_min": coverage_row["date_min"],
            "date_max": coverage_row["date_max"],
            "n_symbols": int(coverage_row["n_symbols"]),
            "factor_direction": direction,
            "direction_flip": bool(direction < 0),
            "rank_ic_raw": raw_mean,
            "icir_raw": raw_icir,
            "rank_ic_std": raw_std,
            "positive_ic_fraction": float(
                (rank_ic_raw.dropna() > 0).mean()
            ),
            "rank_ic_effective": float(summary["rank_ic_mean"]),
            "icir_effective": float(summary["rank_icir"]),
            "hl_annu_ret": float(summary["hl_annu_ret_flipped"]),
            "hl_sharpe": float(summary["hl_sharpe_flipped"]),
            "g10_excess_annu_ret": float(summary["g10_excess_annu_ret"]),
            "g10_excess_sharpe": float(summary["g10_excess_sharpe"]),
            "hl_mdd": float(summary["hl_mdd_flipped"]),
            "avg_hl_turnover": float(summary["avg_hl_turnover"]),
            "implied_annu_fee": float(summary["implied_annu_fee"]),
            "net_annu_after_fee": (
                float(summary["hl_annu_ret_flipped"])
                - float(summary["implied_annu_fee"])
            ),
            "decile_mono_spearman": decile_monotonicity(summary),
            "n_days": int(summary["n_days"]),
            "n_names_avg": float(summary["n_names_avg"]),
            "group_pnl_saved_direction": summary[
                "group_pnl_saved_direction"
            ],
            **stability_fields(yearly_raw, raw_mean),
        }
        rows.append(row)
        print(
            f"[result] raw IC={raw_mean:+.4f} ICIR={raw_icir:+.2f} "
            f"G10={row['g10_excess_sharpe']:.2f} "
            f"H-L={row['hl_sharpe']:.2f}",
            flush=True,
        )

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(POOL_DIR / "candidate_summary.csv", index=False)
    pd.concat(yearly_parts, ignore_index=True).to_csv(
        POOL_DIR / "yearly_ic_all.csv", index=False
    )

    corr_outputs = [
        POOL_DIR / "factor_correlation_spearman.csv",
        POOL_DIR / "intra_family_correlation_pairs.csv",
        POOL_DIR / "redundancy_clusters_080.csv",
        POOL_DIR / "cross_family_correlation.csv",
    ]
    corr_reusable = all(p.exists() for p in corr_outputs)
    if not args.skip_corr and corr_reusable:
        print("[corr] reusing existing correlation outputs", flush=True)
        intra = pd.read_csv(corr_outputs[0], index_col=0)
        annotations = pd.read_csv(corr_outputs[2])
        summary_frame = summary_frame.merge(
            annotations[
                ["factor", "redundancy_cluster_080", "max_corr_peer",
                 "max_abs_corr", "near_alias_observed"]
            ],
            on="factor",
            how="left",
            validate="one_to_one",
        )
        summary_frame.to_csv(POOL_DIR / "candidate_summary.csv", index=False)
    elif not args.skip_corr:
        print("[corr] intra-family daily spearman", flush=True)
        wide_new = _load_narrow_wide(names, start, end)
        intra = mean_daily_cross_sectional_spearman(wide_new, names)
        intra.to_csv(POOL_DIR / "factor_correlation_spearman.csv")
        correlation_pairs(intra).to_csv(
            POOL_DIR / "intra_family_correlation_pairs.csv", index=False
        )
        annotations = redundancy_annotations(intra, threshold=0.80)
        annotations.to_csv(
            POOL_DIR / "redundancy_clusters_080.csv", index=False
        )
        summary_frame = summary_frame.merge(
            annotations[
                ["factor", "redundancy_cluster_080", "max_corr_peer",
                 "max_abs_corr", "near_alias_observed"]
            ],
            on="factor",
            how="left",
            validate="one_to_one",
        )
        summary_frame.to_csv(POOL_DIR / "candidate_summary.csv", index=False)

        cross = _cross_family_correlation(wide_new, names, start, end)
        cross.to_csv(POOL_DIR / "cross_family_correlation.csv", index=False)
        del wide_new
        gc.collect()

    if TWOS_NAME in names:
        print("[audit] timeWeightedOrderSlope stability", flush=True)
        _twos_stability_audit(start, end, mask=mask, ret_matrix=ret_matrix)

    _write_report(summary_frame)
    manifest = {
        "version": "ddb_snapshot_candidate_pool_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_start": str(start.date()),
        "sample_end": str(end.date()),
        "primitive": str(PRIMITIVE_DIR),
        "n_candidates": len(names),
        "factors": names,
        "signal_shift": 1,
        "cost_bps": 7.5,
        "group_direction": "effective",
        "rank_ic_direction": "raw frozen formula",
        "scope_exclusions": [
            "parameter_optimization",
            "factor_combination",
            "machine_learning",
            "weekly_rebalance_optimization",
            "second_neutralization",
            "cross_family_keep_drop",
        ],
    }
    (POOL_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] baseline -> {POOL_DIR}\n"
        + summary_frame[
            ["factor", "rank_ic_raw", "icir_raw",
             "g10_excess_sharpe", "hl_sharpe"]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
