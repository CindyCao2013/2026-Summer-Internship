"""Sidecar registry for generated cut candidates. Never writes candidate_pool_v1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from l2_factor_reproduction.l2_ai_stock_selection.cut_operators.contracts import (
    CANDIDATE_POOL_CSV,
    CUT_REGISTRY_NAME,
    CUT_RESULT_ROOT,
    REGISTRY_COLUMNS,
    time_segment,
)

CUT_TYPE_PREFIX = {
    "time": "time",
    "state": "state",
    "event": "event",
    "contrast": "contrast",
}


def candidate_name(
    base_primitive: str,
    cut_type: str,
    cut_name: str,
    aggregation: str = "",
    contrast_operator: str = "",
) -> str:
    """Machine-readable reconstructable name.

    <base>__<cut_type>_<cut_name>__<agg>
    <base>__contrast_<cut_name>
    """
    base = str(base_primitive).strip()
    ctype = CUT_TYPE_PREFIX[str(cut_type).strip().lower()]
    cname = str(cut_name).strip().lower().replace(" ", "_")
    if ctype == "contrast":
        return "{}__contrast_{}".format(base, cname)
    agg = str(aggregation).strip().lower()
    if not agg:
        raise ValueError("non-contrast candidates require an aggregation")
    return "{}__{}_{}__{}".format(base, ctype, cname, agg)


def parse_candidate_name(name: str) -> Dict[str, str]:
    parts = str(name).split("__")
    if len(parts) < 2:
        raise ValueError("unparseable candidate name {!r}".format(name))
    base = parts[0]
    mid = parts[1]
    if mid.startswith("contrast_"):
        return {
            "base_primitive": base,
            "cut_type": "contrast",
            "cut_name": mid[len("contrast_") :],
            "aggregation": "",
        }
    if len(parts) != 3:
        raise ValueError("unparseable candidate name {!r}".format(name))
    cut_type, cut_name = mid.split("_", 1)
    return {
        "base_primitive": base,
        "cut_type": cut_type,
        "cut_name": cut_name,
        "aggregation": parts[2],
    }


def empty_registry() -> pd.DataFrame:
    return pd.DataFrame(columns=list(REGISTRY_COLUMNS))


def _availability_from_segment(cut_name: str, cut_type: str) -> Dict[str, object]:
    if cut_type == "time":
        try:
            spec = time_segment(cut_name)
        except KeyError:
            spec = None
        if spec is not None:
            return {
                "cut_start_time": spec["start_time"],
                "cut_end_time": spec["end_time"],
                "availability_timestamp": spec["availability_timestamp"],
                "contains_close_auction": bool(spec["contains_close_auction"]),
                "contains_1456_1500": bool(spec["contains_1456_1500"]),
                "latest_source_timestamp": spec["end_time"],
                "factor_available_after": spec["availability_timestamp"],
                "uses_close_auction": bool(spec["contains_close_auction"]),
                "uses_last_5min": bool(spec["uses_last_5min"]),
            }
    # State / event / unspecified time aliases (close, open, ...)
    alias = {
        "open": "OPEN",
        "morning": "MORNING",
        "afternoon": "AFTERNOON",
        "close": "CLOSE",
        "full": "FULL",
        "early_close": "EARLY_CLOSE",
        "late_close": "LATE_CLOSE",
        "close_auction": "CLOSE_AUCTION",
        "close_minus_open": "CLOSE",
        "close_share_full": "CLOSE",
        "reversal_close_vs_open": "CLOSE",
        "highvol_over_full": "FULL",
        "highvol_minus_lowvol": "FULL",
    }
    key = alias.get(str(cut_name).lower())
    if key:
        return _availability_from_segment(key, "time")
    return {
        "cut_start_time": "09:30:00",
        "cut_end_time": "15:00:00",
        "availability_timestamp": "after_continuous_close_T",
        "contains_close_auction": False,
        "contains_1456_1500": True,
        "latest_source_timestamp": "14:59:00",
        "factor_available_after": "after_continuous_close_T",
        "uses_close_auction": False,
        "uses_last_5min": True,
    }


def registry_row(spec: Mapping[str, object]) -> Dict[str, object]:
    cut_type = str(spec.get("cut_type", "")).lower()
    cut_name = str(spec.get("cut_name", spec.get("cut_definition", "")))
    agg = str(spec.get("aggregation", ""))
    contrast = str(spec.get("contrast_operator", "") or "")
    name = spec.get("candidate_name") or candidate_name(
        str(spec["base_primitive"]),
        cut_type,
        cut_name,
        aggregation=agg,
        contrast_operator=contrast,
    )
    avail = _availability_from_segment(cut_name, cut_type)
    row = {c: spec.get(c, "") for c in REGISTRY_COLUMNS}
    row.update(avail)
    row["candidate_name"] = name
    row["base_primitive"] = spec["base_primitive"]
    row["base_family"] = spec.get("base_family", "")
    row["cut_type"] = cut_type
    row["cut_definition"] = spec.get("cut_definition") or cut_name
    row["condition_primitive"] = spec.get("condition_primitive", "") or ""
    row["aggregation"] = agg
    row["contrast_operator"] = contrast
    row["economic_interpretation"] = spec.get("reason") or spec.get(
        "economic_interpretation", ""
    )
    row["parent_factor_if_rescue"] = spec.get("parent_factor_if_rescue", "") or ""
    row["generation_reason"] = spec.get("generation_reason") or spec.get("reason", "")
    row["status"] = spec.get("status", "PROPOSED")
    row["execution_contract_compatible"] = True
    row["production_execution_compatible"] = True
    # Never claim Close[T] execution.
    if bool(row["uses_close_auction"]) or bool(row["uses_last_5min"]):
        row["execution_contract_compatible"] = True  # V2V T+1 still ok
        row["production_execution_compatible"] = True
    return {c: row.get(c, "") for c in REGISTRY_COLUMNS}


def append_rows(existing: Optional[pd.DataFrame], rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    built = [registry_row(r) for r in rows]
    add = pd.DataFrame(built)
    if existing is None or existing.empty:
        return add
    return pd.concat([existing, add], ignore_index=True)


def duplicate_names(frame: pd.DataFrame) -> List[str]:
    if frame.empty:
        return []
    vc = frame["candidate_name"].astype(str).value_counts()
    return [str(n) for n in vc.index[vc > 1]]


def near_duplicate_pairs(
    values: Mapping[str, pd.Series],
    *,
    corr_floor: float = 0.98,
) -> List[tuple]:
    names = list(values.keys())
    hits = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = values[a].align(values[b], join="inner")
            if sa.size < 20:
                continue
            corr = float(sa.corr(sb, method="spearman"))
            if np_abs(corr) >= corr_floor:
                hits.append((a, b, corr))
    return hits


def np_abs(x: float) -> float:
    return abs(float(x)) if x == x else 0.0


def sidecar_registry_path(root: Optional[Path] = None) -> Path:
    return Path(root or CUT_RESULT_ROOT) / CUT_REGISTRY_NAME


def write_registry(frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
    out = Path(path or sidecar_registry_path())
    if out.resolve() == Path(CANDIDATE_POOL_CSV).resolve():
        raise RuntimeError("refusing to write cut registry onto candidate_pool_v1")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_candidate_pool(path: Optional[Path] = None) -> Dict[str, object]:
    p = Path(path or CANDIDATE_POOL_CSV)
    return {
        "path": str(p),
        "exists": p.exists(),
        "sha256": file_sha256(p) if p.exists() else "",
        "nbytes": p.stat().st_size if p.exists() else 0,
    }


def assert_candidate_pool_unchanged(before: Mapping[str, object], path: Optional[Path] = None) -> None:
    after = snapshot_candidate_pool(path)
    if after["sha256"] != before.get("sha256"):
        raise RuntimeError("candidate_pool_v1 was mutated; cut module must not write it")
    if after["nbytes"] != before.get("nbytes"):
        raise RuntimeError("candidate_pool_v1 size changed")
