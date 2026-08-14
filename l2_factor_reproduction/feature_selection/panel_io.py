"""Load FS-1 processed panel partitions for a TradeDate range."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


def iter_quarter_partitions(processed_root: Path) -> List[Path]:
    return sorted(processed_root.glob("year=*/quarter=*/part.parquet"))


def partitions_overlapping(
    processed_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Path]:
    """Heuristic: include quarters that may overlap [start, end]."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    out: List[Path] = []
    for p in iter_quarter_partitions(processed_root):
        # path .../year=YYYY/quarter=Q/part.parquet
        year = int(p.parts[-3].split("=")[1])
        quarter = int(p.parts[-2].split("=")[1])
        q_start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)
        q_end = q_start + pd.offsets.QuarterEnd(startingMonth=3)
        if q_end < start or q_start > end:
            continue
        out.append(p)
    return out


def load_processed_panel_slice(
    processed_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load processed_ind_cap_z_v1 rows with TradeDate in [start, end]."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    parts = partitions_overlapping(processed_root, start, end)
    if not parts:
        return pd.DataFrame(columns=["TradeDate", "Symbol"] + list(columns or []))
    use_cols = None
    if columns is not None:
        use_cols = ["TradeDate", "Symbol"] + [c for c in columns]
    frames = []
    for p in parts:
        if use_cols is not None:
            try:
                import pyarrow.parquet as pq

                available = set(pq.ParquetFile(p).schema.names)
            except Exception:
                available = set(pd.read_parquet(p, columns=[]).columns)
            keep = [c for c in use_cols if c in available]
            df = pd.read_parquet(p, columns=keep) if keep else pd.read_parquet(p)
        else:
            df = pd.read_parquet(p)
        df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
        m = (df["TradeDate"] >= start) & (df["TradeDate"] <= end)
        if m.any():
            frames.append(df.loc[m])
    if not frames:
        return pd.DataFrame(columns=["TradeDate", "Symbol"] + list(columns or []))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["TradeDate", "Symbol"], keep="last")
    return out.sort_values(["TradeDate", "Symbol"]).reset_index(drop=True)


def panel_is_ready(panel_root: Path, *, expect_schema_hash: str) -> tuple[bool, str]:
    """Check FS-1 full/discovery panel readiness for FS-3."""
    inv = panel_root / "feature_inventory.csv"
    processed = panel_root / "processed_ind_cap_z_v1"
    schema = panel_root / "panel_schema.json"
    manifest = panel_root / "manifest.json"
    if not inv.exists():
        return False, "missing feature_inventory.csv"
    if not any(processed.glob("year=*/quarter=*/part.parquet")):
        return False, "missing processed partitions"
    if not schema.exists() and not manifest.exists():
        return False, "missing panel_schema/manifest (build incomplete)"
    return True, "ok"
