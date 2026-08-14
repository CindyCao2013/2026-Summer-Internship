"""Tests for frozen-direction and strict-cache report invariants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.scripts.generate_mid_order_ratio_report_artifacts import (  # noqa: E402
    FROZEN_EFFECTIVE_DIRECTION,
    ORDER_BOUNDS,
    STRICT_SESSION,
    STRICT_SSE_FILTER,
    STRICT_SZSE_FILTER,
    _sha256,
    evaluate_prepared,
    validate_strict_bucket_cache,
)

ARTIFACT_ROOT = (
    PROJ_ROOT / "research/reports/factors/mid_order_ratio/artifacts"
)


def _write_cache_metadata(cache_path: Path, **overrides) -> Path:
    metadata = {
        "output": str(cache_path.resolve()),
        "sha256": _sha256(cache_path),
        "requested_start": "2023-01-01",
        "requested_end": "2024-06-30",
        "boundaries_rmb": ORDER_BOUNDS,
        "session": STRICT_SESSION,
        "sse_trade_filter": STRICT_SSE_FILTER,
        "szse_trade_filter": STRICT_SZSE_FILTER,
    }
    metadata.update(overrides)
    metadata_path = cache_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def test_validate_strict_bucket_cache_checks_hash_and_semantics(tmp_path: Path) -> None:
    cache_path = tmp_path / "strict.parquet"
    cache_path.write_bytes(b"strict-cache-fixture")
    metadata_path = _write_cache_metadata(cache_path)

    validated = validate_strict_bucket_cache(
        cache_path, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-06-30")
    )
    assert validated["metadata_path"] == str(metadata_path.resolve())
    assert validated["validated_sha256"] == _sha256(cache_path)

    _write_cache_metadata(cache_path, szse_trade_filter="Type='011'")
    with pytest.raises(ValueError, match="szse_trade_filter"):
        validate_strict_bucket_cache(
            cache_path, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-06-30")
        )


def test_evaluation_never_reinfers_the_frozen_direction() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    columns = [f"{i:06d}.SZ" for i in range(100)]
    cross_section = np.arange(100, dtype=float)
    signal = pd.DataFrame(
        np.tile(cross_section, (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    returns = signal / 10_000

    result = evaluate_prepared(signal, returns)
    summary = result["summary"]

    assert FROZEN_EFFECTIVE_DIRECTION == -1
    assert summary["rank_ic"] > 0
    assert summary["effective_direction"] == FROZEN_EFFECTIVE_DIRECTION
    assert not summary["effective_hl_mean_positive"]
    assert summary["hl_annu_ret"] < 0


def test_persisted_report_artifacts_keep_frozen_and_matched_conventions() -> None:
    manifest = json.loads(
        (ARTIFACT_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    universe = pd.read_csv(ARTIFACT_ROOT / "universe_comparison.csv")
    second = pd.read_csv(
        ARTIFACT_ROOT / "second_neutralization_comparison.csv"
    )
    decile = pd.read_csv(ARTIFACT_ROOT / "csi1000_decile_summary.csv")

    assert manifest["factor_definition"]["effective_direction"] == -1
    assert manifest["factor_definition"]["direction_policy"].startswith("frozen")
    assert manifest["sources"]["bucket_cache_sha256"] == (
        "ccd2f23475756052c60c32f8b56b8cbb648ab99508bf866c9b2424b7ff61cb1f"
    )
    assert universe["effective_direction"].eq(-1).all()
    assert universe["effective_hl_mean_positive"].all()

    expected_retained = (
        second["rank_ic"].abs() / second["matched_baseline_rank_ic"].abs()
    )
    assert np.allclose(
        second["abs_ic_retained_vs_matched"], expected_retained
    )
    assert np.isclose(
        decile["full_sample_monotonicity_spearman"].iloc[0],
        0.8787878787878788,
    )
