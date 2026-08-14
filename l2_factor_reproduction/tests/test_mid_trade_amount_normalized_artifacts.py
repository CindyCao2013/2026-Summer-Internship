"""Integration gates for the persisted normalized_v1 research package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from l2_factor_reproduction.python.mid_trade_amount_normalization import (
    validate_frozen_config,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE = (
    ROOT
    / "research/results/l2_reproduction/mid_order_ratio/normalized_v1"
)
REPORT = (
    ROOT
    / "research/reports/factors/mid_order_ratio/normalized_v1"
)
ARTIFACTS = REPORT / "artifacts"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_config_exists_and_stage_b_hash_is_valid() -> None:
    path = CACHE / "frozen_config.json"
    assert path.is_file()
    config = json.loads(path.read_text(encoding="utf-8"))
    verified = validate_frozen_config(config)
    assert verified == config["config_sha256"]
    assert config["selection_uses_returns"] is False
    assert config["a1"]["selection_basis"] == "distribution_only_no_returns"
    assert config["effective_direction"] == {
        "A0": -1,
        "A1": -1,
        "A2": -1,
        "A3": -1,
    }


def test_a0_daily_panel_and_formal_metric_parity_gate_passed() -> None:
    parity_path = ARTIFACTS / "a0_parity.json"
    assert parity_path.is_file()
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    assert parity["gate"] == "passed"
    assert parity["canonical_left_only"] == 0
    assert parity["nan_pattern_equal_on_canonical_grid"] is True
    assert parity["spearman"] >= 1.0 - 1e-12
    assert parity["max_abs_error"] <= 1e-10

    metric_path = ARTIFACTS / "a0_formal_metric_parity.json"
    assert metric_path.is_file()
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    assert metric["gate"] == "passed"
    assert metric["rank_ic_abs_error"] <= 1e-12
    assert metric["icir_abs_error"] <= 1e-10
    assert metric["decile_max_abs_error"] <= 1e-10
    assert metric["hl_annu_ret_abs_error"] <= 1e-10


def test_scales_retain_strict_lag_source_dates() -> None:
    path = CACHE / "scales.parquet"
    assert path.is_file()
    columns = [
        "TradeDate",
        "ADV20_lag1",
        "ATS20_lag1",
        "ADV20_history_count",
        "ATS20_history_count",
        "ADV20_source_max_date",
        "ATS20_source_max_date",
    ]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("TradeDate", ">=", pd.Timestamp("2023-01-03")),
            ("TradeDate", "<=", pd.Timestamp("2023-01-31")),
        ],
    )
    trade_date = pd.to_datetime(frame["TradeDate"])
    for scale, count, source in (
        ("ADV20_lag1", "ADV20_history_count", "ADV20_source_max_date"),
        ("ATS20_lag1", "ATS20_history_count", "ATS20_source_max_date"),
    ):
        valid = frame[scale].notna()
        assert frame.loc[valid, count].eq(20).all()
        assert (
            pd.to_datetime(frame.loc[valid, source])
            < trade_date.loc[valid]
        ).all()


def test_sample_segments_never_refreeze_parameters_or_direction() -> None:
    frame = pd.read_csv(ARTIFACTS / "sample_segment_results.csv")
    valid = frame[frame["status"].eq("ok")]
    assert set(valid["segment"]) == {"IS", "validation", "OOS"}
    assert valid["parameters_refrozen"].eq(False).all()
    assert valid["direction_refrozen"].eq(False).all()
    assert valid["effective_direction"].eq(-1).all()

    by_universe = pd.read_csv(
        ARTIFACTS / "sample_segment_results_by_universe.csv"
    )
    valid_all = by_universe[by_universe["status"].eq("ok")]
    assert set(valid_all["universe"]) == {
        "ALL",
        "CSI300",
        "CSI500",
        "CSI1000",
    }
    assert set(valid_all["segment"]) == {"IS", "validation", "OOS"}
    assert valid_all["parameters_refrozen"].eq(False).all()
    assert valid_all["direction_refrozen"].eq(False).all()


def test_manifest_hashes_every_declared_generated_file() -> None:
    path = ARTIFACTS / "artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["frozen_effective_direction"] == -1
    assert manifest["fee"]["one_way_bps"] == 7.5
    for record in manifest["generated_files"]:
        artifact = Path(record["absolute_path"])
        assert artifact.is_file(), artifact
        assert _sha256(artifact) == record["sha256"], artifact

    final_manifest = json.loads(
        (ARTIFACTS / "final_package_manifest.json").read_text(encoding="utf-8")
    )
    for record in final_manifest["files"]:
        artifact = REPORT / record["path"]
        assert artifact.is_file(), artifact
        assert _sha256(artifact) == record["sha256"], artifact


def test_exported_html_embeds_figures_and_pdf_has_pages() -> None:
    html_path = REPORT / "export/mid_trade_amount_normalized_report.html"
    pdf_path = REPORT / "export/mid_trade_amount_normalized_report.pdf"
    html = html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "Missing image:" not in html
    pdf = pdf_path.read_bytes()
    assert pdf.startswith(b"%PDF")
    assert b"/Type /Page" in pdf
    assert len(pdf) > 10_000

