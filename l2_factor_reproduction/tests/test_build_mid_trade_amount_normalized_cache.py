"""Tests for the normalized mid-trade-amount cache orchestration."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.python.ch_mid_trade_amount_normalization import (  # noqa: E402
    DEFAULT_A1_GRID,
    DEFAULT_A2_GRID,
    a1_selected_amount_column,
    a2_selected_amount_column,
)
from l2_factor_reproduction.python.mid_trade_amount_normalization import (  # noqa: E402
    A0_FACTOR_ID,
    A1_FACTOR_ID,
    A2_FACTOR_ID,
    A3_FACTOR_ID,
    validate_frozen_config,
)
from l2_factor_reproduction.scripts import (  # noqa: E402
    build_mid_trade_amount_normalized_cache as builder,
)


PERIODS = {
    "warmup_start": "2022-12-01",
    "factor_start": "2022-12-22",
    "end": "2023-01-05",
    "calibration_start": "2023-01-02",
    "calibration_end": "2023-01-05",
}
SYMBOLS = [f"{value:06d}.SZ" for value in range(1, 11)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _primitive_fetch(
    calls: List[Tuple[pd.Timestamp, pd.Timestamp]],
):
    def fetch(start: Any, end: Any) -> pd.DataFrame:
        first = pd.Timestamp(start).normalize()
        last = pd.Timestamp(end).normalize()
        calls.append((first, last))
        rows: List[Dict[str, Any]] = []
        for date in pd.date_range(first, last, freq="D"):
            for index, symbol in enumerate(SYMBOLS, start=1):
                median = float(10 + index)
                rows.append(
                    {
                        "Symbol": symbol,
                        "TRADE_DT": date,
                        "TotalAmount": float(1_000 + index),
                        "positive_trade_count": 10,
                        "daily_mean_trade_amount": median + 1,
                        "daily_q20": median - 2,
                        "q30": median - 1,
                        "q50": median,
                        "q70": median + 1,
                        "daily_q80": median + 2,
                    }
                )
        return pd.DataFrame(rows)

    return fetch


def _dynamic_frame(
    scale_rows: pd.DataFrame,
    *,
    aliases: bool = False,
    unmatched: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in scale_rows[["symbol", "TradeDate"]].to_dict("records"):
        row: Dict[str, Any] = {
            "symbol": record["symbol"],
            "TradeDate": record["TradeDate"],
            "total_amount": 100.0,
            "a0_abs_4w20w_selected_amount": 30.0,
            "a3_q20_q80_selected_amount": 40.0,
        }
        for index, (lower, upper) in enumerate(DEFAULT_A1_GRID):
            selected = (
                30.0
                if (float(lower), float(upper)) == (1.0, 10.0)
                else 20.0 + index
            )
            row[a1_selected_amount_column(lower, upper)] = selected
        for index, (lower, upper) in enumerate(DEFAULT_A2_GRID):
            selected = (
                35.0
                if (float(lower), float(upper)) == (0.5, 2.0)
                else 10.0 + index
            )
            row[a2_selected_amount_column(lower, upper)] = selected
        rows.append(row)
    frame = pd.DataFrame(rows)
    if unmatched and not frame.empty:
        frame.loc[frame.index[0], "symbol"] = "999999.SZ"
    if aliases:
        frame = frame.rename(
            columns={
                "symbol": "Symbol",
                "TradeDate": "TRADE_DT",
                "total_amount": "TotalAmount",
                "a0_abs_4w20w_selected_amount": "a0_selected_amount",
                "a3_q20_q80_selected_amount": "rollq_selected_amount",
            }
        )
    return frame


def _run_primitives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, Any]:
    monkeypatch.setattr(
        builder,
        "fetch_daily_scale_primitive",
        _primitive_fetch(calls),
    )
    monkeypatch.setattr(
        builder,
        "load_market_calendar",
        lambda start, end: pd.date_range(start, end, freq="D"),
    )
    return builder.run_primitives(output_root=tmp_path, **PERIODS)


def _run_calibrate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, Any]:
    def fetch(
        start: Any,
        end: Any,
        scale_rows: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        assert tuple(kwargs["a1_grid"]) == tuple(DEFAULT_A1_GRID)
        assert tuple(kwargs["a2_grid"]) == tuple(DEFAULT_A2_GRID)
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        return _dynamic_frame(scale_rows, aliases=True)

    dates = pd.date_range(
        PERIODS["calibration_start"], PERIODS["calibration_end"], freq="D"
    )
    membership = pd.DataFrame(1.0, index=dates, columns=SYMBOLS)
    market_cap = pd.DataFrame(
        np.tile(np.arange(1, 11, dtype=float), (len(dates), 1)),
        index=dates,
        columns=SYMBOLS,
    )
    monkeypatch.setattr(builder, "fetch_dynamic_factor_aggregates", fetch)
    monkeypatch.setattr(
        builder,
        "load_csi1000_membership",
        lambda start, end: membership,
    )
    monkeypatch.setattr(
        builder,
        "load_wind_market_cap",
        lambda start, end: market_cap,
    )
    return builder.run_calibrate(output_root=tmp_path, **PERIODS)


def test_defaults_and_concurrency_hard_cap() -> None:
    args = builder.build_parser().parse_args([])
    assert args.stage == "all"
    assert args.warmup_start == "2022-11-01"
    assert args.factor_start == "2023-01-03"
    assert args.end == "2026-07-31"
    assert args.calibration_start == "2023-01-03"
    assert args.calibration_end == "2023-06-30"
    assert args.workers == 2
    assert builder.validate_workers(10) == 10
    with pytest.raises(ValueError, match="between 1 and 10"):
        builder.validate_workers(11)
    with pytest.raises(SystemExit):
        builder.build_parser().parse_args(["--workers", "11"])


def test_primitives_resume_q50_rename_and_hash_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    result = _run_primitives(tmp_path, monkeypatch, calls)
    assert len(calls) == 2

    primitive_path = Path(result["primitive_path"])
    primitive = pd.read_parquet(primitive_path)
    assert "daily_median_trade_amount" in primitive.columns
    assert "q50" not in primitive.columns
    assert not primitive.duplicated(["symbol", "TradeDate"]).any()
    scales = pd.read_parquet(result["scales_path"])
    assert {
        "ADV20_lag1",
        "ATS20_lag1",
        "q20",
        "q80",
    }.issubset(scales.columns)
    assert scales["TradeDate"].min() == pd.Timestamp(PERIODS["factor_start"])

    primitive_metadata = json.loads(
        (tmp_path / builder.PRIMITIVE_METADATA_FILE).read_text(
            encoding="utf-8"
        )
    )
    assert primitive_metadata["artifact_sha256"] == _sha256(primitive_path)
    assert primitive_metadata["metadata_sha256"]

    _run_primitives(tmp_path, monkeypatch, calls)
    assert len(calls) == 2, "verified monthly chunks must be resumed"

    first_chunk = sorted(
        (tmp_path / "daily_trade_size_primitive" / "chunks").glob(
            "chunk_*.parquet"
        )
    )[0]
    tampered = pd.read_parquet(first_chunk)
    tampered.loc[tampered.index[0], "total_amount"] += 1
    tampered.to_parquet(first_chunk, index=False)
    with pytest.raises(builder.CacheBuildError, match="chunk SHA256 mismatch"):
        _run_primitives(tmp_path, monkeypatch, calls)

    builder.run_primitives(output_root=tmp_path, force=True, **PERIODS)
    assert len(calls) == 4


def test_calibrate_freezes_distribution_only_config_with_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive_calls: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    _run_primitives(tmp_path, monkeypatch, primitive_calls)
    dynamic_calls: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    result = _run_calibrate(tmp_path, monkeypatch, dynamic_calls)

    assert len(dynamic_calls) == 1
    config_path = Path(result["frozen_config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert validate_frozen_config(config) == result["config_sha256"]
    assert config["selection_uses_returns"] is False
    assert config["a1"]["lower_bps"] == 1.0
    assert config["a1"]["upper_bps"] == 10.0
    assert config["a2"]["lower_multiple"] == 0.5
    assert config["a2"]["upper_multiple"] == 2.0
    assert set(config["effective_direction"].values()) == {-1}
    assert (
        config_path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
        == result["config_sha256"]
    )

    _run_calibrate(tmp_path, monkeypatch, dynamic_calls)
    assert len(dynamic_calls) == 1, "calibration chunks must also resume"


def test_factors_validate_pin_never_refreeze_and_write_exact_long_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_primitives(tmp_path, monkeypatch, [])
    calibration = _run_calibrate(tmp_path, monkeypatch, [])
    scale_path = tmp_path / builder.SCALES_FILE
    scale_sha_before = builder._sha256_file(scale_path)
    factor_calls: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(
        start: Any,
        end: Any,
        scale_rows: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        assert len(kwargs["a1_grid"]) == 9
        assert len(kwargs["a2_grid"]) == 9
        factor_calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        return _dynamic_frame(scale_rows)

    def forbidden_freeze(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Stage B must never call freeze_config")

    monkeypatch.setattr(builder, "fetch_dynamic_factor_aggregates", fetch)
    monkeypatch.setattr(builder, "freeze_config", forbidden_freeze)
    result = builder.run_factors(
        output_root=tmp_path,
        expected_config_sha256=calibration["config_sha256"],
        **PERIODS,
    )
    assert len(factor_calls) == 2
    assert builder._sha256_file(scale_path) == scale_sha_before

    panels = pd.read_parquet(result["factor_panels"])
    assert list(panels.columns) == list(builder.FACTOR_PANEL_COLUMNS)
    assert panels["factor_id"].nunique() == 22
    assert A0_FACTOR_ID in set(panels["factor_id"])
    assert {A1_FACTOR_ID, A2_FACTOR_ID, A3_FACTOR_ID}.issubset(
        panels["factor_id"]
    )
    assert "mid_order_ratio" not in set(panels["factor_id"])
    sample_key = panels[["TradeDate", "symbol"]].iloc[0]
    sample = panels.loc[
        panels["TradeDate"].eq(sample_key["TradeDate"])
        & panels["symbol"].eq(sample_key["symbol"])
    ].set_index("factor_id")["value"]
    assert sample[A0_FACTOR_ID] == pytest.approx(0.30)
    assert sample[A1_FACTOR_ID] == pytest.approx(0.30)
    assert sample[A2_FACTOR_ID] == pytest.approx(0.35)
    assert sample[A3_FACTOR_ID] == pytest.approx(0.40)

    metadata = json.loads(
        (tmp_path / builder.FACTOR_METADATA_FILE).read_text(encoding="utf-8")
    )
    assert metadata["a0_factor_id"] == A0_FACTOR_ID
    assert metadata["legacy_aliases"] == {
        "mid_order_ratio": A0_FACTOR_ID
    }
    assert metadata["legacy_alias_materialized"] is False

    builder.run_factors(
        output_root=tmp_path,
        expected_config_sha256=calibration["config_sha256"],
        **PERIODS,
    )
    assert len(factor_calls) == 2, "factor chunks must resume"


def test_factors_reject_bad_pin_before_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_primitives(tmp_path, monkeypatch, [])
    _run_calibrate(tmp_path, monkeypatch, [])
    calls: List[int] = []

    def fetch(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(1)
        raise AssertionError("fetch must be gated by frozen config validation")

    monkeypatch.setattr(builder, "fetch_dynamic_factor_aggregates", fetch)
    with pytest.raises(builder.CacheBuildError, match="expected Stage-B"):
        builder.run_factors(
            output_root=tmp_path,
            expected_config_sha256="0" * 64,
            **PERIODS,
        )
    assert calls == []


def test_dynamic_chunk_enforces_scale_join_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_primitives(tmp_path, monkeypatch, [])
    calibration = _run_calibrate(tmp_path, monkeypatch, [])

    def unmatched(
        start: Any,
        end: Any,
        scale_rows: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return _dynamic_frame(scale_rows, unmatched=True)

    monkeypatch.setattr(
        builder,
        "fetch_dynamic_factor_aggregates",
        unmatched,
    )
    with pytest.raises(builder.CacheBuildError, match="join row gate failed"):
        builder.run_factors(
            output_root=tmp_path,
            expected_config_sha256=calibration["config_sha256"],
            **PERIODS,
        )
