"""Fast Discovery Lane 单元测试（不访问 DDB / ClickHouse / Raw L2）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from l2_factor_reproduction.python.fast_discovery import (
    FAMILY_ADAPTERS,
    RESEARCH_GATE,
    STRONG_GATE,
    WINDOWS,
    _file_window_overlap,
    _sha256,
    compute_fast_metrics,
    context_paths,
    gate_label,
    load_fast_context,
)


def _synthetic_group_pnl(monotonic: bool = True) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=120)
    rng = np.random.default_rng(7)
    data = {}
    for i in range(1, 11):
        drift = (i - 5.5) * 1e-3 if monotonic else (5.5 - i) * 1e-3
        data[str(i)] = drift + rng.normal(0, 1e-4, len(dates))
    pnl = pd.DataFrame(data, index=dates)
    pnl["H-L"] = pnl["10"] - pnl["1"]
    return pnl


def _summary_stub(direction: int = 1) -> dict:
    return {
        "rank_ic_mean_raw": 0.02,
        "rank_icir": 1.5,
        "hl_annu_ret_flipped": 0.25,
        "hl_sharpe_flipped": 2.5,
        "g10_excess_sharpe": 1.8,
        "factor_direction": direction,
        "n_days": 120,
        "n_names_avg": 800.0,
    }


def test_window_registry_frozen() -> None:
    assert WINDOWS["discovery"] == (
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2024-12-31"),
    )
    assert WINDOWS["full"] == (
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2026-07-31"),
    )


def test_family_adapters_cover_four_families() -> None:
    assert set(FAMILY_ADAPTERS) == {
        "liquidity_impact",
        "price_formation",
        "order_book",
        "trade_flow",
    }
    for adapter in FAMILY_ADAPTERS.values():
        assert adapter.primitive_dir.exists(), adapter.name
        assert len(adapter.factor_names) > 0


def test_file_window_overlap_prunes_by_name() -> None:
    path = Path("year=2019/order_book_daily_2019-01-01_2019-03-31.parquet")
    start, end = WINDOWS["discovery"]
    assert not _file_window_overlap(path, start, end, 60)
    near = Path("year=2022/order_book_daily_2022-10-01_2022-12-31.parquet")
    assert _file_window_overlap(near, start, end, 60)  # 60d 缓冲覆盖
    inside = Path(
        "year=2023/order_book_daily_2023-04-01_2023-06-30.parquet"
    )
    assert _file_window_overlap(inside, start, end, 60)
    no_date = Path("quarter=2023Q1")
    assert _file_window_overlap(no_date, start, end, 60)


def test_compute_fast_metrics_monotonic() -> None:
    pnl = _synthetic_group_pnl(monotonic=True)
    turnover = pd.DataFrame(
        {"H-L": np.full(len(pnl), 0.1)}, index=pnl.index
    )
    metrics = compute_fast_metrics(pnl, turnover, _summary_stub())
    assert metrics["decile_mono_spearman"] == pytest.approx(1.0)
    assert metrics["adjacent_violations"] == 0
    assert 0.0 <= metrics["positive_hl_month_fraction"] <= 1.0
    assert metrics["cum_hl_time_spearman"] > 0.9
    assert metrics["icir_raw"] == pytest.approx(1.5)


def test_compute_fast_metrics_reversed_direction() -> None:
    pnl = _synthetic_group_pnl(monotonic=False)
    turnover = pd.DataFrame(
        {"H-L": np.full(len(pnl), 0.1)}, index=pnl.index
    )
    metrics = compute_fast_metrics(pnl, turnover, _summary_stub(-1))
    assert metrics["decile_mono_spearman"] == pytest.approx(-1.0)
    assert metrics["icir_raw"] == pytest.approx(-1.5)


def test_gate_thresholds_frozen() -> None:
    assert STRONG_GATE == {
        "hl_sharpe": 3.0,
        "decile_mono_spearman": 0.85,
        "adjacent_violations": 1,
    }
    assert RESEARCH_GATE["hl_sharpe"] == 2.0
    base = {
        "hl_sharpe": 3.5,
        "decile_mono_spearman": 0.9,
        "adjacent_violations": 0,
    }
    assert gate_label(base) == "strong_candidate"
    assert gate_label({**base, "hl_sharpe": 2.5}) == "research_candidate"
    assert (
        gate_label({**base, "hl_sharpe": 2.5, "adjacent_violations": 3})
        == "none"
    )
    assert gate_label({**base, "hl_sharpe": 0.5}) == "none"


def test_ensure_effective_group_pnl_flips_negative_hl() -> None:
    from l2_factor_reproduction.python.fast_discovery import (
        ensure_effective_group_pnl,
    )

    dates = pd.bdate_range("2023-01-02", periods=30)
    # Raw negative factor: G1 high return, G10 low return → H-L < 0
    data = {str(i): np.full(len(dates), (5.5 - i) * 1e-3) for i in range(1, 11)}
    pnl = pd.DataFrame(data, index=dates)
    pnl["H-L"] = pnl["10"] - pnl["1"]
    assert float(pnl["H-L"].mean()) < 0
    effective = ensure_effective_group_pnl(pnl)
    assert float(effective["H-L"].mean()) > 0
    assert list(effective.columns) == [str(i) for i in range(1, 11)] + ["H-L"]
    # After flip, G10 should be former G1
    assert float(effective["10"].mean()) > float(effective["1"].mean())


def test_save_fast_plots_standard_delivery(tmp_path) -> None:
    from l2_factor_reproduction.python.fast_discovery import save_fast_plots

    pnl = _synthetic_group_pnl(monotonic=True)
    metrics = {
        "hl_annu_ret": 0.25,
        "hl_sharpe": 2.5,
        "hl_mdd": -0.12,
        "avg_hl_turnover": 0.8,
        "decile_mono_spearman": 1.0,
        "positive_hl_month_fraction": 0.7,
    }
    path_cum, path_bar = save_fast_plots(
        tmp_path / "demo", "demo_factor", pnl, metrics
    )
    assert path_cum.exists() and path_cum.stat().st_size > 0
    assert path_bar.exists() and path_bar.stat().st_size > 0
    assert path_cum.name == "cumulative_hl.png"
    assert path_bar.name == "decile_bar.png"


def test_context_hash_verification(tmp_path, monkeypatch) -> None:
    window = "discovery"
    paths = context_paths(window)
    mask = pd.DataFrame(
        {"600000.SH": [1.0]},
        index=pd.DatetimeIndex(["2023-01-03"], name="Date"),
    )
    monkeypatch.setattr(
        "l2_factor_reproduction.python.fast_discovery.FAST_CONTEXT_DIR",
        tmp_path,
    )
    paths = context_paths(window)
    paths["manifest"].parent.mkdir(parents=True)
    mask.to_parquet(paths["universe_mask"])
    mask.to_parquet(paths["ret_matrix"])
    manifest = {
        "universe": "000852.SH",
        "files": {
            "ret_matrix": {"sha256": _sha256(paths["ret_matrix"])},
            "universe_mask": {"sha256": _sha256(paths["universe_mask"])},
        },
    }
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    loaded_mask, loaded_ret = load_fast_context(window)
    assert loaded_mask.shape == (1, 1)
    assert loaded_ret.shape == (1, 1)

    # 篡改缓存文件必须触发 hash mismatch
    mask.assign(**{"000001.SZ": [1.0]}).to_parquet(paths["ret_matrix"])
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_fast_context(window)
