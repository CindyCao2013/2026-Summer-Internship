from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (PROJECT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backtest import (  # noqa: E402
    DECILE_LABELS,
    EvaluationResult,
    annual_return,
    annualized_sharpe,
    assign_deciles,
    build_execution_eligible_mask,
    build_decile_returns,
    calibration_composite_eligibility,
    decile_monotonicity,
    evaluate_factor,
    freeze_direction,
    split_calibration_evaluation,
    strict_pass,
)
from data_adapter import (  # noqa: E402
    CompanyDataAdapter,
    DataBundle,
    DataUnavailableError,
    _first_recorded_version,
    assert_factor_inputs_pit,
)
from features import (  # noqa: E402
    _dtw_distance_matrix,
    _dtw_distance_python,
    _load_native_dtw_library,
    _relation_snapshot,
    asof_events_to_daily,
    assert_no_future_edges,
    build_dynamic_graph_features,
    build_industry_excess_return,
    filter_edges_asof,
    relation_features_vary_over_time,
)
from report import plot_factor_result  # noqa: E402
from run import (  # noqa: E402
    _sha256_bytes,
    _write_manifest,
    cache_is_complete,
    load_config,
    validate_registry,
)


def _synthetic_bundle(
    sample_dates: pd.DatetimeIndex,
    symbols,
    *,
    stock_returns: pd.DataFrame = None,
    eligible: pd.DataFrame = None,
    tradable: pd.DataFrame = None,
    industry: pd.DataFrame = None,
    market_caps=None,
) -> DataBundle:
    symbol_list = list(symbols)
    if stock_returns is None:
        stock_returns = pd.DataFrame(
            0.001, index=sample_dates, columns=symbol_list
        )
    calendar = pd.DatetimeIndex(stock_returns.index)
    if eligible is None:
        eligible = pd.DataFrame(
            True, index=sample_dates, columns=symbol_list
        )
    if tradable is None:
        tradable = pd.DataFrame(
            True, index=sample_dates, columns=symbol_list
        )
    if industry is None:
        industry = pd.DataFrame(
            "I1", index=calendar, columns=symbol_list
        )
    derivative = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "market_cap": (
                    market_caps.get(symbol, 100.0)
                    if market_caps is not None
                    else 100.0
                ),
            }
            for date in sample_dates
            for symbol in symbol_list
        ]
    )
    return DataBundle(
        calendar=calendar,
        sample_dates=sample_dates,
        symbols=symbol_list,
        market=pd.DataFrame(),
        derivative=derivative,
        financial=pd.DataFrame(),
        industry=industry,
        universe_mask=pd.DataFrame(
            True, index=sample_dates, columns=symbol_list
        ),
        tradable_mask=tradable,
        eligible_mask=eligible,
        stock_returns=stock_returns,
        index_returns=pd.Series(0.0, index=sample_dates),
        audit={},
    )


def test_factor_t_does_not_use_future_input() -> None:
    valid = pd.DataFrame(
        {
            "factor_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "available_at": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        }
    )
    assert_factor_inputs_pit(valid)
    invalid = valid.copy()
    invalid.loc[1, "available_at"] = pd.Timestamp("2024-01-04")
    with pytest.raises(AssertionError):
        assert_factor_inputs_pit(invalid)


def test_financial_factor_is_available_only_after_announcement() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    events = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "ann_date": [pd.Timestamp("2024-01-02")],
            "available_date": [pd.Timestamp("2024-01-03")],
            "roe_ttm": [12.5],
        }
    )
    panel = asof_events_to_daily(
        events,
        value_column="roe_ttm",
        dates=dates,
        symbols=["000001.SZ"],
    )
    assert pd.isna(panel.loc["2024-01-02", "000001.SZ"])
    assert panel.loc["2024-01-03", "000001.SZ"] == pytest.approx(12.5)


def test_earliest_recorded_duplicate_prevents_revision_backfill() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["A", "A"],
            "value": [1.0, 9.0],
            "opdate": pd.to_datetime(["2024-01-03", "2025-01-03"]),
        }
    )
    selected = _first_recorded_version(frame, ["date", "symbol"])
    assert len(selected) == 1
    assert selected.iloc[0]["value"] == pytest.approx(1.0)


def test_financial_opdate_is_audit_only_not_silent_ann_date_fallback() -> None:
    class FakeSession:
        def run(self, _script):
            return pd.DataFrame(
                {
                    "symbol": ["A", "A"],
                    "ann_date": pd.to_datetime(
                        ["2024-01-02", "2024-01-02"]
                    ),
                    "report_period": pd.to_datetime(
                        ["2023-12-31", "2023-12-31"]
                    ),
                    "statement_type": ["合并报表", "合并报表"],
                    "roe_ttm": [10.0, 99.0],
                    "revenue_ttm": [100.0, 999.0],
                    "profit_ttm": [20.0, 999.0],
                    "debt_ratio": [40.0, 99.0],
                    "opdate": pd.to_datetime(
                        ["2030-01-01", "2031-01-01"]
                    ),
                }
            )

    config = load_config(PROJECT_DIR / "config.yaml")
    adapter = CompanyDataAdapter(config, session=FakeSession())
    calendar = pd.bdate_range("2024-01-01", periods=5)
    financial = adapter._load_financial(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-31"),
        ["A"],
        calendar,
    )
    assert len(financial) == 1
    assert financial.iloc[0]["roe_ttm"] == pytest.approx(10.0)
    assert financial.iloc[0]["available_date"] == pd.Timestamp("2024-01-03")


def test_financial_statement_type_mismatch_fails_closed() -> None:
    class FakeSession:
        def run(self, _script):
            return pd.DataFrame(
                {
                    "symbol": ["A"],
                    "ann_date": pd.to_datetime(["2024-01-02"]),
                    "report_period": pd.to_datetime(["2023-12-31"]),
                    "statement_type": ["母公司报表"],
                    "roe_ttm": [10.0],
                    "revenue_ttm": [100.0],
                    "profit_ttm": [20.0],
                    "debt_ratio": [40.0],
                    "opdate": pd.to_datetime(["2024-01-03"]),
                }
            )

    adapter = CompanyDataAdapter(
        load_config(PROJECT_DIR / "config.yaml"), session=FakeSession()
    )
    with pytest.raises(DataUnavailableError, match="statement type"):
        adapter._load_financial(
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-31"),
            ["A"],
            pd.bdate_range("2024-01-01", periods=5),
        )


def test_schema_audit_covers_every_runtime_table() -> None:
    required = set(CompanyDataAdapter.REQUIRED_SCHEMAS)
    assert {
        "eod",
        "derivative",
        "financial",
        "industry",
        "universe",
        "index_return",
        "calendar",
        "previous_name",
    }.issubset(required)


def test_relation_graph_excludes_future_edges() -> None:
    edges = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "C"],
            "effective_from": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "effective_to": [pd.NaT, pd.NaT],
        }
    )
    active = filter_edges_asof(edges, "2024-01-15")
    assert list(active["target"]) == ["B"]
    assert_no_future_edges(active, "2024-01-15")
    with pytest.raises(AssertionError):
        assert_no_future_edges(edges, "2024-01-15")


def test_relation_features_must_change_over_time() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    dynamic = pd.DataFrame([[1.0, 2.0], [1.2, 1.8]], index=dates)
    broadcast = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], index=dates)
    assert relation_features_vary_over_time({"degree": dynamic})
    assert not relation_features_vary_over_time({"degree": broadcast})
    assert not relation_features_vary_over_time(
        {"degree": dynamic, "dtw_similarity_mean": broadcast}
    )


def test_graph_refresh_drops_nodes_missing_from_new_snapshot() -> None:
    all_dates = pd.bdate_range("2023-12-01", periods=22)
    sample_dates = all_dates[-12:]
    symbols = ["A", "B", "C"]
    random = np.random.RandomState(11)
    stock_returns = pd.DataFrame(
        random.normal(scale=0.01, size=(len(all_dates), len(symbols))),
        index=all_dates,
        columns=symbols,
    )
    eligible = pd.DataFrame(True, index=sample_dates, columns=symbols)
    eligible.loc[sample_dates[2]:, "B"] = False
    bundle = _synthetic_bundle(
        sample_dates,
        symbols,
        stock_returns=stock_returns,
        eligible=eligible,
        industry=pd.DataFrame("I1", index=all_dates, columns=symbols),
    )
    config = {
        "relation": {
            "refresh_every_n_trading_days": 2,
            "historical_return_lookback": 10,
            "dtw_top_k": 1,
            "dtw_band_fraction": 0.1,
            "pearson_absolute_threshold": 0.5,
            "pagerank_alpha": 0.85,
            "max_parallel_workers": 1,
        }
    }
    panels, _ = build_dynamic_graph_features(bundle, config)
    assert pd.notna(panels["pagerank"].loc[sample_dates[1], "B"])
    assert pd.isna(panels["pagerank"].loc[sample_dates[2], "B"])


def test_industry_excess_mean_uses_only_eligible_universe() -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    symbols = ["A", "B", "OUT"]
    returns = pd.DataFrame([[0.0, 0.0, 1.0]], index=dates, columns=symbols)
    eligible = pd.DataFrame([[True, True, False]], index=dates, columns=symbols)
    bundle = _synthetic_bundle(
        dates,
        symbols,
        stock_returns=returns,
        eligible=eligible,
    )
    panel = build_industry_excess_return(bundle)
    assert panel.loc[dates[0], "A"] == pytest.approx(0.0)
    assert panel.loc[dates[0], "B"] == pytest.approx(0.0)
    assert pd.isna(panel.loc[dates[0], "OUT"])


def test_macro_registry_is_not_cross_sectionally_testable() -> None:
    registry = pd.read_csv(PROJECT_DIR / "factor_registry.csv")
    macro = registry[registry["family"] == "macro"]
    assert len(macro) == 4
    assert set(macro["test_status"]) == {"NOT_TESTABLE_CROSS_SECTIONALLY"}


def test_registry_rejects_shifted_required_fields() -> None:
    registry = pd.read_csv(PROJECT_DIR / "factor_registry.csv")
    validate_registry(registry)
    broken = registry.copy()
    broken.loc[broken["factor_id"].eq("turnover"), "neutralization"] = np.nan
    with pytest.raises(ValueError, match="turnover:neutralization"):
        validate_registry(broken)


def test_binary_factor_is_not_forced_into_deciles() -> None:
    values = pd.Series([0] * 50 + [1] * 50)
    assert assign_deciles(values) is None


def test_raw_binary_factor_cannot_gain_deciles_after_neutralization() -> None:
    date = pd.Timestamp("2024-01-02")
    columns = ["S{:03d}".format(index) for index in range(100)]
    processed = pd.DataFrame(
        [np.arange(100, dtype=float)], index=[date], columns=columns
    )
    raw = pd.DataFrame(
        [[0.0] * 50 + [1.0] * 50], index=[date], columns=columns
    )
    returns = processed / 10000.0
    eligible = pd.DataFrame(True, index=[date], columns=columns)
    decile, stats, skipped = build_decile_returns(
        processed,
        returns,
        eligible,
        pd.DatetimeIndex([date]),
        min_stocks=100,
        min_unique_values=10,
        raw_factor=raw,
    )
    assert decile.empty
    assert stats.loc[date, "n_unique_values"] == 2
    assert skipped["insufficient_unique_values"] == 1


def test_decile_labels_are_always_q1_to_q10() -> None:
    labels = assign_deciles(pd.Series(np.arange(100, dtype=float)))
    assert labels is not None
    assert set(labels.unique()) == set(DECILE_LABELS)


def test_hl_is_q10_minus_q1() -> None:
    date = pd.Timestamp("2024-01-02")
    columns = ["S{:03d}".format(index) for index in range(100)]
    factor = pd.DataFrame([np.arange(100)], index=[date], columns=columns)
    returns = pd.DataFrame(
        [np.arange(100) / 10000.0], index=[date], columns=columns
    )
    eligible = pd.DataFrame(True, index=[date], columns=columns)
    decile, _, _ = build_decile_returns(
        factor,
        returns,
        eligible,
        pd.DatetimeIndex([date]),
        min_stocks=100,
        min_unique_values=10,
    )
    assert list(decile.columns) == DECILE_LABELS + ["H-L"]
    assert decile.loc[date, "H-L"] == pytest.approx(
        decile.loc[date, "Q10"] - decile.loc[date, "Q1"]
    )


def test_annual_return_formula() -> None:
    returns = pd.Series([0.01, 0.03, -0.01])
    assert annual_return(returns, 250) == pytest.approx(returns.mean() * 250)


def test_sharpe_uses_sample_std_and_sqrt_annualization() -> None:
    returns = pd.Series([0.01, 0.03, -0.01, 0.02])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(250)
    assert annualized_sharpe(returns, 250) == pytest.approx(expected)


def test_monotonicity_is_signed_spearman_not_absolute() -> None:
    decreasing = np.arange(10, 0, -1, dtype=float)
    assert decile_monotonicity(decreasing) == pytest.approx(-1.0)


def test_pass_requires_all_three_thresholds() -> None:
    thresholds = {
        "hl_annual_return": 0.20,
        "hl_sharpe": 2.00,
        "decile_monotonicity": 0.70,
    }
    assert strict_pass(0.20, 2.00, 0.70, thresholds) == (
        True,
        True,
        True,
        True,
    )
    assert strict_pass(0.19, 2.10, 0.90, thresholds)[-1] is False
    assert strict_pass(0.30, 1.99, 0.90, thresholds)[-1] is False
    assert strict_pass(0.30, 2.10, 0.69, thresholds)[-1] is False


def test_orientation_uses_calibration_not_evaluation() -> None:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    columns = ["S{:02d}".format(index) for index in range(20)]
    factor = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (4, 1)),
        index=dates,
        columns=columns,
    )
    returns = factor.copy()
    returns.loc[dates[2:]] = -factor.loc[dates[2:]]
    eligible = pd.DataFrame(True, index=dates, columns=columns)
    direction, source, calibration_ic = freeze_direction(
        factor,
        returns,
        eligible,
        pd.DatetimeIndex(dates[:2]),
        np.nan,
        "calibration_30pct",
        1,
    )
    assert direction == 1
    assert source == "calibration_30pct_rank_ic"
    assert calibration_ic == pytest.approx(1.0)


def test_calibration_split_embargo_excludes_forward_evaluation_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    calibration, evaluation = split_calibration_evaluation(
        dates, 0.30, embargo_rows=2
    )
    assert evaluation[0] == dates[6]
    assert calibration[-1] == dates[3]
    assert dates.get_loc(calibration[-1]) + 2 < dates.get_loc(evaluation[0])


def test_preprocessing_uses_signal_date_eligible_cross_section() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    symbols = ["A", "B", "OUT"]
    eligible = pd.DataFrame(
        [[True, True, False]] * len(dates), index=dates, columns=symbols
    )
    bundle = _synthetic_bundle(dates, symbols, eligible=eligible)
    raw = pd.DataFrame(
        [[1.0, 2.0, 1e9]] * len(dates), index=dates, columns=symbols
    )
    registry_row = {
        "factor_id": "synthetic",
        "family": "price_volume",
        "factor_type": "atomic",
        "neutralization": "none",
        "default_orientation": 1,
        "orientation_method": "economic",
    }
    result = evaluate_factor(
        "synthetic",
        raw,
        registry_row,
        bundle,
        load_config(PROJECT_DIR / "config.yaml"),
    )
    processed = result.processed_oriented.loc[dates[0]]
    assert pd.isna(processed["OUT"])
    assert processed["A"] == pytest.approx(-processed["B"])


def test_neutralization_cap_transform_uses_only_eligible_cross_section() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    symbols = ["S{:02d}".format(index) for index in range(40)]
    eligible_symbols = symbols[:30]
    eligible = pd.DataFrame(False, index=dates, columns=symbols)
    eligible.loc[:, eligible_symbols] = True
    base_caps = {
        symbol: float(np.exp(1.0 + index / 10.0))
        for index, symbol in enumerate(eligible_symbols)
    }
    caps_one = {
        **base_caps,
        **{symbol: 100.0 + index for index, symbol in enumerate(symbols[30:])},
    }
    caps_two = {
        **base_caps,
        **{
            symbol: float(1e12 * (index + 1))
            for index, symbol in enumerate(symbols[30:])
        },
    }
    raw_values = np.sin(np.arange(40, dtype=float)) + np.arange(40) / 20.0
    raw = pd.DataFrame(
        np.tile(raw_values, (len(dates), 1)),
        index=dates,
        columns=symbols,
    )
    registry_row = {
        "factor_id": "synthetic",
        "family": "price_volume",
        "factor_type": "atomic",
        "neutralization": "industry+log_market_cap",
        "default_orientation": 1,
        "orientation_method": "economic",
    }
    config = load_config(PROJECT_DIR / "config.yaml")
    first = evaluate_factor(
        "synthetic",
        raw,
        registry_row,
        _synthetic_bundle(
            dates, symbols, eligible=eligible, market_caps=caps_one
        ),
        config,
    )
    second = evaluate_factor(
        "synthetic",
        raw,
        registry_row,
        _synthetic_bundle(
            dates, symbols, eligible=eligible, market_caps=caps_two
        ),
        config,
    )
    pd.testing.assert_series_equal(
        first.processed_oriented.loc[dates[0], eligible_symbols],
        second.processed_oriented.loc[dates[0], eligible_symbols],
    )


def test_execution_mask_requires_next_entry_and_exit_tradability() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    symbols = ["A", "B"]
    tradable = pd.DataFrame(True, index=dates, columns=symbols)
    tradable.loc[dates[1], "B"] = False
    bundle = _synthetic_bundle(dates, symbols, tradable=tradable)
    execution = build_execution_eligible_mask(bundle, lag=2)
    assert bool(bundle.eligible_mask.loc[dates[0], "B"])
    assert not bool(execution.loc[dates[0], "B"])
    assert bool(execution.loc[dates[0], "A"])


def test_composite_member_eligibility_uses_calibration_dates_only() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    symbols = ["S{:03d}".format(index) for index in range(100)]
    raw = pd.DataFrame(
        np.tile(np.arange(100, dtype=float), (6, 1)),
        index=dates,
        columns=symbols,
    )
    processed = raw.copy()
    raw.loc[dates[3]:] = np.nan
    eligible = pd.DataFrame(True, index=dates, columns=symbols)
    config = {
        "min_stocks_per_day": 100,
        "min_unique_values_per_day": 10,
        "min_overall_stock_day_coverage": 0.50,
        "min_valid_day_ratio": 0.80,
    }
    usable, reason = calibration_composite_eligibility(
        raw, processed, eligible, dates[:3], config
    )
    assert usable
    assert reason == ""


def test_each_tested_candidate_generates_exactly_two_pngs(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    decile = pd.DataFrame(
        {
            label: np.linspace(-0.001, 0.001, 20) + index * 0.0001
            for index, label in enumerate(DECILE_LABELS)
        },
        index=dates,
    )
    decile["H-L"] = decile["Q10"] - decile["Q1"]
    summary = {
        "factor_id": "synthetic",
        "test_status": "PASS",
        "hl_annual_return": 0.25,
        "hl_sharpe": 2.5,
        "decile_monotonicity": 1.0,
        "evaluation_start": dates.min(),
        "evaluation_end": dates.max(),
    }
    result = EvaluationResult(
        factor_id="synthetic",
        summary=summary,
        decile_daily=decile,
        hl_daily=decile["H-L"],
        processed_oriented=None,
        composite_usable=True,
        skipped_day_reasons={},
    )
    plot_factor_result(result, tmp_path, annualization_days=250)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "cumulative_hl.png",
        "decile_bar.png",
    }


def test_same_inputs_produce_deterministic_deciles() -> None:
    date = pd.Timestamp("2024-01-02")
    columns = ["S{:03d}".format(index) for index in range(100)]
    factor = pd.DataFrame([np.arange(100)], index=[date], columns=columns)
    returns = pd.DataFrame(
        [np.sin(np.arange(100)) / 100.0], index=[date], columns=columns
    )
    eligible = pd.DataFrame(True, index=[date], columns=columns)
    first, first_stats, first_skips = build_decile_returns(
        factor,
        returns,
        eligible,
        pd.DatetimeIndex([date]),
        min_stocks=100,
        min_unique_values=10,
    )
    second, second_stats, second_skips = build_decile_returns(
        factor,
        returns,
        eligible,
        pd.DatetimeIndex([date]),
        min_stocks=100,
        min_unique_values=10,
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_stats, second_stats)
    assert first_skips == second_skips


def test_native_full_universe_dtw_matches_reference_kernel() -> None:
    random = np.random.RandomState(42)
    values = random.normal(size=(101, 8))
    distances = _dtw_distance_matrix(values, band=2, max_workers=2)
    expected = _dtw_distance_python(values[3], values[77], band=2)
    assert distances.shape == (101, 101)
    assert distances[3, 77] == pytest.approx(expected)
    assert distances[77, 3] == pytest.approx(expected)


def test_native_dtw_load_failure_is_fail_closed(monkeypatch) -> None:
    def fail_load(_path):
        raise OSError("broken shared library")

    monkeypatch.setattr("features.ctypes.CDLL", fail_load)
    with pytest.raises(DataUnavailableError, match="library load failed"):
        _load_native_dtw_library()


def test_degree_centrality_is_normalized_by_observed_maximum() -> None:
    random = np.random.RandomState(7)
    history = pd.DataFrame(
        random.normal(size=(12, 4)),
        columns=["A", "B", "C", "D"],
    )
    industry = pd.Series({"A": "I1", "B": "I1", "C": "I2", "D": "I2"})
    snapshot, _ = _relation_snapshot(
        history,
        industry,
        top_k=1,
        band_fraction=0.1,
        correlation_threshold=0.5,
        pagerank_alpha=0.85,
        max_workers=1,
    )
    assert snapshot["degree_centrality"].max() == pytest.approx(1.0)


def test_cache_requires_report_registry_and_matching_statuses(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.csv"
    registry_path = tmp_path / "registry.csv"
    report_path = tmp_path / "report.md"
    manifest_path = tmp_path / "manifest.json"
    graph_path = tmp_path / "graph.csv"
    factor_root = tmp_path / "factors"
    factor_root.mkdir()
    summary = pd.DataFrame(
        [{"factor_id": "x", "test_status": "DATA_UNAVAILABLE"}]
    )
    registry = summary.copy()
    summary.to_csv(summary_path, index=False)
    registry.to_csv(registry_path, index=False)
    report_path.write_text("complete", encoding="utf-8")
    artifact_hashes = {
        "summary": _sha256_bytes(summary_path.read_bytes()),
        "registry": _sha256_bytes(registry_path.read_bytes()),
        "report": _sha256_bytes(report_path.read_bytes()),
    }
    manifest_path.write_text(
        json.dumps(
            {
                "status": "success",
                "cache_hash": "hash",
                "requested_factors": ["x"],
                "factor_statuses": {"x": "DATA_UNAVAILABLE"},
                "graph_diagnostics": [],
                "artifact_hashes": artifact_hashes,
            }
        ),
        encoding="utf-8",
    )
    arguments = (
        ["x"],
        {"cache_hash": "hash"},
        manifest_path,
        summary_path,
        registry_path,
        report_path,
        factor_root,
        graph_path,
    )
    assert cache_is_complete(*arguments)
    report_path.unlink()
    assert not cache_is_complete(*arguments)
    report_path.write_text("complete", encoding="utf-8")
    registry.loc[0, "test_status"] = "ERROR"
    registry.to_csv(registry_path, index=False)
    assert not cache_is_complete(*arguments)
    summary.loc[0, "test_status"] = "ERROR"
    summary.to_csv(summary_path, index=False)
    artifact_hashes = {
        "summary": _sha256_bytes(summary_path.read_bytes()),
        "registry": _sha256_bytes(registry_path.read_bytes()),
        "report": _sha256_bytes(report_path.read_bytes()),
    }
    manifest_path.write_text(
        json.dumps(
            {
                "status": "success",
                "cache_hash": "hash",
                "requested_factors": ["x"],
                "factor_statuses": {"x": "ERROR"},
                "graph_diagnostics": [],
                "artifact_hashes": artifact_hashes,
            }
        ),
        encoding="utf-8",
    )
    assert not cache_is_complete(*arguments)


def test_manifest_persists_skipped_day_reasons(tmp_path: Path) -> None:
    result = EvaluationResult(
        factor_id="x",
        summary={"factor_id": "x", "factor_type": "atomic", "test_status": "UNTESTABLE"},
        decile_daily=pd.DataFrame(),
        hl_daily=pd.Series(dtype=float),
        processed_oriented=None,
        composite_usable=False,
        skipped_day_reasons={"insufficient_unique_values": 7},
    )
    path = tmp_path / "manifest.json"
    _write_manifest(
        path,
        hashes={"cache_hash": "x"},
        requested=["x"],
        results=[result],
        audit={},
        graph_diagnostics=pd.DataFrame(),
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["skipped_day_reasons"]["x"] == {
        "insufficient_unique_values": 7
    }
