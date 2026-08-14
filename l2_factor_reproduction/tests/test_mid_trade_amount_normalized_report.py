"""Pure-DataFrame tests for the normalized mid-trade-amount report."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from l2_factor_reproduction.scripts.generate_mid_trade_amount_normalized_report import (  # noqa: E402
    FIGURE_CLASSES,
    FROZEN_EFFECTIVE_DIRECTION,
    ReportDataError,
    compute_missing_scale_coverage,
    compute_quantile_statistics,
    evaluate_prepared,
    expand_evaluation_summary,
    generate_figure_suite,
    plot_decile_files,
    replace_with_authoritative_a0,
)


def test_summary_extension_keeps_frozen_direction_and_adds_fee() -> None:
    summary = {
        "rank_ic": -0.025,
        "icir": -1.5,
        "effective_direction": -1,
        "hl_turnover": 0.40,
        "hl_annu_ret": 0.30,
        "hl_sharpe": 1.2,
    }
    rank_ic = pd.Series([-0.03, -0.01, 0.01])

    out = expand_evaluation_summary(
        summary,
        factor_id="a1_adv20",
        universe="CSI1000",
        rank_ic=rank_ic,
        role="A1",
    )

    expected_fee = 0.40 * 7.5 / 10_000 * 250
    assert out["effective_direction"] == FROZEN_EFFECTIVE_DIRECTION == -1
    assert np.isclose(out["effective_rank_ic"], 0.025)
    assert np.isclose(out["effective_icir"], 1.5)
    assert np.isclose(out["implied_annu_fee_7p5bps"], expected_fee)
    assert np.isclose(
        out["hl_net_annu_ret_after_implied_fee"], 0.30 - expected_fee
    )
    assert np.isclose(out["raw_ic_negative_day_share"], 2 / 3)


def test_prepared_evaluation_never_reinfers_direction() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    columns = [f"{value:06d}.SZ" for value in range(100)]
    cross_section = np.arange(100, dtype=float)
    signal = pd.DataFrame(
        np.tile(cross_section, (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    returns = signal / 10_000

    result = evaluate_prepared(
        signal,
        returns,
        effective_direction=FROZEN_EFFECTIVE_DIRECTION,
    )

    assert result["summary"]["rank_ic"] > 0
    assert result["summary"]["effective_direction"] == -1
    assert result["summary"]["hl_annu_ret"] < 0


def test_report_replaces_a0_only_after_exact_key_gate(tmp_path: Path) -> None:
    a0 = "mid_trade_amount_share_abs_4w20w"
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    dynamic = pd.DataFrame(
        {
            "TradeDate": [dates[0], dates[1], dates[0], dates[1]],
            "symbol": ["000001.SZ"] * 4,
            "value": [0.1, 0.2, 0.3, 0.4],
            "factor_id": [
                a0,
                a0,
                "mid_trade_amount_share_adv20",
                "mid_trade_amount_share_adv20",
            ],
        }
    )
    authoritative = dynamic.loc[dynamic["factor_id"].eq(a0)].copy()
    authoritative["value"] = [0.11, np.nan]
    path = tmp_path / "a0.parquet"
    authoritative.to_parquet(path, index=False)

    replaced = replace_with_authoritative_a0(dynamic, path)
    observed = replaced.loc[
        replaced["factor_id"].eq(a0), "value"
    ].to_numpy()
    assert observed[0] == pytest.approx(0.11)
    assert np.isnan(observed[1])

    authoritative = authoritative.iloc[:1]
    authoritative.to_parquet(path, index=False)
    with pytest.raises(ReportDataError, match="keys differ"):
        replace_with_authoritative_a0(dynamic, path)


def test_quintile_statistics_use_daily_cross_section_and_report_spread() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    columns = [f"{value:06d}.SZ" for value in range(20)]
    values = np.arange(20, dtype=float)
    signal = pd.DataFrame(
        np.tile(values, (len(dates), 1)), index=dates, columns=columns
    )
    returns = -signal / 1_000
    characteristic = pd.DataFrame(
        np.tile(values + 1, (len(dates), 1)), index=dates, columns=columns
    )

    stats = compute_quantile_statistics(
        signal,
        returns,
        characteristic,
        factor_id="a1_adv20",
        dimension="adv20",
        n_quantiles=5,
        min_group_names=4,
        effective_direction=-1,
    ).set_index("quantile")

    assert set(stats.index) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"}
    assert stats.loc["Q1", "n_names_avg"] == 4
    assert stats.loc["Q5", "factor_mean"] > stats.loc["Q1", "factor_mean"]
    assert np.isclose(stats.loc["Q1", "rank_ic_mean"], -1)
    assert np.isclose(stats.loc["Q5", "effective_rank_ic_mean"], 1)
    assert np.isclose(
        stats.loc["Q5-Q1", "factor_mean"],
        stats.loc["Q5", "factor_mean"] - stats.loc["Q1", "factor_mean"],
    )


def test_missing_scale_coverage_is_explicit() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    symbols = ["000001.SZ", "000002.SZ"]
    scales = pd.DataFrame(
        [
            {
                "TradeDate": date,
                "symbol": symbol,
                "total_amount": 100.0,
                "adv20_lag1": 10.0 if symbol == "000001.SZ" else np.nan,
            }
            for date in dates
            for symbol in symbols
        ]
    )
    factors = pd.DataFrame(
        [
            {
                "TradeDate": date,
                "symbol": "000001.SZ",
                "factor_id": "a1_adv20",
                "value": 0.2,
            }
            for date in dates
        ]
    )

    row = compute_missing_scale_coverage(
        factors,
        scales,
        factor_ids=["a1_adv20"],
        roles={"a1_adv20": "A1"},
    ).iloc[0]

    assert row["expected_stock_days"] == 4
    assert row["factor_stock_days"] == 2
    assert np.isclose(row["factor_coverage_ratio"], 0.5)
    assert row["missing_scale_stock_days"] == 2
    assert np.isclose(row["missing_scale_ratio"], 0.5)
    assert np.isclose(row["factor_coverage_given_scale"], 1.0)


def test_per_variant_decile_figures_are_separate_nonempty_files(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    data = {
        str(group): np.linspace(-0.001, 0.001, len(dates)) + group * 0.0001
        for group in range(1, 11)
    }
    pnl = pd.DataFrame(data, index=dates)
    pnl["H-L"] = pnl["10"] - pnl["1"]

    annualized, cumulative = plot_decile_files(
        pnl,
        tmp_path,
        factor_id="a2_ats20_0p5x_2x",
        factor_label="A2 ATS20 normalized",
    )

    assert annualized != cumulative
    assert "decile_annualized" in annualized.name
    assert "decile_cumulative" in cumulative.name
    assert annualized.is_file() and annualized.stat().st_size > 0
    assert cumulative.is_file() and cumulative.stat().st_size > 0


def test_ten_class_figure_suite_smoke(tmp_path: Path) -> None:
    factor_id = "mid_trade_amount_share_adv20"
    roles = {factor_id: "A1"}
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    pnl = pd.DataFrame(
        {
            str(group): np.linspace(-0.001, 0.001, len(dates))
            + group * 0.0001
            for group in range(1, 11)
        },
        index=dates,
    )
    pnl["H-L"] = pnl["10"] - pnl["1"]
    factor_summary = pd.DataFrame(
        [
            {
                "factor_id": factor_id,
                "factor_role": "A1",
                "effective_icir": 1.2,
                "hl_sharpe": 1.0,
                "hl_net_annu_ret_after_implied_fee": 0.1,
            }
        ]
    )
    universe_summary = pd.DataFrame(
        [
            {
                "factor_role": "A1",
                "universe": universe,
                "effective_icir": 1.0 + index * 0.1,
            }
            for index, universe in enumerate(("ALL", "CSI300", "CSI500", "CSI1000"))
        ]
    )
    daily = pd.DataFrame(
        {
            "TradeDate": dates,
            "factor_id": factor_id,
            "rank_ic_raw": np.linspace(-0.1, 0.02, len(dates)),
        }
    )
    monthly = pd.DataFrame(
        {
            "month": ["2024-01"],
            "factor_id": [factor_id],
            "rank_ic_mean": [-0.03],
        }
    )
    rolling = daily.copy()
    rolling["rank_ic_63d_mean"] = rolling["rank_ic_raw"].expanding().mean()
    quintiles = pd.DataFrame(
        [
            {
                "factor_id": factor_id,
                "quantile": f"Q{number}",
                "rank_ic_mean": -0.01 * number,
            }
            for number in range(1, 6)
        ]
    )
    parameter = pd.DataFrame(
        [
            {
                "factor_id": factor_id,
                "factor_family": "A1",
                "effective_icir": 1.2,
                "is_selected": True,
            }
        ]
    )
    state = pd.DataFrame(
        [
            {
                "factor_role": "A1",
                "turnover_tercile": tercile,
                "rank_ic": value,
            }
            for tercile, value in (("Low", -0.04), ("Mid", -0.03), ("High", -0.02))
        ]
    )
    ols = pd.DataFrame(
        [
            {
                "factor_role": "A1",
                "ols_method": method,
                "effective_icir": value,
            }
            for method, value in (
                ("raw", 1.2),
                ("industry", 1.1),
                ("cap", 0.9),
                ("joint", 0.8),
            )
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "factor_role": "A1",
                "segment": segment,
                "status": "ok",
                "effective_icir": value,
            }
            for segment, value in (("IS", 1.2), ("validation", 1.0), ("OOS", 0.8))
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "factor_role": "A1",
                "factor_coverage_ratio": 0.95,
                "missing_scale_ratio": 0.05,
            }
        ]
    )

    produced = generate_figure_suite(
        output_dir=tmp_path,
        factor_summary=factor_summary,
        universe_summary=universe_summary,
        decile_pnl={factor_id: pnl},
        roles=roles,
        daily_ic=daily,
        monthly_ic=monthly,
        rolling_ic=rolling,
        cap_stats=quintiles,
        adv_stats=quintiles,
        parameter_stability=parameter,
        state_summary=state,
        ols_summary=ols,
        segment_summary=segments,
        coverage=coverage,
    )

    assert list(produced) == list(FIGURE_CLASSES)
    assert all(paths for paths in produced.values())
    paths = [path for group in produced.values() for path in group]
    assert len(paths) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
