"""Tests for strict normalized_v1 Markdown number rendering."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l2_factor_reproduction.python.mid_trade_amount_normalization import (  # noqa: E402
    freeze_config,
)
from l2_factor_reproduction.reporting.render_mid_trade_amount_normalized_markdown import (  # noqa: E402
    CHAPTER_FILES,
    FIGURE_CLASSES,
    MANIFEST_GENERATED_CSVS,
    MarkdownRenderError,
    determine_research_decision,
    render_report,
)


ROLE_TO_FACTOR = {
    "A0": "mid_trade_amount_share_abs_4w20w",
    "A1": "mid_trade_amount_share_adv20",
    "A2": "mid_trade_amount_share_ats20",
    "A3": "mid_trade_amount_share_rollq",
}
ROLES = tuple(ROLE_TO_FACTOR)
UNIVERSES = ("ALL", "CSI300", "CSI500", "CSI1000")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(artifacts: Path, name: str, rows: Iterable[Mapping[str, object]]) -> Path:
    path = artifacts / name
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def _evaluation_values(role: str) -> Dict[str, float]:
    rank_ic = {
        "A0": -0.048,
        "A1": -0.1234,
        "A2": -0.033,
        "A3": -0.014,
    }[role]
    index = ROLES.index(role)
    return {
        "rank_ic": rank_ic,
        "rank_ic_tstat": -3.2 - index * 0.2,
        "icir": -2.4 - index * 0.1,
        "hl_annu_ret": 0.21 + index * 0.01,
        "hl_sharpe": (9.9, -4.0, -3.0, 0.2)[index],
        "hl_mdd": -0.11 - index * 0.01,
        "hl_turnover": 1.21 + index * 0.1,
    }


def _factor_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, role in enumerate(ROLES):
        rows.append(
            {
                "factor_id": ROLE_TO_FACTOR[role],
                "factor_role": role,
                **_evaluation_values(role),
                "factor_coverage_ratio": 0.91 + index * 0.01,
                "implied_annu_fee_7p5bps": 0.031 + index * 0.001,
            }
        )
    return rows


def _universe_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role in ROLES:
        base = _evaluation_values(role)
        for index, universe in enumerate(UNIVERSES):
            rows.append(
                {
                    "factor_id": ROLE_TO_FACTOR[role],
                    "factor_role": role,
                    "universe": universe,
                    "rank_ic": base["rank_ic"] - index * 0.001,
                    "rank_ic_tstat": base["rank_ic_tstat"] - index * 0.1,
                    "icir": base["icir"] - index * 0.1,
                    "hl_sharpe": base["hl_sharpe"] + index * 0.01,
                    "hl_mdd": base["hl_mdd"],
                    "hl_turnover": base["hl_turnover"],
                }
            )
    return rows


def _decile_rows() -> List[Dict[str, object]]:
    return [
        {
            "factor_id": ROLE_TO_FACTOR[role],
            "factor_role": role,
            "decile_monotonicity_spearman": 0.81 - index * 0.03,
            "csi1000_index_excess_hl_annu_ret": 0.17 + index * 0.01,
            "csi1000_index_excess_hl_turnover": 1.31 + index * 0.1,
            "implied_annu_fee_7p5bps": 0.024 + index * 0.001,
        }
        for index, role in enumerate(ROLES)
    ]


def _coverage_rows() -> List[Dict[str, object]]:
    return [
        {
            "factor_id": ROLE_TO_FACTOR[role],
            "factor_role": role,
            "required_scale": {
                "A0": "none",
                "A1": "adv20_lag1",
                "A2": "ats20_lag1",
                "A3": "none",
            }[role],
            "expected_stock_days": 1000,
            "factor_stock_days": 910 + index * 10,
            "factor_coverage_ratio": 0.91 + index * 0.01,
            "missing_scale_ratio": (0.0, 0.08, 0.06, 0.0)[index],
            "factor_coverage_given_scale": 0.97 + index * 0.005,
        }
        for index, role in enumerate(ROLES)
    ]


def _monthly_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, role in enumerate(ROLES):
        for month_index, month in enumerate(("2025-01", "2025-02")):
            rows.append(
                {
                    "factor_id": ROLE_TO_FACTOR[role],
                    "month": month,
                    "rank_ic_mean": -0.0111 - index * 0.002 - month_index * 0.001,
                    "icir": -1.5 - index * 0.1,
                    "negative_ic_day_share": 0.66 + index * 0.02,
                    "n_days": 20,
                }
            )
    return rows


def _rolling_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, role in enumerate(ROLES):
        for day, value in zip(
            ("2025-01-31", "2025-02-28", "2025-03-31"),
            (-0.012 - index * 0.001, -0.015 - index * 0.001, -0.018 - index * 0.001),
        ):
            rows.append(
                {
                    "TradeDate": day,
                    "factor_id": ROLE_TO_FACTOR[role],
                    "rank_ic_raw": value - 0.001,
                    "rank_ic_63d_mean": value,
                    "rank_ic_63d_count": 63,
                }
            )
    return rows


def _quintile_rows(dimension: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role_index, role in enumerate(ROLES):
        for quintile in range(1, 6):
            rows.append(
                {
                    "factor_id": ROLE_TO_FACTOR[role],
                    "dimension": dimension,
                    "quantile": "Q{}".format(quintile),
                    "n_days": 200,
                    "n_names_avg": 190 + quintile,
                    "coverage_rate": 0.93 + quintile * 0.005,
                    "rank_ic_mean": -0.01 * quintile - role_index * 0.001,
                    "icir": -1.0 - quintile * 0.1,
                }
            )
    return rows


def _parameter_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    definitions: Sequence[Sequence[object]] = (
        ("A0", ROLE_TO_FACTOR["A0"], 40000.0, 200000.0, "RMB_per_trade", True),
        ("A1", ROLE_TO_FACTOR["A1"], 2.0, 20.0, "bps_of_ADV20_lag1", True),
        ("A1", "a1_l1_h10", 1.0, 10.0, "bps_of_ADV20_lag1", False),
        ("A2", ROLE_TO_FACTOR["A2"], 0.5, 2.0, "multiple_of_ATS20_lag1", True),
        ("A2", "a2_l0p25_h1p5", 0.25, 1.5, "multiple_of_ATS20_lag1", False),
        ("A3", ROLE_TO_FACTOR["A3"], 0.2, 0.8, "same_day_quantile", True),
    )
    for index, (role, factor_id, lower, upper, unit, selected) in enumerate(
        definitions
    ):
        rows.append(
            {
                "factor_id": factor_id,
                "factor_role": role,
                "factor_family": role,
                "rank_ic": -0.021 - index * 0.001,
                "rank_ic_tstat": -2.6 - index * 0.1,
                "icir": -1.8 - index * 0.1,
                "lower_bound": lower,
                "upper_bound": upper,
                "parameter_unit": unit,
                "is_selected": selected,
            }
        )
    return rows


def _state_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role_index, role in enumerate(ROLES):
        for state_index, state in enumerate(("Low", "Mid", "High")):
            rows.append(
                {
                    "factor_id": ROLE_TO_FACTOR[role],
                    "factor_role": role,
                    "turnover_tercile": state,
                    "rank_ic": -0.02 - state_index * 0.004 - role_index * 0.001,
                    "icir": -1.2 - state_index * 0.2,
                    "negative_ic_day_share": 0.62 + state_index * 0.04,
                    "n_days": 180,
                }
            )
    return rows


def _ols_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role_index, role in enumerate(ROLES):
        for method_index, method in enumerate(("raw", "industry", "cap", "joint")):
            rows.append(
                {
                    "factor_id": ROLE_TO_FACTOR[role],
                    "factor_role": role,
                    "ols_method": method,
                    "rank_ic": -0.03 + method_index * 0.002 - role_index * 0.001,
                    "rank_ic_tstat": -3.1 + method_index * 0.1,
                    "icir": -2.1 + method_index * 0.1,
                    "abs_rank_ic_retained_vs_raw": 1.0 - method_index * 0.09,
                }
            )
    return rows


def _segment_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for role_index, role in enumerate(ROLES):
        for segment_index, segment in enumerate(("IS", "validation", "OOS")):
            oos_tstat = {
                "A0": -3.0,
                "A1": -3.2,
                "A2": -2.7,
                "A3": -1.2,
            }[role]
            rows.append(
                {
                    "factor_id": ROLE_TO_FACTOR[role],
                    "factor_role": role,
                    "segment": segment,
                    "status": "ok",
                    "actual_start": ("2023-01-04", "2023-07-03", "2024-07-01")[
                        segment_index
                    ],
                    "actual_end": ("2024-06-28", "2024-06-28", "2026-07-31")[
                        segment_index
                    ],
                    "rank_ic": (
                        -0.024 - role_index * 0.003
                        if segment != "OOS"
                        else -0.03125 - role_index * 0.001
                    ),
                    "rank_ic_tstat": (
                        -2.4 - role_index * 0.1
                        if segment != "OOS"
                        else oos_tstat
                    ),
                    "icir": -1.7 - role_index * 0.1,
                    "hl_sharpe": {
                        "A0": 8.0,
                        "A1": -9.0,
                        "A2": -8.0,
                        "A3": 0.1,
                    }[role],
                    "hl_mdd": -0.12 - role_index * 0.01,
                    "hl_turnover": 1.3 + role_index * 0.1,
                }
            )
    return rows


def _distribution_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    groups = [
        ("universe", name) for name in UNIVERSES
    ] + [
        ("market_cap_quintile", "Q{}".format(index)) for index in range(1, 6)
    ]
    for index, (group_type, group) in enumerate(groups):
        rows.append(
            {
                "scale": "ADV20_lag1",
                "unit": "bps_of_ADV",
                "group_type": group_type,
                "group": group,
                "quantile": 0.5,
                "value": 0.3142 + index * 0.01,
                "calibration_start": "2023-01-03",
                "calibration_end": "2023-06-30",
            }
        )
    return rows


def _size_rows() -> List[Dict[str, object]]:
    return [
        {
            "variant": "A1_L2_H20",
            "bucket_type": "market_cap",
            "bucket": bucket,
            "amount_coverage": 0.31 + bucket * 0.01,
            "a0_amount_coverage": 0.29 + bucket * 0.01,
            "calibration_start": "2023-01-03",
            "calibration_end": "2023-06-30",
        }
        for bucket in range(1, 6)
    ]


def _adv_grid_rows() -> List[Dict[str, object]]:
    return [
        {
            "lower_adv_bps_exclusive": 2.0,
            "upper_adv_bps_inclusive": 20.0,
            "overall_amount_coverage": 0.3103,
            "a0_overall_amount_coverage": 0.3192,
            "abs_coverage_diff_vs_a0": 0.0089,
            "mean_abs_quintile_coverage_diff_vs_a0": 0.0632,
            "all_quintiles_between_10pct_80pct": True,
            "minimum_quintile_coverage": 0.2439,
            "maximum_quintile_coverage": 0.4153,
            "frozen_main": True,
        },
        {
            "lower_adv_bps_exclusive": 1.0,
            "upper_adv_bps_inclusive": 10.0,
            "overall_amount_coverage": 0.4161,
            "a0_overall_amount_coverage": 0.3192,
            "abs_coverage_diff_vs_a0": 0.0969,
            "mean_abs_quintile_coverage_diff_vs_a0": 0.1272,
            "all_quintiles_between_10pct_80pct": True,
            "minimum_quintile_coverage": 0.3545,
            "maximum_quintile_coverage": 0.4963,
            "frozen_main": False,
        },
    ]


def _build_report(tmp_path: Path) -> Path:
    report_root = tmp_path / "normalized_v1"
    artifacts = report_root / "artifacts"
    figures = report_root / "figures"
    artifacts.mkdir(parents=True)
    figures.mkdir()

    config = freeze_config(
        {
            "version": "mid_trade_amount_normalized_v1",
            "headline_factor_ids": list(ROLE_TO_FACTOR.values()),
            "a0": {
                "factor_id": ROLE_TO_FACTOR["A0"],
                "lower_rmb_exclusive": 40000,
                "upper_rmb_inclusive": 200000,
                "effective_direction": -1,
            },
            "a1": {
                "factor_id": ROLE_TO_FACTOR["A1"],
                "selection_basis": "distribution_only_no_returns",
                "lower_bps": 2.0,
                "upper_bps": 20.0,
                "effective_direction": -1,
            },
            "a2": {
                "factor_id": ROLE_TO_FACTOR["A2"],
                "lower_multiple": 0.5,
                "upper_multiple": 2.0,
                "effective_direction": -1,
            },
            "a3": {
                "factor_id": ROLE_TO_FACTOR["A3"],
                "lower_daily_quantile_exclusive": 0.2,
                "upper_daily_quantile_inclusive": 0.8,
                "effective_direction": -1,
            },
            "effective_direction": {role: -1 for role in ROLES},
            "fee_bps": 7.5,
        }
    )
    (artifacts / "frozen_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (artifacts / "frozen_config.sha256").write_text(
        str(config["config_sha256"]) + "\n",
        encoding="utf-8",
    )
    (artifacts / "a0_parity.json").write_text(
        json.dumps(
            {
                "gate": "passed",
                "pearson": 1.0,
                "spearman": 0.999999999999,
                "max_abs_error": 1.2e-13,
                "mean_abs_error": 6.4e-17,
                "within_1e_12_share": 1.0,
            }
        ),
        encoding="utf-8",
    )

    table_rows = {
        "factor_variant_summary.csv": _factor_rows(),
        "universe_variant_summary.csv": _universe_rows(),
        "csi1000_decile_summary.csv": _decile_rows(),
        "missing_scale_coverage.csv": _coverage_rows(),
        "csi1000_monthly_rank_ic.csv": _monthly_rows(),
        "csi1000_rolling_63d_rank_ic.csv": _rolling_rows(),
        "csi1000_cap_quintile_statistics.csv": _quintile_rows("market_cap"),
        "csi1000_adv_quintile_statistics.csv": _quintile_rows("adv20_lag1"),
        "parameter_stability.csv": _parameter_rows(),
        "state_turnover_tercile_summary.csv": _state_rows(),
        "ols_diagnostics.csv": _ols_rows(),
        "sample_segment_results.csv": _segment_rows(),
        "normalization_distribution_summary_adv.csv": _distribution_rows(),
        "normalization_by_size_bucket_calibration.csv": _size_rows(),
        "parameter_stability_adv_distribution.csv": _adv_grid_rows(),
    }
    written_tables = {
        name: _write_csv(artifacts, name, rows) for name, rows in table_rows.items()
    }

    figure_records = []
    for index, (class_id, description) in enumerate(FIGURE_CLASSES.items()):
        relative = "figures/{}.png".format(class_id)
        path = report_root / relative
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index + 1]) * 24)
        figure_records.append(
            {
                "id": class_id,
                "description": description,
                "files": [relative],
            }
        )

    generated_paths = [
        written_tables[name] for name in MANIFEST_GENERATED_CSVS
    ] + [
        report_root / record["files"][0] for record in figure_records
    ]
    generated_files = [
        {
            "role": "figure" if path.parent == figures else "artifact",
            "path": path.relative_to(report_root).as_posix(),
            "absolute_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in generated_paths
    ]
    manifest = {
        "version": "mid_trade_amount_normalized_report_v1",
        "headline_factor_ids": list(ROLE_TO_FACTOR.values()),
        "factor_roles": {
            factor_id: role for role, factor_id in ROLE_TO_FACTOR.items()
        },
        "frozen_effective_direction": -1,
        "fee": {
            "one_way_bps": 7.5,
            "annualization": 250,
            "formula": "turnover * 7.5/10000 * 250",
        },
        "figure_classes": figure_records,
        "generated_files": generated_files,
        "frozen_config_snapshot": config,
    }
    (artifacts / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_root


def test_renders_real_numbers_all_sections_and_ten_figure_classes(
    tmp_path: Path,
) -> None:
    report_root = _build_report(tmp_path)
    untouched_names = (
        "02_problem_definition.md",
        "03_data_and_strict_trade_construction.md",
        "04_normalized_factor_definitions.md",
        "09_in_sample_out_of_sample.md",
        "README.md",
        "plan.md",
    )
    for name in untouched_names:
        (report_root / name).write_text("do-not-change\n", encoding="utf-8")

    written = render_report(report_root)

    assert tuple(written) == CHAPTER_FILES
    documents = {
        name: path.read_text(encoding="utf-8") for name, path in written.items()
    }
    combined = "\n".join(documents.values())
    assert "-12.34%" in documents["01_executive_summary.md"]
    assert "0.3142" in documents["06_normalization_diagnostics.md"]
    assert "73.00%" in documents["07_exposure_diagnostics.md"]
    assert "-1.11%" in documents["08_time_and_state_robustness.md"]
    assert "-3.12%" in documents["10_research_decision.md"]
    assert "both ADV20 and ATS20 retain the relation" in combined
    assert "A0" in combined and "A1" in combined
    assert "A2" in combined and "A3" in combined
    assert all(universe in combined for universe in UNIVERSES)
    assert all(
        "](figures/{}.png)".format(class_id) in combined
        for class_id in FIGURE_CLASSES
    )
    assert "raw / industry / cap / joint" in documents[
        "07_exposure_diagnostics.md"
    ]
    assert "IS / validation / OOS" in documents[
        "08_time_and_state_robustness.md"
    ]
    assert "factor-library correlation" in documents["10_research_decision.md"]
    assert "portfolio optimization" in documents["10_research_decision.md"]
    for name in untouched_names:
        assert (report_root / name).read_text(encoding="utf-8") == "do-not-change\n"


def test_missing_required_artifact_fails_before_overwriting(
    tmp_path: Path,
) -> None:
    report_root = _build_report(tmp_path)
    sentinels = {}
    for name in CHAPTER_FILES:
        path = report_root / name
        path.write_text("existing {}\n".format(name), encoding="utf-8")
        sentinels[name] = path.read_bytes()
    (report_root / "artifacts" / "ols_diagnostics.csv").unlink()

    with pytest.raises(MarkdownRenderError, match="ols_diagnostics.csv"):
        render_report(report_root)

    for name, expected in sentinels.items():
        assert (report_root / name).read_bytes() == expected


def test_fee_is_bps_and_output_has_no_draft_markers(tmp_path: Path) -> None:
    report_root = _build_report(tmp_path)

    written = render_report(report_root)

    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in written.values()
    )
    assert "7.5 bps" in combined
    assert re.search(r"\b7\.5(?:0+)?\s*%", combined) is None
    assert re.search(r"\b(?:to" + "do|place" + "holder" + r")\b", combined, re.I) is None


@pytest.mark.parametrize(
    "failed_roles,expected_branch",
    [
        ((), "both ADV20 and ATS20 retain the relation"),
        (("A2",), "only ADV20 retains it"),
        (("A1",), "only ATS20 retains it"),
        (("A1", "A2"), "only the fixed-RMB phenomenon remains"),
        (
            ("A0", "A1", "A2"),
            "all normalized variants lose the relation",
        ),
    ],
)
def test_five_way_decision_uses_oos_ic_not_highest_sharpe(
    failed_roles: Sequence[str],
    expected_branch: str,
) -> None:
    segments = pd.DataFrame(_segment_rows())
    universes = pd.DataFrame(_universe_rows())
    for role in failed_roles:
        mask = (segments["factor_role"] == role) & (segments["segment"] == "OOS")
        segments.loc[mask, "rank_ic_tstat"] = -0.5
        segments.loc[mask, "hl_sharpe"] = 999.0

    result = determine_research_decision(segments, universes, direction=-1)

    assert result.branch == expected_branch
