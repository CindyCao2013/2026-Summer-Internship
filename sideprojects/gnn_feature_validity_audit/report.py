"""The only plots and the Markdown report permitted by the audit contract."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import DECILE_LABELS, EvaluationResult, annual_return


TESTED_STATUSES = {
    "PASS",
    "FAIL_RETURN",
    "FAIL_SHARPE",
    "FAIL_MONOTONICITY",
    "FAIL_MULTIPLE",
}
ALLOWED_PNGS = {"cumulative_hl.png", "decile_bar.png"}


def _format_date(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _format_metric(value, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if percent:
        return "{:.2%}".format(float(value))
    return "{:.3f}".format(float(value))


def _clean_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def plot_factor_result(
    result: EvaluationResult,
    output_dir: Path,
    *,
    annualization_days: int,
) -> Sequence[Path]:
    status = str(result.summary["test_status"])
    if status not in TESTED_STATUSES:
        return []
    if result.decile_daily.empty or result.hl_daily.empty:
        raise ValueError("Tested factor has no decile return data")

    _clean_pngs(output_dir)
    factor_id = result.factor_id
    evaluation = "{} to {}".format(
        _format_date(result.summary["evaluation_start"]),
        _format_date(result.summary["evaluation_end"]),
    )

    cumulative = (1.0 + result.hl_daily.astype(float)).cumprod() - 1.0
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(cumulative.index, cumulative.values, color="#1f77b4", linewidth=1.5)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative H-L return")
    axis.set_title("{} — cumulative H-L".format(factor_id))
    annotation = (
        "H-L annual return: {annual}\n"
        "H-L Sharpe: {sharpe}\n"
        "Evaluation: {evaluation}\n"
        "Status: {status}"
    ).format(
        annual=_format_metric(result.summary["hl_annual_return"], percent=True),
        sharpe=_format_metric(result.summary["hl_sharpe"]),
        evaluation=evaluation,
        status=status,
    )
    axis.text(
        0.02,
        0.98,
        annotation,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    cumulative_path = output_dir / "cumulative_hl.png"
    fig.savefig(cumulative_path, dpi=150)
    plt.close(fig)

    annual_returns = [
        annual_return(result.decile_daily[label], annualization_days)
        for label in DECILE_LABELS
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(DECILE_LABELS, annual_returns, color="#4c78a8")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Decile")
    axis.set_ylabel("Annualized mean forward return")
    axis.set_title("{} — Q1 to Q10 annualized returns".format(factor_id))
    axis.text(
        0.02,
        0.98,
        "Decile monotonicity: {}".format(
            _format_metric(result.summary["decile_monotonicity"])
        ),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    decile_path = output_dir / "decile_bar.png"
    fig.savefig(decile_path, dpi=150)
    plt.close(fig)

    generated = {path.name for path in output_dir.glob("*.png")}
    if generated != ALLOWED_PNGS:
        raise AssertionError(
            "Expected exactly two allowed PNGs, got {}".format(sorted(generated))
        )
    return [cumulative_path, decile_path]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[column]):
            display[column] = display[column].map(_format_date)
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else "{:.6g}".format(value)
            )
        else:
            display[column] = display[column].fillna("").astype(str)
        display[column] = display[column].astype(str).str.replace(
            "|", "\\|", regex=False
        )
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def family_phase2_decisions(summary: pd.DataFrame) -> Dict[str, str]:
    decisions: Dict[str, str] = {}
    for family in ("price_volume", "fundamental", "sentiment", "relation"):
        group = summary[summary["family"] == family]
        atomic = group[group["factor_type"] == "atomic"]
        composite = group[group["factor_type"] == "composite"]
        if (atomic["test_status"] == "PASS").any():
            decisions[family] = "YES — at least one atomic factor passed"
        elif (composite["test_status"] == "PASS").any():
            decisions[family] = (
                "CONDITIONAL — composite passed but no atomic factor passed"
            )
        elif atomic["test_status"].isin(TESTED_STATUSES).any():
            decisions[family] = "NO — tested atomic factors produced no PASS"
        else:
            decisions[family] = "UNDETERMINED — no atomic factor was testable"
    decisions["macro"] = (
        "CONDITIONING_ONLY — no independent cross-sectional decile test"
    )
    return decisions


def write_summary(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = summary.copy()
    for column in (
        "calibration_start",
        "calibration_end",
        "evaluation_start",
        "evaluation_end",
    ):
        if column in output.columns:
            output[column] = output[column].map(_format_date)
    output.to_csv(path, index=False)


def write_report(
    summary: pd.DataFrame,
    registry: pd.DataFrame,
    audit: Mapping[str, object],
    path: Path,
    *,
    annualization_days: int,
    source_commit: str,
) -> Dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions = family_phase2_decisions(summary)
    pass_rows = summary[summary["test_status"] == "PASS"]
    fail_rows = summary[summary["test_status"].astype(str).str.startswith("FAIL")]
    untestable_rows = summary[
        ~summary["test_status"].isin(TESTED_STATUSES)
    ]
    atomic_pass_count = int(
        (
            (summary["factor_type"] == "atomic")
            & (summary["test_status"] == "PASS")
        ).sum()
    )
    composite_pass_count = int(
        (
            (summary["factor_type"] == "composite")
            & (summary["test_status"] == "PASS")
        ).sum()
    )
    relation_supported = decisions["relation"].startswith(
        ("YES", "CONDITIONAL")
    )
    non_relation_supported = any(
        decisions[family].startswith(("YES", "CONDITIONAL"))
        for family in ("price_volume", "fundamental", "sentiment")
    )
    gnn_justified = relation_supported and non_relation_supported
    phase2_reason = (
        "Phase 2 is justified because both the relation family and at least "
        "one non-relation cross-sectional family produced qualifying evidence. "
        "This still does not validate any GNN architecture or upstream backtest."
        if gnn_justified
        else "Phase 2 is not justified: a GNN-specific follow-up requires "
        "qualifying relation-family evidence plus a usable non-relation node "
        "feature family. Conventional factor or non-graph composite evidence "
        "alone can be investigated without a GNN."
    )

    availability = (
        summary[
            [
                "factor_id",
                "family",
                "factor_type",
                "data_status",
                "test_status",
                "coverage",
                "failure_reason",
            ]
        ]
        .sort_values(["family", "factor_type", "factor_id"])
        .reset_index(drop=True)
    )
    complete = summary.sort_values(
        ["family", "factor_type", "factor_id"]
    ).reset_index(drop=True)
    family_table = pd.DataFrame(
        [{"family": family, "phase2_decision": decision} for family, decision in decisions.items()]
    )

    limitations = [
        "EOD, derivative, financial, and industry histories are final vendor tables rather than immutable daily snapshots. Duplicate economic keys retain the earliest recorded OPDATE, but an original value overwritten in place cannot be recovered.",
        "Live OPDATE values are often years after their economic dates and are warehouse-maintenance timestamps, not first-availability timestamps. Market/valuation fields use TRADE_DT; financial fields become available on the next trading day after ANN_DT; OPDATE lag counts remain disclosed.",
        "The company new-stock rule is a 60-valid-close-observation proxy, not a verified listing-date table.",
        "No verified continuous news sentiment, LHB, or block-trade PIT table mapping was found.",
        "Concept, shareholder, and supply-chain edge histories were unavailable. Relation graphs use only the frozen verified CITICS, historical DTW, and historical Pearson layers when exact DTW can run.",
        "Relation topology is refreshed every 20 trading days and carried forward between PIT snapshots; it is dynamic over time but not recomputed every day.",
        "H-L results are gross, equal-weight sorting diagnostics and do not include transaction costs or turnover.",
        "The fixed two-row c2c signal lag checks T+1 entry and T+2 exit tradability and differs from the repository's common one-row shift.",
        "Signal-date eligibility is applied before every cross-sectional transform; the calibration/evaluation boundary includes a forward-label embargo.",
        "Composite membership is frozen from calibration-period availability and raw cross-sectional variation, never evaluation performance.",
        "No holding period, factor window, orientation, universe, neutralization variant, or weight was selected from evaluation results.",
    ]
    audit_rows = []
    for key in (
        "sample_start",
        "sample_end",
        "n_sample_dates",
        "n_union_symbols",
        "median_universe_size",
        "median_eligible_size",
        "median_execution_eligible_size",
        "market_rows",
        "derivative_rows",
        "financial_rows",
        "industry_interval_rows",
        "eod_opdate_after_trade_date_rows",
        "derivative_opdate_after_trade_date_rows",
        "financial_rows_with_opdate_after_sample_end",
        "duplicate_version_policy",
        "financial_availability_field",
        "opdate_used_as_first_availability",
    ):
        audit_rows.append({"item": key, "value": audit.get(key, "")})

    lines = [
        "# GNN Feature Validity Audit",
        "",
        "## 0. Project closeout (frozen)",
        "",
        "- **Gate 1 Complete. Decision: Do not proceed to GNN Phase 2.**",
        "- Formal closeout document: `../GATE1_CLOSEOUT.md`.",
        "- This audit answered the only Gate-1 question: whether the upstream "
        "five-family raw features justify a GNN reproduction. They do not.",
        "- The three price-volume PASS factors are one low-risk / low-speculation "
        "cluster (all frozen to direction `-1`), not three independent alphas. "
        "The fundamental equal-weight PASS is a candidate composite only. "
        "Relation features failed under available PIT graph layers.",
        "- This project is frozen. Do not expand the registry, search relation "
        "hyperparameters, or train GNN models from these results.",
        "",
        "## 1. Executive conclusion",
        "",
        "- Source commit: `{}`.".format(source_commit),
        "- Annualization: {} company-standard trading days.".format(
            annualization_days
        ),
        "- Atomic PASS factors: {}.".format(atomic_pass_count),
        "- Composite PASS factors: {}.".format(composite_pass_count),
        "- GNN Phase 2 justified: **{}**. The decision requires relation-family evidence plus at least one usable non-relation family.".format(
            "YES" if gnn_justified else "NO"
        ),
        "- No unavailable or untestable result is converted to a zero-return FAIL.",
        "- Project status: **CLOSED** after Gate 1.",
        "",
        "## 2. Source implementation audit",
        "",
        "The detailed audit is in `../source_audit.md`. Key findings: macro features are declared but absent from the executable feature pipeline; relation values are computed once and broadcast over history; DTW mean is zero in the current pipeline; fundamental rolling calculations operate on daily forward-filled rows; the model target collapses time into one average target per stock; and scan/backtest timing can consume T data and trade at T close.",
        "",
        "## 3. Data availability matrix",
        "",
        _markdown_table(availability),
        "",
        "Observed company-data audit:",
        "",
        _markdown_table(pd.DataFrame(audit_rows)),
        "",
        "## 4. Complete factor result table",
        "",
        _markdown_table(complete),
        "",
        "## 5. PASS factors",
        "",
        _markdown_table(pass_rows),
        "",
        "## 6. FAIL factors",
        "",
        _markdown_table(fail_rows),
        "",
        "## 7. UNTESTABLE factors",
        "",
        _markdown_table(untestable_rows),
        "",
        "Macro variables are market-state or conditioning inputs. They cannot independently sort stocks in a same-day cross section and therefore are not ordinary FAIL factors.",
        "",
        "## 8. Five-family conclusion",
        "",
        _markdown_table(family_table),
        "",
        "Interpretation: volatility/amplitude, market capitalization, and valuation are conventional risk/style exposures. Any PASS only means sortable returns under this frozen test, not graph-specific alpha. Relation degree, PageRank, DTW, and industry-return outcomes are the direct evidence for graph-specific continuation.",
        "",
        "## 9. Whether GNN Phase 2 is justified",
        "",
        phase2_reason,
        "",
        "**Final management decision:** Gate 1 is closed. Do not reproduce the "
        "upstream GNN pipeline. Optional low-risk-cluster deduplication or "
        "fundamental-composite ablation, if ever done, must be a separate "
        "narrow sprint outside this frozen project. See `../GATE1_CLOSEOUT.md`.",
        "",
        "## 10. Exact limitations",
        "",
    ]
    lines.extend("- " + item for item in limitations)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return decisions
