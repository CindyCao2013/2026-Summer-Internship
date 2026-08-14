"""Factor Report Generator v1 — standard single-factor research pack.

Every factor under research/reports/factors/{FACTOR_ID}/ should expose:

    factor_report.md
    metrics.json
    factor_summary.csv
    mechanism.csv          (alias: mechanism_analysis.csv)
    stability.csv          (alias: yearly_stability.csv)
    execution_summary.csv  (optional until execution layer closed)
    artifacts/             (machine copies of the above)
    figures/               (optional)

TGD20 is the frozen template instance (validated_single_factor).
Do not retune TGD formula here — only package / sync / validate.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from factor_eval_metrics import FACTOR_METRICS_SCHEMA, pack_factor_metrics

REPO_ROOT = Path(__file__).resolve().parent
FACTORS_ROOT = REPO_ROOT / "research" / "reports" / "factors"

# Canonical filenames (aliases accepted on read)
CANONICAL_FILES = (
    "factor_report.md",
    "metrics.json",
    "factor_summary.csv",
    "mechanism.csv",
    "stability.csv",
    "execution_summary.csv",
)

ALIASES = {
    "mechanism.csv": ("mechanism_analysis.csv",),
    "stability.csv": ("yearly_stability.csv",),
}


@dataclass
class FactorCard:
    """Lightweight research card for a single factor."""

    factor_id: str
    display_name: str
    category: str
    status: str
    hypothesis: str
    period: str = "2022-01-28_2025-12-31"
    universe: str = "ALL"
    source_column: str = ""
    frozen_formula: bool = False
    next_step: str = "factor_combination_candidate"
    do_not: List[str] = field(default_factory=list)
    source_artifacts: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def factor_dir(factor_id: str, root: Path = FACTORS_ROOT) -> Path:
    return root / factor_id


def ensure_pack_dirs(factor_id: str, root: Path = FACTORS_ROOT) -> Path:
    out = factor_dir(factor_id, root)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def sync_aliases(out: Path) -> None:
    """Keep both canonical and TGD-legacy names in sync."""
    pairs = [
        ("mechanism.csv", "mechanism_analysis.csv"),
        ("stability.csv", "yearly_stability.csv"),
    ]
    for a, b in pairs:
        pa, pb = out / a, out / b
        if pa.exists() and not pb.exists():
            shutil.copy2(pa, pb)
        elif pb.exists() and not pa.exists():
            shutil.copy2(pb, pa)


def mirror_to_artifacts(out: Path) -> None:
    art = out / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    for name in (
        "metrics.json",
        "factor_summary.csv",
        "mechanism.csv",
        "mechanism_analysis.csv",
        "stability.csv",
        "yearly_stability.csv",
        "execution_summary.csv",
    ):
        src = out / name
        if src.exists():
            shutil.copy2(src, art / name)


def build_metrics_bundle(
    *,
    factor: str,
    category: str,
    period: str,
    universe: str,
    summary: pd.DataFrame,
    research_highlights: Optional[Dict[str, Any]] = None,
    production_highlights: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build top-level metrics.json from a factor_summary table."""
    variants: List[dict] = []
    for _, row in summary.iterrows():
        variants.append(
            pack_factor_metrics(
                factor=str(row.get("factor", factor)),
                period=str(row.get("period", period)),
                universe=str(row.get("universe", universe)),
                mode=str(row.get("mode", "raw")),
                rank_ic=float(row["rank_ic"]) if pd.notna(row.get("rank_ic")) else np.nan,
                icir=float(row["icir"]) if pd.notna(row.get("icir")) else np.nan,
                hl_annu_ret=float(row["hl_annu_ret"]) if pd.notna(row.get("hl_annu_ret")) else np.nan,
                hl_sharpe=float(row["hl_sharpe"]) if pd.notna(row.get("hl_sharpe")) else np.nan,
                hl_mdd=float(row["hl_mdd"]) if pd.notna(row.get("hl_mdd")) else np.nan,
                daily_turnover=float(row["daily_turnover"])
                if pd.notna(row.get("daily_turnover"))
                else np.nan,
                implied_annu_fee=float(row["implied_annu_fee"])
                if "implied_annu_fee" in row and pd.notna(row.get("implied_annu_fee"))
                else None,
                net_sharpe=float(row["net_sharpe"])
                if "net_sharpe" in row and pd.notna(row.get("net_sharpe"))
                else None,
                monotonicity=float(row["monotonicity"])
                if "monotonicity" in row and pd.notna(row.get("monotonicity"))
                else None,
                direction=int(row["direction"]) if "direction" in row and pd.notna(row.get("direction")) else 1,
            )
        )

    def _pick(mode: str, key: str, default: Any = None) -> Any:
        sub = summary.loc[summary["mode"].astype(str) == mode]
        if sub.empty or key not in sub.columns:
            return default
        v = sub.iloc[0][key]
        return None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.floating)) else v

    raw_ic = _pick("raw", "rank_ic")
    raw_icir = _pick("raw", "icir")
    raw_sharpe = _pick("raw", "hl_sharpe")
    si_icir = _pick("size_industry", "icir")
    si_sharpe = _pick("size_industry", "hl_sharpe")
    si_net = _pick("size_industry", "net_sharpe")
    ex_net = _pick("execution_best", "net_sharpe")
    ex_to = _pick("execution_best", "daily_turnover")

    payload: Dict[str, Any] = {
        "factor": factor,
        "category": category,
        "period": period,
        "universe": universe,
        "schema_version": "factor_report_v1",
        "metric_definitions": FACTOR_METRICS_SCHEMA,
        "research_score": research_highlights
        or {
            "RankIC": raw_ic,
            "ICIR": raw_icir,
            "ICIR_size_industry": si_icir,
            "Sharpe": raw_sharpe,
            "Sharpe_size_industry": si_sharpe,
            "MDD": _pick("raw", "hl_mdd"),
            "Monotonicity": _pick("raw", "monotonicity"),
        },
        "production_score": production_highlights
        or {
            "NetSharpe_size_industry": si_net,
            "NetSharpe_execution_best": ex_net,
            "Turnover_raw": _pick("raw", "daily_turnover"),
            "Turnover_execution_best": ex_to,
        },
        "variants": variants,
    }
    if extra:
        payload["extra"] = extra
    return payload


def _df_to_md(df: pd.DataFrame, float_prec: int = 4) -> str:
    """Markdown table without requiring tabulate."""
    if df is None or df.empty:
        return "_empty_"
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_prec}g}")
        else:
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else str(x))
    cols = [str(c) for c in show.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in show.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


def render_factor_report_md(
    card: FactorCard,
    *,
    summary: Optional[pd.DataFrame] = None,
    mechanism: Optional[pd.DataFrame] = None,
    stability: Optional[pd.DataFrame] = None,
    execution: Optional[pd.DataFrame] = None,
    gaps: Optional[Sequence[str]] = None,
) -> str:
    """Render a compact standard factor_report.md (not a full research essay)."""
    lines: List[str] = [
        f"# {card.display_name} — Factor Research Pack",
        "",
        f"- **factor_id**: `{card.factor_id}`",
        f"- **category**: `{card.category}`",
        f"- **status**: `{card.status}`",
        f"- **period**: `{card.period}`",
        f"- **universe**: `{card.universe}`",
        f"- **next**: `{card.next_step}`",
        f"- **frozen_formula**: `{card.frozen_formula}`",
        "",
        "## Hypothesis",
        "",
        card.hypothesis.strip(),
        "",
        "## Checklist",
        "",
        "| Stage | Status |",
        "|-------|--------|",
        "| Factor construction | ✅ |",
        "| Mechanism verification | ✅ if mechanism.csv present else 🟡 |",
        "| Metrics schema | ✅ |",
        "| IC / ICIR / Sharpe / MDD | ✅ |",
        "| Neutralization ladder | ✅ if summary has modes else 🟡 |",
        "| Yearly stability | ✅ if stability.csv present else 🟡 |",
        "| Execution optimization | ✅ if execution closed else 🟡 |",
        "| Research report (essay) | 🟡 optional long-form |",
        "",
    ]

    # Fix checklist dynamically
    has_mech = mechanism is not None and not mechanism.empty
    has_stab = stability is not None and not stability.empty
    has_exec_file = execution is not None and not execution.empty
    exec_closed = False
    if has_exec_file and "status" in execution.columns:
        st = execution["status"].astype(str).str.lower()
        exec_closed = st.isin(["closed", "optimized", "best"]).any() and not st.eq(
            "baseline_only"
        ).all()
    elif has_exec_file and "label" in execution.columns:
        # TGD-style: presence of buffer / execution_best style labels
        labs = execution["label"].astype(str)
        exec_closed = labs.str.contains("buffer|execution", case=False, regex=True).any()
    has_neut = (
        summary is not None
        and not summary.empty
        and summary["mode"].astype(str).isin(["size", "industry", "size_industry"]).any()
    )
    lines[lines.index("| Mechanism verification | ✅ if mechanism.csv present else 🟡 |")] = (
        f"| Mechanism verification | {'✅' if has_mech else '🟡'} |"
    )
    lines[lines.index("| Neutralization ladder | ✅ if summary has modes else 🟡 |")] = (
        f"| Neutralization ladder | {'✅' if has_neut else '🟡'} |"
    )
    lines[lines.index("| Yearly stability | ✅ if stability.csv present else 🟡 |")] = (
        f"| Yearly stability | {'✅' if has_stab else '🟡'} |"
    )
    lines[lines.index("| Execution optimization | ✅ if execution closed else 🟡 |")] = (
        f"| Execution optimization | {'✅' if exec_closed else '🟡'} |"
    )

    if summary is not None and not summary.empty:
        lines += ["## Factor summary", "", _df_to_md(summary), ""]

    if mechanism is not None and not mechanism.empty:
        lines += ["## Mechanism", "", _df_to_md(mechanism), ""]

    if stability is not None and not stability.empty:
        lines += ["## Yearly stability", "", _df_to_md(stability), ""]

    if execution is not None and not execution.empty:
        lines += ["## Execution", "", _df_to_md(execution), ""]

    if card.do_not:
        lines += ["## Do not", ""]
        for x in card.do_not:
            lines.append(f"- {x}")
        lines.append("")

    if card.notes:
        lines += ["## Notes", ""]
        for n in card.notes:
            lines.append(f"- {n}")
        lines.append("")

    if gaps:
        lines += ["## Open gaps", ""]
        for g in gaps:
            lines.append(f"- {g}")
        lines.append("")

    lines += [
        "## Artifacts",
        "",
        "```",
        "factor_report.md",
        "metrics.json",
        "factor_summary.csv",
        "mechanism.csv / mechanism_analysis.csv",
        "stability.csv / yearly_stability.csv",
        "execution_summary.csv",
        "artifacts/",
        "figures/",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_pack(
    card: FactorCard,
    *,
    summary: pd.DataFrame,
    mechanism: Optional[pd.DataFrame] = None,
    stability: Optional[pd.DataFrame] = None,
    execution: Optional[pd.DataFrame] = None,
    metrics_extra: Optional[Dict[str, Any]] = None,
    research_highlights: Optional[Dict[str, Any]] = None,
    production_highlights: Optional[Dict[str, Any]] = None,
    gaps: Optional[Sequence[str]] = None,
    root: Path = FACTORS_ROOT,
    long_form_report_src: Optional[Path] = None,
) -> Path:
    """Materialize a full factor research pack."""
    out = ensure_pack_dirs(card.factor_id, root)

    _write_df(summary, out / "factor_summary.csv")
    if mechanism is not None:
        _write_df(mechanism, out / "mechanism.csv")
        _write_df(mechanism, out / "mechanism_analysis.csv")
    if stability is not None:
        _write_df(stability, out / "stability.csv")
        _write_df(stability, out / "yearly_stability.csv")
    if execution is not None:
        _write_df(execution, out / "execution_summary.csv")

    metrics = build_metrics_bundle(
        factor=card.factor_id,
        category=card.category,
        period=card.period,
        universe=card.universe,
        summary=summary,
        research_highlights=research_highlights,
        production_highlights=production_highlights,
        extra={
            "status": card.status,
            "next": card.next_step,
            "source_column": card.source_column,
            "frozen_formula": card.frozen_formula,
            **(metrics_extra or {}),
        },
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    report = render_factor_report_md(
        card,
        summary=summary,
        mechanism=mechanism,
        stability=stability,
        execution=execution,
        gaps=gaps,
    )
    (out / "factor_report.md").write_text(report, encoding="utf-8")

    card_path = out / "factor_card.yaml"
    # Minimal YAML without PyYAML dependency
    card_path.write_text(_card_to_yaml(card), encoding="utf-8")

    readme = "\n".join(
        [
            f"# Factors / {card.factor_id}",
            "",
            f"Status: `{card.status}`  ",
            f"Category: `{card.category}`  ",
            f"Next: `{card.next_step}`",
            "",
            "Generated by `factor_report_generator.py` (Factor Report Template v1).",
            "",
            "Main pack files: `factor_report.md`, `metrics.json`, `*.csv`, `artifacts/`.",
            "",
        ]
    )
    if long_form_report_src is not None:
        readme += f"Long-form essay (optional): `{long_form_report_src.as_posix()}`\n"
    (out / "README.md").write_text(readme, encoding="utf-8")

    sync_aliases(out)
    mirror_to_artifacts(out)
    return out


def _card_to_yaml(card: FactorCard) -> str:
    def q(s: str) -> str:
        if "\n" in s or ":" in s or s.strip() != s:
            return ">\n  " + s.replace("\n", "\n  ").strip()
        return s

    lines = [
        f"factor_id: {card.factor_id}",
        f"display_name: {card.display_name}",
        f"category: {card.category}",
        f"status: {card.status}",
        f"period: {card.period}",
        f"universe: {card.universe}",
        f"source_column: {card.source_column}",
        f"frozen_formula: {str(card.frozen_formula).lower()}",
        f"next_step: {card.next_step}",
        "hypothesis: |",
    ]
    for ln in card.hypothesis.strip().splitlines() or [""]:
        lines.append(f"  {ln}")
    lines.append("do_not:")
    for x in card.do_not or ["none"]:
        lines.append(f"  - {x}")
    lines.append("notes:")
    for n in card.notes or ["none"]:
        lines.append(f"  - {n}")
    lines.append("")
    return "\n".join(lines)


def validate_pack(factor_id: str, root: Path = FACTORS_ROOT) -> Dict[str, Any]:
    """Return checklist for a factor pack."""
    out = factor_dir(factor_id, root)
    required = {
        "factor_report.md": (out / "factor_report.md").exists(),
        "metrics.json": (out / "metrics.json").exists(),
        "factor_summary.csv": (out / "factor_summary.csv").exists(),
        "mechanism": (out / "mechanism.csv").exists() or (out / "mechanism_analysis.csv").exists(),
        "stability": (out / "stability.csv").exists() or (out / "yearly_stability.csv").exists(),
        "execution_summary.csv": (out / "execution_summary.csv").exists(),
        "factor_card.yaml": (out / "factor_card.yaml").exists(),
        "artifacts/": (out / "artifacts").is_dir(),
    }
    missing = [k for k, v in required.items() if not v]
    return {
        "factor_id": factor_id,
        "path": str(out),
        "ok": len(missing) == 0,
        "checks": required,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Built-in assemblers: TGD20 (frozen) + FlowDensity20 (Phase 2)
# ---------------------------------------------------------------------------


def assemble_tgd20(root: Path = FACTORS_ROOT) -> Path:
    src = REPO_ROOT / "research" / "reports" / "tgd_v1" / "export"
    summary = pd.read_csv(src / "factor_summary.csv")
    mechanism = pd.read_csv(src / "mechanism_analysis.csv")
    stability = pd.read_csv(src / "yearly_stability.csv")
    execution = pd.read_csv(src / "execution_summary.csv")

    card = FactorCard(
        factor_id="TGD20",
        display_name="TGD20",
        category="temporal_information",
        status="validated_single_factor",
        hypothesis=(
            "Abnormal downside timing residual (εd ⊥ εu, MA20) predicts next-day "
            "cross-sectional returns. Alpha is temporal residual information, "
            "not raw Gu/Gd or τ=Gd−Gu."
        ),
        period="2022-01-28_2025-12-31",
        universe="ALL",
        source_column="TGD20",
        frozen_formula=True,
        next_step="factor_combination_candidate",
        do_not=[
            "retune MA window (MA10/30/60)",
            "change residual controls",
            "replace residual with ML",
            "mine a new TGD variant under this id",
        ],
        notes=[
            "Template instance for Factor Report Generator v1.",
            "Long-form Chinese report remains under research/reports/tgd_v1/.",
        ],
    )

    # Prefer existing export metrics highlights if present
    research_hl = None
    production_hl = None
    metrics_src = src / "metrics.json"
    if metrics_src.exists():
        old = json.loads(metrics_src.read_text(encoding="utf-8"))
        research_hl = old.get("research_score")
        production_hl = old.get("production_score")

    out = write_pack(
        card,
        summary=summary,
        mechanism=mechanism,
        stability=stability,
        execution=execution,
        research_highlights=research_hl,
        production_highlights=production_hl,
        root=root,
        long_form_report_src=Path(
            "research/reports/tgd_v1/日内分钟收益率时序特征_TGD20因子研究报告.md"
        ),
    )

    # Sync figures from export if available
    fig_src = src / "figures"
    if fig_src.is_dir():
        for p in fig_src.glob("*"):
            if p.is_file():
                _copy_if_exists(p, out / "figures" / p.name)
    return out


def assemble_flow_density20(root: Path = FACTORS_ROOT) -> Path:
    """Assemble FlowDensity20 pack from existing L2 confirmation/validation artifacts."""
    base = REPO_ROOT / "research" / "reports" / "l2_flow_density_v1"
    neut = pd.read_csv(base / "validation_v1" / "neutralization_ladder.csv")
    yearly = pd.read_csv(base / "confirmation_yearly_ic.csv")
    verdict = json.loads((base / "confirmation_verdict.json").read_text(encoding="utf-8"))
    conf = verdict.get("confirmation", {})

    period = "2022-01-28_2025-12-31"
    factor_col = "net_active_flow_mktcap_20d"
    factor_id = "FlowDensity20"

    # Map neutralization ladder → factor_summary
    rows = []
    for _, r in neut.iterrows():
        rows.append(
            {
                "factor": factor_id,
                "period": period,
                "universe": "ALL",
                "mode": str(r["mode"]),
                "rank_ic": float(r["rank_ic"]),
                "annu_ic": float(r["rank_ic"]) * np.sqrt(250.0),
                "icir": float(r["icir"]),
                "hl_annu_ret": float(r["hl_annu_ret"]),
                "hl_sharpe": float(r["hl_sharpe"]),
                "hl_mdd": float(r["hl_mdd"]),
                "daily_turnover": float(r["daily_turnover_hl"]),
                "implied_annu_fee": float(r["implied_annu_fee"]),
                "net_sharpe": float(r["net_sharpe_15bp"]),
                "monotonicity": np.nan,
                "direction": int(r["direction"]) if pd.notna(r.get("direction")) else 1,
            }
        )
    summary = pd.DataFrame(rows)

    mech_path = base / "mechanism" / "mechanism.csv"
    mech_verdict_path = base / "mechanism" / "mechanism_verdict.json"
    amount_orth_path = base / "mechanism" / "mechanism_amount_neutral.csv"
    amount_orth_verdict_path = base / "mechanism" / "amount_orth_verdict.json"
    mech_closed = False
    mech_verdict: Dict[str, Any] = {}
    amount_orth: Dict[str, Any] = {}
    if mech_path.exists():
        mechanism = pd.read_csv(mech_path)
        mech_closed = True
        if mech_verdict_path.exists():
            mech_verdict = json.loads(mech_verdict_path.read_text(encoding="utf-8"))
    else:
        mechanism = pd.DataFrame(
            [
                {
                    "signal": "residual_vs_Base3",
                    "family": "stack_gate",
                    "rank_ic": np.nan,
                    "icir": np.nan,
                    "hl_sharpe": np.nan,
                    "net_sharpe": np.nan,
                    "note": f"residual_ic_t={conf.get('residual_ic_t_base3')}",
                },
                {
                    "signal": "residual_vs_cn_voi_shock",
                    "family": "stack_gate",
                    "rank_ic": np.nan,
                    "icir": np.nan,
                    "hl_sharpe": np.nan,
                    "net_sharpe": np.nan,
                    "note": f"residual_ic_t={conf.get('residual_ic_t_voi')}",
                },
            ]
        )

    if amount_orth_path.exists():
        orth_tbl = pd.read_csv(amount_orth_path)
        # Prefer amount-orth table as primary mechanism view for pack; keep full mech as appendix
        mechanism = orth_tbl.copy()
        mech_closed = True
        if amount_orth_verdict_path.exists():
            amount_orth = json.loads(amount_orth_verdict_path.read_text(encoding="utf-8"))
            mech_verdict = {**mech_verdict, **amount_orth}

    stability = yearly.rename(columns={"ic": "rank_ic"}).copy()
    stability.insert(0, "factor", factor_id)

    # Execution not yet optimized (no buffer grid) — record baseline investability row
    si = summary.loc[summary["mode"] == "size_industry"].iloc[0]
    execution = pd.DataFrame(
        [
            {
                "label": "size_industry|daily|no_buffer",
                "status": "baseline_only",
                "rank_ic": float(si["rank_ic"]),
                "icir": float(si["icir"]),
                "hl_sharpe": float(si["hl_sharpe"]),
                "hl_mdd": float(si["hl_mdd"]),
                "daily_turnover": float(si["daily_turnover"]),
                "net_sharpe": float(si["net_sharpe"]),
                "note": "Execution buffer/grid not closed; NetSharpe from neut ladder @15bp",
            }
        ]
    )

    exec_path = base / "execution" / "execution_summary.csv"
    exec_closed = False
    production_best = None
    if exec_path.exists():
        ranked = pd.read_csv(exec_path)
        if len(ranked) and "net_sharpe" in ranked.columns:
            best = ranked.dropna(subset=["net_sharpe"]).iloc[0]
            execution = ranked.head(15).copy()
            if "status" not in execution.columns:
                execution.insert(1, "status", "optimized")
            else:
                execution["status"] = "optimized"
            exec_closed = True
            production_best = {
                "label": str(best.get("label")),
                "net_sharpe": float(best["net_sharpe"]),
                "gross_sharpe": float(best.get("gross_sharpe", np.nan)),
                "daily_turnover": float(best.get("daily_turnover", np.nan)),
            }
            # Also append execution_best mode onto summary for metrics bundle
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "factor": factor_id,
                                "period": period,
                                "universe": "ALL",
                                "mode": "execution_best",
                                "rank_ic": float(best.get("rank_ic", si["rank_ic"])),
                                "annu_ic": float(best.get("rank_ic", si["rank_ic"])) * np.sqrt(250.0),
                                "icir": float(best.get("icir", si["icir"])),
                                "hl_annu_ret": float(best.get("gross_annu_ret", si["hl_annu_ret"])),
                                "hl_sharpe": float(best.get("gross_sharpe", si["hl_sharpe"])),
                                "hl_mdd": float(best.get("mdd_net", si["hl_mdd"])),
                                "daily_turnover": float(best.get("daily_turnover", si["daily_turnover"])),
                                "implied_annu_fee": float(
                                    best.get("implied_annu_fee", si["implied_annu_fee"])
                                ),
                                "net_sharpe": float(best["net_sharpe"]),
                                "monotonicity": np.nan,
                                "direction": int(best.get("direction", 1)),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    promote = bool(mech_verdict.get("promote_to_validated_single_factor"))
    # Amount-orth is attribution: never promote from entangled interaction alone
    if amount_orth.get("attribution_case") in (
        "case_interaction_entangled",
        "case2_mostly_anti_amount",
    ):
        promote = False
    status = (
        "validated_single_factor"
        if (promote and exec_closed and mech_closed)
        else "validated_single_factor_candidate"
    )
    frozen = bool(mech_verdict.get("freeze_formula")) and promote
    combo_ready = status == "validated_single_factor"
    category = str(
        amount_orth.get("category")
        or amount_orth.get("mechanism_class")
        or "flow_information"
    )

    if amount_orth:
        next_step = "orthogonality_vs_TGD20_raw_and_perp"
    elif status == "validated_single_factor":
        next_step = "orthogonality_vs_TGD20"
    elif exec_closed and not mech_closed:
        next_step = "mechanism_validation"
    elif mech_closed and not promote:
        next_step = "mechanism_gates_incomplete"
    else:
        next_step = "close_execution_then_mechanism_then_orthogonality"

    card = FactorCard(
        factor_id=factor_id,
        display_name="FlowDensity20 (net_active_flow_mktcap_20d)",
        category=category,
        status=status,
        hypothesis=(
            "Net active flow / mktcap (20d) is a Flow × Liquidity interaction: "
            "direction entangled with anti-amount / low-activity. Not pure flow; "
            "complementary to TGD (temporal) but must be treated as microstructure+liquidity."
        ),
        period=period,
        universe="ALL",
        source_column=factor_col,
        frozen_formula=frozen,
        next_step=next_step,
        do_not=[
            "jump to TGD×Flow composite before orthogonality",
            "treat CSI300 standalone as validation target (known concentration)",
            "auto-freeze from amount-orth attribution",
            "rename to pure Flow without documenting liquidity channel",
        ],
        notes=[
            "Assembled from l2_flow_density_v1 confirmation + validation neut ladder.",
            "Soft flag: broad-universe / small-cap concentrated.",
            "Verdict: confirm_pass_enhancer.",
            f"Mechanism: {mech_verdict.get('verdict', amount_orth.get('attribution_case', 'pending'))}.",
            amount_orth.get("interpretation")
            or mech_verdict.get("interpretation", ""),
            f"mechanism_class={amount_orth.get('mechanism_class', 'pending')}",
        ],
    )

    gaps = []
    if not exec_closed:
        gaps.append(
            "execution optimization incomplete — run run_flow_density_execution_opt_v1.py"
        )
    if not mech_closed:
        gaps.append(
            "mechanism analysis incomplete — run run_flow_density_mechanism_v1.py"
        )
    if not amount_orth:
        gaps.append(
            "amount-orthogonalization attribution incomplete — run run_flow_density_amount_orth_v1.py"
        )
    gaps.extend(
        [
            "long-form Chinese research essay not yet written",
            "figures/ not yet standardized",
            "orthogonality vs TGD20 not yet run (raw + Flow_perp_Amount)",
        ]
    )

    out = write_pack(
        card,
        summary=summary,
        mechanism=mechanism,
        stability=stability,
        execution=execution,
        research_highlights={
            "RankIC": float(si["rank_ic"]),
            "ICIR": float(si["icir"]),
            "ICIR_size_industry": float(si["icir"]),
            "Sharpe": float(summary.loc[summary["mode"] == "raw", "hl_sharpe"].iloc[0]),
            "Sharpe_size_industry": float(si["hl_sharpe"]),
            "MDD": float(si["hl_mdd"]),
            "year_ic_pos_ratio": float(conf.get("year_ic_pos_ratio", 1.0)),
            "Flow_perp_Amount_ICIR": (amount_orth.get("icir") or {}).get("Flow_perp_Amount"),
            "Amount_ICIR": (amount_orth.get("icir") or {}).get("Amount"),
        },
        production_highlights={
            "NetSharpe_size_industry": float(si["net_sharpe"]),
            "NetSharpe_execution_best": None
            if production_best is None
            else production_best["net_sharpe"],
            "execution_best_label": None if production_best is None else production_best["label"],
            "Turnover_raw": float(summary.loc[summary["mode"] == "raw", "daily_turnover"].iloc[0]),
            "Turnover_execution_best": None
            if production_best is None
            else production_best["daily_turnover"],
            "annu_one_way_turnover_pct": float(conf.get("annu_one_way_turnover", np.nan)),
            "capacity_cny_approx": conf.get("capacity_cny_approx"),
            "execution": "closed" if exec_closed else "incomplete",
            "mechanism": "closed" if mech_closed else "incomplete",
            "amount_orth": "closed" if amount_orth else "incomplete",
        },
        metrics_extra={
            "source_column": factor_col,
            "confirmation_verdict": verdict.get("verdict"),
            "residual_ic_t_base3": conf.get("residual_ic_t_base3"),
            "residual_ic_t_voi": conf.get("residual_ic_t_voi"),
            "stack_icir_uplift": conf.get("stack_icir_uplift"),
            "formula_frozen": frozen,
            "production_deferred": True,
            "combination_candidate": combo_ready,
            "mechanism_verdict": mech_verdict.get("verdict"),
            "mechanism_gates": mech_verdict.get("gates"),
            "mechanism_class": amount_orth.get("mechanism_class"),
            "attribution_case": amount_orth.get("attribution_case"),
            "category": category,
            "amount_orth": amount_orth.get("icir"),
            "cs_corr_flow_amount": amount_orth.get("cs_corr_flow_amount"),
        },
        gaps=gaps,
        root=root,
        long_form_report_src=Path("research/reports/l2_flow_density_v1/README.md"),
    )
    return out


def write_factors_index(root: Path = FACTORS_ROOT) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("_"):
            continue
        card = p / "factor_card.yaml"
        metrics = p / "metrics.json"
        status = "unknown"
        category = "unknown"
        if metrics.exists():
            m = json.loads(metrics.read_text(encoding="utf-8"))
            status = (m.get("extra") or {}).get("status", status)
            category = m.get("category", category)
        v = validate_pack(p.name, root)
        rows.append(
            {
                "factor_id": p.name,
                "category": category,
                "status": status,
                "pack_complete": v["ok"],
                "missing": ";".join(v["missing"]),
                "path": str(p.relative_to(REPO_ROOT)),
            }
        )
    df = pd.DataFrame(rows)
    path = root / "index.csv"
    df.to_csv(path, index=False)

    md = [
        "# Factor Research Packs (Template v1)",
        "",
        "Goal: every single-factor study lands in the same machine-readable pack.",
        "",
        "## Layout",
        "",
        "```",
        "research/reports/factors/",
        "  TGD20/                 # temporal_information — frozen template",
        "  FlowDensity20/         # flow_information — Phase 2",
        "  FactorCutting/         # reserved",
        "  APM/                   # reserved",
        "  index.csv",
        "  ROADMAP.md",
        "```",
        "",
        "## Pack contents (required)",
        "",
        "- `factor_report.md` — standard pack report",
        "- `metrics.json` — research + production scores",
        "- `factor_summary.csv` — neutralization / mode ladder",
        "- `mechanism.csv` — mechanism / integrity variants",
        "- `stability.csv` — yearly RankIC",
        "- `execution_summary.csv` — investability / buffer grid",
        "- `factor_card.yaml` — status card",
        "- `artifacts/` — mirrored CSVs/JSON",
        "",
        "## Current index",
        "",
        _df_to_md(df) if not df.empty else "_empty_",
        "",
        "## Generator",
        "",
        "```bash",
        "python run_factor_report_generator_v1.py --all",
        "```",
        "",
        "Do **not** retune TGD20. Next research work: close FlowDensity execution gaps, then orthogonality.",
        "",
    ]
    (root / "README.md").write_text("\n".join(md), encoding="utf-8")
    return path
