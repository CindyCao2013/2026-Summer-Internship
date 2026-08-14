#!/usr/bin/env python3
"""Factor Report Generator v2 — schema-driven Research Pack renderer.

Milestone 1C/1D/1D.6: harvest existing artifacts into Template v2 packs.

Does NOT recompute factors, change formulas, or write Registry.
Must NOT special-case factor_id in Python:
  - narrative / code_map → factor_specs/{id}_report_content.yaml
  - artifact copies → factor_specs/{id}.yaml → artifacts:
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent
SCHEMA_DIR = REPO_ROOT / "docs" / "schemas"
SPECS_DIR = REPO_ROOT / "factor_specs"
FACTORS_ROOT = REPO_ROOT / "research" / "reports" / "factors"


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_registries() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return (
        _load_yaml(SCHEMA_DIR / "metric_registry.yaml"),
        _load_yaml(SCHEMA_DIR / "chart_registry.yaml"),
        _load_yaml(SCHEMA_DIR / "factor_report.schema.yaml"),
    )


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _na(reason: str) -> Dict[str, Any]:
    return {"value": None, "source": None, "missing_reason": reason}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return None if np.isnan(x) or np.isinf(x) else x
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _val(value: Any, source: str) -> Dict[str, Any]:
    v = _jsonable(value)
    if v is None:
        return _na("not_in_artifacts")
    return {"value": v, "source": source}


def flatten_metric_defs(metric_reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gname, g in (metric_reg.get("groups") or {}).items():
        if gname in ("risk_adjustment", "execution", "mechanism"):
            continue
        for m in g.get("metrics") or []:
            out.append({**m, "group": gname})
    return out


def normalize_stability_df(df: pd.DataFrame) -> pd.DataFrame:
    """Accept year/period and n/n_days variants from heterogeneous harvests."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "period" not in out.columns and "year" in out.columns:
        out["period"] = out["year"].astype(str)
    if "n_days" not in out.columns and "n" in out.columns:
        out["n_days"] = out["n"]
    if "pos_ic_frac" not in out.columns:
        out["pos_ic_frac"] = np.nan
    if "icir" not in out.columns:
        out["icir"] = np.nan
    if "rank_ic" not in out.columns:
        out["rank_ic"] = np.nan
    return out


def _safe_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_column_map(pack_dir: Path) -> Dict[str, Tuple[Any, str]]:
    """Map lowercase column name → (value, source) from primary summary rows."""
    colmap: Dict[str, Tuple[Any, str]] = {}

    def ingest_row(row: Dict[str, Any], source: str) -> None:
        for k, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            colmap[str(k).lower()] = (v, source)

    summary = pack_dir / "factor_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        # Prefer size_industry for headline harvest; keep raw available separately
        for mode in ("size_industry", "raw", "execution_best"):
            sub = df[df["mode"] == mode] if "mode" in df.columns else df
            if not sub.empty:
                ingest_row(sub.iloc[0].to_dict(), f"factor_summary.csv:{mode}")

    metrics_path = pack_dir / "metrics.json"
    if metrics_path.exists():
        mj = json.loads(metrics_path.read_text(encoding="utf-8"))
        rs = mj.get("research_score") or {}
        for k, v in rs.items():
            colmap[k.lower()] = (v, "metrics.json:research_score")
        ps = mj.get("production_score") or {}
        for k, v in ps.items():
            colmap[k.lower()] = (v, "metrics.json:production_score")

    stab = pack_dir / "yearly_stability.csv"
    if stab.exists():
        sdf = pd.read_csv(stab)
        years = sdf[sdf["kind"] == "year"] if "kind" in sdf.columns else sdf
        if not years.empty and "pos_ic_frac" in years.columns:
            colmap["pos_ic_frac"] = (float(years["pos_ic_frac"].mean()), "yearly_stability.csv:mean_pos")
            colmap["ic_positive_ratio"] = colmap["pos_ic_frac"]

    return colmap


def build_metric_union(metric_reg: Dict[str, Any], colmap: Dict[str, Tuple[Any, str]]) -> Dict[str, Any]:
    union: Dict[str, Any] = {}
    for m in flatten_metric_defs(metric_reg):
        mid = m["id"]
        aliases = [a.lower() for a in (m.get("aliases") or [])] + [mid.lower()]
        hit = None
        for a in aliases:
            if a in colmap:
                hit = colmap[a]
                break
        # Special mappings
        if hit is None and mid == "RankICIR" and "icir" in colmap:
            hit = colmap["icir"]
        if hit is None and mid == "Annualized_RankIC" and "annu_ic" in colmap:
            hit = colmap["annu_ic"]
        if hit is None and mid == "HL_Sharpe" and "hl_sharpe" in colmap:
            hit = colmap["hl_sharpe"]
        if hit is None and mid == "HL_return" and "hl_annu_ret" in colmap:
            hit = colmap["hl_annu_ret"]
        if hit is None and mid == "gross_Sharpe" and "hl_sharpe" in colmap:
            hit = colmap["hl_sharpe"]
        if hit is None and mid == "Sharpe" and "hl_sharpe" in colmap:
            hit = colmap["hl_sharpe"]
        if hit is None and mid == "net_Sharpe" and "net_sharpe" in colmap:
            hit = colmap["net_sharpe"]
        if hit is None and mid == "implied_fee" and "implied_annu_fee" in colmap:
            hit = colmap["implied_annu_fee"]
        if hit is None and mid == "MDD" and "hl_mdd" in colmap:
            hit = colmap["hl_mdd"]
        if hit is None and mid == "annual_return" and "hl_annu_ret" in colmap:
            hit = colmap["hl_annu_ret"]
        if hit is None and mid == "daily_turnover" and "daily_turnover" in colmap:
            hit = colmap["daily_turnover"]

        if hit is None:
            if m.get("missing_ok") or mid in ("IC", "ICIR", "IC_tstat", "IC_mean", "IC_std", "Sortino", "Calmar", "volatility", "cumulative_return", "excess_return", "signal_decay", "long_leg_return", "short_leg_return", "annual_turnover", "stability_score", "Annualized_IC"):
                reason = "not_computed" if mid in ("IC", "ICIR", "IC_tstat", "Sortino", "Calmar") else "not_in_artifacts"
                union[mid] = _na(reason)
            else:
                union[mid] = _na("not_in_artifacts")
        else:
            union[mid] = _val(hit[0], hit[1])
            if mid == "RankICIR":
                union[mid]["note"] = "Mapped from legacy icir (Spearman-based)"
            if mid == "RankIC" and "IC" in union and union["IC"]["value"] is None:
                # Document that legacy packs often stored RankIC under ic
                pass
    return union


def build_risk_ladder(pack_dir: Path) -> List[Dict[str, Any]]:
    path = pack_dir / "factor_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for mode in ("raw", "size", "industry", "size_industry"):
        sub = df[df["mode"] == mode]
        if sub.empty:
            rows.append({"mode": mode, "missing_reason": "not_in_artifacts"})
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "mode": mode,
                "RankIC": float(r["rank_ic"]),
                "RankICIR": float(r["icir"]),
                "HL_Sharpe": float(r["hl_sharpe"]),
                "MDD": float(r["hl_mdd"]),
                "net_Sharpe": float(r["net_sharpe"]) if pd.notna(r.get("net_sharpe")) else None,
                "source": "factor_summary.csv",
            }
        )
    return rows


def ensure_dirs(pack_dir: Path) -> None:
    for d in ("charts", "mechanism", "execution", "diagnostics", "artifacts"):
        (pack_dir / d).mkdir(parents=True, exist_ok=True)


def prepare_charts(
    pack_dir: Path,
    chart_reg: Dict[str, Any],
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    execution: pd.DataFrame,
    content: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Copy legacy figures + generate missing standard charts. Returns missing list."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    missing: List[str] = []
    content = content or {}
    stability = normalize_stability_df(stability)

    def resolve_alias(aliases: List[str]) -> Optional[Path]:
        for a in aliases:
            p = pack_dir / a
            if p.exists():
                return p
            p2 = pack_dir / "figures" / Path(a).name
            if p2.exists():
                return p2
        return None

    for c in chart_reg.get("charts") or []:
        dest = pack_dir / c["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        aliases = [c["path"], Path(c["path"]).name] + list(c.get("aliases") or [])
        src = resolve_alias(aliases)
        if src is not None and src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
            continue
        if dest.exists():
            continue

        cid = c["id"]
        if cid == "construction_diagram":
            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.axis("off")
            steps = list(content.get("construction_steps") or [])
            title = content.get("construction_title") or f"Factor construction ({pack_dir.name})"
            if not steps:
                steps = [pack_dir.name, "signal", "neutralize", "smooth", "shift=1"]
            for i, s in enumerate(steps):
                ax.add_patch(plt.Rectangle((i * 1.6, 0.35), 1.4, 0.4, fill=False, lw=1.5))
                ax.text(i * 1.6 + 0.7, 0.55, s, ha="center", va="center", fontsize=8)
                if i < len(steps) - 1:
                    ax.annotate("", xy=((i + 1) * 1.6, 0.55), xytext=(i * 1.6 + 1.4, 0.55), arrowprops=dict(arrowstyle="->"))
            ax.set_xlim(-0.1, len(steps) * 1.6)
            ax.set_ylim(0, 1.2)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(dest, dpi=140)
            plt.close(fig)
        elif cid == "neutralization_compare" and not summary.empty:
            modes = ["raw", "size", "industry", "size_industry"]
            sub = summary[summary["mode"].isin(modes)].copy()
            title = "Neutralization ladder (not best-only)"
            if sub.empty:
                # e.g. universe diagnostics encoded as mode
                sub = summary.copy()
                title = "Mode / universe ladder (available artifacts)"
            if sub.empty or "icir" not in sub.columns:
                missing.append(c["path"])
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(sub))
            ax.bar(x - 0.2, sub["icir"], 0.4, label="RankICIR")
            if "hl_sharpe" in sub.columns:
                ax.bar(x + 0.2, sub["hl_sharpe"], 0.4, label="H-L Sharpe")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["mode"].astype(str).tolist(), rotation=20, ha="right")
            ax.legend()
            ax.set_title(title)
            ax.set_ylabel("value")
            fig.tight_layout()
            fig.savefig(dest, dpi=140)
            plt.close(fig)
        elif cid == "yearly_stability" and not stability.empty:
            years = stability[stability["kind"] == "year"] if "kind" in stability.columns else stability
            if years.empty:
                years = stability
            if "period" not in years.columns or "rank_ic" not in years.columns:
                missing.append(c["path"])
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(years["period"].astype(str), years["rank_ic"])
            ax.set_title("RankIC stability (yearly or available blocks)")
            ax.set_ylabel("RankIC")
            ax.axhline(0, color="black", lw=0.8)
            fig.tight_layout()
            fig.savefig(dest, dpi=140)
            plt.close(fig)
        elif cid == "turnover_cost" and execution.empty and not summary.empty:
            # fallback: turnover vs net sharpe across summary modes
            sub = summary.copy()
            if "daily_turnover" not in sub.columns:
                missing.append(c["path"])
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            y = sub["net_sharpe"] if "net_sharpe" in sub.columns else sub.get("hl_sharpe")
            ax.scatter(sub["daily_turnover"], y)
            for _, r in sub.iterrows():
                yy = r["net_sharpe"] if pd.notna(r.get("net_sharpe")) else r.get("hl_sharpe")
                ax.annotate(str(r.get("mode", "")), (r["daily_turnover"], yy), fontsize=7)
            ax.set_xlabel("daily turnover")
            ax.set_ylabel("net/gross Sharpe")
            ax.set_title("Turnover vs Sharpe (summary modes; no execution grid)")
            fig.tight_layout()
            fig.savefig(dest, dpi=140)
            plt.close(fig)
        elif cid == "turnover_cost" and not execution.empty:
            top = execution.head(8).copy()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(top["daily_turnover"], top["net_sharpe"])
            for _, r in top.iterrows():
                ax.annotate(str(r["label"])[:28], (r["daily_turnover"], r["net_sharpe"]), fontsize=6)
            ax.set_xlabel("daily turnover")
            ax.set_ylabel("net Sharpe")
            ax.set_title("Execution grid: turnover vs Net Sharpe (top rows)")
            fig.tight_layout()
            fig.savefig(dest, dpi=140)
            plt.close(fig)
        elif cid in ("ic_curve", "decile_return", "cumulative_long_short"):
            # try hml_curve as cumulative alias already in registry
            missing.append(c["path"] + " (no legacy figure found)")
        else:
            missing.append(c["path"])

    return missing


def copy_mechanism_execution(pack_dir: Path) -> None:
    pairs = [
        ("mechanism.csv", "mechanism/mechanism.csv"),
        ("mechanism_analysis.csv", "mechanism/mechanism_analysis.csv"),
        ("execution_summary.csv", "execution/execution_summary.csv"),
        ("yearly_stability.csv", "diagnostics/yearly_stability.csv"),
        ("stability.csv", "diagnostics/stability.csv"),
        ("factor_summary.csv", "artifacts/factor_summary.csv"),
        ("diagnostics_universe_ladder.csv", "diagnostics/universe_ladder.csv"),
        ("diagnostics_ic_decay.csv", "diagnostics/ic_decay.csv"),
    ]
    for src_name, dst_rel in pairs:
        src = pack_dir / src_name
        if src.exists():
            dst = pack_dir / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _normalize_also_dest(entry: Dict[str, Any]) -> List[str]:
    also = entry.get("also_dest")
    if also is None:
        return []
    if isinstance(also, str):
        return [also]
    return [str(x) for x in also]


def apply_artifact_copies(pack_dir: Path, spec: Dict[str, Any]) -> List[str]:
    """Copy harvest artifacts declared in factor_specs/{id}.yaml → artifacts:.

    No factor_id branches. Missing sources are skipped (not fabricated).
    Returns list of applied \"src -> dest\" strings for diagnostics.
    """
    applied: List[str] = []
    arts = spec.get("artifacts") or {}
    if not isinstance(arts, dict):
        return applied

    for entry in arts.get("copy") or []:
        src_rel = entry.get("src")
        dest_rel = entry.get("dest")
        if not src_rel or not dest_rel:
            continue
        src = REPO_ROOT / src_rel
        if not src.exists():
            continue
        dest = pack_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        applied.append(f"{src_rel} -> {dest_rel}")
        for also_rel in _normalize_also_dest(entry):
            also_dest = pack_dir / also_rel
            also_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, also_dest)
            applied.append(f"{src_rel} -> {also_rel}")

    for entry in arts.get("pack_local_copy") or []:
        src_rel = entry.get("src")
        dest_rel = entry.get("dest")
        if not src_rel or not dest_rel:
            continue
        src = pack_dir / src_rel
        if not src.exists():
            continue
        dest = pack_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        applied.append(f"pack:{src_rel} -> {dest_rel}")

    return applied


def render_md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def render_appendix_b(factor_id: str, content: Dict[str, Any]) -> str:
    """Appendix B from standard schema rows + content.code_map (no hardcoded factor paths)."""
    rows: List[List[str]] = [
        ["Report content", f"`factor_specs/{factor_id}_report_content.yaml`"],
        ["Factor spec", f"`factor_specs/{factor_id}.yaml`"],
        ["Metric registry", "`docs/schemas/metric_registry.yaml`"],
        ["Chart registry", "`docs/schemas/chart_registry.yaml`"],
        ["Pack schema", "`docs/schemas/factor_report.schema.yaml`"],
    ]
    for item in content.get("code_map") or []:
        name = str(item.get("item") or "Item")
        path = str(item.get("path") or "")
        # Preserve author formatting when path already embeds backticks
        if path and "`" not in path:
            path = f"`{path}`"
        rows.append([name, path])
    return render_md_table(["Item", "Path"], rows)


def render_factor_report(
    content: Dict[str, Any],
    union: Dict[str, Any],
    risk_ladder: List[Dict[str, Any]],
    mechanism_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    missing_charts: List[str],
    factor_id: str,
) -> str:
    chapters = []

    def u(mid: str) -> str:
        e = union.get(mid) or {}
        if e.get("value") is None:
            return f"N/A ({e.get('missing_reason', 'missing')})"
        return _fmt(e["value"])

    # Headline metrics table
    headline_ids = ["RankIC", "IC", "RankICIR", "ICIR", "IC_tstat", "IC_positive_ratio", "Annualized_RankIC", "Sharpe", "net_Sharpe", "MDD", "daily_turnover", "monotonicity"]
    hrows = [[mid, u(mid)] for mid in headline_ids]

    risk_rows = []
    for r in risk_ladder:
        if "missing_reason" in r:
            risk_rows.append([r["mode"], "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            risk_rows.append(
                [
                    r["mode"],
                    _fmt(r["RankIC"]),
                    _fmt(r["RankICIR"]),
                    _fmt(r["HL_Sharpe"]),
                    _fmt(r["MDD"]),
                    _fmt(r.get("net_Sharpe")),
                ]
            )

    mech_rows = []
    for mr in content.get("mechanism_rows") or []:
        mech_rows.append([mr["hypothesis"], mr["test"], mr["result"], mr["conclusion"]])

    # Full mechanism CSV dump rows
    mech_csv_rows = []
    if not mechanism_df.empty:
        cols = list(mechanism_df.columns)
        for _, r in mechanism_df.iterrows():
            mech_csv_rows.append([_fmt(r[c], 4) if c != "signal" and c != "category" else str(r[c]) for c in cols])

    stab_rows = []
    stability_df = normalize_stability_df(stability_df)
    if not stability_df.empty:
        years = stability_df[stability_df["kind"] == "year"] if "kind" in stability_df.columns else stability_df
        for _, r in years.iterrows():
            n_days = r["n_days"] if pd.notna(r.get("n_days")) else ""
            n_str = str(int(n_days)) if n_days != "" and pd.notna(n_days) else "N/A"
            stab_rows.append(
                [str(r["period"]), _fmt(r["rank_ic"]), _fmt(r["icir"]), _fmt(r.get("pos_ic_frac")), n_str]
            )

    exec_rows = []
    if not execution_df.empty:
        for _, r in execution_df.head(10).iterrows():
            exec_rows.append(
                [
                    str(r["label"])[:40],
                    _fmt(r["gross_sharpe"]),
                    _fmt(r["net_sharpe"]),
                    _fmt(r["daily_turnover"]),
                    _fmt(r["implied_annu_fee"]),
                    _fmt(r["mdd_net"]),
                ]
            )

    dump_rows = []
    for mid, e in sorted(union.items()):
        dump_rows.append(
            [
                mid,
                _fmt(e.get("value")) if e.get("value") is not None else "N/A",
                str(e.get("source") or ""),
                str(e.get("missing_reason") or e.get("note") or ""),
            ]
        )

    missing_block = "None" if not missing_charts else "\n".join(f"- `{m}`" for m in missing_charts)

    formula = content.get("formula") or {}
    mech_chain = (content.get("mechanism_chain") or "").strip() or (
        "diagnostics → see verdict table\n"
        f"{factor_id} → accepted investable expression (sole factor_id)"
    )
    layer_note = (content.get("layer_discipline") or "").strip() or (
        f"**Factor Identity:** `{factor_id}` is the only Registry / investable factor_id. "
        "Rows in Mechanism and Execution are diagnostic variants and portfolio "
        "implementations of this signal — not separate factors."
    )

    body = f"""# {factor_id} — Factor Research Report (Template v2)

> **schema_version:** `factor_report_v2` · Research Pack (schema-driven)  
> **Harvest only** — formulas not recomputed. Metric Union: N/A never silently dropped.

{layer_note}

**Boss reading guide**

| Lens | Look at |
|------|---------|
| Research | RankIC, RankICIR, Gross Sharpe, MDD, Monotonicity |
| Admission | Net Sharpe, Turnover, Implied Fee, Execution |
| Not factors | Mechanism diagnostics · Execution implementation labels |

---

# 1. Executive Summary

{content.get('executive_summary', '').strip()}

### Core metrics (Metric Union headline)

{render_md_table(['Metric', 'Value'], hrows)}

---

# 2. Factor Thesis

{content.get('factor_thesis', '').strip()}

---

# 3. Economic Intuition

{content.get('economic_intuition', '').strip()}

---

# 4. Formula Construction

## 4.1 Raw variables

{formula.get('raw_variables', '').strip()}

## 4.2 Intermediate variables

{formula.get('intermediate_variables', '').strip()}

## 4.3 Transformations / residualization

{formula.get('transformations', '').strip()}

## 4.4 Final investable signal

{formula.get('final_signal', '').strip()}

---

# 5. Signal Pipeline

{content.get('signal_pipeline', '').strip()}

![construction](charts/construction_diagram.png)

---

# 6. Mechanism Validation

> **Layer:** diagnostic variants / signal representations that test *why* `{factor_id}` works.  
> **Not** competing `factor_id`s. Registry still has only `{factor_id}`.

{content.get('mechanism_narrative', '').strip()}

### Verdict table

{render_md_table(['Hypothesis', 'Test', 'Result', 'Conclusion'], mech_rows)}

### Mechanism chain

```
{mech_chain}
```

### Full mechanism artifact (diagnostics — not sibling factors)

{render_md_table(list(mechanism_df.columns), mech_csv_rows) if mech_csv_rows else '_mechanism.csv missing_'}

---

# 7. IC Analysis

{content.get('ic_narrative', '').strip()}

![ic_curve](charts/ic_curve.png)

---

# 8. Portfolio Analysis

{content.get('portfolio_narrative', '').strip()}

![decile](charts/decile_return.png)

![cum](charts/cumulative_long_short.png)

---

# 9. Risk Adjustment

{content.get('risk_narrative', '').strip()}

**All neutralization modes (not best-only):**

{render_md_table(['Mode', 'RankIC', 'RankICIR', 'H-L Sharpe', 'MDD', 'Net Sharpe'], risk_rows)}

![neut](charts/neutralization_compare.png)

---

# 10. Stability

{content.get('stability_narrative', '').strip()}

{render_md_table(['Year', 'RankIC', 'RankICIR', 'IC+ ratio', 'n_days'], stab_rows) if stab_rows else '_no stability_'}

![stability](charts/stability_yearly.png)

---

# 11. Execution (Portfolio Implementation)

> **Layer:** how to trade the **same** `{factor_id}` signal (rebalance / buffer / hold).  
> Labels below are implementation variants — not new factors.

{content.get('execution_narrative', '').strip()}

Top implementation rows (full grid in `execution/execution_summary.csv`):

{render_md_table(['label', 'gross Sharpe', 'net Sharpe', 'daily TO', 'implied fee', 'MDD net'], exec_rows) if exec_rows else '_no execution_'}

![turnover](charts/turnover.png)

---

# 12. Limitations

{content.get('limitations', '').strip()}

### Missing Artifacts

{missing_block}

---

# 13. Final Verdict

{content.get('final_verdict', '').strip()}

---

# Appendix A. Complete Metric Dump (union)

{render_md_table(['metric_id', 'value', 'source', 'note/missing_reason'], dump_rows)}

---

# Appendix B. Data Dictionary & Code Map

{render_appendix_b(factor_id, content)}
"""
    return body


def write_card(pack_dir: Path, content: Dict[str, Any], spec: Dict[str, Any]) -> None:
    cf = content.get("card_fields") or {}
    card = {
        "schema_version": "factor_report_v2",
        "factor_id": content.get("factor_id") or spec.get("factor_id"),
        "display_name": spec.get("display_name") or content.get("factor_id"),
        "family": spec.get("family") or ["unspecified"],
        "source": spec.get("source"),
        "data_level": spec.get("data_level"),
        "status": spec.get("status") or "testing",
        "frozen_formula": bool(spec.get("frozen_formula", False)),
        "hypothesis": cf.get("hypothesis"),
        "mechanism": cf.get("mechanism"),
        "formula": cf.get("formula"),
        "data_requirement": cf.get("data_requirement"),
        "known_failure_modes": cf.get("known_failure_modes"),
        "correlation_cluster": None,
        "data_coverage": spec.get("data_coverage"),
        "coverage_exception": bool(
            (spec.get("data_coverage") or {}).get("exception_reason")
            or (spec.get("data_coverage") or {}).get("exception")
        ),
        "admission": {"requires_manual_review": True},
        "benchmark": {
            "research": {"universe": "ALL", "period_target": "2018-01-01_2025-12-31", "horizon_days": 20},
            "production": {
                "universe": "CSI1000",
                "universe_membership": "declare",
                "period_target": "2018-01-01_2025-12-31",
                "horizon_days": 20,
                "neutralization": "industry_size",
                "cost_bp": 15,
                "note": "Pack harvests existing artifacts; Production Track re-run deferred",
            },
        },
        "do_not": cf.get("do_not") or [],
        "notes": [
            "factor_report_v2 schema-driven pack",
            "Legacy/harvest metrics; Dual Benchmark Production re-run not performed",
        ],
    }
    (pack_dir / "factor_card.yaml").write_text(
        yaml.safe_dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def write_metrics_json(
    pack_dir: Path,
    factor_id: str,
    union: Dict[str, Any],
    risk_ladder: List[Dict[str, Any]],
    summary: pd.DataFrame,
) -> None:
    def pick(mode: str) -> Dict[str, Any]:
        sub = summary[summary["mode"] == mode]
        if sub.empty:
            return {}
        r = sub.iloc[0]
        return {
            "universe": str(r["universe"]),
            "period": str(r["period"]),
            "mode": mode,
            "RankIC": float(r["rank_ic"]),
            "RankICIR": float(r["icir"]),
            "Annualized_RankIC": float(r["annu_ic"]),
            "HL_Sharpe": float(r["hl_sharpe"]),
            "annual_return": float(r["hl_annu_ret"]),
            "MDD": float(r["hl_mdd"]),
            "daily_turnover": float(r["daily_turnover"]),
            "net_Sharpe": float(r["net_sharpe"]) if pd.notna(r.get("net_sharpe")) else None,
            "monotonicity": float(r["monotonicity"]) if pd.notna(r.get("monotonicity")) else None,
        }

    raw = pick("raw")
    si = pick("size_industry")
    ex = pick("execution_best")

    payload = {
        "schema_version": "factor_report_v2",
        "factor_id": factor_id,
        "status": "pack_generated",
        "production": {
            "note": "Legacy harvest — not Protocol CSI1000 Production Track",
            "headline_mode": "size_industry" if si else "raw",
            **(si or raw),
            "execution_best_net_Sharpe": ex.get("net_Sharpe"),
            "execution_best_turnover": ex.get("daily_turnover"),
            "cost_bp_note": "execution grid round_trip_cost=15bp; some v1 implied_fee used 7.5bp",
        },
        "research": {
            "universe": raw.get("universe"),
            "period": raw.get("period"),
            "mode": "raw",
            **{k: raw[k] for k in raw if k not in ("universe", "period", "mode")},
        },
        "metric_union": union,
        "risk_ladder": risk_ladder,
    }
    (pack_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(pack_dir / "metrics.json", pack_dir / "artifacts" / "metrics.json")


def write_summary_md(pack_dir: Path, factor_id: str, union: Dict[str, Any], risk_ladder: List[Dict[str, Any]]) -> None:
    def u(mid: str) -> str:
        e = union.get(mid) or {}
        return _fmt(e["value"]) if e.get("value") is not None else f"N/A ({e.get('missing_reason')})"

    si = next((r for r in risk_ladder if r.get("mode") == "size_industry"), {})
    text = f"""# {factor_id} — Summary (Template v2)

| Field | Value |
|-------|-------|
| Status | validated (formula frozen) |
| Family | temporal_information |
| RankIC (raw harvest) | {u('RankIC')} |
| RankICIR raw / size+ind | {u('RankICIR')} / {_fmt(si.get('RankICIR'))} |
| Gross Sharpe raw / size+ind | {u('Sharpe')} / {_fmt(si.get('HL_Sharpe'))} |
| Net Sharpe (execution best) | see execution/ |
| Monotonicity | {u('monotonicity')} |
| Pearson IC | {u('IC')} |

**PM:** prefer Net Sharpe + turnover in `execution/`. **Researcher:** mechanism chain in `factor_report.md` §6.
"""
    (pack_dir / "summary.md").write_text(text, encoding="utf-8")


def validate_pack(pack_dir: Path, report_schema: Dict[str, Any], chart_reg: Dict[str, Any]) -> Dict[str, Any]:
    req_files = report_schema["pack"]["required_files"]
    req_dirs = report_schema["pack"]["required_dirs"]
    checks = {f: (pack_dir / f).exists() for f in req_files}
    checks.update({d: (pack_dir / d).is_dir() for d in req_dirs})
    for c in chart_reg.get("charts") or []:
        checks[c["path"]] = (pack_dir / c["path"]).exists()
    report = (pack_dir / "factor_report.md").read_text(encoding="utf-8") if (pack_dir / "factor_report.md").exists() else ""
    for ch in report_schema["factor_report_md"]["chapter_order"]:
        checks[f"chapter:{ch}"] = ch.lower() in report.lower() or ch.split()[0].lower() in report.lower()
    # stricter chapter presence
    for i, ch in enumerate(report_schema["factor_report_md"]["chapter_order"], 1):
        checks[f"chapter:{ch}"] = f"# {i}." in report or f"# {i} " in report
    missing = [k for k, v in checks.items() if not v]
    return {"ok": len(missing) == 0, "checks": checks, "missing": missing}


def generate_pack(factor_id: str, factors_root: Path = FACTORS_ROOT) -> Dict[str, Any]:
    metric_reg, chart_reg, report_schema = load_registries()
    pack_dir = factors_root / factor_id
    if not pack_dir.is_dir():
        raise FileNotFoundError(pack_dir)

    content_path = SPECS_DIR / f"{factor_id}_report_content.yaml"
    spec_path = SPECS_DIR / f"{factor_id}.yaml"
    if not content_path.exists():
        raise FileNotFoundError(
            f"Missing {content_path} — Report Generator v2 is schema-driven; "
            "add *_report_content.yaml instead of hardcoding factor branches."
        )
    content = _load_yaml(content_path)
    spec = _load_yaml(spec_path) if spec_path.exists() else {"factor_id": factor_id}

    # Backup v1 checklist report once
    v1 = pack_dir / "factor_report.md"
    v1_bak = pack_dir / "factor_report_v1.md"
    if v1.exists() and not v1_bak.exists():
        shutil.copy2(v1, v1_bak)

    ensure_dirs(pack_dir)
    summary = pd.read_csv(pack_dir / "factor_summary.csv") if (pack_dir / "factor_summary.csv").exists() else pd.DataFrame()
    stability = pd.read_csv(pack_dir / "yearly_stability.csv") if (pack_dir / "yearly_stability.csv").exists() else pd.DataFrame()
    stability = normalize_stability_df(stability)
    execution = pd.read_csv(pack_dir / "execution_summary.csv") if (pack_dir / "execution_summary.csv").exists() else pd.DataFrame()
    mechanism = pd.read_csv(pack_dir / "mechanism.csv") if (pack_dir / "mechanism.csv").exists() else pd.DataFrame()

    colmap = collect_column_map(pack_dir)
    # Headline RankICIR from raw for union default; risk ladder holds all modes
    if not summary.empty:
        raw = summary[summary["mode"] == "raw"]
        if not raw.empty:
            r0 = raw.iloc[0]
            for src_key, dst_key in (
                ("rank_ic", "rank_ic"),
                ("icir", "icir"),
                ("annu_ic", "annu_ic"),
                ("hl_sharpe", "hl_sharpe"),
                ("hl_annu_ret", "hl_annu_ret"),
                ("hl_mdd", "hl_mdd"),
                ("daily_turnover", "daily_turnover"),
                ("net_sharpe", "net_sharpe"),
                ("monotonicity", "monotonicity"),
                ("implied_annu_fee", "implied_annu_fee"),
            ):
                fv = _safe_float(r0.get(src_key))
                if fv is not None:
                    colmap[dst_key] = (fv, "factor_summary.csv:raw")
        if not execution.empty:
            fv = _safe_float(execution.iloc[0].get("annu_one_way_turnover"))
            if fv is not None:
                colmap["annu_one_way_turnover"] = (fv, "execution_summary.csv:best")
                colmap["annual_turnover"] = colmap["annu_one_way_turnover"]

    union = build_metric_union(metric_reg, colmap)
    risk_ladder = build_risk_ladder(pack_dir)
    # Rebuild construction diagram from content (avoid stale TGD-hardcoded PNG)
    cd = pack_dir / "charts" / "construction_diagram.png"
    if cd.exists() and content.get("construction_steps"):
        cd.unlink()
    missing_charts = prepare_charts(pack_dir, chart_reg, summary, stability, execution, content=content)
    copy_mechanism_execution(pack_dir)
    apply_artifact_copies(pack_dir, spec)

    report_md = render_factor_report(
        content, union, risk_ladder, mechanism, stability, execution, missing_charts, factor_id
    )
    (pack_dir / "factor_report.md").write_text(report_md, encoding="utf-8")
    write_card(pack_dir, content, spec)
    write_metrics_json(pack_dir, factor_id, union, risk_ladder, summary)
    write_summary_md(pack_dir, factor_id, union, risk_ladder)

    validation = validate_pack(pack_dir, report_schema, chart_reg)
    (pack_dir / "artifacts" / "pack_validation_v2.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return {"pack_dir": str(pack_dir), "validation": validation, "missing_charts": missing_charts, "n_metrics": len(union)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Factor Report Generator v2 (schema-driven)")
    p.add_argument("--factor", required=True, help="factor_id with factor_specs/{id}_report_content.yaml")
    args = p.parse_args(argv)
    result = generate_pack(args.factor)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
