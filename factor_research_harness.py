"""Factor Research Harness — Milestone 1A skeleton.

Orchestrates Protocol v1 flow without a second evaluation framework:

    factor_spec → compute → evaluate → report pack

Reuses (does not reimplement):
  - factor_runner.py
  - factor_eval_metrics.py
  - factor_report_generator.py
  - execution_layer.py

Milestone 1A scope:
  - CLI + mode/benchmark resolution
  - factor_spec loading
  - adapters for known frozen factors (TGD20 validate-only)
  - NO registry upgrade, NO pack migration, NO formula changes
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = REPO_ROOT / "docs" / "factor_research_protocol_v1.md"
SPECS_ROOT = REPO_ROOT / "factor_specs"
FACTORS_ROOT = REPO_ROOT / "research" / "reports" / "factors"

# Protocol Dual Benchmark — harness-side resolution (not a new eval framework)
BENCHMARK_VERSION = {
    "research": "research_v1",
    "production": "production_v1",
}

MODE_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "research": {
        "benchmark_version": "research_v1",
        "universe": "ALL",
        "period_target": "2018-01-01_2025-12-31",
        "horizon_days": 20,
        "neutralization": "none",
        "cost_bp": None,
        "portfolio": "10_decile_long_short",
        "purpose": "alpha_discovery",
    },
    "production": {
        "benchmark_version": "production_v1",
        "universe": "CSI1000",
        "period_target": "2018-01-01_2025-12-31",
        "horizon_days": 20,
        "neutralization": "industry_size",
        "cost_bp": 15,
        "portfolio": "10_decile_long_short",
        "purpose": "factor_admission",
    },
}


@dataclass
class StageResult:
    name: str
    status: str  # ok | skipped | not_implemented | error
    detail: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class HarnessResult:
    factor_id: str
    mode: str
    benchmark: Dict[str, Any]
    stages: List[StageResult] = field(default_factory=list)
    ok: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "mode": self.mode,
            "benchmark": self.benchmark,
            "ok": self.ok,
            "stages": [asdict(s) for s in self.stages],
        }


def resolve_benchmark(mode: str) -> Dict[str, Any]:
    key = mode.strip().lower()
    if key not in MODE_BENCHMARKS:
        raise ValueError(f"Unknown mode={mode!r}; expected research|production")
    return dict(MODE_BENCHMARKS[key])


def load_factor_spec(factor_id: str, specs_root: Path = SPECS_ROOT) -> Dict[str, Any]:
    """Load factor_specs/{factor_id}.yaml. Returns {} if missing (adapter may still exist)."""
    path = specs_root / f"{factor_id}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required to load factor_spec.yaml") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid factor_spec at {path}")
    data["_spec_path"] = str(path)
    return data


def list_registered_adapters() -> List[str]:
    return sorted(_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Stage implementations (thin adapters — no new metrics)
# ---------------------------------------------------------------------------


def stage_load_spec(factor_id: str, specs_root: Path = SPECS_ROOT) -> StageResult:
    spec = load_factor_spec(factor_id, specs_root)
    if not spec:
        return StageResult(
            name="load_factor_spec",
            status="skipped",
            detail={"factor_id": factor_id, "specs_root": str(specs_root)},
            message=f"No factor_specs/{factor_id}.yaml yet (adapter may still run)",
        )
    return StageResult(
        name="load_factor_spec",
        status="ok",
        detail={
            "factor_id": spec.get("factor_id", factor_id),
            "family": spec.get("family"),
            "frozen_formula": spec.get("frozen_formula"),
            "spec_path": spec.get("_spec_path"),
            "adapter": spec.get("adapter"),
        },
        message="factor_spec loaded",
    )


def stage_compute_default(factor_id: str, mode: str, benchmark: Dict[str, Any]) -> StageResult:
    return StageResult(
        name="compute",
        status="not_implemented",
        detail={"factor_id": factor_id, "mode": mode, "benchmark_version": benchmark.get("benchmark_version")},
        message=(
            "Milestone 1A skeleton: compute not wired for this factor. "
            "Reuse factor_runner / formula modules in a later milestone."
        ),
    )


def stage_evaluate_default(factor_id: str, mode: str, benchmark: Dict[str, Any]) -> StageResult:
    return StageResult(
        name="evaluate",
        status="not_implemented",
        detail={
            "factor_id": factor_id,
            "mode": mode,
            "reuse": ["factor_eval_metrics", "factor_runner", "execution_layer"],
            "benchmark": benchmark,
        },
        message=(
            "Milestone 1A skeleton: evaluate not wired. "
            "Will call existing eval helpers — no new metric definitions."
        ),
    )


def stage_report_default(factor_id: str, mode: str) -> StageResult:
    return StageResult(
        name="generate_report_pack",
        status="not_implemented",
        detail={"factor_id": factor_id, "mode": mode, "reuse": ["factor_report_generator"]},
        message="Milestone 1A skeleton: report generation not wired for this factor.",
    )


def _adapter_tgd20_compute(factor_id: str, mode: str, benchmark: Dict[str, Any]) -> StageResult:
    return StageResult(
        name="compute",
        status="skipped",
        detail={
            "factor_id": factor_id,
            "mode": mode,
            "reason": "frozen_formula",
            "benchmark_version": benchmark.get("benchmark_version"),
            "note": "TGD20 formula must not be recomputed/retuned by harness",
        },
        message="TGD20 compute skipped (frozen); use existing values / pack",
    )


def validate_legacy_pack(factor_id: str, root: Path = FACTORS_ROOT) -> Dict[str, Any]:
    """Read-only pack checklist (Template v1).

    Prefer factor_report_generator.validate_pack when importable; otherwise
    mirror the same file checks without pulling numpy (CI / slim envs).
    """
    try:
        from factor_report_generator import validate_pack

        return validate_pack(factor_id, root)
    except ImportError:
        out = root / factor_id
        required = {
            "factor_report.md": (out / "factor_report.md").exists(),
            "metrics.json": (out / "metrics.json").exists(),
            "factor_summary.csv": (out / "factor_summary.csv").exists(),
            "mechanism": (out / "mechanism.csv").exists()
            or (out / "mechanism_analysis.csv").exists(),
            "stability": (out / "stability.csv").exists()
            or (out / "yearly_stability.csv").exists(),
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
            "validator": "harness_lightweight_fallback",
        }


def _adapter_tgd20_evaluate(factor_id: str, mode: str, benchmark: Dict[str, Any]) -> StageResult:
    # Read-only: do not rewrite pack; validate existing Template v1 artifacts.
    validation = validate_legacy_pack(factor_id, FACTORS_ROOT)
    status = "ok" if validation.get("ok") else "error"
    return StageResult(
        name="evaluate",
        status=status,
        detail={
            "mode": mode,
            "benchmark_version": benchmark.get("benchmark_version"),
            "protocol_note": (
                "Existing TGD20 pack is legacy Template v1 (often ALL / 2022–2025). "
                "Protocol Dual Benchmark not re-run in Milestone 1A."
            ),
            "pack_validation": validation,
            "reuse": ["factor_report_generator.validate_pack", "factor_eval_metrics (schema only)"],
        },
        message="TGD20 evaluate = validate existing pack (no new eval run)",
    )


def _adapter_tgd20_report(factor_id: str, mode: str) -> StageResult:
    pack = FACTORS_ROOT / factor_id
    exists = pack.is_dir()
    charts_present = {
        "ic_curve.png": (pack / "ic_curve.png").exists() or (pack / "charts" / "ic_curve.png").exists(),
        "decile_return.png": (pack / "decile_return.png").exists()
        or (pack / "charts" / "decile_return.png").exists(),
        "cumulative_long_short.png": (pack / "cumulative_long_short.png").exists()
        or (pack / "charts" / "cumulative_long_short.png").exists(),
        "turnover.png": (pack / "turnover.png").exists() or (pack / "charts" / "turnover.png").exists(),
    }
    return StageResult(
        name="generate_report_pack",
        status="ok" if exists else "error",
        detail={
            "mode": mode,
            "pack_path": str(pack),
            "exists": exists,
            "write_pack": False,
            "note": (
                "Milestone 1A does not call assemble_tgd20() (no pack rewrite). "
                "Report stage points at existing research/reports/factors/TGD20/."
            ),
            "charts_present": charts_present,
            "reuse": ["factor_report_generator"],
        },
        message="TGD20 report pack located (read-only adapter)",
    )


AdapterFn = Callable[..., StageResult]


@dataclass
class FactorAdapter:
    factor_id: str
    compute: AdapterFn
    evaluate: AdapterFn
    report: AdapterFn


_ADAPTERS: Dict[str, FactorAdapter] = {
    "TGD20": FactorAdapter(
        factor_id="TGD20",
        compute=_adapter_tgd20_compute,
        evaluate=_adapter_tgd20_evaluate,
        report=_adapter_tgd20_report,
    ),
}


def get_adapter(factor_id: str) -> Optional[FactorAdapter]:
    return _ADAPTERS.get(factor_id)


def run_factor_research(
    factor_id: str,
    mode: str,
    *,
    specs_root: Path = SPECS_ROOT,
    dry_run: bool = False,
) -> HarnessResult:
    """Run Protocol-aligned harness pipeline for one factor.

    dry_run: resolve benchmark + load spec only (skip compute/evaluate/report).
    """
    benchmark = resolve_benchmark(mode)
    result = HarnessResult(factor_id=factor_id, mode=mode.lower(), benchmark=benchmark)

    result.stages.append(stage_load_spec(factor_id, specs_root))
    result.stages.append(
        StageResult(
            name="resolve_benchmark",
            status="ok",
            detail=benchmark,
            message=f"mode={mode} → {benchmark.get('benchmark_version')}",
        )
    )

    if dry_run:
        result.stages.append(
            StageResult(
                name="dry_run",
                status="ok",
                detail={},
                message="dry_run: stopped before compute/evaluate/report",
            )
        )
        result.ok = True
        return result

    adapter = get_adapter(factor_id)
    if adapter is None:
        result.stages.append(stage_compute_default(factor_id, mode, benchmark))
        result.stages.append(stage_evaluate_default(factor_id, mode, benchmark))
        result.stages.append(stage_report_default(factor_id, mode))
        # Skeleton success for wiring; stages may be not_implemented
        result.ok = all(s.status in ("ok", "skipped", "not_implemented") for s in result.stages)
        return result

    result.stages.append(adapter.compute(factor_id, mode, benchmark))
    result.stages.append(adapter.evaluate(factor_id, mode, benchmark))
    result.stages.append(adapter.report(factor_id, mode))
    result.ok = all(s.status in ("ok", "skipped") for s in result.stages)
    return result


def protocol_path() -> Path:
    return PROTOCOL_PATH
