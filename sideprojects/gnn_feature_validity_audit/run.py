"""Single command entry point for the isolated feature validity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backtest import EvaluationResult, evaluate_factor, summary_frame  # noqa: E402
from data_adapter import CompanyDataAdapter, DataUnavailableError  # noqa: E402
from features import (  # noqa: E402
    FUNDAMENTAL_FACTORS,
    MACRO_FACTORS,
    PRICE_VOLUME_FACTORS,
    RELATION_FACTORS,
    SENTIMENT_FACTORS,
    build_all_atomic_features,
    build_equal_weight_composite,
)
from report import (  # noqa: E402
    TESTED_STATUSES,
    plot_factor_result,
    write_report,
    write_summary,
)


ATOMIC_BY_FAMILY = {
    "price_volume": PRICE_VOLUME_FACTORS,
    "fundamental": FUNDAMENTAL_FACTORS,
    "sentiment": SENTIMENT_FACTORS,
    "macro": MACRO_FACTORS,
    "relation": RELATION_FACTORS,
}
COMPOSITE_BY_FAMILY = {
    "price_volume": "price_volume_equal_weight",
    "fundamental": "fundamental_equal_weight",
    "sentiment": "sentiment_equal_weight",
    "relation": "relation_equal_weight",
}
REGISTRY_REQUIRED_COLUMNS = {
    "factor_id",
    "family",
    "factor_type",
    "display_name",
    "source_formula",
    "implemented_formula",
    "required_fields",
    "data_frequency",
    "pit_timestamp_field",
    "lookback",
    "default_orientation",
    "orientation_method",
    "neutralization",
    "coverage",
    "cross_sectional_unique_count",
    "test_status",
    "unavailable_reason",
    "notes",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent PIT audit of the upstream five-family features"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_DIR / "config.yaml"),
        help="Path to frozen YAML configuration",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--audit-only", action="store_true")
    selection.add_argument("--factor", help="Run one atomic factor or composite")
    selection.add_argument(
        "--family",
        choices=sorted(ATOMIC_BY_FAMILY),
        help="Run one frozen family",
    )
    selection.add_argument("--all", action="store_true", help="Run full registry")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore a matching successful cache manifest",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {
        "project",
        "sample",
        "universe",
        "data",
        "timing",
        "preprocessing",
        "orientation",
        "deciles",
        "metrics",
        "relation",
        "composites",
        "output",
    }
    missing = required - set(config)
    if missing:
        raise ValueError("Config missing sections: {}".format(sorted(missing)))
    signal_lag = int(config["timing"]["signal_lag_trading_rows"])
    if signal_lag < 2:
        raise ValueError("Post-close c2c signal lag must be at least 2 rows")
    if int(config["timing"]["calibration_embargo_rows"]) < signal_lag:
        raise ValueError("Calibration embargo must cover the forward-return lag")
    if not bool(config["timing"]["require_entry_and_exit_tradable"]):
        raise ValueError("Entry and exit tradability checks must remain enabled")
    if not bool(
        config["preprocessing"]["apply_signal_eligible_mask_before_transform"]
    ):
        raise ValueError("PIT signal mask must be applied before preprocessing")
    if config["composites"]["member_availability_window"] != "calibration_only":
        raise ValueError("Composite members must freeze from calibration only")
    max_workers = int(config["relation"]["max_parallel_workers"])
    if not 1 <= max_workers <= 10:
        raise ValueError("relation.max_parallel_workers must be between 1 and 10")
    return config


def validate_registry(registry: pd.DataFrame) -> None:
    missing_columns = REGISTRY_REQUIRED_COLUMNS - set(registry.columns)
    if missing_columns:
        raise ValueError(
            "Registry missing columns: {}".format(sorted(missing_columns))
        )
    if registry["factor_id"].duplicated().any():
        duplicates = registry.loc[
            registry["factor_id"].duplicated(keep=False), "factor_id"
        ].astype(str)
        raise ValueError(
            "Registry contains duplicate factor_id values: {}".format(
                sorted(duplicates.unique())
            )
        )
    expected = {
        factor_id
        for members in ATOMIC_BY_FAMILY.values()
        for factor_id in members
    } | set(COMPOSITE_BY_FAMILY.values())
    observed = set(registry["factor_id"].astype(str))
    if observed != expected:
        raise ValueError(
            "Registry factor pool mismatch; missing={} extra={}".format(
                sorted(expected - observed), sorted(observed - expected)
            )
        )
    nonempty_columns = [
        "factor_id",
        "family",
        "factor_type",
        "display_name",
        "source_formula",
        "implemented_formula",
        "required_fields",
        "data_frequency",
        "pit_timestamp_field",
        "lookback",
        "orientation_method",
        "neutralization",
        "test_status",
    ]
    empty = registry[nonempty_columns].isna() | registry[
        nonempty_columns
    ].astype(str).apply(lambda column: column.str.strip().eq(""))
    if empty.any().any():
        locations = [
            "{}:{}".format(
                registry.iloc[row_position]["factor_id"],
                empty.columns[column_position],
            )
            for row_position, column_position in zip(
                *np.where(empty.to_numpy())
            )
        ]
        raise ValueError(
            "Registry has empty required values: {}".format(locations)
        )
    invalid_neutralization = set(registry["neutralization"].astype(str)) - {
        "none",
        "log_market_cap",
        "industry+log_market_cap",
        "component_defined",
    }
    if invalid_neutralization:
        raise ValueError(
            "Registry has invalid neutralization modes: {}".format(
                sorted(invalid_neutralization)
            )
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(PROJECT_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _frozen_registry_hash(registry: pd.DataFrame) -> str:
    dynamic = {
        "coverage",
        "cross_sectional_unique_count",
        "test_status",
        "unavailable_reason",
    }
    frozen = registry[
        [column for column in registry.columns if column not in dynamic]
    ].copy()
    payload = frozen.fillna("").to_csv(index=False).encode("utf-8")
    return _sha256_bytes(payload)


def build_hashes(
    config_path: Path,
    config: Mapping[str, object],
    registry: pd.DataFrame,
) -> Dict[str, str]:
    config_hash = _sha256_bytes(config_path.read_bytes())
    implementation_hash = _implementation_hash()
    registry_hash = _frozen_registry_hash(registry)
    source_commit = str(config["project"]["source_commit"])
    cache_payload = "|".join(
        [config_hash, implementation_hash, registry_hash, source_commit]
    ).encode("utf-8")
    return {
        "config_hash": config_hash,
        "implementation_hash": implementation_hash,
        "registry_hash": registry_hash,
        "source_commit": source_commit,
        "cache_hash": _sha256_bytes(cache_payload),
    }


def _requested_ids(
    args: argparse.Namespace, registry: pd.DataFrame
) -> List[str]:
    all_ids = registry["factor_id"].astype(str).tolist()
    if args.factor:
        if args.factor not in set(all_ids):
            raise ValueError("Unknown factor_id: {}".format(args.factor))
        return [args.factor]
    if args.family:
        return registry.loc[
            registry["family"].eq(args.family), "factor_id"
        ].astype(str).tolist()
    # No selection flag defaults to the complete frozen pool.
    return all_ids


def _dependency_atomic_ids(
    requested: Sequence[str], registry: pd.DataFrame
) -> List[str]:
    required: Set[str] = set(
        registry.loc[
            registry["factor_id"].isin(requested)
            & registry["factor_type"].eq("atomic"),
            "factor_id",
        ].astype(str)
    )
    composite_rows = registry[
        registry["factor_id"].isin(requested)
        & registry["factor_type"].eq("composite")
    ]
    for family in composite_rows["family"].astype(str):
        required.update(ATOMIC_BY_FAMILY[family])
    return [
        factor
        for factor in registry.loc[
            registry["factor_type"].eq("atomic"), "factor_id"
        ].astype(str)
        if factor in required
    ]


def _result_complete(
    factor_id: str,
    status: str,
    factor_root: Path,
) -> bool:
    pngs = set(
        path.name for path in (factor_root / factor_id).glob("*.png")
    )
    if status in TESTED_STATUSES:
        return pngs == {"cumulative_hl.png", "decile_bar.png"}
    return not pngs


def cache_is_complete(
    requested: Sequence[str],
    hashes: Mapping[str, str],
    manifest_path: Path,
    summary_path: Path,
    registry_path: Path,
    report_path: Path,
    factor_root: Path,
    graph_path: Path,
) -> bool:
    if not all(
        path.exists()
        for path in (manifest_path, summary_path, registry_path, report_path)
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        registry = pd.read_csv(registry_path)
    except Exception:
        return False
    if (
        manifest.get("status") != "success"
        or manifest.get("cache_hash") != hashes["cache_hash"]
    ):
        return False
    if not set(requested).issubset(set(manifest.get("requested_factors", []))):
        return False
    artifact_paths = {
        "summary": summary_path,
        "report": report_path,
        "registry": registry_path,
        "graph_diagnostics": graph_path,
    }
    artifact_hashes = manifest.get("artifact_hashes", {})
    for name, expected_hash in artifact_hashes.items():
        path = artifact_paths.get(name)
        if path is None or not path.exists():
            return False
        if _sha256_bytes(path.read_bytes()) != expected_hash:
            return False
    manifest_has_graph = bool(manifest.get("graph_diagnostics"))
    if manifest_has_graph != graph_path.exists():
        return False
    indexed = summary.set_index("factor_id")
    registry_indexed = registry.set_index("factor_id")
    manifest_statuses = manifest.get("factor_statuses", {})
    for factor_id in requested:
        if factor_id not in indexed.index or factor_id not in registry_indexed.index:
            return False
        status = str(indexed.loc[factor_id, "test_status"])
        if status == "ERROR":
            return False
        if (
            str(registry_indexed.loc[factor_id, "test_status"]) != status
            or str(manifest_statuses.get(factor_id)) != status
        ):
            return False
        if not _result_complete(factor_id, status, factor_root):
            return False
    return True


def _registry_row(
    registry: pd.DataFrame, factor_id: str
) -> Mapping[str, object]:
    row = registry.loc[registry["factor_id"].eq(factor_id)]
    if len(row) != 1:
        raise ValueError("Registry row is not unique: {}".format(factor_id))
    return row.iloc[0]


def _error_result(
    factor_id: str,
    registry_row: Mapping[str, object],
    bundle,
    config,
    exc: Exception,
) -> EvaluationResult:
    return evaluate_factor(
        factor_id,
        None,
        registry_row,
        bundle,
        config,
        data_status="ERROR",
        unavailable_reason="{}: {}".format(type(exc).__name__, exc),
    )


def _evaluate_atomic_dependencies(
    atomic_ids: Sequence[str],
    registry: pd.DataFrame,
    feature_result,
    bundle,
    config,
) -> Dict[str, EvaluationResult]:
    results: Dict[str, EvaluationResult] = {}
    for factor_id in atomic_ids:
        row = _registry_row(registry, factor_id)
        data_status = feature_result.data_status.get(
            factor_id, "DATA_UNAVAILABLE"
        )
        reason = feature_result.reasons.get(
            factor_id, "No feature panel was produced"
        )
        try:
            results[factor_id] = evaluate_factor(
                factor_id,
                feature_result.panels.get(factor_id),
                row,
                bundle,
                config,
                data_status=data_status,
                unavailable_reason=reason,
            )
        except Exception as exc:
            results[factor_id] = _error_result(
                factor_id, row, bundle, config, exc
            )
    return results


def _evaluate_composite(
    composite_id: str,
    registry: pd.DataFrame,
    atomic_results: Mapping[str, EvaluationResult],
    bundle,
    config,
) -> EvaluationResult:
    row = _registry_row(registry, composite_id)
    family = str(row["family"])
    error_members = [
        factor_id
        for factor_id in ATOMIC_BY_FAMILY[family]
        if factor_id in atomic_results
        and str(atomic_results[factor_id].summary.get("test_status")) == "ERROR"
    ]
    if error_members:
        return evaluate_factor(
            composite_id,
            None,
            row,
            bundle,
            config,
            data_status="ERROR",
            unavailable_reason=(
                "Atomic member errors prevent frozen composite: {}".format(
                    "|".join(error_members)
                )
            ),
        )
    members = [
        factor_id
        for factor_id in ATOMIC_BY_FAMILY[family]
        if factor_id in atomic_results
        and atomic_results[factor_id].composite_usable
        and atomic_results[factor_id].processed_oriented is not None
    ]
    excluded = [
        factor_id
        for factor_id in ATOMIC_BY_FAMILY[family]
        if factor_id not in members
    ]
    minimum = int(config["composites"]["minimum_members"])
    if len(members) < minimum:
        result = evaluate_factor(
            composite_id,
            None,
            row,
            bundle,
            config,
            data_status="UNTESTABLE",
            unavailable_reason=(
                "Only {} usable frozen atomic members; requires {}".format(
                    len(members), minimum
                )
            ),
        )
        result.summary["composite_members"] = "|".join(members)
        result.summary["composite_excluded_members"] = "|".join(excluded)
        return result
    oriented_panels = {
        factor_id: atomic_results[factor_id].processed_oriented
        for factor_id in members
    }
    composite = build_equal_weight_composite(oriented_panels, members)
    result = evaluate_factor(
        composite_id,
        composite,
        row,
        bundle,
        config,
        data_status="AVAILABLE",
        already_preprocessed=True,
    )
    result.summary["composite_members"] = "|".join(members)
    result.summary["composite_excluded_members"] = "|".join(excluded)
    return result


def _update_registry(
    registry: pd.DataFrame,
    results: Iterable[EvaluationResult],
    path: Path,
) -> pd.DataFrame:
    output = registry.copy()
    output["factor_id"] = output["factor_id"].astype(str)
    for result in results:
        mask = output["factor_id"].eq(result.factor_id)
        summary = result.summary
        output.loc[mask, "coverage"] = summary.get("coverage", np.nan)
        output.loc[
            mask, "cross_sectional_unique_count"
        ] = summary.get("median_unique_values", np.nan)
        output.loc[mask, "test_status"] = summary.get(
            "test_status", "ERROR"
        )
        status = str(summary.get("test_status", "ERROR"))
        if status in (
            "DATA_UNAVAILABLE",
            "INSUFFICIENT_COVERAGE",
            "INSUFFICIENT_CROSS_SECTIONAL_VARIATION",
            "NOT_TESTABLE_CROSS_SECTIONALLY",
            "UNTESTABLE",
            "ERROR",
        ):
            output.loc[mask, "unavailable_reason"] = summary.get(
                "failure_reason", ""
            )
        elif status in TESTED_STATUSES:
            output.loc[mask, "unavailable_reason"] = ""
    output.to_csv(path, index=False)
    return output


def _remove_stale_untested_pngs(
    result: EvaluationResult, factor_root: Path
) -> None:
    directory = factor_root / result.factor_id
    if str(result.summary["test_status"]) in TESTED_STATUSES:
        return
    if directory.exists():
        for path in directory.glob("*.png"):
            path.unlink()
        try:
            directory.rmdir()
        except OSError:
            pass


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_manifest(
    path: Path,
    *,
    hashes: Mapping[str, str],
    requested: Sequence[str],
    results: Sequence[EvaluationResult],
    audit: Mapping[str, object],
    graph_diagnostics: pd.DataFrame,
    family_decisions: Optional[Mapping[str, str]] = None,
    status: str = "success",
    artifact_paths: Optional[Mapping[str, Path]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **dict(hashes),
        "status": status,
        "generated_at": datetime.now().isoformat(),
        "requested_factors": list(requested),
        "factor_statuses": {
            result.factor_id: result.summary.get("test_status")
            for result in results
        },
        "composite_members": {
            result.factor_id: result.summary.get("composite_members", "")
            for result in results
            if result.summary.get("factor_type") == "composite"
        },
        "skipped_day_reasons": {
            result.factor_id: dict(result.skipped_day_reasons)
            for result in results
            if result.skipped_day_reasons
        },
        "data_audit": audit,
        "graph_diagnostics": graph_diagnostics,
        "family_phase2_decisions": family_decisions or {},
        "artifact_hashes": {
            str(name): _sha256_bytes(file_path.read_bytes())
            for name, file_path in (artifact_paths or {}).items()
            if file_path.exists()
        },
    }
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _verify_png_contract(
    results: Sequence[EvaluationResult],
    factor_root: Path,
) -> None:
    expected_tested = {
        result.factor_id
        for result in results
        if str(result.summary["test_status"]) in TESTED_STATUSES
    }
    for result in results:
        pngs = set(
            path.name
            for path in (factor_root / result.factor_id).glob("*.png")
        )
        if result.factor_id in expected_tested:
            if pngs != {"cumulative_hl.png", "decile_bar.png"}:
                raise AssertionError(
                    "{} PNG contract failed: {}".format(
                        result.factor_id, sorted(pngs)
                    )
                )
        elif pngs:
            raise AssertionError(
                "Untested factor {} has PNG output".format(result.factor_id)
            )
    unexpected = [
        path
        for path in factor_root.rglob("*.png")
        if path.name not in {"cumulative_hl.png", "decile_bar.png"}
    ]
    if unexpected:
        raise AssertionError(
            "Unexpected PNG files: {}".format([str(path) for path in unexpected])
        )


def run_pipeline(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    registry_path = PROJECT_DIR / "factor_registry.csv"
    registry = pd.read_csv(registry_path)
    validate_registry(registry)
    hashes = build_hashes(config_path, config, registry)

    output_cfg = config["output"]
    results_dir = PROJECT_DIR / str(output_cfg["results_dir"])
    summary_path = PROJECT_DIR / str(output_cfg["summary_file"])
    report_path = PROJECT_DIR / str(output_cfg["report_file"])
    manifest_path = PROJECT_DIR / str(output_cfg["manifest_file"])
    factor_root = PROJECT_DIR / str(output_cfg["factor_dir"])
    graph_path = PROJECT_DIR / str(output_cfg["graph_diagnostics_file"])
    requested = _requested_ids(args, registry)

    if (
        not args.audit_only
        and not args.force
        and bool(output_cfg["skip_completed_same_hash"])
        and cache_is_complete(
            requested,
            hashes,
            manifest_path,
            summary_path,
            registry_path,
            report_path,
            factor_root,
            graph_path,
        )
    ):
        print(
            "SKIP: matching cache hash already completed ({})".format(
                hashes["cache_hash"][:12]
            )
        )
        return 0

    adapter = CompanyDataAdapter(config)
    if args.audit_only:
        schemas = adapter.schema_audit()
        results_dir.mkdir(parents=True, exist_ok=True)
        audit_path = results_dir / "data_availability_audit.csv"
        schemas.to_csv(audit_path, index=False)
        audit = {
            "schema_audit": schemas,
            "unavailable_sources": config["data"]["unavailable_sources"],
        }
        _write_manifest(
            manifest_path,
            hashes=hashes,
            requested=[],
            results=[],
            audit=audit,
            graph_diagnostics=pd.DataFrame(),
            status="audit_only",
            artifact_paths={"data_availability_audit": audit_path},
        )
        print("WROTE {}".format(audit_path))
        print("WROTE {}".format(manifest_path))
        return 0

    bundle = adapter.load()
    atomic_dependencies = _dependency_atomic_ids(requested, registry)
    include_graph = any(
        factor in atomic_dependencies
        for factor in (
            "degree_centrality",
            "pagerank",
            "dtw_similarity_mean",
        )
    )
    feature_result = build_all_atomic_features(
        bundle, config, include_graph=include_graph
    )
    atomic_results = _evaluate_atomic_dependencies(
        atomic_dependencies,
        registry,
        feature_result,
        bundle,
        config,
    )

    result_by_id: Dict[str, EvaluationResult] = {}
    for factor_id in requested:
        row = _registry_row(registry, factor_id)
        if str(row["factor_type"]) == "atomic":
            result_by_id[factor_id] = atomic_results[factor_id]
        else:
            result_by_id[factor_id] = _evaluate_composite(
                factor_id, registry, atomic_results, bundle, config
            )
    results = [result_by_id[factor_id] for factor_id in requested]

    results_dir.mkdir(parents=True, exist_ok=True)
    factor_root.mkdir(parents=True, exist_ok=True)
    summary = summary_frame(results)
    error_factors = summary.loc[
        summary["test_status"].eq("ERROR"), "factor_id"
    ].astype(str).tolist()
    if error_factors:
        for result in results:
            directory = factor_root / result.factor_id
            if directory.exists():
                for png_path in directory.glob("*.png"):
                    png_path.unlink()
        write_summary(summary, summary_path)
        _update_registry(registry, results, registry_path)
        if report_path.exists():
            report_path.unlink()
        if feature_result.graph_diagnostics.empty:
            if graph_path.exists():
                graph_path.unlink()
        else:
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            feature_result.graph_diagnostics.to_csv(graph_path, index=False)
        error_artifacts = {
            "summary": summary_path,
            "registry": registry_path,
        }
        if graph_path.exists():
            error_artifacts["graph_diagnostics"] = graph_path
        _write_manifest(
            manifest_path,
            hashes=hashes,
            requested=requested,
            results=results,
            audit=bundle.audit,
            graph_diagnostics=feature_result.graph_diagnostics,
            status="error",
            artifact_paths=error_artifacts,
        )
        raise RuntimeError(
            "Factor errors prevent report publication: {}".format(
                "|".join(error_factors)
            )
        )

    annualization = int(config["metrics"]["annualization_days"])
    for result in results:
        if str(result.summary["test_status"]) in TESTED_STATUSES:
            plot_factor_result(
                result,
                factor_root / result.factor_id,
                annualization_days=annualization,
            )
        else:
            _remove_stale_untested_pngs(result, factor_root)

    write_summary(summary, summary_path)
    updated_registry = _update_registry(registry, results, registry_path)
    family_decisions = write_report(
        summary,
        updated_registry,
        bundle.audit,
        report_path,
        annualization_days=annualization,
        source_commit=hashes["source_commit"],
    )
    if feature_result.graph_diagnostics.empty:
        if graph_path.exists():
            graph_path.unlink()
    else:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        feature_result.graph_diagnostics.to_csv(graph_path, index=False)

    _verify_png_contract(results, factor_root)
    _write_manifest(
        manifest_path,
        hashes=hashes,
        requested=requested,
        results=results,
        audit=bundle.audit,
        graph_diagnostics=feature_result.graph_diagnostics,
        family_decisions=family_decisions,
        artifact_paths={
            "summary": summary_path,
            "report": report_path,
            "registry": registry_path,
            **(
                {"graph_diagnostics": graph_path}
                if graph_path.exists()
                else {}
            ),
        },
    )
    print("WROTE {}".format(summary_path))
    print("WROTE {}".format(report_path))
    print("WROTE {}".format(manifest_path))
    print(
        "PASS atomic={} composite={}".format(
            int(
                (
                    (summary["factor_type"] == "atomic")
                    & (summary["test_status"] == "PASS")
                ).sum()
            ),
            int(
                (
                    (summary["factor_type"] == "composite")
                    & (summary["test_status"] == "PASS")
                ).sum()
            ),
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run_pipeline(args)
    except DataUnavailableError as exc:
        print("DATA_UNAVAILABLE: {}".format(exc), file=sys.stderr)
        return 3
    except Exception as exc:
        print(
            "ERROR: {}: {}".format(type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
