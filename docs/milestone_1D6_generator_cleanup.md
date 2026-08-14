# Milestone 1D.6 — Report Generator v2 Schema Compliance Cleanup

**Date:** 2026-07-20  
**Status:** PASS  
**Scope:** Generator architecture only — **no Registry**, no formula changes, no metric redefinition.

---

## Goal

Close the audit P1 defect: production layer must not hardcode factor paths / `factor_id` branches before Registry freezes assets.

```
factor_specs/{id}.yaml          → artifacts.copy / pack_local_copy
factor_specs/{id}_report_content.yaml → narrative + code_map
        ↓
factor_report_generator_v2.py    → schema/spec only
        ↓
research/reports/factors/{id}/
```

---

## Changes

| Area | Before | After |
|------|--------|-------|
| Diagnostics / mechanism copies | `if factor_id == "FlowDensity20": ...` | `spec.artifacts.copy` / `pack_local_copy` |
| Appendix B | Hardcoded TGD essay + `core/l2_features/tgd` for all packs | `content.code_map` per factor |
| Card family default | `temporal_information` | `unspecified` if missing |
| Coverage exception flag | always `True` | true only when `data_coverage.exception_reason` present |

### Spec fields added

```yaml
# factor_specs/{id}.yaml
artifacts:
  copy:
    - src: research/.../file.md
      dest: diagnostics/file.md
      also_dest: [mechanism/file.md]   # optional
  pack_local_copy:
    - src: diagnostics_universe_ladder.csv
      dest: diagnostics/universe_ladder.csv
```

```yaml
# factor_specs/{id}_report_content.yaml
code_map:
  - item: Implementation
    path: "..."
```

---

## Acceptance

| Criterion | Result |
|-----------|--------|
| No `if factor_id ==` / `in (...)` branches in generator | ✅ |
| Missing artifacts remain N/A / skipped — never fabricated | ✅ (Flow charts still missing, listed) |
| TGD20 narrative + metrics + mechanism + execution unchanged | ✅ |
| Appendix B TGD paths identical (content); table separator unified to `\| --- \|` | ✅ substantive |
| D1/Flow/Ideal Appendix B no longer point at TGD essay | ✅ |
| Unit tests | ✅ `tests/test_factor_report_generator_v2.py` (7 tests) |

**Validate packs:**

```bash
/opt/conda/anaconda3/envs/base_93/bin/python -m unittest tests.test_factor_report_generator_v2 -v
/opt/conda/anaconda3/envs/base_93/bin/python factor_report_generator_v2.py --factor TGD20
# Flow / D1 / IdealReversal likewise
```

| Pack | `validation.ok` | Notes |
|------|-----------------|-------|
| TGD20 | true | Golden stable |
| FlowDensity20 | false | Pre-existing 3-chart gap |
| D1_LiquidityQuality60d | true | Appendix B → D1 paths |
| IdealReversal | true | Appendix B → cutting paths |

---

## Explicit non-goals (still deferred)

- Flow visualization fill  
- D1 execution grid  
- Registry (1E)  
- Metric schema changes  

---

## Next

```
1D.6 Generator cleanup     ✅
        ↓
Flow charts + D1 execution (optional completion)
        ↓
1E Factor Registry v1
```

Production layer is now weldable: new factors add YAML, not Python branches.
