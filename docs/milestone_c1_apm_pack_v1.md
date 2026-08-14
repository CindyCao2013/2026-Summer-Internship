# C1.5 — APM_SessionResidual Research Pack v1

**Date:** 2026-07-21  
**Status:** DONE  
**Pack path:** `research/reports/factors/APM_SessionResidual/`  
**Library status:** `testing_candidate`

---

## Admission basis

| Gate | Result |
|------|--------|
| Phase0–2 | Identity + panel + constructability PASS |
| Phase3 Scout | Info PASS · daily invest FAIL |
| Phase4 Execution | **Case A** · Net 1.50 @ `daily\|buffer_10_30` |

## Locks in `summary.yaml`

1. **Execution recipe** frozen: `highAPM|daily|buffer_10_30` @15bp  
2. **Direction** frozen: positive / long_high_apm / no sign flip  
3. **Identity:** `adapted_replication` (not true; not Proxy)

## Library row

```csv
APM_SessionResidual,paper_adapted,session_behavior,testing_candidate,6.55,0.0225,...
```

## Explicit non-goals

- ❌ Registry  
- ❌ Composite  
- ❌ Promote to `validated` without further review  
- ❌ Touch `ActiveTradeProxy`
