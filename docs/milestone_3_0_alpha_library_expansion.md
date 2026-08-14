# Phase III — Alpha Library Expansion

**Date:** 2026-07-20  
**Status:** ACTIVE — III-A CLOSED · **III-B0 design DONE** · III-B1 pending go  
**Goal:** Expand independent alpha sources — **not** improve current portfolio Sharpe.

---

## Capability vs production

| Milestone | Meaning |
|-----------|---------|
| 2.0–2.2 | Research / capability — portfolio **frozen** |
| III-A | Microstructure **CLOSED** · SM `research_candidate` parked |
| III-B0 | SUE identity + data **DONE** |
| III-B1 | SUE_ConsensusEPS implement — after explicit go |

```
Infrastructure     ✅
Microstructure     ✅
Fundamental design ✅  (III-B0)
Fundamental pack   ❌  ← next coding gate
```

---

## Tracks

### Track A — Microstructure ✅ CLOSED

SmartMoney10d = `research_candidate` (parked). APM deferred.

### Track B — Fundamental ← NOW

| Step | Status |
|------|--------|
| III-B0 SUE design | ✅ |
| III-B1 impl design + cache | ✅ [`milestone_3_0_iiib1_sue_consensus_eps_impl_design.md`](milestone_3_0_iiib1_sue_consensus_eps_impl_design.md) |
| Primary id | **`SUE_ConsensusEPS`** · impulse first · decay curve before recipe |
| III-B1 Phase1 coding | pending **go code** (smoke → sanity → scout) |

**P0 note:** prior density useful-fail vs old Base3 ≠ ban; III-B1 re-admits under OS peers (TGD/D1/Flow).

### Track C — Portfolio

**Frozen.**

---

## Constraints

- Freeze TGD/D1/Flow · park SmartMoney  
- One SUE identity at a time · revision ≠ SUE  
- No Composite / Registry until pack  

---

## Related

- III-B0: `docs/milestone_3_0_iiib0_sue_design.md`  
- III-A close: `docs/checkpoint_2026_07_iiia_complete.md`  
- Map: `docs/alpha_information_topology_v1.md`
