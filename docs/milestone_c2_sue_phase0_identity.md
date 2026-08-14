# Milestone C2 Phase0 — SUE_ConsensusEPS Pack Track Identity

**Date:** 2026-07-21  
**Status:** **ACCEPTED** · `design_only` → next Phase1 PIT panel design  
**Parent:** C1 CLOSED · Alpha Library expansion  
**Prior III-B:** `docs/milestone_3_0_iiib0_sue_design.md` · `docs/milestone_3_0_sue_consensus_eps_phase1.md`  
**Next:** [`docs/milestone_c2_sue_phase1_pit_panel_design.md`](milestone_c2_sue_phase1_pit_panel_design.md)

---

## Milestone card

```yaml
milestone: C2_SUE_Pack_Track
phase: Phase0_identity
status: design_only
next: Phase1_event_panel
constraints:
  no_registry: true
  no_composite: true
  no_portfolio: true
  no_fundamental_os: true
reuse: III-B1_PIT_assets
goal: paper_definition_to_factor_pack
```

---

## 0. Why C2 = SUE (not SmartMoney)

Library already covers Temporal · Liquidity · Flow · Session · Efficiency(parked).  
Missing: **fundamental surprise**. SmartMoney execution already answered → park.

**Not** a Fundamental OS. One Factor Pack Track only.

---

## 1. Identity lock (frozen)

```yaml
factor_id: SUE_ConsensusEPS
display_name: "Standardized Unexpected Earnings - Consensus EPS"
family: fundamental_surprise
identity_class: paper_replication   # true_replication_candidate in III-B; Pack Track label
status: design_only
registry: false
direction_paper: positive_ic
data_level: event_fundamental

formula: "(EPS_actual - EPS_consensus) / abs(EPS_consensus)"
known_dt: "first date with finite EPS among express/income"
pit: "est_dt < known_dt"
panel_v0: impulse_only
```

Economic object:

```text
Analyst consensus EPS + realized EPS → expectation error correction
```

---

## 2. Boundary vs other fundamentals

| Factor | Identity | Same id? |
|--------|----------|----------|
| **SUE_ConsensusEPS** | earnings surprise vs consensus | — |
| Revision | analyst expectation **change** | ❌ new id later |
| Quality | business quality | ❌ |
| Value | valuation | ❌ |
| EPS / NP YoY growth | growth, not surprise | ❌ new id if pursued |

**Forbidden under this id:** `SUE = EPS growth` · `SUE = YoY profit growth` · alias Revision as SUE.

---

## 3. III-B1 reuse (with Pack Track re-gate)

Reusable caches:

```text
research/cache/sue_consensus_eps/
  events/ · panels/ · meta/
```

**Rule:** prior III-B1 PASS ≠ automatic library admit.  
Must still pass Pack Track: identity → constructability → scout → execution → Pack.

---

## 4. Phase map

```text
Phase0 Identity              ✅ ACCEPTED
Phase1 PIT Event Panel       ← NEXT (design + audit)
Phase2 Constructability
Phase3 Scout
Phase4 Execution             # event sparsity / hold — not minute buffer
Phase5 Pack v1
```

---

## 5. Explicit non-goals

- ❌ Composite / Portfolio / Registry  
- ❌ Fundamental ranking system / multi-factor blend  
- ❌ Rebuild event store from scratch  
- ❌ Kitchen-sink under `SUE_ConsensusEPS`

---

## Related

- C1 close: `docs/checkpoint_2026_07_c1_apm_closed.md`
- Phase1 design: `docs/milestone_c2_sue_phase1_pit_panel_design.md`
- III-B1: `docs/milestone_3_0_sue_consensus_eps_phase1.md`
