# C2 Phase1 — SUE_ConsensusEPS PIT Event Panel Design

**Date:** 2026-07-21  
**Status:** DESIGN ONLY — prove event object / PIT (no multi-year scout · no Pack)  
**Identity:** `SUE_ConsensusEPS` · `paper_replication` · `design_only`  
**Phase0:** [`docs/milestone_c2_sue_phase0_identity.md`](milestone_c2_sue_phase0_identity.md) (**ACCEPTED**)  
**Reuse:** III-B1 builder + `research/cache/sue_consensus_eps/` (re-audit, do not rebuild OS)

---

## 0. Goal calibration

```text
Phase0: identity
Phase1: announcement / estimate / actual → PIT event object   ← THIS
Phase2: daily CS panel constructability
Phase3: alpha scout
```

Phase1 answers: **is there look-ahead?**  
Phase1 does **not** answer: does SUE make money?

---

## 1. Scope lock

| In | Out |
|----|-----|
| PIT timeline audit | RankICIR / Pack / Registry |
| Event schema + impulse panel check | Revision / YoY / Quality under this id |
| Reuse `sue_consensus_eps_panel.py` | Fundamental OS redesign |
| Coverage / sparsity report | Execution recipe lock |
| Document known_dt / est_dt rules | Premature hold_20d / decay_hl5 |

```text
existing cache
      ↓
Phase0 identity (done)
      ↓
Phase1 PIT event panel audit   ← THIS
      ↓
Phase2 constructability
      ↓
Scout
```

---

## 2. PIT object (frozen)

### 2.1 Timeline fields

| Field | Role |
|-------|------|
| `notice_dt` / `express_dt` / `income_dt` | disclosure provenance |
| `known_dt` | **first date with finite EPS** among express/income |
| `est_dt` | last consensus estimate date used |
| `actual_eps` | EPS on `known_dt` |
| `consensus_eps` | last `eps_avg` with **`est_dt < known_dt`** |
| `sue` | \((\mathrm{actual}-\mathrm{consensus})/|\mathrm{consensus}|\) |

NP-only notice (no EPS) → **no** ConsensusEPS event.

### 2.2 Leakage rules (hard)

| Rule | Pass |
|------|------|
| `est_dt < known_dt` | required |
| signal usable only on `date >= known_dt` | required |
| no consensus ffill across `known_dt` | required |
| `|consensus_eps| >= 1e-6` | required |

### 2.3 Panel v0

**Impulse only:** SUE on `known_dt`, else NaN. No ffill / no decay yet.

---

## 3. What to run in Phase1 (audit, light code OK)

Runner (suggested): `run_milestone_c2_sue_pit_panel.py`

Or extend III-B1 runner with Pack Track report path:

```text
research/reports/sue_consensus_eps_v1/c2_phase1/
├── pit_report.json
├── event_coverage.csv
├── timeline_sample.csv
├── leakage_checks.json
└── build_log.txt
```

### Checks

1. **Leakage sample:** 50–200 events — `est_dt`, `known_dt`, `actual`, `consensus`, `sue`  
2. **Dup keys:** unique `(symbol, fiscal_period)`  
3. **Source mix:** express vs income as `source_actual`  
4. **Coverage:** events / year, CSI1000 ∩ symbols, active days with ≥2 finite impulses  
5. **Unit tests:** existing `core/fundamental/test_sue_consensus_eps.py` must stay green  

Optional smoke RankIC on a short window is **diagnostic only** (already +5.38% in III-B1) — not a soft bar for Phase1.

---

## 4. Acceptance gates

| Gate | Pass |
|------|------|
| **P1** Leakage | zero `est_dt >= known_dt` in events |
| **P2** Schema | required EVENT_COLS present |
| **P3** Impulse | panel NaN off-event; no silent ffill |
| **P4** Coverage | multi-year event build feasible (or documented gap) |
| **P5** Provenance | not aliased as Revision / YoY |

Fail → fix builder; do not Scout.

---

## 5. Module boundary

| Keep | Do not put here |
|------|-----------------|
| `core/fundamental/sue_consensus_eps_panel.py` | IC / execution / Pack generator |
| `sue_data.py` L0 loaders | Composite blend |
| Pack Track reports under `sue_consensus_eps_v1/c2_phase1/` | Registry rows |

---

## 6. After Phase1 PASS

```text
Phase2 Constructability
  → daily impulse panel shape / sparsity / CS rankability
Phase3 Scout
  → CSI1000 multi-year · IC · decay H∈{1,5,10,20,40,60} · yearly
Phase4 Execution
  → event sparsity · hold/decay · refresh cadence (likely not daily buffer)
Phase5 Pack v1
  → research/reports/factors/SUE_ConsensusEPS/
```

---

## 7. Explicit non-goals

- ❌ Multi-year IC as Phase1 gate  
- ❌ Hold/decay lock  
- ❌ Pack / library promote  
- ❌ Registry / Composite / Portfolio  
- ❌ Expand to Revision/Quality under this milestone  

---

## Related

- Phase0: `docs/milestone_c2_sue_phase0_identity.md`
- III-B1 phase1 PASS: `docs/milestone_3_0_sue_consensus_eps_phase1.md`
- Builder: `core/fundamental/sue_consensus_eps_panel.py`
- C1 precedent: APM Phase1 panel discipline
