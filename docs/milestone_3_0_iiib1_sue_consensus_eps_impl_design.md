# III-B1 — SUE_ConsensusEPS Implementation Design + Cache Strategy

**Date:** 2026-07-20 (Phase1 coded 2026-07-21)  
**Status:** DESIGN LOCK · Phase1 smoke+sanity **PASS** · scout gated on explicit go  
**Identity:** locked in `docs/milestone_3_0_iiib0_sue_design.md`  
**Parent:** III-B0 ✅ · III-A CLOSED  
**Phase1 report:** `docs/milestone_3_0_sue_consensus_eps_phase1.md`  
**Positioning:** Fundamental information-source **admission**, not “re-run old P0 fail”

---

## 0. Scope lock

| In | Out |
|----|-----|
| `SUE_ConsensusEPS` only | YoY / Revision / Notice surprises |
| PIT surprise events → daily panel | Formula invention / kitchen-sink |
| Design → cache → builder → smoke → sanity → scout | Registry / Composite / Portfolio |
| IC decay **before** execution recipe | Premature lock of hold_20d XOR decay_hl5 |

**Forbidden:** alias revision as SUE; `est_dt >= known_date`; SmartMoney/APM reopen; Base3-only gate as sole admission bar.

---

## 1. Factor identity (frozen for code)

```yaml
factor_id: SUE_ConsensusEPS
display_name: SUE Consensus EPS Surprise
identity_class: true_replication_candidate
family: fundamental / expectation_error_correction
direction_expected: positive_ic
data_level: event_fundamental

surprise: "(EPS_actual - EPS_consensus) / abs(EPS_consensus)"
known_date: "first date with finite EPS among express/income (notice_dt kept for provenance)"
pit: "est_dt < known_date"   # strict
library_reuse: core/fundamental/sue_consensus_eps_panel.py (+ sue_p0 via sue_data)

panel_transform:
  status: UNLOCKED at design time
  reason: |
    Unlike SmartMoney (diffusion already diagnosed), SUE digestion horizon
    is unknown a priori. Discover via IC decay, then pick hold/decay recipe.
```

Economic object:

```
Analyst expectation  +  realized earnings  →  expectation error correction
```

---

## 2. Exact surprise algorithm (no invention)

### 2.1 Earliest-known timeline (EPS-known)

Reuse `sue_data.load_sue_raw_bundle` for L0 shards. Event builder sets:

```
notice_dt / express_dt / income_dt  = first disclosure per source (provenance)
known_dt = min(express_dt, income_dt) among rows with finite EPS
```

NP-only notice does not create a ConsensusEPS event (answers: when is **actual EPS** public).  
Actual EPS: value on `known_dt` (prefer income if same-day tie).

### 2.2 Consensus match (PIT)

```
expected = last EPS_AVG for (symbol, report_period)
           with EST_DT < known_date
```

If no such consensus → surprise = NaN (do not forward-fill consensus across known_date).

### 2.3 Surprise

\[
\mathrm{SUE}_i = \frac{\mathrm{EPS}_{actual} - \mathrm{EPS}_{consensus}}{|\mathrm{EPS}_{consensus}|}
\]

Require \(|\mathrm{EPS}_{consensus}| \ge \varepsilon\) (existing code uses `1e-6`).

### 2.4 Event → calendar panel (transform **unlocked**)

Event long: `[symbol, known_date, surprise, report_period]`

Daily wide `date × symbol` options to **test later** (do not lock in builder v0):

| Mode | Rule | When to choose |
|------|------|----------------|
| `impulse` | surprise only on known_date (NaN else) | IC decay / event study |
| `hold_H` | ffill ≤ H trading days | after decay prefers short persist |
| `decay_hl` | exp decay half-life | if gradual digestion |

**Builder v0 ships `impulse` + ability to apply hold/decay as transforms** (reuse `apply_event_hold` / `apply_daily_decay`).  
Default for smoke/sanity: **impulse** (honest sparsity).  
Scout reports IC on impulse **and** decay curve on forward returns; then pick panel mode.

---

## 3. Cache architecture (PIT-first)

### Directory layout

```
research/cache/sue_consensus_eps/

├── raw/                          # optional symlink/reuse of sue_p0
│     notice|express|income|consensus/*.parquet
│
├── events/
│     SUE_ConsensusEPS_events_{start}_{end}.parquet
│     columns: symbol, known_date, surprise, report_period,
│              eps_actual, eps_consensus, est_dt, source_actual
│
├── panels/
│     SUE_ConsensusEPS_impulse_{start}_{end}.parquet
│     SUE_ConsensusEPS_hold{H}_{start}_{end}.parquet   # only after mode lock
│     SUE_ConsensusEPS_decay_hl{L}_{start}_{end}.parquet
│
└── meta/
      pit_audit_{tag}.json
      coverage_{tag}.json
```

### Layer roles

| Layer | Content | Rebuild when |
|-------|---------|--------------|
| L0 `sue_p0` / Oracle | Month shards (existing) | Wind refresh |
| L1 `events/` | One row per (symbol, period) surprise with PIT fields | Formula/PIT change |
| L2 `panels/` | Daily wide | Transform mode / calendar |
| L3 artifacts | smoke / sanity / scout reports | eval |

**Reuse:** Prefer `cache_root=research/cache/sue_p0` via `load_sue_raw_bundle(..., keep_cache=True)`.  
Do **not** duplicate Oracle pulls if shards exist.

### PIT audit artifact (required every build)

```json
{
  "n_events": ...,
  "frac_est_before_known": 1.0,
  "n_rejected_est_ge_known": 0,
  "eps_consensus_abs_min": ...,
  "known_date_source_mix": {"notice": ..., "express": ..., "income": ...}
}
```

Any `est_dt >= known_date` in output → **hard fail** build.

---

## 4. Module layout

| Path | Role |
|------|------|
| `sue_data.py` | Reuse as-is (loaders / timeline) |
| `factor_formulas_sue.py` | Reuse `_consensus_eps_sue_events`; thin wrapper id `SUE_ConsensusEPS` |
| `core/fundamental/sue_consensus_eps_panel.py` | **shipped** — events → impulse panel + PIT audit + coverage |
| `run_milestone_3_0_sue_consensus_eps_phase1.py` | Phase1 smoke+sanity (combined) |
| `run_milestone_3_0_sue_consensus_eps_scout.py` | Phase3 CSI1000 scout + IC decay (not yet) |
| `docs/milestone_3_0_sue_consensus_eps_phase1.md` | Phase1 results |

No Registry helper until pack review.

---

## 5. Phased execution (mirror SmartMoney)

### Phase 1 — Builder + smoke (after go code)

1. Load bundle from `sue_p0`  
2. Build L1 events (`SUE_ConsensusEPS`)  
3. Build L2 impulse panel (short window e.g. 2024H1)  
4. Smoke gates: rows>0 · PIT audit pass · no revision/notice columns in event schema  

### Phase 2 — Sanity (before scout)

| Gate | Check |
|------|-------|
| CS dispersion | daily σ(SUE) on non-NaN names (event days may be sparse) |
| Zero / extreme mass | frac≈0, frac \|SUE\|>10 |
| Coverage | mean names on event-active days; pct days with ≥50 names |
| PIT | 100% `est_dt < known_date` |
| Direction probe | raw RankIC on impulse (short window) — **no sign flip** |

Sparse panel expected — do not fail solely because mean coverage ≪ SmartMoney.

### Phase 3 — Scout (CSI1000, 2020–2025)

```
Universe: CSI1000
Period:   2020-01-01 → 2025-12-31
Signal:   impulse (primary) + optional hold/decay for decay study
Peers:    residual IC vs TGD20, D1, FlowDensity20
Cost:     15bp RT (execution only after horizon)
```

Outputs:

- RankIC / ICIR / IC>0 frac (raw)  
- Decile / H-L  
- Yearly 2020–2025  
- **IC decay:** H ∈ {1, 5, 10, 20, 40, 60} on forward cumret  
- Peer IC-series corr (not full Similarity residual matrix)

### Phase 4 — Pack decision

Only after scout:

1. Choose panel transform from decay + TO  
2. Soft bars (library admission, not old Base3 stack)  
3. Pack → `testing` · Registry only on explicit go  

---

## 6. Why not lock hold_20d / decay_hl5 now

| Factor | Mechanism | Horizon prior |
|--------|-----------|---------------|
| SmartMoney | slow diffusion | H→20 IC rose → medium sleeve |
| SUE | earnings surprise digestion | PEAD literature: 5d–60d plausible |

Locking transform before decay would repeat Phase2A mistake (wrong frequency packaging).

---

## 7. Relation to P0 useful fail

| P0 | III-B1 |
|----|--------|
| Gate: resid_t≥2 vs {D1,D4,D5} + stack uplift | Gate: OS scout + residual vs {TGD,D1,Flow} |
| Verdict: sealed fail for production stack | Re-admission as **fundamental layer** candidate |
| 5 mixed identities | **One** identity |

Reuse loaders/events; do not retune five-factor grid.

---

## 8. Explicit non-goals

- ❌ YoY / Revision / Notice under this id  
- ❌ Registry / Composite / Portfolio  
- ❌ Premature execution recipe  
- ❌ Claiming P0 Base3 fail invalidates Fundamental layer  

---

## 9. Coding gate checklist

Go code Phase1 only when:

1. This design accepted  
2. Identity = `SUE_ConsensusEPS` only  
3. Cache root = `sue_p0` + `sue_consensus_eps/`  
4. Impulse panel first; transforms unlocked  
5. Smoke → sanity → scout order enforced  

---

## Related

- III-B0: `docs/milestone_3_0_iiib0_sue_design.md`  
- Identity YAML: `docs/schemas/iiib0_factor_identity_proposals.yaml`  
- Pattern: `docs/milestone_3_0_iiia41_smartmoney10d_impl_design.md`  
- P0 seal: `research/reports/sue_density_v1/README.md`  
- Loaders: `sue_data.py` · `factor_formulas_sue.py`
