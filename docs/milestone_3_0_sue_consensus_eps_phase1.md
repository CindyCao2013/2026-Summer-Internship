# III-B1 Phase1 — SUE_ConsensusEPS Smoke + Sanity

**Date:** 2026-07-21  
**Status:** hard PASS · soft PASS  
**Identity:** `SUE_ConsensusEPS` (true_replication_candidate)  
**Scope:** events builder + PIT cache + impulse panel + smoke/sanity  
**Out of scope:** scout · Registry · hold/decay lock · Composite

---

## Verdict

```text
Gate 1 PIT leakage     PASS  (est_dt < known_dt; signal_date ≥ known_dt)
Gate 2 rankable CS     PASS  (56 active days · mean_n≈15.7 · mean_cs_std≈0.96)
Gate 3 raw RankIC      PASS  (+5.38% · no sign flip · expected positive)
```

**Next:** III-B1 Phase3 scout — CSI1000 2020–2025 + IC decay H∈{1,5,10,20,40,60}  
**Still forbidden:** Registry · Composite · premature hold_20d / decay_hl5

---

## What shipped

| Artifact | Path |
|----------|------|
| Event builder + impulse | `core/fundamental/sue_consensus_eps_panel.py` |
| Unit tests (PIT) | `core/fundamental/test_sue_consensus_eps.py` |
| Runner | `run_milestone_3_0_sue_consensus_eps_phase1.py` |
| Events cache | `research/cache/sue_consensus_eps/events/` |
| Impulse panel | `research/cache/sue_consensus_eps/panels/` |
| PIT meta | `research/cache/sue_consensus_eps/meta/` |
| Report | `research/reports/sue_consensus_eps_v1/phase1/phase1_report.json` |

Formula (locked):

\[
\mathrm{SUE} = \frac{\mathrm{EPS}_{actual} - \mathrm{EPS}_{consensus}}{|\mathrm{EPS}_{consensus}|}
\]

Panel v0 = **impulse only** (event day = SUE; else NaN). No ffill.

---

## known_dt refinement (EPS-known)

Design intent: *when does the market know actual EPS?*

| Field | Role |
|-------|------|
| `notice_dt` / `express_dt` / `income_dt` | full announcement timeline (provenance) |
| `known_dt` | **first date with finite EPS** among express/income |
| consensus | last `eps_avg` with `est_dt < known_dt` |

NP-only notice (no EPS) does **not** create a ConsensusEPS event.  
2024 mix: income 6144 · express 2447 · notice-as-actual 0.

---

## Smoke window

```text
Universe: CSI1000 ∩ event symbols
Period:   2024-01-01 → 2024-12-31
Source:   research/cache/sue_p0
```

| Metric | Value |
|--------|------:|
| Events (hist→end build) | 8591 |
| Events in 2024 window | 3501 |
| Dup (symbol, fiscal_period) | 0 |
| CSI1000 ∩ symbols | 1053 |
| Panel shape | 242 × 1053 |
| Event cells | 893 |
| Active days (≥2 finite) | 56 |
| mean names / active day | 15.7 |
| mean CS σ | 0.96 |
| raw RankIC (shift1) | **+5.38%** |
| IC days | 56 |
| short ICIR | ~1.44 |

Sparsity is expected for impulse — not a fail.

---

## Gates (mirror SmartMoney style)

| Gate | Rule | Result |
|------|------|--------|
| 1 PIT | `est_dt < known_dt` · mapped trade day ≥ `known_dt` · no dups | PASS |
| 2 Rankable | active_days≥5 · mean_cs_std>0 · mean_n≥5 | PASS |
| 3 Direction | raw RankIC finite · **no sign flip** | PASS (+IC) |

Hard = Gate1 + events>0 · Soft = hard + Gate2 + Gate3.

---

## Explicit non-goals (still)

- Scout / IC decay study  
- Hold_H or decay_hl packaging  
- Registry / Composite / Portfolio  
- Alias Revision / Notice / YoY as `SUE_*`

---

## Reproduce

```bash
OMP_NUM_THREADS=1 python -m pytest core/fundamental/test_sue_consensus_eps.py -q
OMP_NUM_THREADS=1 python run_milestone_3_0_sue_consensus_eps_phase1.py --year 2024
```

---

## Related

- Design: `docs/milestone_3_0_iiib1_sue_consensus_eps_impl_design.md`  
- Identity: `docs/milestone_3_0_iiib0_sue_design.md`  
- Pattern: `docs/milestone_3_0_iiia41_smartmoney10d_sanity.md`
