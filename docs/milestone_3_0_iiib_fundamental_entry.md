# III-B — Fundamental Alpha Expansion (Entry)

**Date:** 2026-07-20  
**Status:** OPEN — design before code  
**Parent:** Phase III Alpha Library Expansion  
**Prior:** III-A CLOSED ([`checkpoint_2026_07_iiia_complete.md`](checkpoint_2026_07_iiia_complete.md))

---

## Why now

III-A filled microstructure / efficiency. Library still missing:

```
Fundamental information source
```

SmartMoney parked as `research_candidate`. APM deferred (non-blocking).

---

## Track order

```
SUE (earnings surprise)
        ↓
Earnings / Analyst Revision
        ↓
Quality
        ↓
Value
```

**First slice:** SUE only — III-B0 design ✅ → III-B1 after go.

Design: [`docs/milestone_3_0_iiib0_sue_design.md`](milestone_3_0_iiib0_sue_design.md)  
Primary id: **`SUE_ConsensusEPS`** (true replication; reuse `sue_eps_consensus` body).

---

## Inventory already in repo (do not invent)

| Asset | Path / note |
|-------|-------------|
| `sue_data.py` | data helpers |
| `research/reports/sue_density_v1/` | prior density / independence probes |
| Protocol | Dual Benchmark / pack generator v2 |

Treat prior SUE work as **research inventory** — re-admit through OS path with frozen identity.

---

## III-B0 ✅ DONE

[`docs/milestone_3_0_iiib0_sue_design.md`](milestone_3_0_iiib0_sue_design.md)

| Identity | Decision |
|----------|----------|
| SUE_ConsensusEPS | **GO** III-B1 primary |
| SUE_NP_YoY_Z | later, separate id |
| AnalystRevision20d | **not SUE** → III-B2 |
| Notice surprises | not III-B1 |

Next: explicit **go code III-B1 Phase1** (builder + smoke).  
Impl design: [`docs/milestone_3_0_iiib1_sue_consensus_eps_impl_design.md`](milestone_3_0_iiib1_sue_consensus_eps_impl_design.md)  
Do **not** lock hold_20d / decay_hl5 until IC decay H∈{1,5,10,20,40,60}.

---

## III-B0 tasks (historical checklist — completed)

```
III-B0 SUE feasibility + identity design

1. Locate earnings surprise fields / announcement calendar
2. Compare available fields vs paper / Stage-0 SUE definition
3. Lock factor_id (e.g. SUE_*) and direction
4. Decide: true / adapted / impossible
5. No Registry · no Composite · no SM reopen
```

Coding starts only after III-B0 accepted (mirror III-A4 → III-A4.1).

---

## Constraints

- Freeze TGD/D1/Flow topology  
- Do not pull SmartMoney into Composite  
- Portfolio remains frozen  
- One factor family at a time  

---

## Success for III-B (slice)

At least one fundamental factor with:

```
paper → spec → eval → pack → Registry (testing or candidate)
```

comparable in discipline to microstructure packs — then Similarity / Composite **v2**.
