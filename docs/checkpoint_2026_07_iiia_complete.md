# Checkpoint — III-A Microstructure Completion (CLOSED)

**Date:** 2026-07-20  
**Status:** **III-A CLOSED** · SmartMoney parked as `research_candidate`  
**Next mainline:** **III-B Fundamental** (SUE first)

---

## One-line verdict

> SmartMoney10d is a valid informed-trading alpha with persistent multi-day IC; Phase2A net failure was **wrong trading frequency**, not missing signal. III-A’s job (prove minute microstructure adds a new information layer) is done. Do not optimize SM further now — open Fundamental.

---

## Research chain (SmartMoney) — complete

```
Paper definition → feasibility → impl design → L1/L2/L3 cache
  → smoke → sanity (σ, ranks, raw IC)
  → CSI1000 scout → horizon/TO diagnosis
```

All gates answered. No formula invention. No Active_*. No Registry.

---

## Library map (post III-A)

```
Validated / Core
  TGD20                         temporal structure

Candidate
  D1_LiquidityQuality60d        liquidity quality
  FlowDensity20                 flow / liquidity interaction (enhancer)

Research Candidate (parked)
  SmartMoney10d                 informed VWAP / price-volume efficiency
                                IC persists H=1→20; daily TO fails; best slow Net≈0.31

Testing
  IdealReversal                 cutting / reversal
  IdealAmplitude                cutting / vol-tail (signed mono OK)
  ActiveTradeProxy              daily proxy ≠ paper APM

Deferred (III-A4.2 — not blocking III-B)
  APM_SessionResidual           adapted replication (index minute gap)
```

### Information layers covered

| Layer | Coverage |
|-------|----------|
| Price / temporal | ✅ TGD |
| Liquidity | ✅ D1 |
| Flow | ✅ FlowDensity |
| Cutting / behavioral | ✅ Ideal* |
| Information efficiency (minute) | ✅ SmartMoney (research_candidate) |
| Session APM true | ⚠ deferred |
| **Fundamental** | ❌ **empty → III-B** |

---

## SmartMoney status lock

```yaml
factor_id: SmartMoney10d
status: research_candidate
production_ready: false
registry: false
park_reason: |
  Mechanism + IC + horizon validated.
  Daily investability fail; slowed recipe only Net≈0.31.
  Revisit at Portfolio v2 / slow satellite or when execution framework improves.
do_not:
  - formula retune
  - Composite now
  - more TO optimization as mainline
```

---

## Why not APM next?

APM is still valuable but **adapted** (no index minute) and easy to confuse with ActiveTradeProxy.  
III-A already proved the efficiency layer via SmartMoney.  
**Library gap = Fundamental**, not another microstructure variant.

---

## III-B next (locked mainline)

```
III-B Fundamental Alpha Expansion

  SUE  →  Earnings Revision  →  Quality  →  Value
```

Entry: **SUE feasibility + identity design** (same discipline as III-A4), then implementation.  
Reuse existing `sue_data.py` / density probes only as inventory — do not kitchen-sink.

Constraints unchanged:

- No portfolio / composite retune  
- TGD/D1/Flow topology frozen  
- Provenance-controlled identities  

---

## Related docs

| Doc | Role |
|-----|------|
| `docs/milestone_3_0_iiia41_smartmoney10d_horizon.md` | Horizon proof |
| `docs/milestone_3_0_iiia41_smartmoney10d_phase2a.md` | CSI1000 scout |
| `docs/milestone_3_0_iiia4_smartmoney_apm_design.md` | Identity lock |
| `docs/milestone_3_0_iiib_fundamental_entry.md` | III-B entry |
| `docs/alpha_information_topology_v1.md` | Layer map update |
| `research/reports/factors/ROADMAP.md` | Sequence |

---

## Explicit non-goals (now)

- ❌ SmartMoney Registry / Composite  
- ❌ III-A4.2 APM as blocking mainline  
- ❌ Portfolio Construction v2  
- ❌ DBD-GRU
