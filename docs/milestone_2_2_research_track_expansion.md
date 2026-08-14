# Milestone 2.2 — Alpha Research Expansion Track

**Date:** 2026-07-20  
**Status:** ACTIVE (course correction)  
**Phase:** Factor Library Expansion  
**Do NOT:** portfolio construction · composite optimization · new weighting engines

> **Naming note:** An earlier sidetrack used “2.2 Portfolio Construction.”  
> That work is **parked** as Phase 3 preview  
> (`docs/milestone_2_2_portfolio_construction.md`, `research/reports/portfolio_construction_v1/`).  
> **This document is the official Milestone 2.2.**

---

## Goal

Expand the validated factor library through **paper replication**:

```
5 registry rows (2 mature sources)
        ↓
30–50 candidates over time
        ↓
10–15 validated alpha sources
```

Pipeline (factory advantage):

```
paper
  → factor_spec.yaml
  → implementation
  → evaluation harness
  → Report Generator v2 pack
  → Registry
```

---

## Why not portfolio now?

Only **two** mature independent sources (TGD, D1).  
Institutions do not run portfolio optimization on a two-source library.  
Composite 2.0 answered complementarity; it did **not** open Phase 3.

---

## Priority queue

| # | Factor | Layer | Code | Spec | v2 Pack | Registry | Next action |
|---|--------|-------|------|------|---------|----------|-------------|
| 1 | **IdealReversal** | Trading behavior | ✅ `factor_cutting/ideal_reversal.py` | ✅ | ✅ thin | `testing` | Fill execution; harden Metric Union; keep soft-bar honest |
| 2 | **IdealAmplitude** | Trading behavior | ✅ `factor_cutting/ideal_amplitude.py` | ❌ | ❌ | — | Full OS path (greenfield pack) |
| 3 | **ActiveTrade** | Trading behavior | ✅ `factor_cutting/active_trade.py` | ❌ | ❌ | — | Full OS path after Ideal family |
| 4 | **SmartMoney** | L2 behavior | partial | ❌ | ❌ | — | After L2 cost/economics review |

Frozen: TGD20 / D1 / FlowDensity formulas.  
Inventory only (no kitchen-sink): D4, D5.

---

## IdealReversal — current truth

| Item | State |
|------|-------|
| Cutting Object–Knife–Output | ✅ daily return × ATS → M_high−M_low |
| Research Pack | ✅ `research/reports/factors/IdealReversal/` |
| Soft bar | ❌ mono≈0.44 — correctly `testing` |
| Execution layer | ✅ **2.2.1 filled** — best `every_20d + buffer_5_15` Net≈**1.70** TO≈0.15 (last 252d) |
| Auto-admit to candidate | **No** |

**2.2.1 DONE:** `run_milestone_2_2_1_ideal_reversal.py` · artifacts in pack `execution/` · Registry metrics updated · status stays `testing`.

---

## IdealAmplitude — next greenfield (2.2.2)

```
Object: daily amplitude (high/low - 1)
Knife:  close price state
Output: V_high - V_low
```

Delivers second cutting-family factor to test independence vs IdealReversal / TGD.

---

## Constraints

| Forbidden | Allowed |
|-----------|---------|
| Composite re-optimization | Paper → pack → Registry |
| Portfolio sizing / vol target / capacity as mainline | Factor understanding from 2.0/2.1 archives |
| New TGD/D1/Flow formulas | Cutting-family replication |
| Kitchen-sink D4/D5 into packs | Inventory rows stay |

---

## Success criteria for Expansion Track

1. IdealReversal: execution filled; pack honest; Registry note updated  
2. IdealAmplitude: admitted as `testing` or `candidate` with v2 pack  
3. ActiveTrade: same path queued  
4. Information Layer Map stays the north star (`docs/alpha_information_topology_v1.md`)  
5. No portfolio milestone reopened until ≥~10 validated sources

---

## Immediate next

```
2.2.1 IdealReversal OS gap-fill     ✅
        ↓
2.2.2 IdealAmplitude full replication track   ← NOW
        ↓
2.2.3 ActiveTrade
        ↓
(later) SmartMoney / Fundamental
```
