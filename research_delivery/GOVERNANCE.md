# Research Delivery Governance (frozen)

**Status:** FROZEN for Factor Delivery Sprint  
**Date:** 2026-07-21  
**Scope:** `research_delivery/` only — not Registry / Alpha OS / Portfolio.

---

## 1. Product rule

Deliver **research-grade / paper-grade A-share alpha cards**, aligned with TGD / Flow / Ideal* / APM trajectory.  
Do **not** fill coverage with textbook first-generation factors.

---

## 2. Board schema (`factor_delivery_plan.csv`)

| Field | Meaning |
|-------|---------|
| `factor_id` | Delivery id (PascalCase preferred for cards) |
| `display_name` | Human title |
| `tier` | `A` / `B` / `C` (see §3) |
| `mechanism` | One-line alpha source / economic object |
| `source` | paper / paper_adapted / research_internal / classic_baseline / proxy |
| `family` | Mechanism family tag |
| `difficulty` | easy / medium / hard / done |
| `priority` | Order within batch (1 = first) |
| `batch` | `1_existing` / `2_research` / `3_fundamental` / `4_l2_advanced` / `excluded` |
| `status` | `delivered` / `testing_candidate` / `planned` / `parked` / `excluded` / `next` |
| `validation_level` | How far validation has gone (see §4) |
| `delivery_report` | Path to `report.md` if exists |
| `pack_or_artifacts` | Experiment provenance path / code pointer |
| `counts_toward_library` | `yes` / `yes_soft` / `no` |
| `notes` | Caveats |

---

## 3. Tier definitions (frozen)

### Tier A — Delivery Candidate / Delivered

Must eventually satisfy:

```text
✔ clear economic mechanism
✔ frozen formula
✔ independent implementation
✔ basic backtest
✔ IC / ICIR
✔ portfolio / quantile test
✔ report.md under research_delivery/factors/
```

**Counts toward the ~20 library goal** (`counts_toward_library=yes` or `yes_soft` for testing mono).

`A` + `status=planned` = **A-pending** (in pipeline, not yet card-complete).

### Tier B — Research Candidate

Research value exists, but **missing a delivery link**, e.g.:

- sample / mechanism incomplete  
- portfolio / execution unfinished  
- parked after IC-strong / invest-weak  

**Does not count** until promoted to A with a complete card.

### Tier C — Archive / Baseline

Benchmark / proxy / demoted textbook factors.  
**Never counts** toward library. May remain in code for controls only.

---

## 4. Validation levels

| Level | Meaning |
|-------|---------|
| `none` | Idea only |
| `identity` | Id + mechanism accepted |
| `implemented` | Code exists, no delivery backtest card |
| `confirmation_basic` | IC + quantile + HL on a frozen window |
| `execution_reviewed` | Cost / turnover / buffer reviewed |
| `pack_v1` | Full Pack layout under `research/reports/factors/` |
| `parked_research` | Studied; not promoted |

Promotion rule: **do not invent new factor logic** when promoting — package existing research assets first.

---

## 5. Portfolio metric rule

Every Tier-A card must report:

```text
selected economic long-book return
− exact daily equal-weight return of all valid stocks in its test universe
→ annualized excess Sharpe
```

Positive-direction factors use Group10; negative-direction factors use Group1.
H–L Sharpe and H–L Net Sharpe are diagnostics and cannot substitute for this
market-relative long-book metric.

---

## 6. Standard promote flow (Batch 2 template)

```text
existing research asset
  → identity + exact formula (from code)
  → implementation pointer (no rewrite)
  → link confirmation / pack plots
  → research_delivery/factors/<ID>/report.md
  → update factor_delivery_plan.csv + factor_index.csv
```

---

## 7. Batch map (research-grade)

| Batch | Content |
|-------|---------|
| 1 | Existing delivered cards |
| 2 | AmihudShock → LiquidityResidual → ActiveImbalance → PathMomentum → VolRegime → OvernightDominance |
| 3 | SUE → EarningsRevisionMomentum → QualityResidual → GrossProfitability → EarningsSurpriseConsistency → ValueResidual |
| 4 | L2 advanced fill |

Textbook Momentum/Vol/Turnover/plain Amihud = **Tier C**.
