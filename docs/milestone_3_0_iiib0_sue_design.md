# III-B0 — SUE Feasibility & Identity Design

**Date:** 2026-07-20  
**Status:** DESIGN ONLY — no formula coding · no Registry · no Composite  
**Parent:** III-B Fundamental Entry · III-A CLOSED  
**Discipline:** same as III-A4 SmartMoney (provenance first)

---

## Purpose

Before any III-B1 implementation:

1. Disambiguate **which** “SUE” (do not mix families)  
2. Audit Wind / `sue_p0` fields vs each identity  
3. Lock **one primary** factor identity for III-B1  
4. Decide true / adapted / impossible  
5. Reposition prior P0 “useful fail” under **current** Alpha OS topology  

---

## 0. Scope lock

| In | Out |
|----|-----|
| SUE identity + data audit | APM / SmartMoney reopen |
| One primary III-B1 candidate | Kitchen-sink 5-factor retune |
| Honest use of `sue_data.py` inventory | Claiming P0 sealed fail = “no fundamental alpha forever” |
| Look-ahead rules | Registry / Composite |

---

## 1. Paper / definition families (must not mix)

| Family | Typical formula | Object | Is “SUE”? |
|--------|-----------------|--------|-----------|
| **A. Classic / consensus surprise** | \((E_{\mathrm{act}}-E_{\mathrm{cons}})/\sigma\) or \(/|E_{\mathrm{cons}}|\) | Announcement surprise vs analyst | ✅ canonical SUE |
| **B. Seasonal / RW surprise** | \((NP_t-NP_{t-4})/\sigma(\Delta NP)\) | Time-series unexpected earnings | ✅ classic Foster-style SUE |
| **C. Analyst revision** | \(\Delta\) consensus over 20d | Expectation path, not announcement surprise | ❌ **not SUE** → III-B2 |
| **D. Profit-notice midpoint** | Notice mid vs LY / past NP | China pre-announcement event | ⚠ adapted event — separate id |
| **E. Acceleration** | ΔSUE or surprise change | Second-order | ❌ later, not III-B1 |

**Rule:** One `factor_id` = one family. Never name revision or notice-mid as `SUE_*` without qualifier.

---

## 2. Repo inventory (already built — do not reinvent loaders)

### Data layer — `sue_data.py` + `research/cache/sue_p0/`

| Source | Wind table | Fields used | Role |
|--------|------------|-------------|------|
| Notice | `ASHAREPROFITNOTICE` | first/ann date, NP min/max | early known |
| Express | `ASHAREPROFITEXPRESS` | ACTUAL_ANN_DT/ANN_DT, NP, EPS | early known |
| Income | `ASHAREINCOME` (合并) | ACTUAL_ANN_DT/ANN_DT, NP, EPS | formal |
| Consensus | `ASHARECONSENSUSDATA` | EST_DT, EPS_AVG, NET_PROFIT_AVG (FY1) | expectations |

**Hard PIT rule (locked since P0):**

```
known_date = min(notice, express, formal_income)
```

No peeking past `known_date`. Consensus must be **strictly before** `known_date` for surprise.

Units unified to **yuan** (notice/consensus ×1e4).

Cache present under `research/cache/sue_p0/{notice,express,income,consensus}/` (month parquet shards from ~2017).

### Formula inventory — `factor_formulas_sue.py`

| Library name | Family | Maps to §1 |
|--------------|--------|------------|
| `sue_eps_consensus` | A | Classic consensus SUE |
| `sue_np_yoy_z` | B | Seasonal / YoY z SUE |
| `analyst_np_revision_20d` | C | **Revision — not SUE** |
| `unexpected_profit_notice_surprise_20d` | D | Notice event |
| `profit_notice_mid_surprise` | D | Notice event |

Transforms already exist: event-hold 20d · daily decay HL=5.

### Prior density — `research/reports/sue_density_v1/` (SEALED P0)

Sample 2023-12→2025-12 vs **old Base3** `{D1, D4, D5}`:

| Factor | Best ICIR | resid_t vs Base3 | P0 verdict |
|--------|----------:|-----------------:|------------|
| `sue_eps_consensus` | **3.71** | 1.76 | raw_signal_only (near miss) |
| `sue_np_yoy_z` | 0.54 | 0.32 | raw_signal_only |
| notice / revision | weak | low | drop |

**P0 gate was:** resid \(t\ge2\) **and** stack ICIR uplift vs Base3. All failed stack uplift.

---

## 3. Field ↔ requirement matrix

### Family A — Consensus EPS SUE (preferred III-B1)

| Need | Availability | Gap? |
|------|--------------|------|
| Actual EPS on known_date | income/express EPS ✅ | none |
| Pre-ann consensus EPS | consensus EST_DT < known_date ✅ | none |
| Standardization | `/|cons|` implemented; \(\sigma\) variant optional | optional upgrade |
| Announcement calendar | known_date timeline ✅ | none |
| Look-ahead control | est_dt < known_date ✅ | none |

**Verdict: TRUE REPLICATION — feasible**

### Family B — YoY NP z SUE

| Need | Availability | Gap? |
|------|--------------|------|
| NP series on earliest-known | timeline ✅ | none |
| YoY lag + rolling σ | implemented in `_yoy_sue_events` ✅ | none |

**Verdict: TRUE / CLASSIC-VARIANT — feasible** (Foster-style; label as YoY, not “consensus SUE”)

### Family C — Analyst revision

Feasible data-wise, but **identity ≠ SUE**. Park for **III-B2 Revision**.

### Family D — Notice surprises

Feasible; China-specific **adapted event**. Separate ids; not III-B1 primary.

---

## 4. Factor identity lock (III-B1 primary)

### Chosen primary

```yaml
factor_id: SUE_ConsensusEPS          # NOT registered yet
display_name: SUE Consensus EPS Surprise
identity_class: true_replication_candidate
family: fundamental / earnings_surprise
paper_archetype: classic SUE (actual vs pre-ann consensus)
source: Wind income/express + consensus (PIT)

construction:
  known_date: min(notice, express, income)
  actual: EPS on known_date timeline
  expected: last consensus EPS_AVG with est_dt < known_date
  surprise: (actual - expected) / abs(expected)
  panel: event → hold_20d  OR  decay_hl5   # choose ONE in III-B1 design lock
  direction_expected: positive_ic

banned_aliases:
  - analyst_np_revision_20d
  - unexpected_profit_notice_surprise_20d
  - mixing YoY z under this factor_id
```

Library formula reuse: existing `sue_eps_consensus` body — **rename/admit as OS identity**, do not invent new knife.

### Secondary (optional later, separate ids)

```yaml
factor_id: SUE_NP_YoY_Z              # family B — only after A pack decision
factor_id: AnalystRevision20d       # family C — III-B2
factor_id: ProfitNoticeSurprise*    # family D — satellite / event track
```

---

## 5. Decision summary

| Identity | Class | Decision |
|----------|-------|----------|
| **SUE_ConsensusEPS** | true replication | **GO — III-B1 primary** |
| SUE_NP_YoY_Z | classic variant | GO later (2nd), separate id |
| AnalystRevision20d | not SUE | **Defer III-B2** |
| Notice surprises | adapted event | not III-B1 |
| “Ignore P0 forever” | — | **Rejected** — P0 fail was vs **old Base3 stack gate**, not “data impossible” |

### How to read P0 under Phase III

```
P0 sealed fail  =  failed residual+stack vs {D1,D4,D5} density gate
III-B goal      =  admit a Fundamental information source into Alpha Library
                  under OS path vs current peers {TGD, D1, Flow, SmartMoney}
```

| Reuse from P0 | Do not reuse blindly |
|---------------|----------------------|
| Loaders / PIT / units | “No further SUE work” as permanent ban |
| Consensus SUE definition | Base3 as only residual universe |
| Hold vs decay transforms | Kitchen-sink 5-factor retune |

**III-B1 evaluation peers (proposed):** residual IC vs **TGD20 · D1 · FlowDensity20** (SmartMoney optional).  
Stack-into-Base3 is **not** the III-B1 admission bar (library expansion ≠ production stack).

---

## 6. Look-ahead / integrity checklist (coding gate)

- [ ] `known_date` = earliest notice/express/income  
- [ ] Consensus `est_dt < known_date`  
- [ ] No report_period label used as trade date  
- [ ] Signal only after known_date (shift/hold rules documented)  
- [ ] Coverage report (sparse events → many NaN days — expected)  
- [ ] Do not auto-promote on raw ICIR alone (P0 lesson: size/industry matter)

---

## 7. Explicit non-goals (this milestone)

- ❌ Implement / retune formulas  
- ❌ Registry  
- ❌ Composite / Portfolio  
- ❌ APM / SmartMoney  
- ❌ Rename revision → SUE  

---

## 8. Gate to III-B1 coding

Coding may start only when:

1. This design accepted  
2. Primary id locked: **`SUE_ConsensusEPS`**  
3. Panel mode chosen: **hold_20d XOR decay_hl5** (one default)  
4. Eval window + residual peers agreed (recommend CSI1000 scout + vs TGD/D1/Flow)  

---

## Related

- III-B entry: `docs/milestone_3_0_iiib_fundamental_entry.md`  
- III-A close: `docs/checkpoint_2026_07_iiia_complete.md`  
- P0 seal: `research/reports/sue_density_v1/README.md`  
- Loaders: `sue_data.py` · formulas: `factor_formulas_sue.py`  
- Identity proposals: `docs/schemas/iiib0_factor_identity_proposals.yaml`
