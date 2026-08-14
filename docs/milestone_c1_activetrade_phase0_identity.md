# Milestone C1 Phase0 — ActiveTrade Identity & Data Audit

**Date:** 2026-07-21  
**Status:** ACCEPTED — ADAPTED GO  
**Parent:** Alpha Factor Library · C1 ActiveTrade  
**Companion schema:** [`docs/schemas/c1_factor_identity_proposals.yaml`](schemas/c1_factor_identity_proposals.yaml)  
**Next design:** [`docs/milestone_c1_activetrade_phase1_panel_design.md`](milestone_c1_activetrade_phase1_panel_design.md)  
**Prior design (reuse, do not contradict):** [`docs/milestone_3_0_iiia4_smartmoney_apm_design.md`](milestone_3_0_iiia4_smartmoney_apm_design.md)

---

## Scope lock

```text
C1 ActiveTrade
      |
      v
Phase0 Design          ← THIS DOC
      |
      v
Phase1 Implementation  (only after GO / ADAPTED GO)
```

**This phase does NOT:**

- write formula code
- run IC / scout
- touch `factor_library.csv` as a new paper factor
- enter Registry
- modify `ActiveTradeProxy`

---

## Honesty gate (critical)

```text
ActiveTradeProxy  ≠  paper ActiveTrade / APM
```

Name proximity is the main trap. Phase0 exists to prevent promoting the proxy as paper replication.

---

## 1. Paper provenance audit

### Source of truth (in-repo Stage-0)

| Artifact | Role |
|----------|------|
| `research/factor_cutting/paper_summary.md` | paper object / knife / output |
| `research/factor_cutting/factor_definition.yaml` → `apm` | structured identity |
| `research/factor_cutting/mechanism.md` → APM | economic story |
| `factor_cutting/active_trade.py` | stub + proxy formula |

### Paper definition (locked from Stage-0)

**Paper:** 《主动买卖 / APM》（开源证券 · 因子切割系列）  
**Alias in cutting DSL:** `active_trade` → **same paper as APM**, not a second paper.

| Element | Definition |
|---------|------------|
| **Object** | Overnight vs **afternoon (PM)** residual α |
| **Knife** | Time-of-day bucket (`overnight_am` vs `afternoon_pm`) |
| **Output** | Residualized APM statistic, then **CS residual vs Ret20** |
| **Direction (paper)** | positive IC |
| **Data layer** | session / minute returns + index residualization |

```text
Paper definition
       │
       ▼
Factor object = overnight residual α  vs  afternoon residual α
       │
       ▼
Required data = stock session returns + index session residual + Ret20 panel
       │
       ▼
Implementation feasibility = ADAPTED GO (see §4–§6)
```

### What the paper is *not*

| Not the paper | Why |
|---------------|-----|
| Daily `overnight − daytime` t-stat alone | That is `ActiveTradeProxy` |
| Active buy/sell volume imbalance | Different object (order-flow intensity); Stage-0 APM does **not** use `Active_*` |
| SmartMoney knife \(S_t=\|R\|/V^{0.25}\) | Different paper / different mechanism |

---

## 2. Identity separation

### Existing — do not touch

```yaml
factor_id: ActiveTradeProxy
status: testing
identity_class: research_proxy
data_level: EOD
```

**Formula (shipped):**

\[
t = \frac{\mathrm{mean}(r_{\mathrm{ON}} - r_{\mathrm{DAY}})}{\mathrm{std}/\sqrt{n}},\quad window=20
\]

with \(r_{\mathrm{ON}}=\mathrm{Open}/\mathrm{prevClose}-1\), \(r_{\mathrm{DAY}}=\mathrm{Close}/\mathrm{Open}-1\).

**Mechanism it answers:** *when* does the stock’s return accrue (overnight vs day)?  
**Missing vs paper:** PM minute session, index residual, CS residual vs Ret20.

### Proposed new identity (design_only)

Prefer continuity with III-A4:

```yaml
factor_id: APM_SessionResidual
display_name: APM Session Residual
paper: APM因子模型 / 主动买卖
identity_class: adapted_replication   # until PDF confirms index session method
status: design_only
registry: false
data_level: minute_plus_eod_index
```

**Naming rules:**

| Name | Decision |
|------|----------|
| `ActiveTrade` (bare) | **Forbidden** until PDF checklist signed + pack exists |
| `ActiveTrade_Paper` | Rejected — too easy to confuse with Proxy |
| `ActiveTradeSessionResidual` | Acceptable synonym; prefer `APM_SessionResidual` |
| Rename `ActiveTradeProxy` → APM / ActiveTrade | **Rejected** — provenance break |
| Use `Active_buy_*` under APM id | **Forbidden** — identity theft / formula invention |

If later research wants aggressive-flow imbalance, use a **new** id (e.g. `ActiveBuyImbalance20`), never `APM_*` / `ActiveTrade*`.

---

## 3. Definition mapping

| Component | Paper ActiveTrade / APM | Existing ActiveTradeProxy |
|-----------|-------------------------|---------------------------|
| Signal object | overnight vs PM **residual α** | overnight − daytime **raw return** |
| Economic question | does session trader-mix α predict future returns? | when does return accrue? |
| Time scale | minute / session | daily EOD |
| Inputs | stock session returns + index session + Ret20 | Open, Close only |
| Residualization | vs index (session), then CS vs Ret20 | none |
| Afternoon definition | PM window (Stage-0: ~13:01–15:00 minute Close) | full daytime Open→Close |
| Horizon / window | rolling residual α / t-stat (PDF TBD for exact N) | window=20 t-stat |
| Direction | positive IC (Stage-0) | empirical ICIR ≈ 7 (proxy) |
| Status | `design_only` | `testing` / `research_proxy` |

---

## 4. Data audit (SmartMoney method)

### Primary store

```text
dfs://QV_Trade_to_MinuteBar / Stock_one_minute
```

**Coverage:** ~2018-09-03 → 2026-07-17 (same path TGD / SmartMoney use).

### Minute fields

| Field | Present | Paper APM need? |
|-------|:-------:|:---------------:|
| Symbol, Bartime, Date | ✅ | ✅ |
| Open, High, Low, Close | ✅ | ✅ (PM session) |
| Volume, Amount | ✅ | optional |
| Active_buy_volume / Active_sell_volume | ✅ | **not required by paper** |
| Active_buy_amount / Active_sell_amount | ✅ | **not required** |
| Active_buy_count / Active_sell_count | ✅ | **not required** |

### Case matrix (feasibility)

| Case | Hypothesis | Fields | Verdict for **this** paper id |
|------|------------|--------|-------------------------------|
| **A** Active buy/sell imbalance \(\mathrm{AI}=(B-S)/(B+S)\) | aggressive flow | `Active_*` ✅ | **Out of scope** — different factor |
| **B** Trade intensity residual | intensity − expected | needs intraday baseline / market | **Not Stage-0 APM** |
| **C** Session α (overnight vs PM + index residual) | trader-mix timing | stock minute ✅; index minute ❌ | **Paper path** → adapted |

### Index residual gap

| Need | Availability |
|------|----------------|
| Stock overnight | ✅ EOD Open / prev Close |
| Stock afternoon (PM) | ✅ minute Close 13:01–15:00 |
| Index daily / EOD session | ✅ Wind / `get_Ret_Matrix` |
| Index **minute** in same DFS table | ❌ not in `Stock_one_minute` |

→ Index residualization uses **EOD index session proxies** unless a separate index-minute source is located.  
→ Identity class remains **`adapted_replication`** until PDF confirms EOD-equivalent residual is acceptable for “true”.

### Related caches (not substitutes)

```text
research/cache/l2_daily/...          # Flow / imbalance diagnostics
ActiveTradeProxy pack artifacts      # proxy only — do not reuse as paper proof
```

---

## 5. Distinction from neighbor factors

| Factor | Information source | Object |
|--------|--------------------|--------|
| **SmartMoney10d** | price–volume efficiency | \(S_t=\|R\|/V^{0.25}\) → VWAP ratio |
| **APM_SessionResidual** | session trader-mix / timing α | overnight vs PM residual |
| **FlowDensity20** | capital arrival / active flow | net active amount / mktcap |
| **TGD20** | temporal return structure | time-geometry of returns |
| **ActiveTradeProxy** | daily ON−DAY accrual proxy | raw overnight−day t-stat |

```text
SmartMoney     →  price efficiency (informed prints)
APM / ActiveTrade paper →  trading-session intention / mix
FlowDensity    →  capital movement (active amounts)
```

Three different alpha sources. Do not collapse them under one `factor_id`.

---

## 6. Decision gate

| Gate | Condition | Result |
|------|-----------|--------|
| **GO (true)** | paper definition clear **and** all fields including index minute (or PDF-signed EOD equivalence) | true replication → Phase1 |
| **ADAPTED GO** | core session mechanism reproducible; index residual via EOD proxy | `adapted_replication` → Phase1 |
| **NO-GO** | core session fields missing | close C1 paper track |

### Phase0 verdict

```text
Paper definition:     CLEAR (Stage-0 APM / 主动买卖)
Minute stock fields:  SUFFICIENT for PM session
Index minute:         MISSING → adapted residual
Active_* fields:      NOT required for paper APM
Proxy confusion risk: HIGH → identity locked as APM_SessionResidual

→ ADAPTED GO  (review accepted 2026-07-21)
```

**PDF / index-minute upgrade does not block adapted Phase1.**  
Ship Phase1 as `adapted_replication`; upgrade `identity_class` later if PDF or index minute arrives.

Next: C1.1 Session Panel Builder design → cache → coverage/PIT checks.  
Still no IC / library / Registry in Phase1 panel build.  
Still forbidden until library scale: Composite · Portfolio · Registry.

---

## 7. Explicit non-goals (Phase0)

- ❌ No `compute_apm` body / no session panel builder
- ❌ No IC / smoke / scout runs
- ❌ No Registry row
- ❌ No `ActiveTradeProxy` rename, formula change, or status promotion
- ❌ No Case A imbalance under APM / ActiveTrade ids
- ❌ No SmartMoney reopen / TO retune
- ❌ No SUE / Similarity / Composite

---

## 8. C1 onward (after Phase0 accepted)

```text
C0 Pack Normalization              ✅
C1 ActiveTrade / APM
     Phase0 Identity / Data Audit  ✅ ADAPTED GO (this doc)
          ↓
     Phase1 Session Panel Builder  ← NEXT (design then code)
          ↓
     Phase2 Sanity → Phase3 Scout → Phase4 Pack v1
C2 SmartMoney slow recipe (parked IC-ok / TO-open)
C3 SUE
C4 Revision / Quality / Value
          ↓
Similarity → Composite
```
---

## Related

- Phase1 panel design: `docs/milestone_c1_activetrade_phase1_panel_design.md`
- Proxy honesty: `docs/milestone_3_0_active_trade_proxy.md`
- III-A4 design: `docs/milestone_3_0_iiia4_smartmoney_apm_design.md`
- Identity YAML: `docs/schemas/c1_factor_identity_proposals.yaml`
- III-A4 YAML (historical): `docs/schemas/iiia4_factor_identity_proposals.yaml`
- Library roadmap: `research/reports/factors/ROADMAP.md`
