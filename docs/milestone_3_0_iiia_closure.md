# Phase III-A Microstructure Completion — Closure Review

**Date:** 2026-07-20  
**Status:** III-A **review closed** (not formula retune; no silent status promotion)  
**Scripts:**  
- `run_ideal_amplitude_mechanism_diagnosis.py`  
- Minute audit via DDB `dfs://QV_Trade_to_MinuteBar`

---

## 1. Family inventory (honest)

| Factor | Identity | Status | Role |
|--------|----------|--------|------|
| TGD20 | true factor | validated | WHEN price moves (timing) |
| D1 | true factor | candidate | HOW stable liquidity is |
| FlowDensity20 | true factor | candidate | WHO / flow pressure (enhancer) |
| IdealReversal | true factor | testing | WHICH condition creates reversal |
| IdealAmplitude | true factor | testing | WHICH amplitude state hides dispersion |
| ActiveTradeProxy | **proxy only** | testing | daily overnight−day imbalance — **≠ paper APM** |
| **SmartMoney10d** | true replication | **research_candidate** | informed efficiency; **parked** (IC OK, daily TO fail) |
| APM_SessionResidual | adapted (deferred) | — | non-blocking; index minute gap |

**Reinforced rule:** `ActiveTradeProxy ≠ ActiveTrade paper replication.`

---

## 2. SmartMoney / APM data audit

### Source

```
dfs://QV_Trade_to_MinuteBar / Stock_one_minute
```

Already used by TGD panel builder (`core/l2_features/tgd_panel_builder.py`).

### Schema (available)

| Field | SmartMoney | Paper APM |
|-------|:----------:|:---------:|
| Close / OHLC | ✅ | ✅ session splits |
| Volume / Amount | ✅ | ✅ |
| Active_buy/sell volume·amount·count | ✅ useful | optional |
| Date partition ~2018-09 → 2026-07 | ✅ | ✅ |

Coverage check (2024): ~3.09e8 minute rows — live.

### Feasibility decision

| Question | Answer |
|----------|--------|
| Is minute-level replication **feasible**? | **YES** |
| Is MinuteFeatureStore / paper APM **implemented**? | **NO** |
| Should we start it **before** IdealAmplitude diagnosis + III-A doc freeze? | **No — review first** |
| Recommended next engineering slot | **III-A4** dedicated: paper APM **or** SmartMoney (pick one), server-side aggregates like TGD |

**Cost note:** Full-history SmartMoney/APM is heavy (month-chunk DDB jobs + cache). Feasible because TGD path exists; not “free.”

**Do not** expand ActiveTradeProxy and call it done.

---

## 3. IdealAmplitude mechanism diagnosis

**Window:** last 252d · SI CS-z · no formula change  
**Outputs:** `research/reports/ideal_amplitude_v1/mechanism_diagnosis/`

### Headline conflict to resolve

| Source | Mono | Sharpe / ICIR |
|--------|-----:|--------------:|
| Cutting harvest (3885d) | **0.11** | Sharpe ≈ 3.44 · ICIR ≈ −7.7 |
| This diagnosis (252d) | **0.89** (signed) | Gross Sharpe ≈ 4.70 · ICIR ≈ −10.9 |

### Root cause of “mono 0.11”

`monotonicity_score` in `alpha_research_report.py` counts:

```
fraction of adjacent deciles with return increasing in bucket rank
```

IdealAmplitude has **negative IC** → raw payoff is **decreasing** in V.  
That scores ~0.11 under an **unsigned “higher bucket → higher return”** definition.

After applying the **correct short-high-V book**:

| Metric | Value |
|--------|------:|
| Signed decile mono_frac | **0.89** |
| Spearman(bucket, return) | **0.99** |
| Ventile mono_frac | 0.79 |
| U-shape score | ≈ 0 (not U-shaped) |

**Pattern label:** `mostly_monotonic` (not U-shape / not mono-chaos).

### Payoff shape (signed book, D1=low V … D10=high V after flip)

```
D1  −6.0 bp
D2  +7.0
…
D9  +18.3
D10 +18.0   ← slight top-two inversion only
```

Mostly increasing; soft-bar mono failure on harvest was largely a **metric × sign convention** issue.

### Tail contribution

| Book | Mean daily PnL |
|------|---------------:|
| Extreme 5% H-L | 0.00312 |
| Inner 5–10% H-L | 0.00165 |
| Full 10% H-L | 0.00239 |

Extreme bucket is **stronger** than the full top/bottom 10% → edge is **tail-concentrated** (inner names dilute).  
This explains high Sharpe even when soft-bar mono looked terrible (wrong sign metric) and why payoff shape still matters.

### Status decision (this review)

| Action | Decision |
|--------|----------|
| Retune IdealAmplitude | **No** |
| Auto-promote to `candidate` | **No** (human review; reconcile full-sample **signed** mono first) |
| Update narrative / notes | **Yes** — document metric artifact + tail concentration |
| Keep Registry status | **`testing`** until signed full-sample mono + Dual Benchmark reviewed |

---

## 4. Updated III-A → next

```
III-A Microstructure Completion

DONE (true factors / packs):
  TGD20 · D1 · Flow · IdealReversal · IdealAmplitude

DONE (proxy, labeled):
  ActiveTradeProxy

REVIEW CLOSED:
  SmartMoney/APM feasibility = YES (minute table live)
  IdealAmplitude mono mystery = metric/sign + tail concentration

NEXT (pick order):
  III-A4  Paper APM or SmartMoney (minute, one factor at a time)
     OR
  III-B   SUE (new information layer) — only after accepting III-A4 deferral

THEN:
  Similarity Matrix v2 → Composite / Portfolio v2
```

**Recommendation:** Schedule **III-A4 SmartMoney *or* paper APM** as a scoped engineering milestone (reuse TGD minute pipeline).  
Enter **III-B SUE** in parallel only if minute capacity is blocked; do not pretend ActiveTradeProxy closed the ActiveTrade paper.

---

## 5. Module completion (updated)

| Module | Completion |
|--------|----------:|
| Research OS infrastructure | ~95% |
| Report Generator v2 | ~95% |
| Registry | ~90% |
| Microstructure family | **~85%** (proxy gap + minute stubs) |
| Fundamental family | 0% |
| Portfolio capability | ~80% frozen |

---

## Artifacts

| Path | Content |
|------|---------|
| `research/reports/ideal_amplitude_v1/mechanism_diagnosis/` | payoff CSVs · tail · diagnosis.json · charts |
| `docs/checkpoint_2026_07_paper_replication_track.md` | prior checkpoint |
| This file | III-A closure |

---

## Constraints honored

- No factor formula retune  
- No status change without full signed-sample review  
- ActiveTradeProxy honesty preserved  
