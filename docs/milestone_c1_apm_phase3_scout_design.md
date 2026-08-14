# C1.3 — APM_SessionResidual CSI1000 Scout Design

**Date:** 2026-07-21  
**Status:** Scout **COMPLETE** — verdict `PASS_research_FAIL_invest`  
**Identity:** `APM_SessionResidual` · `adapted_replication`  
**Phase2:** [`docs/milestone_c1_apm_phase2_sanity_design.md`](milestone_c1_apm_phase2_sanity_design.md) (**PASS**)  
**Artifacts:** `research/reports/apm_session_v1/scout/`  
**Script:** `run_milestone_c1_apm_session_scout.py`

---

## 0. Status lock

```text
C0 Pack Normalization                 ✅
C1 ActiveTrade / APM
  Phase0 Identity                     ✅ ACCEPTED · adapted_go
  Phase1 Session Panel                ✅ PASS
  Phase2 Object Constructability      ✅ PASS
  Phase3 Scout                        ← THIS DESIGN
  Phase4 Pack v1                      only if scout PASS
```

Phase2 proved the **paper object is constructible**.  
Phase3 asks: **does that object carry rankable alpha on CSI1000?**

Not yet: Pack v1 · library row as validated · Registry · Composite.

---

## 1. Goal

| Ask | Not ask |
|-----|---------|
| Raw RankIC of paper signal | Flip formula because IC sign ugly |
| Size+industry neutralized RankIC | Full ALL-share 252d production pack |
| Decile mono / H-L Sharpe / TO | Rename Proxy → APM |
| Yearly stability | Claim `true_replication` |
| IC-series corr vs library peers | Similarity Matrix v2 / residual IC stack |

---

## 2. Setup (locked for coding)

| Item | Value |
|------|-------|
| Universe | **CSI1000** |
| Period | **2021-01-01 → 2025-12-31** (primary); optional extend to 2020 if capacity OK |
| Preheat | ≥40 calendar days before start (rolling w=20 + Ret20) |
| Primary signal | **`apm_cs`** (APM_stat CS-residualized vs Ret20) — paper output shape |
| Diagnostic signal | raw `apm_stat` (report separately; not soft-bar driver) |
| Sign on IC | **raw signal — no flip** |
| H-L book | if RankIC > 0 → long high / short low; if RankIC < 0 → long low / short high for **PnL only**; IC metrics stay on raw |
| Eval shift | `signal.shift(1)` → `ret_{t}` (cache unshifted) |
| Cost | 15bp round-trip (same as SmartMoney scout) |
| Neutralization | raw + size+industry (`neutralize_size_industry` + `cs_zscore`) |
| Identity class | remains `adapted_replication` |

### Direction discipline

Paper Stage-0: **positive IC**.

```text
Compute RankIC(raw apm_cs, r_{t+1})
  → if negative: record direction_mismatch = true
  → do NOT multiply signal by −1 under this factor_id
```

Mismatch is a research finding, not a silent formula change.

---

## 3. Build path (reuse Phase1–2 assets)

```text
research/cache/apm_session/
  residual_panel/   (extend months 2020-12 → 2025-12 as needed)
  signal/
    apm_stat_*.parquet
    apm_cs_*.parquet
```

Engineering:

1. Ensure residual months for scout range (+ preheat) via `apm_session_panel_builder`
2. `apm_session_signal.build_apm_stat_panel` + `cs_residualize_vs_ret20`
3. Mask to CSI1000 membership
4. Eval stack (mirror SmartMoney phase2a)

Do **not** rebuild Proxy. Do **not** select `Active_*`.

---

## 4. Tests (ordered)

### 4.1 Raw RankIC

- Daily RankIC(`apm_cs`, `r_{t+1}`)
- Mean RankIC, \|ICIR\|, IC>0 fraction, n days

### 4.2 Neutralized RankIC

- Size + industry neutralize → CS z → RankIC / ICIR

### 4.3 Decile monotonicity

- Decile mean forward returns (signed book consistent with IC sign for display)
- Report mono score; do not retune if weak (IdealAmplitude lesson)

### 4.4 H-L Sharpe

- Gross + Net @15bp RT
- Daily turnover

### 4.5 Yearly stability

- Per calendar year: RankIC, \|ICIR\|, Gross/Net Sharpe
- Count years with RankIC sign matching paper expectation (positive) **and** separately years with Gross Sharpe>0

### 4.6 Peer IC-series correlation

Corr of daily RankIC series (not residual IC):

| Peer | Why |
|------|-----|
| **FlowDensity20** | **priority** — active-flow / timing overlap risk |
| TGD20 | session / temporal structure |
| D1_LiquidityQuality60d | liquidity quality |
| SmartMoney10d | microstructure peer (if panel available; else skip + note) |
| ActiveTradeProxy | optional diagnostic — expect related but not identical |

High corr vs Flow is a **warning**, not auto-reject; pack notes must record it.

---

## 5. Soft bars (pre-agreed)

Mirror SmartMoney style; dual read (research vs investability):

| Metric | Soft threshold | Notes |
|--------|----------------|-------|
| \|RankIC\| (raw) | > 2% | on `apm_cs` |
| \|ICIR\| (raw) | > 1.5 | |
| Direction | record vs paper (+) | mismatch ≠ auto flip |
| Years IC-stable | ≥ 3/5 years RankIC sign consistent | 2021–2025 |
| H-L Gross Sharpe | > 0 | research edge |
| H-L Net Sharpe @15bp | > 1 | investability (may FAIL like SM) |
| Peer corr vs Flow | report; warn if \|ρ\| > 0.7 | not hard fail |

**Coded verdict options:**

| Label | Meaning |
|-------|---------|
| `PASS_scout` | IC soft bars + net investability clear |
| `PASS_research_FAIL_invest` | IC/mechanism OK; net TO/cost fails (SmartMoney pattern) |
| `FAIL_scout` | IC soft bars miss or object broken |
| `direction_mismatch` | flag inside summary; still one of above |

Pack v1 only if `PASS_scout` **or** explicit human admit of `PASS_research` into `testing` pack (same honesty as Ideal*/SmartMoney).

---

## 6. Artifacts

```text
research/reports/apm_session_v1/scout/
├── summary.json
├── ic/
│   ├── factor_summary.csv
│   └── ic_curve.png
├── neutralization/
│   └── neutral_ic.csv
├── quantile/
│   ├── decile_return.csv
│   └── monotonicity.json
├── stability/
│   └── yearly_ic.csv
├── similarity/
│   ├── factor_ic_corr.csv
│   └── factor_signal_corr.csv
└── execution/
    └── turnover_summary.csv
```

Runner: `run_milestone_c1_apm_session_scout.py`

---

## 7. Explicit non-goals

- ❌ Sign-flip “fix”
- ❌ Pack / `factor_library.csv` before verdict
- ❌ Registry / Composite / Portfolio
- ❌ Overwrite `compute_apm()` / touch Proxy
- ❌ Claim `true_replication` (index day proxy still adapted)
- ❌ Full Similarity / residual-IC stack (defer)

---

## 8. Capacity note

Scout needs ~5y minute→PM month chunks + overnight + signal.  
Reuse TGD/SmartMoney month-chunk pattern; prefer CSI1000 symbol filter **after** daily aggregate (or restrict DDB if feasible).  
Expect multi-hour first build; cache months so reruns are cheap.

---

## 9. After Phase3

```text
PASS_scout / admit testing
        ↓
Phase4 Pack v1 → factor_library.csv (status testing|candidate)

PASS_research_FAIL_invest
        ↓
document like SmartMoney; optional hold/buffer grid later; no fake Pack

FAIL_scout
        ↓
close or park paper track; keep Phase1–2 caches as assets
```

---

## 10. Related

- Phase2 sanity: `docs/milestone_c1_apm_phase2_sanity_design.md`
- SmartMoney scout: `docs/milestone_3_0_iiia41_smartmoney10d_phase2a.md`
- Panel builder: `core/l2_features/apm_session_panel_builder.py`
- Signal builder: `core/l2_features/apm_session_signal.py`
- Roadmap: `research/reports/factors/ROADMAP.md`
