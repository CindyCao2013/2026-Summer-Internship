# C1.2 — APM_SessionResidual Phase2 Sanity Design

**Date:** 2026-07-21  
**Status:** Phase2 **PASS** · review **ACCEPTED** (constructability closed; alpha deferred to C1.3)  
**Identity:** `APM_SessionResidual` · `adapted_replication`  
**Phase1:** [`docs/milestone_c1_activetrade_phase1_panel_design.md`](milestone_c1_activetrade_phase1_panel_design.md) (**PASS**)  
**Phase0:** [`docs/milestone_c1_activetrade_phase0_identity.md`](milestone_c1_activetrade_phase0_identity.md) (**ACCEPTED**)  
**Pattern:** SmartMoney Phase1.5 — prove paper object can be built before scout

---

## Goal calibration (critical)

```text
Phase1:  minute/eod → session object → cache
Phase2:  session object → paper statistic object → alignment sanity   ← THIS
Phase3:  IC / scout
```

Phase2 answers: *can the paper object be correctly constructed?*  
Phase2 does **not** answer: *does this factor make money?*

---

## 0. Status lock

```text
C0 Pack Normalization                 ✅
C1 ActiveTrade / APM
  Phase0 Identity                     ✅ ACCEPTED · adapted_go
  Phase1 Session Panel Builder        ✅ PASS
  Phase2 Sanity                       ← THIS DESIGN
  Phase3 Scout                        later
  Phase4 Pack v1                      later
```

Phase1 closed the **data engineering** loop.  
Phase2 asks: can we build the **paper-shaped signal object** from that panel, with correct PIT, without claiming alpha?

---

## 1. Question (sanity, not scout)

From Phase1 residual panel (`alpha_on`, `alpha_pm`, `delta_alpha`):

1. Are residual legs **finite, dispersed, non-degenerate**?
2. Can we form a rolling **`APM_stat`** (window=20) as a Symbol×Date panel?
3. Can we form **CS residual vs Ret20** as a constructability check?
4. Does evaluation alignment use **`signal(T) → return(T+1)`** via `shift(1)` **outside** the cache?

**Not asked in Phase2:** ICIR, Sharpe, decile mono, turnover fence, library admission.

---

## 2. Scope lock

| In | Out |
|----|-----|
| Residual distribution / outlier / NaN | RankICIR / ICIR |
| Rolling `APM_stat` (window=20) | H-L Sharpe / cost |
| Ret20 CS residual **constructability** | Quantile / decile books |
| `shift(1)` PIT alignment report | `factor_library.csv` |
| Sanity reports under `apm_session_v1/sanity/` | Pack v1 / Registry |
| Reuse Phase1 cache (no rebuild required if present) | Touch `ActiveTradeProxy` |
| Keep `Active_*` out | Claim `true_replication` |

```text
panel (Phase1)
   ↓
rolling APM_stat
   ↓
optional Ret20 CS residual (object only)
   ↓
shift(1) alignment check
   ↓
sanity reports
```

---

## 3. Identity / lineage (unchanged)

```yaml
factor_id: APM_SessionResidual
identity_class: adapted_replication
formula_version: apm_session_v1_adapted_eod_index
status: design_only → research_sanity (after Phase2 PASS)
```

**Do not** rename or promote `ActiveTradeProxy`.  
**Do not** implement by overwriting `factor_cutting.active_trade.compute_apm()`.

Future API (Phase2 coding — do not overwrite proxy):

```text
compute_apm_overnight_day_proxy()           # existing daily proxy lineage
compute_apm()                               # keep NotImplementedError
compute_apm_session_residual_signal(...)    # NEW — residual panel → APM_stat
# alias: build_apm_stat_panel(...)
```

Two lineages stay separate:

```text
ActiveTradeProxy          → daily ON−DAY t-stat proxy
APM_SessionResidual       → session residual + Ret20 object (this track)
```

---

## 4. Inputs (from Phase1)

```text
research/cache/apm_session/
  residual_panel/apm_residual_panel_{start}_{end}.parquet
  meta/formula_version.json
  calendar/trade_calendar_*.json
```

Required columns:

| col | role |
|-----|------|
| `date`, `symbol` | keys |
| `alpha_on`, `alpha_pm`, `delta_alpha` | residual legs |
| `r_on`, `r_pm`, `r_on_idx`, `r_day_idx` | diagnostics |

Default sanity window (first run): **same as Phase1 smoke** — e.g. 2024-06 (plus preheat for rolling 20).

Preheat: load residual panel from `start - ~40 calendar days` (or prior month cache) so window=20 is defined on month start.

---

## 5. Exact constructions (no invention)

### 5.1 Residual sanity (distributions)

For each of `{alpha_on, alpha_pm, delta_alpha}` on the eval window:

| Metric | Purpose |
|--------|---------|
| `n`, `pct_nan` | coverage |
| mean / std / min / max | scale |
| p01 / p50 / p99 | tails |
| daily mean CS σ | dispersion (rankability precursor) |
| frac \|x\| > 5·MAD (or \|x\| > 0.2) | gross outlier rate |

**Pass heuristics (soft):**

- `pct_nan` not exploding vs Phase1 (~<5% on legs)
- daily CS σ of `delta_alpha` ≫ 1e-6 (not near-constant)
- no silent fill / ffill of residuals

### 5.2 Rolling `APM_stat` (construct only)

Paper-shaped rolling statistic on the **session residual difference**:

\[
\delta_{i,t} = \alpha^{\mathrm{ON}}_{i,t} - \alpha^{\mathrm{PM}}_{i,t}
= \texttt{delta\_alpha}
\]

\[
\mathrm{APM\_stat}_{i,t}
=
\frac{\mathrm{mean}(\delta_{i,t-w+1:t})}{\mathrm{std}(\delta_{i,t-w+1:t}) / \sqrt{n_{i,t}}}
,\quad w=20
\]

| Rule | Value |
|------|-------|
| Window | 20 trading days |
| Min periods | 10 (match proxy spirit; document) |
| Group | by `symbol`, time-sorted |
| NaN policy | skip NaN in rolling; if n < min_periods → NaN |
| Cache shift | **false** — store raw dated T |

Output long panel:

```text
date, symbol, delta_alpha, apm_stat, n_obs
```

Optional wide: `apm_stat` pivot for CS checks.

**This is not evaluation.** Do not compute RankIC here as a gate (optional diagnostic only if explicitly labeled `diagnostic_not_gate` — default **off**).

### 5.3 Ret20 CS residual (constructability)

Paper output includes CS residual vs Ret20. Phase2 only proves the object can be built:

1. Load Ret20 panel (existing EOD / Factor_Dev_Lib path; same calendar).
2. Align on `(date, symbol)` with `apm_stat`.
3. Cross-sectional residualize **per date**:

\[
\mathrm{APM\_cs}_{i,t}
=
\mathrm{APM\_stat}_{i,t}
-
\hat\beta_t \cdot \mathrm{Ret20}_{i,t}
-
\hat\alpha_t
\]

(simple CS OLS of `apm_stat` on Ret20 + const, per date; min names e.g. ≥ 50).

Store:

```text
date, symbol, apm_stat, ret20, apm_cs
```

**Gate:** finite coverage on majority of days — **not** IC of `apm_cs`.

### 5.4 PIT / shift alignment (mandatory)

| Layer | Rule |
|-------|------|
| Phase1 cache | raw object at T; `cache_shifted_for_backtest: false` |
| Phase2 signal cache | `apm_stat(T)`, `apm_cs(T)` **unshifted** |
| Eval alignment check | join `signal.shift(1)` to `return(T)` **or** `signal(T)` to `return(T+1)` |

Produce `pit_alignment_report.json`:

```json
{
  "cache_shifted": false,
  "eval_convention": "signal(T) predicts return(T+1)",
  "method": "signal.shift(1) joined to ret_t at evaluation only",
  "sample_check": {
    "signal_date": "...",
    "return_date": "...",
    "note": "return_date == signal_date + 1 trading day"
  },
  "no_leakage_claim": "PM/ON legs use bars <= T close; shift applied only in eval join"
}
```

**Forbidden:** writing shifted signal into `research/cache/apm_session/`.

---

## 6. Cache / artifact layout (Phase2)

Signal construction cache (separate from Phase1 raw legs):

```text
research/cache/apm_session/
  signal/
    apm_stat_{start}_{end}_w20.parquet
    apm_cs_{start}_{end}_w20.parquet
  meta/
    phase2_manifest.json
```

Reports:

```text
research/reports/apm_session_v1/sanity/
├── sanity_summary.json
├── residual_distribution.csv
├── signal_distribution.csv
├── daily_coverage.csv
├── ret20_alignment.csv
├── pit_alignment_report.json
└── sample_signal.csv
```

---

## 7. Module layout (when coding)

| Path | Role |
|------|------|
| `core/l2_features/apm_session_signal.py` | rolling APM_stat + Ret20 CS residual (pure pandas) |
| `run_milestone_c1_apm_session_sanity.py` | load Phase1 cache → build signal → reports |
| `factor_cutting/active_trade.py` | **unchanged** in Phase2 |

Do **not** put rolling/Ret20 into `apm_session_panel_builder.py` (keep panel builder = data objects only).

Non-blocking note from Phase1: `S_DQ_TURN` warning comes from shared EOD loader, not APM. Phase2 must not pull execution/turnover metrics into the sanity path.

---

## 8. Phase2 acceptance gates

| Gate | Pass rule |
|------|-----------|
| **S1 Residual finite** | `pct_nan(delta_alpha)` within reason; CS σ(delta_alpha) ≫ near-zero |
| **S2 APM_stat exists** | `n_dates > 0`, mean daily frac finite `apm_stat` > 0.3 |
| **S3 Ret20 object** | `apm_cs` built; mean daily frac finite > 0.3 (or documented Ret20 gap) |
| **S4 PIT** | `cache_shifted=false`; alignment report shows T → T+1 convention |
| **S5 Provenance** | no `Active_*`; identity still `adapted_replication`; Proxy untouched |

**Hard fail** if S4 broken (shifted cache) or Active_* appears.  
**Soft fail** if coverage thin → fix data window / preheat before scout.

Optional diagnostic (default off): one-month raw RankIC sign check labeled non-gate — do **not** use to flip formula.

---

## 9. Explicit non-goals

- ❌ ICIR / Sharpe / decile / execution grid
- ❌ Scout CSI1000 multi-year
- ❌ Pack v1 / `factor_library.csv` / Registry
- ❌ Overwrite `compute_apm()`
- ❌ Rename Proxy → APM
- ❌ Index-minute “true” upgrade claim
- ❌ Formula retune under this `factor_id`

---

## 10. After Phase2 PASS

```text
After Phase2 PASS
        ↓
Phase3 Scout — docs/milestone_c1_apm_phase3_scout_design.md
        ↓
Phase4 Pack v1 (only if scout admits)
```

Still one factor at a time; Composite / Portfolio remain paused.

---

## 11. Related

- Phase1 panel: `docs/milestone_c1_activetrade_phase1_panel_design.md`
- Phase1 runner: `run_milestone_c1_apm_session_panel.py`
- SmartMoney sanity precedent: `docs/milestone_3_0_iiia41_smartmoney10d_sanity.md`
- Identity schema: `docs/schemas/c1_factor_identity_proposals.yaml`
- Roadmap: `research/reports/factors/ROADMAP.md`
