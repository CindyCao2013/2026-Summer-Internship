# C1.1 — APM_SessionResidual Session Panel Builder + Cache Strategy

**Date:** 2026-07-21  
**Status:** Phase1 **PASS** (2024-06 smoke) — panel + cache closed  
**Identity:** `APM_SessionResidual` · `adapted_replication`  
**Phase0:** [`docs/milestone_c1_activetrade_phase0_identity.md`](milestone_c1_activetrade_phase0_identity.md) (**ACCEPTED**)  
**Next:** [`docs/milestone_c1_apm_phase2_sanity_design.md`](milestone_c1_apm_phase2_sanity_design.md)  
**Pattern:** mirror SmartMoney III-A4.1 (month-chunk → cache → coverage), stop at panel

---

## 0. Scope lock

| In (C1.1 Phase1) | Out |
|------------------|-----|
| Session return builder | Factor RankIC / ICIR / H-L |
| Index residual **adapter** (EOD proxy) | Index minute (unavailable) |
| APM panel cache under `research/cache/apm_session/` | Pack v1 / `factor_library.csv` |
| Coverage · missing · alignment · PIT checks | Registry · Composite · Portfolio |
| `ActiveTradeProxy` untouched | Rename / promote proxy |
| No `Active_*` columns in APM cache | Case A imbalance factor |

```text
C1.1
 |
 +-- session return builder
 +-- index residual adapter
 +-- APM panel cache
 |
 +-- (later) smoke → sanity → scout → pack → library
```

**This doc = panel builder + cache strategy.** Design accepted; coding in progress.  
Still **no IC** in Phase1.

### Implementation locks (post-review)

| Lock | Rule |
|------|------|
| PM first Close | **first available bar with Bartime ≥ 13:01**, not “exact 13:01 required” |
| Cache vs eval | Cache stores **raw** legs at date T; **`shift(1)` only in Phase2+ eval** |
| Module | `core/l2_features/apm_session_panel_builder.py` only — do **not** grow `factor_cutting/active_trade.py` |
| Calendar cache | Persist trade calendar + missing-day diagnostics under `calendar/` |

---

## 1. Identity (frozen for code)

```yaml
factor_id: APM_SessionResidual
paper: APM因子模型 / 主动买卖
identity_class: adapted_replication
status_when_coding: design_only → research_panel (after cache exists)
direction_paper: positive_ic
data_level: minute_plus_eod_index
formula_version: apm_session_v1_adapted_eod_index
```

**Does not block:** PDF checklist / index minute (upgrade path later).  
**Does block claiming:** `true_replication` while index residual uses EOD daytime proxy.

---

## 2. Exact session definitions (no invention)

### 2.1 Stock overnight (EOD)

Reuse TGD helper semantics:

\[
r^{\mathrm{ON}}_{i,t} = \frac{\mathrm{Open}_{i,t}}{\mathrm{Close}_{i,t-1}} - 1
\]

Source: existing EOD Open/Close panels (`Factor_Dev_Lib` / same loaders as TGD).  
Reuse `core.l2_features.tgd_panel_builder.overnight_return_long` where practical.

### 2.2 Stock afternoon PM (minute)

Source: `dfs://QV_Trade_to_MinuteBar / Stock_one_minute`

Session filter (PM only):

```text
second(Bartime) ∈ [13:01:00, 15:00:00]
```

Per `(Symbol, Date)`, bars with `Bartime >= 13:01` and `<= 15:00`, sorted by Bartime:

\[
r^{\mathrm{PM}}_{i,t} = \frac{\mathrm{Close}^{\mathrm{last\,PM}}_{i,t}}{\mathrm{Close}^{\mathrm{first\,available\,PM}}_{i,t}} - 1
\]

**`pm_start_rule`:** `first_available_bar_after_13_01`  
(= first bar in the filtered set; **not** “must have exact 13:01 print”).

| Rule | Value |
|------|-------|
| Min PM bars | ≥ 2 (else NaN) |
| Columns selected | `Symbol, Date, Bartime, Close` **only** |
| Forbidden | any `Active_*`, Volume/Amount (not needed) |
| A-share filter | symbol starts with `6\|0\|3` (match TGD/SmartMoney) |

**AM bars are not used** for the PM leg (paper knife = afternoon bucket).

### 2.3 Index session proxy (adapted — honesty)

Index minute **absent** from stock minute table. Adapted definitions:

| Leg | Formula | Fidelity |
|-----|---------|----------|
| Index overnight | \(\mathrm{IdxOpen}_t / \mathrm{IdxClose}_{t-1} - 1\) | ≈ true overnight |
| Index day proxy | \(\mathrm{IdxClose}_t / \mathrm{IdxOpen}_t - 1\) | **full daytime**, not PM |

Default index: `000852.SH` (CSI1000; align with production eval habit).  
Document `index_code` in meta; do not silently switch.

**Adapted mismatch (must stay visible in meta):**

```text
stock PM residual uses stock PM vs index FULL DAY
→ not true session-matched residual
→ identity_class remains adapted_replication
```

### 2.4 Session residuals (panel legs)

\[
\alpha^{\mathrm{ON}}_{i,t} = r^{\mathrm{ON}}_{i,t} - r^{\mathrm{ON}}_{\mathrm{idx},t}
\]

\[
\alpha^{\mathrm{PM}}_{i,t} = r^{\mathrm{PM}}_{i,t} - r^{\mathrm{DAY}}_{\mathrm{idx},t}
\quad\text{(adapted)}
\]

### 2.5 APM residual panel object (Phase1)

Phase1 builds the **daily residual legs + optional difference**, not the final CS-residualized factor:

| Column | Meaning |
|--------|---------|
| `r_on` | stock overnight |
| `r_pm` | stock PM |
| `r_on_idx` | index overnight |
| `r_day_idx` | index day proxy |
| `alpha_on` | residual overnight |
| `alpha_pm` | residual PM (adapted) |
| `delta_alpha` | \(\alpha^{\mathrm{ON}} - \alpha^{\mathrm{PM}}\) |

Rolling APM t-stat / Ret20 CS residual = **Phase2+** (sanity / scout), not required to close Phase1 cache gate.

---

## 3. Architecture

Unlike SmartMoney (rolling 10d minute pool), APM session legs are **mostly single-day aggregates**:

```text
DDB month chunk (PM Close)     →  stock_pm daily
EOD Open/Close                 →  stock_overnight
EOD index Open/Close           →  index_session_proxy
Python join + subtract         →  apm_residual_panel
```

No Python row-loop over full minute history. Prefer DDB `group by Symbol, Date` for PM first/last Close.

---

## 4. Cache layout

```text
research/cache/apm_session/
├── meta/
│     formula_version.json
│     build_manifest.json          # ranges, index_code, adapted flags
├── calendar/
│     trade_calendar_{start}_{end}.json   # source, days, missing diagnostics
├── stock_overnight/
│     stock_overnight_{start}_{end}.parquet
├── stock_pm/
│     stock_pm_YYYYMM.parquet      # month chunks from DDB
│     stock_pm_{start}_{end}.parquet   # optional concat view
├── index_session_proxy/
│     index_session_proxy_{index}_{start}_{end}.parquet
└── residual_panel/
      apm_residual_panel_{start}_{end}.parquet
```

**Calendar purpose:** session-alignment debug (suspend / index holiday / minute gap), not factor math.
User-facing Phase1 deliverables (aliases / final joins OK):

```text
research/cache/apm_session/
  stock_overnight.parquet          # or dated shard above
  stock_pm_return.parquet
  index_session_proxy.parquet
  apm_residual_panel.parquet
```

### Schemas

**stock_overnight**

| col | type |
|-----|------|
| date | datetime64 |
| symbol | str |
| overnight_return | float64 |

**stock_pm_return**

| col | type |
|-----|------|
| date | datetime64 |
| symbol | str |
| pm_return | float64 |
| pm_n_bars | int |
| pm_close_first | float64 |
| pm_close_last | float64 |

**index_session_proxy**

| col | type |
|-----|------|
| date | datetime64 |
| index_code | str |
| index_overnight | float64 |
| index_day | float64 |
| adapted | bool = true |

**apm_residual_panel**

| col | type |
|-----|------|
| date | datetime64 |
| symbol | str |
| r_on, r_pm | float64 |
| r_on_idx, r_day_idx | float64 |
| alpha_on, alpha_pm, delta_alpha | float64 |

### meta flags (required)

```json
{
  "factor_id": "APM_SessionResidual",
  "formula_version": "apm_session_v1_adapted_eod_index",
  "identity_class": "adapted_replication",
  "index_code": "000852.SH",
  "index_pm_matched": false,
  "index_residual_method": "eod_overnight_plus_eod_daytime_proxy",
  "pm_start_rule": "first_available_bar_after_13_01",
  "cache_shifted_for_backtest": false,
  "active_star_columns": false,
  "proxy_factor_id_untouched": "ActiveTradeProxy"
}
```

**PIT note:** Panel cache stores economic objects dated T (bars ≤ T close).  
Evaluation in Phase2+ must use `shift(1)` before joining return(T+1). **Do not shift inside cache.**
---

## 5. DDB / loader sketch

### 5.1 PM daily aggregate (L1)

```dolphindb
t = loadTable('dfs://QV_Trade_to_MinuteBar','Stock_one_minute')
m = select Symbol, Date, second(Bartime) as Bartime, Close
from t
where Date >= {s} and Date <= {e}
  and second(Bartime) >= 13:01:00 and second(Bartime) <= 15:00:00
m = select Symbol, Date, Bartime, Close from m context by Symbol, Date csort Bartime
select Symbol, Date,
  first(Close) as pm_close_first,
  last(Close) as pm_close_last,
  first(Bartime) as pm_bartime_first,
  last(Bartime) as pm_bartime_last,
  count(*) as pm_n_bars
from m
group by Symbol, Date
```

**Important:** `context by` alone keeps all minute rows (~11M/month). Always finish with `group by Symbol, Date`.

Then Python: `pm_return = pm_close_last / pm_close_first - 1` if `pm_n_bars >= 2`.  
`pm_start_rule = first_available_bar_after_13_01`.

### 5.2 Overnight (L2)

EOD panels → `overnight_return_long(open_, close)`.

### 5.3 Index proxy (L3)

Wind / DDB index EOD (`S_DQ_OPEN`, `S_DQ_CLOSE`) for `index_code`.  
Same calendar as stock trading days; left-join; missing index day → NaN residuals.

### 5.4 Residual join (L4)

```text
stock_overnight ⋈ stock_pm ⋈ index_session_proxy
  → alpha_on, alpha_pm, delta_alpha
```

Inner join on date for index; outer for symbols (missing PM → NaN).

---

## 6. Module layout (when coding)

| Path | Role |
|------|------|
| `core/l2_features/apm_session_panel_builder.py` | L1–L4 cache orchestration |
| `run_milestone_c1_apm_session_panel.py` | build cache + coverage/PIT/alignment reports |
| `research/reports/apm_session_v1/phase1/` | coverage · alignment · pit · sample_checks |

**Keep session panel out of** `factor_cutting/active_trade.py` (proxy / knife stub only).  

**Do not create** `factor_specs/APM_SessionResidual.yaml` until scout path.  
**Do not write** Pack v1 under `research/reports/factors/` in Phase1.

### Phase1 report outputs

```text
research/reports/apm_session_v1/phase1/
├── coverage_report.json
├── alignment_report.json
├── pit_report.json
├── build_log.txt
└── sample_checks.csv
```

`sample_checks.csv` (~20 names): `date, symbol, prev_close, open, pm_first_time, pm_last_time, pm_first_close, pm_last_close`.

---

## 7. Phase1 acceptance gates (panel only)

| Gate | Pass rule |
|------|-----------|
| **G1 Coverage** | Smoke window (e.g. 1 month) has stock PM rows for majority of A-shares with minute data |
| **G2 Missing** | Report `pct_nan` for `r_on`, `r_pm`, `alpha_on`, `alpha_pm`; no silent fill |
| **G3 Session alignment** | PM bartime range ⊆ 13:01–15:00; overnight uses prev Close (spot-check 20 names) |
| **G4 PIT / leakage** | Signal date T uses only bars ≤ T close; no future Close; index same-day EOD OK for same-day residual (document; shift(1) before return join in later phases) |
| **G5 Provenance** | Cache columns contain **zero** `Active_*`; meta `adapted=true`; Proxy pack unchanged |

Fail any gate → fix builder; do not proceed to sanity/scout.

---

## 8. Explicit non-goals (C1.1 Phase1)

- ❌ RankIC / ICIR / decile / execution
- ❌ Ret20 CS residual (Phase2+)
- ❌ Rolling APM t-stat as library factor (Phase2+)
- ❌ `factor_library.csv` row for `APM_SessionResidual`
- ❌ Registry
- ❌ Touch `ActiveTradeProxy`
- ❌ Select `Active_*` into APM caches
- ❌ Claim `true_replication`

---

## 9. Later phases (not this design)

```text
Phase1  Session panel + cache + coverage     ✅ PASS
Phase2  Sanity (rolling APM_stat + Ret20 CS + shift)  ← NEXT
Phase3  Scout (CSI1000 window · IC / mono / TO)
Phase4  Pack v1 → factor_library.csv
```

Non-blocking Phase1 note: shared EOD loader may warn `S_DQ_TURN` missing — not an APM field; ignore for panel gates.

Upgrade to `true_replication` only if index minute or PDF-signed EOD equivalence lands.

---

## 10. Related

- Phase2 sanity design: `docs/milestone_c1_apm_phase2_sanity_design.md`
- Phase0 identity: `docs/milestone_c1_activetrade_phase0_identity.md`
- SmartMoney cache pattern: `docs/milestone_3_0_iiia41_smartmoney10d_impl_design.md`
- TGD overnight helper: `core/l2_features/tgd_panel_builder.py`
- Proxy honesty: `docs/milestone_3_0_active_trade_proxy.md`
- Roadmap: `research/reports/factors/ROADMAP.md`
