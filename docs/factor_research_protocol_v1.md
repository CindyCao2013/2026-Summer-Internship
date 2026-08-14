# Factor Research Protocol v1

**Status:** frozen (Milestone 0 + review patch)  
**Date:** 2026-07-17  
**Scope:** research rules and artifact schema only — no code refactor.

This document is the **research contract** for FACTOR_DEV.  
All future factor work (paper replication, formula search, L2, ML enhancement) must obey it.

Related existing packs (legacy Template v1, not yet migrated):

- `research/reports/factors/TGD20/`
- `research/reports/factors/FlowDensity20/`
- `research/reports/factors/README.md`

Schema examples: `docs/schemas/`

**Changelog (review patch):** Pearson IC + RankICIR required; `mechanism/` + `execution/` in pack; family taxonomy frozen; data coverage exception policy; Production horizon locked to 20D; admission = manual review (no numeric bar).

---

## A. Factor lifecycle

### Allowed statuses

| Status | Meaning |
|--------|---------|
| `discovery` | Idea / paper / search hit; no standard pack yet |
| `testing` | Spec + compute exist; evaluation incomplete or non-standard |
| `candidate` | Full pack under Research Track; Production Track pending or provisional |
| `validated` | Production Track complete + **manual admission review** passed; formula frozen |
| `production` | Admitted to live / portfolio construction stack |
| `retired` | Superseded, failed integrity, or redundant; keep pack for audit |

### Forbidden informal labels

Do **not** use as status: `good`, `best`, `strong`, `alpha_xxx`, `validated_single_factor`, `validated_single_factor_candidate`.

Legacy packs may keep historical labels in notes until Milestone 2 migration maps them to the table above.

### Transition rules

```
discovery
    │  write factor_spec.yaml; start compute
    ▼
testing
    │  complete Research Track pack (metrics + required charts)
    ▼
candidate
    │  complete Production Track (20D, CSI1000, size+industry, 15bp);
    │  manual admission review; freeze formula (frozen_formula: true)
    ▼
validated
    │  portfolio / risk / capacity review (Phase 6+)
    ▼
production

Any status ──(integrity fail / superseded / redundant)──► retired
```

Rules:

1. **No skip:** cannot jump `discovery → validated` without Research + Production packs.
2. **Formula freeze at `validated`:** no retuning under the same `factor_id`. New variant → new `factor_id`.
3. **`retired` is terminal** for that `factor_id` (archive only; do not revive without a new id).
4. **Downgrade** (`validated` → `candidate`) only if Production Benchmark definition changes or a data bug is found; document reason in `factor_card.yaml`.

### Admission policy (no numeric bar in v1)

Protocol v1 does **not** freeze ICIR / Net Sharpe / Turnover thresholds. Sample size is insufficient; hard cuts would mis-kill low-frequency but valuable factors.

```yaml
admission:
  status: validated          # target status after review
  requires_manual_review: true
  # no numeric thresholds in Protocol v1
```

Promotion `candidate → validated` requires human review of Production metrics + mechanism integrity, recorded in `factor_card.yaml` notes.

---

## B. Dual Benchmark

Every factor has **two evaluation tracks**. They answer different questions and must not be mixed when ranking factors.

### Research Track (alpha discovery)

| Field | Value |
|-------|-------|
| Universe | ALL tradable A-shares (ST / IPO filters as per existing data loaders) |
| Period | 2018-01-01 ~ 2025-12-31 |
| Forward return | **20 trading days** (official Research horizon) |
| Neutralization | **none** in root Research slot; neutralization ladders → `diagnostics/neutralization/` |
| Portfolio | 10-decile long–short |
| Cost | optional / diagnostic only |
| Purpose | discover whether the signal has real cross-sectional alpha |

### Production Track (factor admission)

| Field | Value |
|-------|-------|
| Universe | CSI1000 |
| Period | 2018-01-01 ~ 2025-12-31 |
| Forward return | **20 trading days only** |
| Neutralization | industry + size |
| Portfolio | 10-decile long–short |
| Cost | **15 bp** round-trip (used for Net Sharpe) |
| Rebalance | daily (unless factor_spec explicitly states otherwise and is documented) |
| Purpose | comparable, investability-aware admission into the registry |

### Horizon policy (frozen)

| Track | Horizon |
|-------|---------|
| Production | **20D only** — no alternate production horizon |
| Research (official slot) | **20D** |
| Diagnostics | **free** — 1D / 5D / 10D / 60D etc. allowed under `diagnostics/` only |

Do not promote a 5D-only result into root `metrics.json` production.

### Universe membership (CSI1000)

| Rule | Policy |
|------|--------|
| Must declare | `universe_membership: pit \| non_pit` in `factor_card.yaml` |
| Preferred | **point-in-time (PIT)** constituents |
| Allowed | non-PIT for research / early packs; must be declared |
| Survivorship | non-PIT results cannot be treated as Production-quality without noting bias |

Harness (Milestone 1) must document which loader convention is used; Protocol does not block non-PIT research.

### Dual-benchmark rationale

- Research-only can overstate tradable alpha (small-cap / liquidity bias).
- Production-only can kill genuine but capacity-sensitive signals before mechanism understanding.
- **Registry primary columns use Production Track only.**
- Research Track numbers may appear in `summary.md` and `diagnostics/`, never as cross-factor “who is stronger” in the registry.

---

## B2. Data coverage exception policy

Not all factors can cover `2018-01-01 ~ 2025-12-31` (especially L2 / minute).

### Required fields

In `factor_card.yaml` and/or `metrics.json`:

```yaml
data_coverage:
  requested: "2018-01-01_2025-12-31"
  actual: "2020-01-01_2025-12-31"
  exception_reason: "minute / L2 data availability"
```

### Rules

1. Always declare **requested** Dual Benchmark period even if incomplete.
2. Always declare **actual** window used for reported metrics.
3. If `actual` ≠ `requested`: set `coverage_exception: true` and fill `exception_reason`.
4. With an open coverage exception, status is at most **`candidate`** unless manual review explicitly accepts the shorter window for `validated` (document in card).
5. Do not silently stretch period labels; never claim 2018–2025 when data start later.

Example (TGD-shaped):

```yaml
data_coverage:
  requested: "2018-01-01_2025-12-31"
  actual: "2022-01-28_2025-12-31"
  exception_reason: "minute return / Gu-Gd lineage availability"
```

---

## C. Standard Metrics

Every evaluation (Research and Production official slots) must report:

| Metric | Role |
|--------|------|
| **IC** | Mean **Pearson** cross-sectional IC (linear predictability) — **required**, not optional |
| **RankIC** | Mean Spearman IC (ranking ability) |
| **ICIR** | `mean(Pearson IC) / std(Pearson IC) * sqrt(250)` |
| **RankICIR** | `mean(RankIC) / std(RankIC) * sqrt(250)` |
| Annualized IC | `mean(Pearson IC) * sqrt(250)` (also report Annualized RankIC if useful) |
| Gross Sharpe | H–L (top−bottom decile) Sharpe before cost |
| Net Sharpe | Sharpe after cost on turnover |
| Maximum Drawdown | MDD of direction-adjusted H–L |
| Turnover | Mean daily \|Δw\| of H–L book |
| Monotonicity | Spearman(decile_id, mean_decile_return) |

Rationale: RankIC answers ordering; Pearson IC answers linear prediction (important for ML factors and regression stacks). Both are required.

Optional auxiliary (not in root admission set): yearly RankIC stability → often under `diagnostics/` or `mechanism/`.

---

## D. Required Charts

Every factor pack **must** contain:

```
charts/
  ic_curve.png
  decile_return.png
  cumulative_long_short.png
  turnover.png
```

Rules:

- Charts under `charts/` are the **official** figures for the Production (or declared) track in root `metrics.json`.
- Mechanism / buffer / exposure figures live under `mechanism/` or `execution/`, not as substitutes for the four required charts.

---

## E. Factor Pack Schema

Canonical path:

```
research/reports/factors/{factor_id}/
```

### Required layout

```
research/reports/factors/{factor_id}/
  factor_card.yaml
  metrics.json
  summary.md
  charts/
    ic_curve.png
    decile_return.png
    cumulative_long_short.png
    turnover.png
  mechanism/                 # why the factor works (required directory; may be sparse)
  diagnostics/
    universe/
    neutralization/
    period/
    horizon/                 # non-20D experiments only
  execution/                 # turnover / buffer / cost grids (investability)
```

### Directory roles

| Path | Role |
|------|------|
| `factor_card.yaml` | Identity, lifecycle, family, coverage, admission |
| `metrics.json` | Official comparable Production numbers (+ optional research block) |
| `summary.md` | Human one-pager |
| `charts/` | Four required admission figures |
| `mechanism/` | Economic / structural attribution (see §E2) |
| `diagnostics/` | Alternate universe / neut / period / horizon — **not** comparable |
| `execution/` | Buffer, rebalance, cost, capacity-style grids |

### Root vs diagnostics rule

- **Root `metrics.json`:** Production Track @ **20D** only for registry ranking.
- **`diagnostics/`:** any other universe, period, neutralization, **horizon**, cost assumption.
- **`mechanism/`:** explains α; not a place to hide a better cherry-picked ICIR as “official.”

Schema examples: `docs/schemas/`.

---

## E2. Factor Mechanism Layer

This platform’s differentiator is not only IC/Sharpe, but **why** the signal works.

Every pack must include a `mechanism/` directory. Contents are factor-specific; empty placeholder + README is acceptable only at `testing`.

Examples:

```
# TGD-shaped
mechanism/
  gu_gd_analysis.csv
  tau_vs_tgd.csv
  residual_validation.csv

# Flow-shaped
mechanism/
  amount_neutral.csv
  buy_sell_component.csv
```

Composite later combines **information layers** (temporal + liquidity-conditioned flow), not raw tickers of factor names — mechanism packs are the evidence for that.

---

## F. Production vs Diagnostic rule

| | Production | Diagnostic |
|---|------------|------------|
| Definition | Official comparable result under frozen Production Track (20D) | Research / sensitivity analysis |
| Where | Root `metrics.json` (+ matching `charts/`) | `diagnostics/**` only |
| Used in registry? | Yes | No |
| Cross-factor comparison? | Yes | **Never** |
| Cherry-picking highest ICIR? | Forbidden | Allowed only inside one factor’s mechanism study |

**Hard rule:** never compare Factor A’s diagnostic CSI300 raw ICIR to Factor B’s production CSI1000 size+industry ICIR and call one “better.”

---

## G. Family taxonomy (frozen)

`family` on every card / registry row must use this enum (multi-label allowed as a list when needed):

| Family | Examples |
|--------|----------|
| `temporal_information` | TGD20 |
| `liquidity` | amount / turnover style; Flow’s liquidity channel |
| `microstructure` | Ideal Reversal, Ideal Amplitude, Active Trade; Flow interaction |
| `momentum` | classic momentum / long-horizon path |
| `value` | PE / PS style |
| `quality` | fundamentals quality |
| `volatility` | amplitude / vol anomaly |
| `event` | earnings / corporate events |
| `alternative_data` | non-standard external |

Examples:

- TGD20 → `temporal_information`
- FlowDensity20 → `[microstructure, liquidity]`
- Ideal Reversal → `microstructure`

Do not invent ad-hoc family strings (`liquidity_flow_interaction` may remain in notes as a mechanism class, but registry `family` uses the enum above).

---

## H. Future research contract

Every **new** factor must follow:

```
factor_spec.yaml
      ↓
factor computation
      ↓
standard evaluation  (Research Track → Production Track @ 20D)
      ↓
report pack          (schema in §E, including mechanism/)
      ↓
registry             (Milestone 2+)
```

### `factor_spec.yaml` (minimum fields)

See `docs/schemas/factor_spec.example.yaml`.

Required ideas:

- `factor_id`, `family` (enum §G), `source`
- `object` / `knife` / `output` when cutting-style
- `data.required`, `data_level` (`EOD` / `L2` / `minute`)
- `data_coverage` when applicable
- `formula` description
- `frozen_formula` boolean

### What Protocol milestones do **not** do yet

- No `run_factor_research.py` (Milestone 1)
- No registry CSV upgrade (Milestone 2)
- No bulk file moves of TGD20 / FlowDensity20 packs
- No Ideal Reversal implementation

---

## I. Legacy pack policy (no bulk move)

Do **not** relocate large existing report trees into a parallel `standard/` copy.

Preferred approach for Milestone 2+:

1. Keep packs at `research/reports/factors/{factor_id}/`.
2. Add Protocol metadata (`schema_version`, lifecycle status, `data_coverage`, family).
3. Optionally add thin links / pointers from long-form legacy dirs (`research/reports/tgd_v1/`, `l2_flow_density_v1/`) without duplicating large artifacts.
4. Gradually place mechanism / execution CSVs into `mechanism/` and `execution/` (or symlink).

TGD20 is already close to v1; migration is **adapter + metadata**, not rewrite.

---

## Compatibility notes (existing assets)

| Topic | Current state | Protocol v1 expectation |
|-------|---------------|-------------------------|
| Status strings | `validated_single_factor*` | Map in Milestone 2 → `validated` / `candidate` |
| Period | Often `2022-01-28_2025-12-31` | `data_coverage.requested` + `actual` |
| Universe in cards | Often `ALL` | Production official = CSI1000 |
| Cost | ~7.5 bp in some metric defs | Production Net Sharpe = **15 bp** |
| Charts | pack root / `figures/` | Official under `charts/` |
| Mechanism files | CSV at pack root | Prefer `mechanism/` |
| Execution files | pack root | Prefer `execution/` |
| Report name | `factor_report.md` | Prefer `summary.md`; alias OK during transition |
| Family | free-form categories | Enum §G |

TGD20 formula remains frozen. Protocol does not authorize retuning.

---

## Resolved vs open (post review)

### Resolved in this patch

| # | Topic | Decision |
|---|--------|----------|
| Q1 | Admission bar | **No numeric thresholds**; `requires_manual_review: true` |
| Q2 | Pearson IC | **Required**; also require **RankICIR** |
| Q3 | Horizon | Production/Research official = **20D only**; other horizons → diagnostics |
| Q4 | CSI1000 PIT | **Declare** membership; PIT preferred; non-PIT allowed if declared |
| Q5 | L2 coverage | **`data_coverage`** block + exception policy (§B2) |
| Q6 | Legacy migration | **No bulk move**; metadata / link adapter |
| Q8 | Family | **Frozen enum** (§G) |

### Still open (Harness / later)

1. Exact loader convention for CSI1000 PIT in code (document in Milestone 1).
2. Earliest reliable L2/minute date for TGD/Flow (fill real `data_coverage.actual`).
3. `summary.md` vs `factor_report.md` dual-file sunset date.
4. When to introduce numeric admission guidelines (after more Production-comparable packs exist) — **not** before Matrix.
