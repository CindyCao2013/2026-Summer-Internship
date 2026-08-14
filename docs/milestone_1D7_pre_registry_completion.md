# Milestone 1D.7 — Pre-Registry Pack Completion

**Date:** 2026-07-20  
**Status:** PASS  
**Scope:** Evaluation / report completeness only — **no formula changes**, **no Registry**.

---

## Goal

Close the two remaining candidate-pack gaps before Registry:

| Pack | Gap | Action |
|------|-----|--------|
| FlowDensity20 | 3 protocol charts missing | Generate from frozen `net_active_flow_mktcap_20d` |
| D1_LiquidityQuality60d | No execution / Net Sharpe | `execution_layer` grid @ 15bp RT |

---

## What ran

```bash
OMP_NUM_THREADS=1 python run_milestone_1d7_pack_completion.py
python factor_report_generator_v2.py --factor FlowDensity20
python factor_report_generator_v2.py --factor D1_LiquidityQuality60d
# TGD20 regen for safety — narrative/metrics unchanged
```

Script: `run_milestone_1d7_pack_completion.py`

---

## Results

### FlowDensity20

| Item | Result |
|------|--------|
| Signal | `size_industry` CS-z of `net_active_flow_mktcap_20d` (confirmation 951d) |
| Charts | `ic_curve` · `decile_return` · `cumulative_long_short` installed |
| Mechanism | unchanged (still interaction / amount-orth) |
| Generator validation | **`ok=true`**, missing_charts=`[]` |
| Status | `candidate` |

Source figures: `research/reports/l2_flow_density_v1/protocol_charts_1d7/`  
Declared in `factor_specs/FlowDensity20.yaml → artifacts.copy`.

### D1_LiquidityQuality60d

| Item | Result |
|------|--------|
| Signal | frozen `low_vol_liquidity_quality_60d` (raw CS-z), 1455d |
| Best execution | `raw\|daily\|buffer_5_15` |
| Gross Sharpe | ≈ 1.81 |
| **Net Sharpe** | **≈ 1.38** |
| Daily TO | ≈ 0.23 |
| Cost | 15bp round-trip |
| Generator validation | **`ok=true`** |
| Status | **`testing` → `candidate`** |

Artifacts: `research/reports/d1_liquidity_density_v1/execution/` + pack `execution_summary.csv`.

Interpretation: investability **improves** vs raw TO drag, but Net Sharpe is modest vs TGD/Flow — correctly stays **candidate**, not validated.

### TGD20

Executive Summary / Mechanism / Execution sections **unchanged** after regen.

---

## Acceptance

| Criterion | Result |
|-----------|--------|
| TGD20 unchanged (core chapters) | ✅ |
| Flow + D1 complete candidate packs | ✅ |
| Missing artifacts reduced (Flow charts filled) | ✅ |
| No Registry creation | ✅ |
| No formula retune | ✅ |

---

## Suggested Registry posture (next: 1E)

```
validated   TGD20
candidate   FlowDensity20, D1_LiquidityQuality60d
testing     IdealReversal
```

Do **not** register diagnostics (`εd`, `τ`, buffers, `M_high`…).

---

## Next

```
1D.7 Pre-Registry completion   ✅
        ↓
1E Factor Registry v1
        ↓
Factor Similarity Matrix
        ↓
Composite Alpha Engine
```
