# Milestone 1E — Factor Registry v1

**Date:** 2026-07-20  
**Status:** PASS  
**Scope:** Registry creation only — **no** formula changes, **no** new evaluation, **no** composite.

---

## Goal

Freeze an **alpha identity inventory** after Research Pack / Generator completion:

```
factor_id
  = one economic hypothesis
  + one investable expression
```

Not:

```
εd / τ / buffer_5_15 / M_high   ← diagnostics & implementations
```

---

## Artifacts

| Path | Role |
|------|------|
| `research/registry/factor_registry.yaml` | Source of truth (nested provenance) |
| `research/registry/factor_registry.csv` | Flat inventory |
| `research/registry/README.md` | Rules |
| `tests/test_factor_registry_v1.py` | Admission / layer-discipline tests |

---

## First inventory

| Status | factor_id | Pack? | Headline snapshot |
|--------|-----------|-------|-------------------|
| **validated** | TGD20 | ✅ | RankICIR≈11.29 (size+ind); NetSharpe≈2.32; TO≈0.30 |
| **candidate** | FlowDensity20 | ✅ | RankICIR≈4.85; NetSharpe≈2.88; interaction |
| **candidate** | D1_LiquidityQuality60d | ✅ | RankICIR≈6.01; NetSharpe≈1.38 (raw exec) |
| **candidate** | D4_WinnerSentimentReversal5d | ❌ library only | Base3 leg; solo metrics pending |
| **candidate** | D5_UpsideFragility20d | ❌ library only | Base3 leg; **mono weak flag** |
| **testing** | IdealReversal | ✅ | Soft bar fail; paper replication |

`production_ready=false` for all rows (CSI1000 Dual Benchmark / Matrix still open).  
`correlation_cluster=null` until Milestone 1F.

---

## Pre-Registry audit note (D1 signal identity)

1D.7 execution tested:

```
low_vol_liquidity_quality_60d → raw CS z-score → execution_layer
```

Not:

```
size+industry residual
```

Documented in:

- `factor_specs/D1_LiquidityQuality60d.yaml` → `evaluation_signal` / `production_signal`
- Registry row notes + YAML fields
- D1 report `execution_narrative`

**Rule:** do not quote confirmation RankICIR and raw-exec NetSharpe as if they were the same neutralization book without labeling.

---

## Layer discipline tests

`tests/test_factor_registry_v1.py` asserts:

1. Unique `factor_id`  
2. YAML ↔ CSV id alignment  
3. Status ∈ Protocol enum  
4. Banned diagnostic / execution substrings cannot be registry ids  
5. First-inventory status table frozen as above  
6. No Base3 / composite stack registered as a factor_id  

```bash
/opt/conda/anaconda3/envs/base_93/bin/python -m unittest tests.test_factor_registry_v1 -v
```

---

## Explicit non-goals

- Factor Similarity Matrix (→ **1F**)  
- Composite Alpha Engine  
- Bulk soft-bar EOD registration (`amount_stability`, …)  
- Auto-admission by ICIR/Sharpe thresholds  

---

## Next

```
1E Registry v1          ✅
        ↓
1F Factor Similarity Matrix
   (IC corr + residual IC: TGD / Flow / D1 / D4 / D5)
        ↓
Composite Alpha Engine
```
