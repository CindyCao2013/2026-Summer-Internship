# Milestone 2.0 — Composite Alpha Engine v1

**Date:** 2026-07-20  
**Status:** PASS (incremental contribution test)  
**Script:** `run_composite_alpha_v1.py`  
**Outputs:** `research/reports/composite_alpha_v1/`

---

## Scope honored

| Constraint | Result |
|------------|--------|
| No Registry writes | ✅ |
| No formula changes | ✅ |
| Registry factors only | TGD20, D1, FlowDensity20 |
| Exclude D4 / D5 / IdealReversal | ✅ |
| IC-weighted (not 50/50) | rolling RankIC lookback=60d, non-neg, renormalized |

**Signal book:** confirmation 951d · size+industry CS-z · `signal_shift=1` · top_frac=0.10 · cost 15bp RT

---

## Models

| Model | Spec |
|-------|------|
| A | TGD20 |
| B | TGD20 + D1 |
| C | TGD20 + D1 + FlowDensity20 |

---

## Results

| Model | RankIC | RankICIR | Gross Sharpe | Net Sharpe | MDD net | Daily TO |
|-------|--------|----------|--------------|------------|---------|----------|
| A | 0.0415 | 11.28 | 4.06 | **1.28** | −0.084 | 0.646 |
| B | 0.0595 | 10.84 | 3.77 | **2.28** | −0.118 | 0.466 |
| C | 0.0577 | 10.87 | 4.47 | **2.88** | −0.094 | 0.449 |

### Incremental

| Contrast | ΔRankICIR | ΔGross Sharpe | ΔNet Sharpe | ΔDaily TO |
|----------|-----------|---------------|-------------|-----------|
| B − A (add D1) | −0.44 | −0.29 | **+1.00** | −0.180 |
| C − B (add Flow) | +0.03 | **+0.70** | **+0.60** | −0.017 |

### Mean IC weights

| Model | TGD20 | D1 | Flow |
|-------|-------|-----|------|
| B | 0.42 | 0.58 | — |
| C | 0.34 | 0.46 | 0.20 |

### Flow residual vs cores

```
Flow ⊥ (TGD, D1):  resid ICIR ≈ −1.74
raw Flow ICIR ≈ 4.85
```

---

## Interpretation

### 1. Two cores stack (B ≫ A on Net)

Adding D1:

- RankICIR slightly **down** (11.28 → 10.84)
- Net Sharpe **up** ~1.0 (1.28 → 2.28)
- Daily turnover **down** ~28% (0.65 → 0.47)

This is exactly the information-decomposition story:

> TGD = high ICIR temporal alpha, high turnover  
> D1 = liquidity-quality core, lower turnover  
> Together = complementary portfolio economics, not just stacked ICIR

**Baseline for further research:** TGD + D1.

### 2. Flow is enhancer, not third core (C > B on Net, resid still absorbed)

Adding Flow:

- Net Sharpe +0.60, Gross +0.70
- Mean weight ~20% under rolling IC
- Joint residual ICIR still **negative** (−1.74)

So Flow’s value here is **marginal contribution inside the book**, not a new independent IC source. That matches Attribution Review:

```
Satellite / liquidity-conditioned enhancer
```

Do **not** promote Flow to Core. Keep optional in IC-weighted overlays.

### 3. What this milestone answered

Primary question:

> Does the alpha library contain complementary information sources?

Answer:

| Source | Complementary? |
|--------|----------------|
| TGD ⊥ D1 | Yes — and it shows up as Net Sharpe lift |
| Flow vs (TGD+D1) | Portfolio enhancer only; residual IC absorbed / wrong-sign |

---

## Artifacts

| File | Content |
|------|---------|
| `model_comparison.csv` | A/B/C metrics |
| `weights.csv` | Mean IC weights |
| `weights_B_daily.csv` / `weights_C_daily.csv` | Daily weights |
| `incremental_contribution.csv` | B−A, C−B deltas |
| `flow_residual_vs_cores.csv` | Flow ⊥ (TGD,D1) |
| `composite_report.md` | Human report |
| `composite_verdict.json` | Machine summary |
| `charts/model_comparison.png` | Bars |

---

## Explicit non-goals (still deferred)

- Registry `alpha_role` field
- D4 / D5 / IdealReversal in composite
- Equal-weight kitchen-sink
- Production CSI1000 Dual Benchmark portfolio track
- D5 direction flip validation

---

## Next

1. Treat **B (TGD+D1)** as Composite baseline.
2. Treat **C** as optional satellite-enhanced variant (document Flow role).
3. Optional: stress Net Sharpe under higher cost / buffer execution.
4. Separate track: D5 `pending_direction_validation`.
