# Composite Alpha Engine v1 — Incremental Contribution Test

**Window:** 2022-01-28 → 2025-12-31 (951d confirmation)
**Signal book:** size+industry CS-z · IC-weighted ranks · lookback=60d · top_frac=0.1
**Cost:** 0.0015 round-trip

## Models

| Model | Spec |
|-------|------|
| A | TGD20 |
| B | TGD20 + D1 (IC-weighted) |
| C | TGD20 + D1 + FlowDensity20 (IC-weighted) |

## Comparison

| Model | RankIC | RankICIR | Gross Sharpe | Net Sharpe | MDD net | Daily TO |
|-------|--------|----------|--------------|------------|---------|----------|
| A_TGD | 0.0415 | 11.28 | 4.06 | 1.28 | -0.084 | 0.646 |
| B_TGD_D1 | 0.0595 | 10.84 | 3.77 | 2.28 | -0.118 | 0.466 |
| C_TGD_D1_Flow | 0.0577 | 10.87 | 4.47 | 2.88 | -0.094 | 0.449 |

## Mean IC weights (confirmation, after warm-up)

        model        factor  mean_weight
     B_TGD_D1         TGD20       0.4227
     B_TGD_D1            D1       0.5773
C_TGD_D1_Flow         TGD20       0.3406
C_TGD_D1_Flow            D1       0.4634
C_TGD_D1_Flow FlowDensity20       0.1961

## Incremental contribution

 contrast           add  delta_rank_icir  delta_gross_sharpe  delta_net_sharpe  delta_daily_turnover  delta_mdd_net
B_minus_A            D1          -0.4398             -0.2906            0.9968               -0.1795        -0.0341
C_minus_B FlowDensity20           0.0256              0.6950            0.6018               -0.0170         0.0244

## Flow residual vs cores

- `Flow_perp_TGD_D1`: raw Flow ICIR=4.85, resid ICIR=-1.74
- CS residual of Flow on [TGD, D1]; IC on aligned panels

## Interpretation

- **B vs A (add D1):** ΔNet Sharpe=+1.00, ΔRankICIR=-0.44. Supports two-core combination.
- **C vs B (add Flow):** ΔNet Sharpe=+0.60, ΔRankICIR=+0.03. Flow acts as portfolio enhancer on this book.
- Flow⊥(TGD,D1) resid ICIR=-1.74 (aligns with Attribution Review: Flow largely D1-overlapping).

## Explicit exclusions

- D4, D5, IdealReversal not in Composite v1 (per Attribution Review).
- No Registry writes. No formula changes.

## Interpretation (locked)

Do **not** promote three cores from Net Sharpe stacking.

| Role | Factor | Why |
|------|--------|-----|
| Primary alpha source | TGD20 | Strong RankICIR; high fee drag alone |
| Independent alpha source | D1 | Resid ⊥ TGD; Net lift via lower TO (trading complementarity) |
| Combination enhancer | FlowDensity20 | Net/Gross lift in C; resid ⊥ cores still negative → not Core |
| Research satellite | D4 | Out of Composite v1 |
| Pending | D5 | Direction validation |

Portfolio engine v1:

```
TGD  → signal generation
D1   → signal stabilization
Flow → implementation improvement (optional)
```

Next: **2.1 Production Stress** (universe · cost · weight · calendar OOS) — not D4/D5 kitchen-sink.


