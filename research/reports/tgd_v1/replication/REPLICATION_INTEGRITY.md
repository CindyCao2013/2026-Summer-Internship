# TGD Replication Integrity

Factor definition frozen. This checks **why** TGD works vs simpler primitives.

## Mechanism decomposition

| Factor | RankIC | ICIR | H-L Sharpe | Net@15bp | Mono |
|--------|--------|------|------------|----------|------|
| `epsilon_u` | 0.0107 | 1.62 | 2.58 | -8.61 | -0.600 |
| `epsilon_d` | 0.0371 | 5.80 | 1.05 | -7.36 | 0.745 |
| `tgd_eps` | 0.0474 | 8.53 | 4.33 | -4.65 | 0.927 |
| `epsilon_u_MA20` | 0.0370 | 3.79 | 0.82 | 0.02 | 0.952 |
| `epsilon_d_MA20` | 0.0458 | 4.74 | 1.39 | 0.75 | 0.915 |
| `TGD20` | 0.0430 | 6.98 | 2.77 | 1.58 | 0.988 |

Interpretation: if `TGD20` ≫ `epsilon_u` / `epsilon_d` alone, alpha comes from the εd⊥εu residual (late-selling / early-buying asymmetry), not raw timing centers.

## Primitive family (MA20-smoothed)

| Factor | RankIC | ICIR | H-L Sharpe | Net@15bp | Mono |
|--------|--------|------|------------|----------|------|
| `Gu_MA20` | 0.0447 | 4.25 | 0.90 | 0.41 | 0.733 |
| `Gd_MA20` | 0.0444 | 4.16 | 0.98 | 0.46 | 0.927 |
| `tau_MA20` | -0.0041 | -0.56 | 0.27 | -1.92 | 0.333 |
| `upsilon_MA20` | -0.0178 | -2.26 | 0.58 | -1.22 | -0.879 |
| `TGD20` | 0.0430 | 6.98 | 2.77 | 1.58 | 0.988 |

Interpretation: if `TGD20` beats `tau_MA20` / `upsilon_MA20`, residualization vs return-structure controls is the incremental alpha source (研报否定的简单 τ).

## Paper vs framework

| Item | Paper | This framework |
|------|-------|----------------|
| Data | 1-min returns | Stock_one_minute Close |
| Gu/Gd | √ | √ (`return_timing.py`) |
| Controls | Rū/Rd̄, R1, R2, overnight | √ conditional means |
| Residual | √ | √ CS OLS |
| TGD | εd~εu → MA20 | √ (`tgd.py`) |
| Groups | 5 | **10 + H-L** |
| Neutral | limited | size / industry / both |
| Cost | typically none | 15bp RT + ImpliedFee(7.5bps) |
| Shift | often implicit | **explicit shift-1** |

