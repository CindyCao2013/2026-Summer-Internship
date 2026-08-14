# Knife Family Incremental IC

**Period:** `2018-01-02 → 2025-12-31` · source=`ddb`  
**Object:** daily ret (W-cut window=20) · **Knives:** `amount`, `volume`, `turnover_proxy`, `ats_trade_count`

## Key questions

1. **volume | amount:** resid_IC=−0.0220 · t=−16.07 · |corr|=**0.92** → residual IC statistically significant, but **same-family** (see synth: amount+volume does **not** beat singles).
2. **ATS | amount:** resid_IC=−0.0220 · t=−14.85 · |corr|=**0.37** → **YES incremental** after controlling amount.  
   ATS | volume: resid_IC=−0.0225 · t=−14.93 · |corr|=0.35.
3. **Dual-knife synth:** best_single=`ats_trade_count` ICIR=−6.48; best_dual=`amount+ats_trade_count:residual_add` ICIR=**−6.88** → **DUAL WINS** (cross-family).

## Effectiveness ranking

| Knife | Family | IC_spread | Separation | Effectiveness |
|-------|--------|-----------|------------|---------------|
| `volume` | participation | −0.0366 | −0.0480 | 0.0411 |
| `amount` | participation | −0.0363 | −0.0476 | 0.0409 |
| `turnover_proxy` | participation | −0.0356 | −0.0463 | 0.0399 |
| `ats_trade_count` | trader_structure | −0.0338 | −0.0451 | 0.0383 |

## Independence (vs closest peer)

| Knife | Peer | \|corr\| | resid_t | Independent? |
|-------|------|-------|---------|--------------|
| `volume` | `amount` | 0.92 | −16.07 | False |
| `amount` | `volume` | 0.92 | −16.99 | False |
| `turnover_proxy` | `amount` | 0.93 | −15.30 | False |
| `ats_trade_count` | `amount` | **0.37** | −14.85 | **True** |

## Cut-factor correlation (corrected; min_overlap=100)

```
                 amount  volume  turnover_proxy  ats_trade_count
amount            1.000   0.918           0.929            0.367
volume            0.918   1.000           0.949            0.354
turnover_proxy    0.929   0.949           1.000            0.382
ats_trade_count   0.367   0.354           0.382            1.000
```

## Residual IC matrix (row residualized vs column)

```
vs_knife         amount  ats_trade_count  turnover_proxy  volume
amount              NaN          -0.0257         -0.0207 -0.0205
ats_trade_count -0.0220              NaN         -0.0232 -0.0225
turnover_proxy  -0.0197          -0.0251             NaN -0.0272
volume          -0.0220          -0.0264         -0.0296     NaN
```

## Residual t-stat matrix

```
vs_knife         amount  ats_trade_count  turnover_proxy  volume
amount              NaN           -13.79          -16.44  -16.99
ats_trade_count  -14.85              NaN          -15.04  -14.93
turnover_proxy   -15.30           -13.19             NaN  -16.49
volume           -16.07           -13.60          -16.68     NaN
```

## Dual-knife synthesis

| Label | Kind | RankIC | ICIR | n_days |
|-------|------|--------|------|--------|
| `amount+ats_trade_count:residual_add` | dual | **−0.0408** | **−6.88** | 1703 |
| `volume+ats_trade_count:residual_add` | dual | −0.0410 | −6.88 | 1703 |
| `amount+ats_trade_count:equal_z` | dual | −0.0417 | −6.83 | 1703 |
| `volume+ats_trade_count:equal_z` | dual | −0.0419 | −6.82 | 1703 |
| `ats_trade_count` | single | −0.0338 | −6.48 | 1703 |
| `amount` | single | −0.0363 | −6.34 | 1921 |
| `amount+volume:equal_z` | dual | −0.0371 | −6.29 | 1921 |
| `amount+volume:residual_add` | dual | −0.0367 | −6.25 | 1921 |
| `volume` | single | −0.0366 | −6.25 | 1921 |
| `turnover_proxy` | single | −0.0356 | −6.20 | 1921 |

## Verdict

| Question | Answer |
|----------|--------|
| volume residual after amount? | Statistically yes, but **no synth uplift** (same participation story) |
| ATS residual after amount? | **Yes** — corr≈0.37, resid t≈−15 |
| Multi-knife value? | **Yes** — `amount+ATS` / `volume+ATS` residual_add beats any single knife |

Cross-family synthesis (participation + trader_structure) is valuable. Within-family (`amount+volume`) is not.

Do **not** replace paper ATS with volume solely because raw effectiveness is slightly higher.

Artifacts: `pairwise_residual_ic.csv`, `dual_knife_synth.csv`, `search_knives.json`, `verdict.json`
