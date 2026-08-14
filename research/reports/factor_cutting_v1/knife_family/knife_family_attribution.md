# Knife Family Attribution

## Effectiveness by family

| Knife | Family | IC_spread | Separation | Effectiveness |
|-------|--------|-----------|------------|---------------|
| `volume` | participation | -0.0366 | -0.0480 | 0.0411 |
| `amount` | participation | -0.0363 | -0.0476 | 0.0409 |
| `turnover_proxy` | participation | -0.0356 | -0.0463 | 0.0399 |
| `ats_trade_count` | trader_structure | -0.0338 | -0.0451 | 0.0383 |

## Independence (vs closest peer)

| Knife | Peer | |corr| | resid_t | Independent? |
|-------|------|-------|---------|--------------|
| `volume` | `amount` | 0.92 | -16.07 | False |
| `amount` | `volume` | 0.92 | -16.99 | False |
| `turnover_proxy` | `amount` | nan | -15.30 | True |
| `ats_trade_count` | `amount` | nan | -14.85 | True |

## Note

volume (participation) ≠ ATS (trader_structure). High corr → same story; low corr + residual IC → two alpha sources. Do not auto-replace paper knife.

## Correlation matrix

```
                 amount  volume  turnover_proxy  ats_trade_count
amount            1.000   0.918             NaN              NaN
volume            0.918   1.000             NaN              NaN
turnover_proxy      NaN     NaN             1.0              NaN
ats_trade_count     NaN     NaN             NaN              1.0
```
