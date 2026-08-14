# Knife Family Attribution

## Effectiveness by family


| Knife              | Family           | IC_spread | Separation | Effectiveness |
| ------------------ | ---------------- | --------- | ---------- | ------------- |
| `volume`           | participation    | -0.0443   | -0.0582    | 0.0499        |
| `amount`           | participation    | -0.0442   | -0.0579    | 0.0497        |
| `trade_count`      | trader_structure | -0.0385   | -0.0527    | 0.0442        |
| `ats_trade_count`  | trader_structure | -0.0372   | -0.0478    | 0.0415        |
| `ats_volume`       | trader_structure | -0.0300   | -0.0367    | 0.0327        |
| `turnover`         | participation    | -0.0300   | -0.0367    | 0.0327        |
| `amihud`           | liquidity        | -0.0239   | -0.0190    | 0.0219        |
| `volatility_state` | liquidity        | -0.0033   | 0.0011     | 0.0024        |


## Independence (vs closest peer)

| Knife | Peer | |corr| | resid_t | Independent? |
|-------|------|-------|---------|--------------|
| `volume` | `amount` | 0.92 | -10.78 | False |
| `amount` | `volume` | 0.92 | -11.52 | False |
| `trade_count` | `ats_trade_count` | 0.17 | -9.02 | True |
| `ats_trade_count` | `trade_count` | 0.17 | -10.76 | True |
| `ats_volume` | `turnover` | 1.00 | -1.06 | False |
| `turnover` | `ats_volume` | 1.00 | -1.06 | False |
| `amihud` | `ats_volume` | 0.03 | -5.51 | True |
| `volatility_state` | `ats_trade_count` | nan | -2.11 | True |

## Note

volume (participation) ≠ ATS (trader_structure). High corr → same story; low corr + residual IC → two alpha sources. Do not auto-replace paper knife.

## Correlation matrix

```
                  ats_trade_count  ats_volume  amount  volume  turnover  trade_count  amihud  volatility_state
ats_trade_count             1.000         NaN     NaN     NaN       NaN        0.173     NaN               NaN
ats_volume                    NaN       1.000     NaN     NaN     1.000          NaN   0.026               NaN
amount                        NaN         NaN    1.00    0.92       NaN          NaN     NaN               NaN
volume                        NaN         NaN    0.92    1.00       NaN          NaN     NaN               NaN
turnover                      NaN       1.000     NaN     NaN     1.000          NaN   0.026               NaN
trade_count                 0.173         NaN     NaN     NaN       NaN        1.000     NaN               NaN
amihud                        NaN       0.026     NaN     NaN     0.026          NaN   1.000               NaN
volatility_state              NaN         NaN     NaN     NaN       NaN          NaN     NaN               1.0
```

