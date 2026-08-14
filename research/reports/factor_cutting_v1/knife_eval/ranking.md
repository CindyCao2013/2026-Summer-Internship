# Knife Evaluator Ranking

Score = 0.6·|IC_spread| + 0.4·|IC_high − IC_low|

| Rank | Knife | IC_high | IC_low | IC_spread | Sep | Effectiveness |
|------|-------|---------|--------|-----------|-----|---------------|
| 1 | `volume` | -0.0426 | 0.0054 | -0.0366 | -0.0480 | 0.0411 |
| 2 | `amount` | -0.0428 | 0.0048 | -0.0363 | -0.0476 | 0.0409 |
| 3 | `ats_trade_count` | -0.0430 | 0.0020 | -0.0338 | -0.0451 | 0.0383 |
| 4 | `ats_volume` | -0.0447 | -0.0136 | -0.0249 | -0.0311 | 0.0273 |
| 5 | `turnover` | -0.0447 | -0.0136 | -0.0249 | -0.0311 | 0.0273 |

Candidates considered: ats_trade_count, ats_volume, amount, volume, turnover
