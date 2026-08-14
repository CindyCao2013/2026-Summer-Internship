# Alpha Report: `relative_liquidity_strength_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0161 | -1.5227 | 0.4601 | 0.0856 | 0.0183 | 0.1533 |
| CSI500 | -0.0355 | -2.8055 | 0.4388 | 0.5844 | 0.1427 | 0.1991 |
| CSI1000 | -0.0447 | -3.6375 | 0.4078 | 1.3140 | 0.3292 | 0.2190 |
| ALL | -0.0497 | -5.0366 | 0.3624 | 2.1598 | 0.4214 | 0.1754 |

## ALL-universe summary

- **factor_name**: relative_liquidity_strength_20d
- **universe**: ALL
- **rank_ic_mean**: -0.0497
- **abs_rank_ic_mean**: 0.0497
- **icir**: -5.0366
- **ic_positive_ratio**: 0.3624
- **hl_sharpe**: 2.1598
- **hl_annu_ret**: 0.4214
- **hl_mdd**: -0.2491
- **hl_avg_turnover**: 0.1754
- **direction**: -1
- **monotonicity_score**: 0.0000
- **stats_title**: H-L, Direction: -1, AnnuRet: 42.14%, Sharpe: 2.16, MDD: -24.91%, Turnover: 0.18, RankIC: -0.0497, ICIR: -5.04

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0441 |
| 5.0000 | -0.0719 |
| 10.0000 | -0.0892 |
| 20.0000 | -0.1124 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.001284 |
| 2 | 0.001163 |
| 3 | 0.000923 |
| 4 | 0.000808 |
| 5 | 0.000627 |
| 6 | 0.000526 |
| 7 | 0.000215 |
| 8 | 0.000051 |
| 9 | -0.000221 |
| 10 | -0.000402 |
| H-L | -0.001686 |

Monotonicity score: **0.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
