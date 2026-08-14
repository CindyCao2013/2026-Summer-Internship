# Alpha Report: `max_daily_return_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0450 | 2.9963 | 0.5833 | 0.1109 | 0.0341 | 0.4282 |
| CSI500 | 0.0483 | 3.7627 | 0.6080 | 0.1801 | 0.0496 | 0.4157 |
| CSI1000 | 0.0577 | 4.5570 | 0.6383 | 1.0269 | 0.2719 | 0.4176 |
| ALL | 0.0586 | 5.1253 | 0.6515 | 1.3431 | 0.3482 | 0.4076 |

## ALL-universe summary

- **factor_name**: max_daily_return_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0586
- **abs_rank_ic_mean**: 0.0586
- **icir**: 5.1253
- **ic_positive_ratio**: 0.6515
- **hl_sharpe**: 1.3431
- **hl_annu_ret**: 0.3482
- **hl_mdd**: -0.2100
- **hl_avg_turnover**: 0.4076
- **direction**: 1
- **monotonicity_score**: 0.7000
- **stats_title**: H-L, Direction: 1, AnnuRet: 34.82%, Sharpe: 1.34, MDD: -21.00%, Turnover: 0.41, RankIC: 0.0586, ICIR: 5.13

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0535 |
| 5.0000 | 0.0716 |
| 10.0000 | 0.0798 |
| 20.0000 | 0.0902 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000551 |
| 2 | 0.000240 |
| 3 | 0.000827 |
| 4 | 0.001056 |
| 5 | 0.001129 |
| 6 | 0.001092 |
| 7 | 0.001135 |
| 8 | 0.001149 |
| 9 | 0.001109 |
| 10 | 0.000842 |
| H-L | 0.001393 |

Monotonicity score: **70.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
