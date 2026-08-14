# Alpha Report: `drawup_drawdown_ratio_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0164 | -1.4444 | 0.4594 | 0.0174 | 0.0038 | 0.8251 |
| CSI500 | -0.0240 | -2.4612 | 0.4457 | 0.4437 | 0.0857 | 0.8508 |
| CSI1000 | -0.0334 | -3.7542 | 0.3893 | 1.1915 | 0.2260 | 0.8664 |
| ALL | -0.0384 | -4.6263 | 0.3680 | 1.9865 | 0.3687 | 0.8722 |

## ALL-universe summary

- **factor_name**: drawup_drawdown_ratio_20d
- **universe**: ALL
- **rank_ic_mean**: -0.0384
- **abs_rank_ic_mean**: 0.0384
- **icir**: -4.6263
- **ic_positive_ratio**: 0.3680
- **hl_sharpe**: 1.9865
- **hl_annu_ret**: 0.3687
- **hl_mdd**: -0.2301
- **hl_avg_turnover**: 0.8722
- **direction**: -1
- **monotonicity_score**: 0.3000
- **stats_title**: H-L, Direction: -1, AnnuRet: 36.87%, Sharpe: 1.99, MDD: -23.01%, Turnover: 0.87, RankIC: -0.0384, ICIR: -4.63

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0309 |
| 5.0000 | -0.0419 |
| 10.0000 | -0.0478 |
| 20.0000 | -0.0555 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000578 |
| 2 | 0.000716 |
| 3 | 0.000795 |
| 4 | 0.000785 |
| 5 | 0.000881 |
| 6 | 0.000721 |
| 7 | 0.000674 |
| 8 | 0.000533 |
| 9 | 0.000195 |
| 10 | -0.000897 |
| H-L | -0.001475 |

Monotonicity score: **30.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
