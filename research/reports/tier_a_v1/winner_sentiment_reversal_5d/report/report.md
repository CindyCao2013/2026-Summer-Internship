# Alpha Report: `winner_sentiment_reversal_5d`

- **Dimension**: D4
- **Role**: base
- **Hypothesis**: Short-horizon winner exhaustion / sentiment reversal

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0286 | 2.4409 | 0.5564 | 0.5461 | 0.1245 | 1.4987 |
| CSI500 | 0.0364 | 3.5258 | 0.5791 | 1.0468 | 0.2162 | 1.4966 |
| CSI1000 | 0.0463 | 4.9602 | 0.6341 | 2.5108 | 0.4956 | 1.5323 |
| ALL | 0.0519 | 5.9584 | 0.6582 | 3.2278 | 0.6239 | 1.5322 |

## ALL-universe summary

- **factor_name**: winner_sentiment_reversal_5d
- **universe**: ALL
- **rank_ic_mean**: 0.0519
- **abs_rank_ic_mean**: 0.0519
- **icir**: 5.9584
- **ic_positive_ratio**: 0.6582
- **hl_sharpe**: 3.2278
- **hl_annu_ret**: 0.6239
- **hl_mdd**: -0.1891
- **hl_avg_turnover**: 1.5322
- **direction**: 1
- **monotonicity_score**: 0.7000
- **stats_title**: H-L, Direction: 1, AnnuRet: 62.39%, Sharpe: 3.23, MDD: -18.91%, Turnover: 1.53, RankIC: 0.0519, ICIR: 5.96

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0370 |
| 5.0000 | 0.0447 |
| 10.0000 | 0.0432 |
| 20.0000 | 0.0468 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.001867 |
| 2 | 0.000362 |
| 3 | 0.000688 |
| 4 | 0.000838 |
| 5 | 0.000922 |
| 6 | 0.000971 |
| 7 | 0.000904 |
| 8 | 0.000740 |
| 9 | 0.000742 |
| 10 | 0.000629 |
| H-L | 0.002496 |

Monotonicity score: **70.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
