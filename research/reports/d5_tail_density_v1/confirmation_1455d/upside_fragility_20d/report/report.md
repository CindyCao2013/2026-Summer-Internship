# Alpha Report: `upside_fragility_20d`

- **Dimension**: D5
- **Role**: base
- **Hypothesis**: Upside tail fragility — crash-prone winners

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0378 | -3.5018 | 0.4017 | 0.3531 | 0.0770 | 1.4197 |
| CSI500 | -0.0444 | -4.8607 | 0.3666 | 0.7990 | 0.1543 | 1.4395 |
| CSI1000 | -0.0484 | -5.6140 | 0.3343 | 1.7745 | 0.3227 | 1.4426 |
| ALL | -0.0552 | -6.8791 | 0.2930 | 2.7378 | 0.4851 | 1.4188 |

## ALL-universe summary

- **factor_name**: upside_fragility_20d
- **universe**: ALL
- **rank_ic_mean**: -0.0552
- **abs_rank_ic_mean**: 0.0552
- **icir**: -6.8791
- **ic_positive_ratio**: 0.2930
- **hl_sharpe**: 2.7378
- **hl_annu_ret**: 0.4851
- **hl_mdd**: -0.2540
- **hl_avg_turnover**: 1.4188
- **direction**: -1
- **monotonicity_score**: 0.2000
- **stats_title**: H-L, Direction: -1, AnnuRet: 48.51%, Sharpe: 2.74, MDD: -25.40%, Turnover: 1.42, RankIC: -0.0552, ICIR: -6.88

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | -0.0436 |
| 5.0000 | -0.0574 |
| 10.0000 | -0.0598 |
| 20.0000 | -0.0642 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000735 |
| 2 | 0.000708 |
| 3 | 0.000789 |
| 4 | 0.000783 |
| 5 | 0.000863 |
| 6 | 0.000772 |
| 7 | 0.000693 |
| 8 | 0.000605 |
| 9 | 0.000238 |
| 10 | -0.001205 |
| H-L | -0.001940 |

Monotonicity score: **20.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
