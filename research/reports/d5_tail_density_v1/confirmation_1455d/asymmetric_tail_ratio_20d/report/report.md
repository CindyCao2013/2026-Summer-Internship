# Alpha Report: `asymmetric_tail_ratio_20d`

- **Dimension**: 
- **Role**: 
- **Hypothesis**: 

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | 0.0165 | 1.3785 | 0.5495 | 0.0416 | 0.0097 | 0.8117 |
| CSI500 | 0.0260 | 2.5539 | 0.5571 | 0.4762 | 0.0948 | 0.8315 |
| CSI1000 | 0.0359 | 3.9405 | 0.6045 | 1.4730 | 0.2821 | 0.8457 |
| ALL | 0.0421 | 4.9211 | 0.6355 | 2.1569 | 0.4065 | 0.8543 |

## ALL-universe summary

- **factor_name**: asymmetric_tail_ratio_20d
- **universe**: ALL
- **rank_ic_mean**: 0.0421
- **abs_rank_ic_mean**: 0.0421
- **icir**: 4.9211
- **ic_positive_ratio**: 0.6355
- **hl_sharpe**: 2.1569
- **hl_annu_ret**: 0.4065
- **hl_mdd**: -0.2142
- **hl_avg_turnover**: 0.8543
- **direction**: 1
- **monotonicity_score**: 0.7000
- **stats_title**: H-L, Direction: 1, AnnuRet: 40.65%, Sharpe: 2.16, MDD: -21.42%, Turnover: 0.85, RankIC: 0.0421, ICIR: 4.92

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0336 |
| 5.0000 | 0.0446 |
| 10.0000 | 0.0513 |
| 20.0000 | 0.0598 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | -0.000989 |
| 2 | 0.000122 |
| 3 | 0.000466 |
| 4 | 0.000708 |
| 5 | 0.000789 |
| 6 | 0.000853 |
| 7 | 0.000883 |
| 8 | 0.000779 |
| 9 | 0.000733 |
| 10 | 0.000637 |
| H-L | 0.001626 |

Monotonicity score: **70.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
