# Alpha Report: `quality_composite`

- **Dimension**: D7
- **Role**: enhancer
- **Hypothesis**: Equal-z(roe_stability, GP/A, CFO/NI)

## IC Statistics (by universe)

| universe | rank_ic_mean | icir | ic_positive_ratio | hl_sharpe | hl_annu_ret | hl_avg_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| CSI300 | -0.0049 | -0.4676 | 0.4938 | 0.0691 | 0.0126 | 0.0929 |
| CSI500 | -0.0049 | -0.5479 | 0.5041 | 0.0939 | 0.0149 | 0.1117 |
| CSI1000 | -0.0007 | -0.0986 | 0.5076 | 0.0932 | 0.0128 | 0.1173 |
| ALL | -0.0002 | -0.0384 | 0.5062 | 0.0962 | 0.0121 | 0.1150 |

## ALL-universe summary

- **factor_name**: quality_composite
- **universe**: ALL
- **rank_ic_mean**: -0.0002
- **abs_rank_ic_mean**: 0.0002
- **icir**: -0.0384
- **ic_positive_ratio**: 0.5062
- **hl_sharpe**: 0.0962
- **hl_annu_ret**: 0.0121
- **hl_mdd**: -0.2831
- **hl_avg_turnover**: 0.1150
- **direction**: -1
- **monotonicity_score**: 0.4000
- **stats_title**: H-L, Direction: -1, AnnuRet: 1.21%, Sharpe: 0.10, MDD: -28.31%, Turnover: 0.11, RankIC: -0.0002, ICIR: -0.04

## IC decay

| horizon_days | rank_ic |
| --- | --- |
| 1.0000 | 0.0004 |
| 5.0000 | -0.0019 |
| 10.0000 | -0.0046 |
| 20.0000 | -0.0097 |

## Decile returns (ALL)

| index | mean_daily_return |
| --- | --- |
| 1 | 0.000379 |
| 2 | 0.000515 |
| 3 | 0.000490 |
| 4 | 0.000578 |
| 5 | 0.000627 |
| 6 | 0.000575 |
| 7 | 0.000644 |
| 8 | 0.000518 |
| 9 | 0.000435 |
| 10 | 0.000330 |
| H-L | -0.000049 |

Monotonicity score: **40.00%**

## Figures

- `quantile_return.png`
- `cumulative_long_short.png`
- `ic_timeseries.png`
- `ic_decay.png`
