# TGD20 — Factor Research Pack

- **factor_id**: `TGD20`
- **category**: `temporal_information`
- **status**: `validated_single_factor`
- **period**: `2022-01-28_2025-12-31`
- **universe**: `ALL`
- **next**: `factor_combination_candidate`
- **frozen_formula**: `True`

## Hypothesis

Abnormal downside timing residual (εd ⊥ εu, MA20) predicts next-day cross-sectional returns. Alpha is temporal residual information, not raw Gu/Gd or τ=Gd−Gu.

## Checklist

| Stage | Status |
|-------|--------|
| Factor construction | ✅ |
| Mechanism verification | ✅ |
| Metrics schema | ✅ |
| IC / ICIR / Sharpe / MDD | ✅ |
| Neutralization ladder | ✅ |
| Yearly stability | ✅ |
| Execution optimization | ✅ |
| Research report (essay) | 🟡 optional long-form |

## Factor summary

| factor | period | universe | mode | rank_ic | annu_ic | icir | hl_annu_ret | hl_sharpe | hl_mdd | daily_turnover | implied_annu_fee | net_sharpe | monotonicity | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TGD20 | 2022-01-28_2025-12-31 | ALL | raw | 0.04301 | 0.68 | 6.98 | 0.3687 | 2.769 | -0.1899 | 0.6471 | 0.1213 | 0.9959 | 0.9879 | 1 |
| TGD20 | 2022-01-28_2025-12-31 | ALL | size | 0.04433 | 0.701 | 8.669 | 0.3675 | 3.518 | -0.07617 | 0.6518 | 0.1222 | 1.507 | 1 | 1 |
| TGD20 | 2022-01-28_2025-12-31 | ALL | industry | 0.04075 | 0.6443 | 8.896 | 0.3499 | 3.189 | -0.1587 | 0.6437 | 0.1207 | 1.163 | 0.9879 | 1 |
| TGD20 | 2022-01-28_2025-12-31 | ALL | size_industry | 0.04152 | 0.6566 | 11.29 | 0.3527 | 4.056 | -0.0604 | 0.6458 | 0.1211 | 1.719 | 0.9879 | 1 |
| TGD20 | 2022-01-28_2025-12-31 | ALL | execution_best | 0.04152 | 0.6566 | 11.28 | 0.3295 | 3.506 | -0.08098 | 0.2965 | 0.0556 | 2.324 |  | 1 |

## Mechanism

| signal | category | rank_ic | icir | hl_sharpe | net_sharpe | monotonicity | daily_turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gu_MA20 | primitive_family | 0.04469 | 4.25 | 0.9008 | 0.4057 | 0.7333 | 0.3587 |
| Gd_MA20 | primitive_family | 0.04437 | 4.162 | 0.9788 | 0.4613 | 0.9273 | 0.3192 |
| tau_MA20 | primitive_family | -0.004065 | -0.5556 | 0.2737 | -1.92 | 0.3333 | 0.808 |
| upsilon_MA20 | primitive_family | -0.01784 | -2.261 | 0.5773 | -1.221 | -0.8788 | 0.5656 |
| TGD20 | primitive_family | 0.04301 | 6.98 | 2.769 | 1.583 | 0.9879 | 0.6471 |
| epsilon_u | mechanism_residual | 0.01073 | 1.621 | 2.581 | -8.608 | -0.6 | 3.128 |
| epsilon_d | mechanism_residual | 0.03708 | 5.799 | 1.054 | -7.363 | 0.7455 | 3.12 |
| tgd_eps | mechanism_residual | 0.04741 | 8.53 | 4.33 | -4.654 | 0.9273 | 3.445 |
| epsilon_u_MA20 | mechanism_residual | 0.037 | 3.785 | 0.8197 | 0.01513 | 0.9515 | 0.3466 |
| epsilon_d_MA20 | mechanism_residual | 0.04582 | 4.738 | 1.385 | 0.7504 | 0.9152 | 0.3373 |

## Yearly stability

| period | kind | n_days | rank_ic | icir | pos_ic_frac |
| --- | --- | --- | --- | --- | --- |
| 2020 | year | 242 | 0.03577 | 8.011 | 0.6983 |
| 2021 | year | 243 | 0.03602 | 6.688 | 0.6543 |
| 2022 | year | 242 | 0.04687 | 9.119 | 0.7273 |
| 2023 | year | 242 | 0.05209 | 8.95 | 0.7314 |
| 2024 | year | 242 | 0.03165 | 4.253 | 0.6405 |
| 2025 | year | 243 | 0.04288 | 7.058 | 0.6831 |
| 2020-2021 | block | 485 | 0.0359 | 7.262 | 0.6763 |
| 2022-2023 | block | 484 | 0.04948 | 9.017 | 0.7293 |
| 2024-2025 | block | 485 | 0.03727 | 5.487 | 0.6619 |

## Execution

| label | stage | rank_ic | icir | gross_sharpe | gross_annu_ret | net_sharpe | net_annu_ret | mdd_net | daily_turnover | annu_one_way_turnover | implied_annu_fee | direction | n_days | rebalance_freq | top_frac | entry_frac | exit_frac | min_hold | weight_method | round_trip_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| size_industry|daily|buffer_5_15 | E2_buffer | 0.04152 | 11.28 | 3.506 | 0.3295 | 2.324 | 0.2183 | -0.08098 | 0.2965 | 37.07 | 0.0556 | 1 | 951 | 1 | 0.1 | 0.05 | 0.15 | 1 | ew | 0.0015 |
| size_industry|best_e1|zscore | E3 | 0.03393 | 9.278 | 3.5 | 0.3451 | 2.244 | 0.2237 | -0.1009 | 0.3237 | 40.46 | 0.06069 | 1 | 951 | 5 | 0.1 |  |  | 1 | zscore | 0.0015 |
| size_industry|daily|buffer_10_30 | E2_buffer | 0.04152 | 11.28 | 3.356 | 0.2376 | 2.207 | 0.1562 | -0.07454 | 0.2172 | 27.15 | 0.04073 | 1 | 951 | 1 | 0.1 | 0.1 | 0.3 | 1 | ew | 0.0015 |
| size_industry|best_e1|hold_10d | E2_hold | 0.03393 | 9.278 | 3.407 | 0.2766 | 2.197 | 0.1801 | -0.09144 | 0.2574 | 32.18 | 0.04827 | 1 | 951 | 5 | 0.1 |  |  | 10 | ew | 0.0015 |
| size_industry|best_e1|buffer_5_15 | E2_buffer | 0.03393 | 9.278 | 3.039 | 0.2986 | 2.197 | 0.2174 | -0.1029 | 0.2164 | 27.05 | 0.04057 | 1 | 951 | 5 | 0.1 | 0.05 | 0.15 | 1 | ew | 0.0015 |
| size_industry|best_e1|buffer_10_30 | E2_buffer | 0.03393 | 9.278 | 2.94 | 0.2173 | 2.094 | 0.1555 | -0.08916 | 0.1648 | 20.6 | 0.0309 | 1 | 951 | 5 | 0.1 | 0.1 | 0.3 | 1 | ew | 0.0015 |
| size_industry|daily_buffer_10_20 | E4_combo | 0.04152 | 11.28 | 3.548 | 0.2718 | 2.092 | 0.16 | -0.07554 | 0.298 | 37.25 | 0.05588 | 1 | 951 | 1 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|daily|buffer_10_20 | E2_buffer | 0.04152 | 11.28 | 3.548 | 0.2718 | 2.092 | 0.16 | -0.07554 | 0.298 | 37.25 | 0.05588 | 1 | 951 | 1 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|best_e1|rank | E3 | 0.03393 | 9.278 | 3.437 | 0.2994 | 2.075 | 0.1829 | -0.0951 | 0.3107 | 38.84 | 0.05826 | 1 | 951 | 5 | 0.1 |  |  | 1 | rank | 0.0015 |
| size_industry|best_e1_buffer_10_20_hold5 | E4_combo | 0.03393 | 9.278 | 3.09 | 0.2418 | 2.072 | 0.1634 | -0.09154 | 0.2092 | 26.15 | 0.03922 | 1 | 951 | 5 | 0.1 | 0.1 | 0.2 | 5 | ew | 0.0015 |
| size_industry|best_e1_buffer_10_20 | E4_combo | 0.03393 | 9.278 | 3.09 | 0.2418 | 2.072 | 0.1634 | -0.09154 | 0.2092 | 26.15 | 0.03922 | 1 | 951 | 5 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|best_e1|buffer_10_20 | E2_buffer | 0.03393 | 9.278 | 3.09 | 0.2418 | 2.072 | 0.1634 | -0.09154 | 0.2092 | 26.15 | 0.03922 | 1 | 951 | 5 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|best_e1_plain | E4_combo | 0.03393 | 9.278 | 3.431 | 0.2963 | 2.061 | 0.1801 | -0.09447 | 0.31 | 38.75 | 0.05812 | 1 | 951 | 5 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|best_e1|ew | E3 | 0.03393 | 9.278 | 3.431 | 0.2963 | 2.061 | 0.1801 | -0.09447 | 0.31 | 38.75 | 0.05812 | 1 | 951 | 5 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|best_e1|hold_1d | E2_hold | 0.03393 | 9.278 | 3.431 | 0.2963 | 2.061 | 0.1801 | -0.09447 | 0.31 | 38.75 | 0.05812 | 1 | 951 | 5 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|best_e1|hold_5d | E2_hold | 0.03393 | 9.278 | 3.431 | 0.2963 | 2.061 | 0.1801 | -0.09447 | 0.31 | 38.75 | 0.05812 | 1 | 951 | 5 | 0.1 |  |  | 5 | ew | 0.0015 |
| size_industry|every_5d | E1 | 0.03393 | 9.278 | 3.431 | 0.2963 | 2.061 | 0.1801 | -0.09447 | 0.31 | 38.75 | 0.05812 | 1 | 951 | 5 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|every_10d | E1 | 0.03038 | 8.24 | 2.952 | 0.2551 | 1.965 | 0.1705 | -0.09294 | 0.2258 | 28.22 | 0.04233 | 1 | 951 | 10 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|weekly_friday | E1 | 0.03343 | 9.353 | 3.32 | 0.2801 | 1.884 | 0.1647 | -0.08551 | 0.3078 | 38.47 | 0.05771 | 1 | 951 | friday | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|daily|hold_10d | E2_hold | 0.04152 | 11.28 | 3.496 | 0.2543 | 1.879 | 0.1365 | -0.07484 | 0.3141 | 39.26 | 0.05889 | 1 | 951 | 1 | 0.1 |  |  | 10 | ew | 0.0015 |
| size_industry|daily|hold_5d | E2_hold | 0.04152 | 11.28 | 3.63 | 0.29 | 1.559 | 0.1244 | -0.0815 | 0.4415 | 55.19 | 0.08279 | 1 | 951 | 1 | 0.1 |  |  | 5 | ew | 0.0015 |
| size_industry|every_20d | E1 | 0.02467 | 6.814 | 2.192 | 0.1825 | 1.459 | 0.1224 | -0.1079 | 0.1601 | 20.02 | 0.03002 | 1 | 951 | 20 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|daily|hold_1d | E2_hold | 0.04152 | 11.28 | 4.064 | 0.3534 | 1.28 | 0.1112 | -0.08415 | 0.6459 | 80.74 | 0.1211 | 1 | 951 | 1 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|daily | E1 | 0.04152 | 11.28 | 4.064 | 0.3534 | 1.28 | 0.1112 | -0.08415 | 0.6459 | 80.74 | 0.1211 | 1 | 951 | 1 | 0.1 |  |  | 1 | ew | 0.0015 |

## Do not

- retune MA window (MA10/30/60)
- change residual controls
- replace residual with ML
- mine a new TGD variant under this id

## Notes

- Template instance for Factor Report Generator v1.
- Long-form Chinese report remains under research/reports/tgd_v1/.

## Artifacts

```
factor_report.md
metrics.json
factor_summary.csv
mechanism.csv / mechanism_analysis.csv
stability.csv / yearly_stability.csv
execution_summary.csv
artifacts/
figures/
```
