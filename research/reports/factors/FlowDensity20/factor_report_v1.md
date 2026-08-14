# FlowDensity20 (net_active_flow_mktcap_20d) — Factor Research Pack

- **factor_id**: `FlowDensity20`
- **category**: `liquidity_flow_interaction`
- **status**: `validated_single_factor_candidate`
- **period**: `2022-01-28_2025-12-31`
- **universe**: `ALL`
- **next**: `orthogonality_vs_TGD20_raw_and_perp`
- **frozen_formula**: `False`

## Hypothesis

Net active flow / mktcap (20d) is a Flow × Liquidity interaction: direction entangled with anti-amount / low-activity. Not pure flow; complementary to TGD (temporal) but must be treated as microstructure+liquidity.

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
| FlowDensity20 | 2022-01-28_2025-12-31 | ALL | raw | 0.01783 | 0.2819 | 2.074 | 0.2338 | 1.515 | -0.1903 | 0.5148 | 0.09652 | -0.1777 |  | 1 |
| FlowDensity20 | 2022-01-28_2025-12-31 | ALL | size | 0.02714 | 0.4291 | 4.411 | 0.3152 | 2.947 | -0.1029 | 0.4744 | 0.08895 | 1.595 |  | 1 |
| FlowDensity20 | 2022-01-28_2025-12-31 | ALL | industry | 0.01516 | 0.2396 | 2.692 | 0.2234 | 1.904 | -0.1293 | 0.4802 | 0.09003 | 0.08408 |  | 1 |
| FlowDensity20 | 2022-01-28_2025-12-31 | ALL | size_industry | 0.02356 | 0.3725 | 4.849 | 0.2923 | 3.381 | -0.09512 | 0.4633 | 0.08688 | 1.85 |  | 1 |
| FlowDensity20 | 2022-01-28_2025-12-31 | ALL | execution_best | 0.02356 | 0.3725 | 4.847 | 0.277 | 3.715 | -0.09819 | 0.1645 | 0.03084 | 2.881 |  | 1 |

## Mechanism

| signal | family | rank_ic | icir | hl_sharpe | hl_annu_ret | hl_mdd | monotonicity | daily_turnover | net_sharpe | direction | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FlowDensity_raw | canonical | 0.02356 | 4.849 | 3.381 | 0.2923 | -0.09512 | 0.8788 | 0.4633 | 1.924 | 1 | size+ind net flow (confirmation signal) |
| Amount | liquidity_channel | -0.04749 | -8.658 | 3.946 | 0.4665 | -0.06895 | -1 | 0.1425 | 3.554 | -1 | size+ind amount/mktcap 20d |
| GrossActive | liquidity_channel | -0.04761 | -8.666 | 3.993 | 0.472 | -0.06922 | -1 | 0.1427 | 3.523 | -1 | size+ind gross active/mktcap 20d |
| Flow_perp_Amount | amount_orthogonal | -0.008764 | -2.489 | 0.1271 | 0.01136 | -0.1427 | 0.3697 | 0.5607 | -2.102 | -1 | ε from Flow_si ~ Amount_si (tradable residual panel) |
| Amount_perp_Flow | amount_orthogonal | -0.03714 | -8.489 | 2.525 | 0.2789 | -0.0989 | -0.8545 | 0.4459 | 0.6742 | -1 | ε from Amount_si ~ Flow_si (liquidity after removing flow) |
| Flow_perp_Amount_then_SI | amount_orthogonal_alt | -0.008533 | -2.422 | 0.07693 | 0.006851 | -0.1538 | 0.3818 | 0.5596 | -2.072 | -1 | ε from Flow~Amount in raw space, then size+ind |

## Yearly stability

| factor | year | n | rank_ic | icir |
| --- | --- | --- | --- | --- |
| FlowDensity20 | 2022 | 223 | 0.02058 | 4.534 |
| FlowDensity20 | 2023 | 242 | 0.02314 | 4.806 |
| FlowDensity20 | 2024 | 242 | 0.02343 | 4.207 |
| FlowDensity20 | 2025 | 243 | 0.02684 | 6.048 |

## Execution

| label | status | stage | rank_ic | icir | gross_sharpe | gross_annu_ret | net_sharpe | net_annu_ret | mdd_net | daily_turnover | annu_one_way_turnover | implied_annu_fee | direction | n_days | rebalance_freq | top_frac | entry_frac | exit_frac | min_hold | weight_method | round_trip_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| size_industry|daily|buffer_10_30 | optimized | E2_buffer | 0.02356 | 4.847 | 3.715 | 0.277 | 2.881 | 0.2153 | -0.09819 | 0.1645 | 20.56 | 0.03084 | 1 | 951 | 1 | 0.1 | 0.1 | 0.3 | 1 | ew | 0.0015 |
| size_industry|best_e1|buffer_10_30 | optimized | E2_buffer | 0.02107 | 4.47 | 3.383 | 0.2546 | 2.859 | 0.2164 | -0.1074 | 0.1018 | 12.72 | 0.01908 | 1 | 951 | 10 | 0.1 | 0.1 | 0.3 | 1 | ew | 0.0015 |
| size_industry|best_e1_buffer_10_20 | optimized | E4_combo | 0.02107 | 4.47 | 3.396 | 0.2654 | 2.762 | 0.2175 | -0.09334 | 0.1277 | 15.97 | 0.02395 | 1 | 951 | 10 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|best_e1_buffer_10_20_hold5 | optimized | E4_combo | 0.02107 | 4.47 | 3.396 | 0.2654 | 2.762 | 0.2175 | -0.09334 | 0.1277 | 15.97 | 0.02395 | 1 | 951 | 10 | 0.1 | 0.1 | 0.2 | 5 | ew | 0.0015 |
| size_industry|best_e1|buffer_10_20 | optimized | E2_buffer | 0.02107 | 4.47 | 3.396 | 0.2654 | 2.762 | 0.2175 | -0.09334 | 0.1277 | 15.97 | 0.02395 | 1 | 951 | 10 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|daily_buffer_10_20 | optimized | E4_combo | 0.02356 | 4.847 | 3.745 | 0.2937 | 2.689 | 0.2113 | -0.09373 | 0.2196 | 27.45 | 0.04118 | 1 | 951 | 1 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|daily|buffer_10_20 | optimized | E2_buffer | 0.02356 | 4.847 | 3.745 | 0.2937 | 2.689 | 0.2113 | -0.09373 | 0.2196 | 27.45 | 0.04118 | 1 | 951 | 1 | 0.1 | 0.1 | 0.2 | 1 | ew | 0.0015 |
| size_industry|best_e1|hold_10d | optimized | E2_hold | 0.02107 | 4.47 | 3.531 | 0.2927 | 2.686 | 0.2258 | -0.09189 | 0.1786 | 22.32 | 0.03348 | 1 | 951 | 10 | 0.1 |  |  | 10 | ew | 0.0015 |
| size_industry|best_e1|hold_5d | optimized | E2_hold | 0.02107 | 4.47 | 3.531 | 0.2927 | 2.686 | 0.2258 | -0.09189 | 0.1786 | 22.32 | 0.03348 | 1 | 951 | 10 | 0.1 |  |  | 5 | ew | 0.0015 |
| size_industry|best_e1_plain | optimized | E4_combo | 0.02107 | 4.47 | 3.531 | 0.2927 | 2.686 | 0.2258 | -0.09189 | 0.1786 | 22.32 | 0.03348 | 1 | 951 | 10 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|best_e1|hold_1d | optimized | E2_hold | 0.02107 | 4.47 | 3.531 | 0.2927 | 2.686 | 0.2258 | -0.09189 | 0.1786 | 22.32 | 0.03348 | 1 | 951 | 10 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|every_10d | optimized | E1 | 0.02107 | 4.47 | 3.531 | 0.2927 | 2.686 | 0.2258 | -0.09189 | 0.1786 | 22.32 | 0.03348 | 1 | 951 | 10 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|best_e1|ew | optimized | E3 | 0.02107 | 4.47 | 3.531 | 0.2927 | 2.686 | 0.2258 | -0.09189 | 0.1786 | 22.32 | 0.03348 | 1 | 951 | 10 | 0.1 |  |  | 1 | ew | 0.0015 |
| size_industry|best_e1|rank | optimized | E3 | 0.02107 | 4.47 | 3.522 | 0.2935 | 2.678 | 0.2263 | -0.09125 | 0.1793 | 22.41 | 0.03362 | 1 | 951 | 10 | 0.1 |  |  | 1 | rank | 0.0015 |
| size_industry|best_e1|zscore | optimized | E3 | 0.02107 | 4.47 | 3.442 | 0.3313 | 2.651 | 0.2582 | -0.08785 | 0.195 | 24.37 | 0.03656 | 1 | 951 | 10 | 0.1 |  |  | 1 | zscore | 0.0015 |

## Do not

- jump to TGD×Flow composite before orthogonality
- treat CSI300 standalone as validation target (known concentration)
- auto-freeze from amount-orth attribution
- rename to pure Flow without documenting liquidity channel

## Notes

- Assembled from l2_flow_density_v1 confirmation + validation neut ladder.
- Soft flag: broad-universe / small-cap concentrated.
- Verdict: confirm_pass_enhancer.
- Mechanism: mechanism_entangled_with_anti_amount.
- Flow⊥Amount ICIR=-2.49 (signed retain=-0.51) — neither pure flow nor pure amount. Classify as Flow × Liquidity interaction.
- mechanism_class=flow_liquidity_interaction

## Open gaps

- long-form Chinese research essay not yet written
- figures/ not yet standardized
- orthogonality vs TGD20 not yet run (raw + Flow_perp_Amount)

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
