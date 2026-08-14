# mid_order_ratio_normalized — Single Factor Research Report

This report tests one question only:

> Does the predictive relation of the fixed-RMB trade-amount bucket survive
> after each Tick execution amount is normalized by the stock's own trading
> scale?

It is a single-factor construction and validation study.  It does not perform
factor-library correlation analysis, factor combination, incremental IC,
portfolio optimization, alpha stacking, or full-sample Sharpe optimization.

## Factor variants

- **A0 — `mid_trade_amount_share_abs_4w20w`**: fixed-RMB benchmark,
  `40,000 < trade amount <= 200,000`; legacy alias `mid_order_ratio`.
- **A1 — `mid_trade_amount_share_adv20`**: Tick amount divided by the
  strictly lagged 20-trading-day mean daily traded amount.
- **A2 — `mid_trade_amount_share_ats20`**: Tick amount divided by the
  strictly lagged median of 20 daily median Tick trade amounts.
- **A3 — `mid_trade_amount_share_rollq`**: amount share between the current
  stock-day Q20 and Q80 Tick-amount quantiles; secondary P1 diagnostic.

All variants retain an amount numerator and an amount denominator.  The Tick
rows are executions, not reconstructed parent orders.

## Report navigation

1. [Executive Summary](01_executive_summary.md)
2. [Problem Definition](02_problem_definition.md)
3. [Data and Strict Trade Construction](03_data_and_strict_trade_construction.md)
4. [Normalized Factor Definitions](04_normalized_factor_definitions.md)
5. [Standalone Validation](05_standalone_validation.md)
6. [Normalization Diagnostics](06_normalization_diagnostics.md)
7. [Exposure Diagnostics](07_exposure_diagnostics.md)
8. [Time and State Robustness](08_time_and_state_robustness.md)
9. [In-sample / Out-of-sample](09_in_sample_out_of_sample.md)
10. [Research Decision](10_research_decision.md)
11. [Code Evidence Map](appendix/code_evidence_map.md)
12. [Reproduction Commands](appendix/reproduction_commands.md)

Machine-readable outputs are under `artifacts/`; figures are under `figures/`;
HTML and PDF exports are under `export/`.

