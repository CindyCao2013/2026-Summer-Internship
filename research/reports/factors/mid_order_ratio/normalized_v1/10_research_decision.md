# 10 — Research Decision

## Final branch

**`only the fixed-RMB phenomenon remains`**

## Frozen decision evidence

| Role | OOS raw RankIC | OOS t-stat | Frozen direction | Significant | Pools same direction | Retained |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | -3.84% | -5.67 | yes | yes | 4/4 | yes |
| A1 | 3.15% | 3.22 | no | no | 0/4 | no |
| A2 | -0.09% | -0.12 | yes | no | 3/4 | no |
| A3 | -3.13% | -4.18 | yes | yes | 4/4 | yes |

A role is retained only if all three primary gates pass:

1. CSI1000 OOS raw RankIC has frozen direction `-1`;
2. its direction-adjusted raw RankIC t-stat is at least `1.96`;
3. raw RankIC has the frozen direction in ALL, CSI300, CSI500 and CSI1000.

The four-pool evidence is the persisted full-common-sample universe table;
the persisted segment table supplies CSI1000 OOS significance. No
segment-by-pool result is invented.

The branch order is fixed: both A1/A2, A1 only, A2 only, A0-only fallback,
then no retained normalized relation. A3 remains a secondary P1 diagnostic
and cannot select a branch.

## Parameter-stability evidence

| Role | Candidates same direction | Share | Selected RankIC | Selected t-stat | Selected ICIR |
| --- | --- | --- | --- | --- | --- |
| A0 | 1/1 | 100.00% | -4.24% | -8.91 | -4.79 |
| A1 | 1/10 | 10.00% | 2.70% | 4.13 | 2.22 |
| A2 | 8/10 | 80.00% | -0.58% | -1.01 | -0.54 |
| A3 | 1/1 | 100.00% | -3.60% | -6.36 | -3.42 |

Parameter stability is supporting evidence after parameters were frozen. It
cannot overturn failed OOS direction/significance gates and cannot select the
highest Sharpe.

## Cost and interpretation limits

- the cost label is one-way **7.5 bps**, display-only;
- H-L is a gross standalone sorting diagnostic, not a portfolio strategy;
- Tick execution amount does not identify investor type;
- no causal claim follows from OLS residualization.

## Explicitly outside scope

This report does **not** perform factor-library correlation analysis, factor
combination, incremental-IC testing, portfolio optimization, alpha stacking,
or return-based parameter optimization.
