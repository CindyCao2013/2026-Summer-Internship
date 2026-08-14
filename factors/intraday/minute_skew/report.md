# Minute Skew — Discovery Report

## Status

`candidate_ddb`: implemented in Python and DolphinDB and registered in the
intraday backend registry. It is not part of the default production factor
list.

## Gates

- Formula and timestamp contract: implemented.
- Explicit future-bar mutation test: implemented.
- Numeric tolerance: maximum absolute difference must be `<= 1e-10`.
- Rank tolerance: minimum cross-sectional Spearman must be `>= 0.999`.
- Full backtest parity: required before promotion.
- Holding-horizon and cost analysis: required before selection.

## Interpretation limits

Minute skew is a distribution-shape candidate, not automatically a standalone
alpha. It must be tested against realized volatility, extreme-return exposure
and existing price-state factors. Its sign must be learned on a research sample
and validated chronologically out of sample.
