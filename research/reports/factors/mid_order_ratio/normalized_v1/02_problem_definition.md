# 02 — Problem Definition

## What A0 already normalizes

The legacy factor is not completely unnormalized.  Its numerator is divided by
the stock's same-day total traded amount:

\[
A0_{i,t} =
\frac{\sum_k amount_{i,t,k}\,
I(40{,}000 < amount_{i,t,k}\le 200{,}000)}
{\sum_k amount_{i,t,k}}.
\]

This removes the mechanical level effect of total daily traded amount from the
final ratio.

## What A0 does not normalize

The classification boundaries remain fixed at RMB 40,000 and RMB 200,000.
They therefore have different economic scale across stocks.  A RMB 100,000
execution is 50 bps of ADV for a stock trading RMB 20 million per day, but only
0.5 bp for a stock trading RMB 2 billion per day.  A0 classifies both
executions identically.

The missing normalization is consequently at the **Tick classification
boundary**, before daily aggregation.  It is not the final division by daily
turnover.

## Why post-factor transforms do not repair it

A daily cross-sectional z-score is a monotone rescaling of already aggregated
stock-day values.  Industry or market-cap OLS residualization removes linear
cross-sectional exposure from the final daily factor.  Neither operation can
reassign individual executions to a different bucket.

The required order is:

1. normalize every execution amount by a lagged stock-specific scale;
2. classify executions using the normalized scale;
3. aggregate selected execution amounts into a daily amount share;
4. optionally standardize or residualize the daily factor;
5. run standalone predictive validation.

## Interpretation boundary

Trade amount is not an investor-identity label.  No parent-order
reconstruction is performed.  The study therefore uses “Tick trade amount,”
“execution amount,” and “trade scale,” never “institutional order size” or
“retail order size.”

