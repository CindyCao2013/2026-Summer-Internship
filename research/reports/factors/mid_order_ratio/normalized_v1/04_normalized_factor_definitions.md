# 04 — Normalized Factor Definitions

## A0 — Fixed RMB benchmark

\[
mid\_trade\_amount\_share\_abs\_4w20w_{i,t} =
\frac{\sum_k amount_{i,t,k}
I(40{,}000 < amount_{i,t,k}\le 200{,}000)}
{\sum_k amount_{i,t,k}}.
\]

`mid_order_ratio` is retained only as a legacy alias.  A0 is the parity
benchmark and is not described as cross-stock scale normalized.

## A1 — Lagged ADV20 normalization

\[
ADV20^{lag1}_{i,t}
= mean(TotalAmount_{i,t-20},\ldots,TotalAmount_{i,t-1}),
\]

\[
relative\_trade\_size^{ADV}_{i,t,k}
= \frac{amount_{i,t,k}}{ADV20^{lag1}_{i,t}}\times 10{,}000
\quad\text{bps of ADV},
\]

\[
A1_{i,t} =
\frac{\sum_k amount_{i,t,k}
I(L_{adv}<relative\_trade\_size^{ADV}_{i,t,k}\le H_{adv})}
{\sum_k amount_{i,t,k}}.
\]

The main bounds are frozen from the 2023-01-03 to 2023-06-30 trade-size
distribution without loading returns.  The candidate grid is
`L={0.5,1,2}` bps and `H={5,10,20}` bps.  The deterministic selection matches
A0's amount coverage subject to non-sparse market-cap-quintile coverage.

## A2 — Lagged typical Tick amount

For each stock-day, ClickHouse first computes the median positive Tick
execution amount.  The historical scale is:

\[
ATS20^{lag1}_{i,t}
= median(DailyMedianTradeAmount_{i,t-20},\ldots,
DailyMedianTradeAmount_{i,t-1}).
\]

\[
A2_{i,t} =
\frac{\sum_k amount_{i,t,k}
I(0.5 < amount_{i,t,k}/ATS20^{lag1}_{i,t}\le 2.0)}
{\sum_k amount_{i,t,k}}.
\]

The main `0.5–2.0` interval is frozen ex ante.  The
`L={0.25,0.5,0.75}`, `H={1.5,2.0,3.0}` grid is a stability diagnostic, not a
parameter-selection exercise.

## A3 — Current-day relative quantiles

\[
A3_{i,t} =
\frac{\sum_k amount_{i,t,k}
I(Q20_{i,t}<amount_{i,t,k}\le Q80_{i,t})}
{\sum_k amount_{i,t,k}}.
\]

A3 is known only after the close and is shifted one trading day before
evaluation.  It describes the middle of the current day's execution-amount
distribution, not deviation from a historical norm.  Because count coverage
would be partly mechanical, the factor remains an amount share.

## No-look-ahead and missing history

ADV20 and ATS20 use the previous 20 market trading dates after reindexing to
the complete trading calendar.  A missing source day makes the scale and
factor NaN; older observations are not pulled forward to fill the window.
Every valid scale row stores a maximum source date no later than `t-1`.

All variants use raw-direction RankIC.  Effective-direction portfolio
diagnostics use the ex-ante frozen direction `-1`; no evaluation window may
re-infer the sign.

