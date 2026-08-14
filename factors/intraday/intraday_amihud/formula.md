# Formula

For symbol \(s\), session \(d\), and minute row \(t\), use raw minute close,
raw traded amount, and the simple minute return:

\[
P_{s,d,t}=Close_{s,d,t}
\]

\[
A_{s,d,t}=Amount_{s,d,t}
\]

\[
r_{s,d,t}=\frac{P_{s,d,t}}{P_{s,d,t-1}}-1.
\]

The factor is:

\[
IntradayAmihud_{s,d,t}
=\frac{\sum_{j=0}^{4}|r_{s,d,t-j}|}
{\sum_{j=0}^{4}A_{s,d,t-j}}.
\]

Both trailing sums use a five-row trading-minute window with
`min_periods=3`. The lag and rolling groups are `(symbol, date)`, so the first
return of every session is missing and no overnight return enters the factor.
Zero or non-economic amount denominators produce no signal.

Only values available at the five standard bartimes are emitted:
`09:59`, `10:29`, `11:29`, `13:29`, and `14:29`.
