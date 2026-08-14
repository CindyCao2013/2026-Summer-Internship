# Formula

For valid minute returns \(r_1,\ldots,r_n\) observed from the session open
through decision time \(t\):

\[
r_i = \frac{P_i}{P_{i-1}} - 1
\]

\[
M_2 = \sum_i (r_i-\bar r)^2,\qquad
M_3 = \sum_i (r_i-\bar r)^3
\]

\[
\operatorname{MinuteSkew}_t =
\frac{n\sqrt{n-1}}{n-2}\frac{M_3}{M_2^{3/2}}
\]

This is the bias-corrected sample skewness used by pandas `expanding().skew()`.
At least three valid returns and positive \(M_2\) are required.

The DDB implementation derives \(M_2\) and \(M_3\) from ordered cumulative raw
moments inside each `Symbol, Date` context. No day-end statistic is stamped at
an earlier bartime.
