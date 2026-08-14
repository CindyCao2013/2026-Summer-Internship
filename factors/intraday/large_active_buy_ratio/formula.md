# Formula

This factor is a **bar-level proxy** necessitated by the absence of true
large-order amount buckets in the live schema.

For symbol-session bar \(t\), define average active-buy size as

\[
buy\_size_t =
\begin{cases}
buy\_amt_t / buy\_count_t, & buy\_count_t > 0 \\
\mathrm{NaN}, & \text{otherwise}.
\end{cases}
\]

Let \(\mu^-_{20,t}\) and \(s^-_{20,t}\) be the mean and **sample** standard
deviation of valid `buy_size` observations in the prior 20-bar rolling window.
Each statistic requires at least 10 valid observations and is shifted by one
bar. The current bar is classified as large when

\[
large_t =
\mathbf{1}\left[
buy\_size_t > \mu^-_{20,t} + s^-_{20,t}
\right].
\]

The factor value is

\[
value_t =
\frac{
\sum_{i=t-19}^{t} buy\_amt_i \cdot large_i
}{
\sum_{i=t-19}^{t} buy\_amt_i
}.
\]

The trailing sums use a 20-bar window with minimum 10 observations. Output
also requires at least 10 bars in that window with valid shifted baselines and
a valid, positive denominator. Values are in \([0,1]\).

Classification applies to an entire minute bar. The numerator therefore
contains all active-buy amount from classified bars, not amounts from
identified large orders. The formula supports neither an individual
large-order interpretation nor an institutional-flow interpretation.
