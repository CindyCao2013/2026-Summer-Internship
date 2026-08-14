# Formula

For symbol \(s\), session \(d\), and minute bar \(t\), define the valid
active-buy average ticket:

\[
q_{s,d,t} =
\frac{
  ActiveBuyAmount_{s,d,t} \times AdjFactor_{s,d,t}
}{
  ActiveBuyCount_{s,d,t}
},
\qquad ActiveBuyCount_{s,d,t} > 0.
\]

A missing or zero adjustment factor is treated as \(1\). Bars with a
non-positive active-buy count have no valid \(q\).

The shifted prior-bar baseline is:

\[
\bar q^{(-1)}_{s,d,t}
=
\operatorname{mean}
\left(
q_{s,d,u}: u \in [t-20,t-1]
\right),
\]

where the 20-bar window must contain at least 10 valid ticket observations.
The current bar is excluded by the one-bar shift.

The factor is:

\[
average\_active\_trade\_size_{s,d,t}
=
\frac{q_{s,d,t}}{\bar q^{(-1)}_{s,d,t}} - 1.
\]

It is emitted only when the current ticket and positive baseline are valid, at
09:59, 10:29, 11:29, 13:29, and 14:29. The timestamp is the bar timestamp:
all inputs are available at or before that time. The measure describes ticket
size only and does not identify institutional or any other trader type.
