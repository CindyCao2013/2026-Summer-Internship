# active_buy_shock — Formula

For symbol \(i\), session \(d\), and bar \(t\), first adjust active-buy amount:

\[
B_{i,d,t}
= ActiveBuyAmount_{i,d,t} \times Adjfactor_{i,d,t},
\]

where a null or zero adjustment factor is treated as \(1\).

Let \(H_{i,d,t}\) contain at most the 20 bars immediately before \(t\) in the
same session. With at least 10 observations, define:

\[
\mu^-_{20,i,d,t} = mean(B_{i,d,s}: s \in H_{i,d,t}),
\]

\[
\sigma^-_{20,i,d,t}
= sampleStd(B_{i,d,s}: s \in H_{i,d,t}).
\]

The factor is:

\[
active\_buy\_shock_{i,d,t}
= \frac{B_{i,d,t} - \mu^-_{20,i,d,t}}
       {\sigma^-_{20,i,d,t}}.
\]

Implementation identity:

```text
(current adjusted active-buy amount
 - move(mavg(adjusted active-buy amount, 20, 10), +1))
/ move(mstd(adjusted active-buy amount, 20, 10), +1)
```

The positive one-bar shift excludes the current bar. Rolling state is grouped
by `Symbol, Date`, and output is restricted to 09:59, 10:29, 11:29, 13:29,
and 14:29. Invalid or zero historical standard deviations produce no signal.
