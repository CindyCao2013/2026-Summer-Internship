# Formula

For symbol \(s\), trading session \(d\), and minute \(i\), define the adjusted
close \(P_{s,d,i}\) and simple minute return

\[
r_{s,d,i} = \frac{P_{s,d,i}}{P_{s,d,i-1}} - 1.
\]

The signal at minute \(t\) is session-to-current realized volatility:

\[
RV_{s,d,t} =
\sqrt{\sum_{i \le t} r_{s,d,i}^{2}}.
\]

The first close in each session has no return and is excluded. A signal is
valid after at least five finite minute returns (`min5`). The cumulative sum
groups by both symbol and trading date, so the afternoon continues the same
session while the next trading date resets the calculation.

Only values at 09:59, 10:29, 11:29, 13:29, and 14:29 are emitted. No
annualization or demeaning is applied.
