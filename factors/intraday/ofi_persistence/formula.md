# Formula

For symbol \(s\), date \(d\), and observed trading-minute bar \(t\), define
bar-level order-flow imbalance as

\[
OFI_{s,d,t} =
\begin{cases}
\dfrac{B_{s,d,t}-S_{s,d,t}}{B_{s,d,t}+S_{s,d,t}},
& B_{s,d,t}+S_{s,d,t}>0,\\
\mathrm{NaN}, & \text{otherwise},
\end{cases}
\]

where \(B\) and \(S\) are raw positive active-buy and active-sell amounts.

Let \(W_{s,d,t}\) contain the current bar and up to 19 preceding observed
trading-minute bars for the same `Symbol, Date`. Let
\(V_{s,d,t}=\{i\in W_{s,d,t}: OFI_{s,d,i}\text{ is valid}\}\). Then

\[
ofi\_persistence_{s,d,t} =
\frac{\sum_{i\in V_{s,d,t}}\mathbf{1}(OFI_{s,d,i}>0)}
{|V_{s,d,t}|},
\qquad |V_{s,d,t}|\ge 5.
\]

No value is produced with fewer than five valid observations. Values are
emitted only at 09:59, 10:29, 11:29, 13:29, and 14:29, and necessarily lie in
\([0,1]\).
