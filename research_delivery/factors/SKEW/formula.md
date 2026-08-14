# SKEW — Formula

## Baseline total skew

\[
r_{i,t}=P_{i,t}/P_{i,t-1}-1,\quad
\mathrm{SKEW}_{i,t}^{(L)}=
\frac{\frac{1}{L}\sum (r-\bar r)^3}
{(\frac{1}{L}\sum (r-\bar r)^2)^{3/2}}
\]

Windows (pre-registered): L ∈ {20, 60, 120}.

## Idiosyncratic skew (headline)

\[
r_{i,s}=\alpha_{i,t}+\beta_{i,t} r_{m,s}+\varepsilon_{i,s},\quad
\mathrm{IdioSKEW}_{i,t}^{(L)}=\mathrm{Skew}(\varepsilon)
\]

Market: CSI300 daily c2c. Windows: L ∈ {60, 120}.

Implementation: `core/factors/skew/` (rolling market-model residual + rolling skew).

## Alpha

\[
\mathrm{Alpha}=-\mathrm{SKEW}
\]

`signal_shift = 1` applied in evaluation.
