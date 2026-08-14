# FlowDensity20 — Formula (not frozen)

## 1. Raw variables

Daily L2 / broker-derived fields:

- `active_buy_amt`, `active_sell_amt` — active buy/sell amounts  
- `float_mktcap` — floating market capitalization  

## 2. Intermediate variables

**Net active flow**

\[
\mathrm{NetActiveFlow}_{i,t} = \mathrm{active\_buy\_amt}_{i,t} - \mathrm{active\_sell\_amt}_{i,t}
\]

**Flow density (daily)**

\[
f_{i,t} = \frac{\mathrm{NetActiveFlow}_{i,t}}{\mathrm{MktCap}_{i,t}}
\]

## 3. Smoothing + cross-sectional z-score

Rolling sum over 20 days, then cross-sectional z-score:

\[
\mathrm{FlowDensity20}_{i,t} = \mathrm{cs\_zscore}\left(\sum_{k=0}^{19} f_{i,t-k}\right)
\]

Module: `factor_formulas_l2_flow_p2.py` → `build_net_active_flow_mktcap(window=20)`

## 4. Diagnostics (not the traded factor)

Amount-orthogonal residual: cross-sectional residual \(F \perp \mathrm{Amount}\) — used to test mechanism, not registered as a separate factor_id.

## 5. Final investable signal

\[
\mathrm{signal}_{i,t} = \mathrm{FlowDensity20}_{i,t-1}
\]

Headline evaluation book: **size+industry** neutralized decile H-L on confirmation harvest.

---

## One-line identity

```text
FlowDensity20 = cs_zscore( MA20_sum( net_active_flow / mktcap ) )
```

Not:

```text
FlowDensity20 ≠ pure smart-money direction
FlowDensity20 ≠ amount channel alone (ICIR ≈ −8.6)
```
