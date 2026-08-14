# Mechanism — ideal_amplitude

Knife: `close_price_state_lambda25`

| Leg | RankIC | ICIR | IC+ |
|-----|--------|------|-----|
| high | -0.0507 | -6.44 | 0.32 |
| low | -0.0227 | -2.68 | 0.42 |
| spread | -0.0378 | -7.66 | 0.29 |

**Knife separation** (IC_high - IC_low): `-0.0280`
**Knife purity** (|IC_spread| / max(|IC_h|,|IC_l|)): `0.746`

Interpretation:
- High leg should carry the alpha; low leg should be near-noise.
- Spread = high − low is information purification, not a second independent factor.

Paper claim: high-price-state amplitude carries stronger negative alpha.
