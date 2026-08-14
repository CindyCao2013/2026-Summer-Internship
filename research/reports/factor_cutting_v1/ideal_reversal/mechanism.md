# Mechanism — ideal_reversal

Knife: `ats_trade_count`

| Leg | RankIC | ICIR | IC+ |
|-----|--------|------|-----|
| high | -0.0430 | -5.73 | 0.35 |
| low | 0.0020 | 0.32 | 0.54 |
| spread | -0.0338 | -6.48 | 0.31 |

**Knife separation** (IC_high - IC_low): `-0.0451`
**Knife purity** (|IC_spread| / max(|IC_h|,|IC_l|)): `0.786`

Interpretation:
- High leg should carry the alpha; low leg should be near-noise.
- Spread = high − low is information purification, not a second independent factor.

Paper claim: high-knife days concentrate alpha; low-knife days are noise. Spread = purification via difference.
