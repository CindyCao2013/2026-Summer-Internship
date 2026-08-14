# Cutting Validation — ideal_amplitude

Knife: `close_price_state`

## Validation chain

```
original factor (mixed)
        ↓
knife partitions states
        ↓
high leg has alpha / low leg ~ noise
        ↓
spread purifies + stabilizes
        ↓
survives neutralization / residual vs Base3
        ↓
tradable (decile / H-L)
```

## IC decomposition

| Name | RankIC | ICIR |
|------|--------|------|
| Amp20 | -0.0596 | -4.44 |
| V_high | -0.0670 | -6.34 |
| V_low | -0.0345 | -2.55 |
| V (spread) | -0.0550 | -7.41 |

**Separation** (IC_high − IC_low): `-0.0324`
**Purity** (|IC_spread| / |IC_high|): `0.822`

Purity high ⇒ low leg is near-noise; knife actually found the information locus.

## Portfolio

- H-L Sharpe: `2.86` (direction=-1)
- H-L ann return: `52.3%`
- Decile monotonicity: `0.11`

## Residual vs Base3

- residual IC mean `-0.0396` · t=`-9.69` · ICIR=`-5.69`

## Plots

- `ic_analysis/rank_ic_bar.png` — cutting before/after
- `ic_analysis/cumulative_ic.png` — persistence of legs
- `mechanism/high_low_leg_ic.png` — core cutting claim
- `mechanism/knife_bucket_return.png` — knife separation
- `portfolio/decile_return.png` + `long_short_curve.png`
- `robustness/neutralization.png` + `raw_vs_residual.png` + `universe_compare.png`
