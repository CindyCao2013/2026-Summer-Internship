# Cutting Validation — ideal_reversal

Knife: `ats_trade_count`

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
| Ret20 | -0.0437 | -4.06 |
| M_high | -0.0476 | -5.97 |
| M_low | 0.0012 | 0.16 |
| M (spread) | -0.0377 | -6.58 |

**Separation** (IC_high − IC_low): `-0.0487`
**Purity** (|IC_spread| / |IC_high|): `0.792`

Purity high ⇒ low leg is near-noise; knife actually found the information locus.

## Portfolio

- H-L Sharpe: `1.60` (direction=-1)
- H-L ann return: `21.5%`
- Decile monotonicity: `0.33`

## Residual vs Base3

- residual IC mean `-0.0294` · t=`-9.96` · ICIR=`-5.85`

## Plots

- `ic_analysis/rank_ic_bar.png` — cutting before/after
- `ic_analysis/cumulative_ic.png` — persistence of legs
- `mechanism/high_low_leg_ic.png` — core cutting claim
- `mechanism/knife_bucket_return.png` — knife separation
- `portfolio/decile_return.png` + `long_short_curve.png`
- `robustness/neutralization.png` + `raw_vs_residual.png` + `universe_compare.png`
