# Factor Cutting — Mechanism Notes

## Why cutting beats formula search

Genetic / formula search asks: *which expression of OHLCV predicts return?*  
Cutting asks: *where is the information hidden inside an already-known additive signal?*

| Approach | Search space | Typical output |
|----------|--------------|----------------|
| Formula search | ops on OHLCV | `rank(delay(close/volume))` |
| Factor cutting | partition boundaries | `diff(agg(return \| knife=high), …)` |

Both can coexist in Alpha Factory as:

```
Formula Search  +  Cutting Operator Search
```

## Economic stories (must survive Stage 3)

### Ideal reversal

- High ATS days ≈ institutional / large-ticket participation.
- Large tickets push price further from value → stronger subsequent reversal.
- Low ATS days are noise / retail churn → weak or opposite (momentum-ish).
- Validation: future reversal strength ↑ in knife high quantiles.

### Ideal amplitude

- High-price-state amplitude embeds “振荡加大 → 状态跃迁” more than low-price-state.
- Traditional low-vol anomaly is partly a mixture; cutting isolates the high-price leg.
- Validation: \(V_\mathrm{high}\) more negative IC than \(V_\mathrm{low}\); difference more stable.

### APM

- Morning / overnight vs afternoon trader mix differs → residual α asymmetry.
- Validation: overnight residual ≠ afternoon residual in signed predictive power.

### Smart Money

- Minutes with high \(|R|/V^{0.25}\) mark informed prints; relative VWAP tracks smart money level.
- Validation: high Q predicts underperformance (crowding at expensive prints).

## Stage 3 diagnostic (framework)

For a knife \(K\) and object return \(r\):

1. Within lookback, bucket days by \(K\) quantile.
2. Aggregate \(r\) per bucket → factor legs.
3. Correlate each leg with future returns (cross-section).
4. Require **monotonic knife → mechanism** story, not only aggregate IC.

## Integration with this repo

- Density gate (same as SUE / event / L2): residual IC vs Base3 + stack λ uplift + turnover fence.
- Cutting factors that collapse after size/industry residual = style pollution, not microstructure alpha.
- Do **not** start with minute feature store; daily W-cut / amplitude first.
