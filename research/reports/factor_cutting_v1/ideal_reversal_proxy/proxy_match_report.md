# Proxy Knife Match — ideal_reversal pre-L2 backfill

Acceptance threshold: IC-series corr vs ATS ≥ **0.80**

| Knife | Role | Corr vs ATS | RankIC (overlap) | ICIR |
|-------|------|-------------|------------------|------|
| `ats_trade_count` | benchmark | 1.000 | -0.0338 | -6.48 |
| `amount` | proxy | 0.757 | -0.0368 | -6.34 |
| `volume` | proxy | 0.731 | -0.0371 | -6.25 |
| `turnover_proxy` | proxy | 0.706 | -0.0359 | -6.19 |
| `ats_volume` | proxy | 0.461 | -0.0244 | -4.73 |

**Best proxy:** `amount` (corr=0.757)

FAIL threshold — best corr 0.757 < 0.80. Still report best proxy with honest label; do not claim paper ATS.

## Full-history proxy factor (stitched or pure proxy)

- knife: `amount`
- mode: `stitch_ats_plus_proxy`
- RankIC: `-0.0281`
- ICIR: `-4.81`
- monthly RankIC: `-0.0563`
- n_days: `3866`

