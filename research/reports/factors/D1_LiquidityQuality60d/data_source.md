# D1_LiquidityQuality60d — Data Source

## Input

| Item | Spec |
|------|------|
| Fields | OHLCV daily (returns, volume/amount) |
| Derived | `volatility_20d`, `amount_cv_20d`, 60d return std |
| Universe (research harvest) | ALL A-shares (0/3/6) |
| Target calendar | 2018–2025 |

## Actual coverage (this pack)

| Field | Value |
|-------|-------|
| Confirmation sample | **confirmation_1455d** (1455 trading days) |
| Stability | Block summary only (no full yearly panel) |
| Exception | Harvested from confirmation pack; full Dual Benchmark pending |

Requested vs actual is recorded in `summary.yaml` (`coverage_exception: true`).  
Do not claim 2018 start for headline IC metrics unless re-run.

## Pipeline layers

```text
OHLCV daily
    → vol_60d + amount stability
    → cross-sectional rank-mean blend
    → D1_LiquidityQuality60d panel (date × symbol)
```

Cache / builders: EOD price-volume stack + `factor_formulas_eod_engine.build_eod_engine_factor`
