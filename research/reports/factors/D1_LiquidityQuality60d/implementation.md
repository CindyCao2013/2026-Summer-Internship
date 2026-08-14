# D1_LiquidityQuality60d — Implementation

## Code map

| Stage | Module | Role |
|-------|--------|------|
| Registry / candidates | `factor_formulas_liquidity_d1.py` | D1 representative id, satellite list |
| Formula | `factor_formulas_eod_engine.py` | `f_low_vol_liquidity_quality_60d` |
| Build | `build_eod_engine_factor("low_vol_liquidity_quality_60d", pv)` | wide panel for eval |
| Execution grid | `run_milestone_1d7_pack_completion.py` | 1D.7 buffer / rebalance grid |
| Spec | `factor_specs/D1_LiquidityQuality60d.yaml` | metadata (frozen) |

## Build steps (conceptual)

```text
1. Load OHLCV cache → vol_60d, amount_cv_20d
2. CS rank each leg → rank-mean blend → D1 score
3. Wide panel + signal_shift=1 → RankIC / decile / execution
4. Optional: cs_zscore(raw) for 1D.7 execution grid
```

## Frozen parameters

| Param | Value |
|-------|------:|
| Vol window | 60 |
| Amount CV window | 20 |
| signal_shift | 1 |
| Cost baseline | 15 bp round-trip |
| Execution signal (1D.7) | raw cs_zscore |

## Do not change under `D1_LiquidityQuality60d`

- Window retune (60d / 20d)  
- Library constructor substitution  
- New variant under same factor_id  

Full narrative: `factor_report.md`
