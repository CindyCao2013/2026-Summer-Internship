# D1_LiquidityQuality60d Research Pack v1 — Index

**Status:** candidate · formula frozen · library status = `candidate`

## Canonical docs

| Doc | Purpose |
|-----|---------|
| [factor_definition.md](factor_definition.md) | Identity + intuition |
| [formula.md](formula.md) | vol_60d + amount stability → rank-mean |
| [data_source.md](data_source.md) | OHLCV + coverage |
| [implementation.md](implementation.md) | Code map |
| [validation.md](validation.md) | IC / decile / stability / exec |
| [summary.yaml](summary.yaml) | Machine-readable card |

## Analysis folders

| Folder | Contents |
|--------|----------|
| `ic_analysis/` | factor_summary.csv |
| `quantile_analysis/` | decile charts (pending harvest) |
| `stability/` | block stability tables |
| `execution/` | cost grid (1D.7) |

## Legacy (keep; not required to read)

- `factor_report.md` / `factor_card.yaml` / `metrics.json` — Template v2 harvest  
- `diagnostics/` — universe ladder, IC decay  
- `mechanism/` — structural diagnostics  

## Reproduce pointers

```text
factor_formulas_eod_engine.py  → f_low_vol_liquidity_quality_60d
factor_formulas_liquidity_d1.py
factor_specs/D1_LiquidityQuality60d.yaml
run_milestone_1d7_pack_completion.py
```
