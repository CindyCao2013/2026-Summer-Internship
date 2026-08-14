# TGD20 Research Pack v1 — Index

**Status:** finalized · formula frozen · library status = `validated`

## Canonical docs

| Doc | Purpose |
|-----|---------|
| [factor_definition.md](factor_definition.md) | Identity + intuition |
| [formula.md](formula.md) | Gu/Gd → residual → MA20 |
| [data_source.md](data_source.md) | Minute data + coverage |
| [implementation.md](implementation.md) | Code map |
| [validation.md](validation.md) | IC / decile / stability / exec |
| [summary.yaml](summary.yaml) | Machine-readable card |

## Analysis folders

| Folder | Contents |
|--------|----------|
| `ic_analysis/` | factor_summary · IC curve |
| `quantile_analysis/` | decile · H-L curve |
| `stability/` | yearly tables · chart |
| `execution/` | cost grid · turnover |

## Legacy (keep; not required to read)

- `factor_report.md` / `factor_card.yaml` / `metrics.json` — Template v2 harvest  
- `research/reports/tgd_v1/` — full narrative research report  

## Reproduce pointers

```text
core/l2_features/return_timing.py
core/l2_features/timing_residual.py
core/l2_features/tgd.py
factor_specs/TGD20.yaml
```
