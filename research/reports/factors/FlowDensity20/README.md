# FlowDensity20 Research Pack v1 — Index

**Status:** candidate · formula not frozen · library status = `candidate`

## Canonical docs

| Doc | Purpose |
|-----|---------|
| [factor_definition.md](factor_definition.md) | Identity + interaction thesis |
| [formula.md](formula.md) | net flow / mktcap → MA20 → cs_z |
| [data_source.md](data_source.md) | L2 active flow + coverage |
| [implementation.md](implementation.md) | Code map |
| [validation.md](validation.md) | IC / decile / stability / exec |
| [summary.yaml](summary.yaml) | Machine-readable card |

## Analysis folders

| Folder | Contents |
|--------|----------|
| `ic_analysis/` | factor_summary.csv |
| `quantile_analysis/` | decile charts (pending harvest) |
| `stability/` | yearly tables |
| `execution/` | cost grid |

## Legacy (keep; not required to read)

- `factor_report.md` / `factor_card.yaml` / `metrics.json` — Template v2 harvest  
- `diagnostics/` — amount-orth, TGD orthogonality  
- `mechanism/` — channel decomposition  

## Reproduce pointers

```text
factor_formulas_l2_flow_p2.py  → build_net_active_flow_mktcap
factor_specs/FlowDensity20.yaml
run_flow_density_mechanism_v1.py
```

Long-form essay (optional): `research/reports/l2_flow_density_v1/README.md`
