# APM_SessionResidual Research Pack v1 — Index

**Status:** Pack v1 · `testing_candidate` · formula frozen · execution recipe frozen  
**Identity class:** `adapted_replication` (EOD index daytime proxy)

## Canonical docs

| Doc | Purpose |
|-----|---------|
| [factor_definition.md](factor_definition.md) | Identity + intuition + locks |
| [formula.md](formula.md) | Session residual → APM_stat → apm_cs |
| [data_source.md](data_source.md) | Minute + EOD index + adapted gap |
| [implementation.md](implementation.md) | Code map |
| [validation.md](validation.md) | IC / decile / stability / peers |
| [execution.md](execution.md) | Frozen recipe `daily\|buffer_10_30` |
| [summary.yaml](summary.yaml) | Machine-readable card |

## Analysis folders

| Folder | Contents |
|--------|----------|
| `ic_analysis/` | factor_summary · neutral IC · IC curve · peer corr |
| `quantile_analysis/` | decile · monotonicity |
| `stability/` | yearly IC |
| `execution/` | grid · horizon IC · verdict · turnover curve |

## Reproduce

```text
run_milestone_c1_apm_session_panel.py
run_milestone_c1_apm_session_sanity.py
run_milestone_c1_apm_session_scout.py
run_milestone_c1_apm_session_execution.py
```

## Distinct from

| ID | Relation |
|----|----------|
| `ActiveTradeProxy` | daily ON−DAY proxy — **not** this factor |
| `SmartMoney10d` | efficiency knife — orthogonal (~0 signal corr) |
| `FlowDensity20` | active flow — orthogonal (~0 signal corr) |

**Not in Registry.** Library asset only.
