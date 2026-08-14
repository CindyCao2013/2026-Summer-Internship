# IdealReversal Research Pack v1 — Index

**Status:** testing · formula not frozen · weak monotonicity (0.44)

## Canonical docs

| Doc | Purpose |
|-----|---------|
| [factor_definition.md](factor_definition.md) | Identity + intuition |
| [formula.md](formula.md) | Ret20 → ATS knife → M spread |
| [data_source.md](data_source.md) | EOD inputs + coverage |
| [implementation.md](implementation.md) | Code map |
| [validation.md](validation.md) | IC / decile / stability / exec |
| [summary.yaml](summary.yaml) | Machine-readable card |

## Analysis folders

| Folder | Contents |
|--------|----------|
| `ic_analysis/` | factor_summary.csv |
| `quantile_analysis/` | decile_means · mechanism_legs |
| `stability/` | block + yearly tables |
| `execution/` | cost grid (252d window) |

## Headline metrics (raw harvest, 1703d)

| Metric | Value |
|--------|------:|
| RankIC (raw / SI) | −0.0311 / −0.0331 |
| ICIR (raw / SI) | −8.60 / −9.46 |
| H-L Sharpe | 1.70 |
| Monotonicity | 0.444 |
| Best net Sharpe | 1.70 @ TO 0.15 |

## Legacy (keep; not required to read)

- `factor_report.md` / `factor_card.yaml` / `metrics.json` / `summary.md` — Template v2 harvest
- `research/reports/factor_cutting_v1/ideal_reversal/` — cutting validation lineage
- `research/reports/ideal_reversal_v1/` — milestone 2.2.1 execution outputs

## Reproduce pointers

```text
factor_cutting/ideal_reversal.py
factor_cutting/w_cut.py
factor_specs/IdealReversal.yaml
run_milestone_2_2_1_ideal_reversal.py
```
