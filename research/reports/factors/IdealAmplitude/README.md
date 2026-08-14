# IdealAmplitude Research Pack v1 — Index

**Status:** testing · formula not frozen · **weak monotonicity (0.11)**

## Canonical docs

| Doc | Purpose |
|-----|---------|
| [factor_definition.md](factor_definition.md) | Identity + intuition |
| [formula.md](formula.md) | Amp → close-state knife → V spread |
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

## Headline metrics (3885d harvest)

| Metric | Value |
|--------|------:|
| RankIC (raw / SI) | −0.0378 / −0.0475 |
| ICIR (raw / SI) | −7.66 / −9.97 |
| H-L Sharpe | 3.44 |
| **Monotonicity** | **0.111** (very weak) |
| Best net Sharpe | 3.40 @ TO 0.45 |

## Legacy (keep; not required to read)

- `factor_report.md` / `factor_card.yaml` / `metrics.json` / `summary.md` — Template v2 harvest
- `mechanism/` — leg diagnostics + mechanism.md
- `research/reports/factor_cutting_v1/ideal_amplitude/` — cutting validation lineage
- `research/reports/ideal_amplitude_v1/` — milestone 3.0 execution outputs

## Reproduce pointers

```text
factor_cutting/ideal_amplitude.py
factor_specs/IdealAmplitude.yaml
run_milestone_3_0_ideal_amplitude.py
run_ideal_amplitude_mechanism_diagnosis.py
```
