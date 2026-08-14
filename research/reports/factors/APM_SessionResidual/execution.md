# APM_SessionResidual — Execution (frozen recipe)

**C1.4 Case A** · artifacts in `execution/`

---

## Frozen recipe (Lock 1)

```yaml
execution_recipe:
  ranking:
    signal: apm_cs
  portfolio:
    side: long_high_apm
  rebalance:
    frequency: daily
  buffer:
    enabled: true
    lower: 0.10
    upper: 0.30
    label: buffer_10_30
  cost:
    round_trip: 15bp
  frozen_label: "highAPM|daily|buffer_10_30"
```

**Direction lock:** long high `apm_cs` — no sign flip.

---

## Horizon diagnosis

| H | RankIC | ICIR |
|--:|-------:|-----:|
| 1 | 2.39% | 4.1 |
| 5 | 2.70% | 4.9 |
| 10 | 3.10% | 6.1 |
| 20 | 4.03% | 8.4 |

`horizon_class = medium_or_longer_persistent` (IC strengthens with H).

---

## Execution grid (@15bp)

| Scheme | Net Sharpe | TO |
|--------|----------:|---:|
| daily plain | 0.92 | 0.75 |
| every_10d | 1.09 | 0.25 |
| every_10d \| buffer_5_15 | 1.37 | 0.21 |
| **daily \| buffer_10_30** | **1.50** | **0.28** |

Buffer keeps session persistence while cutting ranking-noise turnover.

---

## Pack admission note

Unlike SmartMoney10d (best Net≈0.31 → `research_candidate` parked), APM clears Net>1 with a documented recipe → **`testing_candidate`** Pack v1.

**Not Registry.** Recipe changes require updating `summary.yaml` + this file.
