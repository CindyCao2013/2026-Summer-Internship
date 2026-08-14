# L2 Feature Factory — brick layer

**Date:** 2026-07-23  
**Status:** ADOPTED (incremental)

---

## Naming rule

Brick / feature names describe the **observable**, not a causal story.

| Bad (overclaim) | Good (observable) |
|-----------------|-------------------|
| `inst_ratio` / 机构参与度 | `active_size_concentration` |
| “聪明钱强度” as brick id | `active_buy_top20_amt_share` (SmartMoney knife) |

Large active buys ≠ 机构. May be 游资 / 量化 / 散户集中交易.

---

## Target layout

```
core/l2_features/
├── bricks/                 # shared feature bricks (observable)
│   ├── active_size/        # active avg-size concentration
│   ├── active_pressure/    # amount-weighted (buy-sell)/(buy+sell)
│   ├── flow_density20/     # (migrate later)
│   └── tgd20/              # (migrate later)
├── builders/               # brick(s) → factor panel (thin)
├── *_builder.py            # legacy builders (re-export / migrate)
└── ...
```

Pipeline:

```
L2 raw (minute Active_*)
  → feature brick
  → factor
  → neutralized factor
  → portfolio
```

---

## Shared cache

```
research/cache/bricks/active_size/daily_YYYYMM.parquet
research/cache/bricks/active_pressure/daily_YYYYMM.parquet
research/cache/bricks/active_pressure_session/daily_YYYYMM.parquet
research/cache/bricks/active_pressure_smart/daily_YYYYMM.parquet
research/cache/bricks/active_pressure_smartv2/daily_YYYYMM.parquet
```

IdealReversal / future factors that need ASC read `active_size`.
APM delivery (`APM_ActiveV2_SmartV2_1F`) reads `active_pressure_smartv2`.
Legacy `research/cache/ideal_reversal_active_v2/daily_brick/` is promoted on read
(`inst_ratio` → `active_size_concentration`).

---

## Consumers

| Factor | Brick |
|--------|-------|
| IdealReversal_ActiveV2 | `active_size` |
| SmartMoneyActiveV2 | own concentration brick (amount-top20; migrate later) |
| APM delivery `SmartV2_1F` | `active_pressure_smartv2` (ASC gate off) |
| APM baseline / Session / Smart V1 | `active_pressure` / `_session` / `_smart` |
| Flow Density / TGD20 | migrate under `bricks/` |

