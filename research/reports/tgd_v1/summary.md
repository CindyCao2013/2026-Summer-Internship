# TGD v1 — Stage 4 Validation Summary

**Factor:** `TGD20`
**Signal shift:** `1` (TGD20 on day T uses full-session Gu/Gd → predicts T+1)
**Portfolio:** 10-group + H-L (project standard; not paper 5-group)
**Sample:** 2022-01-28 → 2025-12-31 (951d confirmation focus noted below)

## Pipeline (frozen)

```
minute → Gu/Gd → residual(εu,εd) → εd~εu → MA20 → TGD20
```

Formula layer is frozen — this report is research validation only.

## A. Replication sanity (raw, confirmation)

| Metric | Value |
|--------|-------|
| RankIC | 0.0430 |
| ICIR | 6.98 |
| Group10 excess Sharpe vs exact ALL EW | 1.24 |
| H-L Sharpe | 2.77 |
| Decile monotonicity (Spearman) | 0.988 |
| Direction | 1 |

Artifacts: `portfolio/cumulative_long_short.png` (10-group + H-L), `portfolio/decile_return.png`, `portfolio/hml_curve.png`, `ic/rank_ic.csv`

## B. Neutralization ladder

| Mode | RankIC | ICIR | Long-book excess Sharpe | H-L Sharpe | Net Sharpe@15bp | Daily TO(H-L) |
|------|--------|------|-------------------------|------------|-----------------|---------------|
| raw | 0.0430 | 6.98 | 1.24 | 2.77 | 1.00 | 0.65 |
| size | 0.0443 | 8.67 | 1.79 | 3.52 | 1.51 | 0.65 |
| industry | 0.0408 | 8.90 | 1.47 | 3.19 | 1.16 | 0.64 |
| size_industry | 0.0415 | 11.29 | **2.16** | 4.06 | 1.72 | 0.65 |

Artifact: `neutralization/neut_summary.csv`

## C. Period stability

Yearly positive mean-RankIC: **6/6**

| Period | Kind | RankIC | ICIR | Pos IC frac | n |
|--------|------|--------|------|-------------|---|
| 2020 | year | 0.0358 | 8.01 | 0.70 | 242 |
| 2021 | year | 0.0360 | 6.69 | 0.65 | 243 |
| 2022 | year | 0.0469 | 9.12 | 0.73 | 242 |
| 2023 | year | 0.0521 | 8.95 | 0.73 | 242 |
| 2024 | year | 0.0317 | 4.25 | 0.64 | 242 |
| 2025 | year | 0.0429 | 7.06 | 0.68 | 243 |
| 2020-2021 | block | 0.0359 | 7.26 | 0.68 | 485 |
| 2022-2023 | block | 0.0495 | 9.02 | 0.73 | 484 |
| 2024-2025 | block | 0.0373 | 5.49 | 0.66 | 485 |

Artifact: `stability/yearly_ic.csv`

## D. Cost / turnover

| Metric | Raw confirmation |
|--------|------------------|
| Gross Sharpe (tradable) | 2.88 |
| Net Sharpe @15bp RT | 1.00 |
| Daily H-L turnover | 0.65 |
| Implied AnnuFee(7.5%) | 12.13% |
| Annu one-way TO | 67.33 |

Artifact: `cost/turnover_cost.csv`

## Notes vs paper

- Do **not** hard-match paper 5-group IR / RankICIR numbers.
- Compare mechanism (decile ordering) + statistical strength + robustness + cost.

## Status

| Stage | Status |
|-------|--------|
| 0–3 formula / info layer | ✅ frozen |
| 4 validation runner | ✅ this report |

