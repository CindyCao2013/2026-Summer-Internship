# TGD20 — Data Source

## Input

| Item | Spec |
|------|------|
| Bar | Stock one-minute close |
| Session | A-share continuous auction (240 trading minutes; lunch skipped) |
| Extra | Open/close for overnight control |
| Universe (research harvest) | ALL A-shares (0/3/6) |
| Target calendar | 2018–2025 |

## Actual coverage (this pack)

| Field | Value |
|-------|-------|
| Confirmation sample | **2022-01-28 → 2025-12-31** (951 trading days) |
| Yearly stability | 2020–2025 |
| Exception | Full 2018 Gu/Gd lineage not in harvest window |

Requested vs actual is recorded in `summary.yaml` (`coverage_exception: true`).  
Do not claim 2018 start for headline IC metrics unless re-run.

## Pipeline layers

```text
minute close
    → Gu / Gd daily centers
    → εu / εd residualization
    → TGD20 panel (date × symbol)
```

Cache / builders: L2 minute stack + `core/l2_features/tgd_panel_builder.py`
