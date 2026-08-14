# TGD20 — Implementation

## Code map

| Stage | Module | Role |
|-------|--------|------|
| Gu / Gd | `core/l2_features/return_timing.py` | minute → daily time centers |
| Residual | `core/l2_features/timing_residual.py` | controls → εu, εd |
| TGD20 | `core/l2_features/tgd.py` | CS εd~εu → MA20 |
| Panel | `core/l2_features/tgd_panel_builder.py` | wide panel for eval |
| Spec | `factor_specs/TGD20.yaml` | metadata (frozen) |

## Build steps (conceptual)

```text
1. Load minute bars → compute Gu, Gd
2. Residualize vs return-structure controls → εu, εd
3. Daily CS: εd ~ εu → innovation e
4. Per-symbol MA20(e) → TGD20
5. Wide panel + signal_shift=1 → RankIC / decile / execution
```

## Frozen parameters

| Param | Value |
|-------|------:|
| MA window | 20 |
| signal_shift | 1 |
| CS min obs | `DEFAULT_MIN_CS` in timing residual |
| Cost baseline | 15 bp round-trip |

## Do not change under `TGD20`

- MA window retune  
- Control set change  
- Residual → ML substitution  
- New variant under same factor_id  

Full narrative: `research/reports/tgd_v1/TGD_factor_research_report.md`
