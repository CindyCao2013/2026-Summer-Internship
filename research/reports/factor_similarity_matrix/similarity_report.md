# Factor Similarity Matrix v1

**Window:** 2022-01-28 → 2025-12-31 (951d confirmation)
**Signal book:** size+industry CS-z · signal_shift=1
**Universe factors:** TGD20, FlowDensity20, D1_LiquidityQuality60d, D4_WinnerSentimentReversal5d, D5_UpsideFragility20d

## Questions answered

1. Same alpha source? → IC / CS / H-L correlations
2. Incremental alpha? → residual ICIR after CS residualization
3. Combination value? → independence roles (no Composite weights yet)

## IC correlation (daily RankIC series, Pearson)

| index | TGD20 | FlowDensity20 | D1_LiquidityQuality60d | D4_WinnerSentimentReversal5d | D5_UpsideFragility20d |
| --- | --- | --- | --- | --- | --- |
| TGD20 | 1.000 | 0.319 | 0.661 | 0.021 | -0.342 |
| FlowDensity20 | 0.319 | 1.000 | 0.285 | 0.108 | -0.219 |
| D1_LiquidityQuality60d | 0.661 | 0.285 | 1.000 | -0.055 | -0.537 |
| D4_WinnerSentimentReversal5d | 0.021 | 0.108 | -0.055 | 1.000 | -0.312 |
| D5_UpsideFragility20d | -0.342 | -0.219 | -0.537 | -0.312 | 1.000 |

## Factor return correlation (daily H-L)

| index | TGD20 | FlowDensity20 | D1_LiquidityQuality60d | D4_WinnerSentimentReversal5d | D5_UpsideFragility20d |
| --- | --- | --- | --- | --- | --- |
| TGD20 | 1.000 | 0.253 | 0.646 | 0.310 | -0.298 |
| FlowDensity20 | 0.253 | 1.000 | 0.305 | -0.019 | -0.180 |
| D1_LiquidityQuality60d | 0.646 | 0.305 | 1.000 | 0.435 | -0.619 |
| D4_WinnerSentimentReversal5d | 0.310 | -0.019 | 0.435 | 1.000 | -0.605 |
| D5_UpsideFragility20d | -0.298 | -0.180 | -0.619 | -0.605 | 1.000 |

## Residual IC (Y ⊥ X)

| Y | X | CS corr | Raw ICIR(Y) | Resid ICIR | Retention | Role |
|---|---|---------|------------:|-----------:|----------:|------|
| D1_LiquidityQuality60d | D4_WinnerSentimentReversal5d | 0.161 | 9.72 | 7.87 | 0.81 | independent_source |
| D1_LiquidityQuality60d | D5_UpsideFragility20d | -0.341 | 9.72 | 7.25 | 0.75 | independent_source |
| D1_LiquidityQuality60d | FlowDensity20 | 0.344 | 9.72 | 8.71 | 0.90 | independent_source |
| D1_LiquidityQuality60d | TGD20 | 0.367 | 9.72 | 7.21 | 0.74 | independent_source |
| D4_WinnerSentimentReversal5d | D1_LiquidityQuality60d | 0.161 | 5.64 | -0.55 | -0.10 | redundant_or_absorbed |
| D4_WinnerSentimentReversal5d | D5_UpsideFragility20d | -0.470 | 5.64 | -1.23 | -0.22 | mostly_redundant |
| D4_WinnerSentimentReversal5d | FlowDensity20 | -0.027 | 5.64 | 4.21 | 0.75 | independent_source |
| D4_WinnerSentimentReversal5d | TGD20 | 0.075 | 5.64 | 1.92 | 0.34 | mostly_redundant |
| D5_UpsideFragility20d | D1_LiquidityQuality60d | -0.341 | -9.56 | -5.33 | 0.56 | partial_overlap_enhancer |
| D5_UpsideFragility20d | D4_WinnerSentimentReversal5d | -0.470 | -9.56 | -7.59 | 0.79 | independent_source |
| D5_UpsideFragility20d | FlowDensity20 | -0.100 | -9.56 | -8.99 | 0.94 | independent_source |
| D5_UpsideFragility20d | TGD20 | -0.181 | -9.56 | -7.09 | 0.74 | independent_source |
| FlowDensity20 | D1_LiquidityQuality60d | 0.344 | 4.85 | -0.62 | -0.13 | redundant_or_absorbed |
| FlowDensity20 | D4_WinnerSentimentReversal5d | -0.027 | 4.85 | 4.78 | 0.99 | independent_source |
| FlowDensity20 | D5_UpsideFragility20d | -0.100 | 4.85 | 3.48 | 0.72 | independent_source |
| FlowDensity20 | TGD20 | 0.217 | 4.85 | 1.68 | 0.35 | mostly_redundant |
| TGD20 | D1_LiquidityQuality60d | 0.367 | 11.28 | 6.06 | 0.54 | partial_overlap_enhancer |
| TGD20 | D4_WinnerSentimentReversal5d | 0.075 | 11.28 | 8.31 | 0.74 | independent_source |
| TGD20 | D5_UpsideFragility20d | -0.181 | 11.28 | 7.59 | 0.67 | partial_overlap_enhancer |
| TGD20 | FlowDensity20 | 0.217 | 11.28 | 9.12 | 0.81 | independent_source |

## Clusters / role hints

```yaml
schema_version: factor_similarity_v1
signal_book: confirmation_size_industry_cs_z
clusters:
  temporal_core:
  - TGD20
  liquidity_quality:
  - D1_LiquidityQuality60d
  flow_liquidity_interaction:
  - FlowDensity20
  behavioral_reversal:
  - D4_WinnerSentimentReversal5d
  fragility_tail:
  - D5_UpsideFragility20d
independence:
  TGD20:
    vs_TGD20: self
    alpha_role_hint: core
  FlowDensity20:
    vs_TGD20: mostly_redundant
    vs_D1_LiquidityQuality60d: redundant_or_absorbed
    ic_corr_vs_TGD20: 0.3191217562240744
    ic_corr_vs_D1: 0.2853043425749344
    alpha_role_hint: redundant_risk
  D1_LiquidityQuality60d:
    vs_TGD20: independent_source
    vs_D1_LiquidityQuality60d: self
    ic_corr_vs_TGD20: 0.660545109205371
    ic_corr_vs_D1: null
    alpha_role_hint: core_or_satellite
  D4_WinnerSentimentReversal5d:
    vs_TGD20: mostly_redundant
    vs_D1_LiquidityQuality60d: redundant_or_absorbed
    ic_corr_vs_TGD20: 0.020842079333780288
    ic_corr_vs_D1: -0.05474941016687976
    alpha_role_hint: redundant_risk
  D5_UpsideFragility20d:
    vs_TGD20: independent_source
    vs_D1_LiquidityQuality60d: partial_overlap_enhancer
    ic_corr_vs_TGD20: -0.342087237217332
    ic_corr_vs_D1: -0.5371738483626406
    alpha_role_hint: core_or_satellite
notes:
- Clusters are taxonomy seeds + residual roles — not ML clustering.
- D4/D5 lack Template v2 packs; treat as library_inventory candidates.
- No Composite weights produced in 1F.
```

## Verdict (human-readable)

- **TGD20** is the temporal core (`alpha_role_hint=core`). Flow vs TGD: `mostly_redundant` (IC corr=0.3191217562240744).
- **D1** vs TGD: `independent_source` (IC corr=0.660545109205371). Slow liquidity quality — candidate core/satellite, not validated.
- **Flow ⊥ D1:** resid ICIR=-0.62, retention=-0.13, role=`redundant_or_absorbed` — critical for whether Flow is new info vs liquidity repackaging.
- **D4_WinnerSentimentReversal5d**: vs TGD `mostly_redundant`, role hint `redundant_risk` (library inventory; pack incomplete).
- **D5_UpsideFragility20d**: vs TGD `independent_source`, role hint `core_or_satellite` (library inventory; pack incomplete).

## Independent / redundant / enhancer (summary)

| Factor | Independent alpha source? | Redundant risk? | Enhancer-only risk? |
|--------|---------------------------|-----------------|---------------------|
| TGD20 | yes | low/monitor | no |
| FlowDensity20 | partial/no | yes | no |
| D1_LiquidityQuality60d | yes | low/monitor | no |
| D4_WinnerSentimentReversal5d | partial/no | yes | no |
| D5_UpsideFragility20d | yes | low/monitor | no |

## Artifacts

- `factor_ic_corr.csv`
- `factor_return_corr.csv`
- `cs_corr_matrix.csv`
- `residual_ic_matrix.csv`
- `residual_ic_long.csv`
- `factor_clusters.yaml`
- `similarity_verdict.json`
- `figures/`

## Explicit non-goals

- No Composite weights
- No Registry schema changes
- No formula changes

## Next

Composite Alpha Engine only after human review of residual roles (especially Flow⊥D1 and D4/D5 pack completeness).

