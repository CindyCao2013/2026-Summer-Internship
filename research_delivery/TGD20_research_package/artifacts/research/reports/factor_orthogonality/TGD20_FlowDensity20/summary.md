# TGD20 ⟂ FlowDensity20 Orthogonality Report v1

**Window:** confirmation (post discovery-504)  
**Signals:** size+industry neutralized, signal_shift=1  

## Information taxonomy

| Factor | Category | Role |
|--------|----------|------|
| TGD20 | temporal_information | Pure temporal timing residual |
| FlowDensity_raw | liquidity_flow_interaction | Flow × Liquidity (tradable) |
| Flow_perp_Amount | flow_only_candidate | Not validated (ICIR flipped neg) |
| Amount | microstructure / liquidity | Anti-activity anomaly |

## Case design

- **A** TGD vs FlowDensity_raw — do two tradable alphas overlap?
- **B** TGD vs Flow_perp_Amount — any pure-flow leftover vs TGD?
- **C** TGD vs Amount — is TGD independent of liquidity anomaly?

## Correlation matrix (mean CS Spearman)

| index | TGD20 | FlowDensity_raw | Flow_perp_Amount | Amount |
| --- | --- | --- | --- | --- |
| TGD20 | 1.000 | 0.217 | 0.006 | -0.323 |
| FlowDensity_raw | 0.217 | 1.000 | 0.669 | -0.617 |
| Flow_perp_Amount | 0.006 | 0.669 | 1.000 | 0.026 |
| Amount | -0.323 | -0.617 | 0.026 | 1.000 |

## Residual IC (both directions)

| Case | Y ⊥ X | CS corr | Raw ICIR(Y) | Resid ICIR | Resid t | Retention | Class |
|------|-------|--------:|------------:|-----------:|--------:|----------:|-------|
| A | `TGD20_perp_FlowDensity_raw` | 0.217 | 11.28 | 9.12 | 17.77 | 0.81 | mostly_independent |
| A | `FlowDensity_raw_perp_TGD20` | 0.217 | 4.85 | 1.68 | 3.27 | 0.35 | partial_overlap |
| B | `TGD20_perp_Flow_perp_Amount` | 0.006 | 11.28 | 9.60 | 18.71 | 0.85 | independent |
| B | `Flow_perp_Amount_perp_TGD20` | 0.006 | -2.49 | -2.50 | -4.87 | 1.00 | independent |
| C | `TGD20_perp_Amount` | -0.323 | 11.28 | 7.66 | 14.93 | 0.68 | mostly_independent |
| C | `Amount_perp_TGD20` | -0.323 | -8.65 | -5.26 | -10.26 | 0.61 | mostly_independent |
| X | `FlowDensity_raw_perp_Amount` | -0.617 | 4.85 | -1.66 | -3.24 | -0.34 | partial_overlap |
| X | `Amount_perp_FlowDensity_raw` | -0.617 | -8.65 | -7.56 | -14.72 | 0.87 | partial_overlap |

## Equal-rank composite probe (not production weights)

| Label | ICIR A | ICIR B | Composite | Beats max? | Uplift | Corr |
|-------|-------:|-------:|----------:|:----------:|-------:|-----:|
| A_TGD_FlowRaw | 11.28 | 4.85 | 8.50 | N | -2.79 | 0.217 |
| B_TGD_FlowPerp | 11.28 | -2.49 | 5.78 | N | -5.51 | 0.006 |
| C_TGD_Amount | 11.28 | -8.65 | -0.48 | N | -11.76 | -0.323 |
| X_FlowRaw_Amount | 4.85 | -8.65 | -6.45 | N | -11.30 | -0.617 |

## Verdict

- **Overall:** `low_overlap_probe_mixed`
- **Composite readiness:** `composite_v1_worth_testing`

Case A: corr(TGD, FlowRaw)=0.217; TGD⊥Flow resid ICIR=9.12 (mostly_independent); Flow⊥TGD resid ICIR=1.68 (partial_overlap). Equal-rank composite ICIR=8.50 (beats max single: False). Case C: TGD⊥Amount resid ICIR=7.66 (mostly_independent) — TGD is not absorbed by liquidity anomaly. Taxonomy: Temporal (TGD) + Liquidity-conditioned Flow (FlowDensity), not Temporal + Pure Flow. Flow_perp_Amount remains non-candidate.

## Artifacts

- `correlation.csv`
- `residual_ic.csv`
- `composite_probe.csv`
- `orthogonality_verdict.json`
- `figures/factor_overlap_matrix.png`

## Next

- If Case A mostly independent → Composite Alpha Engine v1 is justified
- Prefer **IC-weighted** ranks (equal 0.5/0.5 underperforms TGD alone here)
- Keep FlowDensity as interaction factor (do not freeze as pure flow)
- Do not use Flow_perp_Amount as a long-side enhancer

