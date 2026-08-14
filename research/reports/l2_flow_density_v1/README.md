# P2 L2 Net Active-Flow Density

## Validation v1 (closed before TGD)

Script: `run_l2_flow_density_validation_v1.py`  
Artifacts: `research/reports/l2_flow_density_v1/validation_v1/`

| Check | Result |
|-------|--------|
| Horizon @ 11:29 | Peak **Ret_120 Sharpe 4.17**; Ret_NDay 2.35 (not single-cell only) |
| Period split | See `period_heatmap_*.png` |
| Neut ladder (10-group, conf) | size+ind **ICIR 4.85 / net 1.85**; raw net −0.18 |
| Implied AnnuFee | ~8.7% @ size+ind |

**Next (deferred):** Temporal Feature Layer → Gu/Gd → thin TGD — see `TEMPORAL_FEATURE_LAYER.md`. Do not open `tgd.py` yet.

## Discovery (504d)


**Sample:** 2023-12-05 → 2025-12-31  
**Gate:** size+industry neutralized → residual vs Base3 **and** vs `cn_voi_shock` → stack ICIR uplift

| Factor | ICIR | resid_t Base3 | resid_t VOI | Stack uplift | TO≈ | Verdict |
|--------|------|---------------|-------------|--------------|-----|---------|
| **`net_active_flow_mktcap_20d`** | **5.04** | **4.68** | **7.40** | **+1.02** | 53 | **enhancer_candidate** |
| `net_active_flow_mktcap_5d` | 3.62 | 3.76 | 5.47 | +0.67 | 113 | enhancer_candidate (higher TO) |
| `active_buy_share_20d` | 1.95 | 2.23 | 3.81 | +0.57 | 67 | enhancer_candidate |
| `net_active_flow_shock` | 0.02 | 2.09 | 1.32 | 0 | 196 | weak / overlaps VOI path |

## Confirmation + Investability (951d) — PASSED

**Period:** 2022-01-28 → 2025-12-31 (951d, standard OOS after first-504 discovery split)  
**Script:** `run_l2_flow_confirmation.py`  
**Verdict:** **`confirm_pass_enhancer`** — all hard gates true

| Check | Result | Bar |
|-------|--------|-----|
| ICIR | **4.85** | > 3.0 |
| residual_t vs Base3 | **6.11** | > 2.0 |
| residual_t vs cn_voi_shock | **9.93** | > 2.0 |
| Stack ICIR uplift | **+1.07** (best λ=0.3) | ≥ 0 |
| Yearly IC+ | **100%** (2022–2025 all +) | > 70% |
| Net Sharpe @ 15bp RT | **1.85** | > 0 |
| Window 14/20/26 ICIR | 4.76 / 4.85 / 5.01 | sign stable |
| Annu one-way TO | **52.3%** | < 100% |
| Capacity (ADV×5%) | ~¥21bn | acceptable |

### Soft flag — universe concentration

| Universe | IC | ICIR |
|----------|-----|------|
| ALL | +0.024 | **4.85** |
| CSI1000 | +0.008 | 1.47 |
| CSI500 | −0.005 | −0.63 |
| CSI300 | −0.024 | −2.21 |

Signal is **broad-universe / small-cap concentrated**. Do not expect CSI300 standalone alpha. Combination experiments should keep ALL (or CSI1000+) book unless explicitly testing large-cap overlay.

## Artifacts

- `l2_flow_density_summary.csv` / `l2_flow_density_verdict.json` — discovery
- `confirmation_summary.csv` / `confirmation_verdict.json` — confirmation + investability
- `confirmation_yearly_ic.csv` / `confirmation_universe.csv` / `confirmation_param_stability.csv`
- `p2_deepen_formula_pool.md` — next formulas after freeze

## Standard pack (Factor Report Template v1)

Canonical pack (do not invent a second layout):

`research/reports/factors/FlowDensity20/`

Regenerate:

```bash
python run_factor_report_generator_v1.py --factor FlowDensity20
```

## Execution (closed — same grid as TGD20)

Script: `run_flow_density_execution_opt_v1.py`  
Artifacts: `research/reports/l2_flow_density_v1/execution/`

| Config | Gross | Net@15bp | Daily TO |
|--------|------:|---------:|---------:|
| Baseline size+ind daily | 3.38 | **1.85** | 0.46 |
| **Best: daily\|buffer_10_30** | 3.71 | **2.88** | **0.165** |
| every_10d plain | 3.53 | 2.69 | 0.179 |

Apples-to-apples vs TGD best Net **2.32** (`buffer_5_15`).

## Orthogonality vs TGD20 (closed)

Script: `run_tgd_flow_orthogonality_v1.py`  
Report: `research/reports/factor_orthogonality/TGD20_FlowDensity20/`

| Case | Corr | Residual highlight |
|------|-----:|--------------------|
| A TGD vs Flow raw | **0.22** | TGD⊥Flow ICIR **9.12**; Flow⊥TGD **1.68** |
| B TGD vs Flow⊥Amount | **0.01** | pure flow not positive |
| C TGD vs Amount | **−0.32** | TGD⊥Amount ICIR **7.66** |

Equal-rank composite ICIR **8.50** < TGD alone **11.28** → complementary information, but **do not** use 50/50 ranks; prefer IC-weighted Composite v1.

## Amount-orthogonalization v1 (attribution — not freeze)

Script: `run_flow_density_amount_orth_v1.py`  
Artifacts: `mechanism/mechanism_amount_neutral.csv`, `amount_orth_verdict.json`  
Case: **`case_interaction_entangled`** → category **`microstructure` + `liquidity_flow_interaction`**

| Signal | ICIR | H-L | Net@15bp |
|--------|-----:|----:|---------:|
| FlowDensity_raw | **+4.85** | 3.38 | 1.92 |
| Amount | **-8.66** | 3.95 | 3.55 |
| **Flow ⊥ Amount** | **-2.49** | 0.13 | −2.10 |
| Amount ⊥ Flow | **-8.49** | 2.53 | 0.67 |

cs_corr(Flow, Amount) ≈ **−0.62**. Signed ICIR retention ≈ **−0.51** (sign flips).

**Attribution:** raw FlowDensity ≈ Flow × Liquidity interaction. Pure flow after amount orth does **not** carry the positive alpha. Do not freeze as pure flow; keep candidate.

## Next (unblocked)

1. **Composite Alpha Engine v1** — IC-weighted TGD + FlowDensity_raw (not equal rank; not Flow⊥Amount)
2. Keep FlowDensity as `validated_single_factor_candidate` / interaction factor
3. Optional long-form essay documenting interaction identity
4. Do **not** freeze as pure Flow
