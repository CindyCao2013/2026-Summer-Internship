# C7 Combination — C2_D1_0.60 + λ·P2

**Period:** 2022-01-28 → 2025-12-31 (951d)  
**Script:** `run_combination_c7.py`  
**pass_combo_gate:** True

## Recommended

**`C7_size_tight_λ0.3`** — ALL book + industry/size residualize after blend

| Metric | C2 baseline | C7 recommended | Δ |
|--------|-------------|----------------|---|
| ICIR | 5.73 | **9.63** | +3.90 |
| Net Sharpe (15bp) | 0.67 | **2.25** | +1.58 |
| Annu TO 1-way | 66.2% | 62.5% | −3.7pp |
| size \|exposure\| mean | 0.60 | **0.07** | passes ≤0.2σ |
| Max DD (net) | −16.0% | **−6.9%** | |

Soft miss: aspirational TO ≤50% (actual 62.5%). Hard fence ≤100% OK.

## Additive track (fair P2-only uplift, no extra neut)

| λ | ICIR | Net Sharpe | TO | size\|μ\| |
|---|------|------------|-----|----------|
| 0.1 | 5.90 | 0.87 | 64.5 | 0.57 |
| 0.2 | 6.08 | 1.06 | 62.5 | 0.53 |
| 0.3 | 6.30 | 1.25 | 60.7 | 0.46 |
| 0.4 | 6.57 | 1.47 | 59.2 | 0.37 |
| **0.5** | **6.94** | **1.74** | **57.5** | **0.25** |

Additive λ=0.5 is the cleanest “just add P2” story (+1.20 ICIR, +1.07 net). Size still slightly above 0.2σ → prefer size_tight for production construction.

> Note: size_tight ICIR jump vs raw C2 partly reflects stripping Base3’s own size exposure. Use additive Δ for “P2 incremental alpha”; use size_tight for deployable book.

## Minute heatmaps (separate track)

Bartime × horizon HML heatmaps are in `Intraday_Factor_Test_Process.py` / `intraday_lib.create_group_heatmap` — for **minute-bar** factors, not this daily C7 combo. C7 plots here are λ-grid / excess-curve / size-exposure.

## Artifacts

- `c7_lambda_grid.csv` / `c7_verdict.json`
- `excess_cum_net_vs_c2.png` / `cum_net_c2_vs_best_c7.png`
- `lambda_heatmap.png` / `best_size_exposure_hist.png`
