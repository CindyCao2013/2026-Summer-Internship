# Metric freeze: G10 Excess Sharpe (Mentor protocol)

**Status:** ACTIVE for single-factor delivery  
**Date:** 2026-07-22  
**Convention:** **G1 = short** (lowest factor), **G10 = long** (highest factor), **H-L = G10 − G1**.

## Why

H–L / long–short Net Sharpe mixes two books and often **overstates investability** (especially with high bilateral turnover).  
For delivery, the primary portfolio metric is:

> **Equal-weight G10 (long book) excess return vs exact valid-universe equal-weight, annualized Sharpe.**

Negative-IC raw factors must be **sign-flipped to Alpha** first so G10 remains the long book.

## Exact definition

From the aligned signal and return panels (`signal_shift=1`):

```text
valid_t     = stocks with valid aligned signal and valid forward return
r_EW_t      = mean(return_t of every stock in valid_t)
r_G10,t     = equal-weight return_t of top 10% by factor
r_x,t       = r_G10,t − r_EW_t
G10 Excess Sharpe = mean(r_x)/std(r_x) * sqrt(250)
```

Same `Factor_Dev_Lib.calSharpe` (`riskFree=0`, `n=250`).  
Lib helper: `Factor_Dev_Lib.g10_excess_vs_universe_ew`.  
Also matches `alpha_investability.long_book_excess_performance(..., direction=+1)`.

**Do not** use `mean(G1..G10)` as the EW benchmark.

## Mentor hard gates

| Gate | Rule |
|------|------|
| Absolute | G10 Excess Sharpe **> 3.5** |
| Relative | G10 Excess Sharpe **> every G1…G10 Sharpe and H-L Sharpe** from the same `groupTest` |

Helper: `Factor_Dev_Lib.check_excess_gates`.

Buffer / lower-frequency execution may appear as a **footnote / diagnostic** only.  
Buffer must **not** be the sole means of passing gates (overfit risk).

## Required single-factor modules

Aligned with `Factor_Dev_Lib.groupTest` title metrics:

| Module | Content |
|--------|---------|
| Decile + H-L | `groupTest` → AnnuRet, Sharpe, MDD, Daily Turnover, Implied AnnuFee, Daily IC, Annu ICIR |
| decile_return | Mean daily return bar chart G1…G10 |
| Neutralization | `panel_neutral_size_ind` with `nt_type` in `{ind, cap, ind_cap}` (+ raw) |
| G10 Excess | vs exact universe EW |
| Universes | ALL / HS300 / CSI500 / CSI1000 via **membership mask** (not `base_index` ret) |
| Decay | T+1 / T+5 / T+10 / T+20 forward compound returns |

Tradability: `apply_tradability_mask` (ST ∩ non-limit) on signal and ret before tests.

## Factor matrix

**Paused.** Need ~100 factors with pairwise corr < 0.7 before combination work.  
Focus: **single-factor discovery and optimization** until G10 excess gates pass.

## Artifacts

| File | Role |
|------|------|
| `selected_factor_metrics.csv` | Exact headline table where recomputed |
| Mentor protocol CSV from `run_mentor_single_factor_protocol.py` | Gates + universe + decay |
| `long_book_excess_daily_*.csv` | Daily G10 / EW / excess |

## Cost

Current G10 excess Sharpe is **gross** (fee=0 in source `groupTest`).  
Net G10 excess @15bp is a separate diagnostic; do not invent.
