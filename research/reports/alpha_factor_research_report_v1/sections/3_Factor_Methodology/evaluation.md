# 3. Factor Evaluation Methodology

Shared language used across all Packs. Updated 2026-07-22 for **Mentor single-factor protocol**.

## 3.1 Grouping convention

`Factor_Dev_Lib.groupTest` deciles:

- **G1 = short** (lowest factor value)
- **G10 = long** (highest factor value)
- **H-L = G10 − G1**

Flip negative raw factors to Alpha before grouping so G10 stays the long book.

## 3.2 Predictive metrics

### IC / RankIC

\[
\mathrm{IC}_t = \mathrm{corr}(F_t, R_{t+1}), \quad
\mathrm{RankIC}_t = \mathrm{corr}\big(\mathrm{rank}(F_t),\mathrm{rank}(R_{t+1})\big)
\]

Signal uses `shift(1)` (no same-day leak). Headline tables report **mean RankIC**.

### ICIR

\[
\mathrm{ICIR} = \frac{\mathrm{mean}(\mathrm{IC}_t)}{\mathrm{std}(\mathrm{IC}_t)} \times \sqrt{N_{\mathrm{year}}}
\]

### Neutralization

Report ladder when available (`Factor_Dev_Lib.panel_neutral_size_ind`):

- raw
- `cap` (size)
- `ind` (industry)
- `ind_cap` (size + industry) — preferred headline when present

## 3.3 Quantile / long-short (`groupTest`)

```text
Factor signal
  → tradability mask (ST ∩ non-limit)
  → neutralization
  → cross-sectional rank → 10 groups
  → G10 long / G1 short / H-L
  → report: AnnuRet, Sharpe, MDD, Turnover, Implied AnnuFee, IC, ICIR
```

Always include **decile_return** (mean daily return by group) and **H-L cumulative**.

**Monotonicity:** Spearman of group id vs mean group return.

## 3.4 Headline: G10 Excess Sharpe

\[
r^{excess}_t = r^{G10}_t - \mathrm{EW}(U_t)
\]

where \(U_t\) = stocks with valid aligned signal and return.

### Mentor gates

1. Excess Sharpe **> 3.5**
2. Excess Sharpe **> all G1…G10 Sharpes and H-L Sharpe**

See `research_delivery/METRICS_G10_EXCESS.md`.

Buffer / every-Nd is allowed as execution diagnostic only — **not** the sole way to improve headline excess.

## 3.5 Required market / decay tests

| Test | Method |
|------|--------|
| Universes | Restrict membership: ALL / HS300 / CSI500 / CSI1000 |
| Decay | Forward compound ret T+1 / T+5 / T+10 / T+20 → `groupTest` + G10 excess |

Helpers: `get_index_member_mask`, `calc_forward_returns`, `g10_excess_vs_universe_ew`.

## 3.6 Stability

Yearly (or block) RankIC sign consistency.

## 3.7 What is paused

Factor matrix / combination (need ~100 factors, pairwise corr < 0.7).  
Focus on **single-factor** discovery and optimization.

## 3.8 Evaluation workflow

```text
Signal → ST/limit mask → Neutralize → Deciles/H-L → G10 Excess → Universes → Decay → Gate check
```
