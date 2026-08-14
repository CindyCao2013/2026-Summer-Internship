# RV Exposure Audit v1 (Sprint 4.3)

## Purpose

`realized_volatility` is the strongest retained intraday candidate:

| Field | Value |
|-------|-------|
| factor | `realized_volatility` |
| family | volatility |
| bartime | 14:29 |
| horizon | `Ret_30` |
| direction | -1 |
| train IC / ICIR | -0.0989 / -11.53 |
| train H-L Sharpe | 9.75 |
| train H-L market beta | -0.343 |

This sprint does **not** re-search bartime, horizon, or direction. It answers one
attribution question:

> Is 14:29 RV predicting 14:30–15:00 returns as independent short-horizon
> volatility-reversal alpha, or as a hidden size / beta / liquidity / momentum
> style proxy?

## Frozen signal definition

Session-to-current realized volatility (unchanged from the factor package):

\[
RV_{s,d,t}=\sqrt{\sum_{i\le t} r_{s,d,i}^{2}}
\]

At `t = 14:29`, the signal uses only minute closes at or before 14:29 and
forecasts `Ret_30` over 14:30–15:00. Direction `-1` means long low-RV /
short high-RV.

Prior residualization in the freeze file only orthogonalizes against
same-slot production factors (`close_vwap_deviation`,
`active_buy_sell_imbalance`). That is **not** a market-structure exposure
audit. Sprint 4.3 adds style controls.

## Research cases

### Case A — independent alpha

\[
r^{e}_{i,t}=\alpha+\beta\,RV_{i,t}+\gamma'X_{i,t}+\varepsilon_{i,t}
\]

After controlling for market structure, \(\beta\) stays significant with the
frozen negative sign, and residual signal IC retains a large share of raw IC.

### Case B — hidden style factor

High RV mostly marks small-cap / illiquid / high-beta / high-turnover names.
Once \(X\) enters the regression, \(\beta\) collapses and residual IC dies.

## Canonical methods

All annualization uses 250 trading days. No OOS reselection.

### 1. Exposure diagnostics

Daily cross-sectional Spearman of RV vs each control at 14:29:

- `size` = \(\log(\mathrm{float\_mktcap}_{t-1})\)
- `liquidity` = \(20\mathrm{d}\) mean amount, lagged one day
- `hist_vol` = \(20\mathrm{d}\) close-to-close volatility, lagged one day
- `momentum_20d` = 20-day return, lagged one day
- `session_mom` = open→14:29 session return (available at signal time)

### 2. Fama–MacBeth return regressions

Dependent variable is the exact filtered-constituent market-excess return:

\[
r^{e}_{i,t}=Ret\_30_{i,t}-\overline{Ret\_30}_{t}
\]

Cross-section market level is absorbed by construction; including a constant
market factor in the CS regression would be unidentified.

Daily OLS on rank-z covariates, then time-series mean / Newey–West-style
t-stat (HAC lag = 5 by default):

1. univariate: \(r^{e}=\alpha+\beta RV+\varepsilon\)
2. progressive: add size → liquidity → hist_vol → momentum_20d → session_mom
3. full: all controls jointly
4. industry-adjusted: industry-demean \(r^{e}\) and covariates, then rerun full

### 3. Progressive residual IC chain

Rank-z residualize RV against the same control order, then compute daily
Spearman IC vs `Ret_30` (signed by frozen direction `-1`). Report retention:

\[
\mathrm{retention}=\frac{|\mathrm{IC}_{resid}|}{|\mathrm{IC}_{raw}|}
\]

## Verdict rules

| Verdict | Rule |
|---------|------|
| `case_a_independent_alpha` | full-model \(\lvert t_{\beta_{RV}}\rvert\ge 2\) with frozen sign, and residual signed-IC retention \(\ge 0.50\) |
| `case_b_style_proxy` | full-model \(\lvert t_{\beta_{RV}}\rvert<2\) **or** retention \(< 0.30\) |
| `mixed_partial_alpha` | otherwise |

Dominant drop step is the first progressive control that removes the largest
share of \(|\mathrm{IC}|\).

## Non-goals

- No change to `Intraday_Factor_Test_Process.py`, `intraday_lib.py`, or DDB
  factor packages.
- No freeze mutation; `intraday_alpha_freeze_v1.json` stays locked.
- No cost / turnover re-simulation (already covered by portfolio simulator v1).
- No promotion decision beyond Case A / B / mixed.

## Reproducibility

- Spec: this document
- Pure math: `core/evaluation/rv_exposure_audit.py`
- Runner: `research/run_rv_exposure_audit_v1.py`
- Tests: `tests/test_rv_exposure_audit_v1.py`
- Outputs: `research/results/rv_exposure_audit_v1/`

Default sample is the frozen train window 2024-01-01 → 2024-06-30 on
`000852.SH`. Optional `--period validation_2024H2` / `test_2025_available`
reuses freeze OOS dates without changing the frozen tuple.

## Quick start

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# Unit tests (synthetic panels; no DB)
$PY -m unittest tests.test_rv_exposure_audit_v1 -v

# Full 2024H1 audit (requires DolphinDB + PREHEAT_RET_MATRIX_ZZ1000)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_rv_exposure_audit_v1.py

# Faster path without open→14:29 session momentum
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_rv_exposure_audit_v1.py \
  --skip-session-mom

# OOS windows under the same frozen tuple
OMP_NUM_THREADS=1 PYTHONPATH=. $PY research/run_rv_exposure_audit_v1.py \
  --period validation_2024H2
```
