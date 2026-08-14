# CSI300 Extreme Return Effect Study v1

Short-term reversal / momentum sanity check on CSI300 extreme daily winners & losers (2023–2026).

## Quick start

From repo root (use project conda env):

```bash
PY=/opt/conda/anaconda3/envs/base_93/bin/python

# Unit tests (no DB)
$PY research/extreme_return_study/tests/test_signal_portfolio.py

# Full study
OMP_NUM_THREADS=1 PYTHONPATH=. $PY run_extreme_return_study_v1.py

# Smoke (2024H1)
OMP_NUM_THREADS=1 PYTHONPATH=. $PY run_extreme_return_study_v1.py --smoke
```

## Layout

```
research/extreme_return_study/
  src/           # data_loader, universe, signal, portfolio, backtest, metrics, visualization
  tests/
  results/       # CSV / JSON
  figures/       # publication figures
  data/          # optional caches
  report.md      # quant note
  run_study.py
```

## Design notes

- Dynamic CSI300 membership (historical weights) — no survivorship bias
- Next-open execution: formation close t → buy open t+1 → first o2o return at t+2 (`entry_lag=2`)
- Overlapping equal-weight holds (1/5/10/20D)
- Tradability: limit / ST / suspended / IPO seasoning
- Default cost: 10 bps one-way

## Alpha Factory link

Baseline family **D6: Event / Extreme Movement** — crash / limit-move / vol-shock reversal.
Condition later with volume shock, liquidity, TGD, flow density.
