# APM_SessionResidual — Implementation

## Module map

| Module | Role |
|--------|------|
| `core/l2_features/apm_session_panel_builder.py` | PM / overnight / index proxy / residual legs |
| `core/l2_features/apm_session_signal.py` | `APM_stat` + `apm_cs` (Ret20 CS residual) |
| `factor_cutting/active_trade.py` | **Proxy only** — do not overwrite `compute_apm()` |

## API

```python
# Panel (Phase1)
build_apm_session_panel(start, end, ...)

# Signal (Phase2+)
build_apm_stat_panel(residual_panel, window=20)
compute_apm_session_residual_signal(residual_panel, window=20)
cs_residualize_vs_ret20(apm_stat_long, ret20_long)
```

Keep `compute_apm()` as `NotImplementedError`. Proxy lineage stays `compute_apm_overnight_day_proxy()`.

## Runners

| Script | Phase |
|--------|-------|
| `run_milestone_c1_apm_session_panel.py` | panel + coverage |
| `run_milestone_c1_apm_session_sanity.py` | object constructability |
| `run_milestone_c1_apm_session_scout.py` | CSI1000 alpha scout |
| `run_milestone_c1_apm_session_execution.py` | horizon + execution grid |

## Engineering notes

- DDB PM agg: `context by … csort` then **`group by Symbol, Date`** (collapse to daily).
- No `shift(1)` inside cache; eval layer shifts.
- Panel builder must not pull execution metrics / turnover fields as APM inputs.
