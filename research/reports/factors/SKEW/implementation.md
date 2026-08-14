# SKEW — Implementation

| Item | Path |
|------|------|
| Total skew | `core/factors/skew/skew.py` |
| Idio skew | `core/factors/skew/idio_skew.py` |
| Realized (P1 stub) | `core/factors/skew/realized_skew.py` |
| Engine wrappers | `factor_formulas_eod_engine.py::{skew_20d,skew_60d,skew_120d}` |
| Spec | `factor_specs/SKEW.yaml` |
| Runner | `run_skew_validation_v1.py` |

Engine wrappers expose **Alpha = -SKEW** for harness compatibility.
