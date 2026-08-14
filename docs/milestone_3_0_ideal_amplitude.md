# Phase III A2 — IdealAmplitude

**Date:** 2026-07-20  
**Status:** PASS — admitted as `testing`  
**Script:** `run_milestone_3_0_ideal_amplitude.py`  
**Pack:** `research/reports/factors/IdealAmplitude/`  
**Phase doc:** `docs/milestone_3_0_alpha_library_expansion.md`

---

## Scope

Full OS path for IdealAmplitude (paper cutting family).  
**No** portfolio / composite / TGD-D1-Flow changes.

```
paper → factor_spec → cutting impl → harvest + exec → Generator v2 → Registry
```

---

## Results

| Source | RankICIR | H-L / Net Sharpe | Mono | Note |
|--------|----------|------------------|-----:|------|
| Cutting harvest (3885d) | −7.66 | Sharpe 3.44 | **0.11** | soft-bar mono fail |
| Fresh last 252d | −10.86 | plain Net 2.41 | ~0.11 | confirms sign |
| Exec best | — | **Net 3.40** (`daily|buffer_5_15`) | — | TO≈0.45 |

**Registry:** `IdealAmplitude` · status **`testing`** · `production_ready=false`  
**Pack validation:** Generator v2 `ok=true`, 32 metrics, 7 charts.

---

## Interpretation

- Cutting claim holds (ICIR / Sharpe strong).  
- **Mono blocks candidate** — honest `testing`.  
- Same microstructure family as IdealReversal — residual independence deferred (library expansion first).

---

## Constraints honored

- Portfolio Construction remains **frozen**  
- No composite re-optimization  
- TGD/D1/Flow topology frozen  

---

## Next

```
Phase III Track A3: ActiveTrade
        ↓
Track B: SUE / earnings surprise
```
